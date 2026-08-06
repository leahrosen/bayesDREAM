"""
Recapitulation-simulation stage: simulate data FROM the real fit's trans
summary (same x_true/meta/priors), refit, and compare recovered parameters
to the ones used to simulate. Answers "how accurately can fit_trans recover
its own fitted dose-response curves, given the same noise model" -- distinct
from examples/simulation_study, which simulates from scratch (unknown ground
truth priors) to test recovery under controlled conditions; this stage
starts FROM an already-completed real fit.

Not a bayesDREAM CLI subcommand. Requires ntc + cis + trans already fit and
saved for this gene (needs x_true, sum_factors, alpha_y_prefit, and the
saved trans_summary.csv as ground truth). Deep-copies the loaded model and
swaps in simulated counts for the primary/trans modality, so x_true,
sum_factors, technical_group_code and alpha_y_prefit are IDENTICAL to the
real fit -- only the observed counts fit_trans regresses against are
simulated. This is deliberate: the point is to test recovery of the
dose-response curve itself, not re-derive the whole pipeline.

CHECK BEFORE TRUSTING AT SCALE: simulate_from_trans_summary's returned
DataFrame is reindexed here to match the modality's feature_names/cell_names
order defensively (its docstring doesn't guarantee row/column order matches
the input trans_summary_df/meta exactly) -- but this reindexing has only
been checked by reading the source, not run end-to-end. Sanity-check on one
gene (e.g. compare summary stats of the simulated counts against the real
counts for a couple of features) before trusting the full sweep.

Usage
-----
    python run_recapitulation_sim.py --config <gene_stage_config.yaml> --rep 0

Expects a top-level ``simulation:`` block, e.g.::

    simulation:
      load_ntc: {enabled: true}
      load_cis: {enabled: true}
      load_trans: {enabled: true}
      trans_summary_csv: null    # optional explicit path; default <output_dir>/<label>/trans_feature_summary_<modality>.csv
      sum_factor_col: sum_factor_refit
      group_col: technical_group_code
      fit:                       # passed to fit_trans() on the simulated data; defaults to trans.fit if omitted
        sum_factor_col: sum_factor_refit
        function_type: additive_hill
      output_dir: null           # optional; defaults to <output_dir>/<label>/recapitulation/rep_<rep>
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import (  # noqa: E402
    build_model_from_config,
    load_bayesdream_yaml,
    normalize_stage_args,
    is_enabled,
    apply_sum_factor_adjustments,
)
from git_provenance import save_provenance_json  # noqa: E402

from bayesDREAM.simulation import simulate_from_trans_summary  # noqa: E402


def _seed_for(label: str, rep: int) -> int:
    return abs(hash(f"{label}__recapitulation__rep{rep}")) % (2**32)


def _compare_summaries(ground_truth: "object", recovered: "object", key_cols):
    import pandas as pd

    merged = ground_truth.merge(recovered, on="feature", suffixes=("_true", "_recovered"))
    rows = []
    for col in key_cols:
        c_true, c_rec = f"{col}_true", f"{col}_recovered"
        if c_true not in merged.columns or c_rec not in merged.columns:
            continue
        diff = merged[c_rec] - merged[c_true]
        rows.append({
            "parameter": col,
            "n": diff.notna().sum(),
            "pearson_r": merged[[c_true, c_rec]].dropna().corr().iloc[0, 1] if diff.notna().sum() > 1 else float("nan"),
            "mean_abs_error": diff.abs().mean(),
            "median_abs_error": diff.abs().median(),
        })
    return pd.DataFrame(rows)


def run_recapitulation_sim(cfg: dict, rep: int) -> None:
    import copy
    import numpy as np
    import pandas as pd
    import pyro
    import torch

    model_cfg = cfg.get("model") or {}
    label = model_cfg.get("label")
    seed = _seed_for(label, rep)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_config(cfg)

    sim_cfg = cfg.get("simulation") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}

    if is_enabled(sim_cfg.get("load_ntc", sim_cfg.get("load_technical")), default=True):
        model.load_ntc_fit(**normalize_stage_args(sim_cfg.get("load_ntc") or sim_cfg.get("load_technical")))
    if is_enabled(sim_cfg.get("load_cis"), default=True):
        model.load_cis_fit(**normalize_stage_args(sim_cfg.get("load_cis")))
    if is_enabled(sim_cfg.get("load_trans"), default=True):
        model.load_trans_fit(**normalize_stage_args(sim_cfg.get("load_trans")))

    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {})

    # Must match the real trans run's feature set exactly -- ground_truth
    # (trans_summary.csv, read below) only has rows for genes that survived
    # this same filter originally; skipping it here would make the later
    # reindex-to-modality-features step fail (or worse, silently misalign).
    excl_cfg = cfg.get("exclude_trans_genes") or {}
    if is_enabled(excl_cfg, default=False):
        model.exclude_trans_genes(**normalize_stage_args(excl_cfg))

    modality_name = sim_cfg.get("modality_name") or model.primary_modality
    group_col = sim_cfg.get("group_col", "technical_group_code")
    sum_factor_col = sim_cfg.get("sum_factor_col", "sum_factor_refit")

    output_dir = model_cfg.get("output_dir", "output")
    default_summary = os.path.join(output_dir, label, f"trans_feature_summary_{modality_name}.csv")
    trans_summary_path = sim_cfg.get("trans_summary_csv") or default_summary
    ground_truth = pd.read_csv(trans_summary_path)

    primary_mod = model.get_modality(modality_name)
    cis_mod = model.get_modality("cis")

    x_true = np.asarray(model.x_true)
    x_counts = np.asarray(cis_mod.counts).reshape(-1)
    sim_sum_factor = primary_mod.sum_factors[sum_factor_col].values

    simulated = simulate_from_trans_summary(
        trans_summary_df=ground_truth,
        meta=model.meta,
        x_true=x_true,
        x_counts=x_counts,
        cis_gene=model.cis_gene,
        sim_sum_factor=sim_sum_factor,
        group_col=group_col,
        seed=seed,
    )

    sim_model = copy.deepcopy(model)

    if isinstance(simulated, dict):
        # multinomial: add as a new modality per simulate_from_trans_summary's own
        # docstring example, rather than mutating the 3-D counts in place.
        sim_modality_name = f"{modality_name}_sim"
        sim_model.add_custom_modality(
            name=sim_modality_name,
            counts=simulated["counts"],
            feature_meta=simulated["feature_meta"],
            distribution="multinomial",
            cell_names=simulated["cells"],
        )
        fit_modality_name = sim_modality_name
    else:
        # negbinom/normal/studentt/binomial: DataFrame with an extra cis-gene row.
        sim_counts = simulated.drop(index=model.cis_gene, errors="ignore")
        sim_mod = sim_model.get_modality(modality_name)
        # Defensive reindex -- see module docstring's "CHECK BEFORE TRUSTING" note.
        sim_counts = sim_counts.reindex(index=sim_mod.feature_names, columns=sim_mod.cell_names)
        if sim_counts.isna().any().any():
            raise ValueError(
                "Reindexing simulate_from_trans_summary()'s output to the modality's "
                "feature_names/cell_names produced NaNs -- the returned frame's row/"
                "column labels don't fully match the modality. Inspect `simulated` "
                "directly before proceeding."
            )
        sim_mod.counts = sim_counts.values
        fit_modality_name = modality_name

    fit_args = normalize_stage_args(sim_cfg.get("fit") or (cfg.get("trans") or {}).get("fit"))
    fit_args["modality_name"] = fit_modality_name
    sim_model.fit_trans(**fit_args)

    rec_output_dir = sim_cfg.get("output_dir")
    if not rec_output_dir:
        rec_output_dir = os.path.join(output_dir, label, "recapitulation", f"rep_{rep}")
    os.makedirs(rec_output_dir, exist_ok=True)

    sim_model.save_trans_fit(output_dir=rec_output_dir)
    sim_model.save_trans_summary(output_dir=rec_output_dir, modality_name=fit_modality_name)

    recovered = pd.read_csv(os.path.join(rec_output_dir, f"trans_feature_summary_{fit_modality_name}.csv"))
    key_cols = [c for c in [
        "Vmax_pos_median", "K_pos_median", "n_pos_median",
        "Vmax_neg_median", "K_neg_median", "n_neg_median",
    ] if c in ground_truth.columns]
    comparison = _compare_summaries(ground_truth, recovered, key_cols)
    comparison_path = os.path.join(rec_output_dir, "recapitulation_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    print(f"[recapitulation] rep={rep} seed={seed} -> {rec_output_dir}")
    save_provenance_json(
        os.path.join(rec_output_dir, "provenance.json"),
        extra={"stage": "recapitulation", "label": label, "rep": rep, "seed": seed},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Path to bayesdream-CLI-schema YAML config.")
    parser.add_argument("--rep", type=int, required=True, help="Replicate index (also seeds RNG).")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    run_recapitulation_sim(cfg, args.rep)


if __name__ == "__main__":
    main()
