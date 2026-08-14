"""
Standalone trans-fit stage, used instead of `python -m bayesDREAM fit-trans
--config ...` because the plain CLI's `_run_fit_trans` (bayesDREAM/cli.py)
doesn't call `adjust_ntc_sum_factor()`/`refit_sumfactor()` -- every dataset
here needs both (docs/FIT_TRANS_GUIDE.md's documented workflow is
adjust_ntc_sum_factor -> fit_cis -> refit_sumfactor -> fit_trans, and
neither 'sum_factor_adj' nor 'sum_factor_refit' survives a save/load round
trip -- see config_utils.apply_sum_factor_adjustments's docstring for why).
This script adds that step and is otherwise identical to the CLI's trans
stage (same load_ntc/load_cis/fit/save config shape).

Usage
-----
    python run_trans.py --config <gene_config.yaml>

Config (top-level ``trans:`` and ``sum_factor:`` blocks)::

    sum_factor:
      adjust_ntc_sum_factor:
        enabled: true
        args: {covariates: [lane, cell_line]}
      refit_sumfactor:
        enabled: true
        args: {covariates: [lane, cell_line], sum_factor_col_old: sum_factor_adj}

    exclude_trans_genes:                     # optional; calls model.exclude_trans_genes(**args)
      enabled: true                          # (bayesDREAM/model.py -- min_log2_mu_ntc/genes/
      args: {min_log2_mu_ntc: -4.0}          #  feature_query, any combination)

    trans:
      load_ntc: {args: {input_dir: <ntc_shared_dir>, mask_features: true}}
      load_cis: {enabled: true}
      fit:
        sum_factor_col: sum_factor_refit
        function_type: additive_hill
      save: true
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
from resource_stats import (  # noqa: E402
    is_cuda, new_stats_dict, load_prior_stats, timed_attempt, record_trans_step,
)


def run_trans(cfg: dict) -> None:
    model = build_model_from_config(cfg)
    device_is_cuda = is_cuda(model)

    model_cfg = cfg.get("model") or {}
    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
    stats_path = os.path.join(output_dir, "trans_stats.json")
    prior_stats = load_prior_stats(stats_path)
    stats = new_stats_dict(model, extra={"stage": "trans", "label": model_cfg.get("label")})

    trans_cfg = cfg.get("trans") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}

    if is_enabled(trans_cfg.get("load_ntc", trans_cfg.get("load_technical")), default=True):
        model.load_ntc_fit(**normalize_stage_args(trans_cfg.get("load_ntc") or trans_cfg.get("load_technical")))
    if is_enabled(trans_cfg.get("load_cis"), default=True):
        model.load_cis_fit(**normalize_stage_args(trans_cfg.get("load_cis")))

    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {})

    excl_cfg = cfg.get("exclude_trans_genes") or {}
    if is_enabled(excl_cfg, default=False):
        model.exclude_trans_genes(**normalize_stage_args(excl_cfg))

    fit_args = normalize_stage_args(trans_cfg.get("fit"))
    modality_name = fit_args.get("modality_name") or model.primary_modality
    # Explicit (matches fit_trans()'s own default of output_dir/label anyway)
    # so record_trans_step below reads cumulative_elapsed_sec from the exact
    # checkpoint file this call just wrote/resumed, not a guessed default.
    checkpoint_dir = fit_args.setdefault("checkpoint_dir", output_dir)

    with timed_attempt(device_is_cuda) as this_attempt:
        model.fit_trans(**fit_args)
    record_trans_step(stats, stats_path, "fit_trans", modality_name, checkpoint_dir, this_attempt, prior_stats)

    if is_enabled(trans_cfg.get("save"), default=True):
        model.save_trans_fit(**normalize_stage_args(trans_cfg.get("save")))
        # Also required: run_recapitulation_sim.py reads trans_feature_summary_
        # <modality>.csv as its ground truth (comment there: "Must match the
        # real trans run's feature set exactly"). run_permutation_null.py
        # already calls both save_trans_fit AND save_trans_summary for
        # exactly this reason -- this call was missing here, so the REAL
        # trans fit never produced the file recapitulation depends on
        # (confirmed via a real "FileNotFoundError: trans_feature_summary_
        # gene.csv" traceback, 2026-08-14). output_dir/modality_name match
        # save_trans_fit's own defaults.
        model.save_trans_summary(output_dir=output_dir, modality_name=modality_name)

    save_provenance_json(
        os.path.join(output_dir, "provenance_trans.json"),
        extra={"stage": "trans", "label": model_cfg.get("label")},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    run_trans(load_bayesdream_yaml(Path(args.config)))


if __name__ == "__main__":
    main()
