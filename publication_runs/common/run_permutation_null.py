"""
Permutation-null stage: break cis-trans correlation, refit trans, save the
result as one replicate of the expected-false-positive-rate distribution.

Not a bayesDREAM CLI subcommand. Loads ntc + cis fits like the trans stage,
then permutes ALL of one modality's features' NTC expression into perturbed
cells (``permute_from_ntc``) and resamples ``x_true`` among NTC cells
(``model.permute_x_true``) before calling ``fit_trans()`` -- so any
"significant" hit in the resulting summary is by construction a false
positive. Run this multiple times (``--rep``) per dataset/cis-gene/modality
to build up the null distribution; how many is a per-dataset call (see each
dataset's config.yaml `trans.permutation.n_reps` / `modalities.trans.
permutation.n_reps`).

Works on the PRIMARY modality by default, or on any custom modality (e.g.
Domingo's binomial splicing modalities) via an ``attach_modality:`` config
block, which is resolved and called BEFORE anything else -- the target
modality must exist on `model` before `load_ntc_fit()`/`permute_from_ntc()`
can touch it. See domingo/load_modalities.py's `attach_modality()` for the
function this typically points at.

Usage
-----
    python run_permutation_null.py --config <gene_stage_config.yaml> --rep 0

Expects a top-level ``permutation:`` block, e.g.::

    attach_modality:                     # optional; omit for the primary modality
      module: load_modalities
      function: attach_modality
      kwargs:
        spec: {stype: sj, distribution: binomial, gene_alias_col: gene_for_denominator,
               denominator_mode: gene_expression, clip_violations: true}
        base_dir: /path/to/loader_inputs

    permutation:
      modality_name: splicing_sj         # only needed if NOT using attach_modality
      load_ntc: {enabled: true}
      load_cis: {enabled: true}
      covariates: [cell_line]              # passed to both permute_from_ntc and permute_x_true
      sum_factor_col: sum_factor_adj       # passed to both (permute_x_true always uses the
                                            # PRIMARY modality's sum factor, regardless of which
                                            # modality is being permuted -- x_true/cis are always
                                            # negbinom gene-level, independent of the trans modality)
      fit:                                 # passed to fit_trans(), same schema as trans.fit
        sum_factor_col: sum_factor_refit
        function_type: additive_hill
        min_denominator: 0                 # required by fit_trans() for binomial/multinomial
      output_dir: null                     # optional; defaults to
                                            # <output_dir>/<label>/permutation/[<modality_name>/]rep_<rep>
                                            # (no modality_name subfolder for the primary modality,
                                            # to keep existing paths unchanged)
"""

import argparse
import importlib
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
    ensure_dataset_dir_on_syspath,
)
from git_provenance import save_provenance_json  # noqa: E402

from bayesDREAM.simulation import permute_from_ntc  # noqa: E402


def _seed_for(label: str, rep: int) -> int:
    return abs(hash(f"{label}__permutation__rep{rep}")) % (2**32)


def _attach_modality_if_configured(model, cfg: dict):
    """Returns the attached modality's name, or None if no `attach_modality`
    block is present (i.e. this run targets the primary modality)."""
    spec = cfg.get("attach_modality")
    if not spec:
        return None
    ensure_dataset_dir_on_syspath(cfg)
    mod = importlib.import_module(spec["module"])
    fn = getattr(mod, spec["function"])
    return fn(model, **spec.get("kwargs", {}))


def run_permutation_null(cfg: dict, rep: int) -> None:
    import numpy as np
    import pyro
    import torch

    model_cfg = cfg.get("model") or {}
    label = model_cfg.get("label")
    seed = _seed_for(label, rep)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_config(cfg)

    perm_cfg = cfg.get("permutation") or {}
    attached_modality = _attach_modality_if_configured(model, cfg)
    modality_name = attached_modality or perm_cfg.get("modality_name") or model.primary_modality

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}

    if is_enabled(perm_cfg.get("load_ntc", perm_cfg.get("load_technical")), default=True):
        model.load_ntc_fit(**normalize_stage_args(perm_cfg.get("load_ntc") or perm_cfg.get("load_technical")))
    if is_enabled(perm_cfg.get("load_cis"), default=True):
        model.load_cis_fit(**normalize_stage_args(perm_cfg.get("load_cis")))

    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    # Needed for BOTH permute_x_true/permute_from_ntc (sum_factor_adj, by
    # their own default) and fit_trans below (sum_factor_refit) -- see
    # config_utils.apply_sum_factor_adjustments's docstring. Always operates
    # on the PRIMARY modality's sum factors, regardless of which modality is
    # being permuted.
    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {})

    excl_cfg = cfg.get("exclude_trans_genes") or {}
    if is_enabled(excl_cfg, default=False):
        model.exclude_trans_genes(**normalize_stage_args(excl_cfg))

    covariates = perm_cfg.get("covariates")
    sum_factor_col = perm_cfg.get("sum_factor_col", "sum_factor_adj")

    target_mod = model.get_modality(modality_name)
    permute_from_ntc(
        target_mod, model.meta,
        features2permute="All", covariates=covariates,
        sum_factor_col=sum_factor_col, seed=seed,
    )
    model.permute_x_true(covariates=covariates, sum_factor_col=sum_factor_col)

    fit_args = normalize_stage_args(perm_cfg.get("fit"))
    fit_args.setdefault("modality_name", modality_name)
    model.fit_trans(**fit_args)

    output_dir = perm_cfg.get("output_dir")
    if not output_dir:
        base = model_cfg.get("output_dir", "output")
        if modality_name == model.primary_modality:
            output_dir = os.path.join(base, label, "permutation", f"rep_{rep}")
        else:
            output_dir = os.path.join(base, label, "permutation", modality_name, f"rep_{rep}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_trans_fit(output_dir=output_dir, modalities=[modality_name])
    model.save_trans_summary(output_dir=output_dir, modality_name=modality_name)
    print(f"[permutation_null] modality={modality_name} rep={rep} seed={seed} -> {output_dir}")

    save_provenance_json(
        os.path.join(output_dir, "provenance.json"),
        extra={"stage": "permutation_null", "label": label, "modality": modality_name, "rep": rep, "seed": seed},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Path to bayesdream-CLI-schema YAML config.")
    parser.add_argument("--rep", type=int, required=True, help="Replicate index (also seeds RNG).")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    cfg["_dataset_dir"] = cfg.get("_dataset_dir") or str(Path(args.config).resolve().parents[2])
    run_permutation_null(cfg, args.rep)


if __name__ == "__main__":
    main()
