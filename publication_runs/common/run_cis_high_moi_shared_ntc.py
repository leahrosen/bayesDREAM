"""
High-MOI fit_cis stage that reuses one shared fit_ntc() run across many cis
genes via apply_shared_ntc_high_moi.apply_shared_ntc() -- the high-MOI
counterpart of run_cis_deferred.py (which uses add_cis_gene(), unavailable
in high-MOI mode; see apply_shared_ntc_high_moi.py's docstring for the full
explanation and its validation caveat).

Unlike run_cis_deferred.py, ``model.cis_gene`` here MUST be set in the
config (high-MOI requires it at construction time) -- there is no deferred
state, apply_shared_ntc() just seeds alpha_x/alpha_y onto the already-built,
already-gene-specific model instead of running fit_ntc() again.

Usage
-----
    python run_cis_high_moi_shared_ntc.py --config <gene_config.yaml>

Config needs the usual high-MOI model block (cis_gene, guide_assignment,
guide_meta, guide_target) plus a top-level ``ntc_shared_dir:``::

    model:
      cis_gene: GFI1B
      output_dir: /path/to/output
      label: morris_20260731_GFI1B
      ...

    ntc_shared_dir: /path/to/output/morris_20260731_ntc_shared

    ntc:
      set_technical_groups: [cell_line]

    sum_factor:                       # compute_scran is per-cell-subset (see
      compute_scran:                  # compute_scran_sum_factor.py) -- must be
        enabled: true                 # recomputed HERE, on this gene's own subset,
        args: {batch_col: lane}       # not reused from the shared ntc run.
      adjust_ntc_sum_factor:
        enabled: true
        args: {sum_factor_col_old: sum_factor_new, covariates: [lane]}

    cis:
      fit:
        sum_factor_col: sum_factor_adj
        independent_mu_sigma: true
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
from apply_shared_ntc_high_moi import apply_shared_ntc  # noqa: E402


def run_cis_high_moi_shared_ntc(cfg: dict) -> None:
    model_cfg = cfg.get("model") or {}
    if not model_cfg.get("cis_gene"):
        raise ValueError("run_cis_high_moi_shared_ntc: config's model.cis_gene is required (high-MOI).")

    ntc_shared_dir = cfg.get("ntc_shared_dir")
    if not ntc_shared_dir:
        raise ValueError("run_cis_high_moi_shared_ntc: config needs a top-level 'ntc_shared_dir' key.")

    model = build_model_from_config(cfg)
    if not model.is_high_moi:
        raise ValueError(
            "run_cis_high_moi_shared_ntc: model is not high-MOI -- use run_cis_deferred.py instead."
        )

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    apply_shared_ntc(model, ntc_shared_dir=ntc_shared_dir, modality_name=model_cfg.get("modality_name"))

    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {}, steps=("compute_scran", "adjust_ntc_sum_factor"))

    cis_cfg = cfg.get("cis") or {}
    fit_args = normalize_stage_args(cis_cfg.get("fit"))
    model.fit_cis(**fit_args)

    if is_enabled(cis_cfg.get("save"), default=True):
        model.save_cis_fit(**normalize_stage_args(cis_cfg.get("save")))

    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
    save_provenance_json(
        os.path.join(output_dir, "provenance_cis.json"),
        extra={"stage": "cis_high_moi_shared_ntc", "label": model_cfg.get("label"),
               "cis_gene": model_cfg.get("cis_gene"), "ntc_shared_dir": ntc_shared_dir},
    )
    print(f"[run_cis_high_moi_shared_ntc] {model_cfg.get('cis_gene')}: fit_cis complete, "
          f"ntc reused from {ntc_shared_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    run_cis_high_moi_shared_ntc(load_bayesdream_yaml(Path(args.config)))


if __name__ == "__main__":
    main()
