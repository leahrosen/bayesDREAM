"""
Write trans_feature_summary_<modality>.csv for an ALREADY-completed real
trans fit that predates run_trans.py calling save_trans_summary() (fixed
2026-08-14 -- see that script's module docstring). Loads the existing saved
fit instead of re-running fit_trans() (which can take hours) -- this only
reads posterior_samples_trans back off disk and writes the summary CSV.

Only needed for trans fits that already completed under the OLD run_trans.py
(missing the summary write). Any trans job run after the fix writes this
file itself; you never need this script for those.

Usage
-----
    python backfill_trans_summary.py --config <label>_trans.yaml
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Same <label>_trans.yaml the real trans job used.")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    model = build_model_from_config(cfg)

    model_cfg = cfg.get("model") or {}
    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
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

    # Loads the EXISTING saved posterior -- does not fit anything.
    # subset_features=True: exclude_trans_genes()'s min_log2_mu_ntc filter
    # isn't perfectly reproducible run-to-run (same reason
    # run_recapitulation_sim.py needs it, see 9b8c623) -- this call above
    # mirrors run_trans.py's own ordering exactly, but a fresh evaluation
    # could still land on a slightly different feature set than the
    # ORIGINAL fit_trans() run's did.
    model.load_trans_fit(modalities=[modality_name], subset_features=True)
    model.save_trans_summary(output_dir=output_dir, modality_name=modality_name)
    print(f"[backfill_trans_summary] wrote trans_feature_summary_{modality_name}.csv -> {output_dir}")


if __name__ == "__main__":
    main()
