"""
Standalone fit_ntc stage, using config_utils.build_model_from_config (the
extended builder -- supports exclude_guides/min_count/sparse .npz counts,
none of which bayesDREAM.cli's own `fit-ntc` subcommand forwards, see
config_utils.py's module docstring). Used for every dataset's SHARED
fit_ntc build (cis_gene omitted -- deferred) so every dataset goes through
the same model-construction code path, rather than some datasets using the
plain CLI and others needing a bespoke script.

As of bayesDREAM's "Support deferred cis_gene in high-MOI mode" change,
this works identically for low-MOI and high-MOI datasets: cis_gene omitted,
optionally guide_assignment/guide_meta/guide_target present for high-MOI.
fit_ntc()'s own use_all_cells defaults to True automatically when high-MOI
+ cis_gene is still unset (see bayesDREAM/fitting/ntc.py) -- no special
handling needed here.

Usage
-----
    python run_ntc.py --config <ntc_shared_config.yaml>

Config is the standard bayesdream-CLI-schema (data:/model:/ntc:), with
model.cis_gene omitted::

    data:
      meta: /path/to/meta.csv
      counts: /path/to/gene_counts.csv          # or .npz for sparse
      guide_assignment: /path/to/guide_assignment.npy   # high-MOI only
      guide_meta: /path/to/guide_meta.csv
      guide_target: /path/to/guide_target.csv

    model:
      # cis_gene intentionally absent -- deferred
      exclude_guides: [sgFOO_1, sgFOO_2]   # optional
      min_count: 10                         # optional
      guide_covariates: [lane]
      output_dir: /path/to/output
      label: morris_20260806_ntc_shared

    ntc:
      set_technical_groups: [lane]
      fit: {}
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
    apply_device_override,
)
from git_provenance import save_provenance_json  # noqa: E402
from resource_stats import (  # noqa: E402
    is_cuda, new_stats_dict, load_prior_stats, step_completed,
    carry_forward_step, timed_step,
)


def run_ntc(cfg: dict) -> None:
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("cis_gene"):
        raise ValueError(
            "run_ntc: config's model.cis_gene should be omitted for a shared/deferred "
            "fit_ntc run -- found model.cis_gene={!r}. If cis_gene is meant to be fixed, "
            "use the plain `bayesDREAM fit-ntc` CLI stage instead.".format(model_cfg["cis_gene"])
        )

    model = build_model_from_config(cfg)
    device_is_cuda = is_cuda(model)

    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
    stats_path = os.path.join(output_dir, "ntc_stats.json")
    prior_stats = load_prior_stats(stats_path)
    stats = new_stats_dict(model, extra={"stage": "ntc", "label": model_cfg.get("label")})

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    # fit_ntc() has no internal checkpoint (see resource_stats.py's module
    # docstring) -- a killed/timed-out attempt leaves nothing to resume mid-
    # fit, so "restarting" this job (a manual resubmit; ntc/cis/compensation
    # deliberately do NOT auto-requeue, see publication_runs/README.md)
    # either finds a fully-completed prior attempt (skip, just load + reuse
    # its recorded stats) or starts completely fresh.
    if step_completed(prior_stats, "fit_ntc"):
        carry_forward_step(stats, stats_path, "fit_ntc", prior_stats)
        model.load_ntc_fit()
    else:
        fit_args = normalize_stage_args(ntc_cfg.get("fit"))
        with timed_step("fit_ntc", stats, device_is_cuda, stats_path):
            model.fit_ntc(**fit_args)
        if is_enabled(ntc_cfg.get("save"), default=True):
            model.save_ntc_fit(**normalize_stage_args(ntc_cfg.get("save")))

    save_provenance_json(
        os.path.join(output_dir, "provenance_ntc.json"),
        extra={"stage": "ntc", "label": model_cfg.get("label"), "is_high_moi": model.is_high_moi},
    )
    print(f"[run_ntc] fit_ntc complete (is_high_moi={model.is_high_moi}) -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--device", default=None, help="Explicit device override, e.g. 'cuda:2' (see config_utils.apply_device_override).")
    args = parser.parse_args()
    cfg = load_bayesdream_yaml(Path(args.config))
    apply_device_override(cfg, args.device)
    run_ntc(cfg)


if __name__ == "__main__":
    main()
