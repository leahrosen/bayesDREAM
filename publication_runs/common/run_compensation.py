"""
Standalone "compensation" stage: model.check_systematic_shift().

Not a bayesDREAM CLI subcommand (see publication_runs/README.md), so this
script builds the model itself, loads the ntc + cis fits, and calls
check_systematic_shift() directly. Mirrors the load/fit/save structure of
bayesDREAM.cli._run_fit_cis so it slots into the same config file the cis/
trans stages use.

Usage
-----
    python run_compensation.py --config <path/to/gene_stage_config.yaml>

Also expects the same top-level ``sum_factor:`` block used by run_trans.py
(adjust_ntc_sum_factor + refit_sumfactor are re-run here too -- neither
survives a save/load round trip, see
config_utils.apply_sum_factor_adjustments's docstring), so that
``sum_factor_refit`` matches what fit_trans() actually used.

Expects a top-level ``compensation:`` block in the config, e.g.::

    compensation:
      load_ntc: {enabled: true}
      load_cis: {enabled: true}
      args:
        sum_factor_col: sum_factor_refit   # must match what fit_trans() will use
        exclude_cells: exclude_cells.txt   # optional: path, one cell name per line
        min_cells_per_group: 30
      output: compensation_shift.csv       # optional; defaults to
                                            # <output_dir>/<label>/compensation_<modality>.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import bayesDREAM`

from config_utils import (  # noqa: E402
    build_model_from_config,
    load_bayesdream_yaml,
    normalize_stage_args,
    is_enabled,
    apply_sum_factor_adjustments,
)
from git_provenance import save_provenance_json  # noqa: E402


def _load_exclude_cells(path: str):
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to the config file's directory, not the CWD.
        p = Path(path)
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


def run_compensation(cfg: dict) -> "object":
    import pandas as pd  # local import: keep module import cheap for --help

    model = build_model_from_config(cfg)

    comp_cfg = cfg.get("compensation") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    cis_cfg = cfg.get("cis") or {}

    if is_enabled(comp_cfg.get("load_ntc", comp_cfg.get("load_technical")), default=True):
        load_args = normalize_stage_args(comp_cfg.get("load_ntc") or comp_cfg.get("load_technical"))
        model.load_ntc_fit(**load_args)

    if is_enabled(comp_cfg.get("load_cis"), default=True):
        load_args = normalize_stage_args(comp_cfg.get("load_cis"))
        model.load_cis_fit(**load_args)

    # Same fixup fit_cis/fit_trans apply: a freshly-built model object may not
    # have technical_group_code set even though alpha_x_prefit/x_true loaded fine.
    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {})

    args = normalize_stage_args(comp_cfg.get("args"))
    if isinstance(args.get("exclude_cells"), str):
        args["exclude_cells"] = _load_exclude_cells(args["exclude_cells"])

    result = model.check_systematic_shift(**args)

    modality_name = args.get("modality_name") or model.primary_modality
    output_dir = (cfg.get("model") or {}).get("output_dir", "output")
    label = (cfg.get("model") or {}).get("label")
    default_out = os.path.join(output_dir, label, f"compensation_{modality_name}.csv")
    out_path = comp_cfg.get("output", default_out)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[compensation] wrote {len(result)} rows -> {out_path}")

    save_provenance_json(
        out_path + ".provenance.json",
        extra={"stage": "compensation", "label": label, "modality": modality_name},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Path to bayesdream-CLI-schema YAML config.")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    run_compensation(cfg)


if __name__ == "__main__":
    main()
