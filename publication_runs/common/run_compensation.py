"""
Standalone "compensation" stage: model.check_systematic_shift().

Not a bayesDREAM CLI subcommand (see publication_runs/README.md), so this
script builds the model itself, loads the ntc + cis fits, and calls
check_systematic_shift() directly. Mirrors the load/fit/save structure of
bayesDREAM.cli._run_fit_cis so it slots into the same config file the cis/
trans stages use.

Deliberately does NOT call adjust_ntc_sum_factor()/refit_sumfactor() --
check_systematic_shift() always uses the raw 'sum_factor' column (its own
default) here, by design: zoomed in along the x-axis the raw factor is
close enough, and both reference pipelines (Domingo, Morris) call it with
no sum_factor_col override.

Usage
-----
    python run_compensation.py --config <path/to/gene_stage_config.yaml>

Expects a top-level ``compensation:`` block in the config, e.g.::

    compensation:
      load_ntc: {enabled: true}
      load_cis: {enabled: true}
      args:
        exclude_cells: exclude_cells.txt   # optional: path, one cell name per line
                                            # OR a literal list
                                            # OR {module: ..., function: ..., kwargs: {...}}
                                            #    to compute it dynamically (e.g. Morris'
                                            #    padj-based rule, see
                                            #    morris/compensation_exclude_cells.py)
        min_cells_per_group: 30
      output: compensation_shift.csv       # optional; defaults to
                                            # <output_dir>/<label>/compensation_<modality>.csv
"""

import argparse
import importlib
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
    ensure_dataset_dir_on_syspath,
)
from git_provenance import save_provenance_json  # noqa: E402


def _load_exclude_cells_file(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_exclude_cells(spec, model, cfg):
    """spec is one of: None, a list of cell names, a path (str) to a
    newline-delimited file, or a dict {module, function, kwargs} naming a
    dataset-specific function ``fn(model, cfg, **kwargs) -> list[str]`` to
    call dynamically (e.g. morris/compensation_exclude_cells.py)."""
    if spec is None:
        return None
    if isinstance(spec, list):
        return spec
    if isinstance(spec, str):
        return _load_exclude_cells_file(spec)
    if isinstance(spec, dict):
        # BUG FIX: this used to try (config file's own directory, common/'s
        # parent) as sys.path candidates -- neither is where a dataset's own
        # module (e.g. morris/compensation_exclude_cells.py) actually lives,
        # so this import silently could never have succeeded at runtime.
        # generate_slurm.py now stamps `_dataset_dir` into every rendered
        # config for exactly this.
        ensure_dataset_dir_on_syspath(cfg)
        mod = importlib.import_module(spec["module"])
        fn = getattr(mod, spec["function"])
        return fn(model, cfg, **spec.get("kwargs", {}))
    raise ValueError(f"run_compensation: unsupported exclude_cells spec type: {type(spec)}")


def run_compensation(cfg: dict) -> "object":
    model = build_model_from_config(cfg)

    comp_cfg = cfg.get("compensation") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}

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

    args = normalize_stage_args(comp_cfg.get("args"))
    args["exclude_cells"] = _resolve_exclude_cells(args.get("exclude_cells"), model, cfg)

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
    cfg["_config_dir"] = str(Path(args.config).resolve().parent)
    run_compensation(cfg)


if __name__ == "__main__":
    main()
