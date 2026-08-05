"""
Build and fit the ONE shared fit_ntc() run for a high-MOI dataset, over the
full gene panel (minus one placeholder cis gene -- high-MOI requires
cis_gene at construction, see apply_shared_ntc_high_moi.py's docstring).

Not usable via bayesDREAM.cli's _build_model, because that helper doesn't
forward `exclude_guides` to the constructor (only the keys in its
allowed_model_keys set -- see bayesDREAM/cli.py's _build_model -- which
`exclude_guides` isn't currently in). Rather than editing bayesDREAM/cli.py
(a core-code change; per project convention that needs explicit sign-off,
see CLAUDE.md), this script constructs the model directly for this one
dataset-setup step. Every other stage in this pipeline (cis/compensation/
trans/...) uses the ordinary CLI-schema config + build_model_from_config,
since exclude_guides is only relevant to constructing the ORIGINAL model
guide_assignment/guide_meta.

Usage
-----
    python build_ntc_shared_high_moi.py --config <ntc_shared_config.yaml>

Config schema (its own, NOT the bayesdream-CLI schema)::

    data:
      meta: /path/to/meta.csv
      counts: /path/to/gene_counts.csv
      counts_read_csv_kwargs: {index_col: 0}
      guide_assignment: /path/to/guide_assignment.npy
      guide_meta: /path/to/guide_meta.csv
      guide_target: /path/to/guide_target.csv   # optional

    model:
      placeholder_cis_gene: ACTB     # any gene NOT in your target gene list
      exclude_guides: [sgFOO_1, sgFOO_2]
      guide_covariates: [cell_line]
      guide_covariates_ntc: [cell_line]
      output_dir: /path/to/output
      label: morris_20260731_ntc_shared

    ntc:
      set_technical_groups: [cell_line]
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
    load_bayesdream_yaml,
    normalize_stage_args,
    is_enabled,
    read_table,
    load_guide_assignment,
)
from git_provenance import save_provenance_json  # noqa: E402

from bayesDREAM import bayesDREAM  # noqa: E402


def build_ntc_shared_high_moi(cfg: dict) -> None:
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    ntc_cfg = cfg.get("ntc") or {}

    meta = read_table(data_cfg["meta"], data_cfg.get("meta_read_csv_kwargs"))
    counts = read_table(data_cfg["counts"], data_cfg.get("counts_read_csv_kwargs") or {"index_col": 0})
    guide_assignment = load_guide_assignment(data_cfg["guide_assignment"])
    guide_meta = read_table(data_cfg["guide_meta"], data_cfg.get("guide_meta_read_csv_kwargs"))
    guide_target = None
    if data_cfg.get("guide_target"):
        guide_target = read_table(data_cfg["guide_target"], data_cfg.get("guide_target_read_csv_kwargs"))

    placeholder_cis_gene = model_cfg["placeholder_cis_gene"]
    exclude_guides = model_cfg.get("exclude_guides")

    model = bayesDREAM(
        meta=meta, counts=counts,
        guide_assignment=guide_assignment, guide_meta=guide_meta, guide_target=guide_target,
        cis_gene=placeholder_cis_gene,
        exclude_guides=exclude_guides,
        guide_covariates=model_cfg.get("guide_covariates"),
        guide_covariates_ntc=model_cfg.get("guide_covariates_ntc"),
        output_dir=model_cfg.get("output_dir", "output"),
        label=model_cfg["label"],
    )

    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    fit_args = normalize_stage_args(ntc_cfg.get("fit"))
    model.fit_ntc(**fit_args)

    if is_enabled(ntc_cfg.get("save"), default=True):
        model.save_ntc_fit(**normalize_stage_args(ntc_cfg.get("save")))

    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg["label"])
    save_provenance_json(
        os.path.join(output_dir, "provenance_ntc_shared.json"),
        extra={"stage": "ntc_shared_high_moi", "label": model_cfg["label"],
               "placeholder_cis_gene": placeholder_cis_gene, "exclude_guides": exclude_guides},
    )
    print(f"[build_ntc_shared_high_moi] fit_ntc complete (placeholder_cis_gene={placeholder_cis_gene}) -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    build_ntc_shared_high_moi(load_bayesdream_yaml(Path(args.config)))


if __name__ == "__main__":
    main()
