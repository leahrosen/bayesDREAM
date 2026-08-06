"""
Low-MOI "deferred cis gene" fit_cis stage: reuse one shared fit_ntc() run
across many cis genes via add_cis_gene() (see CLAUDE.md's "Deferred
Cis-Gene Workflow"). Not usable for high-MOI datasets -- add_cis_gene()
itself raises ValueError there; see apply_shared_ntc_high_moi.py for the
high-MOI equivalent.

Why this needs its own script rather than `python -m bayesDREAM fit-cis
--config ...`: the plain CLI's _build_model always passes `model.cis_gene`
straight to the bayesDREAM constructor, which carves the cis gene out of the
primary modality immediately -- there is no way to build a model with
cis_gene deferred and then call add_cis_gene() through the existing CLI
subcommands. This script does exactly that, as a separate process per gene,
loading the ALREADY-FITTED shared ntc posteriors from disk (fit_ntc() itself
only ever needs to run once, in a separate job -- see each dataset's
ntc_shared stage).

Usage
-----
    python run_cis_deferred.py --config <gene_config.yaml>

Config must OMIT model.cis_gene (deferred), and must have a top-level
``cis_gene:`` key naming the gene to commit to via add_cis_gene(), plus an
``ntc_shared_dir:`` key pointing at the shared fit_ntc() output directory::

    model:
      # cis_gene intentionally absent here
      output_dir: /path/to/output
      label: domingo_20260731_GFI1B
      guide_covariates: [cell_line]

    cis_gene: GFI1B
    ntc_shared_dir: /path/to/output/domingo_20260731_ntc_shared

    ntc:
      set_technical_groups: [cell_line]

    sum_factor:                     # only compute_scran/adjust_ntc_sum_factor run at
      adjust_ntc_sum_factor:        # this stage (steps= restricts to a prefix) --
        enabled: true                # refit_sumfactor needs x_true, which doesn't
        args: {covariates: [lane, cell_line]}   # exist until AFTER fit_cis.

    cis:
      fit:
        sum_factor_col: sum_factor_adj
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


def run_cis_deferred(cfg: dict) -> None:
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("cis_gene"):
        raise ValueError(
            "run_cis_deferred: config's model.cis_gene must be omitted (deferred) -- "
            "found model.cis_gene={!r}. Use the plain `bayesDREAM fit-cis` CLI stage "
            "instead if cis_gene is meant to be set at construction time.".format(model_cfg["cis_gene"])
        )

    cis_gene = cfg.get("cis_gene")
    ntc_shared_dir = cfg.get("ntc_shared_dir")
    if not cis_gene or not ntc_shared_dir:
        raise ValueError("run_cis_deferred: config needs top-level 'cis_gene' and 'ntc_shared_dir' keys.")

    model = build_model_from_config(cfg)
    if model.is_high_moi:
        raise ValueError(
            "run_cis_deferred: model is high-MOI -- add_cis_gene() is not supported "
            "for high-MOI models (cis_gene must be set at init). Use "
            "apply_shared_ntc_high_moi.py instead."
        )

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    model.load_ntc_fit(input_dir=ntc_shared_dir, mask_features=True)
    model.add_cis_gene(cis_gene)

    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {}, steps=("compute_scran", "adjust_ntc_sum_factor"))

    cis_cfg = cfg.get("cis") or {}
    fit_args = normalize_stage_args(cis_cfg.get("fit"))
    model.fit_cis(**fit_args)

    if is_enabled(cis_cfg.get("save"), default=True):
        model.save_cis_fit(**normalize_stage_args(cis_cfg.get("save")))

    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
    save_provenance_json(
        os.path.join(output_dir, "provenance_cis.json"),
        extra={"stage": "cis_deferred", "label": model_cfg.get("label"), "cis_gene": cis_gene,
               "ntc_shared_dir": ntc_shared_dir},
    )
    print(f"[run_cis_deferred] {cis_gene}: fit_cis complete, ntc reused from {ntc_shared_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    run_cis_deferred(load_bayesdream_yaml(Path(args.config)))


if __name__ == "__main__":
    main()
