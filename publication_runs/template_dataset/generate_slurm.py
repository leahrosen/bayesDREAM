"""
Template generate_slurm.py -- low-MOI, one shared fit_ntc, per-gene
cis -> compensation -> trans -> permutation -> recapitulation. Copy this
whole directory to publication_runs/<new_name>/, rename it, and adapt:

- If your dataset is high-MOI: `run_cis_deferred.py`/`run_ntc.py` already
  work unchanged (bayesDREAM's high-MOI mode supports deferred `cis_gene` +
  `add_cis_gene()` natively) -- just add `guide_assignment`/`guide_meta`/
  `guide_target` to `data_block`. See morris/generate_slurm.py for the
  worked example, including its per-gene `exclude_guides` handling (a
  bayesDREAM constructor-time filter that has to be threaded consistently
  through every one of a gene's stages -- see morris/README.md) and its
  note on a pre-existing `fit_cis()` dtype bug in high-MOI mode.
- If genes DON'T share one fit_ntc: drop the ntc_shared step and give each
  gene its own `model.cis_gene` set directly + a plain `fit-ntc` stage
  (cis_gene set at construction skips the deferred-workflow machinery
  entirely).
- If you have extra modalities: see domingo/generate_slurm.py +
  domingo/load_modalities.py for the pattern (per-modality config block,
  one job per gene x modality, depends on that gene's cis job).

Usage
-----
    python generate_slurm.py [--config config.yaml] [--outdir slurm]
"""

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR_LOCAL = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common"))
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common" / "slurm"))

from config_utils import load_yaml, write_yaml, render_bayesdream_config  # noqa: E402
from git_provenance import create_stable_snapshot_tag  # noqa: E402
from sbatch_blocks import SbatchStep, SbatchArray  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(THIS_DIR / "config.yaml"))
    parser.add_argument("--outdir", default=str(THIS_DIR / "slurm"))
    parser.add_argument("--no-tag", action="store_true")
    parser.add_argument("--no-push-tag", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    outdir = Path(args.outdir)
    configs_dir = outdir / "configs"
    logs_dir = outdir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag_info = {"bayesdream_tag": None}
    if not args.no_tag:
        tag_info = create_stable_snapshot_tag(prefix=f"{dataset}-run", push=not args.no_push_tag)
    print(f"[generate_slurm] git tag for this batch: {tag_info.get('bayesdream_tag')}")

    paths = cfg["paths"]
    meta_path = paths["meta"].format(**paths)
    counts_path = paths["counts"].format(**paths)
    output_dir = paths["output_dir"]
    repo_dir = paths["repo_dir"]
    python_env = paths["python_env"]

    cluster = cfg["cluster"]
    account = cluster["account"]
    partition_cpu = cluster.get("partition_cpu", "shared")

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    ntc_shared_cfg = cfg["ntc_shared"]
    cis_cfg = cfg["cis"]
    comp_cfg = cfg["compensation"]
    trans_cfg = cfg["trans"]

    base_cfg = {
        "data": {"meta": meta_path, "counts": counts_path},
        "model": {
            "modality_name": model_defaults["modality_name"],
            "guide_covariates": model_defaults["guide_covariates"],
            "guide_covariates_ntc": model_defaults["guide_covariates_ntc"],
            "output_dir": output_dir,
        },
        "ntc": {"set_technical_groups": ntc_shared_cfg["set_technical_groups"]},
    }

    sum_factor_block = {
        "adjust_ntc_sum_factor": {
            "enabled": cis_cfg["adjust_ntc_sum_factor"]["enabled"],
            "args": {"covariates": cis_cfg["adjust_ntc_sum_factor"]["covariates"]},
        },
        "refit_sumfactor": {
            "enabled": True,
            "args": {"covariates": cis_cfg["adjust_ntc_sum_factor"]["covariates"],
                      "sum_factor_col_old": "sum_factor_adj"},
        },
    }

    def bd_cmd(kind: str, config_path, extra_args: str = "") -> str:
        if kind in {"fit-ntc", "fit-cis", "fit-trans", "report"}:
            return f'cd "{repo_dir}" && "{python_env}" -m bayesDREAM {kind} --config "{config_path}"{extra_args}'
        script = f"{repo_dir}/publication_runs/common/run_{kind}.py"
        return f'"{python_env}" "{script}" --config "{config_path}"{extra_args}'

    scripts = []

    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    ntc_bd_cfg = render_bayesdream_config(base_cfg, {"model": {"label": label_ntc}, "ntc": {"fit": {}, "save": True}})
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_bd_cfg)
    ntc_step = SbatchStep(
        job_name=f"{dataset}_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=ntc_shared_cfg["resources"]["time_hours"], cpus=ntc_shared_cfg["resources"]["cores"],
        partition=partition_cpu, repo_dir=repo_dir, commands=[bd_cmd("ntc", ntc_cfg_path)],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))

    for gene in cfg["cis_genes"]:
        label = f"{label_prefix}_{gene}"

        cis_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label}, "cis_gene": gene, "ntc_shared_dir": ntc_shared_dir,
            "cis": {
                "adjust_ntc_sum_factor": sum_factor_block["adjust_ntc_sum_factor"],
                "fit": {"sum_factor_col": "sum_factor_adj" if cis_cfg["adjust_ntc_sum_factor"]["enabled"] else "sum_factor"},
                "save": True,
            },
        })
        cis_cfg_path = configs_dir / f"{label}_cis.yaml"
        write_yaml(cis_cfg_path, cis_bd_cfg)
        cis_step = SbatchStep(
            job_name=f"{dataset}_cis_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=cis_cfg["resources"]["time_hours"], cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[bd_cmd("cis_deferred", cis_cfg_path)],
        )
        scripts.append((f"02_cis_{gene}.sh", cis_step.render()))

        comp_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene}, "sum_factor": sum_factor_block,
            "compensation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "args": {"sum_factor_col": "sum_factor_refit", "exclude_cells": comp_cfg["exclude_cells"].get(gene, [])},
            },
        })
        comp_cfg_path = configs_dir / f"{label}_compensation.yaml"
        write_yaml(comp_cfg_path, comp_bd_cfg)
        comp_step = SbatchStep(
            job_name=f"{dataset}_comp_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=comp_cfg["resources"]["time_hours"], cpus=comp_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[bd_cmd("compensation", comp_cfg_path)],
        )
        scripts.append((f"03_compensation_{gene}.sh", comp_step.render()))

        trans_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene}, "sum_factor": sum_factor_block,
            "trans": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
                "save": True,
            },
        })
        trans_cfg_path = configs_dir / f"{label}_trans.yaml"
        write_yaml(trans_cfg_path, trans_bd_cfg)
        trans_step = SbatchStep(
            job_name=f"{dataset}_trans_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["resources"]["time_hours"], cpus=trans_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[bd_cmd("trans", trans_cfg_path)],
        )
        scripts.append((f"04_trans_{gene}.sh", trans_step.render()))

        perm_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene}, "sum_factor": sum_factor_block,
            "permutation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "covariates": cis_cfg["adjust_ntc_sum_factor"]["covariates"], "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
            },
        })
        perm_cfg_path = configs_dir / f"{label}_permutation.yaml"
        write_yaml(perm_cfg_path, perm_bd_cfg)
        n_perm = trans_cfg["permutation"]["n_reps"]
        perm_step = SbatchArray(
            job_name=f"{dataset}_perm_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["permutation"]["resources"]["time_hours"], cpus=trans_cfg["permutation"]["resources"]["cores"],
            max_index=n_perm - 1, max_concurrent=min(n_perm, 50), partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("permutation_null", perm_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
        )
        scripts.append((f"05_permutation_{gene}.sh", perm_step.render()))

        sim_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene}, "sum_factor": sum_factor_block,
            "simulation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True}, "load_trans": {"enabled": True},
                "sum_factor_col": "sum_factor_refit",
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
            },
        })
        sim_cfg_path = configs_dir / f"{label}_simulation.yaml"
        write_yaml(sim_cfg_path, sim_bd_cfg)
        n_sim = trans_cfg["simulation"]["n_reps"]
        sim_step = SbatchArray(
            job_name=f"{dataset}_sim_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["simulation"]["resources"]["time_hours"], cpus=trans_cfg["simulation"]["resources"]["cores"],
            max_index=n_sim - 1, max_concurrent=min(n_sim, 50), partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("recapitulation_sim", sim_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
        )
        scripts.append((f"06_recapitulation_{gene}.sh", sim_step.render()))

    for filename, text in scripts:
        (outdir / filename).write_text(text)
        os.chmod(outdir / filename, 0o755)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "",
                     'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)', 'echo "ntc_shared: $NTC_JOB"', ""]
    for gene in cfg["cis_genes"]:
        submit_lines += [
            f'CIS_{gene}=$(sbatch --parsable --dependency=afterok:$NTC_JOB 02_cis_{gene}.sh)',
            f'COMP_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 03_compensation_{gene}.sh)',
            f'TRANS_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 04_trans_{gene}.sh)',
            f'sbatch --dependency=afterok:$TRANS_{gene} 05_permutation_{gene}.sh',
            f'sbatch --dependency=afterok:$TRANS_{gene} 06_recapitulation_{gene}.sh', "",
        ]
    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")


if __name__ == "__main__":
    main()
