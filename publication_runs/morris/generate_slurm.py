"""
Generate SLURM scripts + configs for the Morris dataset: high-MOI, one
shared fit_ntc (over the full panel minus a placeholder cis gene -- see
morris/README.md), 5 primary genes with the full pipeline (cis ->
compensation -> trans -> permutation -> recapitulation), and a fit_cis-ONLY
sweep over ~hundreds more genes from genes_all.csv, all reusing the one
shared fit_ntc via apply_shared_ntc_high_moi.py's workaround.

Usage
-----
    python generate_slurm.py [--config config.yaml] [--outdir slurm]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR_LOCAL = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common"))
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common" / "slurm"))

from config_utils import load_yaml, write_yaml, render_bayesdream_config  # noqa: E402
from git_provenance import create_stable_snapshot_tag  # noqa: E402
from sbatch_blocks import SbatchStep, SbatchArray, SbatchGpuNodeQueue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(THIS_DIR / "config.yaml"))
    parser.add_argument("--outdir", default=str(THIS_DIR / "slurm"))
    parser.add_argument("--no-tag", action="store_true")
    parser.add_argument("--no-push-tag", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    outdir = Path(args.outdir)
    configs_dir = outdir / "configs"
    logs_dir = outdir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag_info = {"bayesdream_tag": None}
    if not args.no_tag:
        tag_info = create_stable_snapshot_tag(prefix=f"{cfg['dataset']}-run", push=not args.no_push_tag)
    print(f"[generate_slurm] git tag for this batch: {tag_info.get('bayesdream_tag')}")

    paths = cfg["paths"]
    meta_path = paths["meta"].format(**paths)
    counts_path = paths["counts"].format(**paths)
    guide_assignment_path = paths["guide_assignment"].format(**paths)
    guide_meta_path = paths["guide_meta"].format(**paths)
    guide_target_path = paths.get("guide_target", "").format(**paths) if paths.get("guide_target") else None
    output_dir = paths["output_dir"]
    repo_dir = paths["repo_dir"]
    python_env = paths["python_env"]

    cluster = cfg["cluster"]
    account = cluster["account"]
    partition_cpu = cluster.get("partition_cpu", "shared")

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    exclude_guides = cfg.get("exclude_guides") or []
    ntc_shared_cfg = cfg["ntc_shared"]
    cis_cfg = cfg["cis"]
    comp_cfg = cfg["compensation"]
    trans_cfg = cfg["trans"]
    cis_sweep_cfg = cfg["cis_sweep"]

    data_block = {
        "meta": meta_path, "counts": counts_path,
        "counts_read_csv_kwargs": {"index_col": 0},
        "guide_assignment": guide_assignment_path,
        "guide_meta": guide_meta_path,
    }
    if guide_target_path:
        data_block["guide_target"] = guide_target_path

    base_cfg = {
        "data": data_block,
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

    # ---------------------------------------------------------------- #
    # 1. Shared fit_ntc (high-MOI workaround -- own config schema)       #
    # ---------------------------------------------------------------- #
    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    ntc_own_cfg = {
        "data": data_block,
        "model": {
            "placeholder_cis_gene": ntc_shared_cfg["placeholder_cis_gene"],
            "exclude_guides": exclude_guides,
            "guide_covariates": model_defaults["guide_covariates"],
            "guide_covariates_ntc": model_defaults["guide_covariates_ntc"],
            "output_dir": output_dir,
            "label": label_ntc,
        },
        "ntc": {"set_technical_groups": ntc_shared_cfg["set_technical_groups"], "fit": {}, "save": True},
    }
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_own_cfg)

    ntc_step = SbatchStep(
        job_name="morris_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=ntc_shared_cfg["resources"]["time_hours"], cpus=ntc_shared_cfg["resources"]["cores"],
        partition=partition_cpu, repo_dir=repo_dir,
        commands=[
            f'"{python_env}" "{repo_dir}/publication_runs/common/build_ntc_shared_high_moi.py" --config "{ntc_cfg_path}"'
        ],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))

    def render_gene_cfg(gene: str, label: str, extra: dict) -> dict:
        return render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            **extra,
        })

    # ---------------------------------------------------------------- #
    # 2. Primary genes: cis -> compensation -> trans -> perm/sim          #
    # ---------------------------------------------------------------- #
    for gene in cfg["primary_genes"]:
        label = f"{label_prefix}_{gene}"

        cis_bd_cfg = render_gene_cfg(gene, label, {
            "ntc_shared_dir": ntc_shared_dir,
            "cis": {
                "adjust_ntc_sum_factor": sum_factor_block["adjust_ntc_sum_factor"],
                "fit": {"sum_factor_col": "sum_factor_adj" if cis_cfg["adjust_ntc_sum_factor"]["enabled"] else "sum_factor"},
                "save": True,
            },
        })
        cis_cfg_path = configs_dir / f"{label}_cis.yaml"
        write_yaml(cis_cfg_path, cis_bd_cfg)
        cis_step = SbatchStep(
            job_name=f"morris_cis_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=cis_cfg["resources"]["time_hours"], cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("cis_high_moi_shared_ntc", cis_cfg_path)],
        )
        scripts.append((f"02_cis_{gene}.sh", cis_step.render()))

        comp_bd_cfg = render_gene_cfg(gene, label, {
            "sum_factor": sum_factor_block,
            "compensation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "args": {"sum_factor_col": "sum_factor_refit", "exclude_cells": comp_cfg["exclude_cells"].get(gene, [])},
            },
        })
        comp_cfg_path = configs_dir / f"{label}_compensation.yaml"
        write_yaml(comp_cfg_path, comp_bd_cfg)
        comp_step = SbatchStep(
            job_name=f"morris_comp_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=comp_cfg["resources"]["time_hours"], cpus=comp_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("compensation", comp_cfg_path)],
        )
        scripts.append((f"03_compensation_{gene}.sh", comp_step.render()))

        trans_bd_cfg = render_gene_cfg(gene, label, {
            "sum_factor": sum_factor_block,
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
            job_name=f"morris_trans_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["resources"]["time_hours"], cpus=trans_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("trans", trans_cfg_path)],
        )
        scripts.append((f"04_trans_{gene}.sh", trans_step.render()))

        perm_bd_cfg = render_gene_cfg(gene, label, {
            "sum_factor": sum_factor_block,
            "permutation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "covariates": cis_cfg["adjust_ntc_sum_factor"]["covariates"],
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
            },
        })
        perm_cfg_path = configs_dir / f"{label}_permutation.yaml"
        write_yaml(perm_cfg_path, perm_bd_cfg)
        n_perm = trans_cfg["permutation"]["n_reps"]
        perm_step = SbatchArray(
            job_name=f"morris_perm_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["permutation"]["resources"]["time_hours"],
            cpus=trans_cfg["permutation"]["resources"]["cores"],
            max_index=n_perm - 1, max_concurrent=min(n_perm, 50),
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("permutation_null", perm_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
        )
        scripts.append((f"05_permutation_{gene}.sh", perm_step.render()))

        sim_bd_cfg = render_gene_cfg(gene, label, {
            "sum_factor": sum_factor_block,
            "simulation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "load_trans": {"enabled": True},
                "sum_factor_col": "sum_factor_refit",
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
            },
        })
        sim_cfg_path = configs_dir / f"{label}_simulation.yaml"
        write_yaml(sim_cfg_path, sim_bd_cfg)
        n_sim = trans_cfg["simulation"]["n_reps"]
        sim_step = SbatchArray(
            job_name=f"morris_sim_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=trans_cfg["simulation"]["resources"]["time_hours"],
            cpus=trans_cfg["simulation"]["resources"]["cores"],
            max_index=n_sim - 1, max_concurrent=min(n_sim, 50),
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("recapitulation_sim", sim_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
        )
        scripts.append((f"06_recapitulation_{gene}.sh", sim_step.render()))

    # ---------------------------------------------------------------- #
    # 3. cis-ONLY sweep over genes_all.csv (excluding primary_genes)     #
    # ---------------------------------------------------------------- #
    genes_csv_path = THIS_DIR / cis_sweep_cfg["genes_csv"]
    genes_col = cis_sweep_cfg.get("genes_csv_column", "gene")
    sweep_genes = pd.read_csv(genes_csv_path)[genes_col].tolist()
    sweep_genes = [g for g in sweep_genes if g not in set(cfg["primary_genes"])]

    sweep_configs_list = configs_dir / "cis_sweep_configs.txt"
    tasklist_path = configs_dir / "cis_sweep_tasklist.txt"
    config_paths, task_commands = [], []
    for gene in sweep_genes:
        label = f"{label_prefix}_{gene}"
        sweep_bd_cfg = render_gene_cfg(gene, label, {
            "ntc_shared_dir": ntc_shared_dir,
            "cis": {
                "adjust_ntc_sum_factor": sum_factor_block["adjust_ntc_sum_factor"],
                "fit": {"sum_factor_col": "sum_factor_adj" if cis_cfg["adjust_ntc_sum_factor"]["enabled"] else "sum_factor"},
                "save": True,
            },
        })
        sweep_cfg_path = configs_dir / f"{label}_cis.yaml"
        write_yaml(sweep_cfg_path, sweep_bd_cfg)
        config_paths.append(str(sweep_cfg_path))
        task_commands.append(bd_cmd("cis_high_moi_shared_ntc", sweep_cfg_path))

    sweep_configs_list.write_text("\n".join(config_paths) + "\n")
    tasklist_path.write_text("\n".join(task_commands) + "\n")

    if cis_sweep_cfg["use_gpu_node_queue"]:
        gnq = cis_sweep_cfg["gpu_node_queue"]
        n_nodes = gnq["n_nodes"]
        chunks = [task_commands[i::n_nodes] for i in range(n_nodes)]
        for i, chunk in enumerate(chunks):
            chunk_path = configs_dir / f"cis_sweep_tasklist_node{i}.txt"
            chunk_path.write_text("\n".join(chunk) + "\n")
            gpu_step = SbatchGpuNodeQueue(
                job_name=f"morris_cis_sweep_node{i}", account=account, log_dir=str(logs_dir),
                time_hours=gnq["time_hours"], tasklist_path=str(chunk_path),
                concurrency=gnq["concurrency_per_node"],
                gpu_partition=cluster["gpu_partition"],
                node_queue_script=f"{repo_dir}/publication_runs/common/slurm/run_node_queue.sh",
                repo_dir=repo_dir, gpu_sbatch_lines=["#SBATCH --gpus=1", "#SBATCH -N 1"],
            )
            scripts.append((f"07_cis_sweep_node{i}.sh", gpu_step.render()))
    else:
        # Single CPU array, one task per gene, reading its config path by
        # line number from cis_sweep_configs.txt -- see README.md's
        # MaxSubmitJobs note (this is ONE array submission regardless of
        # sweep_genes count).
        array_commands = [
            f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{sweep_configs_list}")',
            bd_cmd("cis_high_moi_shared_ntc", "$CONFIG"),
        ]
        sweep_step = SbatchArray(
            job_name="morris_cis_sweep", account=account, log_dir=str(logs_dir),
            time_hours=cis_sweep_cfg["resources"]["time_hours"], cpus=cis_sweep_cfg["resources"]["cores"],
            max_index=len(sweep_genes) - 1, max_concurrent=cis_sweep_cfg["array_max_concurrent"],
            partition=partition_cpu, repo_dir=repo_dir, commands=array_commands,
        )
        scripts.append(("07_cis_sweep.sh", sweep_step.render()))

    for filename, text in scripts:
        (outdir / filename).write_text(text)
        os.chmod(outdir / filename, 0o755)

    # ---------------------------------------------------------------- #
    # submit_all.sh                                                      #
    # ---------------------------------------------------------------- #
    submit_lines = [
        "#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "",
        'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)',
        'echo "ntc_shared: $NTC_JOB"', "",
    ]
    for gene in cfg["primary_genes"]:
        submit_lines += [
            f'CIS_{gene}=$(sbatch --parsable --dependency=afterok:$NTC_JOB 02_cis_{gene}.sh)',
            f'echo "cis_{gene}: $CIS_{gene}"',
            f'COMP_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 03_compensation_{gene}.sh)',
            f'TRANS_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 04_trans_{gene}.sh)',
            f'sbatch --dependency=afterok:$TRANS_{gene} 05_permutation_{gene}.sh',
            f'sbatch --dependency=afterok:$TRANS_{gene} 06_recapitulation_{gene}.sh',
            "",
        ]
    if cis_sweep_cfg["use_gpu_node_queue"]:
        n_nodes = cis_sweep_cfg["gpu_node_queue"]["n_nodes"]
        for i in range(n_nodes):
            submit_lines.append(f'sbatch --dependency=afterok:$NTC_JOB 07_cis_sweep_node{i}.sh')
    else:
        submit_lines.append('sbatch --dependency=afterok:$NTC_JOB 07_cis_sweep.sh')

    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")
    print(f"[generate_slurm] cis-only sweep: {len(sweep_genes)} genes "
          f"({'GPU node queue' if cis_sweep_cfg['use_gpu_node_queue'] else 'CPU array'}).")


if __name__ == "__main__":
    main()
