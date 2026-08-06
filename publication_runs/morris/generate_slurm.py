"""
Generate SLURM scripts + configs for the Morris dataset: high-MOI, one
shared fit_ntc (full panel, cis_gene deferred), the full pipeline (cis ->
compensation -> trans -> permutation -> recapitulation) for config.yaml's
`primary_genes`, and a fit_cis-ONLY sweep over every OTHER gene with
padj<0.05 in Morris_gRNA2target_stats.csv, all reusing the one shared
fit_ntc via the SAME deferred-cis_gene + add_cis_gene() mechanism Domingo
uses (bayesDREAM's high-MOI mode now supports this natively -- no more
placeholder-gene/manual-alpha-extraction workaround, see morris/README.md).

Assumes morris/preprocess.py has already been run once (writes the aligned
meta.csv/gene_counts.npz/gene_meta.csv/guide_assignment.npy/guide_meta.csv/
guide_target.csv this script's paths.data_dir points at).

Placement: cis and compensation always run on CPU (bayesdream_cpu env,
`-p shared`); ntc_shared/trans/permutation/recapitulation run on GPU
(bayesdream_rocm env, `-p gpu`) -- per project convention (Domingo is
CPU-only everywhere; Morris's fit_cis specifically is CPU-only regardless of
dataset; everything else in Morris follows its own reference scripts, which
required CUDA). Only trans/permutation/recapitulation auto-resubmit on
timeout (fit_trans has its own checkpoint/resume) -- ntc_shared/cis/
compensation/cis_sweep failures are left for manual review via
common/slurm/list_job_status.py.

Correctness note: `exclude_guides` is a bayesDREAM CONSTRUCTOR-time filter,
so every stage that constructs its OWN fresh model for a given gene (cis via
deferred+add_cis_gene, compensation/trans/permutation/recapitulation via
cis_gene-at-init) must use the IDENTICAL per-gene exclude_guides list, or
their cell/guide composition diverges and load_cis_fit()/load_trans_fit()
would be aligning against a model whose cells don't match what was actually
saved. `_per_gene_exclude_guides()` below is computed once per gene and
threaded through every stage's rendered config for consistency.

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
sys.path.insert(0, str(THIS_DIR))

from config_utils import load_yaml, write_yaml, render_bayesdream_config  # noqa: E402
from git_provenance import create_stable_snapshot_tag  # noqa: E402
from sbatch_blocks import SbatchStep, SbatchArray  # noqa: E402
from snp_exclusion import exclude_guides_for_cis_gene  # noqa: E402

TIME_HOURS = 24.0  # fixed everywhere, per project convention


def _resolve_paths(paths: dict) -> dict:
    """Resolve {placeholder} cross-references, allowing chained references
    (e.g. data_dir: "{raw_data_dir}/preprocessed") by repeatedly formatting
    until nothing changes."""
    resolved = dict(paths)
    for _ in range(len(paths) + 1):
        changed = False
        for k, v in resolved.items():
            if isinstance(v, str) and "{" in v:
                new_v = v.format(**resolved)
                if new_v != v:
                    resolved[k] = new_v
                    changed = True
        if not changed:
            break
    return resolved


def _select_cis_genes(stats_csv: str, gene_use_col: str, padj_col: str, padj_threshold: float,
                       feature_meta_path: str, primary_genes: list):
    stats = pd.read_csv(stats_csv)
    sig_ids = stats.loc[stats[padj_col] < padj_threshold, gene_use_col].unique().tolist()

    feature_meta = pd.read_csv(feature_meta_path)
    id_to_name = dict(zip(feature_meta["gene_id"], feature_meta["gene_name"]))
    name_to_id = {v: k for k, v in id_to_name.items()}

    mapped = []
    unmapped = []
    for gid in sig_ids:
        name = id_to_name.get(gid)
        if name is None:
            unmapped.append(gid)
        else:
            mapped.append((gid, name))
    if unmapped:
        print(f"[generate_slurm] WARNING: {len(unmapped)} padj<{padj_threshold} gene_use id(s) "
              f"not found in feature_meta, skipped: {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")

    sweep_genes = [name for _gid, name in mapped if name not in set(primary_genes)]
    return sweep_genes, name_to_id


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

    paths = _resolve_paths(cfg["paths"])
    output_dir = paths["output_dir"]
    repo_dir = paths["repo_dir"]
    python_env_cpu = paths["python_env_cpu"]
    python_env_gpu = paths["python_env_gpu"]

    cluster = cfg["cluster"]
    account = cluster["account"]
    partition_cpu = cluster.get("partition_cpu", "shared")
    partition_gpu = cluster.get("partition_gpu", "gpu")

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    global_exclude_guides = cfg.get("global_exclude_guides") or []
    sf_cfg = cfg["sum_factor"]
    gene_sel_cfg = cfg["gene_selection"]
    ntc_shared_cfg = cfg["ntc_shared"]
    cis_cfg = cfg["cis"]
    comp_cfg = cfg["compensation"]
    trans_cfg = cfg["trans"]
    cis_sweep_cfg = cfg["cis_sweep"]

    primary_genes = gene_sel_cfg["primary_genes"]
    sweep_genes, name_to_id = _select_cis_genes(
        stats_csv=paths["stats_csv"], gene_use_col=gene_sel_cfg["gene_use_col"],
        padj_col=gene_sel_cfg["padj_col"], padj_threshold=gene_sel_cfg["padj_threshold"],
        feature_meta_path=paths["feature_meta"], primary_genes=primary_genes,
    )
    all_guide_names = pd.read_csv(paths["guide_meta"])["guide"].tolist()

    def per_gene_exclude_guides(gene: str) -> list:
        return sorted(set(global_exclude_guides) | set(exclude_guides_for_cis_gene(all_guide_names, gene)))

    data_block = {
        "meta": paths["meta"], "counts": paths["counts"],
        "feature_meta": paths["feature_meta"], "feature_meta_read_csv_kwargs": {},
        "guide_assignment": paths["guide_assignment"],
        "guide_meta": paths["guide_meta"], "guide_target": paths["guide_target"],
    }
    base_cfg = {
        "data": data_block,
        "model": {
            "modality_name": model_defaults["modality_name"],
            "guide_covariates": model_defaults["guide_covariates"],
            "guide_covariates_ntc": model_defaults["guide_covariates_ntc"],
            "min_count": model_defaults["min_count"],
            "output_dir": output_dir,
        },
        "ntc": {"set_technical_groups": ntc_shared_cfg["set_technical_groups"]},
    }
    sum_factor_cis_block = {
        "compute_scran": {"enabled": True, "args": {"batch_col": sf_cfg["batch_col"]}},
        "adjust_ntc_sum_factor": {
            "enabled": True,
            "args": {"sum_factor_col_old": "sum_factor_new", "covariates": sf_cfg["covariates"]},
        },
    }

    def bd_cmd(kind: str, config_path, python_env: str, extra_args: str = "") -> str:
        if kind in {"fit-ntc", "fit-cis", "fit-trans", "report"}:
            return f'cd "{repo_dir}" && "{python_env}" -m bayesDREAM {kind} --config "{config_path}"{extra_args}'
        script = f"{repo_dir}/publication_runs/common/run_{kind}.py"
        return f'"{python_env}" "{script}" --config "{config_path}"{extra_args}'

    scripts = []
    submitted_rows = []  # (stage, label, script_placeholder) -- jobids filled in by submit_all.sh

    # ---------------------------------------------------------------- #
    # 1. Shared fit_ntc: cis_gene deferred, full gene panel, GPU.        #
    #    Only global_exclude_guides applied here -- per-gene SNP         #
    #    exclusion happens later, per gene (see module docstring).       #
    # ---------------------------------------------------------------- #
    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    ntc_bd_cfg = render_bayesdream_config(base_cfg, {
        "model": {"label": label_ntc, "exclude_guides": global_exclude_guides},
        "ntc": {"fit": {}, "save": True},
    })
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_bd_cfg)

    ntc_step = SbatchStep(
        job_name="morris_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=TIME_HOURS, cpus=ntc_shared_cfg["resources"]["cores"],
        partition=partition_gpu, repo_dir=repo_dir,
        commands=[bd_cmd("ntc", ntc_cfg_path, python_env_gpu)],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))
    submitted_rows.append(("ntc_shared", label_ntc, "01_ntc_shared.sh"))

    def render_gene_cfg(gene: str, label: str, device: str, exclude_guides: list, extra: dict) -> dict:
        overrides = {
            "model": {"label": label, "cis_gene": gene, "device": device, "exclude_guides": exclude_guides},
            **extra,
        }
        return render_bayesdream_config(base_cfg, overrides)

    def render_cis_config(gene: str, label: str, exclude_guides: list) -> Path:
        # Deferred (cis_gene NOT in model:) -- run_cis_deferred.py commits via
        # add_cis_gene(), same mechanism as Domingo, now that bayesDREAM's
        # high-MOI mode supports it directly.
        cis_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "device": "cpu", "exclude_guides": exclude_guides},
            "cis_gene": gene,
            "ntc_shared_dir": ntc_shared_dir,
            "sum_factor": sum_factor_cis_block,
            "cis": {"fit": {"sum_factor_col": "sum_factor_adj", "independent_mu_sigma": True}, "save": True},
        })
        cis_cfg_path = configs_dir / f"{label}_cis.yaml"
        write_yaml(cis_cfg_path, cis_bd_cfg)
        return cis_cfg_path

    def cis_sbatch_step(gene: str, cis_cfg_path: Path) -> SbatchStep:
        return SbatchStep(
            job_name=f"morris_cis_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("cis_deferred", cis_cfg_path, python_env_cpu)],
        )

    # ---------------------------------------------------------------- #
    # 2. Primary genes: cis(CPU) -> compensation(CPU) -> trans(GPU) ->   #
    #    permutation(GPU array) -> recapitulation(GPU array)             #
    # ---------------------------------------------------------------- #
    for gene in primary_genes:
        label = f"{label_prefix}_{gene}"
        exclude_guides = per_gene_exclude_guides(gene)

        cis_cfg_path = render_cis_config(gene, label, exclude_guides)
        cis_step = cis_sbatch_step(gene, cis_cfg_path)
        scripts.append((f"02_cis_{gene}.sh", cis_step.render()))
        submitted_rows.append(("cis", label, f"02_cis_{gene}.sh"))

        cis_gene_ensembl_id = name_to_id.get(gene)
        comp_bd_cfg = render_gene_cfg(gene, label, "cpu", exclude_guides, {
            "compensation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "args": {
                    "exclude_cells": {
                        "module": "compensation_exclude_cells",
                        "function": "compute_padj_exclude_cells",
                        "kwargs": {
                            "stats_csv": paths["stats_csv"],
                            "cis_gene_ensembl_id": cis_gene_ensembl_id,
                            "padj_threshold": comp_cfg["padj_threshold"],
                        },
                    },
                },
            },
        })
        comp_cfg_path = configs_dir / f"{label}_compensation.yaml"
        write_yaml(comp_cfg_path, comp_bd_cfg)
        comp_step = SbatchStep(
            job_name=f"morris_comp_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=comp_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("compensation", comp_cfg_path, python_env_cpu)],
        )
        scripts.append((f"03_compensation_{gene}.sh", comp_step.render()))
        submitted_rows.append(("compensation", label, f"03_compensation_{gene}.sh"))

        sum_factor_trans_block = {
            "compute_scran": sum_factor_cis_block["compute_scran"],
            "adjust_ntc_sum_factor": sum_factor_cis_block["adjust_ntc_sum_factor"],
            "refit_sumfactor": {"enabled": False},  # vestigial for Morris -- fit_trans uses sum_factor_adj
        }
        trans_bd_cfg = render_gene_cfg(gene, label, None, exclude_guides, {
            "sum_factor": sum_factor_trans_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "trans": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
                "save": True,
            },
        })
        trans_cfg_path = configs_dir / f"{label}_trans.yaml"
        write_yaml(trans_cfg_path, trans_bd_cfg)
        trans_step = SbatchStep(
            job_name=f"morris_trans_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=trans_cfg["resources"]["cores"],
            partition=partition_gpu, repo_dir=repo_dir,
            commands=[bd_cmd("trans", trans_cfg_path, python_env_gpu)],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"04_trans_{gene}.sh", trans_step.render()))
        submitted_rows.append(("trans", label, f"04_trans_{gene}.sh"))

        perm_bd_cfg = render_gene_cfg(gene, label, None, exclude_guides, {
            "sum_factor": sum_factor_trans_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "permutation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "covariates": sf_cfg["covariates"],
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
            },
        })
        perm_cfg_path = configs_dir / f"{label}_permutation.yaml"
        write_yaml(perm_cfg_path, perm_bd_cfg)
        n_perm = trans_cfg["permutation"]["n_reps"]
        perm_step = SbatchArray(
            job_name=f"morris_perm_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=trans_cfg["permutation"]["resources"]["cores"],
            max_index=n_perm - 1, max_concurrent=min(n_perm, 50),
            partition=partition_gpu, repo_dir=repo_dir,
            commands=[bd_cmd("permutation_null", perm_cfg_path, python_env_gpu, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"05_permutation_{gene}.sh", perm_step.render()))
        submitted_rows.append(("permutation", label, f"05_permutation_{gene}.sh"))

        sim_bd_cfg = render_gene_cfg(gene, label, None, exclude_guides, {
            "sum_factor": sum_factor_trans_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "simulation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True}, "load_trans": {"enabled": True},
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
            },
        })
        sim_cfg_path = configs_dir / f"{label}_simulation.yaml"
        write_yaml(sim_cfg_path, sim_bd_cfg)
        n_sim = trans_cfg["simulation"]["n_reps"]
        sim_step = SbatchArray(
            job_name=f"morris_sim_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=trans_cfg["simulation"]["resources"]["cores"],
            max_index=n_sim - 1, max_concurrent=min(n_sim, 50),
            partition=partition_gpu, repo_dir=repo_dir,
            commands=[bd_cmd("recapitulation_sim", sim_cfg_path, python_env_gpu, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"06_recapitulation_{gene}.sh", sim_step.render()))
        submitted_rows.append(("recapitulation", label, f"06_recapitulation_{gene}.sh"))

    # ---------------------------------------------------------------- #
    # 3. cis-ONLY sweep (CPU array, one array submission for all genes)  #
    # ---------------------------------------------------------------- #
    sweep_configs_list = configs_dir / "cis_sweep_configs.txt"
    config_paths = []
    for gene in sweep_genes:
        label = f"{label_prefix}_{gene}"
        cis_cfg_path = render_cis_config(gene, label, per_gene_exclude_guides(gene))
        config_paths.append(str(cis_cfg_path))
    sweep_configs_list.write_text("\n".join(config_paths) + "\n")

    array_commands = [
        f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{sweep_configs_list}")',
        bd_cmd("cis_deferred", "$CONFIG", python_env_cpu),
    ]
    sweep_step = SbatchArray(
        job_name="morris_cis_sweep", account=account, log_dir=str(logs_dir),
        time_hours=TIME_HOURS, cpus=cis_sweep_cfg["resources"]["cores"],
        max_index=len(sweep_genes) - 1, max_concurrent=cis_sweep_cfg["array_max_concurrent"],
        partition=partition_cpu, repo_dir=repo_dir, commands=array_commands,
    )
    scripts.append(("07_cis_sweep.sh", sweep_step.render()))
    submitted_rows.append(("cis_sweep", f"{len(sweep_genes)} genes", "07_cis_sweep.sh"))

    for filename, text in scripts:
        (outdir / filename).write_text(text)
        os.chmod(outdir / filename, 0o755)

    # ---------------------------------------------------------------- #
    # submit_all.sh -- writes submitted_jobs.tsv for                     #
    # common/slurm/list_job_status.py (manual review of everything       #
    # except trans/permutation/recapitulation, which auto-requeue).      #
    # ---------------------------------------------------------------- #
    submit_lines = [
        "#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "",
        'TSV="submitted_jobs.tsv"',
        'echo -e "stage\\tlabel\\tjobid\\tscript" > "$TSV"',
        "",
        'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)',
        'echo -e "ntc_shared\\tntc_shared\\t$NTC_JOB\\t01_ntc_shared.sh" >> "$TSV"',
        "",
    ]
    for gene in primary_genes:
        submit_lines += [
            f'CIS_{gene}=$(sbatch --parsable --dependency=afterok:$NTC_JOB 02_cis_{gene}.sh)',
            f'echo -e "cis\\t{gene}\\t$CIS_{gene}\\t02_cis_{gene}.sh" >> "$TSV"',
            f'COMP_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 03_compensation_{gene}.sh)',
            f'echo -e "compensation\\t{gene}\\t$COMP_{gene}\\t03_compensation_{gene}.sh" >> "$TSV"',
            f'TRANS_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 04_trans_{gene}.sh)',
            f'echo -e "trans\\t{gene}\\t$TRANS_{gene}\\t04_trans_{gene}.sh" >> "$TSV"',
            f'PERM_{gene}=$(sbatch --parsable --dependency=afterok:$TRANS_{gene} 05_permutation_{gene}.sh)',
            f'echo -e "permutation\\t{gene}\\t$PERM_{gene}\\t05_permutation_{gene}.sh" >> "$TSV"',
            f'SIM_{gene}=$(sbatch --parsable --dependency=afterok:$TRANS_{gene} 06_recapitulation_{gene}.sh)',
            f'echo -e "recapitulation\\t{gene}\\t$SIM_{gene}\\t06_recapitulation_{gene}.sh" >> "$TSV"',
            "",
        ]
    submit_lines += [
        'SWEEP_JOB=$(sbatch --parsable --dependency=afterok:$NTC_JOB 07_cis_sweep.sh)',
        'echo -e "cis_sweep\\tall\\t$SWEEP_JOB\\t07_cis_sweep.sh" >> "$TSV"',
    ]
    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")
    print(f"[generate_slurm] primary_genes={primary_genes}")
    print(f"[generate_slurm] cis-only sweep: {len(sweep_genes)} genes (CPU array, single submission)")


if __name__ == "__main__":
    main()
