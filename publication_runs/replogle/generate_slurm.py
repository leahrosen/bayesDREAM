"""
Generate SLURM scripts + per-gene/per-cis-variant bayesdream-CLI configs for
the Replogle dataset: one shared fit_ntc (all NTC cells), then per cis gene
(GFI1B/MYB/NFE2/TET2/IKZF1/HHEX/RUNX1) x 2 NTC-subsetting data variants x 3
fit_cis configurations -> matching fit_trans. See
publication_runs/replogle/STRATEGY.md for the full design writeup and
publication_runs/README.md for the general conventions this mirrors
(closest to domingo/generate_slurm.py's shared-ntc + per-gene
subset_per_gene.py + deferred-add_cis_gene()-for-cis /
eager-cis_gene-at-construction-for-trans pattern -- see that module's own
docstring for why those two constructions are equivalent given the SAME
precomputed subset).

Deliberately does NOT include compensation/permutation/recapitulation/extra
modalities -- none of those were part of what was asked for (STRATEGY.md's
scope is preprocess -> ntc_shared -> fit_cis x3 -> fit_trans x3) or shown in
the reference notebooks (tmp/06_.../tmp/10_...). Add them later the same way
Domingo/Morris did, if wanted.

Two NTC-subsetting data variants per gene, both built by ONE
common/subset_per_gene.py call each (with/without the new batch_match_ntc
flag -- see that script's module docstring and STRATEGY.md §2/§6):
    bm/          batch-matched NTC (only NTC cells from a batch where this
                 gene was actually targeted) -- Replogle's own "current"
                 convention, ported from load_gene_model_inputs() in
                 tmp/10_bayesDREAM_fit_trans_MYB.ipynb.
    all/         every NTC cell, no batch restriction (add_cis_gene()'s own
                 default -- no extra flag needed for this one).

Three fit_cis variants per gene (config.yaml's `cis_variants:`), each
reusing ONE of the two data variants above:
    bm_indmu     data=bm,  independent_mu_sigma=True
    bm_noindmu   data=bm,  independent_mu_sigma=False
    all_indmu    data=all, independent_mu_sigma=True
Each gets its own fit_trans, reading that SAME variant's own cis fit (same
label => load_cis_fit()'s default directory lines up) and the SAME data
variant's `full/` subset.

Usage
-----
    python generate_slurm.py [--config config.yaml] [--outdir slurm]

Writes:
    <outdir>/configs/*.yaml                     one bayesdream-CLI config per gene/variant/stage
    <outdir>/logs/                               sbatch log directory
    <outdir>/01_ntc_shared.sh
    <outdir>/01b_subset_<gene>_<data_variant>.sh
    <outdir>/02_cis_<gene>_<cis_variant>.sh
    <outdir>/03_trans_<gene>_<cis_variant>.sh
    <outdir>/submit_all.sh                       dependency-chained submission,
                                                  writes submitted_jobs.tsv for
                                                  common/slurm/list_job_status.py
"""

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR_LOCAL = THIS_DIR.parents[1]  # .../bayesDREAM_forClaude, for local imports only
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common"))
sys.path.insert(0, str(REPO_DIR_LOCAL / "publication_runs" / "common" / "slurm"))

from config_utils import load_yaml, write_yaml, render_bayesdream_config, resolve_paths  # noqa: E402
from git_provenance import create_stable_snapshot_tag  # noqa: E402
from sbatch_blocks import SbatchStep  # noqa: E402

TIME_HOURS_DEFAULT = 24.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(THIS_DIR / "config.yaml"))
    parser.add_argument("--outdir", default=str(THIS_DIR / "slurm"))
    parser.add_argument("--no-tag", action="store_true", help="Skip creating a git snapshot tag.")
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

    paths = resolve_paths(cfg["paths"])
    meta_path = paths["meta"]
    counts_path = paths["counts"]
    gene_meta_path = paths["gene_meta"]
    meta_ntc_path = paths["meta_ntc"]
    counts_ntc_path = paths["counts_ntc"]
    output_dir = paths["output_dir"]
    repo_dir = paths["repo_dir"]
    python_env = paths["python_env"]
    python_env_gpu = paths["python_env_gpu"]

    cluster = cfg["cluster"]
    account = cluster["account"]
    partition_cpu = cluster.get("partition_cpu", "shared")
    partition_gpu = cluster.get("partition_gpu", "gpu")
    gpu_single_sbatch_lines = cluster.get("gpu_single_sbatch_lines", ["#SBATCH --gpus=1"])

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    gene_ids = cfg["cis_gene_ids"]
    ntc_shared_cfg = cfg["ntc_shared"]
    sf_cfg = cfg["sum_factor"]
    subset_cfg = cfg["subset"]
    bm_selected_ntc_guides = cfg["bm_selected_ntc_guides"]
    cis_variants = cfg["cis_variants"]
    cis_cfg = cfg["cis"]
    trans_cfg = cfg["trans"]

    # feature_meta_read_csv_kwargs: {} -- preprocess.py writes gene_meta.csv
    # with to_csv(index=False) (a real gene_id column, no leading unnamed
    # index column). Without this, config_utils's default (index_col=0)
    # would silently consume gene_id AS the index instead of leaving it as
    # a column -- see domingo/generate_slurm.py's identical comment for the
    # real bug this caused there (permute_x_true's IndexError).
    base_cfg = {
        "_dataset_dir": str(THIS_DIR),
        # feature_meta is REQUIRED here, not optional -- Replogle's counts
        # are a sparse .npz with no row labels, and (with model.feature_name_col
        # now set below) resolve_feature_ids' step 1 needs feature_meta to
        # read that column from at all. Without this, write_subset_step()'s
        # own model construction (the only stage that doesn't override
        # data.feature_meta itself via subset_data_block()) would raise
        # "feature_name_col='gene_id' was given but feature_meta was not
        # provided" the first time it actually ran -- never caught by the
        # dry-run config-generation tests already done for this pipeline,
        # which don't construct a real model against real (sparse) data.
        "data": {"meta": meta_path, "counts": counts_path, "meta_read_csv_kwargs": {},
                  "feature_meta": gene_meta_path, "feature_meta_read_csv_kwargs": {}},
        "model": {
            "modality_name": model_defaults["modality_name"],
            "guide_covariates": model_defaults["guide_covariates"],
            "guide_covariates_ntc": model_defaults["guide_covariates_ntc"],
            # Pins primary/'cis' gene identity to gene_id (Ensembl) -- see
            # config.yaml's feature_name_col comment. Propagates to every
            # rendered stage (ntc_shared/subset/cis/trans) via this shared
            # base_cfg, so all of them agree on the same convention.
            "feature_name_col": model_defaults["feature_name_col"],
            "output_dir": output_dir,
            # No explicit 'device' override (unlike Domingo, which forces
            # 'cpu' everywhere) -- ntc_shared and fit_trans's "all" (full-NTC)
            # variant run on GPU (python_env_gpu + partition_gpu below),
            # everything else on CPU (python_env + partition_cpu); bayesDREAM
            # auto-detects cuda vs cpu per bayesDREAM.__init__'s own default,
            # same as morris/generate_slurm.py's base_cfg (also no 'device'
            # key), rather than needing a per-stage override here.
        },
        "ntc": {"set_technical_groups": ntc_shared_cfg["set_technical_groups"]},
    }

    # NOTE: no refit_sumfactor -- Replogle's own reference notebook
    # (tmp/10_bayesDREAM_fit_trans_MYB.ipynb) calls fit_trans() with
    # sum_factor_col="sum_factor_adj" directly, never a refit column. See
    # config.yaml's sum_factor: block comment / STRATEGY.md §7 decision #5.
    sum_factor_block = {
        "adjust_ntc_sum_factor": {"enabled": True, "args": {"covariates": sf_cfg["covariates"]}},
    }

    def bd_cmd(kind: str, config_path: Path, env: str, extra_args: str = "") -> str:
        """kind: 'ntc'/'cis_deferred'/'trans' all go through common/run_<kind>.py
        scripts (same reasoning as domingo/morris: run_trans.py adds
        adjust_ntc_sum_factor(), which the plain CLI's fit-trans never calls).
        `env` is explicit (not a closure over one fixed python_env) since
        ntc_shared and fit_trans's "all" variant run under python_env_gpu,
        everything else under python_env -- mirrors morris/generate_slurm.py's
        bd_cmd signature."""
        script = f"{repo_dir}/publication_runs/common/run_{kind}.py"
        return f'"{env}" "{script}" --config "{config_path}"{extra_args}'

    scripts = []
    submitted_rows = []

    # ---------------------------------------------------------------- #
    # 1. Shared fit_ntc (deferred cis_gene, ALL NTC cells)                #
    # ---------------------------------------------------------------- #
    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    ntc_bd_cfg = render_bayesdream_config(base_cfg, {
        "model": {"label": label_ntc},
        "data": {"meta": meta_ntc_path, "counts": counts_ntc_path,
                  "feature_meta": gene_meta_path, "feature_meta_read_csv_kwargs": {}},
        "ntc": {"fit": ntc_shared_cfg.get("fit", {}), "save": True},
    })
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_bd_cfg)

    ntc_step = SbatchStep(
        job_name="replogle_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=ntc_shared_cfg["resources"].get("time_hours", TIME_HOURS_DEFAULT),
        cpus=ntc_shared_cfg["resources"]["cores"],
        # GPU node (all 85711 NTC cells x 8202 features x 315 technical
        # groups -- the reference notebook itself ran this on cuda:0).
        partition=partition_gpu, extra_sbatch_lines=gpu_single_sbatch_lines,
        repo_dir=repo_dir, commands=[bd_cmd("ntc", ntc_cfg_path, python_env_gpu)],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))
    submitted_rows.append(("ntc_shared", label_ntc, "01_ntc_shared.sh"))

    def subset_dir_for(gene: str, data_variant: str) -> str:
        return f"{output_dir}/{label_prefix}_{gene}_{data_variant}_subset"

    def subset_data_block(gene: str, data_variant: str, mode: str) -> dict:
        d = f"{subset_dir_for(gene, data_variant)}/{mode}"
        return {
            "meta": f"{d}/meta.csv", "counts": f"{d}/gene_counts.npz",
            "feature_meta": f"{d}/gene_meta.csv", "feature_meta_read_csv_kwargs": {},
        }

    def write_subset_step(gene: str, gene_id: str, data_variant: str) -> str:
        subset_label = f"{label_prefix}_{gene}_{data_variant}_subset_input"
        overrides = {"model": {"label": subset_label}, "cis_gene": gene_id}
        if data_variant == "bm":
            # "bm" replicates the ORIGINAL ad hoc pipeline's actual NTC
            # subsetting -- a curated guide list (what NTC_subset/ silently
            # encoded) AND batch-matched (load_gene_model_inputs()'s own
            # logic) -- both compose via AND for the NTC portion. See
            # config.yaml's bm_selected_ntc_guides comment / STRATEGY.md §10.
            overrides["batch_match_ntc"] = {"enabled": True, "batch_col": "batch"}
            overrides["select_ntc_guides"] = {"enabled": True, "guides": bm_selected_ntc_guides}
        subset_input_cfg = render_bayesdream_config(base_cfg, overrides)
        subset_input_cfg_path = configs_dir / f"{subset_label}.yaml"
        write_yaml(subset_input_cfg_path, subset_input_cfg)

        cmd = (
            f'"{python_env}" "{repo_dir}/publication_runs/common/subset_per_gene.py" '
            f'--config "{subset_input_cfg_path}" --outdir "{subset_dir_for(gene, data_variant)}" '
            f'--modes full,cis_only'
        )
        step = SbatchStep(
            job_name=f"replogle_subset_{gene}_{data_variant}", account=account, log_dir=str(logs_dir),
            time_hours=subset_cfg["resources"].get("time_hours", TIME_HOURS_DEFAULT),
            cpus=subset_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[cmd],
        )
        filename = f"01b_subset_{gene}_{data_variant}.sh"
        scripts.append((filename, step.render()))
        return filename

    # ---------------------------------------------------------------- #
    # Per-gene stages                                                    #
    # ---------------------------------------------------------------- #
    submit_lines = ["#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "",
                     'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)', 'echo "ntc_shared: $NTC_JOB"', ""]

    for gene in cfg["cis_genes"]:
        gene_id = gene_ids[gene]

        data_variants_needed = sorted({v["data_variant"] for v in cis_variants.values()})
        subset_scripts = {}
        for data_variant in data_variants_needed:
            script_name = write_subset_step(gene, gene_id, data_variant)
            submitted_rows.append((f"subset_{data_variant}", f"{label_prefix}_{gene}", script_name))
            subset_scripts[data_variant] = script_name
            submit_lines.append(
                f'SUBSET_{gene}_{data_variant}=$(sbatch --parsable --dependency=afterok:$NTC_JOB {script_name})'
            )
        submit_lines.append("")

        for variant_name, variant_spec in cis_variants.items():
            data_variant = variant_spec["data_variant"]
            label = f"{label_prefix}_{gene}_{variant_name}"

            # -- cis (deferred add_cis_gene, reusing shared ntc) --
            cis_bd_cfg = render_bayesdream_config(base_cfg, {
                "model": {"label": label},
                "data": subset_data_block(gene, data_variant, "cis_only"),
                "cis_gene": gene_id,
                "ntc_shared_dir": ntc_shared_dir,
                "sum_factor": {"adjust_ntc_sum_factor": sum_factor_block["adjust_ntc_sum_factor"]},
                "cis": {
                    "fit": {**cis_cfg.get("fit", {}),
                            "sum_factor_col": "sum_factor_adj",
                            "independent_mu_sigma": variant_spec["independent_mu_sigma"]},
                    "save": True,
                },
            })
            cis_cfg_path = configs_dir / f"{label}_cis.yaml"
            write_yaml(cis_cfg_path, cis_bd_cfg)

            cis_step = SbatchStep(
                job_name=f"replogle_cis_{gene}_{variant_name}", account=account, log_dir=str(logs_dir),
                time_hours=cis_cfg["resources"].get("time_hours", TIME_HOURS_DEFAULT),
                cpus=cis_cfg["resources"]["cores"],
                partition=partition_cpu, repo_dir=repo_dir,
                commands=[bd_cmd("cis_deferred", cis_cfg_path, python_env)],
            )
            cis_filename = f"02_cis_{gene}_{variant_name}.sh"
            scripts.append((cis_filename, cis_step.render()))
            submitted_rows.append(("cis", label, cis_filename))

            # -- trans (eager cis_gene at construction, 'full' subset already
            # has the cis gene's row put back in -- equivalent to the
            # reference notebook's deferred add_cis_gene() on the same data,
            # see module docstring) --
            trans_bd_cfg = render_bayesdream_config(base_cfg, {
                "model": {"label": label, "cis_gene": gene_id},
                "data": subset_data_block(gene, data_variant, "full"),
                "sum_factor": sum_factor_block,
                "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
                "trans": {
                    "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                    "load_cis": {"enabled": True},
                    "fit": {**trans_cfg.get("fit", {}),
                            "sum_factor_col": "sum_factor_adj",
                            "function_type": trans_cfg["function_type"]},
                    "save": True,
                },
            })
            trans_cfg_path = configs_dir / f"{label}_trans.yaml"
            write_yaml(trans_cfg_path, trans_bd_cfg)

            # "all" (full-NTC) variant -> one GPU node per gene, fixed 24h,
            # no profiling (see config.yaml's trans.gpu_resources comment).
            # "bm" (batch-matched/subsetted-NTC) variant -> CPU, cores
            # pending real profiling (config.yaml's trans.resources comment).
            if data_variant == "all":
                trans_resources = trans_cfg["gpu_resources"]
                trans_partition = partition_gpu
                trans_extra_sbatch_lines = gpu_single_sbatch_lines
                trans_env = python_env_gpu
            else:
                trans_resources = trans_cfg["resources"]
                trans_partition = partition_cpu
                trans_extra_sbatch_lines = []
                trans_env = python_env

            trans_step = SbatchStep(
                job_name=f"replogle_trans_{gene}_{variant_name}", account=account, log_dir=str(logs_dir),
                time_hours=trans_resources.get("time_hours", TIME_HOURS_DEFAULT),
                cpus=trans_resources["cores"],
                partition=trans_partition, extra_sbatch_lines=trans_extra_sbatch_lines,
                repo_dir=repo_dir, commands=[bd_cmd("trans", trans_cfg_path, trans_env)],
                # fit_trans() has its own internal checkpoint/resume, unlike
                # ntc/cis/subset -- see sbatch_blocks.py's module docstring
                # and publication_runs/README.md's "Restart policy".
                auto_requeue_on_timeout=True,
            )
            trans_filename = f"03_trans_{gene}_{variant_name}.sh"
            scripts.append((trans_filename, trans_step.render()))
            submitted_rows.append(("trans", label, trans_filename))

            submit_lines += [
                f'CIS_{gene}_{variant_name}=$(sbatch --parsable '
                f'--dependency=afterok:$SUBSET_{gene}_{data_variant} {cis_filename})',
                f'sbatch --dependency=afterok:$CIS_{gene}_{variant_name} {trans_filename}',
                "",
            ]

    for filename, text in scripts:
        (outdir / filename).write_text(text)
        os.chmod(outdir / filename, 0o755)

    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    tsv_lines = ["stage\tlabel\tscript"] + [f"{s}\t{l}\t{f}" for s, l, f in submitted_rows]
    (outdir / "submitted_jobs.tsv.template").write_text("\n".join(tsv_lines) + "\n")

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")


if __name__ == "__main__":
    main()
