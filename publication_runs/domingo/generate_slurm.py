"""
Generate SLURM scripts + per-gene bayesdream-CLI configs for the Domingo
dataset: one shared fit_ntc, then per cis-gene (GFI1B/MYB/NFE2/TET2)
fit_cis -> compensation -> fit_trans -> permutation reps -> recapitulation
reps, plus the extra (splicing/velocity/efficiency) modalities per gene
after its cis fit -- and, for BINOMIAL modalities only (not multinomial),
their own permutation + recapitulation reps too. See publication_runs/README.md
for the general conventions, domingo/README.md for the reference pipeline
this mirrors, and domingo/config.yaml for all dataset-specific settings.

Domingo runs on CPU everywhere EXCEPT the two multinomial modalities
(donor_choice/acceptor_choice), which run on GPU, packed together across
ALL genes into ONE SbatchGpuNodeQueue job (len(cis_genes) x 2 = 8 tasks --
exactly one Dardel GPU node's worth of 8 GPUs at concurrency=8) -- see
"GPU-packed multinomial modalities" below. Only trans/permutation/
recapitulation (gene-level AND modality-level) auto-resubmit on timeout
(fit_trans has its own checkpoint/resume) -- ntc_shared/cis/compensation/
modality-fit failures are left for manual review via
common/slurm/list_job_status.py.

GPU-packed multinomial modalities: fit_ntc's own niters default is 2x higher
for multinomial than negbinom/binomial (see bayesDREAM/fitting/ntc.py), and
these two modalities empirically need GPU. Since there's no per-gene
permutation/recapitulation for multinomial (excluded per project
convention, same as the per-gene cis/trans path), the packed job's
tasklist is just the len(cis_genes) x 2 modality-fit calls themselves
(load_modalities.py, same script the per-gene binomial CPU jobs use) --
each individually prefixed with thread-pin + HIP_VISIBLE_DEVICES/
ROCR_VISIBLE_DEVICES env vars (round-robin over concurrency), same pattern
as morris/generate_slurm.py's packed trans/permutation/recapitulation jobs.

Per-gene data subsetting (01b_subset_<gene>.sh): mirrors morris/generate_slurm.py's
own "Per-gene data subsetting" section -- real profiling found
bayesDREAM.__init__ dominating per-gene job cost even for Domingo's much
simpler low-MOI classification (full 20001-cell load, subsetted down to
~4281 for one gene), and EVERY one of cis/compensation/trans/permutation/
recapitulation/modality-fit/modality-permutation/modality-recapitulation
independently re-paid it. Now paid ONCE per gene (common/subset_per_gene.py,
which writes BOTH modes in that one pass -- add_cis_gene() already separates
'cis' from the trans panel internally, so both pieces are in memory either
way):
- `full/`: whole trans panel, cis gene's row put back in -- used by
  compensation/trans/permutation/recapitulation/modality stages (all
  already eager cis_gene-at-construction and stay that way -- they never
  used add_cis_gene()'s alpha extraction to begin with; only fit_cis()
  reads alpha_x_prefit, and none of these call it).
- `cis_only/`: JUST the cis gene's row -- used by 02_cis_<gene>.sh, which
  keeps its deferred+add_cis_gene() mechanism unchanged (that pattern
  tolerates a 1-gene starting panel with zero code changes, confirmed by
  direct test).
ntc_shared separately reads a precomputed NTC-only file (also written by
preprocess.py) instead of the full dataset, since it never fits on non-NTC
cells anyway -- this one stays a single shared fit across all 4 genes
(unlike Morris's primary genes, Domingo has no large sweep-gene set to
justify per-gene fit_ntc's extra jobs for a comparatively small, already-
cheap NTC-only fit).

Per-(gene, modality) subsetting (07a_modality_subset_<gene>_<mod>.sh): same
motivation as 01b above, one level down -- every modality-fit/permutation/
recapitulation job independently re-read+re-aligned the FULL shared raw
splicing directory (modalities.data_dir) against its own cells. Now paid
ONCE per (gene, modality) by domingo/subset_modality_per_gene.py, which
calls the SAME load_modalities.attach_modality() the real jobs used to call
directly, from that gene's OWN precomputed `full` subset (needs only
01b_subset_<gene>.sh, NOT the cis fit -- attach_modality() only touches raw
counts, never a fitted posterior -- so this runs in parallel with
02_cis_<gene>.sh, not after it), and writes the resulting cell-aligned,
already-denominator-computed modality to disk.
07_modality_<gene>_<mod>.sh (fit) and 08/09's `attach_modality:` config
block both switched from attach_modality/base_dir to
attach_modality_precomputed/precomputed_dir -- see load_modalities.py's
module docstring.

Usage
-----
    python generate_slurm.py [--config config.yaml] [--outdir slurm]

Writes:
    <outdir>/configs/*.yaml         one bayesdream-CLI config per gene/stage
    <outdir>/logs/                  sbatch log directory
    <outdir>/01_ntc_shared.sh
    <outdir>/02_cis_<gene>.sh
    <outdir>/03_compensation_<gene>.sh
    <outdir>/04_trans_<gene>.sh
    <outdir>/05_permutation_<gene>.sh    (array over reps)
    <outdir>/06_recapitulation_<gene>.sh (array over reps)
    <outdir>/07a_modality_subset_<gene>_<modality>.sh  (all modalities, one CPU job each)
    <outdir>/07_modality_<gene>_<modality>.sh          (binomial only, one CPU job each)
    <outdir>/07_modality_multinomial_packed.sh         (multinomial only, ONE GPU-node-packed job, all genes)
    <outdir>/08_modality_permutation_<gene>_<modality>.sh    (binomial only, array over reps)
    <outdir>/09_modality_recapitulation_<gene>_<modality>.sh (binomial only, array over reps)
    <outdir>/submit_all.sh           dependency-chained submission,
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
from sbatch_blocks import SbatchStep, SbatchArray, SbatchGpuNodeQueue  # noqa: E402

TIME_HOURS = 24.0  # fixed everywhere, per project convention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(THIS_DIR / "config.yaml"))
    parser.add_argument("--modalities-config", default=str(THIS_DIR / "config_modalities.yaml"))
    parser.add_argument("--outdir", default=str(THIS_DIR / "slurm"))
    parser.add_argument("--no-tag", action="store_true", help="Skip creating a git snapshot tag.")
    parser.add_argument("--no-push-tag", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    modalities_cfg = load_yaml(args.modalities_config)["modalities"]

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
    gpu_node_sbatch_lines = cluster.get("gpu_node_sbatch_lines", ["#SBATCH -N 1", "#SBATCH --gpus=8"])
    node_queue_script = str(REPO_DIR_LOCAL / "publication_runs" / "common" / "slurm" / "run_node_queue.sh")
    GPUS_PER_NODE = 8

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    ntc_shared_cfg = cfg["ntc_shared"]
    sf_cfg = cfg["sum_factor"]
    cis_cfg = cfg["cis"]
    comp_cfg = cfg["compensation"]
    trans_cfg = cfg["trans"]
    modalities_dataset_cfg = cfg["modalities"]

    base_cfg = {
        # Lets common/run_*.py scripts' dynamic-import config blocks (e.g.
        # attach_modality below, or a compensation exclude_cells hook)
        # actually find domingo/-local modules at runtime -- see
        # config_utils.ensure_dataset_dir_on_syspath's docstring.
        "_dataset_dir": str(THIS_DIR),
        "data": {"meta": meta_path, "counts": counts_path},
        "model": {
            "modality_name": model_defaults["modality_name"],
            "guide_covariates": model_defaults["guide_covariates"],
            "guide_covariates_ntc": model_defaults["guide_covariates_ntc"],
            "output_dir": output_dir,
            "device": "cpu",
        },
        "ntc": {"set_technical_groups": ntc_shared_cfg["set_technical_groups"]},
    }

    # NOTE: sum_factor_col_old is left at the library default ('sum_factor')
    # for BOTH adjust_ntc_sum_factor and refit_sumfactor -- matches the
    # reference GFI1B script exactly (refit_sumfactor(covariates=[...]) with
    # no sum_factor_col_old override, i.e. NOT 'sum_factor_adj').
    sum_factor_block = {
        "adjust_ntc_sum_factor": {
            "enabled": True,
            "args": {"covariates": sf_cfg["covariates"]},
        },
        "refit_sumfactor": {
            "enabled": True,
            "args": {"covariates": sf_cfg["covariates"]},
        },
    }

    def bd_cmd(kind: str, config_path: Path, extra_args: str = "") -> str:
        """kind: 'fit-ntc'/'fit-cis'/'fit-trans'/'report' -> bayesDREAM CLI;
        otherwise a common/run_<kind>.py script."""
        if kind in {"fit-ntc", "fit-cis", "fit-trans", "report"}:
            return f'cd "{repo_dir}" && "{python_env}" -m bayesDREAM {kind} --config "{config_path}"{extra_args}'
        script = f"{repo_dir}/publication_runs/common/run_{kind}.py"
        return f'"{python_env}" "{script}" --config "{config_path}"{extra_args}'

    def _pinned(cmd: str, cpus: int, gpu_idx: int) -> str:
        # Same rationale as morris/generate_slurm.py's _pinned(): run_node_queue.sh
        # runs each tasklist line as a separate subshell, so each needs its
        # OWN thread cap (else every concurrent task's BLAS/torch threadpool
        # fights over the whole node's cores) and its OWN GPU device
        # assignment (else every concurrent task defaults to GPU 0 instead
        # of spreading across the node's 8). HIP_VISIBLE_DEVICES is the ROCm
        # analogue of CUDA_VISIBLE_DEVICES; ROCR_VISIBLE_DEVICES set
        # alongside as a fallback for older ROCm builds.
        exports = (
            f"OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} "
            f"VECLIB_MAXIMUM_THREADS={cpus} NUMEXPR_NUM_THREADS={cpus} "
            f"HIP_VISIBLE_DEVICES={gpu_idx} ROCR_VISIBLE_DEVICES={gpu_idx}"
        )
        return f"env {exports} {cmd}"

    scripts = []  # (filename, rendered sbatch text) in submission order
    submitted_rows = []  # (stage, label, script) for submit_all.sh's submitted_jobs.tsv
    multinomial_commands = []  # collected across all (gene, multinomial-modality) pairs, packed after the loop

    # ---------------------------------------------------------------- #
    # 1. Shared fit_ntc (deferred cis_gene -- all genes in one fit)      #
    # ---------------------------------------------------------------- #
    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    ntc_bd_cfg = render_bayesdream_config(base_cfg, {
        "model": {"label": label_ntc},
        "data": {"meta": meta_ntc_path, "counts": counts_ntc_path},
        "ntc": {"fit": ntc_shared_cfg.get("fit", {}), "save": True},
    })
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_bd_cfg)

    ntc_step = SbatchStep(
        job_name="domingo_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=TIME_HOURS, cpus=ntc_shared_cfg["resources"]["cores"],
        partition=partition_cpu, repo_dir=repo_dir,
        commands=[bd_cmd("ntc", ntc_cfg_path)],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))
    submitted_rows.append(("ntc_shared", label_ntc, "01_ntc_shared.sh"))

    def subset_dir_for(label: str) -> str:
        return f"{output_dir}/{label}_subset"

    def modality_subset_dir_for(label: str, mod_name: str) -> str:
        return f"{output_dir}/{label}_modality_subset/{mod_name}"

    def subset_data_block(label: str, mode: str) -> dict:
        # feature_meta_read_csv_kwargs: {} is REQUIRED here, not cosmetic --
        # subset_per_gene.py writes gene_meta.csv with to_csv(index=False)
        # (a real gene_name column, no index column at all); without this
        # override, config_utils.build_model_from_config's default
        # (index_col=0, for datasets whose feature_meta genuinely has a
        # leading unnamed index column) would consume that gene_name column
        # AS the index, leaving zero usable columns.
        d = f"{subset_dir_for(label)}/{mode}"
        return {
            "meta": f"{d}/meta.csv", "counts": f"{d}/gene_counts.npz",
            "feature_meta": f"{d}/gene_meta.csv", "feature_meta_read_csv_kwargs": {},
        }

    def write_subset_step(gene: str, label: str) -> str:
        # Deferred cis_gene, full dataset in -- builds the SAME model
        # add_cis_gene() would, just to extract+write BOTH its NTC+cis-cells-
        # only `full` (whole trans panel) and `cis_only` (just the cis gene,
        # for 02_cis_<gene>.sh -- see module docstring's "Per-gene data
        # subsetting" section) results in one pass, instead of proceeding to
        # fit_cis().
        subset_input_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label},
            "cis_gene": gene,
        })
        subset_input_cfg_path = configs_dir / f"{label}_subset_input.yaml"
        write_yaml(subset_input_cfg_path, subset_input_cfg)

        cmd = (
            f'"{python_env}" "{repo_dir}/publication_runs/common/subset_per_gene.py" '
            f'--config "{subset_input_cfg_path}" --outdir "{subset_dir_for(label)}" --modes full,cis_only'
        )
        step = SbatchStep(
            job_name=f"domingo_subset_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[cmd],
        )
        filename = f"01b_subset_{gene}.sh"
        scripts.append((filename, step.render()))
        return filename

    # ---------------------------------------------------------------- #
    # Per-gene stages                                                    #
    # ---------------------------------------------------------------- #
    for gene in cfg["cis_genes"]:
        label = f"{label_prefix}_{gene}"

        subset_script = write_subset_step(gene, label)
        submitted_rows.append(("subset", label, subset_script))

        # -- 2. cis (deferred add_cis_gene, reusing shared ntc) --
        cis_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label},
            "data": subset_data_block(label, "cis_only"),
            "cis_gene": gene,
            "ntc_shared_dir": ntc_shared_dir,
            "sum_factor": {"adjust_ntc_sum_factor": sum_factor_block["adjust_ntc_sum_factor"]},
            "cis": {
                "fit": {**cis_cfg.get("fit", {}), "sum_factor_col": "sum_factor_adj"},
                "save": True,
            },
        })
        cis_cfg_path = configs_dir / f"{label}_cis.yaml"
        write_yaml(cis_cfg_path, cis_bd_cfg)

        cis_step = SbatchStep(
            job_name=f"domingo_cis_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("cis_deferred", cis_cfg_path)],
        )
        scripts.append((f"02_cis_{gene}.sh", cis_step.render()))
        submitted_rows.append(("cis", label, f"02_cis_{gene}.sh"))

        # -- 3. compensation (raw sum_factor, no adjustments -- see run_compensation.py) --
        comp_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            "data": subset_data_block(label, "full"),
            "compensation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "args": {"exclude_cells": comp_cfg["exclude_cells"].get(gene, [])},
            },
        })
        comp_cfg_path = configs_dir / f"{label}_compensation.yaml"
        write_yaml(comp_cfg_path, comp_bd_cfg)

        comp_step = SbatchStep(
            job_name=f"domingo_comp_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=comp_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("compensation", comp_cfg_path)],
        )
        scripts.append((f"03_compensation_{gene}.sh", comp_step.render()))
        submitted_rows.append(("compensation", label, f"03_compensation_{gene}.sh"))

        # -- 4. trans --
        trans_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            "data": subset_data_block(label, "full"),
            "sum_factor": sum_factor_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "trans": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "fit": {**trans_cfg.get("fit", {}), "sum_factor_col": "sum_factor_refit",
                        "function_type": trans_cfg["function_type"], "restart_from_checkpoint": True},
                "save": True,
            },
        })
        trans_cfg_path = configs_dir / f"{label}_trans.yaml"
        write_yaml(trans_cfg_path, trans_bd_cfg)

        trans_step = SbatchStep(
            job_name=f"domingo_trans_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=trans_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("trans", trans_cfg_path)],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"04_trans_{gene}.sh", trans_step.render()))
        submitted_rows.append(("trans", label, f"04_trans_{gene}.sh"))

        # -- 5. permutation null (array over reps) --
        perm_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            "data": subset_data_block(label, "full"),
            "sum_factor": sum_factor_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "permutation": {
                "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "covariates": sf_cfg["covariates"],
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_refit", "function_type": trans_cfg["function_type"]},
            },
        })
        perm_cfg_path = configs_dir / f"{label}_permutation.yaml"
        write_yaml(perm_cfg_path, perm_bd_cfg)

        n_perm = trans_cfg["permutation"]["n_reps"]
        perm_step = SbatchArray(
            job_name=f"domingo_perm_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS,
            cpus=trans_cfg["permutation"]["resources"]["cores"],
            max_index=n_perm - 1, max_concurrent=min(n_perm, 50),
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("permutation_null", perm_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"05_permutation_{gene}.sh", perm_step.render()))
        submitted_rows.append(("permutation", label, f"05_permutation_{gene}.sh"))

        # -- 6. recapitulation simulation (array over reps) --
        sim_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            "data": subset_data_block(label, "full"),
            "sum_factor": sum_factor_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
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
            job_name=f"domingo_sim_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS,
            cpus=trans_cfg["simulation"]["resources"]["cores"],
            max_index=n_sim - 1, max_concurrent=min(n_sim, 50),
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("recapitulation_sim", sim_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
            auto_requeue_on_timeout=True,
        )
        scripts.append((f"06_recapitulation_{gene}.sh", sim_step.render()))
        submitted_rows.append(("recapitulation", label, f"06_recapitulation_{gene}.sh"))

        # -- 7a. per-(gene, modality) subsetting, then 7. extra modalities --
        # Subsetting reads the ONE shared raw splicing directory
        # (modalities.data_dir) and this gene's OWN precomputed `full`
        # subset (needs only 01b_subset_<gene>.sh -- attach_modality() never
        # touches a fitted posterior, so this runs in parallel with
        # 02_cis_<gene>.sh, not after it), then writes an already
        # cell-aligned, min_count-filtered, denominator-computed modality to
        # modality_subset_dir_for(label, mod_name). The real fit job (and,
        # for binomial, permutation/recapitulation's attach_modality: config
        # block) then read THAT instead of the shared raw directory.
        # BINOMIAL modalities: one CPU job each, as before. MULTINOMIAL
        # (donor_choice/acceptor_choice): GPU, and packed together across
        # all genes -- see "GPU-packed multinomial modalities" below.
        modality_subset_input_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "cis_gene": gene},
            "data": subset_data_block(label, "full"),
        })
        modality_subset_input_cfg_path = configs_dir / f"{label}_modality_subset_input.yaml"
        write_yaml(modality_subset_input_cfg_path, modality_subset_input_cfg)

        for spec in modalities_cfg:
            mod_name = spec["stype"]
            is_multinomial = spec["distribution"] == "multinomial"
            mod_data_dir = modalities_dataset_cfg["data_dir"]
            mod_precomputed_dir = modality_subset_dir_for(label, mod_name)
            # Computed once, reused both below (mod_cfg, so plain
            # `<label>_modality_<mod>.yaml` is directly usable by
            # common/profile_memory.py's --modality-name, matching that
            # script's own docstring) and further down for permutation/
            # recapitulation configs.
            attach_block = {
                "attach_modality": {
                    "module": "load_modalities",
                    "function": "attach_modality_precomputed",
                    "kwargs": {"spec": spec, "precomputed_dir": mod_precomputed_dir},
                },
            }

            subset_mod_cmd = (
                f'"{python_env}" "{repo_dir}/publication_runs/domingo/subset_modality_per_gene.py" '
                f'--config "{modality_subset_input_cfg_path}" --modality-name {mod_name} '
                f'--modality-spec "{repo_dir}/publication_runs/domingo/config_modalities.yaml" '
                f'--data-dir "{mod_data_dir}" --outdir "{mod_precomputed_dir}"'
            )
            subset_mod_step = SbatchStep(
                job_name=f"domingo_modsubset_{gene}_{mod_name}", account=account, log_dir=str(logs_dir),
                time_hours=TIME_HOURS, cpus=modalities_dataset_cfg["resources"]["cores"],
                partition=partition_cpu, repo_dir=repo_dir, commands=[subset_mod_cmd],
            )
            subset_mod_filename = f"07a_modality_subset_{gene}_{mod_name}.sh"
            scripts.append((subset_mod_filename, subset_mod_step.render()))
            submitted_rows.append((f"modality_subset_{mod_name}", label, subset_mod_filename))

            mod_cfg = render_bayesdream_config(base_cfg, {
                "model": {"label": label, "cis_gene": gene, "device": "cuda" if is_multinomial else "cpu"},
                "data": subset_data_block(label, "full"),
                "cis": {"load_cis": {"enabled": True}},
                **attach_block,
            })
            mod_cfg_path = configs_dir / f"{label}_modality_{mod_name}.yaml"
            write_yaml(mod_cfg_path, mod_cfg)

            mod_env = python_env_gpu if is_multinomial else python_env
            mod_cmd = (
                f'cd "{repo_dir}" && "{mod_env}" "{repo_dir}/publication_runs/domingo/load_modalities.py" '
                f'--config "{mod_cfg_path}" --modality-name {mod_name} '
                f'--modality-spec "{repo_dir}/publication_runs/domingo/config_modalities.yaml" '
                f'--precomputed-dir "{mod_precomputed_dir}"'
            )

            if is_multinomial:
                multinomial_commands.append(mod_cmd)
            else:
                mod_step = SbatchStep(
                    job_name=f"domingo_mod_{gene}_{mod_name}", account=account, log_dir=str(logs_dir),
                    time_hours=TIME_HOURS,
                    cpus=modalities_dataset_cfg["resources"]["cores"],
                    partition=partition_cpu, repo_dir=repo_dir,
                    commands=[mod_cmd],
                )
                scripts.append((f"07_modality_{gene}_{mod_name}.sh", mod_step.render()))
                submitted_rows.append((f"modality_{mod_name}", label, f"07_modality_{gene}_{mod_name}.sh"))

            # -- 7b. permutation + recapitulation for BINOMIAL modalities   --
            # only (not multinomial, per instruction). Both depend on THIS
            # modality's own job above (07_modality_...), not just cis --
            # they need that job's saved ntc fit (alpha_y_prefit prior for
            # fit_trans) and, for recapitulation, its saved
            # trans_feature_summary.csv as ground truth.
            if spec["distribution"] == "binomial":
                # attach_block computed once above, reused here.
                mod_fit_args = {
                    "function_type": spec["function_type"],
                    "min_denominator": spec.get("min_denominator", 0),
                }

                mod_perm_cfg = render_bayesdream_config(base_cfg, {
                    "model": {"label": label, "cis_gene": gene},
                    "data": subset_data_block(label, "full"),
                    **attach_block,
                    "sum_factor": sum_factor_block,
                    "permutation": {
                        "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                        "load_cis": {"enabled": True},
                        "covariates": sf_cfg["covariates"],
                        "sum_factor_col": "sum_factor_adj",
                        "fit": mod_fit_args,
                    },
                })
                mod_perm_cfg_path = configs_dir / f"{label}_modality_{mod_name}_permutation.yaml"
                write_yaml(mod_perm_cfg_path, mod_perm_cfg)

                n_mod_perm = modalities_dataset_cfg["trans"]["permutation"]["n_reps"]
                mod_perm_step = SbatchArray(
                    job_name=f"domingo_modperm_{gene}_{mod_name}", account=account, log_dir=str(logs_dir),
                    time_hours=TIME_HOURS, cpus=modalities_dataset_cfg["resources"]["cores"],
                    max_index=n_mod_perm - 1, max_concurrent=min(n_mod_perm, 50),
                    partition=partition_cpu, repo_dir=repo_dir,
                    commands=[bd_cmd("permutation_null", mod_perm_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
                    auto_requeue_on_timeout=True,
                )
                scripts.append((f"08_modality_permutation_{gene}_{mod_name}.sh", mod_perm_step.render()))
                submitted_rows.append((f"modality_permutation_{mod_name}", label,
                                       f"08_modality_permutation_{gene}_{mod_name}.sh"))

                mod_sim_cfg = render_bayesdream_config(base_cfg, {
                    "model": {"label": label, "cis_gene": gene},
                    "data": subset_data_block(label, "full"),
                    **attach_block,
                    "sum_factor": sum_factor_block,
                    "simulation": {
                        "load_ntc": {"args": {"input_dir": ntc_shared_dir, "mask_features": True}},
                        "load_cis": {"enabled": True},
                        "load_trans": {"enabled": True},
                        "fit": mod_fit_args,
                    },
                })
                mod_sim_cfg_path = configs_dir / f"{label}_modality_{mod_name}_simulation.yaml"
                write_yaml(mod_sim_cfg_path, mod_sim_cfg)

                n_mod_sim = modalities_dataset_cfg["trans"]["simulation"]["n_reps"]
                mod_sim_step = SbatchArray(
                    job_name=f"domingo_modsim_{gene}_{mod_name}", account=account, log_dir=str(logs_dir),
                    time_hours=TIME_HOURS, cpus=modalities_dataset_cfg["resources"]["cores"],
                    max_index=n_mod_sim - 1, max_concurrent=min(n_mod_sim, 50),
                    partition=partition_cpu, repo_dir=repo_dir,
                    commands=[bd_cmd("recapitulation_sim", mod_sim_cfg_path, extra_args=" --rep $SLURM_ARRAY_TASK_ID")],
                    auto_requeue_on_timeout=True,
                )
                scripts.append((f"09_modality_recapitulation_{gene}_{mod_name}.sh", mod_sim_step.render()))
                submitted_rows.append((f"modality_recapitulation_{mod_name}", label,
                                       f"09_modality_recapitulation_{gene}_{mod_name}.sh"))

    # ---------------------------------------------------------------- #
    # 7c. GPU-packed multinomial modalities (donor_choice/acceptor_choice) #
    #    across ALL genes: len(cis_genes) x 2 = 8 tasks -- exactly one     #
    #    Dardel GPU node's worth (8 GPUs/node) at concurrency=8, so this   #
    #    is ONE SbatchGpuNodeQueue submission, not one job per (gene,      #
    #    modality) pair. No permutation/recapitulation counterpart        #
    #    (multinomial is excluded from those, same as the per-gene path). #
    # ---------------------------------------------------------------- #
    multinomial_script = None
    if multinomial_commands:
        cpus = modalities_dataset_cfg["resources"]["cores"]
        concurrency = min(len(multinomial_commands), GPUS_PER_NODE)
        pinned_commands = [_pinned(cmd, cpus, i % concurrency) for i, cmd in enumerate(multinomial_commands)]
        tasklist_path = configs_dir / "07_modality_multinomial_packed_tasklist.txt"
        tasklist_path.write_text("\n".join(pinned_commands) + "\n")
        mod_multi_step = SbatchGpuNodeQueue(
            job_name="domingo_mod_multinomial_packed", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, tasklist_path=str(tasklist_path), concurrency=concurrency,
            gpu_partition=partition_gpu, node_queue_script=node_queue_script, repo_dir=repo_dir,
            gpu_sbatch_lines=gpu_node_sbatch_lines, auto_requeue_on_timeout=True,
        )
        multinomial_script = "07_modality_multinomial_packed.sh"
        scripts.append((multinomial_script, mod_multi_step.render()))
        submitted_rows.append(("modality_multinomial_packed", f"{len(cfg['cis_genes'])} genes x 2 modalities (packed)",
                                multinomial_script))

    for filename, text in scripts:
        (outdir / filename).write_text(text)
        os.chmod(outdir / filename, 0o755)

    # ---------------------------------------------------------------- #
    # submit_all.sh -- ntc_shared, then per-gene cis -> {compensation,   #
    # trans (-> permutation, recapitulation, modalities)} in parallel    #
    # branches per gene. Dependencies expressed via --dependency=afterok #
    # against the ntc_shared and per-gene cis job IDs. Writes            #
    # submitted_jobs.tsv for common/slurm/list_job_status.py.            #
    # ---------------------------------------------------------------- #
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "",
        'TSV="submitted_jobs.tsv"',
        'echo -e "stage\\tlabel\\tjobid\\tscript" > "$TSV"',
        "",
        'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)',
        'echo -e "ntc_shared\\tntc_shared\\t$NTC_JOB\\t01_ntc_shared.sh" >> "$TSV"',
        "",
    ]
    all_multinomial_modsubset_deps = []  # accumulated across genes, for the packed multinomial job below
    for gene in cfg["cis_genes"]:
        submit_lines += [
            f'SUBSET_{gene}=$(sbatch --parsable --dependency=afterok:$NTC_JOB 01b_subset_{gene}.sh)',
            f'echo -e "subset\\t{gene}\\t$SUBSET_{gene}\\t01b_subset_{gene}.sh" >> "$TSV"',
            f'CIS_{gene}=$(sbatch --parsable --dependency=afterok:$NTC_JOB:$SUBSET_{gene} 02_cis_{gene}.sh)',
            f'echo -e "cis\\t{gene}\\t$CIS_{gene}\\t02_cis_{gene}.sh" >> "$TSV"',
            f'COMP_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 03_compensation_{gene}.sh)',
            f'echo -e "compensation\\t{gene}\\t$COMP_{gene}\\t03_compensation_{gene}.sh" >> "$TSV"',
            f'TRANS_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 04_trans_{gene}.sh)',
            f'echo -e "trans\\t{gene}\\t$TRANS_{gene}\\t04_trans_{gene}.sh" >> "$TSV"',
            f'PERM_{gene}=$(sbatch --parsable --dependency=afterok:$TRANS_{gene} 05_permutation_{gene}.sh)',
            f'echo -e "permutation\\t{gene}\\t$PERM_{gene}\\t05_permutation_{gene}.sh" >> "$TSV"',
            f'SIM_{gene}=$(sbatch --parsable --dependency=afterok:$TRANS_{gene} 06_recapitulation_{gene}.sh)',
            f'echo -e "recapitulation\\t{gene}\\t$SIM_{gene}\\t06_recapitulation_{gene}.sh" >> "$TSV"',
        ]
        for spec in modalities_cfg:
            mod_name = spec["stype"]
            # Subsetting only needs SUBSET_{gene} (01b) -- attach_modality()
            # never touches a fitted posterior -- so it runs in parallel
            # with CIS_{gene}, not after it.
            submit_lines += [
                f'MODSUBSET_{gene}_{mod_name}=$(sbatch --parsable --dependency=afterok:$SUBSET_{gene} 07a_modality_subset_{gene}_{mod_name}.sh)',
                f'echo -e "modality_subset_{mod_name}\\t{gene}\\t$MODSUBSET_{gene}_{mod_name}\\t07a_modality_subset_{gene}_{mod_name}.sh" >> "$TSV"',
            ]
            if spec["distribution"] == "multinomial":
                all_multinomial_modsubset_deps.append(f"$MODSUBSET_{gene}_{mod_name}")
                continue  # fit packed separately below, one job for ALL genes
            submit_lines += [
                f'MOD_{gene}_{mod_name}=$(sbatch --parsable --dependency=afterok:$CIS_{gene}:$MODSUBSET_{gene}_{mod_name} 07_modality_{gene}_{mod_name}.sh)',
                f'echo -e "modality_{mod_name}\\t{gene}\\t$MOD_{gene}_{mod_name}\\t07_modality_{gene}_{mod_name}.sh" >> "$TSV"',
            ]
            if spec["distribution"] == "binomial":
                submit_lines += [
                    f'sbatch --dependency=afterok:$MOD_{gene}_{mod_name} 08_modality_permutation_{gene}_{mod_name}.sh',
                    f'sbatch --dependency=afterok:$MOD_{gene}_{mod_name} 09_modality_recapitulation_{gene}_{mod_name}.sh',
                ]
        submit_lines.append("")

    if multinomial_script:
        # Packed job needs EVERY gene's cis fit AND every gene's multinomial
        # modality subsetting done (it runs all 4 genes x 2 modalities'
        # worth of tasks) -- multi-job afterok dependency.
        cis_dep = ":".join(f"$CIS_{gene}" for gene in cfg["cis_genes"])
        modsubset_dep = ":".join(all_multinomial_modsubset_deps)
        submit_lines += [
            f'MOD_MULTI=$(sbatch --parsable --dependency=afterok:{cis_dep}:{modsubset_dep} {multinomial_script})',
            f'echo -e "modality_multinomial_packed\\tall_genes\\t$MOD_MULTI\\t{multinomial_script}" >> "$TSV"',
            "",
        ]

    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")
    print(f"[generate_slurm] permutation/recapitulation are array jobs (one array submission each per gene) "
          f"-- check MaxSubmitJobs if this dataset's gene/rep counts grow (see README.md).")


if __name__ == "__main__":
    main()
