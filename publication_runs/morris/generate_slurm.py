"""
Generate SLURM scripts + configs for the Morris dataset: high-MOI. Two
different regimes, chosen so every stage loads the smallest data it can:

- **primary_genes** (5, config.yaml) get the FULL pipeline (fit_ntc -> cis ->
  compensation -> trans -> permutation -> recapitulation), and each gets its
  OWN fit_ntc -- fit on just that gene's own NTC+cis-cells subset (full
  trans panel), not the whole dataset. No shared ntc_shared for these.
- **sweep_genes** (~116, every OTHER padj<0.05 gene) get fit_cis ONLY, and
  DO share one fit_ntc (`01_ntc_shared.sh`, full dataset, cis_gene deferred)
  -- with ~116 of them and no trans-modeling need, one shared fit is cheaper
  in aggregate than 116 separate ones, unlike the primary genes' case.

Both regimes reuse the SAME deferred-cis_gene + add_cis_gene() mechanism
Domingo uses (bayesDREAM's high-MOI mode supports it natively -- no more
placeholder-gene/manual-alpha-extraction workaround, see morris/README.md).

Assumes morris/preprocess.py has already been run once (writes the aligned
meta.csv/gene_counts.npz/gene_meta.csv/guide_assignment.npy/guide_meta.csv/
guide_target.csv this script's paths.data_dir points at).

Placement: cis and compensation always run on CPU (bayesdream_cpu env,
`-p shared`); fit_ntc/trans/permutation/recapitulation run on GPU
(bayesdream_rocm env, `-p gpu`) -- per project convention (Domingo is
CPU-only everywhere; Morris's fit_cis specifically is CPU-only regardless of
dataset; everything else in Morris follows its own reference scripts, which
required CUDA). Only trans/permutation/recapitulation auto-resubmit on
timeout (fit_trans has its own checkpoint/resume) -- ntc/cis/compensation/
cis_sweep failures are left for manual review via common/slurm/list_job_status.py.

GPU node packing: a Dardel GPU node has 8 GPUs (cluster.gpu_node_sbatch_lines
below), and none of a single gene's fit_ntc/trans/permutation/recapitulation
needs a whole node to itself, so each of these is ONE SbatchGpuNodeQueue job
across all 5 primary_genes instead of 5 separate node requests:
- `01d_ntc_packed.sh`: 5 tasks, concurrency=5.
- `04_trans_packed.sh`: 5 tasks, concurrency=5.
- `05_permutation_packed.sh` / `06_recapitulation_packed.sh`: 5 genes x
  n_reps tasks (5 with n_reps=1) at concurrency=min(n_tasks, 8).
Packed jobs still auto-resubmit on timeout (SbatchGpuNodeQueue's
auto_requeue_on_timeout=True) -- see its docstring in sbatch_blocks.py for
the "re-runs already-finished tasks too" caveat this implies for a packed
job specifically (individual per-gene jobs didn't have this caveat).

Per-gene data subsetting (01b_subset_<gene>.sh / 01c_subset_sweep.sh):
real profiling found bayesDREAM.__init__ itself dominating per-gene job cost
(38.7s / 21GB for ONE gene's compensation config -- full 31468-gene x
52852-cell load + high-MOI classification against 1871 guides, THEN
discarding ~40k of those cells). common/subset_per_gene.py pays this cost
ONCE per gene (builds the SAME deferred+add_cis_gene() model construction
fit_cis itself uses, then writes the result to disk instead of proceeding to
fit_cis()) and writes BOTH subsetting modes in that ONE pass (add_cis_gene()
already separates 'cis' from the trans panel internally -- both pieces are
in memory regardless of which mode(s) are requested):
- `full/` (primary_genes' 01b_subset_<gene>.sh): entire trans-gene panel,
  cis gene's row included -- used by that gene's own fit_ntc, compensation,
  trans, permutation, recapitulation (all of which need the whole panel).
- `cis_only/` (both primary_genes' 01b_subset_<gene>.sh AND sweep_genes'
  01c_subset_sweep.sh): ONLY the cis gene's row -- used by 02_cis_<gene>.sh
  and 07_cis_sweep.sh, neither of which touch the trans panel at all. Both
  use the DEFERRED add_cis_gene() pattern, which tolerates a 1-gene starting
  panel with zero code changes (confirmed by direct test -- no equivalent to
  the eager-construction path's "No genes left after filtering!" raise), so
  bayesDREAM's cis_only=True flag isn't needed here; it exists for eager
  construction in general (see bayesDREAM/model.py's cis_only docstring).
subset_per_gene.py itself no longer touches ntc_shared_dir/load_ntc_fit() at
all -- it never needed it for the subsetting itself, and now that primary
genes don't have a shared ntc to load from, requiring one here would be
actively wrong.

Correctness note: `exclude_guides` is a bayesDREAM CONSTRUCTOR-time filter,
so every stage that constructs its OWN fresh model for a given gene (cis via
deferred+add_cis_gene, fit_ntc/compensation/trans/permutation/recapitulation
via cis_gene-at-init or reading the gene's own subset) must use the
IDENTICAL per-gene exclude_guides list, or their cell/guide composition
diverges and load_cis_fit()/load_trans_fit() would be aligning against a
model whose cells don't match what was actually saved.
`per_gene_exclude_guides()` below is computed once per gene and threaded
through every one of that gene's stages -- including its own fit_ntc now,
since that's just another per-gene stage reading the SAME `full` subset the
subsetting step built with this exact list. This does NOT apply to the
sweep genes' SHARED `01_ntc_shared.sh`, which deliberately uses a BROADER
exclude_guides (global_exclude_guides + the full SNP table, unconditionally
-- see `exclude_all_snp_guides`) than any per-gene stage: `add_cis_gene()`
extracts `alpha_y_prefit` per FEATURE (gene), not per cell, from the shared
ntc posteriors, so a wider guide exclusion at that one shared-fit stage
doesn't create a cell-alignment mismatch with the (narrower, per-gene)
exclude_guides used for the sweep genes' own cis-only subset/fit.

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

from config_utils import load_yaml, write_yaml, render_bayesdream_config, resolve_paths  # noqa: E402
from git_provenance import create_stable_snapshot_tag  # noqa: E402
from sbatch_blocks import SbatchStep, SbatchArray, SbatchGpuNodeQueue  # noqa: E402
from snp_exclusion import exclude_guides_for_cis_gene, exclude_all_snp_guides  # noqa: E402

TIME_HOURS = 24.0  # fixed everywhere, per project convention


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

    paths = resolve_paths(cfg["paths"])
    output_dir = paths["output_dir"]
    repo_dir = paths["repo_dir"]
    python_env_cpu = paths["python_env_cpu"]
    python_env_gpu = paths["python_env_gpu"]

    cluster = cfg["cluster"]
    account = cluster["account"]
    partition_cpu = cluster.get("partition_cpu", "shared")
    partition_gpu = cluster.get("partition_gpu", "gpu")
    # Whole-node request for packed GPU jobs (see module docstring's "GPU
    # node packing" section) vs. a single-GPU request for standalone
    # single-task GPU jobs (currently just ntc_shared). Both unconfirmed
    # against the real account/partition -- see config.yaml's comment.
    gpu_node_sbatch_lines = cluster.get("gpu_node_sbatch_lines", ["#SBATCH -N 1", "#SBATCH --gpus=8"])
    gpu_single_sbatch_lines = cluster.get("gpu_single_sbatch_lines", ["#SBATCH --gpus=1"])
    node_queue_script = str(REPO_DIR_LOCAL / "publication_runs" / "common" / "slurm" / "run_node_queue.sh")
    GPUS_PER_NODE = 8

    label_prefix = cfg["label_prefix"]
    model_defaults = cfg["model_defaults"]
    global_exclude_guides = cfg.get("global_exclude_guides") or []
    sf_cfg = cfg["sum_factor"]
    gene_sel_cfg = cfg["gene_selection"]
    ntc_shared_cfg = cfg["ntc_shared"]
    subset_cfg = cfg["subset"]
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
        # meta_read_csv_kwargs/feature_meta_read_csv_kwargs: {} are BOTH
        # REQUIRED, not cosmetic -- preprocess.py writes both meta.csv and
        # gene_meta.csv with to_csv(index=False) (real columns, no leading
        # index column at all). meta_read_csv_kwargs was missing here until
        # 2026-08-12 (same bug as Domingo's identical gap in
        # domingo/generate_slurm.py's base_cfg) -- without it,
        # config_utils._read_counts's default (index_col=0) silently
        # consumes meta.csv's first real column AS self.meta's index instead
        # of a default 0..N-1 range. Harmless for most of the pipeline, but
        # crashes permute_x_true() (bayesDREAM/core.py, the only place that
        # indexes a positional numpy array via self.meta.index directly)
        # with "IndexError: only integers, slices... are valid indices" --
        # i.e. this would have broken 05_permutation_packed.sh the same way
        # it broke Domingo's permutation jobs, the first time it ran.
        "meta_read_csv_kwargs": {},
        "feature_meta": paths["feature_meta"], "feature_meta_read_csv_kwargs": {},
        "guide_assignment": paths["guide_assignment"],
        "guide_meta": paths["guide_meta"], "guide_target": paths["guide_target"],
    }
    base_cfg = {
        # Lets common/run_compensation.py's exclude_cells dynamic-import
        # hook (morris/compensation_exclude_cells.py) actually find that
        # module at runtime -- see
        # config_utils.ensure_dataset_dir_on_syspath's docstring. Previously
        # missing: the hook's own sys.path candidates never included
        # morris/, so this import would have failed at runtime the first
        # time it was actually exercised (never caught by dry-run testing,
        # which only exercises config *generation*, not execution).
        "_dataset_dir": str(THIS_DIR),
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
    # scran is computed ONCE, up front, on the FULL dataset -- morris/
    # preprocess.py writes it straight into meta.csv's 'sum_factor' column
    # (via common/compute_scran_sum_factor.py's _compute_scran_sizefactors),
    # not recomputed per-gene here anymore. This used to be per-gene (via
    # compute_scran, right after add_cis_gene() in subset_per_gene.py,
    # writing 'sum_factor_new') because scran needs the full gene panel to
    # mean anything and can't run on a single-gene cis_only subset -- but
    # that meant fit_ntc's alpha_y_mult/alpha_x_prefit (estimated once,
    # against the shared ntc_shared/each primary gene's OWN 'sum_factor') and
    # fit_cis/fit_trans's sum_factor_adj (derived from a separately, later
    # recomputed 'sum_factor_new') were calibrated against two DIFFERENT
    # normalizations, composed multiplicatively inside bayesDREAM
    # (mu_final = mu_y * alpha_y * sum_factor) -- see
    # compute_scran_sum_factor.py's module docstring for the full rationale.
    # Now every stage -- ntc_shared, each primary gene's own fit_ntc, and
    # every cis/trans/permutation/recapitulation stage's adjust_ntc_sum_factor
    # -- reads the SAME 'sum_factor' column, mirroring Domingo's design
    # (one shared sum_factor, only ever adjusted, never independently
    # recomputed downstream).
    sum_factor_cis_block = {
        "compute_scran": {"enabled": False},
        "adjust_ntc_sum_factor": {
            "enabled": True,
            "args": {"sum_factor_col_old": "sum_factor", "covariates": sf_cfg["covariates"]},
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
    #    global_exclude_guides + the FULL SNP table (unconditionally,    #
    #    no per-gene exception -- see exclude_all_snp_guides) applied    #
    #    here. Per-gene SNP exclusion (SNP table minus that gene's own   #
    #    exception) happens separately, later, per gene.                 #
    # ---------------------------------------------------------------- #
    label_ntc = f"{label_prefix}_ntc_shared"
    ntc_shared_dir = f"{output_dir}/{label_ntc}"
    # ntc_shared isn't "for" any one cis gene (cis_gene is deferred here) --
    # so unlike every per-gene stage below (which excludes the SNP table
    # MINUS that gene's own exception), ntc_shared excludes the SNP table's
    # guides UNCONDITIONALLY (no exception). This matters because high-MOI's
    # deferred-cis_gene fit_ntc() defaults to use_all_cells=True (see
    # bayesDREAM/fitting/ntc.py) -- without this, cells carrying these
    # "extensive trans effects" guides would be included, unfiltered, in the
    # ONE shared alpha_y_prefit every cis gene's add_cis_gene() later
    # extracts from.
    ntc_exclude_guides = sorted(set(global_exclude_guides) | set(exclude_all_snp_guides(all_guide_names)))
    ntc_bd_cfg = render_bayesdream_config(base_cfg, {
        "model": {"label": label_ntc, "exclude_guides": ntc_exclude_guides},
        "ntc": {"fit": {}, "save": True},
    })
    ntc_cfg_path = configs_dir / f"{label_ntc}.yaml"
    write_yaml(ntc_cfg_path, ntc_bd_cfg)

    ntc_step = SbatchStep(
        job_name="morris_ntc_shared", account=account, log_dir=str(logs_dir),
        time_hours=TIME_HOURS, cpus=ntc_shared_cfg["resources"]["cores"],
        partition=partition_gpu, repo_dir=repo_dir,
        extra_sbatch_lines=gpu_single_sbatch_lines,
        commands=[bd_cmd("ntc", ntc_cfg_path, python_env_gpu)],
    )
    scripts.append(("01_ntc_shared.sh", ntc_step.render()))
    submitted_rows.append(("ntc_shared", label_ntc, "01_ntc_shared.sh"))

    def gene_output_dir(label: str) -> str:
        # Every per-gene stage's own output_dir (cis/compensation/trans/etc
        # all already save here) -- also where that gene's OWN fit_ntc now
        # saves to, so it doubles as that gene's "ntc dir" for load_ntc_fit()
        # calls below. No separate directory needed.
        return f"{output_dir}/{label}"

    def render_gene_cfg(gene: str, label: str, device: str, exclude_guides: list, extra: dict) -> dict:
        overrides = {
            "model": {"label": label, "cis_gene": gene, "device": device, "exclude_guides": exclude_guides},
            "data": subset_data_block(label, "full"),
            **extra,
        }
        return render_bayesdream_config(base_cfg, overrides)

    def subset_dir_for(label: str) -> str:
        return f"{output_dir}/{label}_subset"

    def subset_data_block(label: str, mode: str) -> dict:
        # Per-gene precomputed subset (see module docstring's "Per-gene data
        # subsetting" section) -- guide_target is the only file NOT written
        # per-gene by subset_per_gene.py (small lookup table; the ALREADY-
        # pruned guide_meta.csv it's paired with restricts which of its rows
        # are actually relevant, so reusing the global one is harmless).
        d = f"{subset_dir_for(label)}/{mode}"
        return {
            "meta": f"{d}/meta.csv", "counts": f"{d}/gene_counts.npz",
            "feature_meta": f"{d}/gene_meta.csv", "feature_meta_read_csv_kwargs": {},
            "guide_assignment": f"{d}/guide_assignment.npy", "guide_meta": f"{d}/guide_meta.csv",
            "guide_target": paths["guide_target"],
        }

    def render_cis_stage_config(gene: str, label: str, exclude_guides: list, data_block_override: dict, ntc_dir, filename_suffix: str, force: bool = False) -> Path:
        # Deferred (cis_gene NOT in model:) -- add_cis_gene() commits, same
        # mechanism as Domingo, now that bayesDREAM's high-MOI mode supports
        # it directly. Shared by three callers with different
        # data_block_override/ntc_dir: the subsetting step itself (full
        # dataset in, no ntc_dir at all -- subset_per_gene.py ignores it),
        # the real 02_cis_<gene>.sh stage (cis_only subset in, THAT GENE's
        # own ntc dir), and cis_sweep's per-gene task (cis_only subset in,
        # the GLOBAL shared ntc dir).
        # subset_input no longer needs a sum_factor block at all -- 'sum_factor'
        # already exists in data.meta, written once by morris/preprocess.py (see
        # sum_factor_cis_block's comment above); subset_per_gene.py just carries
        # it through unchanged into full/meta.csv and cis_only/meta.csv. The real
        # cis/cis_sweep fit applies adjust_ntc_sum_factor (sum_factor_cis_block).
        # force=True is used ONLY for a subset of sweep_genes whose cis fit hit
        # core.py's "low NTC expression (log2 < -1)" guard on 2026-08-15 --
        # these are secondary sweep genes (fit_cis only, no trans modeling), so
        # we'd rather still get a point estimate and filter unreliable ones
        # manually downstream than drop them from the sweep entirely. NOT set
        # for primary_genes, whose cis fit feeds the full trans pipeline and
        # should keep failing loudly on low expression.
        cis_fit_args = {"sum_factor_col": "sum_factor_adj", "independent_mu_sigma": True}
        if force:
            cis_fit_args["force"] = True
        overrides = {
            "model": {"label": label, "device": "cpu", "exclude_guides": exclude_guides},
            "data": data_block_override,
            "cis_gene": gene,
            "cis": {"fit": cis_fit_args, "save": True},
        }
        if filename_suffix != "subset_input":
            overrides["sum_factor"] = sum_factor_cis_block
        if ntc_dir:
            overrides["ntc_shared_dir"] = ntc_dir
        cis_bd_cfg = render_bayesdream_config(base_cfg, overrides)
        cis_cfg_path = configs_dir / f"{label}_{filename_suffix}.yaml"
        write_yaml(cis_cfg_path, cis_bd_cfg)
        return cis_cfg_path

    def render_cis_config(gene: str, label: str, exclude_guides: list, ntc_dir: str, force: bool = False) -> Path:
        return render_cis_stage_config(gene, label, exclude_guides, subset_data_block(label, "cis_only"), ntc_dir, "cis", force=force)

    def cis_sbatch_step(gene: str, cis_cfg_path: Path) -> SbatchStep:
        return SbatchStep(
            job_name=f"morris_cis_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=cis_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir,
            commands=[bd_cmd("cis_deferred", cis_cfg_path, python_env_cpu)],
        )

    def subset_cmd(config_path, outdir_arg: str, modes: str) -> str:
        script = f"{repo_dir}/publication_runs/common/subset_per_gene.py"
        return f'"{python_env_cpu}" "{script}" --config "{config_path}" --outdir "{outdir_arg}" --modes {modes}'

    def write_subset_step(gene: str, label: str, exclude_guides: list, modes: str) -> str:
        subset_input_cfg_path = render_cis_stage_config(gene, label, exclude_guides, data_block, None, "subset_input")
        cmd = subset_cmd(subset_input_cfg_path, subset_dir_for(label), modes)
        step = SbatchStep(
            job_name=f"morris_subset_{gene}", account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, cpus=subset_cfg["resources"]["cores"],
            partition=partition_cpu, repo_dir=repo_dir, commands=[cmd],
        )
        filename = f"01b_subset_{gene}.sh"
        scripts.append((filename, step.render()))
        return filename

    # ---------------------------------------------------------------- #
    # 2. Primary genes: cis(CPU) -> compensation(CPU) -> trans/          #
    #    permutation/recapitulation (GPU, packed onto one node each      #
    #    across all 5 primary genes -- see module docstring's "GPU node  #
    #    packing" section).                                              #
    # ---------------------------------------------------------------- #
    ntc_commands = []
    trans_commands = []
    perm_commands = []
    sim_commands = []
    for gene in primary_genes:
        label = f"{label_prefix}_{gene}"
        exclude_guides = per_gene_exclude_guides(gene)
        this_gene_ntc_dir = gene_output_dir(label)

        subset_script = write_subset_step(gene, label, exclude_guides, "full,cis_only")
        submitted_rows.append(("subset", label, subset_script))

        # -- own fit_ntc, packed with the other 4 primary genes below --
        # Fit on the SAME `full` subset (NTC+gene cells, whole trans panel)
        # trans/compensation/etc. use -- fit_ntc still needs to estimate
        # alpha_y for every trans gene fit_trans will model, just on a
        # smaller, gene-specific cell population than the old shared fit.
        ntc_gene_bd_cfg = render_bayesdream_config(base_cfg, {
            "model": {"label": label, "exclude_guides": exclude_guides},
            "data": subset_data_block(label, "full"),
            "ntc": {"fit": {}, "save": True},
        })
        ntc_gene_cfg_path = configs_dir / f"{label}_ntc.yaml"
        write_yaml(ntc_gene_cfg_path, ntc_gene_bd_cfg)
        ntc_commands.append(bd_cmd("ntc", ntc_gene_cfg_path, python_env_gpu))

        cis_cfg_path = render_cis_config(gene, label, exclude_guides, this_gene_ntc_dir)
        cis_step = cis_sbatch_step(gene, cis_cfg_path)
        scripts.append((f"02_cis_{gene}.sh", cis_step.render()))
        submitted_rows.append(("cis", label, f"02_cis_{gene}.sh"))

        cis_gene_ensembl_id = name_to_id.get(gene)
        comp_bd_cfg = render_gene_cfg(gene, label, "cpu", exclude_guides, {
            "compensation": {
                "load_ntc": {"args": {"input_dir": this_gene_ntc_dir, "mask_features": True}},
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
                "load_ntc": {"args": {"input_dir": this_gene_ntc_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
                "save": True,
            },
        })
        trans_cfg_path = configs_dir / f"{label}_trans.yaml"
        write_yaml(trans_cfg_path, trans_bd_cfg)
        trans_commands.append(bd_cmd("trans", trans_cfg_path, python_env_gpu))

        perm_bd_cfg = render_gene_cfg(gene, label, None, exclude_guides, {
            "sum_factor": sum_factor_trans_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "permutation": {
                "load_ntc": {"args": {"input_dir": this_gene_ntc_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                "covariates": sf_cfg["covariates"],
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
            },
        })
        perm_cfg_path = configs_dir / f"{label}_permutation.yaml"
        write_yaml(perm_cfg_path, perm_bd_cfg)
        n_perm = trans_cfg["permutation"]["n_reps"]
        for rep in range(n_perm):
            perm_commands.append(bd_cmd("permutation_null", perm_cfg_path, python_env_gpu, extra_args=f" --rep {rep}"))

        sim_bd_cfg = render_gene_cfg(gene, label, None, exclude_guides, {
            "sum_factor": sum_factor_trans_block,
            "exclude_trans_genes": {"enabled": True, "args": trans_cfg["exclude_trans_genes"]},
            "simulation": {
                "load_ntc": {"args": {"input_dir": this_gene_ntc_dir, "mask_features": True}},
                "load_cis": {"enabled": True},
                # subset_features=True -- same rationale as Domingo's
                # identical fix (domingo/generate_slurm.py): this config
                # re-applies the SAME exclude_trans_genes(min_log2_mu_ntc=...)
                # the real trans fit used, but that filter isn't perfectly
                # reproducible run-to-run, so a strict reload can raise
                # "N feature(s) ... not present in the saved trans fit".
                # Recapitulation only needs to match trans's OWN fitted
                # function, so subsetting to whatever it actually covers is
                # correct here, not a workaround.
                "load_trans": {"enabled": True, "args": {"subset_features": True}},
                "sum_factor_col": "sum_factor_adj",
                "fit": {"sum_factor_col": "sum_factor_adj", "function_type": trans_cfg["function_type"]},
            },
        })
        sim_cfg_path = configs_dir / f"{label}_simulation.yaml"
        write_yaml(sim_cfg_path, sim_bd_cfg)
        n_sim = trans_cfg["simulation"]["n_reps"]
        for rep in range(n_sim):
            sim_commands.append(bd_cmd("recapitulation_sim", sim_cfg_path, python_env_gpu, extra_args=f" --rep {rep}"))

    # ---------------------------------------------------------------- #
    # 2b. Packed GPU-node jobs: one submission each for trans/           #
    #    permutation/recapitulation across all 5 primary_genes, sharing  #
    #    one node (see module docstring's "GPU node packing" section).   #
    #    cpus-per-task below sizes EACH concurrent task's own thread     #
    #    pool (trans_cfg["resources"]["cores"], as before) -- packing    #
    #    does not change how many cores an individual fit uses, only     #
    #    how many fits share one node's GPUs at once.                    #
    # ---------------------------------------------------------------- #
    def _pinned(cmd: str, cpus: int, gpu_idx: int) -> str:
        # run_node_queue.sh runs each tasklist line via `eval` in its own
        # subshell -- unlike SbatchStep/SbatchArray (whole-job export), each
        # packed task needs its OWN per-command prefix for two separate
        # reasons:
        # 1. Thread pinning: without it every concurrent task's BLAS/torch
        #    threadpool would try to claim the WHOLE node's cores.
        # 2. GPU device assignment: a whole-node allocation (-N 1 --gpus=8)
        #    exposes ALL 8 GPUs to every process by default -- torch/ROCm
        #    picks device 0 unless told otherwise, so without this EVERY
        #    concurrent task would pile onto the same physical GPU instead
        #    of spreading across the node's 8. HIP_VISIBLE_DEVICES is the
        #    ROCm analogue of CUDA_VISIBLE_DEVICES (python_env_gpu is
        #    bayesdream_rocm); ROCR_VISIBLE_DEVICES set alongside it as a
        #    fallback for older ROCm builds that don't honor the HIP_ name.
        #    Round-robins gpu_idx = task_index % GPUS_PER_NODE across the
        #    tasklist, so this only actually spreads tasks 1:1 across
        #    distinct GPUs when concurrency <= GPUS_PER_NODE (true for every
        #    stage here: trans concurrency=5, permutation/recapitulation
        #    concurrency=min(n_tasks, 8)).
        exports = (
            f"OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} "
            f"VECLIB_MAXIMUM_THREADS={cpus} NUMEXPR_NUM_THREADS={cpus} "
            f"HIP_VISIBLE_DEVICES={gpu_idx} ROCR_VISIBLE_DEVICES={gpu_idx}"
        )
        return f"env {exports} {cmd}"

    def write_packed_gpu_job(step_name: str, job_name: str, commands: list, cpus: int,
                              auto_requeue_on_timeout: bool = True) -> str:
        # NOT verified against the real Dardel GPU node's CPU count:
        # concurrency * cpus must fit within one node's cores (e.g. 5 trans
        # tasks * 16 cores = 80; permutation/recapitulation now reuse this
        # SAME trans.resources.cores value, up to 8 concurrent tasks -- see
        # config.yaml). Confirm via `sinfo -p gpu -o "%P %c %N"` before
        # submitting; lower trans.resources.cores if it doesn't fit.
        # NOT verified: that HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES is
        # actually the right env var for Dardel's ROCm build/driver version
        # -- confirm on a real GPU node (e.g. `rocm-smi` inside two
        # concurrently-launched tasks) before trusting device isolation.
        concurrency = min(len(commands), GPUS_PER_NODE)
        pinned_commands = [_pinned(cmd, cpus, i % concurrency) for i, cmd in enumerate(commands)]
        tasklist_path = configs_dir / f"{step_name}_tasklist.txt"
        tasklist_path.write_text("\n".join(pinned_commands) + "\n")
        step = SbatchGpuNodeQueue(
            job_name=job_name, account=account, log_dir=str(logs_dir),
            time_hours=TIME_HOURS, tasklist_path=str(tasklist_path), concurrency=concurrency,
            gpu_partition=partition_gpu, node_queue_script=node_queue_script, repo_dir=repo_dir,
            gpu_sbatch_lines=gpu_node_sbatch_lines, auto_requeue_on_timeout=auto_requeue_on_timeout,
        )
        filename = f"{step_name}.sh"
        scripts.append((filename, step.render()))
        return filename

    # NOT separately profiled yet: reuses trans_cfg's cores as the closest
    # available analog (same data shape -- full trans panel, NTC+gene cells
    # -- as fit_ntc will see here) until common/profile_memory.py --stage
    # ntc is run against one of these per-gene configs directly.
    # auto_requeue_on_timeout=False: fit_ntc has NO internal checkpoint (see
    # sbatch_blocks.py's module docstring) -- a requeue would just restart
    # every task in the packed tasklist from scratch, same reason
    # SbatchStep/SbatchArray never set this for ntc/cis/compensation. Only
    # trans/permutation/recapitulation (fit_trans's own checkpoint/resume)
    # get the default True below.
    ntc_packed_script = write_packed_gpu_job(
        "01d_ntc_packed", "morris_ntc_packed", ntc_commands, trans_cfg["resources"]["cores"],
        auto_requeue_on_timeout=False,
    )
    submitted_rows.append(("ntc", f"{len(primary_genes)} primary genes (packed)", ntc_packed_script))

    trans_script = write_packed_gpu_job(
        "04_trans_packed", "morris_trans_packed", trans_commands, trans_cfg["resources"]["cores"],
    )
    submitted_rows.append(("trans", f"{len(primary_genes)} primary genes (packed)", trans_script))

    perm_script = write_packed_gpu_job(
        "05_permutation_packed", "morris_perm_packed", perm_commands, trans_cfg["resources"]["cores"],
    )
    submitted_rows.append(("permutation", f"{len(primary_genes)} genes x {trans_cfg['permutation']['n_reps']} reps (packed)", perm_script))

    sim_script = write_packed_gpu_job(
        "06_recapitulation_packed", "morris_sim_packed", sim_commands, trans_cfg["resources"]["cores"],
    )
    submitted_rows.append(("recapitulation", f"{len(primary_genes)} genes x {trans_cfg['simulation']['n_reps']} reps (packed)", sim_script))

    # ---------------------------------------------------------------- #
    # 3a. cis-ONLY sweep: per-gene data subsetting, ONE array job         #
    #    (mode=cis_only -- see module docstring's "Per-gene data          #
    #    subsetting" section). Same gene order as 3b below, so             #
    #    $SLURM_ARRAY_TASK_ID lines up between the two arrays.             #
    # ---------------------------------------------------------------- #
    sweep_subset_input_list = configs_dir / "cis_sweep_subset_input_configs.txt"
    sweep_subset_outdirs_list = configs_dir / "cis_sweep_subset_outdirs.txt"
    sweep_exclude_guides = {gene: per_gene_exclude_guides(gene) for gene in sweep_genes}
    subset_input_paths = []
    subset_outdirs = []
    for gene in sweep_genes:
        label = f"{label_prefix}_{gene}"
        subset_input_cfg_path = render_cis_stage_config(gene, label, sweep_exclude_guides[gene], data_block, None, "subset_input")
        subset_input_paths.append(str(subset_input_cfg_path))
        subset_outdirs.append(subset_dir_for(label))
    sweep_subset_input_list.write_text("\n".join(subset_input_paths) + "\n")
    sweep_subset_outdirs_list.write_text("\n".join(subset_outdirs) + "\n")

    subset_sweep_array_commands = [
        f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{sweep_subset_input_list}")',
        f'OUTDIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{sweep_subset_outdirs_list}")',
        subset_cmd("$CONFIG", "$OUTDIR", "cis_only"),
    ]
    subset_sweep_step = SbatchArray(
        job_name="morris_subset_sweep", account=account, log_dir=str(logs_dir),
        # Same full-dataset-load + high-MOI-classification cost as
        # 01b_subset_<gene>.sh (reads data_block, the raw full dataset, not a
        # precomputed subset) -- reuses subset.resources.cores, NOT
        # cis_sweep.resources.cores (that's sized for 07_cis_sweep.sh's
        # actual cheap fit_cis call on the already-subsetted data instead).
        time_hours=TIME_HOURS, cpus=subset_cfg["resources"]["cores"],
        max_index=len(sweep_genes) - 1, max_concurrent=cis_sweep_cfg["array_max_concurrent"],
        partition=partition_cpu, repo_dir=repo_dir, commands=subset_sweep_array_commands,
    )
    scripts.append(("01c_subset_sweep.sh", subset_sweep_step.render()))
    submitted_rows.append(("subset_sweep", f"{len(sweep_genes)} genes", "01c_subset_sweep.sh"))

    # ---------------------------------------------------------------- #
    # 3b. cis-ONLY sweep (CPU array, one array submission for all genes) #
    #    -- reads each gene's mode=cis_only subset from 3a, same order.  #
    # ---------------------------------------------------------------- #
    sweep_configs_list = configs_dir / "cis_sweep_configs.txt"
    config_paths = []
    for gene in sweep_genes:
        label = f"{label_prefix}_{gene}"
        cis_cfg_path = render_cis_config(gene, label, sweep_exclude_guides[gene], ntc_shared_dir, force=True)
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
        # Shared, sweep-genes-only fit_ntc -- no dependency, runs immediately.
        'NTC_JOB=$(sbatch --parsable 01_ntc_shared.sh)',
        'echo -e "ntc_shared\\tntc_shared\\t$NTC_JOB\\t01_ntc_shared.sh" >> "$TSV"',
        "",
    ]
    # Primary genes' subsetting -- no dependency either (subset_per_gene.py
    # doesn't touch any ntc fit), so these also start immediately, in
    # parallel with ntc_shared.
    for gene in primary_genes:
        submit_lines += [
            f'SUBSET_{gene}=$(sbatch --parsable 01b_subset_{gene}.sh)',
            f'echo -e "subset\\t{gene}\\t$SUBSET_{gene}\\t01b_subset_{gene}.sh" >> "$TSV"',
        ]
    submit_lines.append("")

    # Packed per-primary-gene fit_ntc: needs every gene's own `full` subset
    # ready first (each packed task reads a different gene's subset).
    subset_dep = ":".join(f"$SUBSET_{gene}" for gene in primary_genes)
    submit_lines += [
        f'NTC_PACKED=$(sbatch --parsable --dependency=afterok:{subset_dep} 01d_ntc_packed.sh)',
        'echo -e "ntc\\tall_primary\\t$NTC_PACKED\\t01d_ntc_packed.sh" >> "$TSV"',
        "",
    ]

    for gene in primary_genes:
        submit_lines += [
            f'CIS_{gene}=$(sbatch --parsable --dependency=afterok:$SUBSET_{gene}:$NTC_PACKED 02_cis_{gene}.sh)',
            f'echo -e "cis\\t{gene}\\t$CIS_{gene}\\t02_cis_{gene}.sh" >> "$TSV"',
            f'COMP_{gene}=$(sbatch --parsable --dependency=afterok:$CIS_{gene} 03_compensation_{gene}.sh)',
            f'echo -e "compensation\\t{gene}\\t$COMP_{gene}\\t03_compensation_{gene}.sh" >> "$TSV"',
            "",
        ]
    # Packed trans/permutation/recapitulation (see module docstring's "GPU
    # node packing" section): one job each, covering all primary_genes, so
    # they must wait for EVERY gene's cis job (trans reads that gene's own
    # saved cis fit -- see render_gene_cfg's "load_cis": {"enabled": True})
    # -- and, transitively via CIS_<gene>, on NTC_PACKED too.
    cis_dep = ":".join(f"$CIS_{gene}" for gene in primary_genes)
    submit_lines += [
        f'TRANS_PACKED=$(sbatch --parsable --dependency=afterok:{cis_dep} 04_trans_packed.sh)',
        'echo -e "trans\\tall_primary\\t$TRANS_PACKED\\t04_trans_packed.sh" >> "$TSV"',
        'PERM_PACKED=$(sbatch --parsable --dependency=afterok:$TRANS_PACKED 05_permutation_packed.sh)',
        'echo -e "permutation\\tall_primary\\t$PERM_PACKED\\t05_permutation_packed.sh" >> "$TSV"',
        'SIM_PACKED=$(sbatch --parsable --dependency=afterok:$TRANS_PACKED 06_recapitulation_packed.sh)',
        'echo -e "recapitulation\\tall_primary\\t$SIM_PACKED\\t06_recapitulation_packed.sh" >> "$TSV"',
        "",
        # Sweep genes' subsetting also has no dependency (same reason as
        # primary genes' above).
        'SUBSET_SWEEP_JOB=$(sbatch --parsable 01c_subset_sweep.sh)',
        'echo -e "subset_sweep\\tall\\t$SUBSET_SWEEP_JOB\\t01c_subset_sweep.sh" >> "$TSV"',
        'SWEEP_JOB=$(sbatch --parsable --dependency=afterok:$NTC_JOB:$SUBSET_SWEEP_JOB 07_cis_sweep.sh)',
        'echo -e "cis_sweep\\tall\\t$SWEEP_JOB\\t07_cis_sweep.sh" >> "$TSV"',
    ]
    (outdir / "submit_all.sh").write_text("\n".join(submit_lines) + "\n")
    os.chmod(outdir / "submit_all.sh", 0o755)

    print(f"[generate_slurm] wrote {len(scripts)} sbatch script(s) + submit_all.sh to {outdir}")
    print(f"[generate_slurm] primary_genes={primary_genes}")
    print(f"[generate_slurm] cis-only sweep: {len(sweep_genes)} genes (CPU array, single submission)")


if __name__ == "__main__":
    main()
