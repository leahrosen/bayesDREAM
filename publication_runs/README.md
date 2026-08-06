# publication_runs/

Per-dataset run infrastructure for the publication analyses on Dardel. One
subdirectory per dataset (`domingo/`, `morris/`, ...), each with its own
`config.yaml` and `generate_slurm.py`, plus a `common/` library shared by all
of them.

This directory is orchestration code, not part of the installable `bayesDREAM`
package — it drives `bayesDREAM` (via `python -m bayesDREAM <stage> --config
...`, the existing Typer CLI in `bayesDREAM/cli.py`, for stages it fully
supports) and a growing set of `common/run_*.py` scripts for stages the CLI
doesn't cover yet (compensation, permutation null, recapitulation
simulation) or gets subtly wrong for this pipeline's needs (`run_trans.py`
adds `adjust_ntc_sum_factor()`/`refit_sumfactor()`, which the CLI's
`fit-trans` never calls).

## Layout

```
publication_runs/
├── common/                          # shared across all datasets
│   ├── config_utils.py              # YAML load/deep-merge/render + model builder (see below)
│   ├── git_provenance.py            # commit/branch/dirty capture + optional stable tag
│   ├── profile_memory.py            # real peak-RSS measurement -> cores needed (see "Memory")
│   ├── compute_scran_sum_factor.py  # rpy2 scran sum factors, per-cell-subset (Morris only)
│   ├── exclude_low_ntc_genes.py     # log2(mu_ntc)<threshold filter, applies to every dataset's trans stage
│   ├── run_cis_deferred.py          # low-MOI shared-ntc cis stage (add_cis_gene)
│   ├── run_cis_high_moi_shared_ntc.py / build_ntc_shared_high_moi.py / apply_shared_ntc_high_moi.py
│   │                                 # high-MOI shared-ntc equivalent — see morris/README.md
│   ├── run_compensation.py          # standalone: model.check_systematic_shift() (always raw sum_factor)
│   ├── run_trans.py                 # standalone: adjust/refit sum factor -> exclude_low_ntc_genes -> fit_trans
│   ├── run_permutation_null.py      # standalone: permute_from_ntc + permute_x_true, then fit_trans
│   ├── run_recapitulation_sim.py    # standalone: simulate_from_trans_summary + refit, compare
│   └── slurm/
│       ├── sbatch_blocks.py         # SBATCH header builders (cpu step / array / gpu-node-queue / auto-requeue)
│       ├── run_node_queue.sh        # packs several tasks onto one node allocation
│       ├── submit_chain.sh          # submits a list of scripts with --dependency=afterok chaining
│       └── list_job_status.py       # sacct status report for manual review (see "Restart policy")
│
├── domingo/                         # dataset: 4 cis genes, primary + splicing/velocity modalities
├── morris/                          # dataset: high-MOI, 5 primary + padj-derived cis-only sweep
├── template_dataset/                # copy this to start a new dataset
└── README.md                        # this file
```

## Conventions

**Stages.** Every dataset run is composed of independently-runnable stages:
`ntc` -> `cis` -> `compensation` -> `trans` -> (`permutation`, `recapitulation`).
Some datasets skip stages or share one stage's output across many runs of
another (e.g. Domingo shares one `ntc` fit across 4 cis genes; Morris shares
one `ntc` fit across 5+hundreds of cis genes).

**Config layering.** Two YAML layers, deliberately kept separate:

1. **`<dataset>/config.yaml`** — the *orchestration* config: input/output
   paths, which cis genes to run, per-step core counts, permutation/
   simulation replicate counts, cluster account. Consumed by
   `<dataset>/generate_slurm.py`. Not understood by `bayesDREAM` itself.
2. **Per-gene/per-stage `bayesdream_config` YAML** — close to (but a
   superset of) the schema `bayesDREAM/cli.py` expects (`data:`/`model:`/
   `ntc:`/`cis:`/`trans:`/`report:` keys, as in
   `examples/gfi1b_cli_config.yaml`, plus `sum_factor:`/
   `exclude_low_ntc_genes:`/`compensation:`/`permutation:`/`simulation:`
   blocks the `common/run_*.py` scripts read). These are *generated* by
   `generate_slurm.py` from `config.yaml` (one per gene per stage, written
   to `<outdir>/configs/`) rather than hand-written, so a dataset's
   4-or-500 genes don't need 4-or-500 hand-maintained files.

**`build_model_from_config` is not `bayesDREAM.cli._build_model`.**
`config_utils.py` duplicates that function's data-loading logic with an
*extended* model-kwarg allow-list (adds `exclude_guides`, `min_count` — both
accepted by `bayesDREAM.__init__` but not forwarded by the CLI's own
allow-list) and support for sparse `.npz` counts (the CLI only reads counts
via `pd.read_csv`; Morris's counts matrix is far too large to be dense CSV).
See `config_utils.py`'s module docstring.

**sum_factor recomputation.** `sum_factor_adj`/`sum_factor_refit`/Morris's
`sum_factor_new` are never persisted by `save_cis_fit()`/`save_trans_fit()`
or restored by `load_cis_fit()` — every stage after `fit_cis` that needs one
recomputes it itself, in-process, from the same `sum_factor:` config block
(`config_utils.apply_sum_factor_adjustments`). `compensation` is the one
exception: `check_systematic_shift()` always uses the raw `sum_factor`
column, by design (both datasets' reference pipelines call it with no
override).

**exclude_low_ntc_genes.** Per project convention, genes with
`log2(mu_ntc) < -4` are excluded from trans fitting in every dataset —
wired into `run_trans.py`/`run_permutation_null.py`/`run_recapitulation_sim.py`
via `common/exclude_low_ntc_genes.py` (there's no `fit_trans()` kwarg for
this; it reuses bayesDREAM's own internal feature-subsetting machinery, see
that module's docstring).

**Git provenance.** Every generated `sbatch` script sources
`common/git_provenance.py` at the top of its job (prints commit/branch/dirty
state into the log) and every `run_*.py` script in `common/` writes a
`provenance.json` next to its output. `generate_slurm.py` itself also calls
`create_stable_snapshot_tag()` once per dataset at generation time (not once
per job — see the docstring in `git_provenance.py`) so the whole batch of jobs
is pinned to one tag.

**Memory ("how many cores do I need?").** Dardel's `shared` CPU partition
hands out a FIXED 888MB per core (`DefMemPerCPU=MaxMemPerCPU=888`, confirmed
via `scontrol show partition shared`) — no separate `--mem`, cores ARE your
memory budget. `common/profile_memory.py` measures REAL peak RSS (via
`resource.getrusage`, same pattern as
`examples/simulation_study/run_recovery_fit.py`'s `_peak_rss_mb`) around a
bare model construction and, optionally, a cheap (`--niters 10`) real
`fit_ntc`/`fit_cis` call — peak memory is set by tensor shapes, not
convergence, so you don't need to wait for a real fit to see its real peak.
Constructor-only memory is a lower bound, not a full estimate — fitting
allocates more on top (Adam optimizer state, posterior sample draws,
gradient buffers). `docs/memory_calculator.py`'s `estimate_memory()` is a
faster, closed-form (no data loading) first pass if you want a ballpark
before staging data on Dardel at all.

**Restart policy.** Only `fit_trans`-derived stages (`trans`/`permutation`/
`recapitulation`) auto-resubmit on timeout — `fit_trans()` has its own
built-in checkpoint/resume (writes `trans_checkpoint_{modality}_latest.pt`
every `checkpoint_interval` steps, default 10,000; a resubmitted job picks
back up close to where it left off). `fit_ntc()`/`fit_cis()` have **no**
checkpoint support in bayesDREAM at all — a timeout there means starting
that stage completely over, so `ntc`/`cis`/`compensation`/`cis_sweep`
stages deliberately do NOT auto-resubmit; they fail/time out and stay
failed, for you to review and decide on manually. Auto-resubmit is
implemented via the standard `--signal=B:USR1@120` + trap + `scontrol
requeue` idiom (`common/slurm/sbatch_blocks.py`'s `auto_requeue_on_timeout`
flag on `SbatchStep`/`SbatchArray`). Every dataset's `submit_all.sh` writes
a `submitted_jobs.tsv` (stage, label, jobid, script); run
`common/slurm/list_job_status.py <submitted_jobs.tsv>` to see every job's
current `sacct` state grouped into a "NEEDS ATTENTION" list.

**GPU node packing.** When a step only needs a fraction of a Dardel GPU node,
don't request one node per task. Instead write a plain-text task list (one
CLI invocation per line) and submit a single `sbatch` that requests one node
and runs `common/slurm/run_node_queue.sh <tasklist> <concurrency>`, which
runs up to `<concurrency>` tasks at a time via a bash job-pool (`wait -n`,
requires bash >=4.3). Neither current dataset actually needs this today
(Domingo is CPU-only; Morris's `fit_cis` is CPU-only and its other GPU
stages are one-model-per-job already) — it's available in
`common/slurm/sbatch_blocks.py`'s `SbatchGpuNodeQueue` for whenever a future
dataset does.

**Dardel's MaxSubmitJobs gotcha.** `sbatch` array tasks count against your
account's `MaxSubmitJobs` quota the instant they're submitted, regardless of
`--dependency` (a dependency only delays *running*, not *counting*). This bit
the simulation study (`examples/simulation_study/generate_slurm_dardel.py`,
see its module docstring) with two chained 720-task arrays exceeding a
1024-job quota. `common/slurm/submit_chain.sh` and each dataset's
`generate_slurm.py` keep this in mind: prefer one array with an internal
per-task step sequence over multiple dependent arrays, and check
`sacctmgr show assoc user=$USER format=MaxSubmitJobs` before submitting a
large batch (Morris's padj-derived cis-only sweep is the one to watch here).

## Adding a new dataset

Copy `template_dataset/` to `<new_name>/`, fill in `config.yaml`, and adapt
`generate_slurm.py` (the per-dataset quirks — which stages run, how genes
share an `ntc` fit, any custom modality loading or sum-factor computation —
live here; everything generic lives in `common/`).
