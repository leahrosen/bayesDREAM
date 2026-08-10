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

**Before your first real submission for a dataset, read `VERIFICATION.md`**
— a staged checklist (static checks → dry run → dataset-specific risk
checks → resource sizing → actual submit → output sanity checks) for
confirming the pipeline is actually correct, not just that it renders.

## Layout

```
publication_runs/
├── common/                          # shared across all datasets
│   ├── config_utils.py              # YAML load/deep-merge/render + model builder (see below)
│   ├── git_provenance.py            # commit/branch/dirty capture + optional stable tag
│   ├── profile_memory.py            # real peak-RSS measurement -> cores needed (see "Memory")
│   ├── resource_stats.py            # per-job time/memory tracking, resume-aware (see "Per-job resource stats")
│   ├── compute_scran_sum_factor.py  # rpy2 scran sum factors, per-cell-subset (Morris only)
│   ├── subset_per_gene.py           # precompute per-gene full/cis_only data subsets (see "Per-gene data subsetting")
│   ├── run_ntc.py                   # standalone shared/deferred fit_ntc stage (extended model builder)
│   ├── run_cis_deferred.py          # shared-ntc cis stage (add_cis_gene) -- both low-MOI and high-MOI
│   ├── run_compensation.py          # standalone: model.check_systematic_shift() (always raw sum_factor)
│   ├── run_trans.py                 # standalone: adjust/refit sum factor -> exclude_trans_genes -> fit_trans
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
├── README.md                         # this file
└── VERIFICATION.md                   # checklist: how to verify/test/run a dataset's pipeline
```

## Conventions

**Stages.** Every dataset run is composed of independently-runnable stages:
`ntc` -> `cis` -> `compensation` -> `trans` -> (`permutation`, `recapitulation`).
Some datasets skip stages or share one stage's output across many runs of
another (e.g. Domingo shares one `ntc` fit across all 4 cis genes; Morris's
~116 cis-only sweep genes share one `ntc` fit, but each of its 5 primary
genes -- which get the full pipeline -- gets its OWN `ntc` fit instead,
packed into one GPU job -- see `morris/README.md`'s "Two fit_ntc regimes"
section for why).

**Per-gene data subsetting.** `bayesDREAM.__init__` itself dominated
per-gene job cost when every stage independently loaded and classified the
FULL raw dataset just to subset it down to one gene's cells (real
profiling: 38.7s / 21GB for ONE gene's Morris compensation config). Both
datasets now pay this cost ONCE per gene via `common/subset_per_gene.py`
(builds the same deferred `cis_gene=None` + `add_cis_gene()` model
construction the `cis` stage itself uses, then writes the result to disk
instead of proceeding to `fit_cis()`), producing two on-disk subsets from
one classification pass: `full/` (whole trans panel, cis gene's row
included -- for stages that need the panel: `compensation`/`trans`/
`permutation`/`recapitulation`/modality stages) and `cis_only/` (just the
cis gene's row -- for the `cis` stage itself). Domingo additionally
precomputes a per-(gene, modality) subset of its splicing/velocity data
(`domingo/subset_modality_per_gene.py`) the same way, one level down — see
`domingo/README.md`'s "Per-(gene, modality) subsetting" section.

**Config layering.** Two YAML layers, deliberately kept separate:

1. **`<dataset>/config.yaml`** — the *orchestration* config: input/output
   paths, which cis genes to run, per-step core counts, permutation/
   simulation replicate counts, cluster account. Consumed by
   `<dataset>/generate_slurm.py`. Not understood by `bayesDREAM` itself.
2. **Per-gene/per-stage `bayesdream_config` YAML** — close to (but a
   superset of) the schema `bayesDREAM/cli.py` expects (`data:`/`model:`/
   `ntc:`/`cis:`/`trans:`/`report:` keys, as in
   `examples/gfi1b_cli_config.yaml`, plus `sum_factor:`/
   `exclude_trans_genes:`/`compensation:`/`permutation:`/`simulation:`
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

**Dynamic dataset-module hooks (`attach_modality`, `exclude_cells`).**
`run_compensation.py`, `run_permutation_null.py`, and `run_recapitulation_sim.py`
can call into a dataset-specific Python module at runtime via a config block
shaped `{module, function, kwargs}` (e.g. Morris's padj-based
`compensation_exclude_cells.py`, Domingo's
`load_modalities.attach_modality_precomputed` for modality-level
permutation/recapitulation -- reads that (gene, modality)'s already-
subsetted file rather than re-reading the shared raw splicing directory,
see domingo/README.md's "Modality loading" section). This only works if the
dataset's own directory (`domingo/`, `morris/`) is on `sys.path` in that
job's process — `generate_slurm.py` stamps `_dataset_dir` into every
rendered config for exactly this, and `config_utils.ensure_dataset_dir_on_syspath()`
reads it before the `importlib.import_module()` call. (This was previously
broken for Morris's compensation hook — the old sys.path candidates never
included `morris/`, so the import would have failed the first time it
actually ran; never caught because dry-run testing only exercises config
*generation*, not execution. Fixed now, and any new dynamic hook must go
through `ensure_dataset_dir_on_syspath()` too, not reinvent its own
candidate list.)

**sum_factor recomputation.** `sum_factor_adj`/`sum_factor_refit`/Morris's
`sum_factor_new` are never persisted by `save_cis_fit()`/`save_trans_fit()`
or restored by `load_cis_fit()` — every stage after `fit_cis` that needs one
recomputes it itself, in-process, from the same `sum_factor:` config block
(`config_utils.apply_sum_factor_adjustments`). `compensation` is the one
exception: `check_systematic_shift()` always uses the raw `sum_factor`
column, by design (both datasets' reference pipelines call it with no
override).

**exclude_trans_genes.** Per project convention, genes with
`log2(mu_ntc) < -4` are excluded from trans fitting in every dataset — wired
into `run_trans.py`/`run_permutation_null.py`/`run_recapitulation_sim.py` via
`model.exclude_trans_genes(min_log2_mu_ntc=...)`, bayesDREAM's own public
method for this (also supports excluding by explicit gene name or a
`feature_meta` query — see `bayesDREAM/model.py`'s docstring). An earlier
version of this pipeline had its own `common/exclude_low_ntc_genes.py`
reimplementing the same trim/subset logic by hand, from before bayesDREAM
had a public method for it — deleted now that the library has one.

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

`permutation`/`recapitulation` replicates MUST pass their own unique
`checkpoint_dir` to `fit_trans()` (their own per-rep `output_dir` —
`run_permutation_null.py`/`run_recapitulation_sim.py` do this automatically)
— `fit_trans()`'s own default (`<model.output_dir>/<model.label>`) is shared
by every replicate of a given (gene, modality) AND by that gene's own real
`trans` fit. Since permutation/recapitulation only change cell/feature
*values*, not the shapes `fit_trans()`'s checkpoint validation checks, a
replicate left on the default would silently resume from — and immediately
report as "complete" — the real fit's already-converged, non-permuted
parameters, corrupting the null distribution. See `resource_stats.py`'s
module docstring.

**Per-job resource stats (time/memory, "add on" across restarts).**
`common/resource_stats.py` (used by every `common/run_*.py` script and
`domingo/load_modalities.py`) writes a `<stage>_stats.json` next to each
stage's other output (`ntc_stats.json`, `cis_stats.json`,
`compensation_stats.json`, `trans_stats.json`, or `stats.json` inside each
permutation/recapitulation replicate's own directory) — wall-clock
(`elapsed_sec`), peak CPU RSS (`peak_rss_mb`), and peak GPU memory
(`peak_gpu_mb`, CUDA only), plus hostname/SLURM job+array-task ID for
cross-referencing `sacct`. Written incrementally (after each step), same
pattern as `examples/simulation_study/run_recovery_fit.py`'s
`fit_stats.json`, which it's factored out of. Two different "restart"
behaviors, matching which stages actually have internal checkpointing (see
above):
- `ntc`/`cis` (no internal checkpoint): a manual resubmit either finds a
  fully-completed prior attempt in `<stage>_stats.json` (skips re-fitting
  entirely — loads instead — and carries that step's recorded stats forward
  unchanged) or starts completely fresh; there's no partial progress to add
  onto, since a killed attempt's work was discarded, not built upon.
- `trans`/`permutation`/`recapitulation` (real internal checkpoint): the
  recorded `elapsed_sec` is read back from `fit_trans()`'s own
  `cumulative_elapsed_sec` field (written into every checkpoint, summed
  across every resume attempt) rather than timed by the stats script itself
  — a single process's own timer can't see time spent in earlier,
  separately-killed attempts, so this is the actual "add on when we
  restart" behavior. Peak memory across resumes is `max(prior, this
  attempt)`, never summed — separate process attempts were never running
  simultaneously, so summing would overstate true peak memory pressure at
  any single moment.

**GPU node packing.** When a step only needs a fraction of a Dardel GPU node,
don't request one node per task. Instead write a plain-text task list (one
CLI invocation per line) and submit a single `sbatch` that requests one node
and runs `common/slurm/run_node_queue.sh <tasklist> <concurrency>`, which
runs up to `<concurrency>` tasks at a time via a bash job-pool (`wait -n`,
requires bash >=4.3). `common/slurm/sbatch_blocks.py`'s `SbatchGpuNodeQueue`
implements the sbatch header for this (`gpu_sbatch_lines` for the whole-node
request, `auto_requeue_on_timeout` for the same signal-trap idiom as
`SbatchStep`/`SbatchArray` — see its docstring for the "requeues the whole
packed job, not just the unfinished tasks" caveat this implies). Domingo
runs on CPU everywhere except its two multinomial modalities (donor_choice/
acceptor_choice, which empirically need GPU) — 4 cis genes x 2 modalities =
8 tasks, packed into ONE job across ALL genes (exactly one Dardel node's 8
GPUs) — see `domingo/README.md`'s "GPU-packed multinomial modalities"
section. Morris packs its 5 `primary_genes`' fit_ntc/trans/permutation/
recapitulation jobs the same way (one submission each, instead of one per
gene) — see `morris/README.md`'s "GPU node packing" section; its ~116
sweep genes' shared `ntc_shared` and every gene's `fit_cis` stay unpacked
(one single-task job / CPU-only, respectively). Both datasets' packed
tasklists individually prefix each line
with `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` (round-robinned over
concurrency) so concurrent tasks land on distinct GPUs instead of all
defaulting to device 0 — a whole-node allocation exposes every GPU to every
process by default, and `run_node_queue.sh` runs tasks as separate
subshells, so this can't be a single whole-job export the way thread-pinning
is for `SbatchStep`/`SbatchArray`.

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
