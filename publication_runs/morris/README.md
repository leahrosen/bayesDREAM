# Morris dataset

High-MOI. Two different regimes, chosen so every stage loads the smallest
data it can (see "Per-gene data subsetting" below):

- **5 primary cis genes** (GFI1B, NFE2, IKZF1, HHEX, RUNX1) get the FULL
  pipeline (fit_ntc -> cis -> compensation -> trans -> permutation ->
  recapitulation), and each gets its OWN `fit_ntc()` -- fit on just that
  gene's own NTC+cis-cells subset (full trans panel), not the whole
  dataset. No shared `ntc_shared` for these.
- **~116 sweep genes** (every OTHER gene with padj<0.05 in
  `Morris_gRNA2target_stats.csv`) get `fit_cis()` ONLY, and DO share one
  `fit_ntc()` (`01_ntc_shared.sh`, full dataset, `cis_gene` deferred) --
  with ~116 of them and no trans-modeling need, one shared fit is cheaper
  in aggregate than ~116 separate ones, unlike the primary genes' case.

`fit_cis`/`compensation` always run on CPU (`bayesdream_cpu`, `-p shared`);
`fit_ntc` (primary genes' `01d_ntc_packed.sh` AND the sweep's
`01_ntc_shared.sh`)/`trans`/`permutation`/`recapitulation` run on GPU
(`bayesdream_rocm`, `-p gpu`). Primary genes' `fit_ntc`/`trans`/
`permutation`/`recapitulation` are each ONE packed GPU-node submission
covering all 5 primary genes (and, for permutation/recapitulation, all
reps) rather than one job per gene -- see "GPU node packing" below.

**Before submitting a real run, see `../VERIFICATION.md`** for the general
checklist, plus the Morris-specific risk items it lists (`rpy2`/R
availability, the known `fit_cis()` high-MOI dtype bug).

## Before running

1. **Preprocess once.** `python preprocess.py --indir <raw dir> --outdir
   <raw dir>/preprocessed` -- turns the raw `.npy`/`.npz` files (gene-name
   mapping, guide_assignment alignment+transpose, `sum_factor=sizeFactor`)
   into the clean, positionally-aligned inputs `generate_slurm.py` expects.
   Run once, manually, NOT part of the SLURM pipeline (like Domingo's R
   preprocessing).
2. Fill in `config.yaml`'s `paths:` (repo/env paths, `raw_data_dir`,
   `stats_csv`) and `global_exclude_guides` if any (distinct from the
   per-cis-gene SNP exclusion table, which is computed automatically -- see
   below).
3. Confirm `guide_covariates`/`sum_factor.batch_col`/`sum_factor.covariates`
   (defaulted to `[lane]` everywhere — the reference scripts were
   inconsistent between `lane` and `experiment` on the `set_technical_groups`
   call specifically; `lane` was picked for internal consistency with every
   OTHER sum-factor-related call, confirm/override).
4. Run `python generate_slurm.py`, inspect `slurm/`, then on Dardel:
   `bash slurm/submit_all.sh`.

## Per-gene data subsetting

Real profiling found `bayesDREAM.__init__` itself dominating per-gene job
cost (38.7s / 21GB for ONE gene's compensation config -- full
31468-gene x 52852-cell load + high-MOI classification against 1871 guides,
THEN discarding ~40k of those cells). `common/subset_per_gene.py` pays this
cost ONCE per gene (builds the SAME deferred+`add_cis_gene()` model
construction `fit_cis` itself uses, then writes the result to disk instead
of proceeding to `fit_cis()`) and writes BOTH subsetting modes in one pass
(`add_cis_gene()` already separates `cis` from the trans panel internally,
so both are in memory regardless of which mode(s) are requested):

- `full/` (primary genes' `01b_subset_<gene>.sh`): entire trans-gene panel,
  cis gene's row included -- used by that gene's own `fit_ntc`,
  compensation, trans, permutation, recapitulation (all of which need the
  whole panel).
- `cis_only/` (both primary genes' `01b_subset_<gene>.sh` AND sweep genes'
  `01c_subset_sweep.sh`): ONLY the cis gene's row -- used by
  `02_cis_<gene>.sh` and `07_cis_sweep.sh`, neither of which touch the trans
  panel at all. Both use the DEFERRED `add_cis_gene()` pattern, which
  tolerates a 1-gene starting panel with zero code changes (confirmed by
  direct test -- no equivalent to the eager-construction path's "No genes
  left after filtering!" raise), so bayesDREAM's `cis_only=True` flag isn't
  needed here.

`subset_per_gene.py` doesn't touch `ntc_shared`/`load_ntc_fit()` at all --
it never needed it for the subsetting itself, and since primary genes no
longer have a shared ntc to load from, requiring one here would be actively
wrong.

## Two fit_ntc regimes (high-MOI)

Both reuse bayesDREAM's native deferred-`cis_gene` support for high-MOI
models -- `cis_gene` can be omitted at construction and committed to later
via `add_cis_gene()`, exactly like the low-MOI workflow Domingo uses (see
CLAUDE.md's "Deferred Cis-Gene Workflow" and `common/run_cis_deferred.py`,
which now serves BOTH datasets). This used to require a userland workaround
in this pipeline (a placeholder cis gene + manual alpha extraction via
`torch.load`/`set_alpha_x`) because `add_cis_gene()` originally raised
`ValueError` for high-MOI; that workaround (`common/
apply_shared_ntc_high_moi.py`/`build_ntc_shared_high_moi.py`/
`run_cis_high_moi_shared_ntc.py`) is deleted now that the library supports
this directly.

**Primary genes**: each gets its OWN `fit_ntc()`, fit on that gene's own
`full/` subset (see "Per-gene data subsetting" above) -- packed into ONE
GPU job, `01d_ntc_packed.sh` (see "GPU node packing" below). No cross-gene
sharing; `add_cis_gene()`'s alpha-extraction machinery isn't even needed for
these, since each gene's `fit_ntc()` already only ever saw that one gene's
panel. `fit_cis`/`compensation`/`trans`/etc. all load THAT gene's own saved
ntc fit from `<output_dir>/<label>/` (same directory every one of that
gene's stages writes to -- `gene_output_dir(label)` in `generate_slurm.py`).

**Sweep genes**: share ONE `fit_ntc()` (`01_ntc_shared.sh`, full raw
dataset, `cis_gene` deferred) across all ~116 of them -- `add_cis_gene()`
then extracts each gene's own alpha from that shared fit without re-running
`fit_ntc()`, exactly as Domingo does for its 4 cis genes.

**KNOWN PRE-EXISTING ISSUE** (unrelated to any of the above, not something
this pipeline can work around): `fit_cis()` in high-MOI mode currently
raises `RuntimeError: expected scalar type Float but found Double` from a
dtype mismatch in `bayesDREAM/fitting/cis.py`'s `_model_x`. Reproduces
identically whether `cis_gene` is set eagerly at construction or via
`add_cis_gene()` -- it's a bug in that fitting code, not a config/pipeline
issue. See `CLAUDE.md`/`docs/HIGH_MOI_GUIDE.md`. If every `02_cis_<gene>.sh`
job in this pipeline fails with that exact error, this is why -- it needs an
upstream bayesDREAM fix, not a change here.

## Cis-gene list and SNP exclusion

**Cis-gene list**: `generate_slurm.py` reads `Morris_gRNA2target_stats.csv`,
takes `df[df['padj']<0.05]['gene_use'].unique()` (Ensembl IDs), maps them to
gene NAMES via `feature_meta`'s `gene_id`<->`gene_name` (high-MOI cell
classification matches `cis_gene` against `guide_targets_dict` by NAME, not
Ensembl ID -- passing an ID would silently break cell classification), and
splits into `primary_genes` (full pipeline) vs. everyone else (the cis-only
sweep, `07_cis_sweep.sh` -- one CPU array job, single submission regardless
of sweep size).

**SNP exclusion** (`morris/snp_exclusion.py`): 12 specific SNP-targeting
guides cause extensive trans effects and must be excluded from every cis-gene
run EXCEPT the one gene they themselves target (e.g. SNP-63/498/76 target
GFI1B's two CREs -- excluded everywhere except when `cis_gene='GFI1B'`).
This is a fixed 12-row table (not derivable from the stats CSV), computed
**per cis-gene** at `generate_slurm.py` time (not at runtime) and baked into
each gene's own `model.exclude_guides`. Since this is high-MOI, cells with
guides targeting OTHER genes are otherwise kept (that's the point of
high-MOI modelling) -- this exclusion is deliberately narrow, not a blanket
"exclude any non-cis-gene-targeting guide" rule. Matching is by substring
(`"SNP-63" in guide_name`), matching both reference scripts' own
`.str.contains()` convention -- see `snp_exclusion.py`'s docstring for the
one collision edge case this implies.

**`ntc_shared` gets the FULL table, unconditionally** (`exclude_all_snp_guides()`,
no per-gene exception), not the per-gene subset above. This matters because
`ntc_shared` defers `cis_gene`, and `fit_ntc()` defaults `use_all_cells=True`
whenever high-MOI + `cis_gene` is unset (see `bayesDREAM/fitting/ntc.py`) --
so without this, cells carrying these "extensive trans effects" guides would
have been included, completely unfiltered, in the ONE shared
`alpha_y_prefit` every primary and sweep gene's `add_cis_gene()` extracts
from. This asymmetry (broader exclusion at `ntc_shared` than at any
per-gene stage) is safe: `add_cis_gene()`'s alpha extraction is indexed by
FEATURE (gene), not by cell, so it doesn't create the cell-alignment
mismatch the "identical exclude_guides per gene" rule below is about.

**Consistency requirement**: `exclude_guides` is a bayesDREAM CONSTRUCTOR-time
filter, and every stage that constructs its OWN fresh model for a given
PRIMARY gene (`fit_ntc`, `cis` via deferred+`add_cis_gene()`, and
`compensation`/`trans`/`permutation`/`recapitulation` via `cis_gene` set
directly at construction) MUST use the identical per-gene `exclude_guides`
list, or their cell/guide composition diverges and `load_cis_fit()`/
`load_trans_fit()` would be aligning against a mismatched cell set.
`generate_slurm.py`'s `per_gene_exclude_guides()` computes each gene's
`exclude_guides` (global list + that gene's SNP exclusions) exactly once per
gene and threads it through every one of that gene's rendered configs --
including `01b_subset_<gene>.sh`'s own config and now `fit_ntc` too, since
that's just another per-gene stage reading the SAME `full/` subset the
subsetting step built with this exact list -- don't add a new per-gene
stage without doing the same. This does NOT apply to the sweep genes'
SHARED `01_ntc_shared.sh`, which deliberately uses a BROADER exclusion
(`global_exclude_guides` + the full SNP table, unconditionally --
`exclude_all_snp_guides()`) than any per-gene stage; see the paragraph
above for why that asymmetry is safe.

## Compensation: padj-based exclude_cells

`check_systematic_shift()` is restricted to NTC cells + cells targeting the
cis gene via a guide with `padj<0.05` **for that gene** (from
`Morris_gRNA2target_stats.csv`), via `morris/compensation_exclude_cells.py`
(plugged in through `run_compensation.py`'s dynamic `exclude_cells`
resolver). This REPLACES an earlier exploratory version of the pipeline that
excluded cells whose only targeting guide was SNP-499/500 specifically (that
rule was GFI1B-specific and isn't part of the production pipeline).

## sum_factor: scran is per-cell-subset, not shared

Unlike `alpha_x_prefit`/`alpha_y_prefit` (shared via `add_cis_gene()`,
above), Morris's sum factor is recomputed **separately for every gene**, on
that gene's OWN NTC+target cell subset — this is genuinely per-subset, not
something the shared `ntc_shared` run's own sum factor can stand in for.
`common/compute_scran_sum_factor.py` wraps the exact rpy2
(`quickCluster`+`computeSumFactors`, batched by `lane`) block from the
reference scripts; `config_utils.apply_sum_factor_adjustments`'s
`compute_scran` step calls it wherever needed (`cis`, `trans`,
`permutation`, `recapitulation` — NOT `compensation`, which always uses the
raw `sum_factor` column). Requires `rpy2` + R (`scran`/`Matrix`/
`SingleCellExperiment`/`S4Vectors`) available in whichever conda env runs
these stages — confirm this is installed in `bayesdream_cpu`/
`bayesdream_rocm` before relying on it.

`refit_sumfactor()` is called (creates `sum_factor_refit`) but its output is
**never actually used** -- `fit_trans()`/`fit_cis()` use `sum_factor_adj`
throughout, matching the reference GFI1B script exactly. `generate_slurm.py`
disables `refit_sumfactor` outright rather than compute a column nothing
reads.

## exclude_trans_genes

Every trans-derived stage (trans/permutation/recapitulation) drops genes
with `log2(mu_ntc) < -4` before fitting, via bayesDREAM's own
`model.exclude_trans_genes(min_log2_mu_ntc=...)` (config.yaml's
`trans.exclude_trans_genes`) -- same as Domingo.

## GPU node packing (fit_ntc/trans/permutation/recapitulation)

A Dardel GPU node has 8 GPUs. Running one job per (primary gene [x rep])
for fit_ntc/trans/permutation/recapitulation would waste most of a node's
GPUs per job, so these four stages are each ONE `SbatchGpuNodeQueue`
submission (`common/slurm/sbatch_blocks.py`) instead of N per-gene
`SbatchStep`/`SbatchArray` submissions:

- **`01d_ntc_packed.sh`**: 5 tasks (one `fit_ntc()` per `primary_genes`
  entry, each on that gene's own `full/` subset), concurrency 5.
- **`04_trans_packed.sh`**: 5 tasks (one per `primary_genes` entry),
  concurrency 5.
- **`05_permutation_packed.sh`** / **`06_recapitulation_packed.sh`**: 5 x
  `trans.permutation.n_reps` / `trans.simulation.n_reps` tasks (5 with the
  current n_reps=1), concurrency `min(n_tasks, 8)`.

Each is a plain-text task list of literal shell commands (`configs/
04_trans_packed_tasklist.txt`, etc. -- one `run_trans.py`/
`run_permutation_null.py --rep N`/`run_recapitulation_sim.py --rep N`
invocation per line, generated inside the same per-gene loop that renders
each gene's config YAML) run via `common/slurm/run_node_queue.sh` inside
ONE whole-node `sbatch` allocation (`cluster.gpu_node_sbatch_lines` in
config.yaml, default `-N 1 --gpus=8`). Unlike SbatchStep/SbatchArray (whole-
job export), each tasklist line is individually prefixed with its own `env
...` (`generate_slurm.py`'s `_pinned()`) for TWO reasons -- `run_node_queue.sh`
runs concurrent tasks as separate subshells within the same job, so a
whole-job-level export wouldn't reach them individually:
1. **Thread pinning** (`OMP_NUM_THREADS=...` etc., sized from that stage's
   `resources.cores`) -- otherwise every concurrent task's BLAS/torch
   threadpool would try to claim the whole node's cores.
2. **GPU device assignment** (`HIP_VISIBLE_DEVICES=<i %% concurrency>` +
   `ROCR_VISIBLE_DEVICES=` as a fallback) -- a whole-node `-N 1 --gpus=8`
   allocation exposes all 8 GPUs to every process by default, so without
   this every concurrent task would default to the same device (GPU 0)
   instead of spreading across the node. Round-robins task index modulo
   concurrency, so this is a clean 1:1 mapping onto distinct GPUs for every
   stage here (trans concurrency=5, permutation/recapitulation
   concurrency=min(n_tasks, 8) -- never exceeds `GPUS_PER_NODE=8`).

**Not verified**: (a) that `concurrency * resources.cores` actually fits
within one real Dardel GPU node's core count -- confirm via `sinfo -p gpu -o
"%P %G %c %N"` and reduce a stage's `resources.cores` if it doesn't fit; (b)
that `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` is the env var Dardel's
actual ROCm build/driver honors for device isolation -- confirm with
`rocm-smi` inside two concurrently-launched tasks on a real GPU node before
trusting it.

`01d_ntc_packed.sh` is NOT packed into the SAME node allocation as
`04_trans_packed.sh`: even though both are 5-task packed GPU jobs over the
same 5 primary genes, `01d_ntc_packed.sh` only needs that gene's own `full/`
data subset (from `01b_subset_<gene>.sh`), while `trans` needs that gene's
saved `fit_cis` result -- and CPU-only `cis`/`compensation` sit on the
dependency chain between the two GPU stages, which would leave an expensive
reserved GPU node idle for however long those CPU stages take. So each is
its own `SbatchGpuNodeQueue` submission, chained via
`submit_all.sh`'s `--dependency=afterok`. `ntc_shared` (the SWEEP genes'
single shared fit) is a separate, unpacked single-task GPU job
(`01_ntc_shared.sh`, `--gpus=1`) -- it has nothing to do with the primary
genes' `01d_ntc_packed.sh`.

Packed jobs still auto-resubmit on timeout (`SbatchGpuNodeQueue`'s
`auto_requeue_on_timeout=True`, same `--signal=B:USR1@120` idiom as before),
but note the granularity changed: a timeout now requeues the WHOLE packed
job, re-running every task in its list, not just the unfinished ones (
`run_node_queue.sh` has no "skip if already done" logic). This is wasteful
but not incorrect -- every `run_*.py` stage here overwrites the same saved
output idempotently, and an individual re-run of an already-finished gene's
`fit_trans` still resumes quickly via its own internal Pyro-level
checkpoint. `submit_all.sh`'s dependency chain: `01d_ntc_packed.sh` depends
on ALL 5 primary genes' `01b_subset_<gene>.sh` jobs; each gene's
`02_cis_<gene>.sh` depends on `01d_ntc_packed.sh` + that gene's own subset
job; `04_trans_packed.sh` depends on ALL 5 primary genes' `02_cis_<gene>.sh`
jobs (`--dependency=afterok:$CIS_GFI1B:$CIS_NFE2:...`); `05`/`06` each
depend on just `$TRANS_PACKED` (one dependency instead of one per gene).
