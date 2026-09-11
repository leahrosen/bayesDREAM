# Replogle dataset

Low-MOI (single-guide), CRISPRi-only, transcriptome-wide trans, K562 cells.
7 cis genes (GFI1B, MYB, NFE2, TET2, IKZF1, HHEX, RUNX1 -- same panel as
`comparative/datasets.py`'s `REPLOGLE_GENE_TO_ID`), one shared `fit_ntc()`
across all of them (all NTC cells), then **3 fit_cis configurations per
gene**, each with its own matching `fit_trans()`. See `STRATEGY.md` in this
directory for the full design writeup: what the original ad hoc notebooks
(`tmp/06_bayesDREAM_fit_ntc_combined.ipynb`,
`tmp/10_bayesDREAM_fit_trans_MYB.ipynb`) do, how NTC cells are subset, what
`independent_mu_sigma` does, and every decision made while folding this
into `publication_runs/`.

**Scope**: preprocess -> ntc_shared -> (fit_cis x3, fit_trans x3) per gene.
Deliberately does **not** include compensation/permutation/recapitulation/
extra modalities -- none of those were part of what was asked for, or shown
in the reference notebooks. Add them later the same way Domingo/Morris did,
if wanted.

**Before submitting a real run, see `../VERIFICATION.md`** for the general
checklist.

## Two NTC-subsetting data variants + three fit_cis variants

- **`bm`**: replicates the ORIGINAL ad hoc pipeline's actual NTC subsetting
  (confirmed with the user, 2026-09-09) -- TWO restrictions compose (ANDed
  together for the NTC portion; every cis-gene-targeting cell is always
  kept regardless of either):
  - **guide selection**: only NTC cells carrying one of 5 curated NTC
    guides (`config.yaml`'s `bm_selected_ntc_guides`, given directly by the
    user -- this reproduces what the original pipeline's `NTC_subset/`
    input silently encoded; `common/subset_per_gene.py`'s new
    `select_ntc_guides` flag).
  - **batch match**: of THOSE cells, only ones from a `batch` where this
    gene was actually targeted -- ported from `load_gene_model_inputs()` in
    `tmp/10_bayesDREAM_fit_trans_MYB.ipynb` (see `STRATEGY.md` §2), via
    `common/subset_per_gene.py`'s `batch_match_ntc` flag. Replogle pools
    cells from many technical batches (`K562_combined`), but any one gene's
    guides only appear in a handful of them -- comparing against NTCs from
    unrelated batches would mix in batch-to-batch variation beyond what
    `sum_factor`/`alpha_y` correct for.
- **`all`**: every NTC cell, no restriction at all (`add_cis_gene()`'s own
  default; neither flag applied) -- used for `all_indmu` AND `ntc_shared`
  (see STRATEGY.md §10 for the correction that fixed both to use the
  genuinely full NTC population, not the pre-curated `NTC_subset/` input).

Three fit_cis configurations (`config.yaml`'s `cis_variants:`), each reusing
ONE of the two data variants above:

| variant      | data variant | `independent_mu_sigma` |
|--------------|-------------|-------------------------|
| `bm_indmu`   | `bm`        | `True`                  |
| `bm_noindmu` | `bm`        | `False`                 |
| `all_indmu`  | `all`       | `True`                  |

`independent_mu_sigma=True` fits separate `mu`/`sigma` hyperparameters for
the per-guide effect prior by `target` (NTC vs. cis-gene guides) instead of
pooling them into one shared hyperprior -- see `STRATEGY.md` §3.

Each of the 3 variants gets its own `fit_trans()`, reading that SAME
variant's own cis fit (cis and trans share one label per variant, so
`load_cis_fit()`'s default directory lines up) and that variant's matching
`full/` data subset.

## Pipeline

```
ntc_shared (fit_ntc, ALL NTC cells, cis_gene deferred,
            set_technical_groups(["batch","experiment"]))
   │
   ├── subset(gene, bm)  -> full/, cis_only/   (batch-matched NTC)
   │     ├── cis(gene, bm_indmu)   -> trans(gene, bm_indmu)
   │     └── cis(gene, bm_noindmu) -> trans(gene, bm_noindmu)
   │
   └── subset(gene, all) -> full/, cis_only/   (all NTC cells)
         └── cis(gene, all_indmu)  -> trans(gene, all_indmu)
```

× 7 genes. `generate_slurm.py` writes 1 (ntc_shared) + 7 × [2 subset jobs +
3 × (1 cis + 1 trans)] = 57 sbatch scripts, plus `submit_all.sh` (dependency-
chained: each gene's cis job depends on its own data-variant's subset job;
each trans job depends on its own cis job) and `submitted_jobs.tsv.template`
for `common/slurm/list_job_status.py`.

## Before running

1. **Preprocess once** (see `preprocess.py`'s own docstring for full
   detail). Reads the pre-built, **read-only**
   `for_bayesDREAM/K562_combined/` parquet tree (confirmed 2026-09-08: no
   write access there), reassembles [the FULL NTC population] ∪ [the 7
   chosen cis genes' true target cells], and recomputes `sum_factor`
   **once** on that combined population via `quickCluster`+
   `computeSumFactors` blocked by `batch` (Morris-style; NOT Domingo's
   guide-identity/`ref.clust='ntc'` style) -- discarding whatever
   `sum_factor` values already existed in the input tree, whose own
   computation scope isn't visible anywhere in this repo. Requires `rpy2` +
   R (`scran`/`Matrix`/`SingleCellExperiment`/`S4Vectors`), same as
   Morris's own preprocessing.

   **NTC source (corrected 2026-09-09):** reads `<indir>/NTC/` -- the SAME
   directory `tmp/06_bayesDREAM_fit_ntc_combined.ipynb` itself reads (85711
   cells, every NTC guide) -- NOT `NTC_subset/counts_subset.parquet`, which
   the user confirmed is already subsetted to a chosen set of NTC guides.
   This full NTC population feeds both `meta_ntc.csv`/`gene_counts_ntc.npz`
   (`ntc_shared`) and the combined `meta.csv`/`gene_counts.npz` (the
   `all_indmu` cis_variant, whose own `subset_per_gene.py` step never
   narrows the NTC pool further -- so both now correctly get the full
   population). Each `<gene_id>/counts.parquet` is also no longer trusted
   to hold only that gene's own cells -- its columns are filtered against
   `cell_meta_full`'s `target` column first, dropping any stray NTC (or
   other) cells before combining. See `STRATEGY.md` §10 for the full
   writeup and the synthetic-fixture test that verified this, and the
   "Two NTC-subsetting data variants" section above for the `bm` variant's
   own curated-guide-list restriction (`config.yaml`'s
   `bm_selected_ntc_guides`).

   **Small `batch` values**: any `batch` value with fewer than
   `--min-block-size` cells (default 100, matching `quickCluster`'s own
   `min.size`) is pooled into one bucket for scran's blocking variable
   only (real `batch` column elsewhere is untouched) -- needed on a real
   run, where one batch was too small and `quickCluster` raised `fewer
   cells than the minimum cluster size`. See `STRATEGY.md` §12. Requires
   `fastparquet` (not the default `pyarrow`, which has a confirmed bug
   against these exact files) -- install into `bayesdream_cpu` if missing.

   ```bash
   python preprocess.py \
     --indir /cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/pr_data/for_bayesDREAM/K562_combined \
     --outdir /cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Replogle2022/processed_Leah
   ```

2. Confirm `config.yaml`'s `paths.repo_dir`/`paths.python_env` (assumed
   same as Domingo/Morris -- not independently confirmed for Replogle).
3. Run `python generate_slurm.py`, inspect `slurm/`.
4. After `01_ntc_shared.sh` completes (before submitting the bulk of
   `cis`/`trans` jobs), sanity-check `technical_group_code` consistency --
   see "Technical group consistency" below -- then
   `bash slurm/submit_all.sh` for the rest.

## Known gaps / things NOT independently confirmed (see `STRATEGY.md`)

- **The actual `fit_cis` notebook was never found** anywhere in this repo
  (only its downstream consumer, `tmp/10_bayesDREAM_fit_trans_MYB.ipynb`,
  which loads an already-completed cis fit rather than fitting one). Its
  data-subsetting is assumed identical to the trans notebook's
  `load_gene_model_inputs()` -- confirmed correct by the user (2026-09-08),
  but its other hyperparameters (beyond `independent_mu_sigma`, which was
  explicitly requested) are not otherwise verified.
- `model_defaults.guide_covariates`/`guide_covariates_ntc` are `[]`
  (Replogle is single-cell-line K562 throughout, unlike Domingo's
  CRISPRa/CRISPRi split) -- a reasonable default, not confirmed against the
  missing fit_cis notebook.

## Technical group consistency

bayesDREAM's `set_technical_groups()` numbers groups via
`groupby(covariates).ngroup()`, which is only stable for the exact
dataframe it's run on -- re-running it on a smaller per-gene subset can
renumber groups and desync `technical_group_code` from the
`alpha_x_prefit`/`alpha_y_prefit` tensors fit under `ntc_shared`'s own
numbering (fixed in bayesDREAM as of 2026-09-09: `load_ntc_fit()` now
authoritatively re-derives `technical_group_code` by joining on actual
covariate values against a `technical_group_labels.csv` `ntc_shared`
writes, raising loudly rather than silently misapplying a correction if a
subset's covariate combination was never seen during the NTC fit -- see
`common/run_cis_deferred.py`, already applied here unmodified). Sanity-check
this once `01_ntc_shared.sh` and at least one gene's `01b_subset_*.sh` jobs
have completed, before submitting the rest:

```bash
python ../common/check_technical_groups.sh \
  <output_dir>/<label_prefix>_ntc_shared/meta.csv \
  <output_dir> \
  batch,experiment
```

(`<output_dir>` is `config.yaml`'s `paths.output_dir` = `.../BayesianModel_outs`;
the script globs `<output_dir>/*/cis_only/meta.csv`, which matches
Replogle's `<label_prefix>_<gene>_<bm|all>_subset/cis_only/meta.csv` layout
without modification.) A "MISMATCH ... groups in this subset but NOT in
ntc_shared" result would mean a gene's own batch has no NTC cells at all in
the full population -- a real data gap, not something either `bm`'s or
`all`'s NTC restriction could cause (both draw from `ntc_shared`'s own,
unrestricted coverage).

## Resources (confirmed 2026-09-08)

| stage | data variant | partition | time | cores | status |
|---|---|---|---|---|---|
| `ntc_shared` | (all NTC) | GPU (1 gpu) | 24h | 8 | fixed -- GPU jobs don't need profiling (see below) |
| `subset` (both variants) | full combined panel + `add_cis_gene()` | `main` (full node) | 4h | 128 (placeholder) | **measured 102GB -- see "Full-node CPU stages" below** |
| `cis` (all 3 variants) | `cis_only` (1 gene) + transient full-panel ntc load | `main` (full node) | 24h | 128 (placeholder) | **inferred from subset's measurement -- see below** |
| `trans` (`bm_indmu`/`bm_noindmu`) | `full`, batch-matched | CPU (`shared`) | 24h | 8 | **placeholder -- see profiling below** |
| `trans` (`all_indmu`) | `full`, all NTC | GPU (1 gpu) | 24h | 8 | fixed, not profiled (see rationale below) |

## Full-node CPU stages (`subset`/`cis`, `cluster.partition_main`)

Per-instruction (2026-09-11): any CPU stage whose real profiled peak memory
exceeds ~50% of a node gets a full `main`-partition node instead of a
fractional `--cpus-per-task` request on `shared` (which bills memory
strictly at 888MB/core -- a 100GB+ job would need a ~115-core request
there anyway, most of a node either way, just accounted per-core instead
of as one exclusive allocation).

**`subset` -- real measurement**, `common/profile_memory.py --stage cis
--ntc-shared-dir <ntc_shared output>` against a `bm`-variant
`*_subset_input.yaml` config (this reproduces `subset_per_gene.py`'s own
two most expensive steps exactly -- see `config.yaml`'s comment):
model construction on the full 86,806-cell combined panel (47.5s, 26GB),
then `load_ntc_fit()`+`add_cis_gene()` -- the dominant cost -- jumping to
**102,026 MB peak**.

**`cis` -- inferred, not yet directly measured**: the real `cis` stage pays
the identical `load_ntc_fit()+add_cis_gene()` call against the same
`ntc_shared` posterior; its own construction starts from the already-tiny
`cis_only/` file rather than the full panel, so its real peak is likely
somewhat below `subset`'s 102GB, but almost certainly still over half a
node. Re-profile once a real `01b_subset_<gene>_bm.sh` has actually run
(needed for `data.meta` in the `_cis.yaml` config to exist) and correct
`cis.use_full_node`/`config.yaml`'s comment if it comes in low enough to
fit on `shared` instead.

**`cluster.main_node_cores: 128` is a PLACEHOLDER** -- not independently
confirmed. Run `sinfo -p main -o "%c %m"` and correct it (and
`cluster.partition_main`/`main_node_sbatch_lines` if `main` isn't
literally the right partition name) before submitting either stage for
real.

**`ntc_shared` (and `trans`'s `all_indmu` GPU job) don't need memory
profiling.** Dardel's `888MB/core` rule (`publication_runs/README.md`'s
"Memory" section) is specific to the `shared` **CPU** partition -- a GPU
node's host RAM isn't scaled by `--cpus-per-task`, so `cores:` there just
needs to be "enough for data loading," not a memory proxy. This exact
`ntc_shared` fit (85711 cells x 8202 features) already ran successfully on
a real GPU in `tmp/06_bayesDREAM_fit_ntc_combined.ipynb`. Domingo/Morris's
own single-GPU `ntc_shared` jobs were never profiled either (Morris just
sets `cores: 16` with no "from real profiling" comment, unlike its CPU
stages). The one real unknown for any GPU job here is VRAM, which nothing
in this section checks either way -- `profile_memory.py` only measures CPU
RSS.

**Does `fit_cis` load the trans genes too?** No -- the actual SVI fit only
ever sees the cis gene's own single-feature counts (`cis_only/` subset, 1
row). But `run_cis_deferred.py`'s setup step
(`load_ntc_fit(ntc_shared_dir, mask_features=True)` then `add_cis_gene()`)
**transiently** loads the shared `ntc_shared` fit's FULL ~8195-feature
posterior from disk in order to extract just this gene's own alpha before
discarding the rest. So real peak memory for the `cis` stage is likely
dominated by that transient full-panel load, not by the tiny 1-gene fit
itself -- this is exactly the kind of thing that needs real profiling
rather than a guess, since "1 gene" undersells the actual memory shape.

**Profiling `subset`/`cis`/`trans` (`bm` variants)**: `common/profile_memory.py`
measures real peak RSS around a bare model construction + a cheap
(`--niters 10`) real fit call -- peak memory is set by tensor shapes, not
convergence, so a 10-iteration run already shows the real peak. `subset`
and `cis` are DIFFERENT memory-dominant steps, even though both involve
`add_cis_gene()` -- `subset_per_gene.py` constructs from the FULL combined
panel (all NTC + 7 genes, ~90k+ cells) and classifies it there; the real
`cis` stage constructs from the already-tiny `cis_only/` file `subset`
wrote, but then pays its OWN separate cost re-loading the shared
`ntc_shared` fit's full posterior (see "Does fit_cis load the trans genes
too?" above) -- profile both. Both `subset` and `cis` need a REAL,
already-completed `ntc_shared` run on disk first (deferred configs); `trans`
works standalone (eager config, though note it approximates the real job's
`load_ntc_fit()` step with a cheap in-process `fit_ntc()` instead -- see the
script's own docstring for why that's still shape-equivalent). Run once
`generate_slurm.py` has produced real rendered configs and `ntc_shared` has
actually completed:

```bash
# subset step -- its own rendered config never carries ntc_shared_dir
# (subset_per_gene.py doesn't call load_ntc_fit() itself), so pass it
# explicitly to reproduce the SAME add_cis_gene() classification cost:
python ../common/profile_memory.py \
  --config slurm/configs/<label_prefix>_GFI1B_bm_subset_input.yaml \
  --stage cis --ntc-shared-dir <output_dir>/<label_prefix>_ntc_shared --niters 10

python ../common/profile_memory.py \
  --config slurm/configs/<label_prefix>_GFI1B_bm_indmu_cis.yaml \
  --stage cis --niters 10

python ../common/profile_memory.py \
  --config slurm/configs/<label_prefix>_GFI1B_bm_indmu_trans.yaml \
  --stage trans --niters 10
```

All 3 cis variants (and both `bm_*` trans variants) share the same data
shape (only `independent_mu_sigma`, a tiny flag, differs) -- one profiling
run per gene per stage should be representative of all of them; the `all_*`
variant's `subset`/cis stages read a different (larger, all-NTC) subset and
may be worth profiling separately if their real run's memory looks off.

**`trans`'s `all_indmu` variant is NOT profiled** -- fixed at one GPU node
(1 GPU, 8 cores, 24h) per gene per the user's explicit instruction, since
every `all_indmu` run has the identical full-~8195-gene panel shape
regardless of gene (unlike the `bm` variants and `cis`, whose peak memory
is worth confirming per the transient-full-panel-load reasoning above).

- `cluster.partition_gpu`/`gpu_single_sbatch_lines` and `paths.python_env_gpu`
  are NOT independently confirmed for Replogle -- mirrored from Domingo/
  Morris's own `bayesdream_rocm` env and `gpu`/`--gpus=1` conventions.
  Confirm via `sinfo -p gpu -o "%P %G %c %N"` before submitting.

## sum_factor / no refit_sumfactor

Unlike Domingo/Morris, this pipeline does **not** call `refit_sumfactor()`
at the trans stage -- `tmp/10_bayesDREAM_fit_trans_MYB.ipynb` calls
`fit_trans(sum_factor_col="sum_factor_adj", ...)` directly. `cis`/`trans`
both use `adjust_ntc_sum_factor(covariates=["batch"])` -> `sum_factor_adj`
(not `sum_factor_refit`).

## exclude_trans_genes / function_type

Trans genes with `log2(mu_ntc) < -4.0` are excluded before `fit_trans()`
(`model.exclude_trans_genes(min_log2_mu_ntc=-4.0)`, same convention as
Domingo/Morris). `function_type="single_hill"`, NOT `additive_hill`
(Domingo/Morris's default) -- confirmed from the actual run log in
`tmp/10_bayesDREAM_fit_trans_MYB.ipynb`.

## Feature identity (`model_defaults.feature_name_col`, 2026-09-10)

Replogle's counts are a sparse `.npz` with no row labels, so (per
bayesDREAM commit `24bc2f5`, "Unify feature identity resolution")
per-feature identity would otherwise be resolved from `gene_meta.csv`'s
column-priority cascade rather than an explicit choice.
`model_defaults.feature_name_col: gene_id` pins it explicitly to the
Ensembl ID, matching how every `cis_gene:`/`cis_gene_ids` config value is
already expressed. `preprocess.py`'s `gene_meta.csv` keeps the REAL gene
symbol in `gene_name` (no longer overwritten to equal `gene_id`, unlike an
earlier version of this pipeline) since the explicit override makes that
unnecessary. See `STRATEGY.md` §11 for the full writeup, including a
real (unrelated, pre-existing) bug this surfaced and fixed:
`base_cfg["data"]` never included `feature_meta` at all, which would have
made `01b_subset_<gene>_<variant>.sh`'s own model construction raise the
first time it ran for real (verified via a synthetic sparse-`.npz`
fixture, not caught by this pipeline's earlier dry-run-only testing). See
`domingo/README.md`/`morris/README.md`'s own "Feature identity" sections
for why Morris specifically needed the same fix for a real (not
defense-in-depth) reason.
