# Domingo dataset

Low-MOI, 4 cis genes (GFI1B, MYB, NFE2, TET2), primary (gene) modality plus
splicing/velocity/efficiency modalities. Runs on CPU (`bayesdream_cpu` env,
`-p shared`) EXCEPT the two multinomial modalities (donor_choice/
acceptor_choice), which run on GPU (`bayesdream_rocm` env, `-p gpu`) and are
packed together across ALL 4 genes into ONE job -- see "GPU-packed
multinomial modalities" below.

**Before submitting a real run, see `../VERIFICATION.md`** for the general
checklist, plus the Domingo-specific risk items it lists (unconfirmed
modality directories).

## Pipeline

```
                 ┌─────────────┐
                 │ ntc_shared  │  fit_ntc() once, precomputed NTC-only file, cis_gene deferred
                 └──────┬──────┘
        ┌────────┬──────┼──────┬────────┐
        ▼        ▼      ▼      ▼        ▼
   subset(GFI1B) subset(MYB) subset(NFE2) subset(TET2)  -- one add_cis_gene() classification pass per
        │              │             │            │        gene, writes full/ (whole trans panel + cis
        ▼              ▼             ▼            ▼        row) and cis_only/ (just the cis row)
     cis(GFI1B) cis(MYB) cis(NFE2) cis(TET2)   -- add_cis_gene() on that gene's cis_only/ subset,
        │              │             │            │        reusing ntc_shared
        │              └─────────────┴────────────┴──── donor_choice/acceptor_choice (GPU, ALL 4 genes
        │                                                packed into ONE node-queue job -- 4 genes x 2
        │                                                modalities = 8 tasks, 8 GPUs/node)
        │
        ├── compensation(GFI1B)        check_systematic_shift(), raw sum_factor, full/ subset
        ├── trans(GFI1B)               fit_trans(additive_hill), sum_factor_refit, full/ subset
        │     ├── permutation reps ×5  (array, auto-resubmits on timeout)
        │     └── recapitulation reps ×5 (array, auto-resubmits on timeout)
        └── modality_subset(GFI1B, <mod>)  -- one per (gene, modality), full/ subset + raw splicing dir
              └── modality(GFI1B, sj/exon_skip/intron_retention/mxe/gene_velocity/
                    donor_efficiency/acceptor_efficiency)  -- BINOMIAL only, one CPU job each
                    (own fit_ntc() + fit_trans(additive_hill), reads the precomputed modality subset)
                    ├── modality permutation reps ×5   (array, auto-resubmit)
                    └── modality recapitulation reps ×5 (array, auto-resubmit)
```

Same for MYB, NFE2, TET2. `generate_slurm.py` writes one sbatch script per
box above (146 scripts total: 4 genes × [1 gene-data-subset job + 3
gene-level stages + 2 gene-level array stages + 9 per-modality subset jobs
+ 7 binomial modality-fit jobs + 7 binomial modalities × 2 more array
stages] + the shared ntc job + ONE packed multinomial-modality job covering
all 4 genes) plus `submit_all.sh`, which chains them with
`--dependency=afterok`
per the diagram (each gene's subset job depends only on `ntc_shared`, not
on that gene's own cis fit -- `01b_subset_<gene>.sh` never touches a fitted
posterior; cis depends on `ntc_shared` + that gene's subset job;
compensation/trans/modality-subset all depend on that gene's cis job (they
read the `full/` subset, but still need the cis fit result); each modality's
own fit job depends on BOTH that gene's cis job and its OWN modality-subset
job (`--dependency=afterok:$CIS_<gene>:$MODSUBSET_<gene>_<mod>`); gene-level
permutation/recapitulation depend on gene-level trans; modality-level
permutation/recapitulation depend on THAT modality's own fit job, not just
cis -- see "Modality-level permutation/recapitulation" below; the packed
multinomial job depends on ALL 4 genes' cis jobs AND all 4 genes' multinomial
modality-subset jobs) and writes `submitted_jobs.tsv` for
`common/slurm/list_job_status.py`.

**Per-gene data subsetting (`01b_subset_<gene>.sh`).** Every downstream
stage used to independently re-load and re-classify the FULL 20001-cell
dataset just to subset it down to ~4281 cells for one gene. Now paid ONCE
per gene by `common/subset_per_gene.py`, which builds the SAME deferred
`add_cis_gene()` model construction `02_cis_<gene>.sh` itself uses, then
writes BOTH resulting subsets (`add_cis_gene()` already separates `cis` from
the trans panel internally, so both are in memory regardless of which
mode(s) are requested) instead of proceeding to `fit_cis()`: `full/` (whole
trans panel with the cis gene's row put back in -- used by compensation/
trans/permutation/recapitulation/the per-(gene,modality) subsetting step,
all of which stay eager `cis_gene`-at-construction) and `cis_only/` (just
the cis gene's row -- used by `02_cis_<gene>.sh`, which keeps its deferred
`add_cis_gene()` mechanism unchanged). `ntc_shared` itself reads a separate
precomputed NTC-only file (also written by `preprocess.py`), since it never
fits on non-NTC cells anyway.

**Per-(gene, modality) subsetting (`07a_modality_subset_<gene>_<mod>.sh`).**
Same motivation one level down: every modality-fit/permutation/
recapitulation job independently re-read and re-aligned the FULL shared raw
splicing directory (`modalities.data_dir`) against its own cells. Now paid
ONCE per (gene, modality) by `domingo/subset_modality_per_gene.py`, which
calls the SAME `load_modalities.attach_modality()` the real jobs used to
call directly, from that gene's own precomputed `full/` subset (needs only
`01b_subset_<gene>.sh`, NOT the cis fit -- `attach_modality()` only touches
raw counts, never a fitted posterior), and writes the resulting
cell-aligned, already-denominator-computed modality to disk. See "Modality
loading" below for what the real fit job reads instead.

## GPU-packed multinomial modalities (donor_choice/acceptor_choice)

Unlike every other Domingo stage, these two run on GPU: `fit_ntc()`'s own
`niters` default is 2x higher for multinomial than negbinom/binomial (see
`bayesDREAM/fitting/ntc.py`), and these modalities empirically need GPU.
4 cis genes x 2 modalities = 8 tasks, which happens to be exactly one Dardel
GPU node's worth (8 GPUs/node) -- so `generate_slurm.py` packs all 8 into
ONE `SbatchGpuNodeQueue` job (`07_modality_multinomial_packed.sh`, `-N 1
--gpus=8`, concurrency=8) instead of 8 separate per-(gene, modality) jobs,
using the same `common/slurm/run_node_queue.sh` mechanism as
`morris/generate_slurm.py`'s packed trans/permutation/recapitulation jobs.
Each of the 8 tasklist lines is individually prefixed with its own thread
pins (`OMP_NUM_THREADS=...` etc., from `modalities.resources.cores`) and GPU
device assignment (`HIP_VISIBLE_DEVICES=<task index>` + `ROCR_VISIBLE_DEVICES=`
as a fallback for older ROCm builds) -- `run_node_queue.sh` runs concurrent
tasks as separate subshells, so a whole-job-level export wouldn't reach them
individually, and without per-task device assignment every concurrent task
would default to the same GPU instead of spreading across the node's 8.

There is no permutation/recapitulation for these two modalities (multinomial
is excluded from that everywhere in this pipeline, same as the per-gene
gene-level path), so the packed job's tasklist is just the 8 modality-fit
calls themselves -- the same `load_modalities.py` script the binomial CPU
jobs use, just on `python_env_gpu`/`device: cuda` instead of `python_env`/
`device: cpu`.

**Not verified**: (a) that `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` is
the env var Dardel's actual ROCm build/driver honors for device isolation --
confirm with `rocm-smi` inside two concurrently-launched tasks on a real GPU
node; (b) that `concurrency (8) * modalities.resources.cores` fits within
one real Dardel GPU node's CPU core count -- confirm via `sinfo -p gpu -o
"%P %G %c %N"`.

## Before running

1. **Preprocess once.** `python preprocess.py --indir <dir with
   domingo_cellmeta.txt.gz + domingo_GEXcounts.csv> --outdir <clean output
   dir>` — turns the raw exports into the `cell_meta.csv`/`gene_counts.csv`/
   `gene_meta.csv` `config.yaml`'s `paths.meta`/`paths.counts` expect, plus
   an NTC-only subset (`cell_meta_ntc.csv`/`gene_counts_ntc.npz`,
   `paths.meta_ntc`/`paths.counts_ntc`) that `01_ntc_shared.sh` reads
   instead of the full dataset. Run once, manually, NOT part of the SLURM
   pipeline (like Morris's `preprocess.py`). See "Preprocessing" below for
   what it does and why it's Python+rpy2 now rather than a separate R
   script.
2. Confirm `config.yaml`'s `modalities.data_dir` (one shared directory
   containing `sj/`, `exon_skip/`, ..., `donor_choice/` subdirectories —
   see `load_modalities.py`'s module docstring for the exact per-stype file
   layout) and `config_modalities.yaml`'s `acceptor_choice` entry (added by
   symmetry with `donor_choice`, not yet confirmed against a real
   `loader_inputs/acceptor_choice/` directory).
3. Run `python generate_slurm.py`, inspect `slurm/`, then on Dardel:
   `bash slurm/submit_all.sh`.

## Modality loading (`load_modalities.py`)

Based on real, working loader code (six confirmed modality types; two more
-- `donor_efficiency`/`acceptor_efficiency` -- added per explicit
confirmation that they're binomial, analogous to exon_skip/intron_retention/
mxe/gene_velocity, but not independently verified against real files).
Two loaders, dispatched by `distribution` in `config_modalities.yaml`:

- **`load_binomial_modality`** (sj, exon_skip, intron_retention, mxe,
  gene_velocity, donor_efficiency, acceptor_efficiency): counts +
  denominator, aligned to the gene's own model cells. Denominator comes from
  one of two places (`config_modalities.yaml`'s `denominator_mode`):
  - `file`: a separate `denominator.npz` in the same stype directory
    (exon_skip/intron_retention/mxe/gene_velocity/donor_efficiency/
    acceptor_efficiency).
  - `gene_expression`: the primary (+`cis`, since `add_cis_gene()` carves the
    cis gene OUT of the primary modality) modality's own gene counts,
    features filtered to genes present in the model, with a saved
    counts-vs-denominator diagnostic plot and optional violation clipping
    (`sj` only — SJ counts can occasionally exceed the gene-expression
    denominator).
- **`load_multinomial_modality`** (donor_choice, acceptor_choice): a flat
  2-D `counts.npz` reshaped into `(features, cells, categories)` using
  `row_start`/`row_end` columns in `feature_meta.tsv.gz`.

Each modality then gets its OWN `fit_ntc()` call (a genuinely separate
technical fit — NOT reused from the primary 'gene' modality's ntc fit)
before `fit_trans()`, both with the usual load-if-exists-else-fit-and-save
pattern.

**`data_dir` is ONE shared directory, not per-gene -- but only
`07a_modality_subset_<gene>_<mod>.sh` reads it directly now.** Every cis
gene's subsetting job reads from the SAME `<stype>/` directories (covering
the full cell population) via `attach_modality()`, aligning against that
gene's own `model.meta['L_cell_barcode']` (Domingo's preprocessed meta sets
`cell == L_cell_barcode`, so this lines up with `add_custom_modality()`'s
own internal alignment against `model.meta['cell']`), and writes the result
to `<output_dir>/<label>_modality_subset/<mod>/`. The real
`07_modality_<gene>_<mod>.sh` fit job (and, for binomial modalities, the
permutation/recapitulation `attach_modality:` config block -- see below)
then reads THAT precomputed directory via `attach_modality_precomputed()`
instead -- no re-alignment or (for `sj`) gene-expression-denominator
recomputation needed, both already done when the precomputed file was
written. See `load_modalities.py`'s module docstring for the exact
contract of each function.

## Modality-level permutation/recapitulation

Only for BINOMIAL modalities (sj, exon_skip, intron_retention, mxe,
gene_velocity, donor_efficiency, acceptor_efficiency) -- NOT the multinomial
donor_choice/acceptor_choice. `n_reps` reused from `config.yaml`'s
`modalities.trans.permutation.n_reps`/`modalities.trans.simulation.n_reps`.

Mechanically these are the same `common/run_permutation_null.py`/
`common/run_recapitulation_sim.py` scripts the gene-level jobs use, made
modality-aware via a `modality_name` parameter (previously hardcoded to the
primary modality) plus a new `attach_modality:` config block -- a dynamic
import/call of `load_modalities.attach_modality_precomputed()` (reads that
(gene, modality)'s precomputed subset written by
`07a_modality_subset_<gene>_<mod>.sh`, NOT the raw shared `data_dir` --
see "Modality loading" above), resolved at runtime via
`config_utils.ensure_dataset_dir_on_syspath()` (put `domingo/` on
`sys.path` using the `_dataset_dir` value `generate_slurm.py` stamps into
every rendered config). Since a custom modality only exists on a freshly
built model after `attach_modality_precomputed()` re-attaches it, this happens FIRST,
before `load_ntc_fit()`/`load_cis_fit()`. NOTE: a modality's own ntc fit is
NOT saved into `ntc_shared_dir` (that directory only ever holds the primary
'gene' modality's shared fit) -- it's saved into the gene's OWN
`output_dir/label` by that modality's own `07_modality_<gene>_<mod>.sh` job
(`load_modalities.py`'s `model.save_ntc_fit()` call, no explicit
`output_dir`). A single `load_ntc_fit(input_dir=ntc_shared_dir)` call would
therefore silently find nothing for the custom modality (`load_ntc_fit`
skips missing files rather than erroring) and `fit_trans()` would later fail
with "has not been fit with fit_ntc()". `config_utils.load_ntc_for_stage()`
(used by both scripts) handles this: one `load_ntc_fit()` call restricted to
the primary modality from `ntc_shared_dir`, plus a second one for just the
custom modality from the default directory (`output_dir/label`) when
`modality_name` isn't the primary modality. Binomial's `simulate_from_trans_summary()` call
passes `sim_denominator` (the modality's own aligned denominator) instead of
`sim_sum_factor` -- binomial has no sum-factor concept.

Each modality-level permutation/recapitulation array job depends on that
SAME modality's own `07_modality_<gene>_<modality>.sh` job (not just cis) --
it needs that job's saved ntc fit (`alpha_y_prefit` prior for `fit_trans()`)
and, for recapitulation, its saved `trans_feature_summary_<modality>.csv` as
ground truth.

## Preprocessing (`preprocess.py`, runs once, upstream, NOT part of this SLURM pipeline)

Ports what used to be a separate R script into `domingo/preprocess.py`
(Python + rpy2 for the one R-only step, `scran::calculateSumFactors`) so
the whole pipeline runs from one language/environment, matching Morris's
own `preprocess.py`. Reads two raw exports (confirmed via header/content
inspection on Dardel, NOT the original `domingo_sce.rds` — see the
script's own docstring for exactly what was checked):

- `domingo_cellmeta.txt.gz` — raw `colData(sce)` (comma-separated despite
  the `.txt` extension); has `lane` already derived but not
  `cell`/`target`/`guide`/`sum_factor`.
- `domingo_GEXcounts.csv` — raw `counts(sce)` (genes x cells),
  **unfiltered** (still has the `CRISPRi`/`CRISPRa` pseudo-gene rows and
  un-renamed `CCDC173`) — exactly what `calculateSumFactors()` needs, since
  the original script computes sum factors BEFORE dropping those rows.

`preprocess.py`:

- aligns `domingo_GEXcounts.csv`'s columns to `domingo_cellmeta.txt.gz`'s
  row order by real name-based lookup (its cell-barcode column headers are
  reliable, unlike Morris's `guide_assignment_cells.npy` — see
  `morris/README.md`'s SNP/alignment notes for that contrast), raising if
  any cell is missing;
- computes a **clustered sum factor** via `scran::calculateSumFactors(counts,
  clusters=guide_crispr [any name containing "NTC" collapsed to a single
  'NTC' cluster], ref.clust='NTC')` (clusters = guide identity, NOT
  `quickCluster` -- different from Morris's per-cell-subset scran step, see
  `morris/README.md`) — this becomes the `sum_factor` column, computed ONCE
  for the whole dataset, on the RAW (unfiltered) counts;
- drops the `CRISPRi`/`CRISPRa` pseudo-gene rows and renames `CCDC173` ->
  `CFAP210` in `gene_counts.csv`;
- builds `gene_meta.csv` from the GTF (`gencode.v44.annotation.gtf.gz`),
  filtered to genes present in the (filtered) counts;
- renames columns to bayesDREAM's expected `cell`/`guide`/`target`/
  `sum_factor`.

**Deliberately does NOT compute `adjustment_factor`/`clustered.sum.factor.adj`**
(the guide-level mean-sum-factor-vs-NTC-baseline normalization the original
R script also computed) — that column is always renamed to
`adjustment_factor_old` and never read
(`config_utils.apply_sum_factor_adjustments`'s docstring — it's provably
redundant with `model.adjust_ntc_sum_factor()`, which recomputes an
equivalent column itself), so `preprocess.py` skips reimplementing a step
with no downstream use rather than replicate it for parity's sake.

Requires `rpy2` + R's `scran` package in whichever conda env runs this
(confirm separately from Morris's own rpy2 needs, even though both use
`bayesdream_cpu` — as of 2026-08, `rpy2` was confirmed NOT installed there
yet).

**Why `apply_sum_factor_adjustments` renames `adjustment_factor` before
calling `adjust_ntc_sum_factor()`:** the R output above ALWAYS ships a
precomputed `adjustment_factor` column (from the step above), and
`adjust_ntc_sum_factor()` computes its own internal column of that exact
name — the two would collide. `config_utils.apply_sum_factor_adjustments`
renames the stale one to `adjustment_factor_old` (kept, not dropped)
automatically whenever it's present, before calling
`adjust_ntc_sum_factor()`, so this is handled for you rather than being
something each stage script has to remember.

## Why fit_ntc is shared but cis/trans aren't

`fit_ntc()` estimates per-gene overdispersion from NTC cells only, which
doesn't depend on which gene is "the" cis gene -- so one run covers all 4.
`add_cis_gene()` (called by `common/run_cis_deferred.py`) then extracts each
gene's own alpha from that shared fit without re-running fit_ntc. See
CLAUDE.md's "Deferred Cis-Gene Workflow" and
`common/run_cis_deferred.py`'s docstring.

## sum_factor_adj / sum_factor_refit

Every stage past `fit_cis` that needs one (trans, permutation,
recapitulation — NOT compensation, which always uses the raw `sum_factor`
column, matching the reference GFI1B script) recomputes
`adjust_ntc_sum_factor()`/`refit_sumfactor()` itself, in-process, from the
same `sum_factor:` config block -- neither survives a save/load round trip.
Both use the library-default `sum_factor_col_old='sum_factor'` (the
R-preprocessed, already-clustered column) — `refit_sumfactor()` does NOT
read from `sum_factor_adj`, matching the reference GFI1B script exactly
(`model.refit_sumfactor(covariates=[...])`, no `sum_factor_col_old`
override). See `common/config_utils.apply_sum_factor_adjustments`'s
docstring if this looks redundant; it isn't.

## exclude_trans_genes

Every trans-derived stage (trans/permutation/recapitulation) drops genes
with `log2(mu_ntc) < -4` before fitting, via bayesDREAM's own
`model.exclude_trans_genes(min_log2_mu_ntc=...)` (config.yaml's
`trans.exclude_trans_genes`).
