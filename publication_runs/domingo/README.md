# Domingo dataset

Low-MOI, 4 cis genes (GFI1B, MYB, NFE2, TET2), primary (gene) modality plus
splicing/velocity/efficiency modalities. Runs entirely on CPU
(`bayesdream_cpu` env, `-p shared`) — no GPU stage anywhere in this dataset.

**Before submitting a real run, see `../VERIFICATION.md`** for the general
checklist, plus the Domingo-specific risk items it lists (unconfirmed
modality directories).

## Pipeline

```
                 ┌─────────────┐
                 │ ntc_shared  │  fit_ntc() once, all genes, cis_gene deferred
                 └──────┬──────┘
        ┌────────┬──────┼──────┬────────┐
        ▼        ▼      ▼      ▼        ▼
     cis(GFI1B) cis(MYB) cis(NFE2) cis(TET2)   -- add_cis_gene() per gene, reusing ntc_shared
        │
        ├── compensation(GFI1B)        check_systematic_shift(), raw sum_factor
        ├── trans(GFI1B)               fit_trans(additive_hill), sum_factor_refit
        │     ├── permutation reps ×20  (array, auto-resubmits on timeout)
        │     └── recapitulation reps ×10 (array, auto-resubmits on timeout)
        └── modality(GFI1B, sj/exon_skip/intron_retention/mxe/gene_velocity/
              donor_efficiency/acceptor_efficiency/donor_choice/acceptor_choice)
              (one job per modality: own fit_ntc() + fit_trans(),
               additive_hill except donor/acceptor_choice: single_hill)
```

Same for MYB, NFE2, TET2. `generate_slurm.py` writes one sbatch script per
box above (57 scripts total for 4 genes × 9 modalities + the shared/per-gene
stages) plus `submit_all.sh`, which chains them with `--dependency=afterok`
per the diagram (compensation/trans/modalities all depend on that gene's cis
job; permutation/recapitulation depend on trans) and writes
`submitted_jobs.tsv` for `common/slurm/list_job_status.py`.

## Before running

1. **Data paths.** `config.yaml`'s `paths.meta`/`paths.counts` point at
   `cell_meta.csv`/`gene_counts.csv` under `processed_Leah/` — these are
   written by the R preprocessing script below via relative `fwrite()`
   calls, so confirm the exact output location once it's been (re-)run;
   the filenames themselves should be stable.
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

**`data_dir` is ONE shared directory, not per-gene.** Every cis gene's job
reads from the SAME `<stype>/` directories (covering the full cell
population); per-gene subsetting happens inside the loader by aligning
against that gene's own `model.meta['L_cell_barcode']` (Domingo's
preprocessed meta sets `cell == L_cell_barcode`, so this lines up with
`add_custom_modality()`'s own internal alignment against `model.meta['cell']`
— see `load_modalities.py`'s module docstring).

## R preprocessing (runs once, upstream, NOT part of this SLURM pipeline)

`cell_meta.csv`/`gene_counts.csv` are produced by an R script (outside this
repo) that, starting from `domingo_sce.rds`:

- drops the `CRISPRi`/`CRISPRa` pseudo-gene rows and renames `CCDC173` ->
  `CFAP210`;
- computes a **clustered sum factor** via `scran::calculateSumFactors(counts,
  clusters=guide_crispr, ref.clust='NTC')` (clusters = guide identity, NOT
  `quickCluster` -- different from Morris's per-cell-subset scran step, see
  `morris/README.md`) — this becomes the `sum_factor` column, computed ONCE
  for the whole dataset;
- also computes a guide-level **`adjustment_factor`** (ratio of each guide's
  mean clustered sum factor to its lane+cell_line NTC baseline) and an
  unused `clustered.sum.factor.adj` column;
- renames columns to bayesDREAM's expected `cell`/`guide`/`target`/
  `sum_factor`.

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
