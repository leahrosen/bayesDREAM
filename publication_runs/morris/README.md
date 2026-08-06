# Morris dataset

High-MOI, 5 primary cis genes (GFI1B, NFE2, IKZF1, HHEX, RUNX1) with the full
pipeline, plus a fit_cis-ONLY sweep over every OTHER gene with padj<0.05 in
`Morris_gRNA2target_stats.csv`, reusing the same shared `fit_ntc()`.
`fit_cis`/`compensation` always run on CPU (`bayesdream_cpu`, `-p shared`);
`ntc_shared`/`trans`/`permutation`/`recapitulation` run on GPU
(`bayesdream_rocm`, `-p gpu`).

## Before running

1. **Preprocess once.** `python preprocess.py --indir <raw dir> --outdir
   <raw dir>/preprocessed` -- turns the raw `.npy`/`.npz` files (gene-name
   mapping, guide_assignment alignment+transpose, `sum_factor=sizeFactor`)
   into the clean, positionally-aligned inputs `generate_slurm.py` expects.
   Run once, manually, NOT part of the SLURM pipeline (like Domingo's R
   preprocessing).
2. Fill in `config.yaml`'s `paths:` (repo/env paths, `raw_data_dir`,
   `stats_csv`), `ntc_shared.placeholder_cis_gene` (any real gene NOT in the
   padj<0.05 cis-gene list — see below), and `exclude_guides` is no longer a
   flat list you fill in (see "SNP exclusion" below).
3. Confirm `guide_covariates`/`sum_factor.batch_col`/`sum_factor.covariates`
   (defaulted to `[lane]` everywhere — the reference scripts were
   inconsistent between `lane` and `experiment` on the `set_technical_groups`
   call specifically; `lane` was picked for internal consistency with every
   OTHER sum-factor-related call, confirm/override).
4. Run `python generate_slurm.py`, inspect `slurm/`, then on Dardel:
   `bash slurm/submit_all.sh`.

## Read this before running: the high-MOI shared-fit_ntc workaround

The low-MOI "run fit_ntc once, defer cis_gene" pattern
(`add_cis_gene()`, see CLAUDE.md and `domingo/`) is **explicitly blocked**
for high-MOI data:

```
bayesDREAM/core.py:181-186
    "cis_gene must be provided at initialization time in high-MOI mode.
    Deferred cis-gene specification via add_cis_gene() is not supported
    for high-MOI models."
```

Since running `fit_ntc()` separately for 5 + ~hundreds of genes is exactly
what this dataset needs to avoid, `common/apply_shared_ntc_high_moi.py`
reproduces what `add_cis_gene()` does for low-MOI, using only the public
save/load API (no bayesDREAM source was modified -- see that module's
docstring for the full mechanism and why it doesn't need core-code
sign-off). In short:

1. `common/build_ntc_shared_high_moi.py` builds ONE model over the (near-)
   full gene panel, with `cis_gene` set to a placeholder gene not otherwise
   analysed (`config.yaml`'s `ntc_shared.placeholder_cis_gene`), runs
   `fit_ntc()` once, saves it.
2. For every other gene, `common/run_cis_high_moi_shared_ntc.py` builds a
   normal per-gene high-MOI model (`cis_gene=<that gene>`, required at
   construction) and calls `apply_shared_ntc()` on it *before* `fit_cis()`
   to seed `alpha_x_prefit` and the primary modality's `alpha_y_prefit` from
   the shared run, instead of calling `fit_ntc()` again.

**Validate this once before trusting the full sweep**: compare one of the 5
primary genes' `alpha_x_prefit`/fit_cis results (via this shared-NTC path)
against what you'd get fitting that gene completely standalone. They should
be close (not identical -- the shared run's NTC cell composition differs
slightly per-gene after `exclude_guides` filtering). If they diverge
substantially, this workaround's assumptions don't hold and `add_cis_gene()`
needs an actual upstream fix for high-MOI -- a bayesDREAM core change
needing sign-off first, not something to silently work around further.

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

## Compensation: padj-based exclude_cells

`check_systematic_shift()` is restricted to NTC cells + cells targeting the
cis gene via a guide with `padj<0.05` **for that gene** (from
`Morris_gRNA2target_stats.csv`), via `morris/compensation_exclude_cells.py`
(plugged in through `run_compensation.py`'s dynamic `exclude_cells`
resolver). This REPLACES an earlier exploratory version of the pipeline that
excluded cells whose only targeting guide was SNP-499/500 specifically (that
rule was GFI1B-specific and isn't part of the production pipeline).

## sum_factor: scran is per-cell-subset, not shared

Unlike `alpha_x_prefit`/`alpha_y_prefit` (shared via the workaround above),
Morris's sum factor is recomputed **separately for every gene**, on that
gene's OWN NTC+target cell subset — this is genuinely per-subset, not
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

## exclude_low_ntc_genes

Every trans-derived stage (trans/permutation/recapitulation) drops genes
with `log2(mu_ntc) < -4` before fitting, via `common/exclude_low_ntc_genes.py`
(config.yaml's `trans.exclude_low_ntc_genes_threshold`) -- same as Domingo.
