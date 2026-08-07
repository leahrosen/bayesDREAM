# Morris dataset

High-MOI, 5 primary cis genes (GFI1B, NFE2, IKZF1, HHEX, RUNX1) with the full
pipeline, plus a fit_cis-ONLY sweep over every OTHER gene with padj<0.05 in
`Morris_gRNA2target_stats.csv`, reusing the same shared `fit_ntc()`.
`fit_cis`/`compensation` always run on CPU (`bayesdream_cpu`, `-p shared`);
`ntc_shared`/`trans`/`permutation`/`recapitulation` run on GPU
(`bayesdream_rocm`, `-p gpu`).

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

## Shared fit_ntc across many cis genes (high-MOI)

This now uses bayesDREAM's native deferred-`cis_gene` support for high-MOI
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

**Consistency requirement**: `exclude_guides` is a bayesDREAM CONSTRUCTOR-time
filter, and every stage (`cis`, `compensation`, `trans`, `permutation`,
`recapitulation`) builds its OWN fresh model for a given gene (the `cis`
stage via deferred+`add_cis_gene()`, the rest via `cis_gene` set directly at
construction). All of them MUST use the identical per-gene `exclude_guides`
list, or their cell/guide composition diverges from what `fit_cis()` actually
saved, and `load_cis_fit()`/`load_trans_fit()` would be aligning against a
mismatched cell set. `generate_slurm.py` computes each gene's
`exclude_guides` (global list + that gene's SNP exclusions) exactly once per
gene and threads it through every one of that gene's rendered configs --
don't add a new per-gene stage without doing the same.

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
