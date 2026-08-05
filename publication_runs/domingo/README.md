# Domingo dataset

Low-MOI, 4 cis genes (GFI1B, MYB, NFE2, TET2), primary (gene) modality plus
splicing/velocity/efficiency modalities.

## Pipeline

```
                 ┌─────────────┐
                 │ ntc_shared  │  fit_ntc() once, all genes, cis_gene deferred
                 └──────┬──────┘
        ┌────────┬──────┼──────┬────────┐
        ▼        ▼      ▼      ▼        ▼
     cis(GFI1B) cis(MYB) cis(NFE2) cis(TET2)   -- add_cis_gene() per gene, reusing ntc_shared
        │
        ├── compensation(GFI1B)        check_systematic_shift()
        ├── trans(GFI1B)               fit_trans(additive_hill)
        │     ├── permutation reps ×20  (array)
        │     └── recapitulation reps ×10 (array)
        └── modality(GFI1B, sj/ir/es/mxe/velocity/donor_*/acceptor_*)
              (one job per modality, additive_hill except *_usage: single_hill)
```

Same for MYB, NFE2, TET2. `generate_slurm.py` writes one sbatch script per
box above (57 scripts total for 4 genes × 9 modalities + the shared/per-gene
stages) plus `submit_all.sh`, which chains them with `--dependency=afterok`
per the diagram (compensation/trans/modalities all depend on that gene's cis
job; permutation/recapitulation depend on trans).

## Before running

1. Fill in `config.yaml`'s `paths:` (cluster-absolute) and `cluster:`
   (account, GPU partition if you end up needing one, MaxSubmitJobs) blocks.
2. Plug your real splicing/velocity/efficiency loader into
   `load_modalities.py`'s `_load_one_modality` (currently raises
   `NotImplementedError` -- see that file's docstring for the contract).
3. Run `python generate_slurm.py`, inspect `slurm/`, then on Dardel:
   `bash slurm/submit_all.sh`.

## Why fit_ntc is shared but cis/trans aren't

`fit_ntc()` estimates per-gene overdispersion from NTC cells only, which
doesn't depend on which gene is "the" cis gene -- so one run covers all 4.
`add_cis_gene()` (called by `common/run_cis_deferred.py`) then extracts each
gene's own alpha from that shared fit without re-running fit_ntc. See
CLAUDE.md's "Deferred Cis-Gene Workflow" and
`common/run_cis_deferred.py`'s docstring.

## sum_factor_adj / sum_factor_refit

Every stage past `fit_cis` (compensation, trans, permutation,
recapitulation) recomputes `adjust_ntc_sum_factor()` and `refit_sumfactor()`
itself, in-process, from the same `sum_factor` config block -- these columns
are NOT persisted by save/load. See
`common/config_utils.apply_sum_factor_adjustments`'s docstring if this looks
redundant; it isn't.
