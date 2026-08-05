# Morris dataset

High-MOI, 5 primary cis genes (GFI1B, NFE2, IKZF1, HHEX, RUNX1) with the full
pipeline, plus a fit_cis-ONLY sweep over ~hundreds more genes reusing the
same shared `fit_ntc()`.

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
   analysed (`config.yaml`'s `ntc_shared.placeholder_cis_gene` -- fill this
   in), runs `fit_ntc()` once, saves it.
2. For every other gene, `common/run_cis_high_moi_shared_ntc.py` builds a
   normal per-gene high-MOI model (`cis_gene=<that gene>`, required at
   construction) and calls `apply_shared_ntc()` on it *before* `fit_cis()`
   to seed `alpha_x_prefit` and the primary modality's `alpha_y_prefit` from
   the shared run, instead of calling `fit_ntc()` again.

**Validate this once before trusting the full sweep**: run one of the 5
primary genes through this shared-NTC path (which `generate_slurm.py`
already does for all 5 -- that's `02_cis_<gene>.sh`) and compare its
`alpha_x_prefit`/fit_cis results against what you'd get fitting that gene
completely standalone (its own dedicated `fit_ntc()` + `fit_cis()`, e.g. via
the plain low-MOI-style deferred workflow adapted for one gene, or just a
manual one-off run). They should be close (not identical -- the shared run's
NTC cell composition differs slightly per-gene after `exclude_guides`
filtering.) If they diverge substantially, this workaround's assumptions
don't hold for this dataset and `add_cis_gene()` needs an actual upstream
fix for high-MOI instead of a userland workaround -- that would be a
bayesDREAM core change and needs sign-off first.

## Pipeline

```
                    ┌───────────────┐
                    │  ntc_shared   │  fit_ntc() once, full panel minus placeholder_cis_gene
                    └───────┬───────┘
        ┌──────┬──────┬─────┼─────┬──────┐              ┌─────────────────────────┐
        ▼      ▼      ▼     ▼     ▼      │              │   cis_sweep (~hundreds)  │
     cis(GFI1B)...cis(RUNX1)              └─────────────▶│  fit_cis ONLY per gene,  │
        │                                                │  apply_shared_ntc() each │
        ├── compensation                                 │  (array job or GPU node  │
        ├── trans                                        │   queue, see config.yaml)│
        │     ├── permutation ×20 (array)                └─────────────────────────┘
        │     └── recapitulation ×10 (array)
```

## Before running

1. Fill in `config.yaml`'s `paths:`, `cluster:`, `exclude_guides`,
   `ntc_shared.placeholder_cis_gene`.
2. Fill in `genes_all.csv` with the real ~hundreds-of-genes list (excluding
   `primary_genes` and `placeholder_cis_gene`).
3. Decide `cis_sweep.use_gpu_node_queue` (default `false`, one CPU array job
   on the `shared` partition -- consistent with this repo's own benchmark
   findings that these fits aren't GPU-bound, see
   `examples/simulation_study/generate_slurm_dardel.py`'s docstring). Only
   flip it if you've established this sweep actually needs a GPU.
4. Run `python generate_slurm.py`, inspect `slurm/`, then on Dardel:
   `bash slurm/submit_all.sh`.

## sum_factor_adj / sum_factor_refit

Same note as `domingo/README.md`: every stage past `fit_cis` recomputes
`adjust_ntc_sum_factor()`/`refit_sumfactor()` itself (not persisted across
save/load) -- see `common/config_utils.apply_sum_factor_adjustments`.
