# Single-Hill Recovery Simulation Study — Plan

## Purpose

Assess `fit_cis` / `fit_trans` parameter recovery and null/non-null discrimination
across a wide grid of cell counts, guide designs, cis expression levels, and trans
dose-response shapes, using fully known ground truth. Ground truth is generated as
single-Hill dose-response curves (or null); recovery is fit with
`fit_trans(function_type='additive_hill')` — a deliberately **misspecified**
(more flexible) model relative to the single-Hill generating process, to test
whether `additive_hill` still recovers sensible parameters (correct-direction
component matching truth, opposite-direction component shrinking toward zero)
rather than only testing a matched-model best case. See the 2026-07-29 update in
[§1](#1-decisions-already-made-resolved-with-user-2026-07-27).
Runs on Berzelius via `bayesDREAM.slurm_jobgen`.

This document is the design spec. It does not implement the simulator or the SLURM
scripts — see [Next Steps](#next-steps).

---

## 1. Decisions already made (resolved with user, 2026-07-27)

- **"log2 cis o_y"** in the original request was a typo for **`o_x`** — the cis gene's
  own NB overdispersion (`phi_x = 1/o_x^2`), not the trans-gene `o_y`. Confirmed.
- **"Fits: no effect or single hill"** describes the **ground-truth generative
  scenario** for each simulated trans gene (flat/null vs. a true single-Hill
  dose-response), not a choice among fitting model forms. Confirmed.
- **Replicates**: **5** independent-seed replicates per cell-design scenario.
- **Batching**: each cell-design scenario simulates **all trans-gene parameter
  combinations together as one synthetic gene panel** and fits them in a single
  `fit_cis` + `fit_trans` call (bayesDREAM natively vectorizes across trans features).
  This is the only computationally feasible structure — (one fit per individual
  trans-gene scenario) would be ~187,000 separate fits.

**Update, 2026-07-29 (resolved with user):** Recovery fitting uses
**`fit_trans(function_type='additive_hill')`**, not `single_hill` as originally
planned — this is now deliberately a **misspecified-model** study: the simulator is
unchanged (ground truth stays single-Hill/null only, see §4.2), but fitting uses the
more flexible `additive_hill` form to test whether it still recovers sensible
parameters (correct-direction Hill component tracking the true single-Hill curve,
opposite-direction component's amplitude shrinking toward ~0) rather than only
validating a matched-model best case. This changes total SVI steps per `fit_trans`
call: `additive_hill` runs a `single_hill` curriculum-warmup phase first
(`warmup_steps = round(niters * 0.5/0.9) ≈ 55,556` for the default `niters=100,000`),
then the `niters` `additive_hill` steps proper — **~155,556 total steps**, vs. the
flat 100,000 `single_hill` would have run. Per-step cost is also higher for the
`additive_hill` phase (two Hill components instead of one). The `--min_fit_hours=3`
default in §7 was calibrated against a `single_hill` timing run and has **not** been
re-measured for `additive_hill` — treat it as stale until a real `additive_hill`
timed run is available.

## 2. Parameters that were missing or under-specified, and how they're resolved here

These were not fully specified in the original request. Defaults are proposed below;
flag any of these you want changed before implementation.

| Gap | Resolution proposed here |
|---|---|
| **Units/space of "log2 X" quantities** | All `log2 ...` values in the request (cis expression, `o_x`, `y_ntc`, `o_y`) are treated as log2 of the natural-scale parameter, e.g. `log2 cis gene expression = 1` → `X_NTC = 2`. |
| **Sum factor units** | "Normal(mean 0, variance 0.5)" is applied in **log2 space**: `log2(sum_factor) ~ Normal(0, sqrt(0.5))`, `sum_factor = 2^that`. A sum factor drawn directly from `Normal(0, 0.5)` would be negative ~50% of the time, which is invalid as a multiplicative NB scale factor — every other quantity in the request is specified in log2 units, so this is the consistent reading. **Flag if you intended something else** (e.g. variance 0.5 on the linear scale, or a Gamma/log-normal with a different parameterization). |
| **Non-integer cells/guide** | `[100,500,1000] cells / (4 or 6 guides)` doesn't always divide evenly (e.g. 100/6 = 16.67). Rule: `floor(total/n_guides)` cells per targeting guide; the remainder is added to the NTC guide's cell count. This makes NTC slightly larger than target guides at 100 and 1000 cells with 6 guides (100/6 → 5 guides × 16 + NTC × 20; 1000/6 → 5×166 + NTC×170). Flag if you'd rather round differently or drop the remainder. |
| **Technical groups / batches** | "1 batch" is implemented as a single, constant `cell_line` value passed to `set_technical_groups(['cell_line'])`. `fit_ntc`/`fit_cis`/`fit_trans` all have an explicit `C == 1` code path (no group-effect sampling), so this is a supported, non-degenerate case — `alpha_x`/`alpha_y` are implicitly 1.0 for all cells. |
| **Random seeding mechanism** | The existing pipeline scripts (`run_technical.py` etc.) derive seeds via Python's builtin `hash(label + gene) % 2**32`. **This is not reproducible across separate process launches** unless `PYTHONHASHSEED` is fixed, because `hash()` on strings is salted per-process by default. This simulation study uses an explicit, saved integer-seed table instead (§6) — do not reuse the `hash()` pattern for this study. |
| **Low-count edge case (`y_ntc` = 2^-4 ≈ 0.06)** | At `cells_per_gene=100` (≤25 cells/guide) this mean is low enough that some genes may be all-zero across a guide group, or even across the whole scenario in the worst case. This looks like a deliberate stress test of the low-expression `A`-prior regime (see `docs/HILL_FUNCTION_PRIORS.md`, "NTC-anchored interpolation", and memory note `gfi1b-false-positives`). Flag as expected/intended rather than an error; the simulator should record per-feature total counts so all-zero features can be identified and excluded from evaluation rather than silently breaking the fit (bayesDREAM's `_refilter_zero_count_features` will already drop true all-zero features from the panel — the ground-truth table must be joined back by `feature` name, not by row position, to survive this). |
| **Evaluation metrics** | Not part of "structure and plan" as literally requested, but sketched in §8 so the ground-truth schema (§5) captures what's needed. Not implemented yet. |
| **`n_guides` group-count mismatch with `cells_per_gene`** | Guide count (3 or 5 target + 1 NTC) and the three effect-shape patterns (even/gap/small) are combined freely, i.e. all `2 × 3 = 6` combinations are run, not just "3-guide-even, 5-guide-gap" pairings. Flag if you intended a fixed pairing instead. |

**Found during implementation** (not anticipated at design time): the ground-truth
`effect_type` column originally used the literal string `'null'` for the no-effect
category. `pandas.read_csv`'s default `na_values` list includes the string `"null"`
(along with `"None"`, `"NA"`, `"NaN"`, etc.), so writing `'null'` to CSV and reading it
back silently turns it into `NaN` — confirmed by the local smoke test (§7 verification
below). Fixed by using `'no_effect'` instead. Worth remembering for any future string
columns in this study's CSVs.

**Nothing else appears to be missing** for a well-posed generative model — every term
in the observation model (`alpha_x`, `x_true`, `sum_factor`, `phi_x` for cis;
`alpha_y`, `y_pred`, `sum_factor`, `phi_y` for trans) is now pinned to a value or a
swept grid.

---

## 3. Parameter grid

### 3.1 Cell-design scenario (drives `fit_cis`; shared `x_true` for a whole trans panel)

| Parameter | Values | Count |
|---|---|---|
| `cells_per_gene` | 100, 500, 1000 | 3 |
| `n_guides` (targeting only; +1 NTC) | 3, 5 | 2 |
| guide effect shape | even, gap, small | 3 |
| `sigma_eff` (log2 within-guide SD) | 0.7 (fixed) | 1 |
| `log2(X_NTC)` (cis expression at NTC) | -1, 0, 1, 2 | 4 |
| `log2(o_x)` (cis overdispersion) | -1.5, 0 | 2 |

Guide log2FC patterns (relative to NTC, guide count implied by pattern):

| Guides | even | gap | small |
|---|---|---|---|
| 3 | [-3, -2, -1] | [-3, -2.5, -0.5] | [-1.5, -1, -0.5] |
| 5 | [-4, -3, -2, -1, 0] | [-4, -3.5, -3, -1, -0.5] | [-1.5, -1.25, -1, -0.75, -0.5] |

`n_guides × shape` = 6 guide-design combinations.

**Cell-design scenarios = 3 × 6 × 4 × 2 = 144**, each replicated 5× → **720 total
(`fit_ntc` + `fit_cis` + `fit_trans`) pipeline runs.**

### 3.2 Trans-gene scenario (one panel of features per cell-design scenario)

| Parameter | Values | Count |
|---|---|---|
| `log2(y_ntc)` | -4, -1, 1, 4 | 4 |
| `log2(o_y)` | -1.5, 0 (for y_ntc ∈ {-1,1,4}); -0.3, 2 (for y_ntc = -4) | 2 per y_ntc level |
| ground-truth response | null (no effect), or single_hill | — |
| `n` (Hill coefficient, signed) | ±0.5, ±1, ±5 | 6 |
| `K_log2FC` (EC50 offset from `X_NTC`) | -4,-3,-2,-1,0,1,2,3,4 | 9 |
| `full_log2FC` (amplitude) | 0.5, 1, 2, 4 | 4 |

Per `y_ntc × o_y` combo: 1 null + 6×9×4 = 217 single_hill combos → 217 trans-gene
scenarios. Total trans panel size: **8 × 217 = 1736 features per cell-design
scenario** (constant across all 144 scenarios — the trans grid doesn't depend on the
cis-side design).

---

## 4. Generative model

### 4.1 Cis gene (mirrors `_model_x`, see `docs/CIS_MODEL_PARAMETERS.md`)

For each cell `c` with guide `g`:

```
x_eff_g        = X_NTC * 2^(guide_log2FC[g])          # guide_log2FC[NTC] = 0
log2(x_true_c) ~ Normal(log2(x_eff_g), sigma_eff)       # sigma_eff = 0.7
x_true_c       = 2^log2(x_true_c)
log2(sf_c)     ~ Normal(0, sqrt(0.5))                   # per-cell sum factor
sf_c           = 2^log2(sf_c)
x_obs_c        ~ NegBinom(mu = x_true_c * sf_c, phi = 1/o_x^2)     # alpha_x = 1 (C=1)
```

`sf_c` is shared across the cis gene and every trans gene for that cell (it's a
library-size factor, not gene-specific).

### 4.2 Trans genes — reuse `simulate_from_trans_summary`

Rather than re-implementing the Hill/NB sampling, build a synthetic
`trans_summary_df` in the **fold-change parameterization** already supported by
`bayesDREAM.simulation.simulation.simulate_from_trans_summary` /
`_compute_AV_from_fc` and call it directly:

Required columns per feature row:
- `feature`: unique gene id, e.g. `f{y_ntc_idx}_{o_y_idx}_null` or
  `f{y_ntc_idx}_{o_y_idx}_n{n}_K{K_log2FC}_F{full_log2FC}`
- `function_type = 'single_hill'`
- `distribution = 'negbinom'`
- `o_y_median` = `o_y` for that row
- `y_ntc_median` = `y_ntc` for that row
- `x_ntc_median` = `X_NTC` (the *cis* NTC level for this cell-design scenario)
- `n_a_median` = `n` (0 for null rows)
- `K_log2FC_a_median` = `K_log2FC` (irrelevant for null rows, any finite value)
- `full_log2FC_a_median` = `full_log2FC` (0 for null rows)

Then:

```python
counts_df = simulate_from_trans_summary(
    trans_summary_df=panel_df,       # 1736 rows
    meta=meta,                       # cell, guide, target (+ technical_group_code, unused: C=1)
    x_true=x_true,                   # from §4.1
    x_counts=x_obs,                  # from §4.1
    cis_gene='CisGene',
    sim_sum_factor=sf,               # same per-cell sf as §4.1
    fdr_threshold=None,              # no gating — use raw fold-change params, not a fitted posterior
    seed=<scenario_seed>,
)
```

This handles the NB sampling, null-row detection (`n==0 or full_log2FC==0`), and
`A`/`V`/`K` reconstruction internally — it's the exact code path already used to
validate real `fit_trans` output, so simulated and fitted data share the same
semantics by construction. `_compute_AV_from_fc` docstring confirms: `K = x_ntc *
2^K_log2FC`, `A`/`V` derived so that `y(x_ntc) = y_ntc` exactly.

No `group_col` / `alpha_y` columns are included (C=1 ⇒ no technical correction).

### 4.3 Sum factor: simulated vs. recomputed

`sim_sum_factor` above (§4.1's `sf`, drawn independently per cell) is what
`simulate_from_trans_summary` uses to *generate* the counts — it is **not** what gets
fit against. `simulate_from_trans_summary`'s own docstring is explicit: "sum factors
used for simulation are NOT the same as sum factors for downstream fitting... After
simulation, recalculate sum factors from the simulated counts (e.g. via
`scran::calculateSumFactors`)". So after `counts.csv` is written,
`recompute_sum_factor_scran()` (`bayesDREAM/simulation/cis_panel_simulation.py`) shells
out to `Rscript` running `calculateSumFactors(counts, clusters=guide×cell_line,
ref.clust='NTC')` on the realized counts, and **that** value — not the true simulated
one — is what ends up in `meta.csv`'s `sum_factor` column (mirroring what a real
analyst actually has). The true simulated value is preserved as `sum_factor_true` in
`cis_ground_truth.csv`.

Because `sum_factor` is now estimated from *realized*, post-perturbation counts,
guide identity is genuinely correlated with it (strong cis perturbations shift a
measurable fraction of the 1737-feature library's composition) — so
`run_recovery_fit.py` follows the full documented workflow
(`docs/FIT_TRANS_GUIDE.md`): `fit_ntc` → `adjust_ntc_sum_factor` (→
`sum_factor_adj`) → `fit_cis` → `refit_sumfactor` (→ `sum_factor_refit`) →
`fit_trans`, rather than feeding raw `sum_factor` straight into `fit_cis`/`fit_trans`.

---

## 5. Ground truth to save (per cell-design scenario × replicate)

```
outdir/<label>/scenario_<sid>/rep_<r>/
├── config.json              # full resolved parameter set + seeds (see §6)
├── meta.csv                 # cell, guide, target, cell_line(constant),
│                             # sum_factor (scran-recomputed, §4.3 — NOT the true value)
├── counts.csv               # CisGene + 1736 trans genes × cells
├── cis_ground_truth.csv     # guide, x_eff_g_true, sigma_eff (per guide);
│                             # cell, guide, log2_x_true, x_true, sum_factor_true (per cell)
├── guide_ground_truth.csv   # guide, target, guide_log2FC, x_eff_g_true, sigma_eff, n_cells
├── trans_ground_truth.csv   # feature, y_ntc, o_y, effect_type(no_effect/single_hill),
│                             # n_true, K_log2FC_true, full_log2FC_true,
│                             # A_true, Vmax_true, K_true (reconstructed absolute values)
├── sum_factor_scran/        # R script + intermediate CSVs from §4.3 (kept, not cleaned up)
└── fit/recovery/
    ├── posterior_samples_ntc.pt, posterior_samples_cis.pt, posterior_samples_trans.pt
    ├── trans_checkpoint_gene_*.pt     # fit_trans's own checkpointing
    ├── trans_feature_summary_gene.csv # from save_trans_summary(); fitted A/n/K/full_log2FC + fdr_alpha
    └── fit_stats.json                 # per-step (fit_ntc/fit_cis/fit_trans) wall-clock time,
                                        # peak CPU RSS, peak GPU memory, hostname, SLURM job/array-task ID
```

`trans_ground_truth.csv` and the fitted `trans_summary.csv` are both keyed by
`feature`, so downstream comparison is a join, robust to any features dropped by
`_refilter_zero_count_features`.

A single top-level `design_matrix.csv` (one row per `scenario_id × replicate`) records
every swept parameter value and the seed, so any scenario can be identified and
re-run standalone without re-deriving indices.

---

## 6. Reproducibility / seeding

- One **master seed** for the whole study (e.g. `MASTER_SEED = 20260727`).
- `design_matrix.csv` is generated once, deterministically, by enumerating the grid in
  a fixed nested order (cells_per_gene → guide design → X_NTC → o_x → replicate) and
  assigning `scenario_seed = MASTER_SEED + 1000 * scenario_id + replicate_id`
  (plain integer arithmetic — no `hash()`).
- Each scenario's `config.json` records: every parameter value, `scenario_id`,
  `replicate_id`, `scenario_seed`, and git provenance for the exact bayesDREAM code
  used to run it — `bayesdream_commit` (`git rev-parse HEAD`; sufficient on its own to
  reproduce the code content, since git is content-addressed), `bayesdream_branch`
  (aids discovery if the commit ever becomes unreachable and is garbage-collected),
  `bayesdream_git_dirty` (**tracked** files only — modified/staged/deleted; `dirty=True`
  means the recorded commit hash does *not* fully capture what ran), and
  `bayesdream_untracked_count` (informational only, does not count as dirty — a stray
  untracked file, e.g. leftover pre-refactor modules or docs on a lived-in clone, isn't
  a reproducibility hazard unless it's actually imported/used, which git can't tell
  from its mere presence; confirmed in practice necessary on the Berzelius clone used
  for this study). A stable git tag pinning the exact commit is also created once per
  `design_matrix.csv` build — see `examples/simulation_study/BERZELIUS_GUIDE.md`'s
  "Reproducibility" section and `_git_provenance.py`.
- Within a scenario, `scenario_seed` seeds `numpy`, `torch`, and `pyro`
  (`np.random.default_rng`, `torch.manual_seed`, `pyro.set_rng_seed`) once before any
  sampling, and is also passed as `seed=` to `simulate_from_trans_summary` (which uses
  its own `np.random.default_rng(seed)` internally — same value, so cis and trans
  noise streams are both pinned by the one scenario seed).
- SVI fitting (`fit_ntc`/`fit_cis`/`fit_trans`) is itself stochastic (minibatching,
  variational sampling) — the same `scenario_seed` is reused there too, so a rerun of
  a given scenario+replicate is fully deterministic end to end.

---

## 7. Execution plan (Berzelius)

1. **Simulate** all 720 datasets locally or in a CPU array job (cheap: no SVI, just
   NB/RNG sampling, plus one Rscript/scran subprocess per scenario — see §4.2 update
   below) — write `meta.csv`/`counts.csv`/ground-truth CSVs per scenario/replicate.
2. **`fit_ntc`**: single technical group (C=1) is cheap; can run as CPU or 1 thin GPU
   per `SlurmJobGenerator`'s auto-selection. One job (array of 720) since it's shared
   input to `fit_cis`.
3. **`fit_cis`**: CPU partition, per `SlurmJobGenerator` default and confirmed by
   `docs/FIT_TRANS_GUIDE.md`/`CIS_MODEL_PARAMETERS.md` — cheap even at 1000 cells.
   Array job of 720 tasks.
4. **`fit_trans`**: T=1736 features × up to 1000 cells is the expensive step — likely
   1 thin/fat GPU per task per `SlurmJobGenerator`'s memory-based auto-selection.
   Array job of 720 tasks, `function_type='additive_hill'` only (per §1, updated
   2026-07-29 — misspecified-model recovery against single-Hill-only ground truth).
5. Use `bayesDREAM.slurm_jobgen.SlurmJobGenerator.estimate_memory_requirements()` /
   `estimate_time_requirements()` on one representative simulated scenario (largest:
   `cells_per_gene=1000`) **before** generating the full 720-task array, to get
   real (not hand-estimated) memory/time numbers and set `--array=0-719%<N>`
   throttling appropriately. `SlurmJobGenerator` doesn't currently have a notion of
   "720 independent tiny datasets sharing one script" — it's built around N cis genes
   within one dataset. **Resolved**: `examples/simulation_study/generate_slurm.py`
   writes a thin custom wrapper (two array scripts templated over
   `design_matrix.csv` row index, not `cis_genes`) that calls `SlurmJobGenerator`
   only for its memory/time *estimation* math, not `generate_all_scripts()`. Each
   generated script also carries `#SBATCH --account=<account>` (required CLI arg).

   **Found during implementation**: `estimate_memory_requirements()` was completely
   broken (two pre-existing bugs, unrelated to this study, fixed with user approval):
   (1) `self.sparsity = (counts == 0).sum() / counts.size` divides a per-column Series
   by a scalar instead of computing an overall fraction, breaking the very next
   f-string; (2) `_recommend_resources()` read `memory['fit_technical_ram_gb']` /
   `['fit_technical_vram_gb']`, but `docs/memory_calculator.estimate_memory()` actually
   returns `fit_ntc_ram_gb`/`fit_ntc_vram_gb` — a leftover key-name mismatch from a
   `fit_technical`→`fit_ntc` rename. Both fixed directly in `slurm_jobgen.py`.

   **Also found**: `estimate_time_requirements()`'s wall-time formula scales purely by
   `(T/20000)*(N/30000)` relative to a large production-scale baseline. At this
   study's scale (T≈1737, N≤1000) that factor is ~0.003, so the raw estimate rounds to
   `00:00:00` — real wall time at this scale is dominated by the fixed per-SVI-iteration
   cost (set by `niters`, independent of dataset size), which the estimator doesn't
   model. Rather than change that heuristic in `slurm_jobgen.py` itself (a judgment
   call, not a clear bug), `generate_slurm.py` floors the fit step's time budget with
   a `--min_fit_hours` flag and prints both the raw estimate and the floored value it
   used. **Calibrated (2026-07-28) against `single_hill`, now stale**: default was
   set to 3h based on a real `run_recovery_fit.py` run on the largest
   (`cells_per_gene=1000`) scenario on Berzelius with `function_type='single_hill'`,
   predicted to take just over 2h. Since 2026-07-29 the fit call uses `additive_hill`
   instead (see §1 update), which runs ~1.56x the total SVI steps (curriculum
   warmup + main phase, vs. `single_hill`'s flat step count) at a higher per-step
   cost for the main phase — **`--min_fit_hours` needs to be re-measured from a
   real `additive_hill` timed run before trusting it for the full 720-task array.**

   **Resolved (niters/nsamples)**: `run_recovery_fit.py` does not expose `niters`/
   `nsamples` at all — every `fit_ntc`/`fit_cis`/`fit_trans` call uses bayesDREAM's own
   library defaults (`fit_ntc`: 50,000; `fit_cis`: 100,000; `fit_trans`: 100,000–
   200,000 `niters` depending on distribution, plus an automatic `single_hill`
   curriculum-warmup phase before `additive_hill`'s main phase — see §1 update). This
   was a deliberate decision: the point of this study is to validate the *default*
   settings, not some study-specific iteration count. This makes the
   `--min_fit_hours` calibration above straightforward to determine (one real timed
   run at the actual defaults, not a parameter sweep).

## 7b. Execution plan (Dardel) — added 2026-07-29/30

Moved to Dardel (PDC) CPU-only, using its `shared` partition (request `--cpus-per-task`,
memory follows automatically — no separate `--mem`). `examples/simulation_study/
generate_slurm_dardel.py` is the Dardel counterpart to `generate_slurm.py`: no GPU-tier
memory estimation needed, just a fixed `--cores` per fit task.

**Found during the move**: `run_recovery_fit.py` never pinned OMP/MKL/OpenBLAS/NumExpr
thread counts, and calling `bayesDREAM.utils.set_max_threads()` the normal way wouldn't
have fixed it either — that helper imports numpy/torch itself, by which point their
threadpools have already initialized (those env vars are read once at library load, not
per-call). Harmless on Berzelius (GPU jobs get an exclusive node slice), but Dardel's
`shared` partition routinely co-locates multiple array tasks per physical node, so an
unpinned task could try to use every core visible on the node rather than just the cores
it was allocated, oversubscribing every co-located job at once. Fixed by resolving
`--cores`/`$SLURM_CPUS_PER_TASK` and setting the env vars (plus `torch.set_num_threads()`)
before any numpy/pandas/pyro/torch import in the file — verified empirically that the env
var is set before each of those imports fires.

**Core-count scaling benchmark (2026-07-29/30)**, largest scenario
(`scenario_id=96`, `cells_per_gene=1000`), one replicate per core count
(2/4/8/16/32 cores). All 5 jobs hit their 3h `--time` budget mid-`fit_trans` — the
budget at the time was based on Berzelius's `single_hill` CPU timing, not
`additive_hill`. Real per-step throughput during `fit_trans` was recovered from
checkpoint timestamps (`trans_checkpoint_gene_stepNNNNNN.pt`, saved every 10,000
steps, plus `trans_checkpoint_gene_warmup.pt` at the Phase 1→2 boundary — its
timestamp always lands between the step-50000 and step-60000 checkpoints, confirming
`warmup_steps=55,556` for `niters=100,000`, i.e. 155,556 total steps):

| cores | fit_ntc | fit_cis | fit_trans rate | fit_trans (155,556 steps, extrapolated) | **total** | core-hours/job |
|---|---|---|---|---|---|---|
| 2 | 4590s | 1305s | 3.97 steps/s | 39,201s (10.9h) | **12.53h** | 25.1 |
| 4 | 2878s | 1288s | 6.49 steps/s | 23,955s (6.65h) | **7.81h** | 31.2 |
| 8 | 1790s | 1091s | 8.01 steps/s | 19,415s (5.39h) | **6.19h** | 49.5 |
| 16 | 2477s | 1338s | 10.42 steps/s | 14,933s (4.15h) | **5.21h** | 83.3 |
| 32 | 1351s | 1155s | 8.33 steps/s | 18,667s (5.19h) | **5.88h** | 188.2 |

Caveats: `fit_trans` rate is extrapolated linearly across all 155,556 steps from each
job's observed checkpoint interval; the 2-core rate comes from a single 42-minute
interval (least reliable of the five), and Phase 2 (`additive_hill` proper, two Hill
components) plausibly costs somewhat more per step than Phase 1 (`single_hill`
warmup) — a flat-rate extrapolation from mostly-Phase-1 data likely understates true
total time somewhat. `fit_ntc`/`fit_cis` (unlike `fit_trans`) still show ~flat/noisy
scaling with core count, consistent with §7's Berzelius finding — but `fit_trans`
(85-95% of total time) does show real scaling from 2→16 cores before dipping at 32
(plausibly a NUMA/noisy-neighbor effect crossing Dardel's 2-socket, 64-core/socket
node boundary — jobs also landed on different physical nodes on this shared
partition, so some of the non-monotonicity, especially the 16-vs-32 reversal, may be
contention noise rather than a true core-count effect). This revises §7's Berzelius
conclusion that "cores don't matter much for this workload" — that held for
`fit_ntc`/`fit_cis` but not for `fit_trans`, the step that actually dominates total
time.

**Throughput**: core-hours/job is lowest at 2 cores (25.1 vs. 49.5 at 8 cores) — for a
fixed core budget, 2 cores/job completes roughly 2x more jobs than 8 cores/job, so it
remains the throughput-optimal choice despite its longer 12.5h per-job wall-clock
(vs. 5.2h at 16 cores). `generate_slurm_dardel.py --cores 2` (default) sets
`--fit_time_hours=18` (raw 12.53h estimate + margin for the caveats above) —
recalibrate if `--cores` is changed, using the table above as a starting point, and
confirm 18h fits within Dardel's `shared` partition's max walltime before submitting
the full 720-task array.

---

## 8. Evaluation plan (sketch — not implemented)

**Update, 2026-07-29**: since fitting now uses `additive_hill` against single-Hill-only
ground truth (§1), the fitted model reports **two** Hill components (e.g. `n_a`/`K_a`/
`Vmax_a` for the positive-direction component, `n_b`/`K_b`/`Vmax_b` for the
negative-direction one — see `docs/FIT_TRANS_GUIDE.md`/`HILL_FUNCTION_PRIORS.md` for
exact naming) rather than one. The bullets below need to be read against whichever
component's direction matches the ground-truth `full_log2FC`'s sign as "the" recovered
curve, with the opposite-direction component treated as a nuisance parameter expected
to shrink toward `Vmax≈0` — this comparison logic is not yet designed in detail and
should be worked out before implementing this section.

For each fitted `trans_summary.csv` joined to `trans_ground_truth.csv`:

- **Discrimination**: for null-truth features, false-positive rate = fraction with
  `fdr_alpha ≤ 0.05`; for single_hill-truth features, power = fraction with
  `fdr_alpha ≤ 0.05`, stratified by every grid dimension (`n`, `K_log2FC`,
  `full_log2FC`, `y_ntc`, `o_y`, `cells_per_gene`, guide design, `X_NTC`, `o_x`).
- **Parameter recovery** (single_hill-truth, detected only): bias and RMSE of the
  matching-direction fitted component's `n`/`K` vs. true `n_a`, `K_a` (convert
  `K_log2FC_true` → absolute `K_true` for comparison, since the fitted model reports
  `K` in absolute cis-expression units, not log2FC — see
  `docs/HILL_FUNCTION_PRIORS.md` "Interpretation" section), and `full_log2FC`
  (reconstruct from the matching component's fitted `A`/`Vmax` the same way
  `_compute_AV_from_fc` does it in reverse). Also report the opposite-direction
  component's fitted `Vmax` (expected ≈0 if `additive_hill` is behaving well under
  misspecification).
- **Calibration**: coverage of 95% credible intervals for the matching-direction
  component's `n`/`K`/`Vmax` against true values, across replicates.

---

## Next Steps

**Done**: simulator (`bayesDREAM/simulation/cis_panel_simulation.py`), driver scripts
in `examples/simulation_study/` (design matrix, simulate/fit CLIs, SLURM generation),
the two original `slurm_jobgen.py` bugs plus a third (`docs.memory_calculator` import
failing when `generate_slurm.py` is invoked by full path outside the repo root — the
exact way `BERZELIUS_GUIDE.md` instructs running it), the `effect_type='null'`/
pandas-na_values bug, the `device` auto-detect bug, scran-based `sum_factor`
recomputation + `adjust_ntc_sum_factor`/`refit_sumfactor` in the fit pipeline, the
niters/nsamples decision (always library defaults), the `fit_ntc` NaN crash at
production scale (root-caused and fixed — `AutoIAFNormal`'s auto-selection ignored
data sparsity; `AutoNormal` is now the default, verified against the exact crashing
scenario/seed), git-tag provenance for the whole study, and incremental
`save_ntc_fit`/`save_cis_fit`/`save_trans_fit` after each step rather than only at the
end. Design matrix is now 144 scenarios × 5 replicates = 720 rows. Berzelius
deployment is confirmed working end-to-end (env, account, git push access all
verified live on the cluster).

`--min_fit_hours` default is calibrated at **3h** (§7 step 5), based on a real
`run_recovery_fit.py` run on the largest scenario on Berzelius, predicted to take just
over 2h.

**Still open**:

1. **The validation run above hasn't been confirmed complete/successful yet** — it was
   predicted to take just over 2h; confirm it actually finished without error and
   inspect the recovered parameters before trusting the fix at scale.
2. **Evaluation/aggregation script doesn't exist yet** — §8 is a sketch, no code.
3. **Storage footprint not estimated** — 720 scenarios × (`counts.csv`, several
   `posterior_samples_*.pt`, `fit_trans` checkpoint files, `sum_factor_scran/`
   intermediates) could be a meaningful chunk of quota; worth estimating from one
   completed scenario's directory size before committing to the full run.
