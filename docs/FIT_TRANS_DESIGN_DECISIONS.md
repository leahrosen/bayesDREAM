# fit_trans Technical Design Decisions

This document records every significant technical design choice in `fit_trans()` and `_model_y()`, the reasoning behind each, the commit and date it was introduced or changed, and the known pros/cons. It is intended as a living reference so that future changes can be made with full context.

---

## Model Formulation

### Additive Hill Function

**Current form** (`additive_hill`):
```
y = A + alpha * Hill_a(x; Vmax_a, K_a, n_a)
      + beta  * Hill_b(x; Vmax_b, K_b, n_b)
```
where `Hill(x; Vmax, K, n) = Vmax * x^n / (K^n + x^n)` and `n` can be positive (activation) or negative (repression), so a single component can model either direction.

**Alternatives that were considered or exist**:
- `single_hill`: one component only — simpler, less prone to overfitting
- `nested_hill`: Hill_b applied to the *output* of Hill_a (cascade model)
- `polynomial`: arbitrary smooth function in log-space

**Why additive_hill is the default**: captures non-monotonic / biphasic responses (e.g., a gene activated at low perturbation and repressed at high perturbation), while reducing to single_hill when one of alpha/beta collapses to zero.

**Pros**: flexible, biologically interpretable, handles both directions
**Cons**: twice as many parameters as single_hill; overfitting risk when data are sparse or one-directional (see Sparsity section and Curriculum Warmup section)

---

## 1. K Prior: Distribution Family

**Current default**: LogNormal (`use_lognormal_priors=True`)
**Alternative**: Gamma (`use_lognormal_priors=False`)

### History

| Commit | Date | Change |
|--------|------|--------|
| (pre-repo, ~Oct–Nov 2025) | ~Oct–Nov 2025 | LogNormal introduced for binomial/multinomial; later unified to all distributions |
| `754f4ab` | 2026-01-22 | Reverted negbinom/normal/studentt K and Vmax to **Gamma** (`use_lognormal_priors=False` as default) after observing two-component overfitting and convergence gap |
| `9884956` | 2026-01-23 | Restored **LogNormal as default** after identifying the real cause of the overfitting as a bug in x_true point-estimate handling (`c5b1024`, 2025-12-17) rather than the prior choice |

### Reasoning for LogNormal

- **Unified code**: same parameterization for all distributions (binomial, multinomial, negbinom, normal)
- **AutoNormal guide compatibility**: `log_K` lives on the real line → Normal variational distribution is well-suited; Gamma requires a constrained-to-positive parameterization
- **Numerical stability**: working in log-space avoids near-zero K values causing gradient explosions
- **Natural positivity**: K > 0 is guaranteed by the exp transform without explicit clamping

### Pros/Cons

| | LogNormal (current default) | Gamma |
|--|--|--|
| **Parameterization** | log_K ~ Normal → natural for AutoNormal guide | Gamma ~ direct, matched to archive |
| **Tail behaviour** | Heavier right tail → more exploration of large K | Lighter tail → more conservative |
| **Convergence** | Slightly slower (when it was the *only* change before x_true bug was fixed) | Slightly faster in archive comparison |
| **Sparsity enforcement** | Good; recovered after x_true bug fix | Good |
| **Code unification** | Yes (same for all distributions) | No (separate path for negbinom/normal) |

**Current status**: LogNormal is the default and the intended long-term choice. The `use_lognormal_priors=False` option is retained for comparison / regression testing against the archive.

---

## 2. K Prior: Width (CV-Based vs Fixed)

**Current default**: CV-based
`K_std_prior = K_mean_prior * x_true_CV`
where `x_true_CV = std(x_true_mean) / mean(x_true_mean)`

**Alternative**: Gamma-equivalent fixed width
`K_std_prior = K_max / (2 * sqrt(K_alpha))` (K_alpha = 2 → CV ≈ 0.71)

### History

| Commit | Date | Change |
|--------|------|--------|
| `aebe85a` | 2025-12-15 | Confirmed that `fit_trans` already uses CV-based K width; plotting code was corrected to match |
| `f5a674b` | 2026-03-25 | Attempted fix: use Gamma-equivalent width for negbinom/normal LogNormal path to prevent overfitting |
| `2fa6d5d` | 2026-03-25 | Reverted: CV-based width is the intended design; curriculum warmup is the correct remedy |

### Why CV-Based

- **Scale-invariant**: works regardless of cis expression magnitude (raw counts vs log-normalised)
- **Works without guides**: computed from global x_true statistics, so does not require multiple guide levels
- **Interpretable**: CV = 0.5 means K prior std is 50% of the prior mean
- **K is in x_true space regardless of trans modality**, so the same x_true CV applies to all modalities; the K prior width should reflect the actual spread of x_true values

### The Overfitting Concern

When `x_true_CV` is large (≥ 0.7, typical for full CRISPR data with both KO and CA guides), the LogNormal K prior becomes very wide (`K_log_sigma ≈ 0.7–0.8`). During the high-temperature phase of training, both K_a and K_b get gradient signals and can drift to different positions, fitting single-Hill data with two components. The sparsity prior should correct this at convergence, but a sticky local minimum can form before temperature drops.

**The chosen remedy** is the curriculum warmup (see §8 below), not tightening the K prior. Rationale: tightening the prior would bias K_a and K_b toward `K_max/2` in genuine two-Hill systems.

### Pros/Cons

| | CV-based (current) | Fixed Gamma-equivalent |
|--|--|--|
| **Scale invariance** | Yes | No |
| **Works without guides** | Yes | Partially |
| **Bias in true 2-Hill** | None | Mild pull toward K_max/2 |
| **Risk of overfitting** | Higher when CV is large | Lower |
| **Remedy for overfitting** | Curriculum warmup | N/A (prior itself limits K divergence) |

---

## 3. Vmax Prior: Distribution Family

**Current default**: LogNormal (`use_lognormal_priors=True`)
**Vmax_log_sigma** is computed from within-guide variance (raw, not CV):
```python
Vmax_std_prior = sqrt(mean_within_guide_var)
ratio_Vmax     = Vmax_std_prior / Vmax_mean_prior
Vmax_log_sigma = sqrt(log1p(ratio_Vmax^2))
```

### Why Raw Variance (Not CV) for Vmax

- Vmax is an *absolute* amplitude (expression units), not a relative quantity
- Using raw variance preserves scale information needed to constrain the response magnitude
- Contrast with K (a *position* on the x-axis), where scale-invariance is desirable

### History

Same as K prior history — LogNormal was introduced ~Oct–Nov 2025, reverted Jan 22, restored Jan 23. See §1.

---

## 4. A (Baseline) Prior

Distribution depends on data type:

| Distribution | A Prior | Rationale |
|--|--|--|
| negbinom | `Exponential(1/Amean_adjusted)` | A must be a positive count; exponential is the conjugate for Poisson-like data |
| normal / studentt | `Normal(Amean_adjusted, abs(Amean_adjusted))` | A can be negative (e.g., SpliZ scores below zero) |
| binomial | `Beta(α=1, β=(1−Amean)/Amean)` | Constrained to [0,1]; α=1 gives weak directional push toward 0 (baseline PSI should be low) |
| multinomial | `Dirichlet(mean_normalized * K)` | Per-category, sums to 1; weak concentration ≈ 1 per category |

`Amean_adjusted = (1−weight) * Amean + weight * Vmax_mean`
where `weight = o_y / (o_y + beta_o_beta/beta_o_alpha)` adaptively blends toward Vmax when the gene is very overdispersed (i.e., baseline is uncertain).

**Why the blended Amean**: prevents the A prior from being degenerate (too close to zero) when overdispersion is large and the distinction between A and A + Vmax is unclear from data alone.

---

## 5. n (Hill Coefficient) Prior — Hierarchical

```
sigma_n_a ~ Exponential(1/5)   # global hyperprior (shared across all T genes)
n_a_raw   ~ Normal(0, sigma_n_a)
n_a       = alpha * n_a_raw     # effective regularization via sparsity
```

### Why Hierarchical

- Shares information across all T genes: if most genes have weak, near-zero n, sigma_n_a is learned to be small, which regularises all genes simultaneously
- `n = alpha * n_a_raw` means that when alpha → 0 (component inactive), n → 0 and `Hill(x)` becomes 0.5 everywhere — the contribution collapses gracefully without requiring a separate n prior

### Why *n* Gets This Treatment but K and Vmax Do Not

The user's design intent: K and Vmax priors should be **per-gene and data-driven** (no information sharing across genes, as their magnitudes are gene-specific). n is a *shape* parameter (cooperativity) that should plausibly share information — most genes are expected to have moderate cooperativity, and the hyperprior enforces this. This is analogous to a random-effects model for shape.

### Archive Consistency

Unchanged from archive: `sigma_n_a ~ Exponential(1/5)`, `n_mu = 0`. Same for `sigma_n_b`.

---

## 6. Sparsity Prior — RelaxedBernoulli

```
alpha ~ RelaxedBernoulli(temperature=T, logit(p_n))   p_n = 1e-6
beta  ~ RelaxedBernoulli(temperature=T, logit(p_n))
```

`logit(1e-6) ≈ −13.8`, which strongly pushes alpha and beta toward 0.

### Why p_n = 1e-6

The KL cost for a gene to have a non-zero component is approximately `|logit(p_n)| ≈ 13.8` nats per gene. Over N cells, the likelihood improvement needed to activate a component is ~13.8/N nats per cell (≈ 0.07 nats/cell for N=200). This is a meaningful but not excessively strict threshold: a genuine single-direction responder clears it; a null gene does not.

### Temperature Annealing

Linear schedule: `T(t) = T_init + (T_final − T_init) * t/niters`
with `T_init = 1.0`, `T_final = 0.1`.

At T=1.0: alpha ≈ 0.5 (soft, both components can receive gradient signal)
At T=0.1: alpha is near 0 or 1 (near-discrete)

When `warmup_steps > 0`, the schedule **restarts** from T_init at the switch from single_hill to additive_hill, so beta gets its own full annealing schedule (see §8).

**Unchanged from archive.**

### Pros/Cons

| | p_n = 1e-6 |
|--|--|
| **Sparsity** | Strong: most genes stay near zero |
| **Sensitivity** | Any gene with > ~0.07 nats/cell improvement per component activates |
| **Sticky local minima** | Possible when two components both receive gradient at high T (motivation for curriculum warmup) |

---

## 7. Data-Driven Prior Computation

### 7a. A_mean and Vmax_mean: Percentiles vs Min/Max

**Current** (`use_archive_prior_computation=False`, default since `9884956`, 2026-01-23):
- `A_mean = 5th percentile of guide means`
- `Vmax_mean = 95th percentile − 5th percentile (range)`

**Archive** (`use_archive_prior_computation=True`):
- `A_mean = min(guide means)`
- `Vmax_mean = max(guide means)` (absolute, not range)

Introduced in `39e7558` / `d9034cd` (fixing a critical Vmax prior bug where Vmax was set to the absolute max rather than the range). Percentiles chosen for robustness against outlier guides.

**Pros of percentiles**: robust to outlier guides, stable with small guide counts
**Cons**: potentially underestimates range (5th–95th instead of full range)

### 7b. Technical Correction Before Prior Computation

**Current** (`correct_priors_for_technical=True`, default since `9884956`, 2026-01-23):
Removes batch effects from `y_obs_for_prior` before computing A_mean and Vmax_mean, using the inverse of the alpha_y transform.

**Archive** (`correct_priors_for_technical=False`): priors computed from raw sum-factor-normalised data; technical correction only applied during likelihood computation.

**Commit**: introduced as an option in `6fc5410` (2026-01-23), made default in `9884956` (2026-01-23).

**Pros**: priors are in the "true" biological space rather than batch-confounded space
**Cons**: if alpha_y is noisy or large, division can amplify variance in the prior estimates; small risk of overcorrection

---

## 8. Curriculum Warmup (`warmup_steps`)

**Introduced**: `3253e8a`, 2026-03-25

**Motivation**: During the high-temperature phase of additive_hill fitting, both K_a/Vmax_a and K_b/Vmax_b receive gradient signals. With a wide K prior (CV-based), K_a and K_b can drift to different positions. By the time temperature drops and the sparsity prior tries to kill one component, the model may be in a sticky local minimum where keeping both components slightly active is an ELBO optimum.

**Mechanism**: When `warmup_steps > 0` and `function_type='additive_hill'` (or `nested_hill`):
1. **Phase 1** (steps 0 to warmup_steps): model runs as `single_hill`. K_a, Vmax_a, A, o_y, alpha_y, and alpha converge to their single-component optima. The param store accumulates these values.
2. **Phase 2** (steps warmup_steps to niters): model switches to `additive_hill`. K_b, Vmax_b, n_b_raw, sigma_n_b, and beta are **initialised fresh from the prior** (lazy initialisation by AutoNormalMessenger encountering these sites for the first time). K_a and Vmax_a carry over from Phase 1.
Temperature **restarts from T_init** at the switch, so beta has a full annealing schedule.

**How sticky is the transition?** In a true two-Hill system (e.g., K_a=1, K_b=4):
- Phase 1: K_a settles at a compromise position (~K=2)
- Phase 2: K_b initialises near K_max/2 and moves toward K=4 via gradient; K_a also adjusts from ~2 toward 1
- The Adam momentum from Phase 1 initially resists K_a moving, but Phase 2 gradient signals accumulate and K_a reaches the correct position
- Any gene where the second component improves the ELBO by > ~13.8 nats will activate beta

**Compute overhead**: approximately `warmup_steps / niters * ~90%` (single_hill steps are marginally faster). For warmup_steps = niters/10: ~9% overhead.

**Suggested starting point**: `warmup_steps = niters // 10`

**Pros**:
- Prevents K_b from contaminating the single-Hill gradient during warmup
- Does not bias K_a/K_b toward K_max/2 in true two-Hill cases
- No change to prior width (prior remains uninformative/CV-based)

**Cons**:
- In Phase 2, K_a needs to "unlearn" the compromise position from Phase 1; if Phase 2 is too short or learning rate too low, K_a may not fully converge to its true position
- Adds ~10% training time

**Default**: `warmup_steps=0` (no warmup, original behaviour preserved)

---

## 9. Variational Family — AutoNormalMessenger

All fitting (technical, cis, trans) uses `pyro.infer.autoguide.AutoNormalMessenger`.

**Why Messenger variant** (not plain `AutoNormal`):
`AutoNormalMessenger` handles plates lazily and is more memory-efficient for large T (many genes). It is also compatible with the curriculum warmup: new latent sites (K_b, Vmax_b, beta, etc.) can be added at runtime when the model switches from single_hill to additive_hill, because parameters are initialised on first encounter.

**Introduced**: `79e6375` (2026-03-11) for the technical fitter; already used in trans fitter.

---

## 10. Optimizer

| Function type | Optimizer | Parameters | Rationale |
|--|--|--|--|
| `single_hill`, `additive_hill`, `nested_hill` | ClippedAdam | lr=1e-3, clip_norm=10.0 | Matches archive; gradient clipping prevents instability from large n_a gradients |
| `polynomial` | PyroLRScheduler (OneCycleLR + Adam) | base_lr=1e-3, max_lr=1e-2, pct_start=0.1 | OneCycleLR provides warm-up + cosine decay, beneficial for the higher-dimensional polynomial space |

**Commit**: Hill-based revert in `c325714`; polynomial OneCycleLR in `3850aa8`.

---

## 11. Numerical Stability

### logK Parameterisation for Hill Function

For negbinom/normal when `use_lognormal_priors=True`, the Hill function is evaluated as:
```python
Hill_based_positive_logK(x, Vmax, A=0, logK, n)
```
using `x^n / (exp(n * logK) + x^n)` to avoid computing `K^n` directly (which can overflow for large n).

**Introduced**: `53abfb0`, 2026-02-19.

### use_epsilon

`use_epsilon=False` (default since `e53ffb6`, 2026-01-23): no epsilon added to NegativeBinomial logits. Matches archive exactly.
`use_epsilon=True`: adds 1e-8 to the log-normalisation term for numerical safety. Can cause a small but consistent ELBO gap vs archive.

---

## 12. Known Issues and Open Questions

| Issue | Status | Notes |
|--|--|--|
| Massive CIs when subsetting to CRISPRi-only or CRISPRa-only with additive_hill | **Open** | Likely identifiability issue: one-directional data cannot constrain both K and Vmax of the Hill plateau. Curriculum warmup may help but does not fully resolve. Investigate whether single_hill or a Vmax regularisation improves this. |
| Two-component overfitting when fitting additive_hill to single-Hill data | **Partially addressed** | Curriculum warmup reduces early-phase K_b contamination. The CV-based K prior remains wide by design (uninformative). If warmup is insufficient, increase warmup_steps. |
| correct_priors_for_technical noisiness | **Open** | When alpha_y is large and variable, dividing y_obs_for_prior by it can introduce instability in prior estimates. Consider capping the correction factor. |
| No cross-gene sharing of K/Vmax | **By design** | User preference: K and Vmax are gene-specific. Only n has a hierarchical prior (sigma_n_a). |

---

## 13. Parameter Defaults Summary

| Parameter | Default | Archive | Rationale for difference |
|--|--|--|--|
| `use_lognormal_priors` | `True` | `False` (Gamma) | Unified parameterisation, better guide compatibility |
| `use_data_driven_priors` | `True` | `True` (implicit) | Both use guide means; percentiles vs min/max differ |
| `use_archive_prior_computation` | `False` | `True` | Percentile method more robust |
| `correct_priors_for_technical` | `True` | `False` | Priors in biological space (not batch-confounded) |
| `use_epsilon` | `False` | `False` | Matches archive |
| `p_n` (sparsity) | `1e-6` | `1e-6` | Unchanged |
| `init_temp` | `1.0` | `1.0` | Unchanged |
| `final_temp` | `0.1` | `0.1` | Unchanged |
| `K_alpha` | `2` | `2` | Unchanged |
| `warmup_steps` | `0` | N/A | New; default preserves original behaviour |

To exactly replicate archive behaviour:
```python
model.fit_trans(
    ...,
    use_lognormal_priors=False,
    correct_priors_for_technical=False,
    use_archive_prior_computation=True,
    use_epsilon=False,
)
```

---

## Related Documentation

- `HILL_FUNCTION_PRIORS.md` — detailed reference for all prior distributions and their formulas
- `FIT_TRANS_GUIDE.md` — user guide (function types, workflow, interpretation)
- `ARCHITECTURE.md` — codebase structure
- `OUTSTANDING_TASKS.md` — current development priorities
