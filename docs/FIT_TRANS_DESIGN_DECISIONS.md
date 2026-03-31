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

**Current**: LogNormal (only option — Gamma path removed)

### History

| Commit | Date | Change |
|--------|------|--------|
| (pre-repo, ~Oct–Nov 2025) | ~Oct–Nov 2025 | LogNormal introduced for binomial/multinomial; later unified to all distributions |
| `754f4ab` | 2026-01-22 | Reverted negbinom/normal/studentt K and Vmax to **Gamma** after observing two-component overfitting |
| `9884956` | 2026-01-23 | Restored **LogNormal as default** after identifying the real cause of overfitting as an x_true bug |
| `b966bfd` | 2026-03-31 | Removed `use_lognormal_priors` parameter; LogNormal is now the only path |

### Reasoning for LogNormal

- **Unified code**: same parameterization for all distributions (binomial, multinomial, negbinom, normal)
- **AutoNormal guide compatibility**: `log_K` lives on the real line → Normal variational distribution is well-suited
- **Numerical stability**: working in log-space avoids near-zero K values causing gradient explosions
- **Natural positivity**: K > 0 is guaranteed by the exp transform without explicit clamping

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

## 3. Vmax Prior: Distribution Family and Width

**Current**: LogNormal with a minimum log_sigma floor of **1.0** (negbinom/normal/studentt):
```python
Vmax_sigma     = Vmax_prior_mean / sqrt(Vmax_alpha)
Vmax_log_sigma = max(sqrt(log1p((Vmax_sigma / Vmax_prior_mean)^2)), 1.0)
Vmax_log_mu    = log(Vmax_prior_mean) - 0.5 * Vmax_log_sigma^2
```

The floor of 1.0 means the 95% CI upper bound is ≈ **4.3× Vmax_mean**, regardless of the data-driven spread. This is necessary because for one-sided subsets (CRISPRa-only or CRISPRi-only), `Vmax_mean` underestimates the true dynamic range and a tight prior would prevent the posterior from moving to the true value.

For **binomial/multinomial**, `Vmax_a ~ Beta(Vmax_mean × 2, (1 − Vmax_mean) × 2)` — concentration=2 gives CV ≈ 0.6–1.0, replacing the previous concentration=10.

### History

| Commit | Date | Change |
|--------|------|--------|
| ~Oct–Nov 2025 | | LogNormal introduced alongside K |
| `754f4ab` | 2026-01-22 | Reverted to Gamma |
| `9884956` | 2026-01-23 | Restored LogNormal as default |
| `b966bfd` | 2026-03-31 | Removed Gamma path; LogNormal is now the only option |
| `05b0375` | 2026-03-31 | Added log_sigma floor=1.0; lowered binomial/multinomial concentration 10→2 |

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
sigma_n_a  ~ Exponential(1/2)                  # global hyperprior (mean = 2)
n_mu_raw   = half_n * atanh((n_mu - center_n) / half_n)   # inverse soft_clamp of n_mu = 0
n_a_raw    ~ Normal(n_mu_raw, sigma_n_a)        # per-gene in unconstrained space
n_a        = soft_clamp(n_a_raw, nmin, nmax)   # constrained to safe range
```

where `center_n = (nmin + nmax) / 2`, `half_n = (nmax - nmin) / 2`, and `nmin`/`nmax` are physically-derived overflow bounds (see §11).

### Why Hierarchical

- Shares information across all T genes: if most genes have weak, near-zero n, sigma_n_a shrinks, regularising all genes simultaneously.
- `n_a` gates the Hill shape; the sparsity prior on `alpha` / `beta` gates whether the component has any effect at all. The two work in concert: a component with alpha ≈ 0 contributes nothing regardless of n_a.

### Why *n* Gets This Treatment but K and Vmax Do Not

K and Vmax priors are **per-gene and data-driven** (magnitudes are gene-specific; no sharing is appropriate). n is a *shape* parameter (cooperativity) where sharing information across genes is reasonable — most genes are expected to have moderate cooperativity. This is analogous to a random-effects model for shape.

### sigma_n Prior Width History

| Commit | Date | Change | Rationale |
|--------|------|--------|-----------|
| (archive) | pre-2026 | `Exp(1/5)`, mean = 5 | Original value |
| `8d8332d` | 2026-03-30 | `Exp(1/2)`, mean = 2 | Exp(1/5) allowed sigma_n to grow so large that n routinely hit the physical overflow bounds (nmin/nmax ≈ ±38). Exp(1/2) covers biological Hill coefficients ~0.5–8 without systematic boundary hits. |

### Hard Clamp → Soft Clamp (`_soft_clamp`)

**Original** (`torch.clamp`):
```python
n_a = n_a_raw.clamp(nmin, nmax)
```
`torch.clamp` has zero gradient outside [nmin, nmax]. Once n_a_raw overshoots the boundary, the optimizer receives no signal to pull it back ("dead-gradient" problem).

**Current** (`_soft_clamp`):
```python
def _soft_clamp(x, lo, hi):
    half = 0.5 * (hi - lo);  center = 0.5 * (hi + lo)
    return center + half * tanh(x / half)
```
Gradient = sech²(x/half) ≥ sech²(1) ≈ 0.42 at the boundary, never zero. **Introduced**: `a75bee7`, 2026-03-30.

### Prior Miscalibration Bug (Fixed `8350c35`, 2026-03-30)

With the soft_clamp, `Normal(0, sigma)` on n_a_raw does **not** mean the constrained prior mode is 0. The prior mode of n_a is:

$$\text{mode}(n_a) = \text{soft\_clamp}(0,\, n_{\min},\, n_{\max}) = \frac{n_{\min} + n_{\max}}{2}$$

For typical data spanning x ∈ [0.3, 5] (KO to CA guides):
- nmin ≈ −74, nmax ≈ 55 → prior mode ≈ **−9.3** — systematically negative!

**Fix**: Use the inverse soft_clamp of n_mu (= 0) as the prior mean for n_a_raw:
```python
center_n     = 0.5 * (nmax + nmin)
half_n       = 0.5 * (nmax - nmin)
n_mu_raw     = half_n * atanh((n_mu - center_n) / half_n)
```
By construction, `soft_clamp(n_mu_raw, nmin, nmax) = n_mu = 0`, so the constrained prior mode is always at the intended value regardless of dataset asymmetry.

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

For negbinom/normal/studentt, the Hill function is evaluated as:
```python
Hill_based_positive_logK(x, Vmax, A=0, logK, n)
```
using `x^n / (exp(n * logK) + x^n)` to avoid computing `K^n` directly (which can overflow for large n).

**Introduced**: `53abfb0`, 2026-02-19.

### use_epsilon

`use_epsilon=False` (default since `e53ffb6`, 2026-01-23): no epsilon added to NegativeBinomial logits. Matches archive exactly.
`use_epsilon=True`: adds 1e-8 to the log-normalisation term for numerical safety. Can cause a small but consistent ELBO gap vs archive.

---

## 12. K–n Joint Identifiability and the Smoothing-Out Bias

### The Identifiability Problem

The Hill function has a well-known joint non-identifiability between K and n when the data do not span both sides of K:

$$h_+(x;\, K, n) = \frac{x^n}{K^n + x^n}$$

When all observed x values are **far below K** (e.g., K >> max(x)), the function is approximately:
$$h_+(x;\, K, n) \approx \left(\frac{x}{K}\right)^n$$

This is a power law. Any pair (K′, n′) satisfying `n′ * log(K′) ≈ n * log(K)` (i.e., the same log-ratio) fits the sub-threshold data equally well. The likelihood surface is flat along this ridge, so the **prior dominates** in this regime.

### Why This Manifests as "Smoothing Out"

The NTC-centred K prior has `E[K] = x_NTC` and `sigma = 5·ln(2)/2`, giving a 95% CI of ±5 log2FC around NTC. The n prior has mode at 0 (no cooperativity → near-linear response).

When the true EC50 lies far above the observed x range (e.g., CRISPRi-only data where all x < x_NTC):
1. The data provides no gradient pulling K above the observed range.
2. The K prior pulls K back toward x_NTC.
3. With K ≈ x_NTC and all x below K, the Hill function is quasi-linear near 0: h+(x) ≈ (x/K)^n.
4. The n prior, pulling toward 0, makes the response even shallower.
5. The net result: the model fits a **gently rising curve** rather than a steep sigmoid whose plateau is off-screen.

This behaviour is called "smoothing out" — the model acts as if the dose-response is still rising at the limit of the observed range, rather than acknowledging that the plateau may be unreachable.

### When This Is a Problem vs. Acceptable

| Fit type | K outside observed range? | Effect | Status |
|----------|--------------------------|--------|--------|
| **Full dataset** (CRISPRi + NTC + CRISPRa) | Only if true K >> CA expression level | Rare; CA guides usually reach > 3 log2FC, covering most plausible K | Acceptable |
| **CRISPRa-only** | If true K << NTC (K below the activation range) | Unusual; most activatable genes have K near or below NTC | Occasionally an issue |
| **CRISPRi-only** | Almost always: true K ≥ NTC, but all x < NTC | Common and systematic | **Known limitation** |

### Practical Implication

**EC50 estimates from one-sided fits (CRISPRi-only or CRISPRa-only) are unreliable when the true EC50 lies outside the observed x-range.** The NTC-centred prior is the most informative default available without external data, but it will pull K toward NTC and underestimate EC50 when the true value is far above (or below) NTC.

For ALAS2: full and CRISPRa fits agree on log2(EC50) ≈ +3.2; the CRISPRi-only fit returns ≈ +0.95. The CRISPRi data cannot distinguish EC50 = 0.95 from EC50 = 3.2 because all observations lie below both values. This is not a model failure — it is a fundamental data limitation.

**Recommended practice**: Use the full-dataset fit for EC50 estimation. One-sided fits remain valid for effect-direction classification and FDR-controlled detection of active components.

### The Deliberate Bias Is in the Right Direction

Smoothing out is a better failure mode than the alternative (K drifting to implausible extremes):
- A smoothed-out Hill curve still fits the observed data well.
- A Hill curve with K wildly off-range can produce near-zero gradients for the observed data, destabilising training.
- The FDR criterion for component activity (`fdr_alpha < 0.05`) uses `P(alpha * Vmax / A > ε)`, which correctly captures whether an *observable* effect exists in the data, not whether the fitted K is the true EC50.

---

## 13. Known Issues and Open Questions

| Issue | Status | Notes |
|--|--|--|
| EC50 underestimation in CRISPRi-only / CRISPRa-only fits | **By design / documented limitation** | K–n joint non-identifiability when true EC50 lies outside observed x-range. See §12 for full analysis. Use full-dataset fits for EC50 estimation. |
| Two-component overfitting when fitting additive_hill to single-Hill data | **Partially addressed** | Curriculum warmup reduces early-phase K_b contamination. The CV-based K prior remains wide by design. If warmup is insufficient, increase warmup_steps. |
| correct_priors_for_technical noisiness | **Open** | When alpha_y is large and variable, dividing y_obs_for_prior by it can introduce instability in prior estimates. Consider capping the correction factor. |
| No cross-gene sharing of K/Vmax | **By design** | User preference: K and Vmax are gene-specific. Only n has a hierarchical prior (sigma_n_a). |

---

## 14. Parameter Defaults Summary

| Parameter | Default | Archive | Rationale for difference |
|--|--|--|--|
| Vmax/K prior family | LogNormal (only) | Gamma | LogNormal removed Gamma path entirely (b966bfd) |
| Vmax log_sigma floor | 1.0 | none | Allows posterior to explore 4× above data-driven estimate |
| Beta concentration (bin/multi) | 2 | 10 | More diffuse; allows one-sided subsets to fit properly |
| `use_data_driven_priors` | `True` | `True` (implicit) | Both use guide means; percentiles vs min/max differ |
| `use_archive_prior_computation` | `False` | `True` | Percentile method more robust |
| `correct_priors_for_technical` | `True` | `False` | Priors in biological space (not batch-confounded) |
| `use_epsilon` | `False` | `False` | Matches archive |
| `p_n` (sparsity) | `1e-6` | `1e-6` | Unchanged |
| `init_temp` | `1.0` | `1.0` | Unchanged |
| `final_temp` | `0.1` | `0.1` | Unchanged |
| `K_alpha` | `2` | `2` | Unchanged |
| `warmup_steps` | `0` | N/A | New; default preserves original behaviour |

To approximately replicate archive behaviour (note: Gamma path no longer available):
```python
model.fit_trans(
    ...,
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
