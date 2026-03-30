# Hill Function Priors for Trans Effects

This document describes all priors for Hill function parameters (A, Vmax, K, n) across different distribution types in `fit_trans()`.

## Overview

The trans effects model uses Hill-based dose-response functions to model how trans gene expression depends on cis gene expression. The functional form is:

```
y = A + Vmax_sum * (alpha * Hill_a(x; K_a, n_a) + beta * Hill_b(x; K_b, n_b))
```

Where:
- **A**: Baseline expression (minimum value)
- **Vmax**: Maximum amplitude of response
- **K**: Half-saturation constant (inflection point of Hill curve)
- **n**: Hill coefficient (cooperativity, steepness of sigmoid)
- **alpha, beta**: Sparsity parameters (RelaxedBernoulli, controls which Hill function is active)

## Data-Driven Prior Computation

All priors are computed from observed data using **percentiles and robust statistics**:

### 1. Filter by count threshold (binomial/multinomial only)
- Only use observations where denominator ≥ 3 (binomial) or total_counts ≥ 3 (multinomial)
- This prevents unreliable estimates from low-coverage features

### 2. Compute guide-level statistics (WITH GUIDES)
When guides are available (`guide_code` in metadata):

```python
guide_means = []  # Mean expression per guide
guide_vars = []   # Variance within each guide
for g in unique_guides:
    vals_g = y_obs_for_prior[guides_tensor == g, ...]
    guide_means.append(nanmean(vals_g, dim=0))
    guide_vars.append(nanvar(vals_g, dim=0))

guide_means = torch.stack(guide_means, dim=0)  # [G, T] or [G, T, K]
guide_vars = torch.stack(guide_vars, dim=0)    # [G, T] or [G, T, K]
```

**Derive prior parameters from guide statistics:**
- **A_mean**: 5th percentile across guides (`nanquantile(guide_means, 0.05, dim=0)`)
- **Vmax_mean**: 95th percentile across guides (`nanquantile(guide_means, 0.95, dim=0)`)
- **A_mean constraint**: Clamped to minimum of 1e-3
- **mean_within_guide_var**: Average variance within guides (`nanmean(guide_vars, dim=0)`)

### 3. Fallback to overall statistics (WITHOUT GUIDES)
When no guides are available (NTC-only or single-guide data):

```python
for t in range(y_obs_for_prior.shape[1]):
    vals_t = y_obs_for_prior[:, t]
    valid_t = vals_t[~torch.isnan(vals_t)]

    A_mean_tensor.append(torch.quantile(valid_t, 0.05))      # 5th percentile
    Vmax_mean_tensor.append(torch.quantile(valid_t, 0.95))   # 95th percentile
    overall_vars.append(torch.var(valid_t))

A_mean_tensor = A_mean_tensor.clamp_min(1e-3)  # Enforce minimum
```

### 4. Distribution-specific clamping
For **binomial and multinomial** (probabilities must be in [0, 1]):
```python
Vmax_mean_tensor = Vmax_mean_tensor.clamp(min=1e-3, max=1.0 - 1e-6)
A_mean_tensor = A_mean_tensor.clamp(min=1e-3, max=1.0 - 1e-6)
```

### 5. K computation using Coefficient of Variation (CV)
**Works with or without guides:**

```python
# Global statistics of x_true (cis expression)
x_true_mean_global = x_true_mean.mean()
x_true_std_global = x_true_mean.std()
x_true_CV = x_true_std_global / x_true_mean_global  # Coefficient of variation

# K_max from guides if available, otherwise from overall max
if guides available:
    guide_x_means = [x_true_mean[guides_tensor == g].mean() for g in unique_guides]
    K_max_tensor = max(guide_x_means)
else:
    K_max_tensor = x_true_mean.max()

# K prior parameters
K_mean_prior = (K_max_tensor / 2.0).clamp_min(epsilon)
K_std_prior = K_mean_prior * x_true_CV  # CV-based variance
```

**Key insight**: CV is scale-invariant, so it works regardless of expression magnitude or guide structure.

---

## Distribution-Specific Priors

### 1. **Negative Binomial** (Gene Expression Counts)

| Parameter | Prior Distribution | Mean | Variance | Shape |
|-----------|-------------------|------|----------|-------|
| **A** | Exponential | `Amean_adjusted = (1-weight)*Amean + weight*Vmax_mean` | `1/Amean_adjusted` | `[T]` |
| **Vmax_a** | Log-Normal | `Vmax_mean` | Raw variance (data-driven) | `[T]` |
| **Vmax_b** | Log-Normal | `Vmax_mean` | Raw variance (data-driven) | `[T]` |
| **K_a** | Log-Normal | `K_max / 2` | CV-based (data-driven) | `[T]` |
| **K_b** | Log-Normal | `K_max / 2` | CV-based (data-driven) | `[T]` |
| **n_a** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_a ~ Exponential(1/2)` | `[T]` |
| **n_b** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_b ~ Exponential(1/2)` | `[T]` |
| **alpha** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` |
| **beta** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` |

**Notes**:
- `weight = o_y / (o_y + beta_o_beta/beta_o_alpha)` (adaptively blends min and max based on overdispersion)
- `n_a = alpha * n_a_raw` (multiplied by sparsity parameter)
- **NEW**: Vmax and K use Log-Normal (more stable than Gamma for AutoNormal Guide)
- **NEW**: K variance is CV-based (works with or without guides)

**Log-Normal parameterization for Vmax**:
```python
# Use raw variance from mean_within_guide_var
Vmax_mean_prior = Vmax_mean.clamp_min(epsilon)
Vmax_var_prior = mean_within_guide_var  # Raw variance (within-guide)
Vmax_std_prior = sqrt(Vmax_var_prior.clamp_min(epsilon))

# Log-Normal parameters
ratio_Vmax = (Vmax_std_prior / Vmax_mean_prior).clamp_min(1e-6)
Vmax_log_sigma = sqrt(log1p(ratio_Vmax^2))
Vmax_log_mu = log(Vmax_mean_prior) - 0.5 * Vmax_log_sigma^2

log_Vmax_a ~ Normal(Vmax_log_mu, Vmax_log_sigma)
Vmax_a = exp(log_Vmax_a)
```

**Log-Normal parameterization for K**:
```python
# Use CV-based variance
K_mean_prior = (K_max / 2.0).clamp_min(epsilon)
K_std_prior = K_mean_prior * x_true_CV  # CV-based

# Log-Normal parameters
ratio_K = (K_std_prior / K_mean_prior).clamp_min(1e-6)
K_log_sigma = sqrt(log1p(ratio_K^2))
K_log_mu = log(K_mean_prior) - 0.5 * K_log_sigma^2

log_K_a ~ Normal(K_log_mu, K_log_sigma)
K_a = exp(log_K_a)
```

---

### 2. **Binomial** (Proportions with Denominator)

#### Reparameterization
Instead of sampling A and Vmax separately, we use:
```
A ~ Beta(α=1, β)             # Minimum probability (pushes toward 0)
upper_limit ~ Beta(α, β=1)   # Maximum probability (pushes toward 1)
Vmax_sum = upper_limit - A   # Total amplitude
y = A + Vmax_sum * (alpha * Hill_a + beta * Hill_b)
```

| Parameter | Prior Distribution | Mean | Variance Formula | Shape | Data-Driven? |
|-----------|-------------------|------|------------------|-------|--------------|
| **A** | Beta (α=1) | `A_mean` (5th %ile) | Weak prior toward 0 | `[T]` | ✓ (mean only) |
| **upper_limit** | Beta (β=1) | `Vmax_mean` (95th %ile, max 1-1e-6) | Weak prior toward 1 | `[T]` | ✓ (mean only) |
| **Vmax_sum** | Deterministic | `upper_limit - A` (clamped ≥ 0) | - | `[T]` | ✓ (derived) |
| **K_a** | Log-Normal | `K_mean / 2` | CV-based (data-driven) | `[T]` | ✓ (mean & var) |
| **K_b** | Log-Normal | `K_mean / 2` | CV-based (data-driven) | `[T]` | ✓ (mean & var) |
| **n_a** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_a ~ Exponential(1/2)` | `[T]` | Partially |
| **n_b** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_b ~ Exponential(1/2)` | `[T]` | Partially |
| **alpha** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` | No |
| **beta** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` | No |

**Simplified Beta priors (NEW)**:
```python
# A ~ Beta(α=1, β) with mean = A_mean
# For Beta: mean = α/(α+β) = 1/(1+β) = A_mean
# Solving: β = (1-A_mean)/A_mean

beta_A = (1.0 - A_mean_tensor) / A_mean_tensor  # [T]
alpha_A = 1.0

A ~ Beta(alpha_A, beta_A)  # Pushes toward 0 (α=1 creates bias toward lower values)

# upper_limit ~ Beta(α, β=1) with mean = Vmax_mean
# For Beta: mean = α/(α+β) = α/(α+1) = Vmax_mean
# Solving: α = Vmax_mean/(1-Vmax_mean)

alpha_upper = Vmax_mean_tensor / (1.0 - Vmax_mean_tensor)  # [T]
beta_upper = 1.0

upper_limit ~ Beta(alpha_upper, beta_upper)  # Pushes toward 1 (β=1 creates bias toward upper values)
```

**K prior (Log-Normal, CV-based)**:
```python
# Same as negbinom (see above)
K_mean_prior = (K_max / 2.0).clamp_min(epsilon)
K_std_prior = K_mean_prior * x_true_CV  # CV-based

log_K_a ~ Normal(K_log_mu, K_log_sigma)
K_a = exp(log_K_a)
```

**Notes**:
- **NEW**: Simplified Beta priors with α=1 or β=1 (weak priors with correct means)
- Hills computed with `Vmax=1` (output [0,1]), then scaled by `Vmax_sum`
- No clamps on combined_hill or y_dose_response (naturally in valid range)
- `Vmax_sum` is the **same** for both Hill_a and Hill_b
- **NEW**: K uses unified Log-Normal with CV-based variance

---

### 3. **Multinomial** (Categorical, K Categories)

#### Reparameterization
Uses **Dirichlet distributions** for K-dimensional probabilities:
```
A ~ Dirichlet(concentration_A)           # K-dimensional, sums to 1
upper_limit ~ Dirichlet(concentration_upper) # K-dimensional, sums to 1
Vmax_sum = upper_limit - A (clamped ≥ 0)  # K-dimensional amplitudes

# Fit K-1 independent Hill functions:
for k in 1..(K-1):
    y_k = A_k + Vmax_sum_k * (alpha * Hill_a_k + beta * Hill_b_k)

# Kth category is residual:
y_K = 1 - sum(y_1, ..., y_{K-1})
```

| Parameter | Prior Distribution | Mean | Variance Formula | Shape | Data-Driven? |
|-----------|-------------------|------|------------------|-------|--------------|
| **A** | Dirichlet | `A_mean_normalized` (5th %ile) | Weak concentration | `[T, K]` | ✓ (mean only) |
| **upper_limit** | Dirichlet | `upper_mean_normalized` (95th %ile) | Weak concentration | `[T, K]` | ✓ (mean only) |
| **Vmax_sum** | Deterministic | `upper_limit - A` (clamped ≥ 0) | - | `[T, K]` | ✓ (derived) |
| **K_a** | Log-Normal | `K_mean / 2` | CV-based (data-driven) | `[T, K-1]` | ✓ (mean & var) |
| **K_b** | Log-Normal | `K_mean / 2` | CV-based (data-driven) | `[T, K-1]` | ✓ (mean & var) |
| **n_a** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_a ~ Exponential(1/2)` | `[K-1, T]` | Partially |
| **n_b** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_b ~ Exponential(1/2)` | `[K-1, T]` | Partially |
| **alpha** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` | No |
| **beta** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` | No |

**Simplified Dirichlet priors (NEW)**:
```python
# Weak concentration: mean_normalized * K gives ~1 per category
K_dim = A_mean_tensor.shape[-1]  # Number of categories

# Normalize means to sum to 1
A_mean_clamped = A_mean_tensor.clamp(min=epsilon, max=1.0 - epsilon)
A_mean_normalized = A_mean_clamped / A_mean_clamped.sum(dim=-1, keepdim=True)  # [T, K]

Vmax_clamped = Vmax_mean_tensor.clamp(min=epsilon, max=1.0 - epsilon)
upper_mean_normalized = Vmax_clamped / Vmax_clamped.sum(dim=-1, keepdim=True)  # [T, K]

# Weak concentration: mean_normalized * K gives ~1 per category
concentration_A = A_mean_normalized * K_dim  # [T, K]
concentration_upper = upper_mean_normalized * K_dim  # [T, K]

A ~ Dirichlet(concentration_A)
upper_limit ~ Dirichlet(concentration_upper)
```

**Why this works**:
- For Dirichlet with `concentration = [c₁, c₂, ..., cₖ]`, the mean is `μₖ = cₖ / Σcⱼ`
- Setting `concentration_A = A_mean_normalized * K` gives:
  - `Σcⱼ = K` (total concentration)
  - `μₖ = (A_mean_normalized_k * K) / K = A_mean_normalized_k` (correct mean)
  - Each category gets concentration ≈ 1 (very weak prior)

**Notes**:
- **NEW**: Simplified Dirichlet with weak concentration (mean * K)
- **K-1 structure**: Only K-1 Hill functions are fit, Kth category is residual `1 - sum(K-1)`
- Dirichlet naturally ensures sum = 1 across all K categories (no normalization needed)
- `n_a`, `K_a`, `n_b`, `K_b` are all `[K-1]` dimensional (Kth has no Hill function)
- Hills computed with `Vmax=1`, then scaled by `Vmax_sum_k` for each category
- Sum of K-1 probabilities is clamped to `< 1-epsilon` to ensure valid residual
- **NEW**: K uses unified Log-Normal with CV-based variance

---

### 4. **Normal** (Continuous, Unbounded)

| Parameter | Prior Distribution | Mean | Variance | Shape |
|-----------|-------------------|------|----------|-------|
| **A** | Normal | `Amean_adjusted = (1-weight)*Amean + weight*Vmax_mean` | `abs(Amean_adjusted)` | `[T]` |
| **Vmax_a** | Log-Normal | `Vmax_mean` | Raw variance (data-driven) | `[T]` |
| **Vmax_b** | Log-Normal | `Vmax_mean` | Raw variance (data-driven) | `[T]` |
| **K_a** | Log-Normal | `K_max / 2` | CV-based (data-driven) | `[T]` |
| **K_b** | Log-Normal | `K_max / 2` | CV-based (data-driven) | `[T]` |
| **n_a** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_a ~ Exponential(1/2)` | `[T]` |
| **n_b** | Normal (raw) | `n_mu` (default: 0) | `sigma_n_b ~ Exponential(1/2)` | `[T]` |
| **alpha** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` |
| **beta** | RelaxedBernoulli | - | `temperature=1.0→0.1` | `[T]` |

**Notes**:
- A can be negative (uses Normal instead of Exponential/Beta)
- `y_dose_response = A + alpha * Hill_a + beta * Hill_b` (can be negative!)
- **NEW**: Vmax and K use Log-Normal (same as negbinom)
- **NEW**: K variance is CV-based (works with or without guides)

---

### 5. **Student's t** (Heavy-Tailed Continuous)

**Identical priors to Normal**, with additional degrees of freedom parameter:

| Parameter | Prior Distribution | Mean | Variance | Shape |
|-----------|-------------------|------|----------|-------|
| **nu_y** | Gamma | ~5 | - | `[T]` |

- `nu_y ~ Gamma(10.0, 2.0)` ensures df > 2 for valid variance
- Alternative: Fixed `nu_y = 3.0` (simpler, faster)
- **NEW**: Uses same Log-Normal Vmax and K as Normal

---

## Hyperparameters: K_max and x_true_CV

### K_max (Data-Driven)

**Computation (with guides)**:
```python
# K_max is the maximum of guide-specific means of x_true (cis expression)
guide_x_means = [mean(x_true[guides_tensor == g]) for g in unique_guides]
K_max_tensor = max(guide_x_means)
```

**Computation (without guides)**:
```python
# K_max is the overall maximum of x_true
K_max_tensor = x_true_mean.max()
```

**Usage**: Prior mean for K is `K_max / 2` for **all distributions**

**Interpretation**: K represents the cis expression level at which the trans effect is half-maximal. Using `K_max / 2` as the prior mean centers K in the middle of the observed cis expression range.

---

### x_true_CV (Data-Driven, NEW)

**Computation (works with or without guides)**:
```python
# Coefficient of variation (scale-invariant measure of variability)
x_true_mean_global = x_true_mean.mean()
x_true_std_global = x_true_mean.std()
x_true_CV = x_true_std_global / x_true_mean_global
```

**Usage**: Controls the variance of K prior for **all distributions**:
```python
K_mean_prior = (K_max / 2.0).clamp_min(epsilon)
K_std_prior = K_mean_prior * x_true_CV  # CV-based variance
```

**Why CV instead of raw variance?**
- CV is **scale-invariant**: works regardless of expression magnitude
- Works **with or without guides**: computed from global x_true statistics
- More **robust**: doesn't require guide structure
- **Interpretable**: CV = 0.5 means K varies by ±50% of its mean

---

## Summary Table: Data-Driven Status

| Parameter | Negbinom | Binomial | Multinomial | Normal | Student-t |
|-----------|----------|----------|-------------|--------|-----------|
| **A mean** | ✓ (5th %ile) | ✓ (5th %ile) | ✓ (5th %ile) | ✓ (5th %ile) | ✓ (5th %ile) |
| **A variance** | ✗ (fixed) | ✗ (weak α=1) | ✗ (weak conc) | ✗ (fixed) | ✗ (fixed) |
| **Vmax mean** | ✓ (95th %ile) | ✓ (95th %ile, ≤1-1e-6) | ✓ (95th %ile) | ✓ (95th %ile) | ✓ (95th %ile) |
| **Vmax variance** | ✓ (raw var) | ✗ (weak β=1) | ✗ (weak conc) | ✓ (raw var) | ✓ (raw var) |
| **K mean** | ✓ (`K_max/2`) | ✓ (`K_max/2`) | ✓ (`K_max/2`) | ✓ (`K_max/2`) | ✓ (`K_max/2`) |
| **K variance** | ✓ (CV-based) | ✓ (CV-based) | ✓ (CV-based) | ✓ (CV-based) | ✓ (CV-based) |
| **n mean** | Partially | Partially | Partially | Partially | Partially |
| **n variance** | ✗ (hierarchical) | ✗ (hierarchical) | ✗ (hierarchical) | ✗ (hierarchical) | ✗ (hierarchical) |
| **K_max** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **x_true_CV** | ✓ | ✓ | ✓ | ✓ | ✓ |

**Key**:
- ✓: Fully data-driven (computed from observed data)
- ✓ (`value`): Data-driven with shown computation
- ✗ (`param`): Uses fixed or weak prior with minimal assumptions
- ✗ (hierarchical): Uses global hyperprior (`sigma_n ~ Exponential(1/2)`)
- Partially: Mean is fixed (default 0), variance is hierarchical

---

## Key Insights

### Why Percentiles Instead of Min/Max? (NEW)

**Previous approach**: Used `min(guide_means)` and `max(guide_means)` for A and Vmax

**NEW approach**: Uses 5th and 95th percentiles

**Benefits**:
1. **Robustness to outliers**: Extreme values don't dominate priors
2. **Better typical range**: Represents the range where 90% of guides fall
3. **Stability**: Less sensitive to noise in small-sample guides
4. **Constraint**: A_mean ≥ 1e-3 ensures numerical stability

**Fallback (without guides)**:
- Uses overall 5th/95th percentiles instead of guide-level
- Same robustness benefits
- Enables running on NTC-only or single-guide data

---

### Why Simplified Beta/Dirichlet Priors? (NEW)

**For binomial (Beta)**:

**Previous approach**: Complex concentration parameters derived from within-guide variance

**NEW approach**:
- `A ~ Beta(α=1, β)` with `β = (1-A_mean)/A_mean`
  - **α=1** creates weak prior biased toward 0 (baseline should be low)
  - Mean is correctly set to A_mean
- `upper_limit ~ Beta(α, β=1)` with `α = Vmax_mean/(1-Vmax_mean)`
  - **β=1** creates weak prior biased toward 1 (upper limit should be high)
  - Mean is correctly set to Vmax_mean

**Benefits**:
1. **Simpler**: Only uses means, no variance calculation
2. **Interpretable**: α=1 or β=1 has clear directional bias
3. **Robust**: Works when variance estimates are unreliable
4. **Correct means**: Mathematical guarantee that priors match observed data

**For multinomial (Dirichlet)**:

**Previous approach**: Per-category concentration from within-guide variance

**NEW approach**:
- `concentration = mean_normalized * K` where K is number of categories
- Gives ~1 per category (very weak prior)
- Mean is correctly set to observed proportions

**Benefits**:
1. **Minimal assumptions**: Weak prior lets data dominate
2. **Correct means**: Proportions match observed data
3. **Simpler**: No variance-to-concentration conversion
4. **Stable**: Works even with sparse multinomial data

---

### Why Log-Normal for Vmax and K? (NEW)

**Previous approach**: Gamma distribution for negbinom/normal, various for binomial/multinomial

**NEW approach**: **Unified Log-Normal for ALL distributions**

**Benefits**:
1. **AutoNormal Guide compatibility**: Log-Normal is more stable for variational inference
2. **Unified code**: Same parameterization across all distributions
3. **Natural positivity**: Log-Normal ensures Vmax, K > 0 without explicit constraints
4. **Data-driven variance**: Easy to incorporate observed variance via log-space

**Parameterization**:
```python
# For Log-Normal: log(X) ~ Normal(μ, σ)
# Given mean m and std s:
ratio = s / m
log_sigma = sqrt(log(1 + ratio^2))
log_mu = log(m) - 0.5 * log_sigma^2
```

---

### Why CV for K Variance? (NEW)

**Previous approach**: Between-guide variance of x_true means

**NEW approach**: **Coefficient of Variation (CV)** of x_true

**Computation**:
```python
x_true_CV = std(x_true) / mean(x_true)
K_std_prior = K_mean_prior * x_true_CV
```

**Benefits**:
1. **Scale-invariant**: Works regardless of expression magnitude (log2 vs raw counts)
2. **Works without guides**: Uses global statistics instead of guide structure
3. **Interpretable**: CV=0.5 means "K varies by ±50% around its mean"
4. **Robust**: Single statistic captures relative variability

**Raw variance for A/Vmax**:
- A and Vmax represent absolute expression levels (not relative)
- Use raw variance (`mean_within_guide_var`) to preserve scale information
- For negbinom/normal/studentt: directly translates to Log-Normal variance

---

### Hierarchical n Prior

The Hill coefficient `n` uses a **hierarchical prior** with a soft-clamping transformation:
```
sigma_n_a  ~ Exponential(1/2)                          # Global variance (shared across all T genes)
n_mu_raw   = half_n * atanh((0 - center_n) / half_n)  # Inverse soft_clamp of n_mu=0
n_a_raw    ~ Normal(n_mu_raw, sigma_n_a)               # Per-gene unconstrained value
n_a        = soft_clamp(n_a_raw, nmin, nmax)           # Constrained to physically safe range
```

where `center_n = (nmin + nmax) / 2` and `half_n = (nmax - nmin) / 2`.

The `n_mu_raw` correction ensures the **constrained prior mode is exactly at n_mu = 0** regardless of the asymmetry of [nmin, nmax]. Without it, `Normal(0, sigma)` on n_a_raw has its constrained mode at center_n, which can be strongly negative when nmin << nmax (e.g., all-CA data with nmin = −100, nmax ≈ 38, center ≈ −31).

This allows:
- Global learning of typical variability in Hill coefficients across all T genes
- Per-gene flexibility while sharing information
- The sparsity prior on `alpha` / `beta` gates whether a component has any effect at all; `n_a` controls the shape conditional on the component being active

---

## Implementation Details

### nanquantile Helper (NEW)

```python
def nanquantile(x, q, dim):
    """Compute quantile ignoring NaN values."""
    if x.ndim == 2:  # [G, T]
        result = []
        for t in range(x.shape[1]):
            vals = x[:, t]
            valid = vals[~torch.isnan(vals)]
            if valid.numel() > 0:
                result.append(torch.quantile(valid, q))
            else:
                result.append(torch.tensor(float('nan'), device=x.device))
        return torch.stack(result)

    elif x.ndim == 3:  # [G, T, K]
        result = []
        for t in range(x.shape[1]):
            result_t = []
            for k in range(x.shape[2]):
                vals = x[:, t, k]
                valid = vals[~torch.isnan(vals)]
                if valid.numel() > 0:
                    result_t.append(torch.quantile(valid, q))
                else:
                    result_t.append(torch.tensor(float('nan'), device=x.device))
            result.append(torch.stack(result_t))
        return torch.stack(result)
```

### NaN Handling Fallback

```python
# After computing A_mean and Vmax_mean, check for NaN
nan_mask = torch.isnan(A_mean_tensor) | torch.isnan(Vmax_mean_tensor)

if nan_mask.any():
    # Some features have all observations filtered (e.g., denominator < 3)
    # Use median of valid values as fallback
    valid_means = guide_means[~torch.isnan(guide_means)]

    if valid_means.numel() > 0:
        valid_mean = torch.median(valid_means)
        fallback_A = torch.clamp(valid_mean * 0.5, min=0.01, max=0.99)
        fallback_Vmax = torch.clamp(valid_mean * 1.5, min=0.01, max=0.99)
    else:
        # Last resort: generic defaults
        fallback_A = 0.3
        fallback_Vmax = 0.7

    # Replace NaN with fallback
    A_mean_tensor = torch.where(torch.isnan(A_mean_tensor), fallback_A, A_mean_tensor)
    Vmax_mean_tensor = torch.where(torch.isnan(Vmax_mean_tensor), fallback_Vmax, Vmax_mean_tensor)
```

### Log-Normal Parameterization (Unified)

```python
def log_normal_params(mean, std):
    """Convert mean and std to Log-Normal(μ, σ) parameters."""
    epsilon = 1e-6
    ratio = (std / mean).clamp_min(epsilon)
    log_sigma = torch.sqrt(torch.log1p(ratio ** 2))
    log_mu = torch.log(mean) - 0.5 * log_sigma ** 2
    return log_mu, log_sigma

# For Vmax (negbinom/normal/studentt):
Vmax_log_mu, Vmax_log_sigma = log_normal_params(Vmax_mean_prior, Vmax_std_prior)
log_Vmax_a = pyro.sample("log_Vmax_a", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
Vmax_a = pyro.deterministic("Vmax_a", torch.exp(log_Vmax_a))

# For K (all distributions):
K_log_mu, K_log_sigma = log_normal_params(K_mean_prior, K_std_prior)
log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))
K_a = pyro.deterministic("K_a", torch.exp(log_K_a))
```

---

## Future Improvements

1. **Per-feature n hyperpriors**: Allow `sigma_n` to vary by feature (currently global)
2. **Adaptive epsilon/clamping**: Data-driven minimum values instead of fixed 1e-3, 1e-6
3. **Separate A and upper_limit variances**: Currently use same `mean_within_guide_var` for both (could compute separately)
4. **Adaptive weak priors**: Instead of fixed α=1 or β=1, could use data-driven weak concentration (e.g., 0.5 to 2.0)
5. **Alternative CV definitions**: Could use median absolute deviation (MAD) for even more robustness

---

## References

### Code Location

- **File**: `bayesDREAM/fitting/trans.py`
- **Method**: `_model_y()` (lines 74-565)
- **Prior computation**: Lines 1087-1283 in `fit_trans()`
- **Percentile computation**: Lines 1116-1224 (nanquantile helper)
- **K and CV computation**: Lines 1267-1283
- **Binomial Beta priors**: Lines 199-217 in `_model_y()`
- **Multinomial Dirichlet priors**: Lines 175-197 in `_model_y()`
- **Unified Log-Normal K priors**: Lines 291-351 in `_model_y()`
- **Log-Normal Vmax priors**: Lines 324-347 in `_model_y()`

### Related Documentation

- `MEMORY_AUTO_DETECTION.md`: Automatic memory-aware batching
- `REFACTORING_SUMMARY.md`: Code organization details
- `ARCHITECTURE.md`: Overall system design
- `INITIALIZATION.md`: Empirical Bayes initialization strategies

---

## Changelog

### 2025-01-26: Comprehensive Prior Refactoring

**Major changes**:
1. **Percentiles instead of min/max**: A_mean and Vmax_mean now use 5th and 95th percentiles
2. **Without-guides fallback**: Code works when no guides are available
3. **Simplified Beta priors**: Binomial uses α=1 (A) and β=1 (upper_limit) for weak directional priors
4. **Simplified Dirichlet priors**: Multinomial uses weak concentration (mean * K)
5. **Unified Log-Normal for K**: All distributions use same Log-Normal parameterization with CV-based variance
6. **Log-Normal for Vmax**: Negbinom/normal/studentt switched from Gamma to Log-Normal
7. **CV-based K variance**: Coefficient of variation replaces between-guide variance
8. **Raw variance for A/Vmax**: Within-guide variance used directly (not CV)
9. **Constraints**: A_mean ≥ 1e-3, binomial Vmax_mean ≤ 1-1e-6
10. **x_true_CV parameter**: New parameter passed to _model_y() for K prior variance
