# Cis Model Output Documentation

## Overview

The cis model (`fit_cis`) estimates the true expression level of a targeted gene (cis gene)
for each cell, accounting for guide-level effects and technical variation. The model uses
a hierarchical Bayesian framework with a negative binomial likelihood.

## Parameters

### `x_true` (per-cell true expression)

**Shape:** `(n_samples, n_cells)` or `(n_cells,)` for point estimate

**Meaning:** The estimated "true" expression level of the cis gene for each cell,
deconvolved from technical noise. This is on the **linear scale** (not log-transformed).

**Generative model:**
```
log2(x_true) ~ Normal(log2(x_eff_g[guide]), sigma_eff[guide])
x_true = 2^(log2(x_true))
```

Where `guide` is the guide assigned to that cell.

**Usage:** Use `x_true` as the input to the trans model (`fit_trans`) to model downstream
effects. It represents the perturbation "dose" for each cell.

---

### `x_eff_g` (per-guide mean effect)

**Shape:** `(n_samples, n_guides)` or `(n_guides,)` for point estimate

**Meaning:** The mean true expression level for cells with each guide. This captures
the average perturbation strength of each guide.

**Generative model:**
```
log2(x_eff_g) ~ Normal(mu, sigma) * eps,  where eps ~ StudentT(df=3)
x_eff_g = 2^(log2(x_eff_g))
```

Where `mu` and `sigma` are shared hyperparameters across all guides (or per-target if
`independent_mu_sigma=True`).

**Interpretation:**
- Higher `x_eff_g` = stronger knockdown/activation effect (depending on perturbation type)
- Guides targeting the same gene should have similar `x_eff_g` values
- NTC guides should have `x_eff_g` close to the baseline expression level

---

### `sigma_eff` (per-guide variability)

**Shape:** `(n_samples, n_guides)` or `(n_guides,)` for point estimate

**Meaning:** The standard deviation of log2(x_true) for cells with each guide. This
captures guide-specific heterogeneity in perturbation effect.

**Generative model:**
```
sigma_eff ~ Gamma(sigma_eff_alpha, sigma_eff_beta)
```

Where `sigma_eff_alpha` and `sigma_eff_beta` are shared hyperparameters.

**Interpretation:**
- Higher `sigma_eff` = more variable perturbation effect across cells with that guide
- Could indicate: guide inefficiency, cell-state-dependent effects, or technical issues
- Typically ranges from 0.1 to 2.0 in log2 space

---

### `alpha_x` (technical correction factor)

**Shape:** `(n_samples, C-1)` or `(C-1,)` for point estimate, where C = number of technical groups

**Meaning:** Multiplicative correction factor for each technical group (e.g., cell line,
batch) **relative to the reference group** (group 0, typically the first alphabetically).

**Generative model:**
```
alpha_x ~ Gamma(alpha_alpha, alpha_alpha/alpha_mu)
```

**How it's used in the observation model:**
```
mu_obs = alpha_x[group] * x_true * sum_factor
x_obs ~ NegativeBinomial(mu=mu_obs, phi=phi_x)
```

**Interpretation:**
- `alpha_x ≈ 1.0` means the technical group has similar expression to reference
- `alpha_x > 1.0` means higher baseline expression in that group
- `alpha_x < 1.0` means lower baseline expression in that group
- Reference group (index 0) implicitly has `alpha_x = 1.0`

**Note:** If `alpha_x ≈ 1.0` for all groups, technical correction is minimal.

---

### `o_x` (overdispersion parameter)

**Shape:** `(n_samples,)` - scalar per posterior sample

**Meaning:** Controls the overdispersion of the negative binomial distribution. Related to
the dispersion parameter `phi_x` by `phi_x = 1 / o_x^2`.

**Generative model:**
```
beta_o ~ Gamma(beta_o_alpha, beta_o_beta)  # hyperprior
o_x ~ Exponential(beta_o)                   # overdispersion
phi_x = 1 / o_x^2                           # NB dispersion parameter
```

**Two versions are exported:**

1. **`o_x_cis_posterior.csv`**: Overdispersion estimated during `fit_cis()`. This is
   estimated jointly with the guide effects and captures the noise structure when
   modeling the cis gene expression.

2. **`o_x_trans_posterior.csv`**: Overdispersion estimated during `fit_ntc()` on
   NTC cells only. This serves as a baseline estimate of technical noise without
   perturbation effects.

**Interpretation:**
- Smaller `o_x` → larger `phi_x` → less overdispersion (closer to Poisson)
- Larger `o_x` → smaller `phi_x` → more overdispersion (more variance than Poisson)
- Typical values: 0.1 to 1.0

---

## Full Observation Model

The complete generative model for observed counts is:

```
x_obs ~ NegativeBinomial(
    mu = alpha_x[technical_group] * x_true * sum_factor,
    phi = phi_x
)
```

Where:
- `x_obs`: Observed UMI counts for the cis gene
- `alpha_x`: Technical group correction (multiplicative)
- `x_true`: True expression (derived from guide effect + cell-level noise)
- `sum_factor`: Library size normalization factor
- `phi_x`: Overdispersion parameter (shared or per-group)

---

## File Descriptions

| File | Shape | Description |
|------|-------|-------------|
| `guide_meta.csv` | (n_guides, cols) | Guide metadata with mean x_eff_g and sigma_eff |
| `cell_meta.csv` | (n_cells, cols) | Cell metadata with mean x_true |
| `x_true_posterior.csv` | (n_samples, n_cells) | Full posterior samples for x_true |
| `x_eff_g_posterior.csv` | (n_samples, n_guides) | Full posterior samples for x_eff_g |
| `sigma_eff_posterior.csv` | (n_samples, n_guides) | Full posterior samples for sigma_eff |
| `o_x_cis_posterior.csv` | (n_samples,) | Overdispersion from cis fit |
| `o_x_trans_posterior.csv` | (n_samples,) | Overdispersion from technical fit (NTC only) |
| `alpha_x_posterior.csv` | (n_samples, C-1) | Technical correction factors from technical fit |

**Note:** Rows in posterior CSVs are posterior samples, columns are cells/guides.
Guide order matches `guide_code` in guide_meta (sorted by guide_code).
Technical group order: index 0 = reference group (alpha_x = 1.0, not stored), index i = group i+1.
