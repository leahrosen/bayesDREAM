# NTC Fitting Initialization Guide

## Overview

This document describes the empirical Bayes initialization strategies used in bayesDREAM's NTC fitting step (`fit_ntc`) for removing batch effects. The initialization approach varies by distribution type (negative binomial, binomial, multinomial) but follows a consistent philosophy: **use reference group data to set baseline priors and initialize correction parameters based on empirical group differences**.

## Key Concepts

### Reference Group

The **reference group** (group 0) represents the baseline technical condition. All other technical groups are modeled as corrections relative to this baseline. When computing prior hyperparameters and guide initialization values, we preferentially use only the reference group data to avoid diluting the baseline signal with data from other conditions.

### Initialization Components

For each distribution, we initialize two key components:

1. **Baseline Prior Hyperparameters**: Parameters of the prior distribution for the baseline mean/probability (e.g., Gamma shape/rate for negbinom, Beta concentration for binomial, Dirichlet concentration for multinomial)

2. **Technical Correction Parameters** (`log2_alpha_y`): Group-specific additive or multiplicative corrections that model how each non-reference group differs from the baseline

### Transformation Scales

Different distributions require different transformation scales for modeling corrections:

- **Negative binomial**: log2-ratio scale (multiplicative effects on mean expression)
- **Binomial**: logit-difference scale (additive effects on log-odds)
- **Multinomial**: log-ratio scale (additive effects on log-probabilities)

## Distribution-Specific Details

### Summary Table

| Distribution | Baseline Prior | Data Source | Guide Type | Correction Init | Transformation |
|-------------|----------------|-------------|------------|-----------------|----------------|
| **negbinom** | Gamma(shape, rate) | ✅ Ref group | AutoIAFNormal | ✅ Ref group | log2-ratio |
| **binomial** | Beta(a, b) | ✅ Ref group | AutoNormal | ✅ Ref group | logit-difference |
| **multinomial** | Dirichlet(α) | ✅ Ref group | AutoDelta + AutoNormal | ✅ Ref group | log-ratio |

### 1. Negative Binomial (Gene Expression Counts)

**Use Case**: Modeling gene expression counts, transcript counts, or other count data.

**Baseline Prior** (`bayesDREAM/fitting/ntc.py:196-239`):

```python
# Compute from reference group (group 0) only
ref_mask = (groups_ntc_tensor == 0)

if ref_mask.sum() > 0:
    y_sum_ref = y_obs_ntc_tensor[ref_mask, :].sum(dim=0).float()    # [T]
    sf_sum_ref = sum_factors_ntc_tensor[ref_mask].sum().float()     # scalar
else:
    # Fallback to all data if no reference group
    y_sum_ref = y_obs_ntc_tensor.sum(dim=0).float()
    sf_sum_ref = sum_factors_ntc_tensor.sum().float()

# Method-of-moments estimates for Gamma prior
mu_hat = (y_sum_ref + 1.0) / (sf_sum_ref + 1e-6)  # [T]
cv = 0.1  # assumed coefficient of variation

gamma_shape = 1.0 / (cv**2)
gamma_rate = gamma_shape / mu_hat  # [T]

with f_plate:
    mu_ntc = pyro.sample("mu_ntc", dist.Gamma(gamma_shape, gamma_rate))
```

**Why Reference Group**: The Gamma hyperparameters set the baseline expectation for gene expression. Using only the reference group ensures this baseline accurately reflects the "standard" condition rather than an average across different batch effects.

**Batch Correction Initialization** (`bayesDREAM/fitting/ntc.py:307-338`):

```python
# Initialize with log2-ratio differences between groups
group_codes_np = groups_ntc_tensor.cpu().numpy()
y_obs_np = y_obs_ntc_tensor.cpu().numpy()
sf_np = sum_factors_ntc_tensor.cpu().numpy()

# Reference group mean (log2 scale)
ref_mask = (group_codes_np == 0)
if ref_mask.sum() > 0:
    ref_mean = (y_obs_np[ref_mask, :].sum(axis=0) + 1.0) / (sf_np[ref_mask].sum() + 1e-6)
else:
    ref_mean = (y_obs_np.sum(axis=0) + 1.0) / (sf_np.sum() + 1e-6)

ref_mean_log2 = np.log2(np.maximum(ref_mean, 1e-6))

# Initialize alpha for each non-reference group
init_values = []
non_ref_groups = sorted(set(group_codes_np) - {0})

for g in non_ref_groups[:C-1]:
    group_mask = (group_codes_np == g)
    if group_mask.sum() > 0:
        group_mean = (y_obs_np[group_mask, :].sum(axis=0) + 1.0) / (sf_np[group_mask].sum() + 1e-6)
        group_mean_log2 = np.log2(np.maximum(group_mean, 1e-6))

        # log2-ratio difference: log2(mean_group) - log2(mean_ref)
        init_values.append(group_mean_log2 - ref_mean_log2)
    else:
        init_values.append(np.zeros(T_local, dtype=np.float32))

init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_local), dtype=np.float32)
return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)
```

**Why log2-ratio**: For count data, batch effects are often multiplicative (e.g., 2x higher expression in one batch). The log2 scale converts multiplicative effects to additive ones, making them easier to model. A log2 difference of 1.0 means a 2-fold change.

**Guide**: AutoIAFNormal (Inverse Autoregressive Flow with Normal base) provides flexible posterior approximation for the Student-T prior on `log2_alpha_y`.

---

### 2. Binomial (Splicing PSI, Proportions)

**Use Case**: Modeling proportions with denominators, such as splicing percent-spliced-in (PSI), raw splice junction counts, or allelic ratios.

**Baseline Prior** (`bayesDREAM/fitting/ntc.py:240-268`):

```python
# Compute PSI from reference group (group 0) only
ref_mask = (groups_ntc_tensor == 0)

if ref_mask.sum() > 0:
    y_sum_ref = y_obs_ntc_tensor[ref_mask, :].sum(dim=0).float()       # [T]
    den_sum_ref = denominator_ntc_tensor[ref_mask, :].sum(dim=0).float()  # [T]
else:
    # Fallback to all data if no reference group
    y_sum_ref = y_obs_ntc_tensor.sum(dim=0).float()
    den_sum_ref = denominator_ntc_tensor.sum(dim=0).float()

# Smooth p-hat with pseudocounts
p_hat = (y_sum_ref + 0.5) / (den_sum_ref + 1.0)  # [T] in (0,1)
p_hat = torch.clamp(p_hat, 1e-6, 1 - 1e-6)

# Cap effective sample size for stable but not razor-sharp prior
kappa = torch.clamp(den_sum_ref, min=20.0, max=200.0)

# Beta concentration parameters
a = p_hat * kappa + 1e-3
b = (1.0 - p_hat) * kappa + 1e-3

with f_plate:
    mu_ntc = pyro.sample("mu_ntc", dist.Beta(a, b))  # [T]
```

**Why Reference Group**: The Beta prior represents our belief about the baseline PSI. Using the reference group ensures the prior centers on the "true" PSI in the standard condition, not a weighted average across different batch effects.

**Why Bounded kappa**: The effective sample size `kappa` controls how informative the prior is. We bound it between 20 and 200 to provide reasonable informativeness without overfitting to the observed data. Too low would make the prior uninformative; too high would make it inflexible.

**Batch Correction Initialization** (`bayesDREAM/fitting/ntc.py:354-394`):

```python
# Initialize with logit-scale differences between groups
group_codes_np = groups_ntc_tensor.cpu().numpy()
y_obs_np = y_obs_ntc_tensor.cpu().numpy()      # [N, T]
denom_np = denominator_ntc_tensor.cpu().numpy()  # [N, T]

epsilon = 1e-6

# Reference group PSI (logit scale)
ref_mask = (group_codes_np == 0)
if ref_mask.sum() > 0:
    ref_numer = y_obs_np[ref_mask, :].sum(axis=0) + 0.5  # [T]
    ref_denom = denom_np[ref_mask, :].sum(axis=0) + 1.0   # [T]
    ref_psi = np.clip(ref_numer / ref_denom, epsilon, 1 - epsilon)
else:
    ref_psi = np.ones(T_local) * 0.5

ref_logit = np.log(ref_psi / (1 - ref_psi))

# Initialize alpha for each non-reference group
init_values = []
non_ref_groups = sorted(set(group_codes_np) - {0})

for g in non_ref_groups[:C-1]:
    group_mask = (group_codes_np == g)
    if group_mask.sum() > 0:
        group_numer = y_obs_np[group_mask, :].sum(axis=0) + 0.5  # [T]
        group_denom = denom_np[group_mask, :].sum(axis=0) + 1.0   # [T]
        group_psi = np.clip(group_numer / group_denom, epsilon, 1 - epsilon)

        group_logit = np.log(group_psi / (1 - group_psi))

        # Logit-difference: logit(PSI_group) - logit(PSI_ref)
        init_values.append(group_logit - ref_logit)
    else:
        init_values.append(np.zeros(T_local, dtype=np.float32))

init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_local), dtype=np.float32)
return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)
```

**Why logit-difference**: The logit transformation maps probabilities from [0,1] to (-∞, +∞), making them easier to model with additive effects. A logit difference of 0.5 represents a modest shift in PSI; 1.0 represents a larger shift. The logit scale is more natural for proportion data than log-ratio because it respects the bounded nature of probabilities.

**Guide**: AutoNormal (Gaussian) provides a simple but effective posterior approximation for the Student-T prior on `log2_alpha_y`.

---

### 3. Multinomial (Isoform Usage, Donor/Acceptor Usage)

**Use Case**: Modeling categorical outcomes where each feature has multiple mutually exclusive categories, such as transcript isoform usage, donor site usage, or acceptor site usage.

**Baseline Prior** (`bayesDREAM/fitting/ntc.py:85-98, 273`):

```python
# Step 1: Identify structurally absent categories from ALL NTC data
# (We need all data to know which categories are impossible)
total_counts_per_feature = y_obs_ntc_tensor.sum(dim=0)  # [T, K]
zero_cat_mask = (total_counts_per_feature == 0)         # [T, K] bool
pyro.deterministic("zero_cat_mask", zero_cat_mask)

# Step 2: Compute counts from reference group (group 0) for Dirichlet prior
ref_mask = (groups_ntc_tensor == 0)
if ref_mask.sum() > 0:
    # Use reference group only for accurate baseline
    total_counts_ref = y_obs_ntc_tensor[ref_mask, :, :].sum(dim=0)  # [T, K]
else:
    # Fallback to all data if no reference group
    total_counts_ref = total_counts_per_feature  # [T, K]

# Later in the model (line 273):
concentration = total_counts_ref + 1.0  # [T, K], strictly > 0

with f_plate:
    probs0 = pyro.sample("probs_baseline_raw", dist.Dirichlet(concentration))  # [T, K]
```

**Why Two Separate Computations**:
- `zero_cat_mask` uses **all NTC data** to identify categories that are structurally absent (zero counts across all samples and groups). This ensures we correctly mask impossible outcomes.
- `concentration` uses **reference group only** to set the baseline category probabilities. This ensures the Dirichlet prior accurately reflects the reference condition's category usage pattern.

**Why Reference Group for Dirichlet**: The Dirichlet concentration parameter determines the expected proportion of each category. Using the reference group ensures these baseline proportions match the standard condition, not an average across batch effects that might shift category usage.

**Batch Correction Initialization** (`bayesDREAM/fitting/ntc.py:421-465`):

```python
# Initialize with log-ratio differences between groups
group_codes_np = groups_ntc_tensor.cpu().numpy()
y_obs_np = y_obs_ntc_tensor.cpu().numpy()  # [N, T, K]

# Reference group probabilities (log scale)
ref_mask = (group_codes_np == 0)
if ref_mask.sum() > 0:
    ref_counts = y_obs_np[ref_mask, :, :].sum(axis=0) + 1.0  # [T, K]
else:
    ref_counts = y_obs_np.sum(axis=0) + 1.0

ref_probs = ref_counts / ref_counts.sum(axis=1, keepdims=True)  # [T, K]
ref_log_probs = np.log(np.maximum(ref_probs, 1e-10))  # [T, K]

# Initialize alpha for each non-reference group
init_values = []
non_ref_groups = sorted(set(group_codes_np) - {0})

for g in non_ref_groups[:C-1]:
    group_mask = (group_codes_np == g)
    if group_mask.sum() > 0:
        group_counts = y_obs_np[group_mask, :, :].sum(axis=0) + 1.0  # [T, K]
        group_probs = group_counts / group_counts.sum(axis=1, keepdims=True)  # [T, K]
        group_log_probs = np.log(np.maximum(group_probs, 1e-10))  # [T, K]

        # log-ratio difference: log(P_group) - log(P_ref)
        # This is centered by subtracting the mean across categories
        diff = group_log_probs - ref_log_probs  # [T, K]
        diff_centered = diff - diff.mean(axis=1, keepdims=True)  # [T, K]
        init_values.append(diff_centered)
    else:
        init_values.append(np.zeros((T_local, K), dtype=np.float32))

init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_local, K), dtype=np.float32)
return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)
```

**Why log-ratio with centering**: For multinomial data, batch effects shift category probabilities. Working in log-space makes these shifts additive. Centering ensures the corrections sum to zero across categories (maintaining the constraint that probabilities must sum to 1).

**Why Centering is Crucial**: Since probabilities must sum to 1, we cannot change all categories independently. Centering the log-ratio differences ensures that the corrections "balance out" - if one category increases, others must decrease proportionally.

**Guide**: Combination of AutoDelta (point-mass) for the baseline probabilities and AutoNormal (Gaussian) for the batch corrections. This reflects that we're very confident about the baseline but uncertain about the corrections.

**Category Masking**: After sampling category probabilities, we apply `zero_cat_mask` twice:
1. After initial sampling (line ~280)
2. After centering the group corrections (line ~290)

This ensures that categories with zero counts remain at exactly zero probability, preventing the model from assigning non-zero probability to impossible outcomes.

---

## Common Patterns

### 1. Reference Group Fallback

All three distributions include fallback logic if no reference group exists:

```python
ref_mask = (groups_ntc_tensor == 0)
if ref_mask.sum() > 0:
    # Use reference group only
    ref_data = data[ref_mask, ...]
else:
    # Fallback to all data
    ref_data = data
```

This ensures the code works even if the user doesn't have a properly coded reference group, though results will be less accurate.

### 2. Pseudocounts for Stability

All computations add small pseudocounts to avoid division by zero and extreme parameter values:

- Negative binomial: `+ 1.0` for counts and sum factors
- Binomial: `+ 0.5` for numerator, `+ 1.0` for denominator
- Multinomial: `+ 1.0` for counts

### 3. Zero-Centered Priors, Non-Zero Guide Initialization

The Pyro models use **zero-centered priors** for technical corrections:
```python
log2_alpha_y ~ StudentT(df=3, loc=0, scale=20)
```

This reflects prior uncertainty: we don't know a priori which direction or magnitude the batch effects will have.

However, the **guide initialization** uses empirical group differences, giving the variational inference a good starting point near the true posterior.

### 4. Student-T Priors for Robustness

All corrections use Student-T priors with `df=3`, which have heavier tails than Gaussian priors. This makes the model more robust to outlier features with unusually large batch effects.

---

## Implementation Notes

### Location in Codebase

- **Main implementation**: `bayesDREAM/fitting/ntc.py`
- **Model definition**: `_model_ntc` method (lines ~70-300)
- **Guide initialization**: `init_loc_fn` function (lines ~305-470)
- **Fitting interface**: `fit_ntc` method (lines ~475-600)

### Guide Selection

The guide type is automatically selected based on distribution:

```python
if distribution == 'multinomial':
    guide_class = pyro.infer.autoguide.AutoDelta
else:  # negbinom or binomial
    guide_class = pyro.infer.autoguide.AutoNormal

# For negbinom with many features, use IAF for more flexibility
if distribution == 'negbinom' and n_features > 1000:
    guide_class = pyro.infer.autoguide.AutoIAFNormal
```

### Optimization Settings

Different distributions use different optimization strategies:

- **Negative binomial**: Adam with learning rate 0.01, 5000 steps
- **Binomial**: Adam with learning rate 0.01, 3000 steps
- **Multinomial**: Adam with learning rate 0.01, 5000 steps

These settings are tuned for typical dataset sizes but can be adjusted via `fit_ntc` parameters.

---

## Usage Examples

### Example 1: Gene Expression (Negative Binomial)

```python
from bayesDREAM import bayesDREAM

# Load data
meta = pd.read_csv('cell_meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line'],
    output_dir='./output'
)

# Set technical groups (must be called before fit_ntc)
# Group 0 will be the reference
model.set_technical_groups(['cell_line'])

# Fit technical model (uses reference group for initialization)
model.fit_ntc(
    sum_factor_col='sum_factor',
    num_steps=5000,
    learning_rate=0.01
)

# Access corrected parameters
alpha_y_prefit = model.get_modality('gene').alpha_y_prefit  # [N_samples, N_groups, N_features]
```

### Example 2: Splicing PSI (Binomial)

```python
# Add splicing modality (automatically creates binomial distribution for raw SJs)
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj']  # Raw splice junction counts as binomial
)

# Set technical groups
model.set_technical_groups(['cell_line'])

# Fit technical model for splicing modality
# Note: Currently only 'gene' modality supports fit_ntc
# For other modalities, alpha_y must be set manually
```

### Example 3: Isoform Usage (Multinomial)

```python
# Add transcript modality with isoform usage
model.add_transcript_modality(
    transcript_counts=tx_counts,
    transcript_meta=tx_meta,
    modality_types=['usage']  # Creates multinomial distribution
)

# Set technical groups
model.set_technical_groups(['cell_line'])

# Fit technical model for transcript usage
# Note: Currently only 'gene' modality supports fit_ntc
# For other modalities, alpha_y must be set manually
```

---

## Benefits of Reference Group Initialization

1. **More Accurate Baselines**: Prior hyperparameters reflect the true baseline condition, not a mixture across different batch effects.

2. **Faster Convergence**: Starting the guide near empirical group differences means fewer SVI steps are needed to reach the posterior.

3. **Better Separation**: By centering on the reference group, technical corrections more clearly represent deviations from the standard condition.

4. **Improved Interpretability**: Effects are relative to a meaningful reference, making them easier to interpret than arbitrary averages.

5. **Robust to Imbalanced Groups**: Even if non-reference groups have very different sample sizes, the baseline accurately represents the reference condition.

---

## Troubleshooting

### Problem: No convergence / high loss

**Solution**: Increase `num_steps` or adjust `learning_rate`. For difficult datasets, try:
```python
model.fit_ntc(num_steps=10000, learning_rate=0.005)
```

### Problem: Correction parameters are all near zero

**Possible causes**:
- Technical groups are not properly coded (reference should be group 0)
- Not enough samples per group for stable empirical estimates
- Batch effects are genuinely very small

**Diagnosis**:
```python
# Check group coding
print(model.meta_ntc['technical_group_code'].value_counts())

# Check group sample sizes
print(model.meta_ntc.groupby('technical_group_code').size())
```

### Problem: Category masking not working (multinomial)

**Solution**: Verify that `zero_cat_mask` is computed from all NTC data:
```python
# After fitting, check the mask
gene_mod = model.get_modality('gene')
print(gene_mod.zero_cat_mask.sum(dim=1))  # Should show number of masked categories per feature
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): Overall system architecture
- [API_REFERENCE.md](API_REFERENCE.md): Detailed API documentation
- [QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md): Multi-modal usage examples
- [DATA_ACCESS.md](DATA_ACCESS.md): Accessing fitted parameters and posteriors

---

## Version History

**2025-11-04**:
- Added reference group initialization for binomial Beta prior
- Added logit-scale initialization for binomial log2_alpha_y
- Fixed multinomial Dirichlet to use reference group (separated from zero_cat_mask computation)
- Documented all three distributions comprehensively

**2024-10**: Initial implementation of technical fitting with empirical Bayes for negative binomial

---

## Contact

For questions or issues related to initialization, please file an issue on the GitHub repository or consult the full documentation.
