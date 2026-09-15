# bayesDREAM Trans Fitting Guide

This guide explains the `fit_trans()` method, which models how trans genes (or features) respond as a function of cis gene expression.

## Overview

The trans fitting step models downstream effects of perturbations by fitting dose-response functions between the cis feature (e.g., GFI1B expression) and trans features (e.g., other genes, ATAC peaks, splice junctions).

### Three-Step Pipeline

1. **fit_ntc()**: Model technical variation in NTC cells → estimate `alpha_y` (overdispersion parameters)
2. **fit_cis()**: Model direct effects on the cis feature → estimate `x_true` (true cis expression per cell)
3. **fit_trans()**: Model trans features as functions of `x_true` → discover dose-response relationships

## Distribution-Specific Implementation

bayesDREAM applies trans functions differently depending on the distribution type. This ensures the function output matches the natural space of each distribution.

### Negative Binomial (Gene/Transcript Counts, ATAC Peaks)

**Function Application:**
- **Hill functions**: Applied in count space (natural scale)
  ```
  y_count = A + alpha * Hill(x)
  ```
- **Polynomial**: Applied in log2 space to ensure multiplicative effects
  ```
  log2(y) = log2(A) + alpha * polynomial(log2(x))
  y_count = 2^(log2(y))
  ```

**Priors:**
- `A` (baseline): `Exponential(1/Amean)` → positive counts
- `Vmax` (Hill amplitude): `Gamma(mean², mean/σ²)` → positive amplitude
- `K` (EC50/IC50): `Gamma((K_max/2)², (K_max/2)/σ²)` → positive half-max point
- `n` (Hill coefficient): `Normal(0, σ_n)` clamped to `[nmin, nmax]` to avoid overflow

**Technical Correction:**
```
mu_final = y_dose_response * alpha_y[group] * sum_factor
```
Applied multiplicatively on count scale (default: no correction if `set_technical_groups()` not called).

---

### Binomial (Exon Skipping PSI, Raw SJ Counts)

**Function Application:**
- **Hill functions**: Applied in probability space [0, 1], clamped
  ```
  p = A + alpha * Hill(x)
  p = clamp(p, ε, 1-ε)
  ```
- **Polynomial**: Applied in logit space (unbounded), then sigmoid
  ```
  logit(p) = logit(A) + alpha * polynomial(x)
  p = sigmoid(logit(p))  # Maps to [0, 1]
  ```

**Priors:**
- `A` (baseline probability): `Beta(α, β)` with mean = Amean → constrained to [0, 1]
- `Vmax` (Hill amplitude): `Beta(α, β)` → probability in [0, 1]
- `K` (EC50/IC50): `Gamma((K_max/2)², (K_max/2)/σ²)` → positive half-max point
- `n` (Hill coefficient): `Normal(0, σ_n)` clamped to `[nmin, nmax]`

**Data-Driven Priors:**
For binomial distributions, observed counts (numerator) are normalized by the denominator to get probabilities [0, 1] before computing `Amean` and `Vmax_mean`. This ensures priors are in the correct probability space, preventing parameter explosions.

**Technical Correction:**
```
logit(p_final) = logit(p_baseline) + log(alpha_y[group])
```
Applied on logit scale (additive in log-odds space).

---

### Multinomial (Donor/Acceptor Usage, Isoform Proportions)

**Function Application:**
- **Hill functions**: K-1 independent Hill functions for K-1 categories, Kth is residual
  ```
  p_k = A_k + alpha * Hill_k(x)  for k=1..K-1
  p_K = 1 - sum(p_1, ..., p_{K-1})  # Residual category
  ```
  The K-1 approach avoids identifiability issues in probability space.

- **Polynomial**: K independent polynomials in logit space, softmax to get probabilities
  ```
  logit_k = logit(A_k) + alpha * polynomial_k(x)  for k=1..K
  p = softmax(logit_1, ..., logit_K)  # Sum to 1
  ```
  The K-in-logit-space approach uses softmax for normalization.

**Priors (per category k=1..K-1 for Hill, k=1..K for polynomial):**
- `A_k` (baseline probability): `Beta(α, β)` → each category in [0, 1]
- `Vmax_k` (Hill amplitude): `Beta(α, β)` → probability amplitude per category
- `K_k` (EC50): `Gamma((K_max/2)², (K_max/2)/σ²)` → per-category half-max
- `n_k` (Hill coefficient): `Normal(0, σ_n)` per category, clamped

**Data-Driven Priors:**
For multinomial distributions, observed counts [N, T, K] are normalized by the total across categories (sum over K) to get proportions [0, 1] before computing `Amean` and `Vmax_mean`. This ensures priors are in probability space.

**Technical Correction:**
```
log(p_final[k]) = log(p_baseline[k]) + alpha_y[group, k]
```
Applied on log scale for each category k, similar to binomial logit correction. Implemented in commit 59f08d9 (December 2025).

---

### Normal (SpliZ Scores, Continuous Measurements)

**Function Application:**
- **Hill functions**: Applied in natural value space (can be negative)
  ```
  y = A + alpha * Hill(x)  # No clamping
  ```
- **Polynomial**: Applied directly in natural space
  ```
  y = A + alpha * polynomial(x)  # Can be negative
  ```

**Priors:**
- `A` (baseline): `Normal(Amean, |Amean|)` → can be negative
- `Vmax` (Hill amplitude): `Gamma(mean², mean/σ²)` → positive amplitude
- `K` (EC50/IC50): `Gamma((K_max/2)², (K_max/2)/σ²)` → positive half-max point
- `n` (Hill coefficient): `Normal(0, σ_n)` clamped to `[nmin, nmax]`

**Technical Correction:**
```
y_final = y_baseline + alpha_y[group]
```
Applied additively (since values can be negative).

---

### Student-t (Robust SpliZ, Heavy-Tailed Continuous)

**Function Application:**
Same as Normal distribution (natural value space).

**Priors:**
Same as Normal distribution, plus:
- `nu_y` (degrees of freedom): `Gamma(10, 2)` → mean ~5, ensures df > 2

**Technical Correction:**
Same as Normal (additive).

---

## Function Types

bayesDREAM supports three function types for modeling trans effects:

### 1. Single Hill Function (`single_hill`)

Models either positive OR negative Hill-type dose-response:

```python
y = y_inf + A * (x^n / (K^n + x^n))
```

**Parameters:**
- `A`: Amplitude (effect size)
- `K`: Half-maximal point (EC50/IC50)
- `n`: Hill coefficient (cooperativity)
- `y_inf`: Baseline expression

**Use when:**
- Expecting monotonic responses (only up OR only down)
- Want interpretable EC50/IC50 values
- Have strong biological prior about response direction

**Example:**
```python
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='single_hill',
    modality_name='gene'
)
```

### 2. Additive Hill Function (`additive_hill`)

Models BOTH positive AND negative Hill responses simultaneously:

```python
y = y_inf + A_pos * (x^n_pos / (K_pos^n_pos + x^n_pos)) - A_neg * (x^n_neg / (K_neg^n_neg + x^n_neg))
```

**Parameters:**
- `A_pos`, `K_pos`, `n_pos`: Positive (activation) Hill function
- `A_neg`, `K_neg`, `n_neg`: Negative (repression) Hill function
- `y_inf`: Baseline expression

**Use when:**
- Expecting biphasic or non-monotonic responses
- Want to model both activation and repression
- Analyzing transcription factors with both activating and repressing targets

**Example:**
```python
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    modality_name='gene'
)
```

**Why this is the default:**
- Most flexible for discovering unexpected response patterns
- Can capture both activation and repression in single model
- Reduces to single Hill if one component has negligible amplitude
- Recommended for exploratory analysis

### 3. Polynomial Function (`polynomial`)

Models arbitrary smooth functions using polynomial basis:

```python
y = β₀ + β₁*x + β₂*x² + ... + βₖ*xᵏ
```

**Parameters:**
- `degree`: Polynomial degree (default: 6)
- Coefficients: β₀, β₁, ..., βₖ

**Use when:**
- No prior biological model
- Purely empirical curve fitting
- Comparing against non-parametric baseline
- Need maximum flexibility without mechanistic assumptions

**Example:**
```python
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='polynomial',
    degree=6  # Can adjust degree
)
```

## Method Signature

```python
def fit_trans(
    self,
    sum_factor_col: str = 'sum_factor',
    function_type: str = 'additive_hill',
    modality_name: str = None,
    lr: float = 1e-3,
    niters: int = 100_000,
    nsamples: int = 1000,
    alpha_ewma: float = 0.05,
    tolerance: float = 1e-4,
    alpha_dirichlet: float = 0.1,
    **kwargs
)
```

### Key Parameters

#### Required Parameters

- **sum_factor_col** (str, default='sum_factor'):
  - Column in `meta` containing size factors for normalization
  - Typical workflow:
    1. Original: `'sum_factor'` (from scran or library size)
    2. Adjusted for guides: `'sum_factor_adj'` (from `adjust_ntc_sum_factor()`)
    3. Adjusted for cis effects: `'sum_factor_refit'` (from `refit_sumfactor()`)
  - **Recommendation**: Use `'sum_factor_refit'` for trans fitting to remove cis gene contribution

- **function_type** (str, default='additive_hill'):
  - `'single_hill'`: Single Hill equation (positive or negative)
  - `'additive_hill'`: Additive positive + negative Hills (**recommended**)
  - `'polynomial'`: Polynomial function (flexible, non-parametric)

#### Optional Parameters

- **modality_name** (str, optional):
  - Which modality to fit trans effects on
  - Default: `None` uses primary modality (usually `'gene'`)
  - Options: Any modality name from `model.list_modalities()`
  - Examples: `'gene'`, `'atac'`, `'splicing_donor'`, `'spliz'`

- **lr** (float, default=1e-3):
  - Learning rate for Adam optimizer
  - Lower values (1e-4) for more stable but slower convergence
  - Higher values (1e-2) for faster but potentially unstable convergence

- **niters** (int, default=100,000):
  - Maximum number of SVI iterations
  - Will stop early if convergence criteria met
  - Typical convergence: 10,000-50,000 iterations

- **nsamples** (int, default=1000):
  - Number of posterior samples to draw after fitting
  - More samples = better posterior estimates but slower
  - 1000 is usually sufficient; use 100 for quick testing

- **tolerance** (float, default=1e-4):
  - Convergence tolerance for early stopping
  - Smaller values require tighter convergence
  - Recommended: keep at 1e-4

- **alpha_dirichlet** (float, default=0.1):
  - Dirichlet prior concentration for sparsity
  - Lower values encourage stronger sparsity (more genes at baseline)
  - Higher values allow more genes to respond

## Excluding Genes Before Fitting

Call `model.exclude_trans_genes()` before `fit_trans()` to permanently drop
features from a modality (typically the primary `'gene'` modality). Any
combination of three criteria can be given; a feature is dropped if it
matches ANY of them:

```python
# 1. Named features
model.exclude_trans_genes(genes=['MALAT1', 'XIST'])

# 2. feature_meta selection (pandas DataFrame.eval() syntax)
model.exclude_trans_genes(feature_query='protein_coding == False')

# 3. Low NTC expression (requires fit_ntc() to have been run first)
model.exclude_trans_genes(min_log2_mu_ntc=-4)

# Criteria combine freely in one call
model.exclude_trans_genes(
    genes=['MALAT1'],
    feature_query='protein_coding == False',
    min_log2_mu_ntc=-4,
)
```

`alpha_y_prefit_mult`/`alpha_y_prefit_add` and `posterior_samples_ntc` already
stored on the modality are trimmed to match, so `fit_trans()` (and, if called
again, `fit_ntc()`) sees a consistent, reduced feature set. If the modality
already has `posterior_samples_trans` from a previous `fit_trans()` call, it
is dropped with a warning since it would no longer match the reduced feature
count — re-run `fit_trans()` afterward.

## Complete Workflow Example

### Step-by-Step Trans Fitting

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Initialize model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./output',
    label='gfi1b_screen'
)

# Step 1: Technical fitting (NTC cells only)
model.set_technical_groups(['cell_line'])
model.fit_ntc()

# Step 2: Adjust sum factors for guide-level effects
model.adjust_ntc_sum_factor(
    sum_factor_col_old='sum_factor',
    sum_factor_col_adj='sum_factor_adj',
    covariates=['cell_line']
)

# Step 3: Cis fitting
model.fit_cis(
    sum_factor_col='sum_factor_adj',
    technical_covariates=['cell_line']
)

# Step 4: Remove cis gene contribution from sum factors
model.refit_sumfactor(
    sum_factor_col_old='sum_factor_adj',
    sum_factor_col_refit='sum_factor_refit',
    covariates=['cell_line']
)

# Step 5: Trans fitting
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    modality_name='gene',
    lr=1e-3,
    niters=100_000,
    nsamples=1000
)

# Save results
model.save_trans_fit(suffix='additive_hill')
```

### Multi-Modal Trans Fitting

```python
# Fit trans effects for multiple modalities
# (After completing technical + cis fitting)

# 1. Fit gene expression responses
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    modality_name='gene'
)
model.save_trans_fit(modalities=['gene'], suffix='additive_hill')

# 2. Fit ATAC peak responses
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    modality_name='atac'
)
model.save_trans_fit(modalities=['atac'], suffix='additive_hill')

# 3. Fit splicing responses
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    modality_name='splicing_donor'
)
model.save_trans_fit(modalities=['splicing_donor'], suffix='additive_hill')
```

### Comparing Function Types

```python
# Fit with different function types to compare
function_types = ['additive_hill', 'single_hill', 'polynomial']

for func_type in function_types:
    print(f"Fitting with {func_type}...")
    model.fit_trans(
        sum_factor_col='sum_factor_refit',
        function_type=func_type,
        modality_name='gene'
    )
    model.save_trans_fit(suffix=func_type)

# Compare fits visually
from bayesDREAM.plotting import plot_xy_data

for gene in ['TET2', 'MYB', 'NFE2']:
    model.plot_xy_data(
        feature=gene,
        show_hill_function=True,
        title=f'{gene} - {func_type}'
    )
```

## Accessing Trans Fitting Results

### Posterior Samples

```python
# Access posterior samples for trans fitting
posterior = model.posterior_samples_trans

# Available samples (varies by function_type):
# - For additive_hill:
#   - 'A': Amplitudes (positive effects)
#   - 'A_neg': Negative amplitudes (repression)
#   - 'alpha': EC50 values (positive)
#   - 'alpha_neg': IC50 values (negative)
#   - 'beta': Hill coefficients (positive)
#   - 'beta_neg': Hill coefficients (negative)
#   - 'y_inf': Baseline expression levels
#   - 'y_pred': Predicted trans feature values

# Example: Get posterior means
import torch

A_mean = posterior['A'].mean(dim=0)  # Mean amplitudes across samples
alpha_mean = posterior['alpha'].mean(dim=0)  # Mean EC50 values

# Get credible intervals (95%)
A_lower = torch.quantile(posterior['A'], 0.025, dim=0)
A_upper = torch.quantile(posterior['A'], 0.975, dim=0)
```

### Identifying Significant Responders

```python
import numpy as np

# Posterior samples shape: (n_samples, n_features)
A_samples = model.posterior_samples_trans['A'].cpu().numpy()

# Calculate posterior probability of positive effect (A > threshold)
threshold = 0.5  # log2 fold-change threshold
prob_positive = (A_samples > threshold).mean(axis=0)

# Get significant responders (e.g., > 95% posterior probability)
significant_idx = np.where(prob_positive > 0.95)[0]

# Map to gene names
gene_modality = model.get_modality('gene')
significant_genes = gene_modality.feature_meta.iloc[significant_idx]['gene'].tolist()

print(f"Found {len(significant_genes)} significant positive responders:")
print(significant_genes[:10])  # Show first 10
```

## Advanced Usage

### Custom Convergence Settings

```python
# For difficult-to-fit models, adjust convergence parameters
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    lr=5e-4,  # Lower learning rate
    niters=200_000,  # More iterations
    tolerance=1e-5,  # Tighter convergence
    alpha_ewma=0.01  # Slower ELBO smoothing
)
```

### Polynomial Degree Selection

```python
# Try different polynomial degrees
for degree in [3, 6, 9]:
    model.fit_trans(
        sum_factor_col='sum_factor_refit',
        function_type='polynomial',
        degree=degree
    )
    model.save_trans_fit(suffix=f'polynomial_deg{degree}')

    # Compare model fits (e.g., using AIC/BIC or cross-validation)
```

### Memory Management for Large Datasets

```python
# For very large datasets, reduce memory usage
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill',
    nsamples=500,  # Fewer posterior samples
    minibatch_size=100  # Process in minibatches (if implemented)
)
```

## Interpretation Guidelines

### Additive Hill Function Parameters

**Amplitude (A, A_neg)**:
- Magnitude of effect (log2 fold-change)
- `A > 0.5`: Strong positive response
- `A_neg > 0.5`: Strong negative response
- Both small: Gene is likely non-responsive

**Half-Maximal Points (alpha, alpha_neg)**:
- EC50 (alpha): Cis expression level for half-maximal activation
- IC50 (alpha_neg): Cis expression level for half-maximal repression
- Values in cis expression units (not log2)
- Lower values → responds at lower cis expression

**Hill Coefficients (beta, beta_neg)**:
- Cooperativity/steepness of response
- β = 1: Simple binding (Michaelis-Menten)
- β > 1: Cooperative activation (sharp threshold)
- β < 1: Gradual response

**Baseline (y_inf)**:
- Expression level when cis gene = 0
- Similar to NTC expression
- Useful for identifying constitutively expressed vs. induced genes

### Response Patterns

**Strong Positive Responder:**
- High `A` (> 1.0 log2FC)
- Low posterior uncertainty (narrow CI)
- `A_neg` ≈ 0 (no repression component)

**Biphasic Responder:**
- Both `A` and `A_neg` > 0.5
- Different EC50/IC50 values (alpha ≠ alpha_neg)
- Suggests dual regulatory mechanisms

**Non-Responder:**
- Both `A` and `A_neg` < 0.5
- High posterior uncertainty
- Likely y ≈ y_inf (flat response)

## Troubleshooting

### Convergence Issues

**Symptom**: Model doesn't converge within `niters` iterations

**Solutions**:
1. Lower learning rate: `lr=5e-4` or `lr=1e-4`
2. Increase iterations: `niters=200_000`
3. Adjust tolerance: `tolerance=1e-3` (looser) for faster convergence
4. Check data quality: Ensure cis fitting completed successfully

### Unexpected Parameter Values

**Symptom**: Parameters (A, alpha, beta) have unrealistic values

**Solutions**:
1. Check sum_factor: Use `'sum_factor_refit'` not raw `'sum_factor'`
2. Verify x_true: Ensure `fit_cis()` completed before `fit_trans()`
3. Check data scale: Counts should be raw (not log-transformed)
4. Try different function_type: Compare `additive_hill` vs `polynomial`

### Memory Errors

**Symptom**: Out of memory during fitting or sampling

**Solutions**:
1. Reduce `nsamples`: Try 500 or 100 instead of 1000
2. Use CPU instead of GPU: `device='cpu'` in initialization
3. Fit modalities separately: One `fit_trans()` call per modality
4. Subset features: Pre-filter to high-variance genes

## Best Practices

1. **Always use adjusted sum factors**: Use `sum_factor_refit` for trans fitting to remove cis gene effects

2. **Start with additive_hill**: Most flexible and interpretable for biological data

3. **Compare function types**: Fit multiple function types and compare for key genes

4. **Check convergence**: Monitor loss curves to ensure model has converged

5. **Validate with plots**: Visualize fitted curves using `plot_xy_data()` with `show_hill_function=True`
   - **Multi-modality support**: Plotting now works for all modalities (genes, splicing, ATAC, etc.)
   - The plotting function automatically detects which modality was fitted and retrieves the correct posterior samples
   - For non-gene modalities, pass the feature identifier that matches the modality's feature_meta column (e.g., 'coord.intron' for splice junctions)

6. **Save intermediate results**: Save after each major step (technical, cis, trans)

7. **Use appropriate covariates**: Match technical covariates across all fitting steps

## Related Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Quick start guide
- **[PLOTTING_GUIDE.md](PLOTTING_GUIDE.md)** - Visualization of trans fits
- **[DATA_ACCESS.md](DATA_ACCESS.md)** - Accessing posterior samples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Trans fitting implementation details
