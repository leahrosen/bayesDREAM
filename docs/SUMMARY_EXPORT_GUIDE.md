# Summary Export Guide

## Overview

bayesDREAM provides methods to export model results as R-friendly CSV files for downstream plotting and analysis. The summary exports include:

- **Mean and 95% credible intervals** for all parameters
- **Cell-wise and feature-wise** summaries
- **Observed log2FC**, predicted log2FC, and inflection points for trans fits
- **Compatible with different function types** (Hill, polynomial) and distributions (negbinom, normal, binomial, multinomial)

## Quick Reference

| Method | Creates | Contents |
|--------|---------|----------|
| `save_ntc_summary()` | `ntc_feature_summary_{modality}.csv` | Feature-wise overdispersion parameters per group |
| `save_cis_summary()` | `cis_guide_summary.csv`<br>`cis_cell_summary.csv` | Guide-level and cell-level x_true with credible intervals |
| `save_trans_summary()` | `trans_feature_summary_{modality}.csv` | Feature-wise dose-response parameters, log2FC, inflection points |

## Usage

### Basic Workflow

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Initialize and fit model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line'],
    output_dir='./results'
)

# Run 3-step pipeline
model.set_technical_groups(['cell_line'])
model.fit_ntc(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Export summaries
model.save_ntc_summary()
model.save_cis_summary()
model.save_trans_summary()
```

## NTC Fit Summary

### Method Signature

```python
model.save_ntc_summary(
    output_dir=None,
    modality_name=None
)
```

**Parameters:**
- `output_dir` (str, optional): Output directory (default: `model.output_dir`)
- `modality_name` (str, optional): Modality to export (default: primary modality)

### Output File

**`ntc_feature_summary_{modality}.csv`**

Columns:
- `feature`: Feature name
- `modality`: Modality name
- `distribution`: Distribution type
- `group_{i}_alpha_y_mean`: Mean overdispersion (alpha_y) for group i
- `group_{i}_alpha_y_lower`: 2.5% quantile for group i
- `group_{i}_alpha_y_upper`: 97.5% quantile for group i

### Example

```python
# After fit_ntc()
df = model.save_ntc_summary()

print(df.head())
#   feature modality distribution  group_0_alpha_y_mean  group_0_alpha_y_lower  ...
# 0    GATA1     gene     negbinom              0.523                  0.412
# 1     TAL1     gene     negbinom              0.892                  0.721
# 2    RUNX1     gene     negbinom              0.447                  0.338
```

## Cis Fit Summary

### Method Signature

```python
model.save_cis_summary(
    output_dir=None,
    include_cell_level=True
)
```

**Parameters:**
- `output_dir` (str, optional): Output directory (default: `model.output_dir`)
- `include_cell_level` (bool): Whether to save cell-level summary (default: True)

### Output Files

**1. `cis_guide_summary.csv`** (Guide-level)

Columns:
- `guide`: Guide RNA name
- `target`: Target gene
- `n_cells`: Number of cells per guide
- `x_true_mean`: Mean cis expression
- `x_true_lower`: 2.5% quantile
- `x_true_upper`: 97.5% quantile
- `raw_counts_mean`: Average raw counts

**2. `cis_cell_summary.csv`** (Cell-level, optional)

Columns:
- `cell`: Cell barcode
- `guide`: Guide RNA name
- `target`: Target gene
- `cell_line`: Cell line (if available)
- `x_true_mean`: Mean cis expression for this guide
- `x_true_lower`: 2.5% quantile
- `x_true_upper`: 97.5% quantile
- `raw_counts`: Raw counts for this cell

### Example

```python
# After fit_cis()
guide_df, cell_df = model.save_cis_summary()

print(guide_df.head())
#      guide   target  n_cells  x_true_mean  x_true_lower  x_true_upper  raw_counts_mean
# 0  gRNA1_K562  GFI1B       52     5.234         4.981         5.498            123.5
# 1  gRNA2_K562  GFI1B       48     2.134         1.892         2.387             45.2
# 2     ntc_K562    ntc      150     6.892         6.734         7.045            234.1
```

## Trans Fit Summary

### Method Signature

```python
model.save_trans_summary(
    output_dir=None,
    modality_name=None,
    compute_inflection=True,
    compute_full_log2fc=True
)
```

**Parameters:**
- `output_dir` (str, optional): Output directory (default: `model.output_dir`)
- `modality_name` (str, optional): Modality to export (default: primary modality)
- `compute_inflection` (bool): Whether to compute inflection points for Hill functions (default: True)
- `compute_full_log2fc` (bool): Whether to compute full log2FC range (default: True)

### Output File

**`trans_feature_summary_{modality}.csv`**

**Common columns (all function types):**
- `feature`: Feature name
- `modality`: Modality name
- `distribution`: Distribution type
- `function_type`: Function type (additive_hill, single_hill, polynomial)
- `observed_log2fc`: Observed log2FC (perturbed vs NTC)
- `observed_log2fc_se`: Standard error of observed log2FC

**Additive Hill parameters:**
- `Vmax_a_mean`, `Vmax_a_lower`, `Vmax_a_upper`: Component A magnitude
- `EC50_a_mean`, `EC50_a_lower`, `EC50_a_upper`: Component A half-max point
- `n_a_mean`, `n_a_lower`, `n_a_upper`: Component A Hill coefficient (cooperativity)
- `Vmax_b_mean`, `Vmax_b_lower`, `Vmax_b_upper`: Component B magnitude
- `EC50_b_mean`, `EC50_b_lower`, `EC50_b_upper`: Component B half-max point
- `n_b_mean`, `n_b_lower`, `n_b_upper`: Component B Hill coefficient (cooperativity)
- `pi_y_mean`, `pi_y_lower`, `pi_y_upper`: Sparsity weight (optional)
- `alpha_mean`, `alpha_lower`, `alpha_upper`: Component A weight (optional)
- `beta_mean`, `beta_lower`, `beta_upper`: Component B weight (optional)
- `inflection_a_mean`, `inflection_a_lower`, `inflection_a_upper`: Component A inflection x
- `inflection_b_mean`, `inflection_b_lower`, `inflection_b_upper`: Component B inflection x
- `full_log2fc_mean`, `full_log2fc_lower`, `full_log2fc_upper`: Full dynamic range

**Single Hill parameters:**
- `B_mean`, `B_lower`, `B_upper`: Hill magnitude
- `K_mean`, `K_lower`, `K_upper`: Hill coefficient
- `xc_mean`, `xc_lower`, `xc_upper`: Half-max point (EC50 or IC50)
- `inflection_mean`, `inflection_lower`, `inflection_upper`: Inflection x
- `full_log2fc_mean`, `full_log2fc_lower`, `full_log2fc_upper`: Full dynamic range

**Polynomial parameters:**
- `coef_{i}_mean`, `coef_{i}_lower`, `coef_{i}_upper`: Coefficient i (for each degree)
- `full_log2fc_mean`, `full_log2fc_lower`, `full_log2fc_upper`: Full dynamic range

### Example

```python
# After fit_trans()
df = model.save_trans_summary()

print(df[['feature', 'observed_log2fc', 'B_pos_mean', 'EC50_pos_mean', 'full_log2fc_mean']].head())
#   feature  observed_log2fc  B_pos_mean  EC50_pos_mean  full_log2fc_mean
# 0   GATA1             0.85       1.234          4.521             1.892
# 1    TAL1            -0.45      -0.823          5.234             1.456
# 2   RUNX1             0.23       0.567          6.123             0.892
```

## Understanding Inflection Points

For Hill functions, the inflection point is where the dose-response curve has maximum curvature (steepest slope).

**Formula:**
```
x_inflection = xc * ((K - 1) / (K + 1))^(1/K)
```

**Only defined for K > 1** (sigmoidal curves). For K ≤ 1, inflection point is NaN.

**Interpretation:**
- The x value at which the effect changes most rapidly
- Useful for identifying the dynamic range of the response
- For K > 1, inflection occurs before the half-max point (xc)

## Understanding Full Log2FC

**Definition:** The total magnitude of the dose-response effect across the observed x_true range.

**Calculation:**
- **Additive Hill**: `B_pos + B_neg` (sum of positive and negative effects)
- **Single Hill**: `|B|` (absolute magnitude)
- **Polynomial**: `|y(x_max) - y(x_min)|` (range across x_true)

**Interpretation:**
- Total change in expression from minimum to maximum cis expression
- For negbinom: log2 fold change in expression level
- For normal: absolute change in score

## Using Summaries in R

### Complete Example Script

A comprehensive R plotting script is available at **[docs/example_summary_plots.R](example_summary_plots.R)**. This script demonstrates:
- Loading all summary CSVs
- Creating volcano plots, overdispersion plots, cis expression plots
- Visualizing Hill parameters and inflection points
- Computing summary statistics

Run it with:
```bash
Rscript docs/example_summary_plots.R
```

### Load Data

```r
library(tidyverse)

# Load summaries
tech_summary <- read_csv("results/ntc_feature_summary_gene.csv")
cis_guide <- read_csv("results/cis_guide_summary.csv")
cis_cell <- read_csv("results/cis_cell_summary.csv")
trans_summary <- read_csv("results/trans_feature_summary_gene.csv")
```

### Plot Technical Fit

```r
# Plot overdispersion by group
tech_summary %>%
  select(feature, starts_with("group_")) %>%
  pivot_longer(cols = -feature,
               names_to = c("group", ".value"),
               names_pattern = "group_(\\d+)_alpha_y_(.+)") %>%
  ggplot(aes(x = mean, y = reorder(feature, mean))) +
  geom_pointrange(aes(xmin = lower, xmax = upper, color = group)) +
  labs(x = "Overdispersion (alpha_y)", y = "Feature") +
  theme_minimal()
```

### Plot Cis Expression

```r
# Plot x_true by guide
cis_guide %>%
  ggplot(aes(x = reorder(guide, x_true_mean), y = x_true_mean, color = target)) +
  geom_pointrange(aes(ymin = x_true_lower, ymax = x_true_upper)) +
  labs(x = "Guide", y = "Cis Expression (x_true)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

### Plot Trans Effects

```r
# Volcano plot with confidence intervals
trans_summary %>%
  mutate(significant = abs(observed_log2fc) > 0.5) %>%
  ggplot(aes(x = observed_log2fc, y = -log10(observed_log2fc_se))) +
  geom_point(aes(color = significant), alpha = 0.6) +
  geom_errorbarh(aes(xmin = observed_log2fc - 1.96*observed_log2fc_se,
                     xmax = observed_log2fc + 1.96*observed_log2fc_se),
                 alpha = 0.3) +
  labs(x = "Observed Log2FC", y = "-log10(SE)") +
  theme_minimal()

# Hill parameter scatter
trans_summary %>%
  ggplot(aes(x = EC50_pos_mean, y = K_pos_mean)) +
  geom_point(aes(color = B_pos_mean), alpha = 0.6) +
  geom_errorbar(aes(ymin = K_pos_lower, ymax = K_pos_upper), alpha = 0.2) +
  geom_errorbarh(aes(xmin = EC50_pos_lower, xmax = EC50_pos_upper), alpha = 0.2) +
  scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(x = "EC50", y = "Hill Coefficient (K)", color = "Magnitude (B)") +
  theme_minimal()
```

## Advanced Usage

### Modality-Specific Summaries

```python
# Export summaries for specific modality
model.save_ntc_summary(modality_name='atac')
model.save_trans_summary(modality_name='splicing_donor')
```

### Cell-Level Only

```python
# Skip cell-level export (guide-level only)
guide_df = model.save_cis_summary(include_cell_level=False)
```

### Custom Output Directory

```python
# Save to custom directory
model.save_ntc_summary(output_dir='./plots_data')
model.save_cis_summary(output_dir='./plots_data')
model.save_trans_summary(output_dir='./plots_data')
```

### Skip Optional Calculations

```python
# Skip inflection and full log2FC calculations (faster)
model.save_trans_summary(
    compute_inflection=False,
    compute_full_log2fc=False
)
```

## Complete Example Workflow

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
    guide_covariates=['cell_line'],
    output_dir='./results'
)

# Run full pipeline
model.set_technical_groups(['cell_line'])
model.fit_ntc(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Export all summaries
print("Exporting summaries...")
model.save_ntc_summary()
model.save_cis_summary()
model.save_trans_summary()

print("Done! CSV files saved to ./results/")
print("Files created:")
print("  - ntc_feature_summary_gene.csv")
print("  - cis_guide_summary.csv")
print("  - cis_cell_summary.csv")
print("  - trans_feature_summary_gene.csv")
```

## Troubleshooting

### Error: "Technical fit not found"

**Cause:** Trying to save technical summary before running `fit_ntc()`

**Solution:**
```python
model.fit_ntc(sum_factor_col='sum_factor')
model.save_ntc_summary()  # Now works
```

### Error: "Cis fit not found"

**Cause:** Trying to save cis summary before running `fit_cis()`

**Solution:**
```python
model.fit_cis(sum_factor_col='sum_factor')
model.save_cis_summary()  # Now works
```

### Error: "Trans fit not found"

**Cause:** Trying to save trans summary before running `fit_trans()`

**Solution:**
```python
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
model.save_trans_summary()  # Now works
```

### Error: "No technical fit parameters found for modality"

**Cause:** Trying to export a modality that wasn't fit in `fit_ntc()`

**Solution:**
```python
# Fit the specific modality first
model.fit_ntc(modality_name='atac', sum_factor_col='sum_factor')
model.save_ntc_summary(modality_name='atac')  # Now works
```

### Empty or NaN inflection points

**Cause:** Inflection points are only defined for Hill functions with K > 1

**Explanation:** For K ≤ 1, the Hill curve is not sigmoidal and has no inflection point. This is expected behavior.

## Testing

The summary export functionality is fully tested in **[tests/test_summary_export_simple.py](../tests/test_summary_export_simple.py)**. This test validates:

✓ Technical summary export with group-wise overdispersion parameters
✓ Cis summary export (guide-level and cell-level)
✓ Trans summary export with additive Hill functions
✓ Trans summary export with polynomial functions
✓ Inflection point calculations for Hill functions
✓ Full log2FC calculations for all function types
✓ CSV file creation and structure validation

Run the test:
```bash
python tests/test_summary_export_simple.py
```

All tests pass successfully, confirming that the summary export methods work correctly with all supported distributions and function types.

## See Also

- [example_summary_plots.R](example_summary_plots.R) - Complete R plotting examples
- [SAVE_LOAD_GUIDE.md](SAVE_LOAD_GUIDE.md) - Save/load fitted parameters
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [PLOTTING_GUIDE.md](PLOTTING_GUIDE.md) - Visualization examples
- [DATA_ACCESS.md](DATA_ACCESS.md) - Accessing model data directly
