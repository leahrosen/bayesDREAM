# Save/Load Guide for bayesDREAM

## Overview

bayesDREAM provides methods to save and load fitted parameters at each stage of the pipeline:
1. **Technical fit** (`fit_technical`) → `save_ntc_fit()` / `load_ntc_fit()`
2. **Cis fit** (`fit_cis`) → `save_cis_fit()` / `load_cis_fit()`
3. **Trans fit** (`fit_trans`) → `save_trans_fit()` / `load_trans_fit()`

This allows you to:
- Run expensive fits once and reuse results
- Run pipeline stages separately (e.g., on different compute nodes)
- Experiment with different downstream parameters without refitting

## Quick Reference

### Save Methods

| Method | Saves | File(s) Created | Modality Control |
|--------|-------|-----------------|------------------|
| `save_ntc_fit()` | Technical parameters | `alpha_x_prefit.pt`, `alpha_y_prefit.pt`, `posterior_samples_ntc.pt`, `alpha_y_prefit_{modality}.pt` | `modalities=` |
| `save_cis_fit()` | Cis parameters | `x_true.pt`, `posterior_samples_cis.pt` | N/A (cis is model-level) |
| `save_trans_fit()` | Trans parameters | `posterior_samples_trans.pt`, `posterior_samples_trans_{modality}.pt` | `modalities=` |

### Load Methods

| Method | Loads | Required Files | Modality Control |
|--------|-------|----------------|------------------|
| `load_ntc_fit()` | Technical parameters | `alpha_x_prefit.pt`, `alpha_y_prefit.pt`, etc. | `modalities=` |
| `load_cis_fit()` | Cis parameters | `x_true.pt`, `posterior_samples_cis.pt` | N/A (cis is model-level) |
| `load_trans_fit()` | Trans parameters | `posterior_samples_trans.pt`, etc. | `modalities=` |

## Detailed Usage

### 1. Save Technical Fit

After running `fit_ntc()`:

```python
model = bayesDREAM(...)
model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='gene', sum_factor_col='sum_factor')

# Save all modalities to default output_dir
model.save_ntc_fit()

# Or specify custom directory
model.save_ntc_fit(output_dir='./my_results/technical/')

# Save specific modalities only
model.save_ntc_fit(modalities=['gene', 'atac'])

# Note: Model-level parameters are automatically saved when primary modality is included
# If you save only non-primary modalities, model-level params are skipped automatically
model.save_ntc_fit(modalities=['atac'])  # Skips model-level (primary is 'gene')
```

**What Gets Saved**:

Model-level (automatically saved when primary modality is included):
- `alpha_x_prefit.pt`: Overdispersion for cis gene (if exists)
- `alpha_y_prefit.pt`: Overdispersion for trans genes (primary modality, backward compat)
- `posterior_samples_ntc.pt`: Full posterior samples (primary modality)

Per-modality (for each modality in `modalities` list):
- `alpha_y_prefit_{modality}.pt`: Per-modality overdispersion

**File Structure**:
```
output_dir/
├── alpha_x_prefit.pt
├── alpha_y_prefit.pt
├── posterior_samples_ntc.pt
├── alpha_y_prefit_gene.pt
├── alpha_y_prefit_splicing_donor.pt
└── ...
```

### 2. Load Technical Fit

Before running `fit_cis()`:

```python
model = bayesDREAM(...)

# Load all modalities from default output_dir
model.load_ntc_fit()

# Or specify custom directory
model.load_ntc_fit(input_dir='./my_results/technical/')

# Use posterior samples (default)
model.load_ntc_fit(use_posterior=True)

# Or use point estimates (posterior mean)
model.load_ntc_fit(use_posterior=False)

# Load specific modalities only
model.load_ntc_fit(modalities=['gene', 'atac'])

# Note: Model-level parameters are automatically loaded when primary modality is included
# If you load only non-primary modalities, model-level params are skipped automatically
model.load_ntc_fit(modalities=['atac'])  # Skips model-level (primary is 'gene')
```

**Parameters**:
- `input_dir`: Directory containing saved files (default: `self.output_dir`)
- `use_posterior`: If `True`, loads full posterior samples. If `False`, uses posterior mean as point estimate
- `modalities`: List of modality names to load (default: all available modalities)

**What Happens**:
- Sets `self.alpha_x_prefit` and `self.alpha_x_type`
- Sets `self.alpha_y_prefit` and `self.alpha_y_type`
- Loads `self.posterior_samples_ntc`
- Loads per-modality `alpha_y_prefit` for each modality

### 3. Save Cis Fit

After running `fit_cis()`:

```python
model.fit_cis(sum_factor_col='sum_factor')

# Save
model.save_cis_fit()
# or
model.save_cis_fit(output_dir='./my_results/cis/')
```

**What Gets Saved**:
- `x_true.pt`: True cis gene expression (posterior samples)
- `posterior_samples_cis.pt`: Full posterior samples from cis fit

### 4. Load Cis Fit

Before running `fit_trans()`:

```python
model = bayesDREAM(...)

# Load technical fit first (required for alpha_y)
model.load_ntc_fit()

# Load cis fit
model.load_cis_fit()

# Or with point estimates
model.load_cis_fit(use_posterior=False)
```

**What Happens**:
- Sets `self.x_true` and `self.x_true_type`
- Loads `self.posterior_samples_cis`

### 5. Save Trans Fit

After running `fit_trans()`:

```python
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Save all modalities
model.save_trans_fit()

# Save specific modalities only
model.save_trans_fit(modalities=['gene', 'atac'])

# Note: Model-level parameters are automatically saved when primary modality is included
model.save_trans_fit(modalities=['atac'])  # Skips model-level (primary is 'gene')
```

**What Gets Saved**:

Model-level (automatically saved when primary modality is included):
- `posterior_samples_trans.pt`: Model-level posterior samples (primary modality, backward compat)

Per-modality (for each modality in `modalities` list):
- `posterior_samples_trans_{modality}.pt`: Per-modality posterior samples

### 6. Load Trans Fit

For downstream analysis:

```python
model = bayesDREAM(...)

# Load all previous fits
model.load_ntc_fit()
model.load_cis_fit()
model.load_trans_fit()

# Load specific modalities only
model.load_trans_fit(modalities=['gene', 'atac'])

# Note: Model-level parameters are automatically loaded when primary modality is included
model.load_trans_fit(modalities=['atac'])  # Skips model-level (primary is 'gene')
```

**Parameters**:
- `input_dir`: Directory containing saved files (default: `self.output_dir`)
- `modalities`: List of modality names to load (default: all available modalities)

## Complete Pipeline Examples

### Example 1: Standard Workflow (Save Everything)

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)

# Initialize
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./results/',
    label='GFI1B_analysis'
)

# Stage 1: Technical
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.save_ntc_fit()

# Stage 2: Cis
model.fit_cis(sum_factor_col='sum_factor')
model.save_cis_fit()

# Stage 3: Trans
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
model.save_trans_fit()
```

### Example 2: Separate Scripts (Load and Continue)

**Script 1: `run_technical.py`**
```python
from bayesDREAM import bayesDREAM
import pandas as pd

meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)

model = bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B', output_dir='./results/')
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.save_ntc_fit()
```

**Script 2: `run_cis.py`**
```python
from bayesDREAM import bayesDREAM
import pandas as pd

meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)

model = bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B', output_dir='./results/')

# Load previous fit
model.load_ntc_fit()

# Continue with cis
model.fit_cis(sum_factor_col='sum_factor')
model.save_cis_fit()
```

**Script 3: `run_trans.py`**
```python
from bayesDREAM import bayesDREAM
import pandas as pd

meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)

model = bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B', output_dir='./results/')

# Load previous fits
model.load_ntc_fit()
model.load_cis_fit()

# Continue with trans
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
model.save_trans_fit()
```

### Example 3: Multi-Modal with ATAC

```python
# Stage 1: Fit technical on gene expression (primary modality)
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B', guide_covariates=['cell_line'])
model.add_atac_modality(atac_counts, region_meta)

model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='gene', sum_factor_col='sum_factor')
model.save_ntc_fit()  # Saves alpha_x_prefit, alpha_y_prefit, alpha_y_prefit_atac

# Stage 2: Load and fit cis
model2 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B', guide_covariates=['cell_line'])
model2.add_atac_modality(atac_counts, region_meta)

model2.load_ntc_fit()  # Loads all technical parameters including per-modality
model2.fit_cis(sum_factor_col='sum_factor')
model2.save_cis_fit()

# Stage 3: Load and fit trans
model3 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B', guide_covariates=['cell_line'])
model3.add_atac_modality(atac_counts, region_meta)

model3.load_ntc_fit()
model3.load_cis_fit()
model3.fit_trans(modality_name='atac', sum_factor_col='sum_factor_adj', function_type='additive_hill')
model3.save_trans_fit()  # Saves posterior_samples_trans_atac.pt
```

### Example 4: Modality-Specific Save/Load

```python
# Fit technical on multiple modalities
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model.add_atac_modality(atac_counts, region_meta)
model.add_splicing_modality(sj_counts, sj_meta, splicing_types=['donor'])

model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='gene', sum_factor_col='sum_factor')
model.fit_technical(modality_name='atac', sum_factor_col='sum_factor')
model.fit_technical(modality_name='splicing_donor', sum_factor_col='sum_factor')

# Save only specific modalities
model.save_ntc_fit(modalities=['gene', 'atac'])  # Skip splicing_donor

# In a new session, load only what you need
model2 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model2.add_atac_modality(atac_counts, region_meta)
# Note: No need to add splicing_donor if we're not loading it

model2.load_ntc_fit(modalities=['gene'])  # Load only gene
model2.fit_cis(sum_factor_col='sum_factor')
model2.save_cis_fit()

# Later, load gene for cis, then fit trans on ATAC
model3 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model3.add_atac_modality(atac_counts, region_meta)

model3.load_ntc_fit(modalities=['gene', 'atac'])  # Load both
model3.load_cis_fit()

# Fit trans on ATAC, save only ATAC trans results
model3.fit_trans(modality_name='atac', sum_factor_col='sum_factor_adj')
model3.save_trans_fit(modalities=['atac'])  # Model-level skipped automatically (primary is 'gene')
```

## Posterior Samples vs Point Estimates

By default, `load_*_fit()` methods load full posterior samples. You can optionally use point estimates (posterior means) for:
- **Faster loading**: Smaller memory footprint
- **Compatibility**: Some downstream tools may expect point estimates
- **Speed**: Faster computation in subsequent stages

### Using Point Estimates

```python
# Load as point estimates
model.load_ntc_fit(use_posterior=False)  # alpha_x_type='point', alpha_y_type='point'
model.load_cis_fit(use_posterior=False)        # x_true_type='point'
```

### When to Use Each

| Scenario | Recommendation |
|----------|---------------|
| Full Bayesian uncertainty propagation | `use_posterior=True` (default) |
| Quick exploratory analysis | `use_posterior=False` |
| Memory constrained environment | `use_posterior=False` |
| Final publication-quality results | `use_posterior=True` |

## Modality-Specific Parameters

### When to Use `modalities` Parameter

**Use Case 1: Save Storage Space**
```python
# If you only need gene expression results, don't save ATAC
model.save_ntc_fit(modalities=['gene'])
model.save_trans_fit(modalities=['gene'])
```

**Use Case 2: Selective Loading for Speed**
```python
# Load only what you need for this analysis
model.load_ntc_fit(modalities=['gene'])  # Skip loading large ATAC arrays
```

**Use Case 3: Incremental Fitting**
```python
# Fit and save modalities one at a time
model.fit_technical(modality_name='gene', ...)
model.save_ntc_fit(modalities=['gene'])  # Includes model-level (primary)

model.fit_technical(modality_name='atac', ...)
model.save_ntc_fit(modalities=['atac'])  # Skips model-level (not primary)
```

**Use Case 4: Different Compute Resources**
```python
# Fit heavy modalities on HPC, lighter ones locally
# On HPC:
model.fit_technical(modality_name='atac', ...)
model.save_ntc_fit(modalities=['atac'])  # Skips model-level automatically

# On local machine:
model.load_ntc_fit(modalities=['gene'])  # From previous run
model.load_ntc_fit(modalities=['atac'])  # From HPC
```

### Automatic Model-Level Behavior

**When model-level parameters are saved/loaded:**
- Automatically included when primary modality is in the `modalities` list
- Automatically skipped when primary modality is NOT in the `modalities` list
- No explicit flag needed - it's automatic based on whether you're saving/loading the primary modality

**Example**:
```python
# Primary modality is 'gene'
model.save_ntc_fit(modalities=['gene'])  # Saves model-level params
model.save_ntc_fit(modalities=['atac'])  # Skips model-level params
model.save_ntc_fit(modalities=['gene', 'atac'])  # Saves model-level params
```

## Advanced: Manual Save/Load

If you need finer control, you can save/load individual components:

```python
import torch

# Save individual components
torch.save(model.alpha_x_prefit, 'my_alpha_x.pt')
torch.save(model.x_true, 'my_x_true.pt')

# Load individual components
model.alpha_x_prefit = torch.load('my_alpha_x.pt')
model.alpha_x_type = 'posterior'  # or 'point'
model.x_true = torch.load('my_x_true.pt')
model.x_true_type = 'posterior'
```

## Troubleshooting

### Files Not Found

If you get "file not found" errors:
- Check that `output_dir` matches between save and load
- Ensure you're loading in the correct order (technical → cis → trans)
- Verify files exist: `ls ./results/`

### Missing Modalities

If per-modality files aren't loaded:
- Ensure you create the same modalities before loading
- Modality names must match exactly
- Call `add_*_modality()` before `load_*_fit()`

### Memory Issues

If loading large posteriors causes memory issues:
- Use `use_posterior=False` to load point estimates
- Consider loading only necessary components manually

## Migration from Old run_pipeline Scripts

If you're updating from the old `run_pipeline/` scripts:

**Old way**:
```python
torch.save(model.alpha_y_prefit, f'{outdir}/alpha_y_prefit.pt')
alpha_y = torch.load(f'{outdir}/alpha_y_prefit.pt')
model.set_alpha_x(alpha_y[:,:,model.counts.index.values == model.cis_gene].mean(dim=0), ...)
```

**New way**:
```python
model.save_ntc_fit()
model.load_ntc_fit()  # Automatically handles alpha_x and alpha_y extraction
```

The new methods:
- ✅ Handle cis gene extraction automatically
- ✅ Support per-modality parameters
- ✅ Provide consistent interface
- ✅ Include proper type tracking ('posterior' vs 'point')

## Summary

- Use `save_*_fit()` after each pipeline stage
- Use `load_*_fit()` before the next stage
- Set `use_posterior=False` for faster loading with point estimates
- Per-modality parameters are saved/loaded automatically
- Default `output_dir` is used unless specified

For complete API documentation, see `docs/API_REFERENCE.md`.
