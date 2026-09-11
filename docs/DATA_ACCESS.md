# Data Access Guide

Guide to accessing and working with data in bayesDREAM.

## Table of Contents

- [Accessing Modalities](#accessing-modalities)
  - [Understanding the Cis Modality](#understanding-the-cis-modality)
  - [Accessing Cis vs Trans Features](#accessing-cis-vs-trans-features)
- [Working with Count Data](#working-with-count-data)
- [Accessing Metadata](#accessing-metadata)
- [Subsetting Data](#subsetting-data)
  - [Understanding Cell Subsetting](#understanding-cell-subsetting)
- [Converting Data Formats](#converting-data-formats)
- [Accessing Model Results](#accessing-model-results)
- [Working with Posterior Samples](#working-with-posterior-samples)
- [Working with Cis Modality Data](#working-with-cis-modality-data)

---

## Accessing Modalities

### Understanding the Cis Modality

bayesDREAM uses a **separate 'cis' modality** for the targeted feature:

```python
# The 'cis' modality is automatically created during initialization
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,      # 1000 genes including GFI1B
    cis_gene='GFI1B',
    output_dir='./output'
)

# Check what was created:
print(model.list_modalities())
```

Output:
```
                  name distribution  n_features  n_cells  n_categories
0                  cis     negbinom           1      500           NaN
1                 gene     negbinom         999      500           NaN
2   splicing_donor  multinomial          50      500          10.0
3               spliz       normal        1000      500           NaN
```

**Key Points:**
- The **'cis' modality** contains just the targeted feature (e.g., GFI1B)
- The **primary modality** (e.g., 'gene') contains all OTHER features (trans features only)
- `model.counts` still contains ALL features (for technical fitting)
- All modalities are automatically subset to cells present in the 'cis' modality

### List All Modalities

```python
# Get summary of all modalities
summary = model.list_modalities()
print(summary)
```

### Get Specific Modality

```python
# Get a modality by name
donor_mod = model.get_modality('splicing_donor')

# Access modality attributes
print(donor_mod.name)           # 'splicing_donor'
print(donor_mod.distribution)   # 'multinomial'
print(donor_mod.dims)           # {'n_features': 50, 'n_cells': 500, 'n_categories': 10}
```

### Check if Modality Exists

```python
if 'splicing_donor' in model.modalities:
    donor_mod = model.get_modality('splicing_donor')
else:
    print("Modality not found")
```

### Accessing Cis vs Trans Features

```python
# Access cis feature data (targeted gene)
cis_mod = model.get_modality('cis')
cis_counts = cis_mod.counts  # shape: (1, n_cells)
cis_gene_name = cis_mod.feature_meta.index[0]  # e.g., 'GFI1B'

print(f"Cis gene: {cis_gene_name}")
print(f"Cis counts shape: {cis_counts.shape}")

# Access trans feature data (all other genes)
gene_mod = model.get_modality('gene')
trans_counts = gene_mod.counts  # shape: (n_trans_genes, n_cells)
trans_gene_names = gene_mod.feature_meta.index

print(f"Number of trans genes: {len(trans_gene_names)}")
print(f"Trans counts shape: {trans_counts.shape}")

# Access ALL features (cis + trans) for technical fitting
all_counts = model.counts  # shape: (n_all_genes, n_cells) - includes cis gene
print(f"All genes (for technical fit): {all_counts.shape}")
```

**When to use each:**
- **`cis_mod`**: For `fit_cis()` - modeling direct perturbation effects
- **`gene_mod`**: For `fit_trans()` - modeling downstream effects
- **`model.counts`**: For `fit_ntc()` - includes cis gene for cell-line effect estimation

---

## Working with Count Data

### Access Raw Counts

```python
# IMPORTANT DISTINCTION:
# model.counts vs modality counts

# 1. model.counts: ALL features (includes cis gene)
#    Used for fit_ntc() on primary modality
all_gene_counts = model.counts  # pandas DataFrame (all genes × cells)
print(all_gene_counts.shape)    # (1000, 500) - includes GFI1B

# 2. Cis modality: ONLY cis feature
cis_mod = model.get_modality('cis')
cis_counts = cis_mod.counts  # numpy array (1 × cells)
print(cis_counts.shape)      # (1, 500)

# 3. Primary modality: ONLY trans features (excludes cis)
gene_mod = model.get_modality('gene')
trans_counts = gene_mod.counts  # numpy array (trans genes × cells)
print(trans_counts.shape)       # (999, 500) - excludes GFI1B

# 4. Other modalities
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # numpy array (donors × cells × acceptors)
print(donor_counts.shape)        # (50, 500, 10)
```

### 2D Count Data (negbinom, normal, binomial)

```python
# Gene counts (negbinom)
gene_mod = model.get_modality('gene')
gene_counts = gene_mod.counts  # shape: (n_genes, n_cells)

# Access specific gene
gene_idx = 0
gene_expression = gene_counts[gene_idx, :]  # expression across all cells

# Access specific cell
cell_idx = 10
cell_profile = gene_counts[:, cell_idx]  # all genes in this cell
```

### 3D Count Data (multinomial, mvnormal)

```python
# Donor usage (multinomial)
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # shape: (n_donors, n_cells, n_acceptors)

# Access specific donor
donor_idx = 5
donor_usage = donor_counts[donor_idx, :, :]  # cells × acceptors

# Access specific cell
cell_idx = 10
cell_donors = donor_counts[:, cell_idx, :]  # donors × acceptors for this cell

# SpliZVD (mvnormal)
splizvd_mod = model.get_modality('splizvd')
splizvd_data = splizvd_mod.counts  # shape: (n_genes, n_cells, 3)
# 3rd dimension: [z0, z1, z2]
```

### Binomial with Denominator

```python
# Exon skipping
exon_mod = model.get_modality('splicing_exon_skip')
inclusion = exon_mod.counts         # numerator (events × cells)
total = exon_mod.denominator        # denominator (events × cells)

# Calculate PSI (proportion spliced in)
psi = inclusion / (total + 1e-9)  # add small constant to avoid division by zero

# For exon skipping, raw junction counts are also available
if exon_mod.is_exon_skipping():
    inc1 = exon_mod.inc1  # Inclusion junction 1 (d1->a2)
    inc2 = exon_mod.inc2  # Inclusion junction 2 (d2->a3)
    skip = exon_mod.skip  # Skipping junction (d1->a3)

    print(f"Aggregation method: {exon_mod.exon_aggregate_method}")  # 'min' or 'mean'
```

---

## Accessing Metadata

### Cell Metadata

```python
# Access full cell metadata
meta = model.meta  # pandas DataFrame

# Key columns
print(meta['cell'])         # Cell barcodes
print(meta['guide'])        # Guide RNA names
print(meta['target'])       # Target genes
print(meta['cell_line'])    # Cell line identifiers
print(meta['sum_factor'])   # Normalization factors

# Check for adjusted sum factors
if 'sum_factor_adj' in meta.columns:
    print(meta['sum_factor_adj'])
```

### Feature Metadata

```python
# Gene metadata
gene_mod = model.get_modality('gene')
gene_meta = gene_mod.feature_meta
print(gene_meta.index)  # Gene names

# Donor metadata
donor_mod = model.get_modality('splicing_donor')
donor_meta = donor_mod.feature_meta
print(donor_meta.columns)
# Output: ['chrom', 'strand', 'donor', 'acceptors', 'n_acceptors']

# Access specific donor info
print(donor_meta.loc[0, 'donor'])      # Donor position
print(donor_meta.loc[0, 'acceptors'])  # List of acceptor positions
print(donor_meta.loc[0, 'n_acceptors']) # Number of acceptors

# Exon skipping metadata
exon_mod = model.get_modality('splicing_exon_skip')
exon_meta = exon_mod.feature_meta
print(exon_meta.columns)
# Output: ['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3', 'sj_inc1', 'sj_inc2', 'sj_skip']
```

### Guide-Level Metadata

```python
# Access guide-level summaries
guide_meta = model.meta.groupby('guide').first()
print(guide_meta['target'])  # Target for each guide
```

---

## Subsetting Data

### Understanding Cell Subsetting

**All modalities are automatically subset to cells in the 'cis' modality:**

```python
# During initialization, bayesDREAM:
# 1. Creates 'cis' modality from cis_gene
# 2. Filters cells (e.g., removes cells with zero counts)
# 3. Subsets ALL other modalities to match these cells

# Check cell alignment
cis_mod = model.get_modality('cis')
gene_mod = model.get_modality('gene')
donor_mod = model.get_modality('splicing_donor')

print(f"Cis cells: {cis_mod.dims['n_cells']}")
print(f"Gene cells: {gene_mod.dims['n_cells']}")
print(f"Donor cells: {donor_mod.dims['n_cells']}")
# All should be the same!

# Verify cell names match
if cis_mod.cell_names is not None and gene_mod.cell_names is not None:
    assert cis_mod.cell_names == gene_mod.cell_names
    print("✓ Cell names match across modalities")
```

**Why this matters:**
- Ensures all modalities have the same cells
- Prevents dimension mismatches during fitting
- The 'cis' modality defines the reference cell set

### Subset by Features

```python
# Get specific genes
gene_names = ['GFI1B', 'GATA1', 'TAL1']
gene_mod = model.get_modality('gene')

# Find indices
gene_indices = [i for i, name in enumerate(gene_mod.feature_meta.index)
                if name in gene_names]

# Subset modality
subset_mod = gene_mod.get_feature_subset(gene_indices)
print(subset_mod.counts.shape)  # (3, n_cells)
```

### Subset by Cells

```python
# Get cells from specific guide
guide_name = 'gRNA1'
cell_indices = model.meta[model.meta['guide'] == guide_name].index

# Subset modality
donor_mod = model.get_modality('splicing_donor')
subset_mod = donor_mod.get_cell_subset(cell_indices)
print(subset_mod.counts.shape)  # (n_donors, n_guide_cells, n_acceptors)
```

### Filter by Cell Line

```python
# Get cells from K562 cell line
k562_cells = model.meta[model.meta['cell_line'] == 'K562']
cell_indices = k562_cells.index

# Subset all modalities
for mod_name in model.modalities.keys():
    mod = model.get_modality(mod_name)
    subset = mod.get_cell_subset(cell_indices)
    print(f"{mod_name}: {subset.dims}")
```

### Filter by NTC vs Perturbed

```python
# NTC cells
ntc_cells = model.meta[model.meta['target'] == 'ntc']
ntc_indices = ntc_cells.index

# Perturbed cells (targeting cis gene)
cis_cells = model.meta[model.meta['target'] == model.cis_gene]
cis_indices = cis_cells.index

# Get trans gene counts for each
gene_mod = model.get_modality('gene')  # Trans genes only
ntc_trans = gene_mod.get_cell_subset(ntc_indices).counts
cis_trans = gene_mod.get_cell_subset(cis_indices).counts

# Get cis gene counts for each
cis_mod = model.get_modality('cis')  # Cis gene only
ntc_cis = cis_mod.get_cell_subset(ntc_indices).counts
cis_cis = cis_mod.get_cell_subset(cis_indices).counts

print(f"Trans genes - NTC: {ntc_trans.shape}, Perturbed: {cis_trans.shape}")
print(f"Cis gene - NTC: {ntc_cis.shape}, Perturbed: {cis_cis.shape}")

# Compare cis gene expression between NTC and perturbed
cis_gene_mean_ntc = ntc_cis.mean()
cis_gene_mean_perturbed = cis_cis.mean()
print(f"Cis gene fold change: {cis_gene_mean_perturbed / cis_gene_mean_ntc:.2f}")
```

---

## Converting Data Formats

### Convert to PyTorch Tensors

```python
# Convert counts to tensor
gene_mod = model.get_modality('gene')
gene_tensor = gene_mod.to_tensor(device='cuda')  # or 'cpu'
print(type(gene_tensor))  # torch.Tensor
print(gene_tensor.device)  # cuda:0 or cpu

# Convert 3D data
donor_mod = model.get_modality('splicing_donor')
donor_tensor = donor_mod.to_tensor(device='cpu')
print(donor_tensor.shape)  # torch.Size([50, 500, 10])
```

### Convert to Pandas DataFrame

```python
# Counts are already pandas for 2D primary modality
gene_counts = model.counts  # pandas DataFrame

# Convert numpy array to DataFrame
donor_mod = model.get_modality('splicing_donor')
# For 2D slice
donor_cell0 = donor_mod.counts[:, 0, :]  # donors × acceptors
donor_df = pd.DataFrame(
    donor_cell0,
    index=donor_mod.feature_meta.index,
    columns=[f'acceptor_{i}' for i in range(donor_cell0.shape[1])]
)
```

### Convert to AnnData (for scanpy)

```python
import anndata

# Create AnnData from gene modality
gene_mod = model.get_modality('gene')
adata = anndata.AnnData(
    X=gene_mod.counts.T,  # AnnData expects (cells × genes)
    obs=model.meta,       # Cell metadata
    var=gene_mod.feature_meta  # Gene metadata
)

# Add additional layers from other modalities
spliz_mod = model.get_modality('spliz')
adata.layers['spliz'] = spliz_mod.counts.T
```

---

## Accessing Model Results

### Technical Model Results

```python
# After running fit_ntc() on primary modality
# Technical fit uses model.counts (includes cis gene)
# It extracts SEPARATE parameters for cis vs trans

# 1. Alpha for cis gene (extracted during fit_ntc)
if hasattr(model, 'alpha_x_prefit'):
    alpha_x = model.alpha_x_prefit  # Cis gene overdispersion
    print(f"Alpha_x shape: {alpha_x.shape}")  # (n_samples, n_groups)
    print(f"Cis gene: {model.cis_gene}")

# 2. Alpha for trans genes (stored per-modality)
gene_mod = model.get_modality('gene')
if hasattr(gene_mod, 'alpha_y_prefit') and gene_mod.alpha_y_prefit is not None:
    alpha_y = gene_mod.alpha_y_prefit  # Trans gene overdispersion
    print(f"Alpha_y shape: {alpha_y.shape}")  # (n_samples, n_groups, n_trans_genes)

    # Convert to pandas for easier viewing
    # Take mean across samples if multiple samples
    if alpha_y.ndim == 3:
        alpha_mean = alpha_y.mean(dim=0)  # (n_groups, n_trans_genes)
    else:
        alpha_mean = alpha_y

    alpha_df = pd.DataFrame(
        alpha_mean.cpu().numpy().T,  # Transpose to (n_trans_genes, n_groups)
        index=model.trans_genes,
        columns=[f"group_{i}" for i in range(alpha_mean.shape[0])]
    )
    print(alpha_df.head())

# 3. Technical fit metadata (which features were excluded)
if hasattr(model, 'counts_meta'):
    print("\nTechnical fit metadata:")
    print(model.counts_meta.head())
    # Columns: ntc_zero_count, ntc_zero_std, ntc_single_category,
    #          ntc_excluded_from_fit, technical_correction_applied
```

### Cis Model Results

```python
# After running fit_cis()
if hasattr(model, 'x_true'):
    x_true = model.x_true  # Cis expression per guide
    print(f"Shape: {x_true.shape}")  # (n_guides,)

    # Create DataFrame with guide info
    guide_expr = pd.DataFrame({
        'guide': model.meta.groupby('guide').first().index,
        'x_true': x_true.cpu().numpy()
    })
    print(guide_expr)
```

### Trans Model Results

```python
# After running fit_trans()
if hasattr(model, 'posterior_samples_trans'):
    posterior = model.posterior_samples_trans

    # Available parameters depend on function_type
    # For 'additive_hill':
    print(posterior.keys())
    # ['Vmax_a', 'K_a', 'n_a', 'Vmax_b', 'K_b', 'n_b', 'pi_y', 'alpha', 'beta', ...]

    # Get component A Hill parameters
    Vmax_a = posterior['Vmax_a']  # (n_trans_genes,) - magnitude
    EC50_a = posterior['K_a']  # (n_trans_genes,) - half-max point
    n_a = posterior['n_a']  # (n_trans_genes,) - Hill coefficient
    print(f"Component A Vmax (magnitude): {Vmax_a}")
    print(f"Component A EC50 (half-max): {EC50_a}")
    print(f"Component A n (Hill coefficient): {n_a}")
```

---

## Working with Posterior Samples

### Extract Point Estimates

```python
# Cis posterior
x_true_samples = model.posterior_samples_cis['x_true']  # (n_samples, n_guides)
x_true_mean = x_true_samples.mean(dim=0)  # Mean across samples
x_true_std = x_true_samples.std(dim=0)    # Std across samples

# Trans posterior (additive_hill)
Vmax_a = model.posterior_samples_trans['Vmax_a']  # (n_samples, n_genes)
Vmax_a_mean = Vmax_a.mean(dim=0)  # Mean Vmax_a across samples
Vmax_a_std = Vmax_a.std(dim=0)    # Std Vmax_a across samples
```

### Credible Intervals

```python
import torch

# 95% credible interval for x_true
x_true_samples = model.posterior_samples_cis['x_true']
x_lower = torch.quantile(x_true_samples, 0.025, dim=0)
x_upper = torch.quantile(x_true_samples, 0.975, dim=0)

# Create summary DataFrame
x_summary = pd.DataFrame({
    'guide': model.meta.groupby('guide').first().index,
    'mean': x_true_mean.cpu().numpy(),
    'lower': x_lower.cpu().numpy(),
    'upper': x_upper.cpu().numpy()
})
```

### Save Posterior Samples

```python
import torch

# Save all posteriors
torch.save({
    'cis': model.posterior_samples_cis,
    'trans': model.posterior_samples_trans
}, 'posteriors.pt')

# Load later
posteriors = torch.load('posteriors.pt')
```

### Reconstruct Predictions

```python
import torch

# For additive Hill model
x_grid = torch.linspace(
    model.x_true.min(),
    model.x_true.max(),
    100
)

# Get mean parameters for component A and B
Vmax_a_mean = model.posterior_samples_trans['Vmax_a'].mean(dim=0)  # (n_genes,)
EC50_a_mean = model.posterior_samples_trans['K_a'].mean(dim=0)
n_a_mean = model.posterior_samples_trans['n_a'].mean(dim=0)

Vmax_b_mean = model.posterior_samples_trans['Vmax_b'].mean(dim=0)
EC50_b_mean = model.posterior_samples_trans['K_b'].mean(dim=0)
n_b_mean = model.posterior_samples_trans['n_b'].mean(dim=0)

# Predict for first gene
gene_idx = 0
# Component A
Hill_a = Vmax_a_mean[gene_idx] * (x_grid ** n_a_mean[gene_idx]) / (EC50_a_mean[gene_idx] ** n_a_mean[gene_idx] + x_grid ** n_a_mean[gene_idx])
# Component B
Hill_b = Vmax_b_mean[gene_idx] * (x_grid ** n_b_mean[gene_idx]) / (EC50_b_mean[gene_idx] ** n_b_mean[gene_idx] + x_grid ** n_b_mean[gene_idx])
# Total effect
y_pred = Hill_a + Hill_b

# Plot
import matplotlib.pyplot as plt
plt.plot(x_grid.cpu(), y_pred.cpu())
plt.xlabel('Cis expression (x)')
plt.ylabel('Trans expression (y)')
plt.title(f'Dose-response: {model.trans_genes[gene_idx]}')
```

---

## Common Data Access Patterns

### Compare NTC vs Perturbed Expression

```python
import numpy as np
import pandas as pd

# Get cell masks
ntc_mask = model.meta['target'] == 'ntc'
perturbed_mask = model.meta['target'] == model.cis_gene

# 1. Compare TRANS genes (from 'gene' modality)
gene_mod = model.get_modality('gene')
trans_counts = gene_mod.counts  # numpy array

# Calculate mean expression
ntc_trans_expr = trans_counts[:, ntc_mask].mean(axis=1)
pert_trans_expr = trans_counts[:, perturbed_mask].mean(axis=1)

# Calculate log fold change
trans_log2fc = np.log2((pert_trans_expr + 1) / (ntc_trans_expr + 1))

# Create DataFrame
trans_df = pd.DataFrame({
    'gene': gene_mod.feature_meta.index,
    'ntc_mean': ntc_trans_expr,
    'perturbed_mean': pert_trans_expr,
    'log2fc': trans_log2fc
})

# Find top changed trans genes
top_trans = trans_df.nlargest(20, 'log2fc', key=abs)
print("Top changed trans genes:")
print(top_trans)

# 2. Compare CIS gene (from 'cis' modality)
cis_mod = model.get_modality('cis')
cis_counts = cis_mod.counts[0, :]  # 1D array

ntc_cis_expr = cis_counts[ntc_mask].mean()
pert_cis_expr = cis_counts[perturbed_mask].mean()
cis_log2fc = np.log2((pert_cis_expr + 1) / (ntc_cis_expr + 1))

print(f"\nCis gene ({model.cis_gene}):")
print(f"  NTC mean: {ntc_cis_expr:.2f}")
print(f"  Perturbed mean: {pert_cis_expr:.2f}")
print(f"  Log2 FC: {cis_log2fc:.2f}")

# 3. For ALL genes together (from model.counts)
all_counts = model.counts  # pandas DataFrame
ntc_all = all_counts.loc[:, model.meta[ntc_mask]['cell']].mean(axis=1)
pert_all = all_counts.loc[:, model.meta[perturbed_mask]['cell']].mean(axis=1)
all_log2fc = np.log2((pert_all + 1) / (ntc_all + 1))

print(f"\nAll genes (including cis):")
print(all_log2fc.nlargest(20))
```

### Extract Guide-Level Summaries

```python
# Average expression per guide
guide_summary = []
for guide in model.meta['guide'].unique():
    guide_cells = model.meta[model.meta['guide'] == guide]['cell']
    guide_expr = model.counts[guide_cells].mean(axis=1)
    guide_summary.append({
        'guide': guide,
        'target': model.meta[model.meta['guide'] == guide]['target'].iloc[0],
        'n_cells': len(guide_cells),
        'mean_expression': guide_expr
    })

guide_df = pd.DataFrame(guide_summary)
```

### Access Splicing Changes

```python
# Get donor usage for NTC vs perturbed
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # (donors × cells × acceptors)

ntc_mask = model.meta['target'] == 'ntc'
cis_mask = model.meta['target'] == model.cis_gene

# Average usage across cells
ntc_usage = donor_counts[:, ntc_mask, :].mean(axis=1)  # (donors × acceptors)
cis_usage = donor_counts[:, cis_mask, :].mean(axis=1)

# Calculate usage difference
usage_diff = cis_usage - ntc_usage

# Find most changed donors
max_diff = np.abs(usage_diff).max(axis=1)
top_donors = np.argsort(max_diff)[-10:]  # Top 10

# Get metadata for top donors
print(donor_mod.feature_meta.iloc[top_donors])
```

---

## Changing Exon Skipping Aggregation

For exon skipping modalities, you can change how `inc1` and `inc2` are aggregated to compute inclusion:

```python
# Get exon skipping modality
exon_mod = model.get_modality('splicing_exon_skip')
print(f"Current method: {exon_mod.exon_aggregate_method}")  # e.g., 'min'

# Change to mean aggregation
exon_mod.set_exon_aggregate_method('mean')
# This recomputes exon_mod.counts (inclusion) and exon_mod.denominator (total)

# After fit_ntc(), changing is prevented by default
model.fit_ntc(covariates=['cell_line'], distribution='binomial', denominator=exon_mod.denominator)
exon_mod.mark_technical_fit_complete()  # Lock the aggregation method

try:
    exon_mod.set_exon_aggregate_method('min')  # Will raise ValueError
except ValueError as e:
    print(f"Prevented: {e}")

# Override if you really want to (invalidates technical fit parameters)
exon_mod.set_exon_aggregate_method('min', allow_after_technical_fit=True)
```

**Why lock after technical fit?**

The technical model estimates overdispersion parameters (`alpha_y`) based on the current aggregation method. Changing the aggregation method changes the inclusion counts, which would make the prefit parameters incorrect. The lock prevents accidental invalidation of these parameters.

---

## Working with Cis Modality Data

### Why the Cis Modality is Special

The 'cis' modality serves as the **reference for direct perturbation effects**:

```python
# The cis modality defines:
# 1. Which feature is being directly perturbed
# 2. The reference cell set (all other modalities subset to match)
# 3. The x_true values used in trans modeling

cis_mod = model.get_modality('cis')
print(f"Cis feature: {cis_mod.feature_meta.index[0]}")
print(f"Number of cells: {cis_mod.dims['n_cells']}")
print(f"Distribution: {cis_mod.distribution}")
```

### Accessing Cis Expression

```python
# Raw counts from cis modality
cis_mod = model.get_modality('cis')
cis_raw_counts = cis_mod.counts[0, :]  # 1D array (n_cells,)

# After fit_cis(), get posterior cis expression
if hasattr(model, 'x_true'):
    # x_true: guide-level cis expression
    x_true = model.x_true  # shape: (n_guides,)

    # Map to cells
    guide_to_x = dict(zip(
        model.meta.groupby('guide').first().index,
        x_true.cpu().numpy()
    ))

    # Add to metadata
    model.meta['x_true'] = model.meta['guide'].map(guide_to_x)
    print(model.meta[['cell', 'guide', 'x_true']].head())
```

### Verifying Cis Modality Extraction

```python
# Check that cis gene was correctly extracted
print(f"Cis gene specified: {model.cis_gene}")

cis_mod = model.get_modality('cis')
print(f"Cis modality feature: {cis_mod.feature_meta.index[0]}")
assert cis_mod.feature_meta.index[0] == model.cis_gene, "Cis gene mismatch!"

# Check that gene modality excludes cis gene
gene_mod = model.get_modality('gene')
assert model.cis_gene not in gene_mod.feature_meta.index, "Cis gene should not be in trans genes!"

# Check that model.counts includes cis gene
assert model.cis_gene in model.counts.index, "Cis gene should be in model.counts!"

print("✓ Cis modality correctly extracted")
```

### Using Cis Data in Analysis

```python
# Example: Plot cis gene expression vs posterior x_true
import matplotlib.pyplot as plt

cis_mod = model.get_modality('cis')
cis_counts = cis_mod.counts[0, :]

# Get guide-level means
guide_means = {}
for guide in model.meta['guide'].unique():
    guide_cells = model.meta[model.meta['guide'] == guide].index
    guide_means[guide] = cis_counts[guide_cells].mean()

# Compare to x_true
if hasattr(model, 'x_true'):
    guides = model.meta.groupby('guide').first().index
    x_true_vals = model.x_true.cpu().numpy()

    raw_means = [guide_means[g] for g in guides]

    plt.figure(figsize=(8, 6))
    plt.scatter(raw_means, x_true_vals, alpha=0.6)
    plt.xlabel('Raw count mean')
    plt.ylabel('Posterior x_true')
    plt.title(f'Cis gene ({model.cis_gene}): Raw vs Posterior')
    plt.plot([min(raw_means), max(raw_means)],
             [min(raw_means), max(raw_means)],
             'r--', alpha=0.5)
    plt.show()
```

### When Cis is Not a Gene

For ATAC or other modalities:

```python
# Initialize with ATAC as primary and cis feature
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    modality_name='atac',
    cis_feature='chr9:132283881-132284881',  # Use cis_feature instead of cis_gene
    guide_covariates=['cell_line']
)

# Access cis ATAC region
cis_mod = model.get_modality('cis')
cis_region = cis_mod.feature_meta.index[0]
print(f"Cis region: {cis_region}")

# Same workflow as with genes
cis_counts = cis_mod.counts[0, :]
# ... rest of analysis ...
```

---

## Tips and Best Practices

### 1. Always Check Data Dimensions

```python
mod = model.get_modality('my_modality')
print(f"Distribution: {mod.distribution}")
print(f"Dimensions: {mod.dims}")
print(f"Counts shape: {mod.counts.shape}")
```

### 2. Handle Missing Values

```python
# Check for NaN
print(f"NaN in counts: {np.isnan(mod.counts).sum()}")

# Check for Inf
print(f"Inf in counts: {np.isinf(mod.counts).sum()}")

# Filter out features with too many zeros
zero_frac = (mod.counts == 0).mean(axis=1)
keep_features = zero_frac < 0.9  # Keep features with <90% zeros
```

### 3. Memory-Efficient Access

```python
# For large datasets, don't load all data at once
# Instead, iterate over features
gene_mod = model.get_modality('gene')
n_genes = gene_mod.dims['n_features']

for i in range(0, n_genes, 100):  # Process 100 genes at a time
    subset_counts = gene_mod.counts[i:i+100, :]
    # ... process subset ...
```

### 4. Verify Cell-Modality Alignment

```python
# Check that all modalities have same cells
cell_counts = {}
for mod_name in model.modalities.keys():
    mod = model.get_modality(mod_name)
    cell_counts[mod_name] = mod.dims['n_cells']

print(f"All modalities aligned: {len(set(cell_counts.values())) == 1}")
```

### 5. Save and Load Modalities

```python
import pickle

# Save a modality
donor_mod = model.get_modality('splicing_donor')
with open('donor_mod.pkl', 'wb') as f:
    pickle.dump(donor_mod, f)

# Load later
with open('donor_mod.pkl', 'rb') as f:
    loaded_mod = pickle.load(f)
```
