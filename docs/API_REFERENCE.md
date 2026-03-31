# bayesDREAM API Reference

Complete reference for all classes, methods, and functions in bayesDREAM.

## Table of Contents

- [bayesDREAM Class](#bayesdream-class)
- [Modality Class](#modality-class)
- [Distribution Functions](#distribution-functions)
- [Splicing Functions](#splicing-functions)
- [Utility Functions](#utility-functions)
- [Plotting Methods](#plotting-methods)

---

## bayesDREAM Class

The main class for multi-modal Bayesian modeling of perturbation effects.

### Initialization

```python
bayesDREAM(
    meta,
    counts=None,
    modality_name='gene',
    feature_meta=None,
    cis_gene=None,
    cis_feature=None,
    guide_covariates=None,
    guide_covariates_ntc=None,
    sum_factor_col='sum_factor',
    output_dir='./model_out',
    label=None,
    device=None,
    random_seed=2402,
    cores=1
)
```

**Parameters:**
- `meta` (pd.DataFrame): Cell metadata with required columns:
  - `cell`: Cell barcode
  - `guide`: Guide RNA identifier
  - `target`: Target gene (use `'ntc'` for non-targeting controls)
  - `sum_factor`: Normalization factor per cell (column name set by `sum_factor_col`)
  - Additional covariates (e.g., `cell_line`) as needed
- `counts` (pd.DataFrame, optional): Count matrix for primary modality (features × cells)
  - Must represent negbinom count data for cis/trans modeling
  - If not provided, you must add a primary modality via `add_custom_modality()` before fitting
- `modality_name` (str): Name/type of the primary modality. Default: `'gene'`
  - Pre-set types: `'gene'`, `'atac'` (gene is most common)
  - Custom types: Any string (creates custom negbinom modality with that name)
  - The primary modality MUST be negative binomial for cis/trans modeling
- `feature_meta` (pd.DataFrame, optional): Feature-level metadata for primary modality
  - For `modality_name='gene'`: Recommended columns: `gene`, `gene_name`, `gene_id`
  - For other modalities: Relevant feature annotations
  - If not provided, minimal metadata created from `counts.index`
- `cis_gene` (str, optional): Feature to extract as 'cis' modality (gene name when `modality_name='gene'`)
  - Example: `cis_gene='GFI1B'`
  - Alias for `cis_feature` when `modality_name='gene'`
  - The cis feature will be extracted as a separate 'cis' modality and removed from primary modality
- `cis_feature` (str, optional): Alternative to `cis_gene` for non-gene modalities
  - For `'atac'`: region ID (e.g., `'chr9:132283881-132284881'`)
  - For custom modalities: feature identifier
  - Note: Cannot specify both `cis_gene` and `cis_feature`
- `guide_covariates` (list, optional): Covariates for guide grouping (e.g., `['cell_line', 'batch']`)
- `guide_covariates_ntc` (list, optional): Covariates for NTC guide grouping (if different from `guide_covariates`)
- `sum_factor_col` (str): Column name in meta containing size factors. Default: `'sum_factor'`
- `output_dir` (str): Output directory. Default: `'./model_out'`
- `label` (str, optional): Run label for organizing outputs
- `device` (str, optional): PyTorch device (`'cpu'` or `'cuda'`). Auto-detects if None
- `random_seed` (int): Random seed for reproducibility. Default: 2402
- `cores` (int): Number of CPU cores for parallelization. Default: 1

**Returns:** bayesDREAM instance

**Cis Modality Extraction:**
- The 'cis' modality is **ONLY** extracted during `bayesDREAM()` initialization
- The primary modality will contain **trans features only** (cis feature excluded)
- All modalities are automatically subset to cells present in the filtered metadata
- When calling `add_*_modality()` later, **NO** cis extraction occurs

**Examples:**
```python
# Basic usage with gene expression
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line'],
    output_dir='./output'
)
# Creates: 'cis' modality (GFI1B) + 'gene' modality (all other genes)

# With feature metadata
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    feature_meta=gene_metadata,
    cis_gene='GFI1B',
    guide_covariates=['cell_line']
)

# Custom negbinom modality
model = bayesDREAM(
    meta=meta,
    counts=my_counts,
    modality_name='my_custom_modality',
    cis_feature='feature_123',
    guide_covariates=['batch']
)
```

---

### Adding Modalities

#### add_modality()

```python
model.add_modality(name, modality, overwrite=False)
```

Add a pre-constructed Modality object.

**Parameters:**
- `name` (str): Name for this modality
- `modality` (Modality): Pre-constructed Modality object
- `overwrite` (bool): Whether to overwrite if name exists. Default: False

**Example:**
```python
custom_mod = Modality(
    name='my_data',
    counts=data_array,
    feature_meta=feature_df,
    distribution='normal'
)
model.add_modality('my_data', custom_mod)
```

---

#### add_transcript_modality()

```python
model.add_transcript_modality(
    transcript_counts,
    transcript_meta,
    modality_types='counts',
    gene_col=None
)
```

Add transcript-level data as counts and/or isoform usage.

**Parameters:**
- `transcript_counts` (pd.DataFrame): Transcript counts (transcripts × cells)
- `transcript_meta` (pd.DataFrame): Transcript metadata with required columns:
  - `transcript_id`: Transcript identifier
  - One of: `gene`, `gene_name`, or `gene_id` (gene assignment)
- `modality_types` (str or list): Type(s) to create:
  - `'counts'`: Transcript counts as negative binomial (2D)
  - `'usage'`: Isoform usage as multinomial (3D, grouped by gene)
  - `['counts', 'usage']`: Create both
- `gene_col` (str, optional): Column name for gene assignment. Auto-detects from: `gene`, `gene_name`, `gene_id`

**Modalities Created:**
- `'transcript_counts'`: Distribution = `negbinom`, shape = (transcripts, cells)
- `'transcript_usage'`: Distribution = `multinomial`, shape = (genes, cells, max_isoforms)

**Example:**
```python
model.add_transcript_modality(
    transcript_counts=tx_counts,
    transcript_meta=tx_meta,
    modality_types=['counts', 'usage']
)
```

---

#### add_splicing_modality()

```python
model.add_splicing_modality(
    sj_counts,
    sj_meta,
    splicing_types=['donor', 'acceptor'],
    gene_counts=None,
    min_cell_total=1,
    min_total_exon=2,
    exon_aggregate='min'
)
```

Add splicing data (splice junctions, donor/acceptor usage, exon skipping).

**Parameters:**
- `sj_counts` (pd.DataFrame): Splice junction counts (junctions × cells)
- `sj_meta` (pd.DataFrame): Junction metadata with required columns:
  - `coord.intron`: Junction ID (e.g., "chr1:12345:67890:+")
  - `chrom`: Chromosome
  - `intron_start`, `intron_end`: Junction coordinates
  - `strand`: Strand ('+', '-', 1, or 2)
  - `gene_name_start`, `gene_name_end`: Gene names at junction boundaries
  - Optional: `gene_id_start`, `gene_id_end` (Ensembl IDs)
- `splicing_types` (list): Types to create:
  - `'sj'`: Raw SJ counts (binomial with gene expression denominator)
  - `'donor'`: Donor site usage (multinomial)
  - `'acceptor'`: Acceptor site usage (multinomial)
  - `'exon_skip'`: Exon skipping events (binomial)
- `gene_counts` (pd.DataFrame, optional): Gene counts for 'sj' type denominator. Defaults to `self.counts`
- `min_cell_total` (int): Minimum reads per donor/acceptor. Default: 1
- `min_total_exon` (int): Minimum reads for exon skipping. Default: 2
- `exon_aggregate` (str): How to aggregate inc1/inc2 for exon skipping: `'min'` or `'mean'`

**Modalities Created:**
- `'splicing_sj'`: Distribution = `binomial`, shape = (junctions, cells)
- `'splicing_donor'`: Distribution = `multinomial`, shape = (donors, cells, acceptors)
- `'splicing_acceptor'`: Distribution = `multinomial`, shape = (acceptors, cells, donors)
- `'splicing_exon_skip'`: Distribution = `binomial`, shape = (events, cells)

**Example:**
```python
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip']
)
```

---

#### add_custom_modality()

```python
model.add_custom_modality(
    name,
    counts,
    feature_meta,
    distribution,
    denominator=None
)
```

Add a custom modality with user-defined measurements.

**Parameters:**
- `name` (str): Name for this modality
- `counts` (np.ndarray or pd.DataFrame): Measurement data
  - 2D for: `negbinom`, `normal`, `studentt`, `binomial`
  - 3D for: `multinomial`
- `feature_meta` (pd.DataFrame): Feature-level metadata
- `distribution` (str): Distribution type:
  - `'negbinom'`: Negative binomial (count data)
  - `'normal'`: Normal (continuous measurements)
  - `'studentt'`: Student's t (heavy-tailed continuous)
  - `'binomial'`: Binomial (proportions with denominator)
  - `'multinomial'`: Multinomial (categorical)
- `denominator` (np.ndarray, optional): Required for `binomial` distribution

**Example:**
```python
# SpliZ scores (normal distribution)
model.add_custom_modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=gene_meta,
    distribution='normal'
)
```

---

#### add_atac_modality()

```python
model.add_atac_modality(
    atac_counts,
    region_meta,
    name='atac',
    cis_region=None,
    cell_names=None,
    overwrite=False
)
```

Add ATAC-seq modality with genomic region annotations.

**Parameters:**
- `atac_counts` (np.ndarray or pd.DataFrame): Fragment counts per region (regions × cells)
- `region_meta` (pd.DataFrame): Region metadata with required columns:
  - `region_id`: Unique region identifier
  - `region_type`: One of `['promoter', 'gene_body', 'distal']`
  - `chrom`: Chromosome
  - `start`, `end`: Coordinates (0-based)
  - `gene`: Associated gene (NA for distal regions)
  - Optional: `gene_name`, `gene_id`, `strand`, `tss_distance`
- `name` (str): Modality name. Default: `'atac'`
- `cis_region` (str, optional): Region ID to use as cis proxy
  - Only creates 'cis' modality if no 'cis' modality exists
  - Otherwise ignored with warning
- `cell_names` (list, optional): Cell identifiers (only for np.ndarray counts)
- `overwrite` (bool): Overwrite existing modality. Default: False

**Notes:**
- ATAC counts are treated as negative binomial data (like gene expression)
- Zero-std regions are automatically filtered
- Cis extraction only happens if no 'cis' modality exists AND `cis_region` is specified

**Examples:**
```python
# Add ATAC as secondary modality (gene expression is primary/cis)
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta
)

# Use ATAC region as cis proxy (creates 'cis' modality)
model = bayesDREAM(meta=meta, counts=gene_counts)  # No cis_gene specified
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    cis_region='chr9:132283881-132284881'  # Creates 'cis' from this region
)
```

---

### Accessing Modalities

#### get_modality()

```python
model.get_modality(name)
```

Retrieve a specific modality by name.

**Parameters:**
- `name` (str): Modality name

**Returns:** Modality object

**Raises:** KeyError if modality doesn't exist

**Example:**
```python
donor_mod = model.get_modality('splicing_donor')
print(donor_mod.dims)  # {'n_features': 100, 'n_cells': 500, 'n_categories': 10}
```

---

#### list_modalities()

```python
model.list_modalities()
```

Get a summary table of all modalities.

**Returns:** pd.DataFrame with columns:
- `name`: Modality name
- `distribution`: Distribution type
- `n_features`: Number of features
- `n_cells`: Number of cells
- `n_categories` (if applicable): Number of categories (multinomial)

**Example:**
```python
print(model.list_modalities())
#                   name distribution  n_features  n_cells  n_categories
# 0                 gene     negbinom        1000      500           NaN
# 1    splicing_donor  multinomial          50      500          10.0
# 2                spliz       normal        1000      500           NaN
```

---

### Modeling Pipeline

#### set_technical_groups()

```python
model.set_technical_groups(covariates)
```

Set technical group codes based on covariates. **Must be called before `fit_technical()`**.

**Parameters:**
- `covariates` (list of str): Column names in meta to group by (e.g., `['cell_line']`)

**Side Effects:**
- Adds `technical_group_code` column to `self.meta`

**Example:**
```python
model.set_technical_groups(['cell_line'])
```

---

#### fit_technical()

```python
model.fit_technical(
    sum_factor_col='sum_factor',
    lr=1e-3,
    niters=None,
    nsamples=1000,
    alpha_ewma=0.05,
    tolerance=1e-4,
    beta_o_beta=3,
    beta_o_alpha=9,
    epsilon=1e-6,
    minibatch_size=None,
    distribution=None,
    denominator=None,
    modality_name=None,
    use_all_cells=False
)
```

Fit technical model to estimate baseline overdispersion and cell-line effects from non-targeting controls.

**Important:** Call `set_technical_groups()` first to define technical groups.

**Primary Modality Behavior:**
- When fitting the **primary modality**, uses original counts that **include the cis feature**
- Automatically extracts `alpha_x_prefit` for the cis feature after fitting
- Stores `alpha_y_prefit` for all trans features (excluding cis)
- Feature metadata stored in `self.counts_meta` (for all features including cis)

**Parameters:**
- `sum_factor_col` (str): Column in `meta` with normalization factors. Default: `'sum_factor'`
  - Required for `negbinom` distribution
  - Not used for `normal`
- `lr` (float): Learning rate. Default: 1e-3
- `niters` (int, optional): Number of optimization steps. Default: None (auto-selects based on distribution)
  - Univariate distributions (negbinom, normal, etc.): 50,000
  - Multivariate distributions (multinomial): 100,000
- `nsamples` (int): Number of posterior samples. Default: 1,000
- `alpha_ewma` (float): EWMA smoothing parameter for convergence. Default: 0.05
- `tolerance` (float): Convergence tolerance. Default: 1e-4
- `beta_o_beta` (float): Beta prior for overdispersion. Default: 3
- `beta_o_alpha` (float): Alpha prior for overdispersion. Default: 9
- `epsilon` (float): Small constant for numerical stability. Default: 1e-6
- `minibatch_size` (int, optional): Minibatch size for predictive sampling
- `distribution` (str, optional): Distribution type. Defaults to modality's distribution
  - `'negbinom'`: Negative binomial (count data)
  - `'normal'`: Normal (continuous measurements)
  - `'studentt'`: Student's t (heavy-tailed continuous)
  - `'binomial'`: Binomial (proportions)
  - `'multinomial'`: Multinomial (categorical)
- `denominator` (np.ndarray, optional): Required for `binomial` distribution
- `modality_name` (str, optional): Modality to fit. Defaults to primary modality
- `use_all_cells` (bool): If False (default), fit using NTC cells only. If True, fit using all cells.
  - Use True for high MOI experiments where technical effects are batch/lane specific
  - Use False (default) when technical groups correlate with cis gene expression

**Side Effects:**
- Sets `self.alpha_y_prefit` (overdispersion parameters for trans features)
- Sets `self.alpha_x_prefit` (overdispersion for cis feature, if primary modality)
- Sets `self.alpha_y_type` and `self.alpha_x_type` to `'posterior'`
- Sets `self.posterior_samples_technical` (full posterior samples)
- Sets `self.loss_technical` (optimization loss history)
- For modalities: sets `modality.alpha_y_prefit_mult` and `modality.alpha_y_prefit_add`
- For primary modality with original counts: creates `self.counts_meta` DataFrame with filtering flags

**Example:**
```python
# Standard usage with gene counts
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
# For 92 genes (including GFI1B cis gene):
# - Fits all 92 features
# - Extracts alpha_x_prefit for GFI1B (shape: [nsamples, 2])
# - Stores alpha_y_prefit for 91 trans genes (shape: [nsamples, 2, 91])

# Continuous scores (normal distribution)
model.set_technical_groups(['cell_line'])
model.fit_technical(distribution='normal')

# Specific modality
model.fit_technical(modality_name='splicing_donor', distribution='multinomial')

# High MOI: use all cells for technical estimation
model.set_technical_groups(['lane', 'batch'])
model.fit_technical(use_all_cells=True)
```

---

#### fit_cis()

```python
model.fit_cis(
    technical_covariates=None,
    sum_factor_col='sum_factor',
    cis_feature=None,
    manual_guide_effects=None,
    prior_strength=1.0,
    lr=1e-3,
    niters=100000,
    nsamples=1000,
    alpha_ewma=0.05,
    tolerance=1e-4,
    beta_o_beta=3,
    beta_o_alpha=9,
    alpha_alpha_mu=5.8,
    epsilon=1e-6,
    alpha_dirichlet=0.1,
    minibatch_size=None,
    independent_mu_sigma=False
)
```

Fit cis model to estimate direct effects on the targeted feature.

**Modality Used:**
- Always uses the **'cis' modality** (extracted during initialization)
- Consistent interface regardless of primary modality type (gene, ATAC, etc.)

**Parameters:**
- `technical_covariates` (list, optional): Technical covariates for correction (e.g., `['cell_line']`)
  - If provided and `technical_group_code` not already set, creates the grouping
- `sum_factor_col` (str): Column with normalization factors. Default: `'sum_factor'`
- `cis_feature` (str, optional): Specific feature in 'cis' modality to use
  - If None, uses the first (and typically only) feature in 'cis' modality
  - Rarely needed since 'cis' modality usually contains exactly one feature
- `manual_guide_effects` (pd.DataFrame, optional): Manual guide effect estimates as priors
  - Required columns: `guide` (identifier), `log2FC` (expected log2 fold-change vs NTC)
  - Used for prior-informed fitting (infrastructure in place, integration in progress)
- `prior_strength` (float): Weight for manual guide effects. Default: 1.0
  - 0 = ignore manual effects, higher = trust more
- `lr` (float): Learning rate. Default: 1e-3
- `niters` (int): Number of optimization steps. Default: 100,000
- `nsamples` (int): Number of posterior samples. Default: 1,000
- `alpha_ewma` (float): EWMA smoothing parameter. Default: 0.05
- `tolerance` (float): Convergence tolerance. Default: 1e-4
- `beta_o_beta` (float): Beta prior for overdispersion. Default: 3
- `beta_o_alpha` (float): Alpha prior for overdispersion. Default: 9
- `alpha_alpha_mu` (float): Mean for alpha hyperprior. Default: 5.8
- `epsilon` (float): Small constant for numerical stability. Default: 1e-6
- `alpha_dirichlet` (float): Dirichlet concentration. Default: 0.1
- `minibatch_size` (int, optional): Minibatch size for predictive sampling
- `independent_mu_sigma` (bool): Whether to use independent mu/sigma per target type. Default: False
  - Requires `target` column in meta with >1 unique values

**Side Effects:**
- Sets `self.x_true` (posterior cis expression per cell)
- Sets `self.x_true_type` to `'posterior'`
- Sets `self.posterior_samples_cis` (full posterior samples)
- Sets `self.loss_x` (optimization loss history)

**Example:**
```python
model.fit_cis(sum_factor_col='sum_factor')
# Uses 'cis' modality (e.g., just GFI1B for gene modality)
# Sets self.x_true with shape: [nsamples, n_cells]

# With technical covariates
model.fit_cis(
    technical_covariates=['cell_line'],
    sum_factor_col='sum_factor'
)

# With independent mu/sigma per target type
model.fit_cis(
    sum_factor_col='sum_factor',
    independent_mu_sigma=True
)
```

---

#### fit_trans()

```python
model.fit_trans(
    sum_factor_col=None,
    function_type='single_hill',
    polynomial_degree=6,
    lr=None,
    niters=None,
    nsamples=1000,
    alpha_ewma=0.05,
    tolerance=1e-4,
    beta_o_beta=3,
    beta_o_alpha=9,
    alpha_alpha_mu=5.8,
    K_alpha=2,
    Vmax_alpha=2,
    n_mu=0,
    p_n=1e-6,
    epsilon=1e-6,
    init_temp=1.0,
    final_temp=0.1,
    minibatch_size=None,
    distribution=None,
    denominator=None,
    modality_name=None,
    min_denominator=None,
    use_data_driven_priors=True,
    correct_priors_for_technical=True,
    use_archive_prior_computation=False,
    use_epsilon=False
)
```

Fit trans model to estimate downstream effects as a function of cis expression.

**Parameters:**
- `sum_factor_col` (str, optional): Column with normalization factors. Required for `negbinom`
- `function_type` (str): Dose-response function type. Default: `'single_hill'`
  - `'single_hill'`: Single Hill equation
  - `'additive_hill'`: Sum of positive and negative Hill functions
  - `'polynomial'`: Polynomial function
- `polynomial_degree` (int): Degree for polynomial function. Default: 6
- `lr` (float, optional): Learning rate. Default: None (auto-selects based on distribution)
  - negbinom: 0.05, others: 0.01
- `niters` (int, optional): Number of optimization steps. Default: None (auto-selects)
  - negbinom: 50,000, multinomial: 100,000, others: 50,000
- `nsamples` (int): Number of posterior samples. Default: 1,000
- `alpha_ewma` (float): EWMA smoothing parameter. Default: 0.05
- `tolerance` (float): Convergence tolerance. Default: 1e-4
- `beta_o_beta` (float): Beta prior for overdispersion. Default: 3
- `beta_o_alpha` (float): Alpha prior for overdispersion. Default: 9
- `alpha_alpha_mu` (float): Mean for alpha hyperprior. Default: 5.8
- `K_alpha` (float): Alpha for K (EC50) prior. Default: 2
- `Vmax_alpha` (float): Alpha for Vmax prior. Default: 2
- `n_mu` (float): Mean for Hill coefficient prior. Default: 0
- `p_n` (float): Prior probability for effect. Default: 1e-6
- `epsilon` (float): Small constant for numerical stability. Default: 1e-6
- `init_temp` (float): Initial temperature for relaxed Bernoulli. Default: 1.0
- `final_temp` (float): Final temperature (annealed during training). Default: 0.1
- `minibatch_size` (int, optional): Minibatch size for predictive sampling
- `distribution` (str, optional): Distribution type. Auto-detected from modality if None
  - `'negbinom'`: Negative binomial (count data)
  - `'normal'`: Normal (continuous measurements)
  - `'studentt'`: Student's t (heavy-tailed continuous)
  - `'binomial'`: Binomial (proportions)
  - `'multinomial'`: Multinomial (categorical)
- `denominator` (np.ndarray, optional): Required for `binomial`. Auto-detected from modality
- `modality_name` (str, optional): Modality to fit. Defaults to primary modality
- `min_denominator` (int, optional): Minimum denominator for binomial observations
  - Observations with denominator < min_denominator are masked (excluded)
  - Useful for filtering low-coverage splicing junctions
- `use_data_driven_priors` (bool): Use data-driven Beta/Dirichlet priors. Default: True
- `correct_priors_for_technical` (bool): Correct data for technical effects before computing priors. Default: True
- `use_archive_prior_computation` (bool): Use archive method for Amean/Vmax_mean. Default: False
- `use_epsilon` (bool): Add epsilon for numerical stability in NegativeBinomial. Default: False

**Side Effects:**
- Sets `self.posterior_samples_trans` (dose-response parameters)
- Sets `self.loss_trans` (optimization loss history)
- For modalities: sets `modality.posterior_samples_trans`

**Example:**
```python
# Gene counts (negbinom)
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    distribution='negbinom'
)

# Continuous (normal)
model.fit_trans(
    distribution='normal',
    function_type='polynomial'
)

# Exon skipping (binomial) with minimum coverage filter
model.fit_trans(
    distribution='binomial',
    denominator=total_counts,
    function_type='single_hill',
    min_denominator=5
)

# Specific modality with custom priors
model.fit_trans(
    modality_name='splicing_donor',
    distribution='multinomial',
    use_data_driven_priors=True
)
```

---

### Save/Load Methods

#### save_technical_fit()

```python
model.save_technical_fit(
    file_path=None,
    modalities=None,
    metadata=None
)
```

Save technical fitting results to HDF5 file.

**Parameters:**
- `file_path` (str, optional): Custom save path. Defaults to `{output_dir}/{label}/technical_fit.h5`
- `modalities` (list, optional): Modalities to save. Defaults to all with technical fits
- `metadata` (dict, optional): Additional metadata to store

**Saves:**
- Posterior samples for alpha_y (overdispersion)
- Loss history
- Modality-specific alpha_y parameters (mult/add)
- Technical group information

**Example:**
```python
model.save_technical_fit()
# Saves to: ./output/my_run/technical_fit.h5
```

---

#### load_technical_fit()

```python
model.load_technical_fit(
    file_path=None,
    modalities=None
)
```

Load technical fitting results from HDF5 file.

**Parameters:**
- `file_path` (str, optional): Path to load from. Defaults to `{output_dir}/{label}/technical_fit.h5`
- `modalities` (list, optional): Modalities to load. Defaults to all in file

**Restores:**
- `self.posterior_samples_technical`
- `self.loss_technical`
- `modality.alpha_y_prefit_mult` and `modality.alpha_y_prefit_add`

**Example:**
```python
model.load_technical_fit()
```

---

#### save_cis_fit()

```python
model.save_cis_fit(
    file_path=None,
    metadata=None
)
```

Save cis fitting results to HDF5 file.

**Parameters:**
- `file_path` (str, optional): Custom save path. Defaults to `{output_dir}/{label}/cis_fit.h5`
- `metadata` (dict, optional): Additional metadata

**Saves:**
- Posterior samples for x_true (cis expression per guide)
- Posterior samples for cis model parameters
- Loss history

**Example:**
```python
model.save_cis_fit()
```

---

#### load_cis_fit()

```python
model.load_cis_fit(
    file_path=None
)
```

Load cis fitting results from HDF5 file.

**Parameters:**
- `file_path` (str, optional): Path to load from

**Restores:**
- `self.x_true`
- `self.posterior_samples_cis`
- `self.loss_cis`

**Example:**
```python
model.load_cis_fit()
```

---

#### save_trans_fit()

```python
model.save_trans_fit(
    file_path=None,
    modalities=None,
    suffix=None,
    metadata=None
)
```

Save trans fitting results to HDF5 file.

**Parameters:**
- `file_path` (str, optional): Custom save path. Defaults to `{output_dir}/{label}/trans_fit_{suffix}.h5`
- `modalities` (list, optional): Modalities to save. Defaults to last fit modality
- `suffix` (str, optional): Filename suffix (e.g., function type). Default: 'default'
- `metadata` (dict, optional): Additional metadata

**Saves:**
- Posterior samples for dose-response parameters (A, K, n, etc.)
- Loss history
- Function type and modality information

**Example:**
```python
# Save with descriptive suffix
model.save_trans_fit(suffix='additive_hill')
# Saves to: ./output/my_run/trans_fit_additive_hill.h5
```

---

#### load_trans_fit()

```python
model.load_trans_fit(
    file_path=None,
    modalities=None,
    suffix='default'
)
```

Load trans fitting results from HDF5 file.

**Parameters:**
- `file_path` (str, optional): Path to load from
- `modalities` (list, optional): Modalities to load
- `suffix` (str): Filename suffix to load. Default: 'default'

**Restores:**
- `self.posterior_samples_trans`
- `self.loss_trans`
- Modality-specific posterior samples

**Example:**
```python
model.load_trans_fit(suffix='additive_hill')
```

---

#### save_technical_summary()

```python
model.save_technical_summary(
    output_dir=None,
    modalities=None
)
```

Export technical fit results as CSV files for downstream analysis (e.g., R plotting).

**Parameters:**
- `output_dir` (str, optional): Directory for CSV files. Defaults to `{output_dir}/{label}/summaries/technical/`
- `modalities` (list, optional): Modalities to export. Defaults to all with technical fits

**Exports (per modality):**
- `alpha_y_summary.csv`: Overdispersion parameter summaries (mean, SD, quantiles)
- Modality-specific parameter files

**Example:**
```python
model.save_technical_summary()
```

---

#### save_cis_summary()

```python
model.save_cis_summary(
    output_dir=None
)
```

Export cis fit results as CSV files.

**Parameters:**
- `output_dir` (str, optional): Directory for CSV files. Defaults to `{output_dir}/{label}/summaries/cis/`

**Exports:**
- `x_true_summary.csv`: Cis expression summaries per guide

**Example:**
```python
model.save_cis_summary()
```

---

#### save_trans_summary()

```python
model.save_trans_summary(
    output_dir=None,
    modalities=None,
    suffix='default'
)
```

Export trans fit results as CSV files.

**Parameters:**
- `output_dir` (str, optional): Directory for CSV files. Defaults to `{output_dir}/{label}/summaries/trans_{suffix}/`
- `modalities` (list, optional): Modalities to export. Defaults to last fit modality
- `suffix` (str): Identifier for this fit. Default: 'default'

**Exports (varies by function_type):**
- `A_summary.csv`: Amplitude parameters
- `K_summary.csv`: EC50/IC50 parameters
- `n_summary.csv`: Hill coefficients
- Feature metadata with statistical summaries

**Example:**
```python
model.save_trans_summary(suffix='additive_hill')
```

---

### Utility Methods

#### adjust_ntc_sum_factor()

```python
model.adjust_ntc_sum_factor(
    covariates,
    sum_factor_col_old='sum_factor',
    sum_factor_col_adj='sum_factor_adj'
)
```

Adjust sum factors based on NTC mean within covariate groups.

**Parameters:**
- `covariates` (list of str): Covariates to group by
- `sum_factor_col_old` (str): Input sum factor column. Default: `'sum_factor'`
- `sum_factor_col_adj` (str): Output adjusted column. Default: `'sum_factor_adj'`

**Side Effects:**
- Adds `sum_factor_col_adj` column to `self.meta`
- Adds `adjustment_factor` column to `self.meta`

**Example:**
```python
model.adjust_ntc_sum_factor(covariates=['lane', 'cell_line'])
```

---

#### set_alpha_x()

```python
model.set_alpha_x(alpha_x, is_posterior=False, covariates=None)
```

Set overdispersion parameters for cis gene.

**Parameters:**
- `alpha_x` (array-like): Overdispersion values
- `is_posterior` (bool): Whether these are posterior samples. Default: False
- `covariates` (list, optional): Covariate names if applicable

---

#### set_alpha_y()

```python
model.set_alpha_y(alpha_y, is_posterior=False, covariates=None)
```

Set overdispersion parameters for trans genes.

**Parameters:**
- `alpha_y` (array-like): Overdispersion values
- `is_posterior` (bool): Whether these are posterior samples. Default: False
- `covariates` (list, optional): Covariate names if applicable

---

#### set_x_true()

```python
model.set_x_true(x_true, is_posterior=False)
```

Set cis expression values (e.g., from external estimates).

**Parameters:**
- `x_true` (array-like): Cis expression values
- `is_posterior` (bool): Whether these are posterior samples. Default: False

---

#### set_o_x()

```python
model.set_o_x(o_x, is_posterior=False)
```

Set overdispersion parameter for cis gene.

**Parameters:**
- `o_x` (float or array-like): Overdispersion values
  - If posterior: shape `(S,)`, `(S, 1)`, or `(S, 1, 1)` where S = number of samples
  - If point estimate: scalar or 1-element tensor
- `is_posterior` (bool): Whether these are posterior samples. Default: False

---

#### set_o_x_grouped()

```python
model.set_o_x_grouped(o_x, is_posterior, covariates=None)
```

Set grouped overdispersion parameter (one per technical group).

**Parameters:**
- `o_x` (array-like): Overdispersion values
  - Point estimate: shape `(C, 1)` where C = number of technical groups
  - Posterior: shape `(S, C, 1)` where S = number of samples
- `is_posterior` (bool): Whether these are posterior samples
- `covariates` (list, optional): Technical group covariates (e.g., `['cell_line']`)
  - If provided, creates `technical_group_code` column
  - If None, uses existing `technical_group_code`

---

#### subset_cells()

```python
model.subset_cells(cell_mask=None, query=None, preserve_fits=True)
```

Create a new model instance with a subset of cells.

Useful for testing without technical correction by subsetting to a single cell_line
(e.g., CRISPRi or CRISPRa only).

**Parameters:**
- `cell_mask` (np.ndarray, pd.Series, or list, optional): Boolean mask or list of cell names to keep
- `query` (str, optional): Pandas query string to filter cells (e.g., `"cell_line == 'CRISPRi'"`)
  - Must provide either `cell_mask` or `query`, not both
- `preserve_fits` (bool): Whether to preserve existing fit results. Default: True
  - If True, copies `posterior_samples_technical`, `alpha_y_prefit`, etc.
  - If False, creates a fresh model without fit results

**Returns:** New bayesDREAM instance with subset of cells

**Example:**
```python
# Subset to CRISPRi cells only
crispri_model = model.subset_cells(query="cell_line == 'CRISPRi'")

# Subset using boolean mask
mask = model.meta['target'] != 'ntc'
perturbed_model = model.subset_cells(cell_mask=mask)
```

---

#### permute_genes()

```python
model.permute_genes(
    genes2permute,
    sum_factor_col='sum_factor',
    function_type='additive_hill',
    n_steps=10000,
    lr=0.01
)
```

Permute guide-gene associations for null testing.

**Parameters:**
- `genes2permute` (list): Gene names to permute
- `sum_factor_col` (str): Normalization factor column
- `function_type` (str): Dose-response function
- `n_steps` (int): Optimization steps
- `lr` (float): Learning rate

**Side Effects:**
- Sets `self.posterior_samples_trans_permuted`

---

#### refit_sumfactor()

```python
model.refit_sumfactor(
    covariates,
    sum_factor_col_old='sum_factor',
    sum_factor_col_new='sum_factor_refit'
)
```

Re-estimate sum factors based on posterior cis expression.

**Parameters:**
- `covariates` (list): Covariates for adjustment
- `sum_factor_col_old` (str): Input sum factor column
- `sum_factor_col_new` (str): Output refitted column

---

## Modality Class

Container for modality-specific data and metadata.

### Initialization

```python
Modality(
    name,
    counts,
    feature_meta,
    distribution,
    denominator=None
)
```

**Parameters:**
- `name` (str): Modality name
- `counts` (np.ndarray): Measurement data
  - 2D `(features, cells)`: negbinom, normal, studentt, binomial
  - 3D `(features, cells, categories)`: multinomial
- `feature_meta` (pd.DataFrame): Feature-level metadata
- `distribution` (str): One of: `'negbinom'`, `'normal'`, `'studentt'`, `'binomial'`, `'multinomial'`
- `denominator` (np.ndarray, optional): For binomial only

**Attributes:**
- `name`: Modality name
- `counts`: Data array
- `feature_meta`: Metadata DataFrame
- `distribution`: Distribution type
- `denominator`: Denominator array (binomial only)
- `inc1`: Inclusion junction 1 counts (exon skipping only)
- `inc2`: Inclusion junction 2 counts (exon skipping only)
- `skip`: Skipping junction counts (exon skipping only)
- `exon_aggregate_method`: How inc1/inc2 are aggregated ('min' or 'mean', exon skipping only)
- `dims`: Dictionary with dimensions:
  - `n_features`: Number of features
  - `n_cells`: Number of cells
  - `n_categories` (multinomial): Number of categories

---

### Methods

#### to_tensor()

```python
modality.to_tensor(device='cpu')
```

Convert counts to PyTorch tensor.

**Parameters:**
- `device` (str): PyTorch device

**Returns:** torch.Tensor

---

#### get_feature_subset()

```python
modality.get_feature_subset(feature_indices)
```

Subset to specific features.

**Parameters:**
- `feature_indices` (list or array): Feature indices or names

**Returns:** New Modality object

---

#### get_cell_subset()

```python
modality.get_cell_subset(cell_indices)
```

Subset to specific cells.

**Parameters:**
- `cell_indices` (list or array): Cell indices

**Returns:** New Modality object

---

#### set_exon_aggregate_method()

```python
modality.set_exon_aggregate_method(method, allow_after_technical_fit=False)
```

Change the exon skipping aggregation method and recompute inclusion counts.

Only available for exon skipping modalities (those with `inc1`, `inc2`, `skip` data).

**Parameters:**
- `method` (str): Aggregation method - `'min'` or `'mean'`
- `allow_after_technical_fit` (bool): Allow changing after technical fit. Default: False

**Raises:** ValueError if invalid method, not an exon skipping modality, or changing after technical fit without permission

**Example:**
```python
exon_mod = model.get_modality('splicing_exon_skip')
exon_mod.set_exon_aggregate_method('mean')  # Changes from 'min' to 'mean'
```

---

#### is_exon_skipping()

```python
modality.is_exon_skipping()
```

Check if this is an exon skipping modality with inc1/inc2/skip data.

**Returns:** bool

---

#### mark_technical_fit_complete()

```python
modality.mark_technical_fit_complete()
```

Mark that technical fit has been performed with current aggregation method.

This locks the aggregation method to prevent accidental changes that would invalidate the prefit overdispersion parameters.

---

## Distribution Functions

From `bayesDREAM.distributions` module.

### get_observation_sampler()

```python
get_observation_sampler(distribution, model_type)
```

Get appropriate observation sampler for a distribution and model.

**Parameters:**
- `distribution` (str): Distribution name
- `model_type` (str): Either `'technical'` or `'trans'`

**Returns:** Sampler function

**Example:**
```python
from bayesDREAM import get_observation_sampler
sampler = get_observation_sampler('multinomial', 'trans')
```

---

### requires_denominator()

```python
requires_denominator(distribution)
```

Check if distribution requires a denominator.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'binomial'`)

---

### requires_sum_factor()

```python
requires_sum_factor(distribution)
```

Check if distribution requires sum factors.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'negbinom'`)

---

### is_3d_distribution()

```python
is_3d_distribution(distribution)
```

Check if distribution uses 3D data.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'multinomial'`)

---

### supports_cell_line_effects()

```python
supports_cell_line_effects(distribution)
```

Check if distribution supports cell-line covariate effects.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool

---

### get_cell_line_effect_type()

```python
get_cell_line_effect_type(distribution)
```

Get how cell-line effects are applied for a distribution.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** str - One of:
- `'multiplicative'`: Effect multiplies mu (negbinom)
- `'additive'`: Effect adds to mu (normal)
- `'logit'`: Effect on logit scale (binomial)
- `None`: Not supported (multinomial)

---

## Splicing Functions

From `bayesDREAM.splicing` module.

### create_splicing_modality()

```python
create_splicing_modality(
    sj_counts,
    sj_meta,
    splicing_type,
    gene_counts=None,
    min_cell_total=1,
    min_total_exon=2,
    exon_aggregate='min'
)
```

High-level function to create a splicing modality.

**Parameters:**
- `sj_counts` (pd.DataFrame): Splice junction counts
- `sj_meta` (pd.DataFrame): Junction metadata
- `splicing_type` (str): Type to create (`'sj'`, `'donor'`, `'acceptor'`, `'exon_skip'`)
- `gene_counts` (pd.DataFrame, optional): Gene counts for 'sj' denominator
- `min_cell_total` (int): Minimum reads for donor/acceptor
- `min_total_exon` (int): Minimum reads for exon skipping
- `exon_aggregate` (str): Aggregation method for exon inclusion

**Returns:** Modality object

---

### process_donor_usage()

```python
process_donor_usage(sj_counts, sj_meta, min_cell_total=1)
```

Process splice junctions into donor site usage.

**Returns:** Tuple of (counts_3d, metadata_df)

---

### process_acceptor_usage()

```python
process_acceptor_usage(sj_counts, sj_meta, min_cell_total=1)
```

Process splice junctions into acceptor site usage.

**Returns:** Tuple of (counts_3d, metadata_df)

---

### process_exon_skipping()

```python
process_exon_skipping(
    sj_counts,
    sj_meta,
    min_total=2,
    aggregate='min'
)
```

Detect cassette exon triplets and compute inclusion/total.

**Returns:** Tuple of (inclusion_2d, total_2d, metadata_df)

---

### process_sj_counts()

```python
process_sj_counts(sj_counts, sj_meta, gene_counts)
```

Process raw splice junction counts with gene expression denominator.

**Returns:** Tuple of (sj_counts_2d, gene_denominator_2d, metadata_df)

---

## Utility Functions

From `bayesDREAM.utils` module.

### Hill_based_positive()

```python
Hill_based_positive(x, params)
```

Positive Hill equation (activation).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, EC50]

**Returns:** torch.Tensor

---

### Hill_based_negative()

```python
Hill_based_negative(x, params)
```

Negative Hill equation (repression).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, IC50]

**Returns:** torch.Tensor

---

### Hill_based_piecewise()

```python
Hill_based_piecewise(x, params)
```

Piecewise Hill equation (can be positive or negative).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, xc, indicator]

**Returns:** torch.Tensor

---

### Polynomial_function()

```python
Polynomial_function(x, params, degree=6)
```

Polynomial dose-response function.

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Polynomial coefficients
- `degree` (int): Polynomial degree. Default: 6

**Returns:** torch.Tensor

---

### find_beta()

```python
find_beta(mu, phi, method='mle', tolerance=1e-9)
```

Numerical solver for beta parameter in negative binomial.

**Parameters:**
- `mu` (float): Mean parameter
- `phi` (float): Overdispersion parameter
- `method` (str): Solving method. Default: 'mle'
- `tolerance` (float): Convergence tolerance

**Returns:** float

---

### check_tensor()

```python
check_tensor(x, name='tensor')
```

Check tensor for NaN/Inf values.

**Parameters:**
- `x` (torch.Tensor): Tensor to check
- `name` (str): Name for error message

**Raises:** ValueError if NaN or Inf detected

---

## Distribution Registry

The `DISTRIBUTION_REGISTRY` dictionary maps distribution names to their samplers:

```python
from bayesDREAM import DISTRIBUTION_REGISTRY

print(DISTRIBUTION_REGISTRY.keys())
# dict_keys(['negbinom', 'multinomial', 'binomial', 'normal', 'studentt'])
```

Each entry contains:
- `'technical'`: Sampler function for technical model
- `'trans'`: Sampler function for trans model
- `'requires_denominator'`: bool
- `'requires_sum_factor'`: bool
- `'is_3d'`: bool
- `'supports_cell_line_effects'`: bool
- `'cell_line_effect_type'`: str or None

---

## Plotting Methods

bayesDREAM provides comprehensive plotting functions. For detailed usage, see [PLOTTING_GUIDE.md](PLOTTING_GUIDE.md).

### Model Methods

These methods are available on `bayesDREAM` model instances:

#### plot_technical_fit()

```python
model.plot_technical_fit(
    param='alpha_y',
    modality_name=None,
    technical_group_index=None,
    order_by='mean',
    subset_features=None,
    plot_type='auto',
    metric='posterior_coverage'
)
```

Plot prior vs posterior for technical fit parameters.

**Parameters:**
- `param` (str): Parameter to plot: `'beta_o'`, `'alpha_x'`, `'alpha_y'`, `'mu_ntc'`, `'o_y'`
- `modality_name` (str, optional): Modality name (default: primary modality)
- `technical_group_index` (int, optional): Technical group index (1, 2, ...). Cannot be 0 (baseline).
- `order_by` (str): Feature ordering: `'mean'`, `'difference'`, `'alphabetical'`, `'input'`
- `subset_features` (list, optional): Subset to specific features
- `plot_type` (str): `'auto'`, `'violin'`, or `'scatter'`
- `metric` (str): Comparison metric: `'overlap'`, `'kl_divergence'`, `'posterior_coverage'`

**Returns:** matplotlib Figure

---

#### plot_cis_fit()

```python
model.plot_cis_fit(
    param='x_true',
    order_by='mean',
    metric='posterior_coverage'
)
```

Plot prior vs posterior for cis fit parameters.

**Parameters:**
- `param` (str): Parameter to plot: `'x_true'`
- `order_by` (str): Guide ordering: `'mean'`, `'difference'`, `'alphabetical'`, `'input'`
- `metric` (str): Comparison metric

**Returns:** matplotlib Figure

---

#### plot_trans_fit()

```python
model.plot_trans_fit(
    param='theta',
    modality_name=None,
    subset_features=None,
    order_by='mean',
    plot_type='auto',
    function_type='additive_hill',
    metric='posterior_coverage'
)
```

Plot prior vs posterior for trans fit parameters.

**Parameters:**
- `param` (str): Parameter to plot: `'theta'`, `'gamma'`, `'mu_y'`
- `modality_name` (str, optional): Modality name (default: primary modality)
- `subset_features` (list, optional): Subset to specific features
- `order_by` (str): Feature ordering
- `plot_type` (str): `'auto'`, `'violin'`, or `'scatter'`
- `function_type` (str): Function type used: `'additive_hill'`, `'single_hill'`, `'polynomial'`
- `metric` (str): Comparison metric

**Returns:** matplotlib Figure

---

#### plot_xy_data()

```python
model.plot_xy_data(
    feature,
    modality_name=None,
    window=100,
    show_correction='corrected',
    min_counts=3,
    show_hill_function=True
)
```

Plot raw x-y data showing relationship between cis gene expression and feature values.

**Parameters:**
- `feature` (str): Feature name (gene, junction, etc.)
- `modality_name` (str, optional): Modality name (default: primary modality)
- `window` (int): k-NN window size for smoothing. Default: 100
- `show_correction` (str): `'uncorrected'`, `'corrected'`, or `'both'`. Default: `'corrected'`
- `min_counts` (int): Minimum denominator for binomial. Default: 3
- `show_hill_function` (bool): Overlay Hill function if trans model fitted. Default: True

**Returns:** matplotlib Figure

---

#### plot_parameter_ci_panel()

```python
model.plot_parameter_ci_panel(
    params,
    modality_name=None,
    genes=None,
    ci_level=95.0,
    sort_by='none',
    filter_dependent=False,
    max_genes=100
)
```

Forest plot (dot + whisker CI) for posterior parameters across trans genes.

**Parameters:**
- `params` (list): Parameter names to plot (e.g., `['n_a', 'n_b']`)
- `modality_name` (str, optional): Modality name
- `genes` (list, optional): Specific genes to plot
- `ci_level` (float): Credible interval level. Default: 95.0
- `sort_by` (str): How to sort: `'none'`, `'alphabetical'`, `'median'`, `'abs_median'`, `'effect'`
- `filter_dependent` (bool): Only show genes where CI excludes 0. Default: False
- `max_genes` (int): Maximum genes to plot. Default: 100

**Returns:** Tuple of (matplotlib Figure, matplotlib Axes)

---

#### extract_posterior_dataframe()

```python
model.extract_posterior_dataframe(
    params,
    modality_name=None,
    include_samples=False
)
```

Extract posterior parameters into a long-format DataFrame.

**Parameters:**
- `params` (list): Parameter names to extract (e.g., `['n_a', 'n_b', 'K_a', 'K_b']`)
- `modality_name` (str, optional): Modality name
- `include_samples` (bool): Include all posterior samples (can be large). Default: False

**Returns:** pd.DataFrame with columns: `gene`, `gene_idx`, `param`, `median`, `lo`, `hi`, `mean`, `std`, `ci_excludes_zero`

---

### Standalone Plotting Functions

Available via `from bayesDREAM.plotting import ...`:

#### Prior/Posterior Comparison

- `plot_scalar_parameter(prior, posterior, name, metric, ...)`: Plot scalar parameter
- `plot_1d_parameter(prior, posterior, names, param_name, ...)`: Plot 1D parameter (per-feature)
- `plot_2d_parameter(prior, posterior, feature_names, group_names, ...)`: Plot 2D parameter

#### Basic X-True Plots

- `scatter_by_guide(model, gene, log2, ...)`: Scatter plot by guide
- `scatter_ci95_by_guide(model, gene, ...)`: Scatter with 95% CI
- `violin_by_guide_log2(model, gene, ...)`: Violin plot in log2 space
- `filled_density_by_guide_log2(model, gene, ...)`: Filled density plot

#### Posterior Density Plots

- `plot_posterior_density_lines(model, param, ...)`: Posterior density lines
- `plot_xtrue_density_by_guide(model, ...)`: X-true density by guide
- `plot_parameter_density_with_xtrue(model, ...)`: Parameter density with x-true

#### DE Comparison Plots

- `compute_log2fc_metrics(...)`: Compute log2FC metrics
- `scatter_and_heatmap_edger_vs_bayes(...)`: Compare with edgeR
- `plot_edger_vs_bayes_full_range(...)`: Full range comparison
- `plot_edger_vs_bayes_observed_range(...)`: Observed range comparison

#### Diagnostics

- `plot_sum_factor_comparison(...)`: Compare sum factor methods

#### Color Utilities

- `ColorScheme`: Color scheme management class
- `build_guide_colors(model)`: Build guide-specific colors
- `lighten(color, amount)`: Lighten a color
- `darken(color, amount)`: Darken a color
