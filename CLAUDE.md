# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

bayesDREAM is a Bayesian framework for modeling perturbation effects across multiple molecular modalities. The model consists of three sequential steps:

1. **Technical fit** (`fit_ntc`): Models technical variation in non-targeting controls (NTC) to estimate gene-specific overdispersion parameters (`alpha_y`)
2. **Cis fit** (`fit_cis`): Models direct effects on the targeted gene expression (`model_x`)
3. **Trans fit** (`fit_trans`): Models downstream effects on other genes as a function of the cis gene expression (`model_y`)

The codebase uses PyTorch and Pyro for probabilistic programming and variational inference.

## Repository Structure

```
bayesDREAM_forClaude/
├── bayesDREAM/
│   ├── __init__.py          # Package exports
│   ├── model.py             # Main bayesDREAM class
│   ├── core/                # _BayesDREAMCore base class (package)
│   │   ├── __init__.py      # _BayesDREAMCore
│   │   ├── data_setup.py    # Input/metadata validation, cell subsetting at init
│   │   └── technical_state.py # set_alpha_x / set_alpha_y (TechnicalStateMixin)
│   ├── modality.py          # Modality class for multi-modal data
│   ├── guides.py            # Guide-level helpers
│   ├── diagnostics.py       # Post-fit diagnostics (DiagnosticsMixin)
│   ├── slurm_jobgen.py      # SLURM job generation
│   ├── utils.py             # Shared utility functions
│   ├── archive/             # Retired code (not imported)
│   ├── fitting/             # Fitting methods (modular)
│   │   ├── __init__.py
│   │   ├── ntc.py           # NTCFitter class (overdispersion estimation)
│   │   ├── ntc_tuning.py    # fit_ntc memory estimation / minibatching
│   │   ├── cis.py           # CisFitter class
│   │   ├── trans.py         # TransFitter class
│   │   └── distributions.py # Distribution-specific observation samplers
│   ├── io/                  # Save/load/summary functionality
│   │   ├── __init__.py
│   │   ├── save.py          # ModelSaver class
│   │   ├── load.py          # ModelLoader class
│   │   └── summary.py       # Summary export
│   ├── modalities/          # Modality-specific methods
│   │   ├── __init__.py
│   │   ├── transcript.py    # TranscriptModalityMixin
│   │   ├── splicing_modality.py  # SplicingModalityMixin (empty stub; see docs/SPLICING_LOADER_GUIDE.md)
│   │   ├── atac.py          # ATACModalityMixin
│   │   ├── custom.py        # CustomModalityMixin
│   │   └── plotting_mixin.py # PlottingMixin
│   ├── plotting/            # Visualization subpackage
│   │   ├── __init__.py
│   │   ├── basic.py         # Basic plots
│   │   ├── model_plots.py   # Model-specific plots
│   │   ├── posterior.py     # Posterior visualization
│   │   ├── prior_posterior.py
│   │   ├── prior_sampling.py
│   │   ├── xy_plots.py      # x/y dose-response plots
│   │   ├── de_comparison.py # DE comparison plots
│   │   ├── diagnostics.py   # Diagnostic plots
│   │   ├── colors.py        # Color palettes
│   │   ├── helpers.py       # Plotting helpers
│   │   └── utils.py         # Plotting utilities
│   └── simulation/          # Simulation and permutation
│       ├── __init__.py
│       ├── simulation.py    # Data simulation
│       └── permutation.py   # Permutation testing utilities
├── tests/                   # Test suite (24 test files)
├── toydata/                 # Test datasets (genes, splicing, metadata)
├── examples/                # Example scripts
├── docs/                    # Documentation
├── testing/                 # Ad-hoc testing notebooks/scripts
└── apptainer/               # Container build definitions
```

Analysis/publication scripts (`publication_runs/`, `comparative/`, `visualisation/`) live in the sibling repo `bayesdream_analyses`, not here.

**Note**: The codebase was recently refactored from a single 4,537-line `model.py` file into a modular structure. This improves maintainability while preserving backward compatibility. See `docs/archive/planning/REFACTORING_SUMMARY.md` for details.

## Core Architecture

### bayesDREAM Class

The main class in `bayesDREAM/model.py` implements multi-modal Bayesian modeling with the three-step pipeline:

**Initialization:**
- Takes cell metadata DataFrame (`meta`) with columns: `cell`, `guide`, `cell_line`, `target`, `sum_factor`, etc.
- Takes counts DataFrame (`counts`) with genes as rows, cell barcodes as columns
- Optionally takes feature metadata DataFrame (`feature_meta`) with gene annotations
  - Recommended columns: `gene`, `gene_name`, `gene_id`
  - If not provided, minimal metadata is created from counts.index
  - Feature identity: `feature_name_col` pins which column is the feature id (e.g. `gene_id` when `gene_name` has duplicates); `feature_names` gives ids directly
  - The primary modality's name is set with `modality_name` (default `'gene'`)
- Creates guide-level metadata by grouping cells by guide and specified covariates
- Supports both CPU and CUDA devices
- `cis_gene` is **optional at init** (low-MOI only); omit it to run `fit_ntc()` once across all cis genes, then call `add_cis_gene()` before `fit_cis()`. When `cis_gene` is omitted, `label` must be provided explicitly (it normally defaults to the gene name).

**Key Methods:**

- `set_technical_groups(covariates)`: Sets technical_group_code based on covariates (must be called before fit_ntc)
- `fit_ntc(sum_factor_col, modality_name, ...)`: Fits NTC-only model to estimate `alpha_y_prefit`
- `add_cis_gene(cis_gene, allow_ntc_only=False)`: Specifies the cis gene after initialization (see **Deferred Cis-Gene Workflow** below). Can be called before or after `fit_ntc()`.
- `set_alpha_x(alpha_x, covariates)`: Sets cis gene overdispersion parameters
- `set_alpha_y(alpha_y, covariates)`: Sets trans gene overdispersion parameters
- `adjust_ntc_sum_factor(covariates, ...)`: Adjusts NTC sum factors for covariates
- `fit_cis(sum_factor_col, ...)`: Fits cis effects using `_model_x`. `independent_mu_sigma=True` fits `mu`/`sigma` per group = (NTC | cis) x covariates: cis guides by `mu_sigma_covariates` (default `guide_covariates`), NTC by `mu_sigma_covariates_ntc` (default `guide_covariates_ntc`). `guide_covariates` splits only non-NTC guides and `guide_covariates_ntc` only NTC guides (never combined)
- `set_x_true(x_true)`: Sets true cis expression for trans modeling
- **Where fit results live (issue #14):** per-modality only. `model.posterior_samples_trans` is a property reading the primary modality's `posterior_samples_trans`; `model.alpha_x_prefit` is a property reading the `'cis'` modality's `alpha_y_prefit_mult` (`[C, 1]`, returned as `[C]`). Assigning either still works but raises a `DeprecationWarning` and writes through to the modality; package code uses `_set_posterior_samples_trans()` / `_set_alpha_x_prefit()`.
- `permute_genes(genes2permute, ...)`: Permutes guide-gene associations for null testing
- `refit_sumfactor(covariates, ...)`: Re-estimates sum factors based on posterior cis expression
- `fit_trans(sum_factor_col, function_type, modality_name, ...)`: Fits trans effects using `_model_y`. Default `x_level='cell', x_uncertainty='sample'`: each SVI step conditions on a fresh fit_cis posterior draw of x_true (fixed data, never a latent of `_model_y`); needs full `posterior_samples_cis` and raises otherwise. `x_uncertainty='median'` and `x_level='guide'` exist but are not recommended (see docs/FIT_TRANS_GUIDE.md)

**Function Types for Trans Modeling:**

The `fit_trans` method supports multiple functional forms for modeling how trans gene expression depends on cis gene expression:

- `single_hill`: Single Hill equation (positive or negative)
- `additive_hill`: Additive combination of positive and negative Hill functions
- `nested_hill`: Hill function of a Hill function

(`polynomial` was removed in PR #34.)

### Probabilistic Models

Three Pyro models implement the statistical framework:

1. **`_model_ntc`** (`fitting/ntc.py`): Models NTC cells to estimate technical-group effects and baselines
   - Distribution-specific likelihood (negbinom, normal, studentt, binomial, multinomial)
   - Estimates per-feature `alpha_y` (technical-group effects), `mu_ntc`, and dispersion
   - Its data-driven prior parameters are recorded in `model.ntc_prior_params[modality]` (used for prior/posterior plots)

2. **`_model_x`**: Models cis effects on the targeted gene
   - Accounts for guide-level and cell-line-level variation
   - Estimates true gene expression `x_true` for each guide
   - Uses sum factors for normalization

3. **`_model_y`**: Models trans effects as functions of cis expression
   - Hill-based functions (`single_hill`, `additive_hill`, `nested_hill`)
   - Conditions on fixed technical-group effects from `fit_ntc` (`alpha_y_prefit`); it does not estimate them
   - Includes sparsity priors (gamma distribution on effect sizes)
   - Models gene-specific dose-response curves

## Common Development Tasks

### Testing

Run infrastructure tests (requires `bayesdream` conda environment at `/Users/leahrosen/miniconda3/envs/bayesdream/`):

```bash
cd "/Users/leahrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/Code/bayesDREAM code/bayesDREAM_forClaude"
export PYTHONPATH="."
/Users/leahrosen/miniconda3/envs/bayesdream/bin/python -m pytest tests/ -v
```

### Modifying the Model

The codebase uses a modular structure with delegation:

1. **Core fitting logic**: Pyro models and fitting methods are in `fitting/` directory
   - `fitting/ntc.py`: NTCFitter class with `_model_ntc` and `fit_ntc`
   - `fitting/cis.py`: CisFitter class with `_model_x` and `fit_cis`
   - `fitting/trans.py`: TransFitter class with `_model_y` and `fit_trans`
   - `fitting/distributions.py`: Distribution-specific observation samplers (shared across fitters)

2. **Base class**: `core/__init__.py` contains `_BayesDREAMCore` which:
   - Initializes fitter objects (`_ntc_fitter`, `_cis_fitter`, `_trans_fitter`)
   - Delegates method calls to appropriate fitters
   - Contains utility methods (parameter setters, permutation, etc.)

3. **Main class**: `model.py` contains `bayesDREAM` which:
   - Inherits from `_BayesDREAMCore` and modality mixins
   - Handles multi-modal initialization
   - Provides modality management methods

4. **Modality mixins**: `modalities/` directory contains methods for adding different data types:
   - `transcript.py`: `add_transcript_modality`
   - `splicing_modality.py`: empty stub (`add_splicing_modality` was removed; load splicing via `add_custom_modality`, see `docs/SPLICING_LOADER_GUIDE.md`)
   - `atac.py`: `add_atac_modality`
   - `custom.py`: `add_custom_modality`

5. **I/O operations**: `io/` directory contains save/load/summary functionality
   - `save.py`: ModelSaver class
   - `load.py`: ModelLoader class
   - `summary.py`: Summary export for downstream analysis

**When adding new functionality**:
- Distribution sampling logic → `fitting/distributions.py`
- New Pyro models → appropriate fitter in `fitting/` (follow `_model_<name>` convention)
- Modality-specific code → appropriate mixin in `modalities/`
- Save/load methods → `io/save.py` or `io/load.py`
- All fitters access model attributes via `self.model.*` (not `self.*`)

### Adding New Function Types

To add a new dose-response function:

1. Define the function in the helper section (e.g., `def my_function(x, params)`)
2. Add a conditional branch in `_model_y` to handle the new function type
3. Update `fit_trans` to set appropriate priors and optimization settings

### Deferred Cis-Gene Workflow (`add_cis_gene`)

In the standard pipeline, `cis_gene` is provided at initialization and the model is subset to NTC + that gene's cells immediately. This means `fit_ntc()` must be re-run separately per cis gene.

**`add_cis_gene(cis_gene)`** allows a single `fit_ntc()` call to serve all cis genes:

```python
# One model, one fit_ntc call — shared across all cis genes
model = bayesDREAM(meta=meta, counts=gene_counts, label='run1')
model.set_technical_groups(['cell_line'])
model.fit_ntc(sum_factor_col='sum_factor')
model.adjust_ntc_sum_factor(covariates=['cell_line'])  # optional, also shareable

# Then for each cis gene, fork from the fitted model:
import copy
for gene in ['GFI1B', 'MYB', 'TET2']:
    m = copy.deepcopy(model)
    m.add_cis_gene(gene)
    m.fit_cis(sum_factor_col='sum_factor')
    m.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
```

**What `add_cis_gene()` does internally (`model.py`):**

1. Finds the cis gene by name in the primary modality's `feature_meta`
2. Extracts its counts into a new `'cis'` modality
3. If `fit_ntc()` has already run: calls `_extract_cis_alpha_from_ntc_posteriors()` — pulls the gene's alpha from the primary modality posteriors into the cis modality and trims it from the primary posteriors; sets `self.alpha_x_prefit`
4. Removes the cis gene from the primary modality (counts + feature_meta, index reset)
5. Subsets `self.meta` and all modalities to NTC + cis cells
6. Rebuilding each modality on the retained cells (`get_cell_subset()`) drops features that became zero-count or zero-variance there; `_transfer_ntc_results()` trims the stored NTC results (`posterior_samples_ntc`, `alpha_y_prefit_*`) to the surviving features **by feature id** (issue #24). Raises if the cis gene itself is zero-count/constant across the retained cells
7. Recomputes `guide_code` as compact integers over the retained cells
8. Reinitialises `sum_factors` on the final cell set

**Constraints:**
- `label` must be provided explicitly when `cis_gene` is omitted at init
- `fit_cis()` and `fit_trans()` require `add_cis_gene()` to have been called first
- Cis cells come from `meta['target']`, or from the guide mapping (`guide_targets_dict`) in high-MOI / `guide_target` mode. If `cis_gene` doesn't appear there, `add_cis_gene()` tries the gene's other identifiers (feature id, `gene_id`/`gene_name`/... in its feature_meta row) and relabels a match to `cis_gene`; if nothing matches it raises instead of silently fitting NTC-only (pass `allow_ntc_only=True` to allow that) — issue #22. These checks run before the model is modified, so a failed call can be retried
- `adjust_ntc_sum_factor()` and `fit_ntc()` can safely be called before `add_cis_gene()`
- `load_ntc_fit()` on a model whose cis gene is already set (at init or via `add_cis_gene()`) works with a shared, all-genes NTC fit: when there is no `posterior_samples_ntc_cis.pt`, it locates the cis gene in the primary modality's saved posterior by any of its identifiers (feature id, `gene_id`/`gene_name`/... from either side's feature metadata) and fills the `'cis'` modality and `alpha_x_prefit` from it — issue #23
- `guide_code` is computed over all cells at init (harmless), then recomputed compactly inside `add_cis_gene()` (single-guide mode only — stays `-1` in high-MOI mode)

**Supported in high-MOI mode too** (`guide_assignment`/`guide_meta` provided at init, `cis_gene` omitted):
- At init: cells are classified as `'excluded'` / `'ntc'` / `'other'` (cis identity unknown yet); excluded cells are dropped immediately; the full guide panel is kept (no pruning) since which guides are "cis" isn't known until `add_cis_gene()`.
- `fit_ntc(use_all_cells=...)` defaults to `True` when `cis_gene` is still unset in high-MOI mode (cell classification into NTC/cis/other isn't possible yet, so a per-gene NTC-only fit isn't well-defined). Pass `use_all_cells=False` explicitly to force NTC-only fitting instead (still correct, since 'ntc' cells are already properly classified at this point).
- `add_cis_gene(gene)` additionally: reclassifies any cell carrying a guide targeting `gene` from `'ntc'`/`'other'` to `gene` (excluded cells are never reclassified — same `excluded > cis > ntc > other` priority used in eager high-MOI init), then prunes `guide_assignment`/`guide_meta`/`guide_targets_dict` down to NTC + cis-gene guides only and rebuilds `guide_assignment_tensor`.
- Fixed: `fit_cis()` in high-MOI mode used to raise a Float/Double dtype mismatch in `_model_x`, because the high-MOI guide means were computed in float64 from the numpy `guide_assignment`; they are now cast to float32 in `fitting/cis.py`.

### Single-Guide Mode with `guide_target` (Ambiguous/Multi-Target Guides)

Single-guide mode normally requires a `target` column in `meta`, pre-computed one-to-one per cell. When a guide's true target is ambiguous (e.g. predicted off-target effects on more than one gene), pass a `guide_target` DataFrame instead — same schema as the high-MOI many-to-many mapping (`{'guide', 'target'}` rows, multiple rows allowed per guide):

```python
guide_target = pd.DataFrame({
    'guide':  ['guide_A', 'guide_C', 'guide_C', 'ntc_1'],
    'target': ['GFI1B',   'GFI1B',   'MYB',     'ntc'],  # guide_C is ambiguous: GFI1B or MYB
})
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B', guide_target=guide_target)
```

- `meta` does NOT need a `'target'` column when `guide_target` is supplied — each cell's target is derived from its `'guide'` column: resolves to `cis_gene` if `cis_gene` is among the guide's plausible targets, else `'ntc'` if any plausible target is an NTC variant, else `'other'` (dropped).
- Because resolution depends on `cis_gene`, the same ambiguous guide resolves differently across separate fits/models for different cis genes — no need to hand-edit `meta['target']` per gene.
- Works with the deferred-cis-gene workflow too: omit `cis_gene` at init (all cells kept, target is `'ntc'`/`'other'` only), then `add_cis_gene(gene)` reclassifies any cell whose guide targets `gene` — mirroring the high-MOI `add_cis_gene()` reclassification step, but without the guide-panel pruning (there's no `guide_assignment` matrix to prune in single-guide mode).
- `exclude_targets` also works in this mode (previously high-MOI only): a guide targeting any excluded gene drops all its cells, even if it also targets `cis_gene`.
- Implementation: `core/__init__.py` builds `self.guide_targets_dict` from `guide_target` in the single-guide branch (mirrors the high-MOI dict, but is never `None`-checked the same way — high-MOI always populates it or raises, single-guide leaves it `None` when no `guide_target` is given); `add_cis_gene()` in `model.py` has an `elif self.guide_targets_dict is not None:` branch parallel to its `if self.is_high_moi:` branch for the reclassification step.

### Testing Changes

The `toydata/` directory contains small test datasets. Use these for quick validation before running on full data:

- `gene_counts.csv`, `gene_meta.csv`: Gene expression data
- `Josie_cell_meta.csv`, `Josie_gene_meta.csv`: Alternative cell/gene metadata
- `SJ_meta.csv` / `SpliZ_meta.csv`: Splice junction / SpliZ metadata (the matching `*_counts.csv` files are not tracked in git; they may exist only locally)

## Multi-Modal Architecture

bayesDREAM supports multiple molecular modalities beyond gene expression, allowing modeling of transcripts, splicing, and custom measurements within a unified framework.

### Modality Class

The `Modality` class (`bayesDREAM/modality.py`) provides a standardized container for different data types:

**Supported Distributions:**
- `negbinom`: Negative binomial (gene counts, transcript counts)
- `multinomial`: Categorical/proportional data (isoform usage, donor/acceptor usage)
- `binomial`: Binary outcomes with denominators (exon skipping PSI)
- `normal`: Continuous measurements (SpliZ scores)
- `studentt`: Heavy-tailed continuous (robust SpliZ)

**Data Structures:**
- **2D data** (negbinom, normal, studentt, binomial): `(features, cells)` or `(cells, features)`
- **3D multinomial**: `(features, cells, categories)` - e.g., donor sites × cells × acceptor options
- **Binomial**: 2D counts + 2D denominator array

**Key Features:**
- Feature-level metadata (gene names, junction coordinates, etc.)
- Cell subsetting and feature subsetting
- Automatic validation of shapes and distribution requirements
- Conversion to PyTorch tensors

### bayesDREAM Class (Multi-Modal)

The `bayesDREAM` class (`bayesDREAM/model.py`) provides full multi-modal support:

**Initialization (standard):**
```python
from bayesDREAM import bayesDREAM

model = bayesDREAM(
    meta=cell_metadata,
    counts=gene_counts,              # Primary modality (genes)
    feature_meta=gene_metadata,      # Optional: gene annotations
    cis_gene='GFI1B',               # Optional — can be set later via add_cis_gene()
    modality_name='gene',            # Name of the primary modality (drives cis/trans)
    output_dir='./output',
    label='multimodal_run'
)
```

**Initialization (deferred cis gene — fit_ntc once for all genes):**
```python
model = bayesDREAM(
    meta=cell_metadata,
    counts=gene_counts,
    label='run1',                   # Required when cis_gene is omitted
)
model.set_technical_groups(['cell_line'])
model.fit_ntc(sum_factor_col='sum_factor')
# model now has fit_ntc posteriors for ALL genes;
# call add_cis_gene() to commit to one cis gene before fit_cis()
model.add_cis_gene('GFI1B')
```

**Adding Modalities:**

1. **Transcript counts** (as counts and/or isoform usage):
   ```python
   # Add both counts and usage in one call
   model.add_transcript_modality(
       transcript_counts=tx_counts,
       transcript_meta=tx_meta,      # Must have: transcript_id + (gene/gene_name/gene_id)
       modality_types=['counts', 'usage']  # 'counts', 'usage', or both
   )

   # Or add just one type
   model.add_transcript_modality(
       transcript_counts=tx_counts,
       transcript_meta=tx_meta,
       modality_types='counts'       # Just negbinom counts
   )
   ```

2. **Splicing data** (SJ / donor / acceptor / exon skipping): computed outside bayesDREAM and
   loaded with `add_custom_modality()` (binomial with a denominator, or 3D multinomial).
   The standard input layout and loader helpers are in `docs/SPLICING_LOADER_GUIDE.md`.
   (`add_splicing_modality()` and `splicing.py` were removed; `splicing.py` is in `bayesDREAM/archive/`.)

3. **Custom modalities** (SpliZ, etc.):
   ```python
   # SpliZ scores (normal distribution)
   model.add_custom_modality(
       name='spliz',
       counts=spliz_scores,          # 2D: genes × cells
       feature_meta=gene_meta,
       distribution='normal'
   )
   ```

**Working with Modalities:**
```python
# List all modalities
print(model.list_modalities())

# Access specific modality
donor_mod = model.get_modality('splicing_donor')
print(donor_mod.dims)                    # {'n_features': 100, 'n_cells': 500, 'n_categories': 10}
print(donor_mod.feature_meta.head())     # Metadata: chrom, strand, donor, acceptors, etc.

# Subset modality
subset = donor_mod.get_feature_subset(['feature1', 'feature2'])
```

### Splicing Data

Splicing quantities are prepared outside bayesDREAM and added with `add_custom_modality()`;
see `docs/SPLICING_LOADER_GUIDE.md` for the directory layout, distribution per type, and loader
helpers. Typical encodings:

- **Raw SJ / exon skipping**: `binomial`, `(features, cells)` numerator + denominator
- **Donor / acceptor usage**: `multinomial`, `(features, cells, categories)`

Note: multinomial fit results are laid out `[..., features, categories]` (or `[..., features]`
for per-feature scalars); always locate the feature axis with `utils.feature_axis()`.

### Current Limitations

1. **Cis/Trans fitting**: Currently only the primary modality (usually genes) is used for cis and trans modeling. Future versions will support modality-specific fits.

2. **Technical fitting**: `fit_ntc(modality_name=...)` runs per modality. Only the primary modality's fit also yields the cis gene's `alpha_x`.

3. **Permutation testing**: `permute_genes()` operates on the primary modality.

4. **Sum factors**: Calculated only for gene-level data. Other modalities may need alternative normalization strategies.

### Example Workflows

**Comprehensive example**:
```python
from bayesDREAM import bayesDREAM

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create multi-modal model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    feature_meta=gene_meta,  # Optional: provide gene annotations
    cis_gene='GFI1B',
    output_dir='./output',
    label='multimodal_run'
)

# Add a splicing modality (prepared outside bayesDREAM; see docs/SPLICING_LOADER_GUIDE.md)
model.add_custom_modality(
    name='splicing_donor',
    counts=donor_counts_3d,          # (donors, cells, acceptor options)
    feature_meta=donor_meta,
    distribution='multinomial',
)

# Inspect modalities
print(model.list_modalities())

# Set technical groups first (required before fit_ntc)
model.set_technical_groups(['cell_line'])

# Run standard pipeline (operates on primary 'gene' modality)
model.fit_ntc(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access splicing data for downstream analysis
donor_modality = model.get_modality('splicing_donor')
donor_counts = donor_modality.counts        # 3D array
donor_meta = donor_modality.feature_meta    # Donor site annotations
```

See `tests/` directory for complete examples including transcripts, custom modalities, and advanced usage.

---

## Development Workflow

### Running Tests

All tests are in the `tests/` directory. The conda environment is `bayesdream` at `/Users/leahrosen/miniconda3/envs/bayesdream/bin/python`.

Run the full suite with pytest:

```bash
cd "/Users/leahrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/Code/bayesDREAM code/bayesDREAM_forClaude"
export PYTHONPATH="."
/Users/leahrosen/miniconda3/envs/bayesdream/bin/python -m pytest tests/ -v
```

Or run individual test files:

```bash
PYTHON=/Users/leahrosen/miniconda3/envs/bayesdream/bin/python
$PYTHON tests/test_multimodal_fitting.py       # Full pipeline (slow, requires toydata)
$PYTHON tests/test_negbinom_compat.py          # Negbinom backward compat
$PYTHON tests/test_technical_compat.py         # Technical fitting compat
$PYTHON tests/test_per_modality_fitting.py     # Multi-modality fitting
$PYTHON tests/test_gene_meta.py                # Gene metadata handling
$PYTHON tests/test_modality_save_load.py       # Save/load
$PYTHON tests/test_summary_export.py           # Summary export (slow)
$PYTHON tests/test_summary_export_simple.py    # Summary export (mocked, faster)
$PYTHON tests/test_system_quick_pipeline.py    # Quick end-to-end pipeline
$PYTHON tests/test_trans_all_distributions.py  # Trans fitting all distributions
$PYTHON tests/test_high_moi.py                 # High-MOI mode
$PYTHON tests/test_imports.py                  # Import smoke tests
$PYTHON tests/test_filtering_simple.py         # Feature filtering
$PYTHON tests/test_matrix_types.py             # Sparse/dense matrix support
$PYTHON tests/test_modality_atac.py            # ATAC modality
$PYTHON tests/test_cell_names_numpy.py         # Cell name handling
$PYTHON tests/test_exon_skip_aggregation.py    # Exon skipping aggregation
$PYTHON tests/test_exclude_trans_genes.py      # exclude_trans_genes()
$PYTHON tests/test_guide_target_low_moi.py     # guide_target in single-guide mode
$PYTHON tests/test_cis_group_mu_prior.py       # Per-group mu prior (independent_mu_sigma)
$PYTHON tests/test_independent_mu_sigma_groups.py  # guide_covariates / guide_covariates_ntc
$PYTHON tests/test_lean_load.py                # lean=True loading
$PYTHON tests/test_trans_x_uncertainty.py      # fit_trans x_level / x_uncertainty
```

**Do not write new tests or edit existing ones unless explicitly asked.** If tests need changing,
raise it with the user as a short draft issue instead.

### Documentation

All documentation is in the `docs/` directory:

- `API_REFERENCE.md`: Public API documentation
- `ARCHITECTURE.md`: System architecture and design decisions
- `INITIALIZATION.md`: Technical fitting initialization strategies (empirical Bayes for negbinom, binomial, multinomial)
- `OUTSTANDING_TASKS.md`: **Current outstanding tasks and known issues**
- `SAVE_LOAD_GUIDE.md`: Guide to save/load functionality
- `SUMMARY_EXPORT_GUIDE.md`: Exporting results for R/plotting
- `QUICKSTART_MULTIMODAL.md`: Quick start for multi-modal analysis
- `SLURM_JOB_GENERATOR.md`: HPC job generation guide
- `MEMORY_REQUIREMENTS.md`: Memory estimation guide
- `FIT_TRANS_GUIDE.md`: Trans fitting guide
- `FIT_TRANS_DESIGN_DECISIONS.md`: Design decisions for trans fitting
- `HIGH_MOI_GUIDE.md`: High-MOI mode guide
- `SPLICING_LOADER_GUIDE.md`: Splicing data loading guide
- `CIS_MODEL_PARAMETERS.md`: Cis model parameter documentation
- `HILL_FUNCTION_PRIORS.md`: Hill function prior specification
- `BAYESIAN_FDR.md`: Bayesian FDR approach
- `CELL_NAMES_GUIDE.md`: Cell name handling
- `DATA_ACCESS.md`: Data access patterns
- `FALSE_POSITIVES_GFI1B_DIAGNOSIS.md`: False positive diagnosis for GFI1B run
- `OX_PRIOR_DIAGNOSIS.md`: o_x prior diagnosis and fix
- Historical planning documents available in `docs/archive/`

**Important**: Check `docs/OUTSTANDING_TASKS.md` for current development priorities, including the guide-prior infrastructure that needs to be integrated into `fit_cis()`.

### Code Organization

- **Python source**: `bayesDREAM/` directory
- **Tests**: `tests/` directory
- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Test data**: `toydata/` directory

### Making Changes

1. Read `docs/OUTSTANDING_TASKS.md` to understand current priorities
2. Make changes in appropriate module (see "Modifying the Model" section above)
3. Run relevant tests to ensure nothing breaks
4. Update documentation if public API changes
5. Add tests for new functionality
