# bayesDREAM Tests

This directory contains test scripts for bayesDREAM functionality.

## Current Test Suite (16 Essential Tests)

### Core Integration Tests

- **`test_multimodal_fitting.py`** - Core multi-modal fitting infrastructure test
- **`test_per_modality_fitting.py`** - Per-modality technical fitting with different distributions
- **`test_trans_all_distributions.py`** - Comprehensive trans fitting test for all distributions

### Compatibility Tests

- **`test_negbinom_compat.py`** - Backward compatibility test for negative binomial distribution
- **`test_technical_compat.py`** - Backward compatibility test for `fit_technical()` with negbinom

### Feature-Specific Tests

- **`test_modality_atac.py`** - Tests ATAC-seq modality integration
- **`test_cell_names_numpy.py`** - Tests cell_names parameter for numpy arrays
- **`test_exon_skip_aggregation.py`** - Tests exon skipping aggregation methods (min vs mean)
- **`test_filtering_simple.py`** - Distribution-specific filtering at modality creation
- **`test_gene_meta.py`** - Gene metadata handling and auto-creation
- **`test_high_moi.py`** - High MOI (multiplicity of infection) workflows
- **`test_matrix_types.py`** - Matrix type handling (sparse/dense)
- **`test_modality_save_load.py`** - Modality save/load functionality

### Export and Summary Tests

- **`test_summary_export.py`** - Full pipeline summary export (runs complete pipeline)
- **`test_summary_export_simple.py`** - Summary export with mock posterior data (fast)

### Quick Validation

- **`test_imports.py`** - Quick smoke test for package imports

## Running Tests

Tests require the `pyroenv` conda environment:

```bash
# Set PYTHONPATH to repository root
cd "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude"
export PYTHONPATH="."

# Run a specific test
/opt/anaconda3/envs/pyroenv/bin/python tests/test_multimodal_fitting.py

# Run multiple tests
/opt/anaconda3/envs/pyroenv/bin/python tests/test_filtering_simple.py
/opt/anaconda3/envs/pyroenv/bin/python tests/test_per_modality_fitting.py
/opt/anaconda3/envs/pyroenv/bin/python tests/test_trans_all_distributions.py
```

## Archived Tests

The `archive/` subdirectory contains older test scripts organized by category:

- **`archive/debug_scripts/`** - One-off debugging scripts (debug_alpha_*.py, quick_test_*.py, verify_extraction.py)
- **`archive/bug_fix_tests/`** - Tests for specific bug fixes (sparse sum flatten, cis extraction)
- **`archive/plotting_tests/`** - Old plotting validation tests
- **`archive/redundant_tests/`** - Tests superseded by more comprehensive versions

These archived tests are kept for historical reference but are no longer actively maintained.

## Test Organization Principles

**Essential tests** (kept in main tests/ directory):
- Core integration tests for main functionality
- Backward compatibility tests
- Feature-specific tests for actively maintained features
- Export and summary functionality tests

**Archived tests** (moved to tests/archive/):
- One-off debugging scripts
- Tests for specific historical bugs
- Redundant tests superseded by more comprehensive versions
- Development-phase validation scripts
