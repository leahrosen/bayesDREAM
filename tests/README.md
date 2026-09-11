# bayesDREAM Tests

This directory contains test scripts for bayesDREAM functionality.

## Current Test Suite (16 Essential Tests)

### Core Integration Tests

- **`test_multimodal_fitting.py`** - Core multi-modal fitting infrastructure test
- **`test_per_modality_fitting.py`** - Per-modality technical fitting with different distributions
- **`test_trans_all_distributions.py`** - Comprehensive trans fitting test for all distributions

### Compatibility Tests

- **`test_negbinom_compat.py`** - Backward compatibility test for negative binomial distribution
- **`test_technical_compat.py`** - Backward compatibility test for `fit_ntc()` with negbinom

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

### CLI and full pipeline tests

- **`test_cli_system.py`** - test the CLI
- **`test_system_quick_pipeline.py`** - run a full pipeline

## Running Tests

```bash
pytest .
```
