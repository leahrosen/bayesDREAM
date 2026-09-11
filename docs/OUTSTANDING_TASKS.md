# Outstanding Tasks for bayesDREAM

This document tracks remaining implementation tasks and known issues.

Last updated: 2025-01-27

---

## High Priority

### 1. Multinomial and Student-T Trans Fitting

**Status**: Implemented, needs thorough testing

**Description**: Trans fitting for multinomial (splicing donor/acceptor usage) and Student-T (continuous data with heavy tails) distributions.

**What Works**:
- Distribution-specific observation samplers in `distributions.py`
- Multinomial technical correction enabled
- Basic Hill function plotting for multinomial

**What's Needed**:
- Comprehensive testing on real biological data
- Validation that posteriors are reasonable for splicing data
- Performance benchmarking with large multinomial features
- Document expected data formats and preprocessing

---

### 2. High-MOI Guide Additivity

**Status**: Not implemented

**Description**: Support for experiments where cells receive multiple guides (high multiplicity of infection). Currently the model assumes one guide per cell.

**What's Needed**:
1. Design decision: How to model additive/multiplicative effects of multiple guides
2. Modify guide assignment logic to handle cells with multiple guides
3. Update `_model_x` to model combined guide effects
4. Update `_model_y` to propagate combined cis effects to trans
5. Add tests with synthetic high-MOI data

**Design Considerations**:
- Additive effects in log space? `log2(x_true) = sum(log2(effect_g) for g in cell_guides)`
- How to handle NTC in high-MOI context?
- Cell-level vs population-level modeling of guide combinations

---

## Medium Priority

### 3. Multi-Modal Workflow Documentation

**Status**: Implementation complete, documentation needed

**Description**: The multi-modal fitting infrastructure is fully implemented but lacks comprehensive examples and documentation.

**What Works**:
- `fit_cis()` always fits the primary modality (by design - the primary modality defines the cis gene)
- `fit_trans()` can fit ANY modality via `modality_name` parameter (tested on splicing_sj, etc.)
- Any negbinom modality can be primary (gene, ATAC, custom)
- All distribution types supported in trans fitting (negbinom, binomial, multinomial, normal, studentt)

**What's Needed**:
1. Add examples showing `fit_trans()` for splicing/custom modalities
2. Document best practices for multi-modal cis/trans workflows
3. Example showing full workflow: fit_cis on genes → fit_trans on multiple modalities

**Note**: Cis fitting is always negative binomial on the primary modality. This is by design - the primary modality defines what "cis" means (deconvolved count-based expression of the perturbed gene/feature).

---

### 4. Cell Cycle Modality

**Status**: Not implemented

**Description**: Model cell cycle state as a modality. Two potential approaches:

1. **Multinomial**: Probability of being in each cell cycle phase (G1, S, G2/M)
   - Input: Classification probabilities from cell cycle scoring tools
   - Distribution: `multinomial` with categories = phases
   - Similar to splicing donor/acceptor usage

2. **Multivariate continuous**: Phase-specific scores
   - Input: Continuous scores for each phase (e.g., from Seurat/Scanpy)
   - Distribution: `gamma` (non-negative) or `studentt` (robust to outliers)
   - Could use independent univariate fits or correlated multivariate

**What's Needed**:
1. Decide on data format and distribution choice
2. Add `add_cell_cycle_modality()` method (or use `add_custom_modality()`)
3. If using Gamma: add to distribution registry (`distributions.py`)
4. Test with real cell cycle data
5. Document preprocessing requirements (normalization, etc.)

**Design Considerations**:
- Cell cycle is per-cell, not per-feature - may need special handling
- Scores may be highly correlated (cells transitioning between phases)
- Consider whether to model phase as categorical vs continuous "progress"

---

### 5. Implement Guide-Prior Infrastructure in fit_cis

**Status**: Infrastructure prepared, not yet integrated into _model_x

**Location**: `bayesDREAM/fitting/cis.py` lines 330-395

**Description**: The `fit_cis()` method accepts `manual_guide_effects` parameter (DataFrame with columns `guide` and `log2FC`) and prepares tensors for use as priors, but these tensors are not yet passed to or used in `_model_x()`.

**What's Done**:
- Parameter parsing and validation in `fit_cis()` (lines 337-368)
- Tensor preparation: `manual_guide_log2fc_tensor` and `manual_guide_mask_tensor`
- Pseudocode documentation explaining intended usage (lines 370-394)

**What's Needed**:
1. Pass `manual_guide_log2fc_tensor`, `manual_guide_mask_tensor`, and `prior_strength` to `_model_x()`
2. Modify `_model_x()` to use manual priors when sampling `mu_x` or `log2_x_eff_g`:
   ```python
   # Inside guides_plate:
   for g in range(G):
       if manual_guide_mask_tensor[g] == 1.0:
           # This guide has a manual prior
           prior_mean = manual_guide_log2fc_tensor[g]
           prior_sd = 1.0 / prior_strength
           # Sample with informative prior
       else:
           # Use standard hierarchical prior
   ```

**Design Decisions to Make**:
1. Should `prior_sd = 1.0 / prior_strength`, or another function?
2. Should manual effects override hierarchical priors completely, or combine them?
3. Should NTC guide always have log2FC=0 enforced, or learned?
4. How to handle cell-line-specific effects (alpha_x) with manual priors?

**Testing**: Once implemented, add test in `tests/test_manual_guide_priors.py`

---

## Low Priority / Future Work

### 6. ~~Polynomial Degree Configuration~~ ✅ COMPLETED

**Status**: ✅ Already implemented

**Location**: `bayesDREAM/fitting/trans.py` line 839

**Description**: Polynomial degree is configurable via `polynomial_degree` parameter in `fit_trans()` with default value of 6.

**Usage**:
```python
model.fit_trans(
    sum_factor_col='sum_factor',
    function_type='polynomial',
    polynomial_degree=8  # Change from default of 6
)
```

**What's Still Needed**:
- Add tests for different polynomial degrees in test suite

---

### 7. Documentation Updates

**Status**: Mostly complete

**What's Done**:
- CLAUDE.md updated with modular structure and new API (2025-11-19)
- README.md updated with refactoring note and new API examples (2025-11-24)
- API_REFERENCE.md updated with new initialization parameters (2025-11-10)
- QUICKSTART_MULTIMODAL.md updated with new API examples (2025-11-10)
- ARCHITECTURE.md updated with new parameter references (2025-11-10)
- INITIALIZATION.md updated with new API examples (2025-11-07)
- OUTSTANDING_TASKS.md updated (2025-01-23)
- HILL_FUNCTION_PRIORS.md documents all Hill priors (2025-11-26)
- Repository cleanup and documentation archiving (2025-12-12)

**What's Needed**:
- Add comprehensive API reference for all public methods
- Create usage examples for each modality type
- Add troubleshooting guide
- Document common error messages and solutions

---

### 8. Example Workflows

**Status**: Basic examples exist

**Location**: `examples/` directory

**What's Needed**:
- Comprehensive multi-modal workflow example
- ATAC modality example
- Splicing modality example with all types (sj, donor, acceptor, exon_skip)
- Custom modality example (SpliZ)
- Example showing save/load workflow
- Example showing guide-prior usage (after #5 is implemented)

---

### 9. Performance Optimization

**Status**: Not started

**Ideas**:
- Profile code to identify bottlenecks
- Optimize tensor operations in Pyro models
- Consider GPU memory optimization strategies
- Investigate minibatch training for very large datasets

---

### 10. Additional Distributions

**Status**: Core distributions implemented

**Implemented**: negbinom, multinomial, binomial, normal, studentt

**Not Currently Planned**:
- Gamma distribution (for non-negative continuous data) - may revisit if use case arises

---

## Long-term / Exploratory

These tasks represent significant architectural changes and are not planned for near-term implementation.

### 11. Combinatorial Cis Effects (Multiple Cis Genes)

**Status**: Not planned - long-term research direction

**Description**: Support experiments with multiple cis genes being perturbed simultaneously (e.g., double knockouts, combinatorial screens). Currently the model assumes a single cis gene.

**Challenges**:
- How to model interaction effects between multiple cis genes
- Exponential growth in combinations (2 genes = 4 states, 3 genes = 8 states, etc.)
- Statistical power requirements for estimating interaction terms
- Computational complexity of fitting combinatorial models

**Potential Approaches**:
1. **Additive model**: `x_true = x_cis1 + x_cis2 + ...` (no interactions)
2. **Multiplicative model**: `x_true = x_cis1 * x_cis2 * ...` (in log space: additive)
3. **Full factorial**: Fit separate effects for each combination (limited scalability)
4. **Hierarchical interaction model**: Sparse priors on interaction terms

**Prerequisites**:
- High-MOI support (#2) should be fully tested first
- Guide-prior infrastructure (#5) may help constrain combinatorial fits
- Significant theoretical work needed on identifiability

**Note**: This is a research direction, not a planned feature. Would require substantial model redesign.

---

### 12. Dynamic ATAC Region Identification

**Status**: Not planned - long-term research direction

**Description**: Rather than using pre-determined ATAC peaks/motifs/regions, dynamically identify key variable regions that respond to perturbations. Currently, users must specify regions of interest upfront.

**Challenges**:
- Computational cost of scanning across all genomic positions
- Multiple testing burden across thousands/millions of potential regions
- How to define "region boundaries" in a data-driven way
- Integration with existing peak-calling pipelines

**Potential Approaches**:
1. **Two-stage**: Initial broad scan to identify candidate regions, then detailed fitting
2. **Sliding window**: Test windows of varying sizes across the genome
3. **Bayesian changepoint detection**: Identify boundaries where accessibility changes
4. **Variational feature selection**: Learn which regions are informative during fitting

**Related Work**:
- Differential accessibility methods (e.g., DiffBind, csaw)
- Topic modeling approaches for single-cell ATAC
- Peak-free methods in scATAC-seq analysis

**Note**: This is a research direction that would significantly expand scope beyond current perturbation-response modeling.

---

## Completed Recently ✓

- ✅ Save/Load for Cis Fit (2025-01-23)
  - `save_cis_fit()` and `load_cis_fit()` working
  - Tested with posterior and point estimate types
- ✅ Binomial/multinomial trans fitting bug fixes (2025-12-12)
  - Fixed critical alpha_y_full construction (ones → zeros)
  - Fixed Hill function plotting for binomial/multinomial
  - Enabled multinomial technical correction
  - Fixed losses_trans storage on model/modality objects
- ✅ Repository cleanup and documentation archiving (2025-12-12)
  - Moved historical technical docs to docs/archive/technical/
  - Deleted legacy environment.yml
  - Updated documentation index
- ✅ Technical correction in trans priors (2025-11-27)
  - Apply inverse correction before computing guide means for A/Vmax priors
  - Prevents biased priors from confounding technical and biological variation
- ✅ Modular refactoring (2025-10-22)
  - Reduced model.py from 4537 to 311 lines
  - Delegation pattern for fitters, savers, loaders
  - Multi-modal infrastructure
  - Distribution registry system
  - Per-modality save/load
- ✅ API refactoring (2025-11-07)
  - modality_name, feature_meta, guide_covariates parameters
  - Generic negbinom modality creation - ATAC/custom as primary now supported
  - Documentation updates for new API across major docs
- ✅ Empirical Bayes initialization for binomial and multinomial (2025-11-04)
- ✅ Multinomial per-group zero-category masking (2025-11-05)
- ✅ Binomial per-group boundary checking (2025-11-05)

---

## Known Issues

### ~~guide_covariates not implemented for high-MOI mode~~ (fixed 2026-09-09)

**Status**: Implemented (`core.py`'s `_expand_guide_assignment_by_covariates`, called from `__init__`)

**Description**: `guide_covariates`/`guide_covariates_ntc` previously only built the
single-guide-mode `guide_used` column (guide name + covariate values -> per-covariate-group
guide effect). High-MOI mode has no per-cell single-guide identity (`guide_assignment` is a
multi-hot matrix), so these params were accepted but silently ignored there — a config could
claim to model a covariate split that wasn't actually happening. Discovered via
`publication_runs/morris` (high-MOI, `guide_covariates: [lane]` in config, which did nothing;
fixed to `[]` there — lane is already handled via `set_technical_groups`/`sum_factor.covariates`,
and this dataset has no other reason to want a guide covariate).

**Implementation**: `_expand_guide_assignment_by_covariates()` splits each guide's column in
`guide_assignment`/`guide_meta` into one column per distinct combination of covariate values
observed among the cells carrying that guide (NTC-classified guides split by
`guide_covariates_ntc`, all others by `guide_covariates`) — the high-MOI analogue of
single-guide's `guide_used`. `_model_x` needed no changes — it already treats each
`guide_assignment` column as an independent latent effect summed per-cell via matmul, so this
is purely a data-prep step, called once at the end of `__init__` (works for both eager
`cis_gene=...` and deferred `add_cis_gene()` workflows without any changes to the latter —
`guide_meta['guide']` keeps the original, possibly-duplicated guide name, so
`guide_targets_dict` lookups elsewhere keep working unchanged). No-op when both covariate
lists are empty (the default), so existing pipelines are unaffected. Tests:
`tests/test_high_moi.py`'s `test_covariate_expansion_*` group.

**Known follow-on (not fixed)**: per-guide plots/summaries that key off
`guide_meta['guide']` (e.g. `plotting/basic.py`'s guide bar plots) will show one row per
(guide, covariate) column rather than one row per guide when covariates are used — correct
information, but cosmetically unlabeled (no dedicated column render distinguishing the split
in those specific plots yet; `guide_meta['guide_covariate_key']` has the value).

If you discover issues, please document them here with:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Suggested fix (if known)

---

## Notes

- Cis fitting is always negative binomial (core model design)
- All public API methods should maintain backward compatibility
- Tests should pass before merging new features
- Documentation should be updated alongside code changes
- Use TodoWrite tool for tracking active work
- See `docs/archive/RECENT_CHANGES_2025-12.md` for historical change details
