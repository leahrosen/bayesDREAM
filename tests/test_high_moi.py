"""Test high MOI (multiple guides per cell) functionality.

Verifies:
1. Initialization with guide_assignment matrix and guide_meta
2. Backward compatibility (single-guide mode still works)
3. Additive guide effects in fit_cis
4. Proper handling of NTC cells
"""

import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow

NITERS_CIS = 100


def _make_high_moi_data(n_cells=100, n_genes=50, n_guides=6, seed=42):
    import torch
    import pyro
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'cell_line': np.random.choice(['K562', 'MOLM13'], n_cells),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)],
    )
    guide_assignment = np.zeros((n_cells, n_guides), dtype=int)
    guide_assignment[0:30, 0] = 1
    guide_assignment[0:30, 1] = 1
    guide_assignment[30:60, 2] = 1
    guide_assignment[60:80, 3] = 1
    guide_assignment[60:80, 4] = 1
    guide_assignment[80:100, 5] = 1
    guide_meta = pd.DataFrame({
        'guide': ['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'],
        'target': ['GFI1B', 'GFI1B', 'MYB', 'MYB', 'MYB', 'ntc'],
    }, index=['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'])
    return meta, counts, guide_assignment, guide_meta, n_guides


def _make_single_guide_data(n_cells=50, n_genes=30, seed=42):
    import torch
    import pyro
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['guide_A'] * 25 + ['ntc'] * 25,
        'target': ['GFI1B'] * 25 + ['ntc'] * 25,
        'cell_line': ['K562'] * n_cells,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)],
    )
    return meta, counts


# High MOI Initialization Tests
@pytest.fixture(scope='module')
def high_moi_model():
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM
    
    meta, counts, guide_assignment, guide_meta, n_guides = _make_high_moi_data()
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_high_moi',
        device='cpu',
    )
    return {'model': model, 'n_guides': n_guides}


def test_high_moi_mode_active(high_moi_model):
    model = high_moi_model['model']
    assert model.is_high_moi


def test_guide_assignment_attribute(high_moi_model):
    model = high_moi_model['model']
    assert hasattr(model, 'guide_assignment')
    assert hasattr(model, 'guide_meta')
    assert hasattr(model, 'guide_assignment_tensor')


def test_guide_meta_length(high_moi_model):
    # Only NTC and cis-gene guides are kept; MYB guides (C, D, E) are pruned
    # guide_A (GFI1B), guide_B (GFI1B), ntc_1 (ntc) → 3 guides remain
    model = high_moi_model['model']
    assert len(model.guide_meta) == 3


def test_target_column_created(high_moi_model):
    model = high_moi_model['model']
    assert 'target' in model.meta.columns


def test_ntc_cell_count(high_moi_model):
    model = high_moi_model['model']
    ntc_cells = (model.meta['target'] == 'ntc').sum()
    assert ntc_cells == 20


def test_targeting_cell_count(high_moi_model):
    model = high_moi_model['model']
    targeting_cells = (model.meta['target'] == 'GFI1B').sum()
    assert targeting_cells == 30


def test_guide_code_is_placeholder(high_moi_model):
    model = high_moi_model['model']
    assert (model.meta['guide_code'] == -1).all()


# Verify single-guide mode is unaffected by high-MOI changes.
@pytest.fixture(scope='module')
def single_guide_model():
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM
    
    meta, counts = _make_single_guide_data()
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_single_guide',
        device='cpu',
    )
    return model


def test_not_high_moi_mode(single_guide_model):
    assert not single_guide_model.is_high_moi


def test_no_guide_assignment_attribute(single_guide_model):
    assert not hasattr(single_guide_model, 'guide_assignment')


def test_guide_code_not_placeholder(single_guide_model):
    assert 'guide_code' in single_guide_model.meta.columns
    assert not (single_guide_model.meta['guide_code'] == -1).all()


# High MOI CIS Fitting Tests
@pytest.fixture(scope='module')
def high_moi_fitted_model():
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM
    
    meta, counts, guide_assignment, guide_meta, _ = _make_high_moi_data()
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_high_moi_cis',
        device='cpu',
    )
    model.fit_cis(
        sum_factor_col='sum_factor',
        lr=1e-2,
        niters=NITERS_CIS,
        nsamples=10,
        tolerance=1e-3,
    )
    return model


def test_posterior_samples_cis_exists(high_moi_fitted_model):
    assert hasattr(high_moi_fitted_model, 'posterior_samples_cis')


def test_x_true_in_posterior(high_moi_fitted_model):
    assert 'x_true' in high_moi_fitted_model.posterior_samples_cis


def test_x_eff_g_in_posterior(high_moi_fitted_model):
    assert 'x_eff_g' in high_moi_fitted_model.posterior_samples_cis


def test_x_true_cell_count(high_moi_fitted_model):
    x_true = high_moi_fitted_model.posterior_samples_cis['x_true']
    assert x_true.shape[1] == len(high_moi_fitted_model.meta)


def test_x_eff_g_guide_count(high_moi_fitted_model):
    x_eff_g = high_moi_fitted_model.posterior_samples_cis['x_eff_g']
    assert x_eff_g.shape[1] == high_moi_fitted_model.guide_assignment.shape[1]
