"""Test multi-modal fitting infrastructure.

Tests:
1. Backward compatibility: bayesDREAM works exactly like bayesDREAM for gene expression
2. fit_modality_technical() delegates correctly to fit_technical()
3. fit_modality_trans() delegates correctly to fit_trans()
4. Distribution registry is properly loaded
"""

import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


# --- Distribution registry tests (no setup) ---

def test_all_distributions_registered():
    from bayesDREAM import DISTRIBUTION_REGISTRY
    for dist in ('negbinom', 'multinomial', 'binomial', 'normal'):
        assert dist in DISTRIBUTION_REGISTRY


def test_helper_functions():
    from bayesDREAM import requires_denominator, is_3d_distribution
    assert requires_denominator('binomial')
    assert not requires_denominator('negbinom')
    assert is_3d_distribution('multinomial')
    assert not is_3d_distribution('negbinom')


def test_get_observation_sampler():
    from bayesDREAM import get_observation_sampler
    sampler = get_observation_sampler('negbinom', 'trans')
    assert callable(sampler)


# --- Multi-modal model tests ---

@pytest.fixture(scope='module')
def multimodal_model(shared_test_data):
    from bayesDREAM import bayesDREAM
    meta = shared_test_data['meta']
    gene_counts = shared_test_data['gene_counts']
    n_genes = gene_counts.shape[0] - 1
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_multimodal',
        device='cpu',
        cores=1,
    )
    return {'model': model, 'n_genes': n_genes}


def test_model_creation(multimodal_model):
    model = multimodal_model['model']
    assert model.primary_modality == 'gene'
    assert len(model.modalities) > 0


def test_core_fitting_methods_exist(multimodal_model):
    model = multimodal_model['model']
    for method in ('fit_technical', 'fit_cis', 'fit_trans', 'set_technical_groups'):
        assert hasattr(model, method), f"Missing method: {method}"


def test_gene_modality_excludes_cis_gene(multimodal_model):
    model = multimodal_model['model']
    n_genes = multimodal_model['n_genes']
    gene_modality = model.get_modality('gene')
    gene_names = gene_modality.feature_meta['gene'].tolist()
    assert 'GFI1B' not in gene_names, "Cis gene should be excluded from gene modality"
    assert len(gene_names) == n_genes


def test_base_class_retains_cis_gene(multimodal_model):
    model = multimodal_model['model']
    assert 'GFI1B' in model.counts.index
    assert model.cis_gene == 'GFI1B'
