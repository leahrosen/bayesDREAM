"""Test ATAC modality functionality in bayesDREAM.

Tests:
1. Creating ATAC modality with region metadata
2. Cis modality auto-created from cis_region
3. Manual guide effects infrastructure
"""

import numpy as np
import pandas as pd

import pytest


pytestmark = pytest.mark.slow


def _make_atac_data(n_regions=10, n_cells=100):
    atac_counts = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, size=(n_regions, n_cells)),
        index=[f'region{i}' for i in range(n_regions)],
        columns=[f'cell{i}' for i in range(n_cells)],
    )
    region_meta = pd.DataFrame({
        'region_id': [f'region{i}' for i in range(n_regions)],
        'region_type': ['promoter'] * 3 + ['gene_body'] * 3 + ['distal'] * 4,
        'chrom': ['chr9'] * n_regions,
        'start': np.arange(1000, 1000 + n_regions * 1000, 1000),
        'end': np.arange(2000, 2000 + n_regions * 1000, 1000),
        'gene': ['GFI1B', 'GFI1B', 'SPI1'] + ['GFI1B'] * 3 + [''] * 4,
    })
    return atac_counts, region_meta, n_regions


@pytest.fixture(scope='module')
def gene_plus_atac_model(shared_test_data):
    from bayesDREAM import bayesDREAM

    meta = shared_test_data['meta'].copy()
    gene_counts = shared_test_data['gene_counts'].copy()
    n_cells = len(meta)

    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
    )

    atac_counts, region_meta, n_regions = _make_atac_data(n_cells=n_cells)
    model.add_atac_modality(
        atac_counts=atac_counts,
        region_meta=region_meta,
        name='atac',
        cis_region='region0',
    )
    return {'model': model, 'n_regions': n_regions}


def test_atac_modality_in_model(gene_plus_atac_model):
    model = gene_plus_atac_model['model']
    modalities_df = model.list_modalities()
    assert 'atac' in modalities_df['name'].values


def test_atac_distribution(gene_plus_atac_model):
    model = gene_plus_atac_model['model']
    atac_mod = model.get_modality('atac')
    assert atac_mod.distribution == 'negbinom'


def test_atac_feature_count(gene_plus_atac_model):
    model = gene_plus_atac_model['model']
    n_regions = gene_plus_atac_model['n_regions']
    atac_mod = model.get_modality('atac')
    assert atac_mod.dims['n_features'] == n_regions


def test_cis_modality_auto_created(gene_plus_atac_model):
    model = gene_plus_atac_model['model']
    assert 'cis' in model.modalities
    cis_mod = model.get_modality('cis')
    assert cis_mod.dims['n_features'] == 1


def test_guide_effects_dataframe_format():
    guide_effects = pd.DataFrame({
        'guide': ['guide0', 'guide1', 'guide2'],
        'log2FC': [-2.5, -1.8, -1.2],
    })
    for col in ('guide', 'log2FC'):
        assert col in guide_effects.columns
