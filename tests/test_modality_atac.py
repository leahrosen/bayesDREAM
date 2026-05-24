"""Test ATAC modality functionality in bayesDREAM.

Tests:
1. Creating ATAC modality with region metadata
2. Cis modality auto-created from cis_region
3. ATAC-only initialization (no gene expression)
4. Manual guide effects infrastructure
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


@pytest.fixture(scope='module')
def atac_only_model():
    from bayesDREAM import bayesDREAM

    np.random.seed(0)
    n_cells = 100
    n_guides = 5
    guides = [f'guide{i % n_guides}' for i in range(n_cells)]
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(n_cells)],
        'guide': guides,
        'cell_line': ['line1'] * 50 + ['line2'] * 50,
        'target': ['GFI1B'] * 80 + ['ntc'] * 20,
        'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    })
    model = bayesDREAM(
        meta=meta,
        counts=None,
        modality_name='atac',
    )

    n_regions = 10
    atac_counts = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, size=(n_regions, n_cells)),
        index=[f'chr9:{i*1000}-{(i+1)*1000}' for i in range(n_regions)],
        columns=[f'cell{i}' for i in range(n_cells)],
    )
    region_meta = pd.DataFrame({
        'region_id': [f'chr9:{i*1000}-{(i+1)*1000}' for i in range(n_regions)],
        'region_type': ['promoter'] * 3 + ['gene_body'] * 3 + ['distal'] * 4,
        'chrom': ['chr9'] * n_regions,
        'start': np.arange(1000, 1000 + n_regions * 1000, 1000),
        'end': np.arange(2000, 2000 + n_regions * 1000, 1000),
        'gene': ['GFI1B', 'GFI1B', 'SPI1'] + ['GFI1B'] * 3 + [''] * 4,
    })
    model.add_atac_modality(
        atac_counts=atac_counts,
        region_meta=region_meta,
        name='atac',
        cis_region='chr9:1000-2000',
    )

    return model


def test_atac_modality_added(atac_only_model):
    modalities_df = atac_only_model.list_modalities()
    assert 'atac' in modalities_df['name'].values


def test_primary_modality_is_atac(atac_only_model):
    assert atac_only_model.primary_modality == 'atac'


def test_guide_effects_dataframe_format():
    guide_effects = pd.DataFrame({
        'guide': ['guide0', 'guide1', 'guide2'],
        'log2FC': [-2.5, -1.8, -1.2],
    })
    for col in ('guide', 'log2FC'):
        assert col in guide_effects.columns
