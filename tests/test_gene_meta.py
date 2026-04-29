"""Test gene metadata handling in bayesDREAM."""

import numpy as np
import pandas as pd

import pytest


def _make_base_data():
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(1, 21)],
        'guide': ['g1', 'g2', 'g3', 'g4', 'g5'] * 4,
        'target': ['GFI1B'] * 10 + ['ntc'] * 10,
        'cell_line': ['A'] * 10 + ['B'] * 10,
        'sum_factor': [1.0] * 20,
    })
    gene_counts = pd.DataFrame(
        np.random.randint(10, 100, (10, 20)),
        index=[f'GENE{i}' for i in range(10)],
        columns=[f'cell{i}' for i in range(1, 21)],
    )
    gene_counts.loc['GFI1B'] = np.random.randint(50, 150, 20)
    return meta, gene_counts


def _make_model(meta, gene_counts, **kwargs):
    from bayesDREAM import bayesDREAM
    return bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        **kwargs,
    )


@pytest.fixture(scope='module')
def gene_meta_test_data():
    np.random.seed(0)
    return _make_base_data()


def test_no_gene_meta_creates_minimal_metadata(gene_meta_test_data):
    meta, gene_counts = gene_meta_test_data
    model = _make_model(meta, gene_counts, label='test_no_meta')
    assert model.gene_meta is not None
    assert model.gene_meta.shape[0] > 0


def test_full_gene_meta_accepted(gene_meta_test_data):
    meta, gene_counts = gene_meta_test_data
    gene_meta = pd.DataFrame({
        'gene': [f'GENE{i}' for i in range(10)] + ['GFI1B'],
        'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
        'gene_id': [f'ENSG{i:08d}' for i in range(11)],
        'chromosome': ['chr1'] * 11,
        'biotype': ['protein_coding'] * 11,
    }, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])
    model = _make_model(meta, gene_counts, feature_meta=gene_meta, label='test_with_meta')
    assert 'gene' in model.gene_meta.columns


def test_gene_meta_with_gene_name_only(gene_meta_test_data):
    meta, gene_counts = gene_meta_test_data
    gene_meta_simple = pd.DataFrame({
        'gene_name': [f'GENE{i}' for i in range(10)] + ['GFI1B'],
    }, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])
    model = _make_model(meta, gene_counts, feature_meta=gene_meta_simple, label='test_simple_meta')
    assert 'gene' in model.gene_meta.columns, "'gene' column should be created from 'gene_name'"


def test_gene_meta_index_becomes_gene_column(gene_meta_test_data):
    meta, gene_counts = gene_meta_test_data
    gene_meta_indexed = pd.DataFrame({
        'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
        'gene_id': [f'ENSG{i:08d}' for i in range(11)],
    })
    gene_meta_indexed.index = [f'GENE{i}' for i in range(10)] + ['GFI1B']
    gene_meta_indexed.index.name = 'gene_symbol'
    model = _make_model(meta, gene_counts, feature_meta=gene_meta_indexed, label='test_indexed_meta')
    assert 'gene' in model.gene_meta.columns, "'gene' column should be created from index"
