"""Test gene metadata handling in bayesDREAM."""

import pandas as pd

import pytest


@pytest.fixture(scope='module')
def gene_meta_model_factory(shared_test_data):
    from bayesDREAM import bayesDREAM

    meta = shared_test_data['meta'].copy()
    gene_counts = shared_test_data['gene_counts'].copy()

    def make_model(**kwargs):
        return bayesDREAM(
            meta=meta.copy(),
            counts=gene_counts.copy(),
            cis_gene='GFI1B',
            output_dir='./test_output',
            **kwargs,
        )

    return make_model


def _gene_feature_meta(model):
    return model.get_modality('gene').feature_meta


def test_no_gene_meta_creates_minimal_metadata(gene_meta_model_factory):
    model = gene_meta_model_factory(label='test_no_meta')
    feature_meta = _gene_feature_meta(model)
    assert feature_meta is not None
    assert feature_meta.shape[0] > 0


def test_full_gene_meta_accepted(gene_meta_model_factory, shared_test_data):
    gene_names = shared_test_data['gene_counts'].index.tolist()
    gene_meta = pd.DataFrame(
        {
            'gene': gene_names,
            'gene_name': [f'GeneSymbol_{gene}' for gene in gene_names],
            'gene_id': [f'ENSG{i:08d}' for i in range(len(gene_names))],
            'chromosome': ['chr1'] * len(gene_names),
            'biotype': ['protein_coding'] * len(gene_names),
        },
        index=gene_names,
    )
    model = gene_meta_model_factory(feature_meta=gene_meta, label='test_with_meta')
    assert 'gene' in _gene_feature_meta(model).columns


def test_gene_meta_with_gene_name_only(gene_meta_model_factory, shared_test_data):
    gene_names = shared_test_data['gene_counts'].index.tolist()
    gene_meta_simple = pd.DataFrame(
        {'gene_name': gene_names},
        index=gene_names,
    )
    model = gene_meta_model_factory(feature_meta=gene_meta_simple, label='test_simple_meta')
    feature_meta = _gene_feature_meta(model)
    assert 'gene_name' in feature_meta.columns


def test_gene_meta_index_becomes_gene_column(gene_meta_model_factory, shared_test_data):
    gene_names = shared_test_data['gene_counts'].index.tolist()
    gene_meta_indexed = pd.DataFrame(
        {
            'gene_name': [f'GeneSymbol_{gene}' for gene in gene_names],
            'gene_id': [f'ENSG{i:08d}' for i in range(len(gene_names))],
        },
        index=gene_names,
    )
    gene_meta_indexed.index.name = 'gene_symbol'
    model = gene_meta_model_factory(feature_meta=gene_meta_indexed, label='test_indexed_meta')
    feature_meta = _gene_feature_meta(model)
    assert 'gene_name' in feature_meta.columns
