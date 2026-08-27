"""Test exclude_trans_genes(): by name, feature_meta query, and log2(mu_ntc) threshold."""
import numpy as np
import pandas as pd
import torch

import pytest


def _make_model():
    from bayesDREAM import bayesDREAM
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(1, 11)],
        'guide': ['g1', 'g2', 'g3', 'g4', 'g5', 'ntc1', 'ntc2', 'ntc3', 'ntc4', 'ntc5'],
        'target': ['GFI1B'] * 5 + ['ntc'] * 5,
        'cell_line': ['A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'B'],
        'sum_factor': [1.0] * 10,
    })
    gene_counts = pd.DataFrame(
        {f'cell{i}': [10 + i, 20 + i, 30 + i, 40 + i, 50 + i,
                      100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 1000 + i]
         for i in range(1, 11)},
        index=['GFI1B', 'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5',
               'GENE6', 'GENE7', 'GENE8', 'GENE9', 'GENE10'],
    )
    gene_meta = pd.DataFrame({
        'gene': ['GFI1B', 'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5',
                 'GENE6', 'GENE7', 'GENE8', 'GENE9', 'GENE10'],
        'protein_coding': [True, True, False, True, False, True,
                            True, True, False, True, True],
    })
    return bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=gene_meta,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='exclude_trans_genes_test',
    )


def _trans_feature_names(model):
    return model.get_modality(model.primary_modality).feature_names


def test_exclude_by_name():
    model = _make_model()
    n_before = model.get_modality(model.primary_modality).dims['n_features']
    model.exclude_trans_genes(genes=['GENE1', 'GENE3'])
    mod = model.get_modality(model.primary_modality)
    assert mod.dims['n_features'] == n_before - 2
    names = _trans_feature_names(model)
    assert 'GENE1' not in names
    assert 'GENE3' not in names
    assert 'GENE2' in names


def test_exclude_by_name_unknown_warns():
    model = _make_model()
    with pytest.warns(UserWarning, match='not found'):
        model.exclude_trans_genes(genes=['NOT_A_GENE'])


def test_exclude_by_feature_query():
    model = _make_model()
    mod = model.get_modality(model.primary_modality)
    non_coding = mod.feature_meta.loc[
        ~mod.feature_meta['protein_coding'], 'gene'
    ].tolist()
    assert len(non_coding) > 0

    model.exclude_trans_genes(feature_query='protein_coding == False')
    mod = model.get_modality(model.primary_modality)
    names = _trans_feature_names(model)
    for g in non_coding:
        assert g not in names
    assert (mod.feature_meta['protein_coding'] == True).all()


def test_exclude_by_min_log2_mu_ntc():
    model = _make_model()
    mod = model.get_modality(model.primary_modality)
    n_features = mod.dims['n_features']
    names = mod.feature_names

    # Fake an NTC posterior: GENE2 and GENE5 are very lowly expressed.
    mu_ntc_vals = np.full(n_features, 10.0)
    low_expr = {'GENE2', 'GENE5'}
    for i, n in enumerate(names):
        if n in low_expr:
            mu_ntc_vals[i] = 2 ** -6  # log2 = -6

    mod.posterior_samples_ntc = {
        'mu_ntc': torch.tensor(mu_ntc_vals, dtype=torch.float32).unsqueeze(0)  # [1, T]
    }

    model.exclude_trans_genes(min_log2_mu_ntc=-4)
    mod = model.get_modality(model.primary_modality)
    remaining_names = mod.feature_names
    for g in low_expr:
        assert g not in remaining_names
    assert mod.dims['n_features'] == n_features - len(low_expr)
    # posterior_samples_ntc trimmed to match
    assert mod.posterior_samples_ntc['mu_ntc'].shape[-1] == mod.dims['n_features']


def test_exclude_min_log2_mu_ntc_without_fit_ntc_warns():
    model = _make_model()
    n_before = model.get_modality(model.primary_modality).dims['n_features']
    with pytest.warns(UserWarning, match='requires fit_ntc'):
        model.exclude_trans_genes(min_log2_mu_ntc=-4)
    assert model.get_modality(model.primary_modality).dims['n_features'] == n_before


def test_exclude_no_criteria_raises():
    model = _make_model()
    with pytest.raises(ValueError, match='Provide at least one'):
        model.exclude_trans_genes()


def test_exclude_combined_criteria():
    model = _make_model()
    mod = model.get_modality(model.primary_modality)
    n_features = mod.dims['n_features']
    names = mod.feature_names
    mu_ntc_vals = np.full(n_features, 10.0)
    idx_gene4 = names.index('GENE4')
    mu_ntc_vals[idx_gene4] = 2 ** -8
    mod.posterior_samples_ntc = {
        'mu_ntc': torch.tensor(mu_ntc_vals, dtype=torch.float32).unsqueeze(0)
    }

    model.exclude_trans_genes(
        genes=['GENE1'],
        feature_query='protein_coding == False',
        min_log2_mu_ntc=-4,
    )
    mod = model.get_modality(model.primary_modality)
    remaining = set(mod.feature_names)
    assert 'GENE1' not in remaining
    assert 'GENE4' not in remaining
    assert 'GENE2' not in remaining  # non-protein-coding
    assert 'GENE6' in remaining
