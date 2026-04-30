"""Test summary export functionality using mock posterior data.

Validates all summary export methods without requiring a full fitting pipeline.
"""

import os
import numpy as np
import pandas as pd

import pytest


def _make_toy_data(n_genes=10, n_cells=100, n_guides=6, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['K562', 'HEL'], n_cells),
        'target': ['GFI1B'] * 60 + ['ntc'] * 40,
        'sum_factor': np.random.lognormal(0, 0.2, n_cells),
        'sum_factor_adj': np.random.lognormal(0, 0.2, n_cells),
    })
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, (len(genes), n_cells)),
        index=genes,
        columns=meta['cell'],
    )
    return meta, gene_counts


@pytest.fixture(scope='module')
def summary_model(tmp_path_factory):
    import torch
    from bayesDREAM import bayesDREAM

    tmpdir = str(tmp_path_factory.mktemp('summary_simple'))
    meta, gene_counts = _make_toy_data()
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        output_dir=tmpdir,
        label='summary_test_simple',
        device='cpu',
    )
    model.set_technical_groups(['cell_line'])

    # Inject mock technical fit results (point estimate style: [C, T]).
    gene_mod = model.modalities['gene']
    n_features = gene_mod.dims['n_features']
    n_groups = 2
    gene_mod.alpha_y_prefit = torch.ones((n_groups, n_features), device=model.device)

    # Inject mock CIS fit posteriors.
    n_cells_total = len(model.meta)
    model.x_true = torch.randn(n_cells_total, device=model.device) + 5
    model.posterior_samples_cis = {
        'x_true': torch.randn((100, n_cells_total), device=model.device) + 5
    }

    # Inject mock trans fit posteriors (additive_hill).
    n_features_trans = model.modalities['gene'].dims['n_features']
    n_samples = 100
    model.function_type = 'additive_hill'
    model.posterior_samples_trans = {
        'Vmax_a': torch.randn((n_samples, n_features_trans), device=model.device),
        'K_a': torch.randn((n_samples, n_features_trans), device=model.device) + 5,
        'n_a': torch.abs(torch.randn((n_samples, n_features_trans), device=model.device)) + 1.5,
        'Vmax_b': torch.randn((n_samples, n_features_trans), device=model.device),
        'K_b': torch.randn((n_samples, n_features_trans), device=model.device) + 5,
        'n_b': torch.abs(torch.randn((n_samples, n_features_trans), device=model.device)) + 1.5,
        'pi_y': torch.rand((n_samples, n_features_trans), device=model.device) * 0.5 + 0.5,
        'alpha': torch.rand((n_features_trans,), device=model.device),
        'beta': torch.rand((n_features_trans,), device=model.device),
    }

    return {
        'model': model,
        'outdir': os.path.join(tmpdir, 'summary_test_simple'),
    }


def test_technical_summary_columns(summary_model):
    model = summary_model['model']
    tech_df = model.save_technical_summary()
    for col in ('feature', 'modality', 'distribution'):
        assert col in tech_df.columns
    assert any('alpha_y' in col for col in tech_df.columns)


def test_technical_summary_csv_created(summary_model):
    model = summary_model['model']
    outdir = summary_model['outdir']
    model.save_technical_summary()
    csv_path = os.path.join(outdir, 'technical_feature_summary_gene.csv')
    assert os.path.exists(csv_path)


def test_cis_summary_columns(summary_model):
    model = summary_model['model']
    guide_df, _ = model.save_cis_summary()
    for col in ('guide', 'target', 'x_true_mean', 'x_true_lower', 'x_true_upper'):
        assert col in guide_df.columns


def test_cis_summary_csvs_created(summary_model):
    model = summary_model['model']
    outdir = summary_model['outdir']
    model.save_cis_summary()
    assert os.path.exists(os.path.join(outdir, 'cis_guide_summary.csv'))
    assert os.path.exists(os.path.join(outdir, 'cis_cell_summary.csv'))


def test_trans_summary_additive_hill_columns(summary_model):
    model = summary_model['model']
    trans_df = model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
    for col in ('feature', 'function_type', 'observed_log2fc', 'Vmax_a_mean',
                'EC50_a_mean', 'inflection_a_mean', 'full_log2fc_mean'):
        assert col in trans_df.columns


def test_trans_summary_csv_created(summary_model):
    model = summary_model['model']
    outdir = summary_model['outdir']
    model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
    assert os.path.exists(os.path.join(outdir, 'trans_feature_summary_gene.csv'))


def test_trans_summary_polynomial_columns(summary_model):
    import torch

    model = summary_model['model']
    n_features_trans = model.modalities['gene'].dims['n_features']
    n_samples = 100
    degree = 6
    model.function_type = 'polynomial'
    model.posterior_samples_trans = {
        'poly_coefs': torch.randn((n_samples, n_features_trans, degree), device=model.device)
    }
    poly_df = model.save_trans_summary(compute_inflection=False, compute_full_log2fc=True)
    assert 'coef_0_mean' in poly_df.columns
    assert 'full_log2fc_mean' in poly_df.columns
