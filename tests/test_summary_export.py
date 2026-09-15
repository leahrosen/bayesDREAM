"""Test summary export functionality with a full fitting pipeline.

Runs technical → cis → trans on toy CSV data and validates that all
summary export methods produce R-friendly CSV files with the expected columns.
"""

import os
from pathlib import Path
import pandas as pd


import torch
import pyro
import pytest
pytestmark = pytest.mark.slow

TOYDATA_DIR = Path(__file__).resolve().parents[1] / 'toydata'
TOYDATA_REQUIRED_FILES = ('cell_meta.csv', 'gene_counts.csv', 'gene_meta.csv')
TOYDATA_AVAILABLE = all((TOYDATA_DIR / fname).exists() for fname in TOYDATA_REQUIRED_FILES)

NITERS_TECH = 1000
NITERS_CIS = 100
NITERS_TRANS = 100


@pytest.fixture(scope='module')
def summary_export_model(tmp_path_factory):
    from bayesDREAM import bayesDREAM

    tmpdir = str(tmp_path_factory.mktemp('summary_export'))

    meta = pd.read_csv(TOYDATA_DIR / 'cell_meta.csv')
    gene_counts = pd.read_csv(TOYDATA_DIR / 'gene_counts.csv', index_col=0)
    gene_meta = pd.read_csv(TOYDATA_DIR / 'gene_meta.csv')
    gene_meta.set_index('gene_name', inplace=True)

    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=gene_meta,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        output_dir=tmpdir,
        label='summary_test',
        device='cpu',
    )
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=NITERS_TECH, lr=0.001)
    model.fit_cis(sum_factor_col='sum_factor', niters=NITERS_CIS, lr=0.01)
    model.fit_trans(
        sum_factor_col='sum_factor',
        function_type='additive_hill',
        niters=NITERS_TRANS,
        lr=0.01,
    )
    return {
        'model': model,
        'tmpdir': tmpdir,
        'outdir': os.path.join(tmpdir, 'summary_test'),
    }


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_technical_summary_structure(summary_export_model):
    model = summary_export_model['model']
    tech_df = model.save_ntc_summary()
    for col in ('feature', 'modality', 'distribution'):
        assert col in tech_df.columns
    alpha_cols = [c for c in tech_df.columns if 'alpha_y' in c]
    assert len(alpha_cols) > 0


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_technical_summary_csv(summary_export_model):
    model = summary_export_model['model']
    outdir = summary_export_model['outdir']
    model.save_ntc_summary()
    assert os.path.exists(os.path.join(outdir, 'technical_feature_summary_gene.csv'))


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_cis_summary_guide_level_columns(summary_export_model):
    model = summary_export_model['model']
    guide_df, _ = model.save_cis_summary(include_cell_level=True)
    for col in ('guide', 'target', 'n_cells', 'x_true_mean', 'x_true_lower', 'x_true_upper'):
        assert col in guide_df.columns


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_cis_summary_cell_level_columns(summary_export_model):
    model = summary_export_model['model']
    _, cell_df = model.save_cis_summary(include_cell_level=True)
    for col in ('cell', 'guide', 'x_true_mean'):
        assert col in cell_df.columns


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_cis_summary_csvs(summary_export_model):
    model = summary_export_model['model']
    outdir = summary_export_model['outdir']
    model.save_cis_summary()
    assert os.path.exists(os.path.join(outdir, 'cis_guide_summary.csv'))
    assert os.path.exists(os.path.join(outdir, 'cis_cell_summary.csv'))


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_trans_summary_additive_hill_columns(summary_export_model):
    model = summary_export_model['model']
    trans_df = model.save_trans_summary(
        compute_inflection=True, compute_full_log2fc=True
    )
    expected_cols = (
        'feature', 'modality', 'distribution', 'function_type',
        'observed_log2fc', 'observed_log2fc_lower', 'observed_log2fc_upper',
        'Vmax_a_mean', 'K_a_mean', 'EC50_a_mean',
        'Vmax_b_mean', 'K_b_mean', 'EC50_b_mean',
        'inflection_a_mean', 'inflection_b_mean', 'full_log2fc_mean'
    )
    for col in expected_cols:
        assert col in trans_df.columns


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_trans_summary_csv(summary_export_model):
    model = summary_export_model['model']
    outdir = summary_export_model['outdir']
    model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
    assert os.path.exists(os.path.join(outdir, 'trans_feature_summary_gene.csv'))


@pytest.mark.skipif(not TOYDATA_AVAILABLE, reason="toydata not found")
def test_trans_summary_polynomial_columns(summary_export_model):
    model = summary_export_model['model']
    model.fit_trans(
        sum_factor_col='sum_factor',
        function_type='polynomial',
        niters=NITERS_TRANS,
        lr=0.01,
    )
    poly_df = model.save_trans_summary(compute_inflection=False, compute_full_log2fc=True)
    coef_cols = [c for c in poly_df.columns if 'coef_' in c]
    assert len(coef_cols) > 0
    assert 'full_log2fc_mean' in poly_df.columns
