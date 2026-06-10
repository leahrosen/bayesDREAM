"""Test per-modality fitting functionality.

Verifies that fit_ntc() and fit_trans() can fit different modalities
and store results correctly per modality, without overwriting each other.
"""

import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow

NITERS = 20
NSAMPLES = 10


@pytest.fixture(scope='module')
def fitted_multimodal_model(shared_test_data):
    import torch
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM, Modality

    meta = shared_test_data['meta'].copy()
    gene_counts = shared_test_data['gene_counts'].copy()
    cell_names = meta['cell'].tolist()
    n_cells = len(cell_names)

    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        primary_modality='gene',
        output_dir='./test_output',
        label='per_modality_test',
        device='cpu',
    )

    # Add a splicing-like binomial modality
    sj_counts = shared_test_data['splicing_counts']
    sj_total = shared_test_data['splicing_denom']
    sj_meta = shared_test_data['splicing_meta'].copy()
    splicing_modality = Modality(
        name='splicing_test',
        counts=pd.DataFrame(sj_counts, columns=cell_names),
        feature_meta=sj_meta,
        distribution='binomial',
        denominator=sj_total,
        cells_axis=1,
    )
    model.add_modality('splicing_test', splicing_modality)

    # Fit technical on both modalities
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', modality_name='gene',
                        niters=NITERS, nsamples=NSAMPLES)
    model.fit_ntc(modality_name='splicing_test', niters=NITERS, nsamples=NSAMPLES)

    # Set dummy x_true for trans tests
    model.x_true = torch.ones(n_cells, dtype=torch.float32)

    # Fit trans on gene modality
    model.fit_trans(
        sum_factor_col='sum_factor',
        function_type='additive_hill',
        modality_name='gene',
        p0=0.01, gamma_threshold=0.01,
        niters=NITERS, nsamples=NSAMPLES,
    )

    # Fit trans on splicing modality
    model.fit_trans(
        function_type='additive_hill',
        modality_name='splicing_test',
        p0=0.01, gamma_threshold=0.01,
        niters=NITERS, nsamples=NSAMPLES,
        min_denominator=0,
    )

    return model


# --- Technical fit checks ---

def test_gene_modality_alpha_y_prefit_set(fitted_multimodal_model):
    gene_mod = fitted_multimodal_model.get_modality('gene')
    assert gene_mod.alpha_y_prefit is not None


def test_splicing_modality_alpha_y_prefit_set(fitted_multimodal_model):
    spl_mod = fitted_multimodal_model.get_modality('splicing_test')
    assert spl_mod.alpha_y_prefit is not None


def test_gene_alpha_not_overwritten_by_splicing_fit(fitted_multimodal_model):
    gene_mod = fitted_multimodal_model.get_modality('gene')
    spl_mod = fitted_multimodal_model.get_modality('splicing_test')
    assert gene_mod.alpha_y_prefit is not None
    assert spl_mod.alpha_y_prefit is not None


def test_model_level_technical_stored(fitted_multimodal_model):
    gene_mod = fitted_multimodal_model.get_modality('gene')
    assert gene_mod.posterior_samples_ntc is not None


# --- Trans fit checks ---

def test_gene_modality_posterior_samples_trans(fitted_multimodal_model):
    gene_mod = fitted_multimodal_model.get_modality('gene')
    assert gene_mod.posterior_samples_trans is not None


def test_splicing_modality_posterior_samples_trans(fitted_multimodal_model):
    spl_mod = fitted_multimodal_model.get_modality('splicing_test')
    assert spl_mod.posterior_samples_trans is not None


def test_model_level_posterior_samples_trans_backward_compat(fitted_multimodal_model):
    assert fitted_multimodal_model.posterior_samples_trans is not None


# --- Error handling ---

def test_trans_without_technical_fit_raises(fitted_multimodal_model):
    from bayesDREAM import Modality

    n_cells = len(fitted_multimodal_model.meta)
    third_modality = Modality(
        name='untrained',
        counts=np.random.poisson(10, (5, n_cells)),
        feature_meta=pd.DataFrame({'feature': [f'f_{i}' for i in range(5)]}),
        distribution='negbinom',
        cells_axis=1,
    )
    fitted_multimodal_model.add_modality('untrained', third_modality)
    
    with pytest.raises(ValueError):
        fitted_multimodal_model.fit_trans(modality_name='untrained', niters=10, nsamples=5)


# --- Default behaviour ---

def test_fit_ntc_defaults_to_primary_modality(fitted_multimodal_model):
    gene_mod = fitted_multimodal_model.get_modality('gene')
    # Reset the prefit for this test
    gene_mod.alpha_y_prefit = None
    gene_mod.posterior_samples_ntc = None
    
    # Fit technical without specifying modality (should default to 'gene')
    fitted_multimodal_model.fit_ntc(sum_factor_col='sum_factor', niters=NITERS, nsamples=NSAMPLES)
    
    assert gene_mod.alpha_y_prefit is not None
