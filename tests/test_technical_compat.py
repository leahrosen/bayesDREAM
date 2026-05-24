"""Test backward compatibility of fit_technical with negbinom distribution."""

import numpy as np

import pytest
pytestmark = pytest.mark.slow

NITERS = 20
NSAMPLES = 10


@pytest.fixture(scope='module')
def technical_compat_model(shared_test_data):
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM

    meta = shared_test_data['meta'].copy()
    counts = shared_test_data['gene_counts'].copy()
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='technical_compat_test',
    )
    model.set_technical_groups(['cell_line'])
    model.fit_technical(
        sum_factor_col='sum_factor',
        distribution='negbinom',
        niters=NITERS,
        nsamples=NSAMPLES,
        lr=1e-2,
    )
    return model


def test_fit_technical_negbinom_runs(technical_compat_model):
    gene_modality = technical_compat_model.get_modality('gene')
    assert gene_modality.alpha_y_prefit is not None


def test_alpha_y_prefit_set_in_modality(technical_compat_model):
    gene_modality = technical_compat_model.get_modality('gene')
    assert hasattr(gene_modality, 'alpha_y_prefit')
    assert gene_modality.alpha_y_prefit is not None


def test_alpha_y_prefit_correct_shape(technical_compat_model):
    gene_modality = technical_compat_model.get_modality('gene')
    # Shape should be (n_samples, n_groups, n_genes)
    assert len(gene_modality.alpha_y_prefit.shape) == 3
