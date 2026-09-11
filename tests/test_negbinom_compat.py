"""Test backward compatibility with negbinom distribution for fit_trans."""

import numpy as np

import pytest
pytestmark = pytest.mark.slow

NITERS = 20
NSAMPLES = 10


@pytest.fixture(scope='module')
def negbinom_fitted_model(shared_test_data):
    import torch
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
        label='negbinom_compat_test',
    )
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=NITERS, nsamples=NSAMPLES)

    # Set dummy x_true for trans testing
    n_guides = len(model.meta)
    model.x_true = torch.ones(n_guides, dtype=torch.float32, device=model.device)
    model.log2_x_true = torch.log2(model.x_true)

    model.fit_trans(
        sum_factor_col='sum_factor',
        distribution='negbinom',
        function_type='single_hill',
        niters=NITERS,
        lr=1e-2,
        p0=0.01,
        gamma_threshold=0.01,
        nsamples=NSAMPLES,
    )
    return model


def test_fit_trans_negbinom_runs(negbinom_fitted_model):
    gene_modality = negbinom_fitted_model.get_modality('gene')
    assert hasattr(gene_modality, 'posterior_samples_trans')


def test_posterior_samples_created(negbinom_fitted_model):
    gene_modality = negbinom_fitted_model.get_modality('gene')
    assert hasattr(gene_modality, 'posterior_samples_trans')
    assert gene_modality.posterior_samples_trans is not None
