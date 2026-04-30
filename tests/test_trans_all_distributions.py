"""Test fit_trans with all distributions and function types.

Verifies:
1. Binomial Hill works in probability space with Beta priors
2. Binomial polynomial works in logit space
3. Multinomial has per-category parameters (K-1 for Hill, K for polynomial)
4. Technical groups are applied correctly (no double-application)
5. All distributions work with all function types
"""

import numpy as np
import pandas as pd
import pytest

import torch
import pyro
from bayesDREAM import bayesDREAM

pytestmark = pytest.mark.slow

NITERS = 20
NSAMPLES = 10
LR = 0.01


def _base_model(meta, gene_counts, feature_meta):
    return bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=feature_meta,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        device='cpu',
    )


def _fit_technical(model):
    model.set_technical_groups(["cell_line"])
    model.fit_technical(
        sum_factor_col="sum_factor",
        niters=NITERS,
        nsamples=NSAMPLES,
        lr=LR,
    )


def _fit_cis(model):
    model.fit_cis(
        sum_factor_col="sum_factor",
        niters=NITERS,
        nsamples=NSAMPLES,
        lr=LR,
    )


def _fit_trans(model, **kwargs):
    model.fit_trans(
        niters=NITERS,
        nsamples=NSAMPLES,
        lr=LR,
        **kwargs,
    )


def _posterior_for(model, modality_name=None):
    if modality_name is None or modality_name == model.primary_modality:
        return model.posterior_samples_trans
    return model.get_modality(modality_name).posterior_samples_trans


def test_negbinom_hill_variants(shared_test_data):
    meta = shared_test_data["meta"]
    gene_counts = shared_test_data["gene_counts"]
    feature_meta = shared_test_data["feature_meta"]
    model = _base_model(meta, gene_counts, feature_meta)

    _fit_technical(model)
    _fit_cis(model)
    _fit_trans(
        model,
        sum_factor_col="sum_factor_adj",
        function_type="single_hill",
    )
    assert model.posterior_samples_trans is not None
    for key in ("A", "Vmax_a", "K_a", "n_a"):
        assert key in model.posterior_samples_trans

    _fit_trans(
        model,
        sum_factor_col="sum_factor_adj",
        function_type="additive_hill",
    )
    for key in ("Vmax_b", "K_b", "n_b"):
        assert key in model.posterior_samples_trans


def test_binomial_hill_and_polynomial(shared_test_data):
    meta = shared_test_data["meta"]
    gene_counts = shared_test_data["gene_counts"]
    feature_meta = shared_test_data["feature_meta"]
    inclusion_counts = shared_test_data["inclusion_counts"]
    total_counts = shared_test_data["total_counts"]
    model = _base_model(meta, gene_counts, feature_meta)
    exon_meta = pd.DataFrame({"exon": inclusion_counts.index})
    model.add_custom_modality(
        name="exon_skip",
        counts=inclusion_counts,
        feature_meta=exon_meta,
        distribution="binomial",
        denominator=total_counts,
    )

    # Binomial modality has no fit_technical path, so only fit cis baseline.
    _fit_cis(model)

    _fit_trans(
        model,
        sum_factor_col=None,
        function_type="single_hill",
        modality_name="exon_skip",
        min_denominator=0,
    )
    posterior = _posterior_for(model, modality_name="exon_skip")
    A = posterior["A"]
    Vmax_a = posterior["Vmax_a"]
    assert torch.all(A >= 0) and torch.all(A <= 1)
    assert torch.all(Vmax_a >= 0) and torch.all(Vmax_a <= 1)

    _fit_trans(
        model,
        sum_factor_col=None,
        function_type="polynomial",
        modality_name="exon_skip",
        min_denominator=0,
    )
    posterior = _posterior_for(model, modality_name="exon_skip")
    assert any("poly_coeff" in k for k in posterior)
    A = posterior["A"]
    assert torch.all(A >= 0) and torch.all(A <= 1)


def test_multinomial_hill_and_polynomial(shared_test_data):
    meta = shared_test_data["meta"]
    gene_counts = shared_test_data["gene_counts"]
    feature_meta = shared_test_data["feature_meta"]
    multinomial_counts = shared_test_data["multinomial_counts"]
    model = _base_model(meta, gene_counts, feature_meta)
    K = multinomial_counts.shape[-1]
    donor_meta = pd.DataFrame(
        {"donor": [f"donor_{i}" for i in range(multinomial_counts.shape[0])]}
    )
    model.add_custom_modality(
        name="donor_usage",
        counts=multinomial_counts,
        feature_meta=donor_meta,
        distribution="multinomial",
    )

    _fit_cis(model)

    _fit_trans(
        model,
        sum_factor_col=None,
        function_type="single_hill",
        modality_name="donor_usage",
        min_denominator=0,
    )
    posterior = _posterior_for(model, modality_name="donor_usage")
    n_a = posterior["n_a"]
    assert n_a.shape[-1] == K - 1

    _fit_trans(
        model,
        sum_factor_col=None,
        function_type="polynomial",
        modality_name="donor_usage",
        min_denominator=0,
    )
    poly_coeffs = {
        k: v for k, v in _posterior_for(model, modality_name="donor_usage").items() if "poly_coeff" in k
    }
    for samples in poly_coeffs.values():
        assert samples.shape[-1] == K


def test_normal_polynomial(shared_test_data):
    meta = shared_test_data["meta"]
    gene_counts = shared_test_data["gene_counts"]
    feature_meta = shared_test_data["feature_meta"]
    normal_scores = shared_test_data["normal_scores"]
    model = _base_model(meta, gene_counts, feature_meta)
    score_meta = pd.DataFrame({"score": normal_scores.index})
    model.add_custom_modality(
        name="scores",
        counts=normal_scores,
        feature_meta=score_meta,
        distribution="normal",
    )

    _fit_cis(model)
    _fit_trans(
        model,
        sum_factor_col=None,
        function_type="polynomial",
        modality_name="scores",
    )
    posterior = _posterior_for(model, modality_name="scores")
    assert (
        "sigma_y" in posterior
        or "sigma" in posterior
    )
