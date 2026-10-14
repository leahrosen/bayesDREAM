"""Tests for the Polya-Gamma CAVI prototype of the negative binomial NTC model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bayesDREAM.fitting.ntc_cavi import _exact_newton_terms, fit_ntc_negbinom_cavi


def _simulate(seed=0, N=800, T=40, C=3):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, C, N)
    s = np.exp(rng.normal(0, 0.3, N))
    mu = np.exp(rng.uniform(np.log(0.5), np.log(50), T))
    phi = np.exp(rng.uniform(np.log(2), np.log(30), T))
    l2a = np.vstack([np.zeros(T), rng.normal(0, 0.5, (C - 1, T))])
    mean = mu[None] * 2.0 ** l2a[g] * s[:, None]
    y = rng.negative_binomial(phi[None], phi[None] / (phi[None] + mean))
    return y, g, s, mu, phi, l2a


def test_exact_gradient_and_hessian_match_autograd():
    d = torch.float64
    rng = np.random.default_rng(1)
    N, T, C = 200, 3, 3
    g = torch.as_tensor(rng.integers(0, C, N))
    s = torch.as_tensor(np.exp(rng.normal(0, 0.3, N)), dtype=d)
    y = torch.as_tensor(rng.negative_binomial(3, 0.4, (N, T)), dtype=d)
    Xg = torch.eye(C, dtype=d)
    Xg[:, 0] = 1.0
    G = torch.nn.functional.one_hot(g, C).to(d)
    m = torch.as_tensor(rng.normal(0, 0.3, (T, C)), dtype=d)
    m0 = torch.zeros_like(m)
    log_phi = torch.as_tensor(rng.normal(1, 0.5, T), dtype=d)
    prior = torch.as_tensor(rng.uniform(0.1, 1.1, (T, C)), dtype=d)
    beta_o = torch.tensor(3.0, dtype=d)
    grad, H = _exact_newton_terms(y, g, Xg, G, torch.log(s).unsqueeze(-1), m, log_phi, prior, m0, beta_o, C)

    for t in range(T):
        def logp(z):
            th, l = z[:C], z[C]
            r = torch.exp(l)
            psi = Xg[g] @ th + torch.log(s) - l
            ll = (torch.lgamma(y[:, t] + r) - torch.lgamma(r) + y[:, t] * psi
                  - (y[:, t] + r) * torch.nn.functional.softplus(psi)).sum()
            return ll - 0.5 * (prior[t] * th ** 2).sum() - beta_o * torch.exp(-l / 2) - l / 2

        z = torch.cat([m[t], log_phi[t:t + 1]])
        H_ad = -torch.autograd.functional.hessian(logp, z)
        g_ad = torch.autograd.functional.jacobian(logp, z)
        assert torch.allclose(H[t], H_ad, atol=1e-8)
        assert torch.allclose(grad[t], g_ad, atol=1e-8)


def test_cavi_recovers_parameters():
    y, g, s, mu, phi, l2a = _simulate()
    r = fit_ntc_negbinom_cavi(y, g, s, verbose=False)
    assert r["converged"]
    assert r["log2_alpha_mean"].shape == (2, y.shape[1])
    err_a = r["log2_alpha_mean"] - l2a[1:]
    err_mu = r["log_mu_mean"] - np.log(mu)
    assert np.sqrt(np.mean(err_a ** 2)) < 0.2
    assert np.sqrt(np.mean(err_mu ** 2)) < 0.15
    # intervals roughly calibrated (loose bound for a small simulation)
    z = np.abs(err_a) / r["log2_alpha_sd"]
    assert 0.75 < np.mean(z < 1.6449) <= 1.0
