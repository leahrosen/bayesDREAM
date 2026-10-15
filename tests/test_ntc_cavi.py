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


def test_fit_ntc_method_cavi_end_to_end(tmp_path):
    """fit_ntc(method='cavi') stores SVI-shaped results that fit_cis and fit_trans can use."""
    import pandas as pd
    from bayesDREAM import bayesDREAM

    rng = np.random.default_rng(3)
    n_cells, n_genes = 120, 12
    target = np.where(np.arange(n_cells) < 60, "ntc", "GFI1B")
    meta = pd.DataFrame({
        "cell": [f"c{i}" for i in range(n_cells)],
        "guide": np.where(target == "ntc", "ntc_1", np.where(np.arange(n_cells) % 2, "g1", "g2")),
        "target": target,
        "cell_line": np.tile(["A", "B"], n_cells // 2),
        "sum_factor": rng.uniform(0.8, 1.2, n_cells),
    })
    genes = ["GFI1B"] + [f"gene_{i}" for i in range(n_genes - 1)]
    counts = pd.DataFrame(rng.poisson(30, (n_genes, n_cells)), index=genes, columns=meta["cell"])
    model = bayesDREAM(meta=meta, counts=counts, cis_gene="GFI1B", output_dir=str(tmp_path),
                       label="cavi_e2e", device="cpu")
    model.set_technical_groups(["cell_line"])
    model.fit_ntc(sum_factor_col="sum_factor", nsamples=50, method="cavi")

    gene_mod = model.get_modality("gene")
    ps = gene_mod.posterior_samples_ntc
    T = gene_mod.dims["n_features"]
    assert ps["log2_alpha_y"].shape == (50, 1, T)
    assert ps["mu_ntc"].shape == (50, 1, T)
    assert ps["o_y"].shape == (50, 1, T)
    assert gene_mod.alpha_y_prefit_mult.shape == (2, T)
    assert model.alpha_x_prefit is not None
    assert torch.isfinite(gene_mod.alpha_y_prefit_mult).all()

    model.fit_cis(sum_factor_col="sum_factor", niters=50, nsamples=10)
    model.fit_trans(sum_factor_col="sum_factor", function_type="single_hill", niters=20, nsamples=10)
    assert gene_mod.posterior_samples_trans is not None


def test_fit_ntc_method_cavi_rejects_other_distributions(tmp_path):
    import pandas as pd
    from bayesDREAM import bayesDREAM

    meta = pd.DataFrame({"cell": ["a", "b", "c", "d"], "guide": ["n", "n", "g", "g"],
                         "target": ["ntc", "ntc", "X", "X"], "cell_line": ["A"] * 4,
                         "sum_factor": [1.0] * 4})
    counts = pd.DataFrame([[1, 2, 3, 4], [5, 1, 2, 7]], index=["X", "Y"], columns=meta["cell"])
    model = bayesDREAM(meta=meta, counts=counts, cis_gene="X", output_dir=str(tmp_path),
                       label="cavi_reject", device="cpu")
    model.set_technical_groups(["cell_line"])
    with pytest.raises((NotImplementedError, ValueError)):
        model.fit_ntc(distribution="normal", method="cavi")
    with pytest.raises(ValueError):
        model.fit_ntc(method="nuts")
