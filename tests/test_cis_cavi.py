"""Tests for the Polya-Gamma CAVI prototype of the single-guide negative binomial cis model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bayesDREAM.fitting.cis_cavi import _exact_terms, _gaussian_block, fit_cis_negbinom_cavi

D = torch.float64


def _small_problem(seed=1, N=40, G=5, K=2):
    rng = np.random.default_rng(seed)
    guides = torch.as_tensor(np.sort(rng.integers(0, G, N)))
    tgt = torch.as_tensor(np.arange(G) % K)
    y = torch.as_tensor(rng.negative_binomial(4, 0.4, N), dtype=D)
    o = torch.as_tensor(rng.normal(-1.0, 0.3, N), dtype=D)
    tau = torch.as_tensor(rng.uniform(0.5, 4.0, G), dtype=D)
    p = torch.as_tensor(rng.uniform(0.2, 2.0, G), dtype=D)
    p0 = torch.as_tensor(rng.uniform(0.1, 1.0, K), dtype=D)
    m0 = torch.as_tensor(rng.normal(0, 1, K), dtype=D)
    ell = torch.as_tensor(rng.normal(1, 0.5, N), dtype=D)
    theta = torch.as_tensor(rng.normal(1, 0.5, G), dtype=D)
    mu = torch.as_tensor(rng.normal(1, 0.5, K), dtype=D)
    return dict(guides=guides, tgt=tgt, y=y, o=o, tau=tau, p=p, p0=p0, m0=m0,
                ell=ell, theta=theta, mu=mu, N=N, G=G, K=K, phi=3.0)


def _F(z, P):
    N, G = P['N'], P['G']
    return _exact_terms(z[:N], z[N:N + G], z[N + G:], P['y'], P['o'], P['phi'], P['guides'],
                        P['tgt'], P['tau'], P['p'], P['p0'], P['m0'])[0]


def test_exact_gradient_and_curvature_match_autograd():
    P = _small_problem()
    N, G = P['N'], P['G']
    F, g_ell, g_th, g_mu, h = _exact_terms(P['ell'], P['theta'], P['mu'], P['y'], P['o'], P['phi'],
                                           P['guides'], P['tgt'], P['tau'], P['p'], P['p0'], P['m0'])
    z = torch.cat([P['ell'], P['theta'], P['mu']])
    g_ad = torch.autograd.functional.jacobian(lambda z: _F(z, P), z)
    assert torch.allclose(torch.cat([g_ell, g_th, g_mu]), g_ad, atol=1e-10)
    # h is minus the second derivative of the likelihood part only
    H_ad = torch.autograd.functional.hessian(lambda z: _F(z, P), z)
    prior_curv = -torch.diagonal(H_ad)[:N] - P['tau'][P['guides']]
    assert torch.allclose(h, prior_curv, atol=1e-10)


def test_gaussian_block_matches_dense_inverse():
    P = _small_problem(seed=2)
    N, G, K = P['N'], P['G'], P['K']
    rng = np.random.default_rng(3)
    w = torch.as_tensor(rng.uniform(0.1, 5.0, N), dtype=D)
    r = torch.as_tensor(rng.normal(0, 2.0, N), dtype=D)
    # dense precision Q and linear term of the quadratic form in (ell, theta, mu)
    n = N + G + K
    Q = torch.zeros(n, n, dtype=D)
    lin = torch.zeros(n, dtype=D)
    for i in range(N):
        g = N + int(P['guides'][i])
        t = float(P['tau'][int(P['guides'][i])])
        Q[i, i] += w[i] + t
        Q[g, g] += t
        Q[i, g] -= t
        Q[g, i] -= t
        lin[i] = r[i]
    for g_ in range(G):
        g, k = N + g_, N + G + int(P['tgt'][g_])
        Q[g, g] += P['p'][g_]
        Q[k, k] += P['p'][g_]
        Q[g, k] -= P['p'][g_]
        Q[k, g] -= P['p'][g_]
    for k_ in range(K):
        Q[N + G + k_, N + G + k_] += P['p0'][k_]
        lin[N + G + k_] += P['p0'][k_] * P['m0'][k_]
    S = torch.linalg.inv(Q)
    mean = S @ lin
    m = _gaussian_block(w, r, P['guides'], P['tgt'], G, K, P['tau'], P['p'], P['p0'], P['m0'])
    assert torch.allclose(torch.cat([m['ell_m'], m['th_m'], m['mu_m']]), mean, atol=1e-10)
    var = torch.diagonal(S)
    assert torch.allclose(torch.cat([m['ell_v'], m['th_v'], m['mu_v']]), var, atol=1e-10)
    idx = torch.arange(N)
    assert torch.allclose(m['c_ell_th'], S[idx, N + P['guides']], atol=1e-10)
    gi = torch.arange(G)
    assert torch.allclose(m['c_th_mu'], S[N + gi, N + G + P['tgt']], atol=1e-10)


def test_newton_step_and_laplace_covariance_match_autograd_hessian():
    """The block solve with the exact curvature is the Newton step, and its variances are
    the diagonal of the inverse negative Hessian (the Laplace covariance)."""
    P = _small_problem(seed=4)
    N, G = P['N'], P['G']
    z = torch.cat([P['ell'], P['theta'], P['mu']])
    H = -torch.autograd.functional.hessian(lambda z: _F(z, P), z)
    g = torch.autograd.functional.jacobian(lambda z: _F(z, P), z)
    newton = z + torch.linalg.solve(H, g)
    _, _, _, _, h = _exact_terms(P['ell'], P['theta'], P['mu'], P['y'], P['o'], P['phi'],
                                 P['guides'], P['tgt'], P['tau'], P['p'], P['p0'], P['m0'])
    lik_grad = P['y'] - (P['y'] + P['phi']) * torch.sigmoid(P['ell'] + P['o'])
    m = _gaussian_block(h, lik_grad + h * P['ell'], P['guides'], P['tgt'], G, P['K'],
                        P['tau'], P['p'], P['p0'], P['m0'])
    assert torch.allclose(torch.cat([m['ell_m'], m['th_m'], m['mu_m']]), newton, atol=1e-9)
    assert torch.allclose(torch.cat([m['ell_v'], m['th_v'], m['mu_v']]),
                          torch.diagonal(torch.linalg.inv(H)), atol=1e-10)


def _simulate(seed=0, N=1500, G=12, n_ntc=4, phi=5.0):
    rng = np.random.default_rng(seed)
    l2eff = np.concatenate([np.full(n_ntc, 3.0), 3.0 + np.linspace(-0.5, -3.0, G - n_ntc)])
    seff = rng.uniform(0.2, 0.5, G)
    g = rng.integers(0, G, N)
    a = np.array([1.0, 1.4, 0.7])[rng.integers(0, 3, N)]
    s = np.exp(rng.normal(0, 0.3, N))
    l2x = rng.normal(l2eff[g], seff[g])
    m = a * 2.0 ** l2x * s
    y = rng.negative_binomial(phi, phi / (phi + m))
    return y, g, s, a, phi, l2eff, l2x


def test_cavi_recovers_guide_and_cell_expression():
    y, g, s, a, phi, l2eff, l2x = _simulate()
    r = fit_cis_negbinom_cavi(y, g, s, a, phi, verbose=False)
    assert r["converged"] and r["laplace_converged"]
    assert r["max_abs_grad"] < 1e-6
    assert r["log2_x_eff_mean"].shape == (12,) and r["log2_x_true_mean"].shape == y.shape
    err_g = r["log2_x_eff_mean"] - l2eff
    err_c = r["log2_x_true_mean"] - l2x
    assert np.sqrt(np.mean(err_g ** 2)) < 0.2
    assert np.sqrt(np.mean(err_c ** 2)) < 0.5
    # per-cell intervals roughly calibrated; the correction widens them
    cov_c = np.mean(np.abs(err_c) <= 1.6449 * r["log2_x_true_sd"])
    assert 0.83 < cov_c < 0.96
    assert np.mean(r["log2_x_true_sd"]) > np.mean(r["log2_x_true_sd_cavi"])
    assert np.allclose(r["x_true"], 2.0 ** r["log2_x_true_mean"])


def test_independent_targets_run():
    y, g, s, a, phi, l2eff, l2x = _simulate(seed=1, N=800)
    tpg = (np.arange(12) >= 4).astype(int)
    r = fit_cis_negbinom_cavi(y, g, s, a, phi, targets_per_guide=tpg, verbose=False)
    assert r["mu_log2_mean"].shape == (2,)
    assert r["mu_log2_mean"][0] > r["mu_log2_mean"][1]
    assert np.sqrt(np.mean((r["log2_x_eff_mean"] - l2eff) ** 2)) < 0.25


def test_fit_cis_method_cavi_end_to_end(tmp_path):
    """fit_cis(method='cavi') stores SVI-shaped results that fit_trans can use."""
    import pandas as pd
    from bayesDREAM import bayesDREAM

    rng = np.random.default_rng(3)
    n_cells, n_genes = 160, 10
    target = np.where(np.arange(n_cells) < 80, "ntc", "GFI1B")
    guide = np.where(target == "ntc", np.where(np.arange(n_cells) % 2, "n1", "n2"),
                     np.where(np.arange(n_cells) % 2, "g1", "g2"))
    meta = pd.DataFrame({"cell": [f"c{i}" for i in range(n_cells)], "guide": guide, "target": target,
                         "cell_line": np.tile(["A", "B"], n_cells // 2),
                         "sum_factor": rng.uniform(0.8, 1.2, n_cells)})
    cis = rng.poisson(np.where(target == "ntc", 40, 10))
    counts = pd.DataFrame(np.vstack([cis, rng.poisson(30, (n_genes - 1, n_cells))]),
                          index=["GFI1B"] + [f"gene_{i}" for i in range(n_genes - 1)], columns=meta["cell"])
    model = bayesDREAM(meta=meta, counts=counts, cis_gene="GFI1B", output_dir=str(tmp_path),
                       label="cis_cavi_e2e", device="cpu")
    model.set_technical_groups(["cell_line"])
    model.fit_ntc(sum_factor_col="sum_factor", nsamples=20, method="cavi")
    model.fit_cis(sum_factor_col="sum_factor", nsamples=30, method="cavi")

    ps = model.posterior_samples_cis
    G = model.meta["guide_code"].nunique()
    assert ps["x_true"].shape == (30, n_cells)
    assert ps["log_x_true"].shape == (30, n_cells)
    assert ps["x_eff_g"].shape == (30, G)
    assert torch.isfinite(model.x_true).all() and model.x_true.shape == (n_cells,)
    # knocked-down guides sit below the NTC guides
    codes = dict(model.meta[["guide", "guide_code"]].drop_duplicates().values)
    xg = ps["x_eff_g"].median(0).values
    assert max(xg[codes["g1"]], xg[codes["g2"]]) < min(xg[codes["n1"]], xg[codes["n2"]])

    model.fit_trans(sum_factor_col="sum_factor", function_type="logistic_hill", niters=50, nsamples=10)
    assert model.get_modality("gene").posterior_samples_trans is not None
