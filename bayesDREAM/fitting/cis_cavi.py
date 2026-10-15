"""
Polya-Gamma coordinate-ascent variational inference (CAVI) for the negative binomial cis
model, single-guide mode.

Model, for cell i with guide g(i), technical group c(i) and target k(g) of its guide
(k = 0 for every guide unless targets_per_guide is given):

    y_i ~ NB(mean = a_{c(i)} x_i s_i, total_count = phi)          a, phi fixed (from fit_ntc)
    ell_i = ln x_i ~ Normal(theta_g, v_g)                          per-cell expression
    theta_g ~ Normal(mu_k, sigma_k^2 / lambda_g),  lambda_g ~ Gamma(nu/2, nu/2)
                                                   (theta_g ~ StudentT(nu, mu_k, sigma_k))
    mu_k ~ Normal(m0, s0^2)
    sigma_k^2 | a_k ~ InvGamma(1/2, 1/a_k),  a_k ~ InvGamma(1/2, 1/A^2)
                                                   (sigma_k ~ HalfCauchy(A), Wand et al. 2011)
    v_g ~ InvGamma(a_v, b_v),  b_v ~ Gamma(c_b, d_b)

theta_g / ln 2 is log2 x_eff_g and ell_i / ln 2 is log2 x_true_i in CisFitter._model_x.
The SVI model puts a Gamma prior on the per-guide SD sigma_eff_g; here the variance
v_g = sigma_eff_g^2 has an inverse-gamma prior, which is conjugate.

With psi_i = ell_i + o_i, o_i = ln(a_{c(i)} s_i) - ln phi, the likelihood is
exp(y psi) / (1 + exp(psi))^(y + phi), and the Polya-Gamma identity gives a Gaussian
conditional in ell_i with precision omega_i and linear term kappa_i - omega_i o_i,
kappa_i = (y_i - phi) / 2.

Variational family:
    q(ell, theta, mu) jointly Gaussian (structured; not mean field)
    q(omega_i) = PG(y_i + phi, c_i)
    q(lambda_g), q(sigma_k^2), q(a_k), q(v_g), q(b_v) in their conjugate families.

The Gaussian block has an arrow structure (cells -> guide -> target), so its means,
marginal variances and the covariances needed by the other updates follow from eliminating
ell_i, then theta_g, in O(N + G). A mean-field split q(ell) q(theta) would make the
marginal variance of theta_g far too small: the cell-level measurement noise
1 / omega_i would not enter it.

Laplace correction: the PG curvature E[omega_i] = b tanh(c/2) / (2c) exceeds the exact
curvature b sigma(psi)(1 - sigma(psi)) away from psi = 0, so the CAVI variances are too
narrow. After CAVI, Newton steps on the exact log posterior in (ell, theta, mu), with the
hyperparameter factors held at their current expectations, find the mode; the same
elimination with the exact curvature gives the Laplace covariance. The hyperparameter
factors are then refreshed from the Laplace moments, and the two steps alternate until the
hyperparameters settle.

Both loops are fixed-point iterations that converge slowly when cell-level count noise is
large relative to v_g (much of the information about v_g is "missing"); they are
accelerated with SQUAREM (Varadhan and Roland 2008). Per-cell work is O(N) and vectorised.
"""

import math
import time

import numpy as np
import torch

LN2 = math.log(2.0)


def _gaussian_block(w, r, guides, tgt, G, K, tau, p, p0, m0):
    """
    Moments of the Gaussian with log density

        sum_i [-w_i ell_i^2 / 2 + r_i ell_i] - sum_i tau_g (ell_i - theta_g)^2 / 2
        - sum_g p_g (theta_g - mu_k)^2 / 2 - sum_k p0 (mu_k - m0)^2 / 2,

    computed by eliminating ell_i, then theta_g. All arguments are 1-D tensors
    (w, r over cells; tau, p, tgt over guides; p0, m0 over targets).

    Returns a dict with means and marginal variances of ell, theta, mu and the
    covariances Cov(ell_i, theta_g(i)) and Cov(theta_g, mu_k(g)).
    """
    dtype = w.dtype
    wt = tau[guides]                                     # [N]
    k = wt / (w + wt)                                    # [N]  d E[ell_i | theta] / d theta
    A = p + torch.zeros(G, dtype=dtype, device=w.device).index_add_(0, guides, w * k)
    B = torch.zeros(G, dtype=dtype, device=w.device).index_add_(0, guides, r * k)
    zK = torch.zeros(K, dtype=dtype, device=w.device)
    P_mu = p0 + zK.clone().index_add_(0, tgt, p - p ** 2 / A)
    L_mu = p0 * m0 + zK.clone().index_add_(0, tgt, p * B / A)
    mu_m = L_mu / P_mu
    mu_v = 1.0 / P_mu
    pa = p / A
    th_m = (B + p * mu_m[tgt]) / A
    th_v = 1.0 / A + pa ** 2 * mu_v[tgt]
    c_th_mu = pa * mu_v[tgt]
    ell_m = (r + wt * th_m[guides]) / (w + wt)
    ell_v = 1.0 / (w + wt) + k ** 2 * th_v[guides]
    c_ell_th = k * th_v[guides]
    return dict(ell_m=ell_m, ell_v=ell_v, c_ell_th=c_ell_th,
                th_m=th_m, th_v=th_v, c_th_mu=c_th_mu, mu_m=mu_m, mu_v=mu_v)


def _squarem(F, x, tol, max_iter):
    """
    SQUAREM-3 acceleration (Varadhan and Roland 2008) of the fixed-point map x -> F(x).

    Each iteration costs three evaluations of F. Falls back to two plain steps if the
    extrapolated point is not finite. Stops when max |F(x) - x| < tol.
    Returns (x, n_iter, converged).
    """
    for it in range(1, max_iter + 1):
        x1 = F(x)
        r = x1 - x
        if r.abs().max().item() < tol:
            return x1, it, True
        x2 = F(x1)
        v = x2 - 2.0 * x1 + x
        nv = v.norm().item()
        alpha = -r.norm().item() / nv if nv > 0 else -1.0
        alpha = min(alpha, -1.0)
        x_new = x - 2.0 * alpha * r + alpha ** 2 * v
        x_new = F(x_new) if torch.isfinite(x_new).all() else x2
        if not torch.isfinite(x_new).all():
            x_new = x2
        x = x_new
    return x, max_iter, False


def _exact_terms(ell, theta, mu, y, o, phi, guides, tgt, tau, p, p0, m0):
    """
    Exact log posterior F(ell, theta, mu) with the hyperparameters fixed (tau_g = E[1/v_g],
    p_g = E[lambda_g] E[1/sigma_k^2]), up to a constant, plus its gradient and the
    curvature of the likelihood term.

    F = sum_i [y_i psi_i - (y_i + phi) softplus(psi_i)] - sum_i tau_g (ell_i - theta_g)^2 / 2
        - sum_g p_g (theta_g - mu_k)^2 / 2 - sum_k p0 (mu_k - m0)^2 / 2,   psi_i = ell_i + o_i.

    Returns (F, grad_ell [N], grad_theta [G], grad_mu [K], h [N]) with
    h_i = (y_i + phi) sigma(psi_i)(1 - sigma(psi_i)) = -d2 loglik_i / d ell_i^2.
    """
    psi = ell + o
    sig = torch.sigmoid(psi)
    b = y + phi
    d_ell = ell - theta[guides]
    d_th = theta - mu[tgt]
    F = ((y * psi - b * torch.nn.functional.softplus(psi)).sum()
         - 0.5 * (tau[guides] * d_ell ** 2).sum()
         - 0.5 * (p * d_th ** 2).sum()
         - 0.5 * (p0 * (mu - m0) ** 2).sum())
    g_ell = y - b * sig - tau[guides] * d_ell
    g_th = torch.zeros_like(theta).index_add_(0, guides, tau[guides] * d_ell) - p * d_th
    g_mu = torch.zeros_like(mu).index_add_(0, tgt, p * d_th) - p0 * (mu - m0)
    h = b * sig * (1.0 - sig)
    return F, g_ell, g_th, g_mu, h


def fit_cis_negbinom_cavi(
    x_obs,
    guides,
    sum_factor,
    alpha_x_per_cell,
    phi_x,
    n_guides=None,
    targets_per_guide=None,
    mu_prior_mean_log2=None,
    mu_prior_sd_log2=None,
    sigma_prior_scale_log2=2.0,
    guide_df=3.0,
    v_shape=2.0,
    v_rate_prior_shape=1.0,
    sigma_eff_prior_log2=None,
    max_iter=2000,
    tol=1e-6,
    laplace=True,
    laplace_rounds=200,
    newton_steps=20,
    epsilon=1e-6,
    device=None,
    dtype=torch.float64,
    verbose=True,
):
    """
    Fit the single-guide negative binomial cis model by Polya-Gamma CAVI plus a Laplace
    correction.

    Parameters
    ----------
    x_obs : array-like [N]
        Cis-gene counts.
    guides : array-like of int [N]
        Guide code per cell, 0 .. G-1.
    sum_factor : array-like [N]
        Size factors.
    alpha_x_per_cell : array-like [N] or None
        Technical-group multiplier of the cis gene per cell (alpha_x_prefit[group]);
        None means 1.
    phi_x : float
        NB total count (1 / o_x^2 from fit_ntc).
    n_guides : int, optional
        G (default max(guides) + 1).
    targets_per_guide : array-like of int [G], optional
        Target code per guide; gives each target its own mu and sigma
        (as independent_mu_sigma=True). Default: one shared mu and sigma.
    mu_prior_mean_log2, mu_prior_sd_log2 : float, optional
        Normal prior on mu in log2 units. Defaults follow fit_cis: mean and SD of the
        per-guide log2 means of x / (alpha s).
    sigma_prior_scale_log2 : float
        HalfCauchy scale on sigma in log2 units (fit_cis uses 2).
    guide_df : float
        Student-t degrees of freedom of the guide effects (fit_cis uses 3).
    v_shape : float
        Shape a_v of the inverse-gamma prior on v_g.
    v_rate_prior_shape : float
        Shape c_b of the Gamma hyperprior on b_v; its mean is set so that the prior mean of
        v_g equals sigma_eff_prior_log2^2 (in log2 units, converted).
    sigma_eff_prior_log2 : float, optional
        Prior guess of the per-guide cell SD in log2 units; default: the mean of
        1.4826 * MAD of log2(x / (alpha s)) per guide, as in fit_cis, clamped to [0.05, 2].
    max_iter, tol : int, float
        Maximum SQUAREM iterations of the CAVI phase; both phases stop when the largest
        change of the fixed-point state (means in natural log, log variances, log
        hyperparameter expectations) is below tol.
    laplace : bool
        Apply the Newton + Laplace correction (default True).
    laplace_rounds, newton_steps : int
        Maximum SQUAREM iterations of the Laplace phase (each runs three rounds of Newton to
        the mode followed by a hyperparameter refresh), and Newton steps per round.

    Returns
    -------
    dict with numpy arrays (log2 scale unless stated):
        log2_x_true_mean, log2_x_true_sd   [N]   per-cell log2 x_i
        x_true                             [N]   2 ** log2_x_true_mean (posterior median)
        log2_x_eff_mean, log2_x_eff_sd     [G]   per-guide log2 x_eff_g
        x_eff_g                            [G]
        mu_log2_mean, mu_log2_sd           [K]
        sigma_log2                         [K]   sqrt(E[sigma^2]) under q
        sigma_eff_log2                     [G]   sqrt(E[v_g]) under q
        *_cavi                                   the same means / SDs before the correction
        log2_x_eff_sd_meanfield            [G]   diagnostic, see the code
        n_iter, converged, laplace_rounds_run, laplace_converged, max_abs_grad, time_s
    """
    t0 = time.time()
    if device is None:
        device = torch.device("cpu")
    t = lambda a: torch.as_tensor(np.array(a, dtype=float), device=device, dtype=dtype)
    y = t(x_obs).reshape(-1)
    guides = torch.as_tensor(np.asarray(guides), device=device, dtype=torch.long).reshape(-1)
    s = t(sum_factor).reshape(-1)
    a_cell = torch.ones_like(s) if alpha_x_per_cell is None else t(alpha_x_per_cell).reshape(-1)
    phi = float(phi_x)
    N = y.shape[0]
    G = int(guides.max().item()) + 1 if n_guides is None else int(n_guides)
    if targets_per_guide is None:
        tgt = torch.zeros(G, device=device, dtype=torch.long)
    else:
        tgt = torch.as_tensor(np.asarray(targets_per_guide), device=device, dtype=torch.long)
    K = int(tgt.max().item()) + 1

    o = torch.log(a_cell * s) - math.log(phi)
    b = y + phi
    kappa = (y - phi) / 2.0
    n_g = torch.zeros(G, dtype=dtype, device=device).index_add_(0, guides, torch.ones_like(y))
    G_k = torch.zeros(K, dtype=dtype, device=device).index_add_(0, tgt, torch.ones(G, dtype=dtype, device=device))

    # ---------------- data-driven priors (as in fit_cis) ----------------
    xf = (y / (a_cell * s)).cpu().numpy()
    g_np = guides.cpu().numpy()
    cnt = np.bincount(g_np, minlength=G).clip(min=1)
    gmean_log2 = np.log2(np.clip(np.bincount(g_np, weights=xf, minlength=G) / cnt, epsilon, None))
    if mu_prior_mean_log2 is None:
        mu_prior_mean_log2 = float(np.mean(gmean_log2))
    if mu_prior_sd_log2 is None:
        mu_prior_sd_log2 = max(float(np.std(gmean_log2, ddof=1)), 0.1)
    if sigma_eff_prior_log2 is None:
        lx = np.log2(np.clip(xf, epsilon, None))
        mads = [1.4826 * np.median(np.abs(lx[g_np == g] - np.median(lx[g_np == g])))
                for g in range(G) if np.any(g_np == g)]
        sigma_eff_prior_log2 = float(np.clip(np.mean(mads), 0.05, 2.0))
    m0 = torch.full((K,), mu_prior_mean_log2 * LN2, dtype=dtype, device=device)
    p0 = torch.full((K,), 1.0 / (mu_prior_sd_log2 * LN2) ** 2, dtype=dtype, device=device)
    A2 = (sigma_prior_scale_log2 * LN2) ** 2
    nu = float(guide_df)
    a_v = float(v_shape)
    c_b = float(v_rate_prior_shape)
    # E[v] = b_v / (a_v - 1) under the prior; choose E[b_v] = c_b / d_b to match the guess
    d_b = c_b / ((a_v - 1.0) * (sigma_eff_prior_log2 * LN2) ** 2)

    # ---------------- hyperparameter factors ----------------
    # The state of the hyperparameter factors is h = (tau_g = E[1/v_g], E[lambda_g],
    # E[1/sigma_k^2], E[1/a_k], E[b_v]); the Gaussian block depends on it only through
    # tau and p_g = E[lambda_g] E[1/sigma_k^2].
    def hyper_update(mom, h):
        tau, E_lam, E_inv_s2, E_inv_a, E_bv = h
        # E[(ell_i - theta_g)^2] and E[(theta_g - mu_k)^2] under the Gaussian block
        D_ell = ((mom['ell_m'] - mom['th_m'][guides]) ** 2 + mom['ell_v']
                 + mom['th_v'][guides] - 2.0 * mom['c_ell_th'])
        D_th = ((mom['th_m'] - mom['mu_m'][tgt]) ** 2 + mom['th_v']
                + mom['mu_v'][tgt] - 2.0 * mom['c_th_mu'])
        # q(lambda_g) = Gamma((nu + 1)/2, (nu + E[1/sigma^2] D_th)/2)
        E_lam = (nu + 1.0) / (nu + E_inv_s2[tgt] * D_th)
        # q(sigma_k^2) = IG(1/2 + G_k/2, E[1/a_k] + sum_g E[lambda_g] D_th / 2)
        rate_s2 = E_inv_a + 0.5 * torch.zeros(K, dtype=dtype, device=device).index_add_(0, tgt, E_lam * D_th)
        E_inv_s2 = shape_s2 / rate_s2
        # q(a_k) = IG(1, E[1/sigma^2] + 1/A^2)
        E_inv_a = 1.0 / (E_inv_s2 + 1.0 / A2)
        # q(v_g) = IG(a_v + n_g/2, E[b_v] + sum_i D_ell / 2)
        rate_v = E_bv + 0.5 * torch.zeros(G, dtype=dtype, device=device).index_add_(0, guides, D_ell)
        tau = shape_v / rate_v
        # q(b_v) = Gamma(c_b + G a_v, d_b + sum_g E[1/v_g])
        E_bv = ((c_b + G * a_v) / (d_b + tau.sum())).reshape(1)
        return (tau, E_lam, E_inv_s2, E_inv_a, E_bv), dict(rate_s2=rate_s2, rate_v=rate_v)

    shape_s2 = 0.5 + 0.5 * G_k
    shape_v = a_v + 0.5 * n_g
    sizes = [G, G, K, K, 1]

    def pack(h):
        return torch.log(torch.cat(h))

    def unpack(x):
        return tuple(torch.split(torch.exp(x), sizes))

    def prec(h):
        return h[1] * h[2][tgt]

    # ---------------- initial values ----------------
    th0 = torch.as_tensor(gmean_log2 * LN2, dtype=dtype, device=device)
    h = (torch.full((G,), 1.0 / (sigma_eff_prior_log2 * LN2) ** 2, dtype=dtype, device=device),
         torch.ones(G, dtype=dtype, device=device),
         torch.full((K,), 1.0 / (mu_prior_sd_log2 * LN2) ** 2, dtype=dtype, device=device),
         torch.full((K,), 1.0 / A2, dtype=dtype, device=device),
         torch.tensor([c_b / d_b], dtype=dtype, device=device))

    # ---------------- CAVI ----------------
    # One sweep: q(omega) from the current q(ell); the Gaussian block; the hyperparameter
    # factors. The sweep is a fixed-point map on (E[ell_i], log Var[ell_i], log h), which is
    # accelerated by SQUAREM (Varadhan and Roland 2008); the coupling between v_g and the
    # cell-level noise makes plain coordinate ascent converge slowly.
    def cavi_sweep(state):
        ell_m, ell_v = state[:N], torch.exp(state[N:2 * N])
        hh = unpack(state[2 * N:])
        c = torch.sqrt((ell_m + o) ** 2 + ell_v)
        E_omega = torch.where(c < 1e-4, b / 4.0, b / (2.0 * c.clamp_min(1e-12)) * torch.tanh(c / 2.0))
        mom = _gaussian_block(E_omega, kappa - E_omega * o, guides, tgt, G, K, hh[0], prec(hh), p0, m0)
        hh, extra = hyper_update(mom, hh)
        return torch.cat([mom['ell_m'], torch.log(mom['ell_v']), pack(hh)]), mom, hh, extra

    state0 = torch.cat([th0[guides], torch.zeros(N, dtype=dtype, device=device), pack(h)])
    state, n_iter, converged = _squarem(lambda z: cavi_sweep(z)[0], state0, tol, max_iter)
    _, mom, h, extra = cavi_sweep(state)
    mom_cavi = {k_: v_.clone() for k_, v_ in mom.items()}
    # For comparison only: the theta_g variance a mean-field q(ell) q(theta) would give at the
    # same hyperparameters, 1 / (n_g E[1/v_g] + p_g), which ignores cell-level count noise.
    th_v_meanfield = 1.0 / (n_g * h[0] + prec(h))
    if verbose:
        print(f"[cis CAVI] CAVI {'converged' if converged else 'stopped'} after {n_iter} SQUAREM iterations")

    # ---------------- Newton + Laplace correction ----------------
    rounds_run, lap_converged, max_abs_grad = 0, False, float("nan")
    if laplace:
        mode = [mom['ell_m'].clone(), mom['th_m'].clone(), mom['mu_m'].clone()]

        def find_mode(hh):
            ell, theta, mu = mode
            args = (y, o, phi, guides, tgt, hh[0], prec(hh), p0, m0)
            for _ in range(newton_steps):
                F, g_ell, g_th, g_mu, hc = _exact_terms(ell, theta, mu, *args)
                # Newton step: maximiser of the quadratic expansion of the likelihood
                lik_grad = y - b * torch.sigmoid(ell + o)
                sol = _gaussian_block(hc, lik_grad + hc * ell, guides, tgt, G, K, hh[0], prec(hh), p0, m0)
                d_ell, d_th, d_mu = sol['ell_m'] - ell, sol['th_m'] - theta, sol['mu_m'] - mu
                step = 1.0
                for _ls in range(30):
                    F_new = _exact_terms(ell + step * d_ell, theta + step * d_th, mu + step * d_mu, *args)[0]
                    if F_new >= F - 1e-12 * abs(F.item()):
                        break
                    step *= 0.5
                ell, theta, mu = ell + step * d_ell, theta + step * d_th, mu + step * d_mu
                if max(d_ell.abs().max().item(), d_th.abs().max().item(), d_mu.abs().max().item()) * step < 1e-10:
                    break
            mode[:] = [ell, theta, mu]
            F, g_ell, g_th, g_mu, hc = _exact_terms(ell, theta, mu, *args)
            gmax = max(g_ell.abs().max().item(), g_th.abs().max().item(), g_mu.abs().max().item())
            # Laplace moments: the same elimination with the exact curvature at the mode
            mom = _gaussian_block(hc, torch.zeros_like(hc), guides, tgt, G, K, hh[0], prec(hh), p0, m0)
            mom.update(ell_m=ell.clone(), th_m=theta.clone(), mu_m=mu.clone())
            return mom, gmax

        def laplace_round(x):
            mom, _ = find_mode(unpack(x))
            return pack(hyper_update(mom, unpack(x))[0])

        x, rounds_run, lap_converged = _squarem(laplace_round, pack(h), tol, laplace_rounds)
        h = unpack(x)
        mom, max_abs_grad = find_mode(h)
        _, extra = hyper_update(mom, h)
        if verbose:
            print(f"[cis CAVI] Laplace {'converged' if lap_converged else 'stopped'} after "
                  f"{rounds_run} SQUAREM iterations, max |grad| = {max_abs_grad:.1e}")

    # E[sigma^2] and E[v_g] under q, for reporting
    E_s2 = extra['rate_s2'] / (shape_s2 - 1.0)
    E_s2 = torch.where(shape_s2 > 1.0, E_s2, torch.full_like(E_s2, float('nan')))
    E_v = extra['rate_v'] / (shape_v - 1.0)
    it = n_iter

    to = lambda x: x.detach().cpu().numpy()
    l2m = mom['ell_m'] / LN2
    out = {
        "log2_x_true_mean": to(l2m),
        "log2_x_true_sd": to(mom['ell_v'].sqrt() / LN2),
        "x_true": to(2.0 ** l2m),
        "log2_x_eff_mean": to(mom['th_m'] / LN2),
        "log2_x_eff_sd": to(mom['th_v'].sqrt() / LN2),
        "x_eff_g": to(2.0 ** (mom['th_m'] / LN2)),
        "mu_log2_mean": to(mom['mu_m'] / LN2),
        "mu_log2_sd": to(mom['mu_v'].sqrt() / LN2),
        "sigma_log2": to(E_s2.sqrt() / LN2),
        "sigma_eff_log2": to(E_v.sqrt() / LN2),
        "log2_x_true_mean_cavi": to(mom_cavi['ell_m'] / LN2),
        "log2_x_true_sd_cavi": to(mom_cavi['ell_v'].sqrt() / LN2),
        "log2_x_eff_mean_cavi": to(mom_cavi['th_m'] / LN2),
        "log2_x_eff_sd_cavi": to(mom_cavi['th_v'].sqrt() / LN2),
        "log2_x_eff_sd_meanfield": to(th_v_meanfield.sqrt() / LN2),
        "n_iter": it,
        "converged": converged,
        "laplace_rounds_run": rounds_run,
        "laplace_converged": lap_converged,
        "max_abs_grad": max_abs_grad,
        "time_s": time.time() - t0,
    }
    if verbose:
        print(f"[cis CAVI] done in {out['time_s']:.2f} s")
    return out
