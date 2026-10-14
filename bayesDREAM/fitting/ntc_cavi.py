"""
Polya-Gamma coordinate-ascent variational inference (CAVI) for the negative binomial
NTC model.

Model, for feature t and NTC cell i in technical group c(i) (c = 0 is the reference):

    y_it ~ NB(mean = mu_t * a_{c(i),t} * s_i, total_count = phi_t)

Written with the NB logit link used elsewhere in the package,

    psi_it = log(mu_t a_{c(i),t} s_i) - log(phi_t) = x_i' theta_t + o_it,
    theta_t = (log mu_t, log a_1t, ..., log a_{C-1,t}),   o_it = log s_i - log phi_t,

and the likelihood p(y | psi) = C(y, phi) exp(y psi) / (1 + exp(psi))^(y + phi).

Priors (matching _model_ntc in spirit):

    log mu_t   ~ Normal(m0_t, s0_t^2)
    log a_ct   ~ StudentT(nu = 3, 0, scale), written as Normal(0, scale^2 / tau_ct),
                 tau_ct ~ Gamma(nu/2, nu/2)
    o_t = phi_t^(-1/2) ~ Exponential(beta_o),   beta_o ~ Gamma(9, 3)

The Polya-Gamma identity (Polson, Scott and Windle 2013)

    exp(y psi) / (1 + exp(psi))^(y + phi)
        = 2^-(y + phi) exp(kappa psi) E_{omega ~ PG(y + phi, 0)}[exp(-omega psi^2 / 2)],
    kappa = (y - phi) / 2,

makes theta conditionally Gaussian. The variational family is

    q(theta_t) = Normal(m_t, V_t),   q(omega_it) = PG(y_it + phi_t, c_it),
    q(tau_ct) = Gamma,

with closed-form CAVI updates. The dispersion phi_t has no conjugate prior here; it is
updated by maximising the expected log likelihood plus log prior (a variational EM step),
and a Laplace approximation on log phi_t gives its uncertainty. beta_o is updated to its
conditional mode. Features are independent given phi and beta_o, so every update is
vectorised over features.
"""

import math
import time

import numpy as np
import torch


def _gauss_hermite(n, device, dtype):
    x, w = np.polynomial.hermite_e.hermegauss(n)  # probabilists' Hermite: weight exp(-x^2/2)
    w = w / w.sum()
    return torch.as_tensor(x, device=device, dtype=dtype), torch.as_tensor(w, device=device, dtype=dtype)


def _exact_newton_terms(y, groups, Xg, G, log_s, m, log_phi, prior_diag, m0, beta_o, C):
    """
    Gradient and negative Hessian of the exact log posterior in z = (theta, log phi), per feature.

    log p = sum_i [lgamma(y + r) - lgamma(r) + y psi_i - (y + r) softplus(psi_i)] + priors,
    psi_i = x_i' theta + log s_i - log r, r = exp(log phi). The Student-t prior on the group
    effects enters through its current Gaussian scale-mixture precision (prior_diag).
    Returns (grad [T, C+1], neg_hessian [T, C+1, C+1]).
    """
    dtype, device = m.dtype, m.device
    T = m.shape[0]
    r = torch.exp(log_phi)                                  # [T]
    psi = (Xg[groups] @ m.T) + log_s - log_phi.unsqueeze(0)  # [N, T]
    sig = torch.sigmoid(psi)
    b = y + r.unsqueeze(0)
    w = b * sig * (1 - sig)                                 # -d2/dpsi2, [N, T]
    # d/dpsi of the per-cell term: y - b*sig; psi depends on log r with derivative -1,
    # and the lgamma / (y + r) terms depend on log r directly.
    d_softplus_r = torch.nn.functional.softplus(psi)        # d/dr of -(y+r) softplus = -softplus
    # Hessian blocks (negative), per feature
    W = G.T @ w                                             # [C, T]
    H_tt = torch.einsum('cd,ct,ce->tde', Xg, W, Xg)          # theta-theta
    # theta - log phi: d2f/(dtheta dlogr) = x_i (w_i - r sig_i), since d g_psi / d log r
    # = -r sig (b grows by r) + w (psi shifts by -1); the negative Hessian takes the opposite sign.
    cross = r.unsqueeze(0) * sig - w
    H_tl = torch.einsum('cd,ct->td', Xg, G.T @ cross)        # [T, C]
    # log phi - log phi: second derivative of the per-cell term in log r
    #   f(l) = lgamma(y+r) - lgamma(r) - (y+r) softplus(psi) + y psi, psi = ... - l
    #   df/dl = r[digamma(y+r) - digamma(r) - softplus(psi)] - (y - b sig)
    #   d2f/dl2 = r[digamma(y+r) - digamma(r) - softplus(psi)]
    #            + r^2[trigamma(y+r) - trigamma(r)] + r sig + (r sig - w)
    rr = r.unsqueeze(0)
    d2 = (rr * (torch.digamma(y + rr) - torch.digamma(rr) - d_softplus_r)
          + rr ** 2 * (torch.polygamma(1, y + rr) - torch.polygamma(1, rr))
          + 2 * rr * sig - w)
    H_ll = -d2.sum(0)                                       # [T]
    # prior on log phi: log p(l) = const - beta_o exp(-l/2) - l/2; d2 = -beta_o exp(-l/2) / 4
    H_ll = H_ll + beta_o * torch.exp(-0.5 * log_phi) / 4.0
    H = torch.zeros(T, C + 1, C + 1, device=device, dtype=dtype)
    H[:, :C, :C] = H_tt + torch.diag_embed(prior_diag)
    H[:, :C, C] = H_tl
    H[:, C, :C] = H_tl
    H[:, C, C] = H_ll
    # Guard against a non-positive-definite Hessian far from the mode
    eye = torch.eye(C + 1, device=device, dtype=dtype)
    H = H + 1e-8 * eye

    grad = torch.zeros(T, C + 1, device=device, dtype=dtype)
    grad[:, :C] = (Xg.T @ (G.T @ (y - b * sig))).T - prior_diag * (m - m0)
    grad[:, C] = (rr * (torch.digamma(y + rr) - torch.digamma(rr) - d_softplus_r) - (y - b * sig)).sum(0) \
        + 0.5 * beta_o * torch.exp(-0.5 * log_phi) - 0.5
    return grad, H


def fit_ntc_negbinom_cavi(
    y,
    groups,
    sum_factor,
    n_groups=None,
    log_mu_prior_scale=2.0,
    alpha_prior_df=3.0,
    alpha_prior_scale_log2=20.0,
    beta_o_alpha=9.0,
    beta_o_beta=3.0,
    max_iter=1000,
    tol=1e-5,
    phi_every=5,
    phi_newton_steps=2,
    n_quadrature=10,
    newton_polish_steps=5,
    device=None,
    dtype=torch.float64,
    verbose=True,
):
    """
    Fit the negative binomial NTC model by Polya-Gamma CAVI.

    Parameters
    ----------
    y : array-like, shape [N, T]
        Counts for NTC cells.
    groups : array-like of int, shape [N]
        Technical group code per cell, 0 .. C-1; group 0 is the reference.
    sum_factor : array-like, shape [N]
        Size factors.
    n_groups : int, optional
        Number of groups C (default: max(groups) + 1).
    log_mu_prior_scale : float
        SD of the Normal prior on log mu_t, centred on log of the normalised mean.
    alpha_prior_df, alpha_prior_scale_log2 : float
        Student-t prior on log2 a_ct (same defaults as _model_ntc: df 3, scale 20).
    beta_o_alpha, beta_o_beta : float
        Gamma hyperprior on beta_o (same defaults as fit_ntc).
    max_iter, tol : int, float
        Stop when the largest change in m (natural-log units) falls below tol.
    phi_every, phi_newton_steps, n_quadrature : int
        Update log phi every phi_every iterations with this many Newton steps, using
        Gauss-Hermite quadrature with n_quadrature nodes for E_q[log(1 + exp(psi))].
    newton_polish_steps : int
        Newton steps on the exact log posterior from the CAVI solution before the Laplace
        covariance is computed (0 keeps the CAVI means).

    Returns
    -------
    dict with numpy arrays:
        log_mu_mean, log_mu_sd               [T]      natural log of mu_t
        log2_alpha_mean, log2_alpha_sd       [C-1, T] log2 a_ct for c >= 1
        theta_mean [T, C], theta_cov [T, C, C]        natural-log scale
        log_phi_mean, log_phi_sd             [T]      Laplace approximation on log phi_t
        phi, o_y                             [T]      point values at the mode
        beta_o, n_iter, converged, time_s
    """
    t0 = time.time()
    if device is None:
        device = torch.device("cpu")
    y = torch.as_tensor(np.asarray(y), device=device, dtype=dtype)
    groups = torch.as_tensor(np.asarray(groups), device=device, dtype=torch.long)
    s = torch.as_tensor(np.asarray(sum_factor), device=device, dtype=dtype)
    N, T = y.shape
    C = int(groups.max().item()) + 1 if n_groups is None else int(n_groups)

    # Design: intercept + one indicator per non-reference group, X [N, C]
    X = torch.zeros(N, C, device=device, dtype=dtype)
    X[:, 0] = 1.0
    if C > 1:
        X[torch.arange(N, device=device), groups] = 1.0  # column 0 doubles as group 0
    # Design row of each group, Xg [C, C]: a cell's row depends only on its group,
    # so every per-cell quadratic form is computed per group and then indexed.
    Xg = torch.eye(C, device=device, dtype=dtype)
    Xg[:, 0] = 1.0
    G = torch.nn.functional.one_hot(groups, C).to(dtype)  # [N, C]

    log_s = torch.log(s).unsqueeze(-1)  # [N, 1]

    # ---------------- priors ----------------
    y_norm_mean = (y / s.unsqueeze(-1)).mean(0).clamp_min(1e-3)
    m0 = torch.zeros(T, C, device=device, dtype=dtype)
    m0[:, 0] = torch.log(y_norm_mean)
    s_alpha = alpha_prior_scale_log2 * math.log(2.0)  # natural-log scale of the t prior
    nu = alpha_prior_df
    prior_prec_mu = 1.0 / log_mu_prior_scale ** 2

    # ---------------- initial values ----------------
    m = m0.clone()
    V = torch.diag_embed(torch.full((T, C), 0.01, device=device, dtype=dtype))
    E_tau = torch.ones(T, max(C - 1, 1), device=device, dtype=dtype)
    ybar = y.mean(0)
    var = y.var(0)
    log_phi = torch.log((ybar ** 2 / (var - ybar).clamp_min(1e-3 * ybar.clamp_min(1e-3))).clamp(0.05, 1e3))
    beta_o = torch.tensor(beta_o_alpha / beta_o_beta, device=device, dtype=dtype)
    gh_x, gh_w = _gauss_hermite(n_quadrature, device, dtype)

    def psi_var():
        # x_c' V_t x_c per group, then per cell: [N, T]
        v = torch.einsum('cd,tde,ce->ct', Xg, V, Xg).clamp_min(0.0)
        return v[groups]

    def expected_loglik(log_phi_, mean_wo_phi, var_):
        """E_q[log NB(y | psi, phi)] summed over cells, as a function of log phi [T]."""
        phi = torch.exp(log_phi_)
        psi_mean = mean_wo_phi - log_phi_.unsqueeze(0)
        sd = torch.sqrt(var_)
        nodes = psi_mean.unsqueeze(-1) + sd.unsqueeze(-1) * gh_x  # [N, T, Q]
        e_softplus = (torch.nn.functional.softplus(nodes) * gh_w).sum(-1)
        ll = (torch.lgamma(y + phi) - torch.lgamma(phi) - torch.lgamma(y + 1.0)
              + y * psi_mean - (y + phi) * e_softplus)
        return ll.sum(0)  # [T]

    def log_prior_logphi(log_phi_, beta_o_):
        o = torch.exp(-0.5 * log_phi_)
        return torch.log(beta_o_) - beta_o_ * o + torch.log(o) - math.log(2.0)

    converged = False
    for it in range(1, max_iter + 1):
        m_old = m.clone()
        phi = torch.exp(log_phi)

        # ---- q(omega): E[omega] = (y + phi) / (2c) tanh(c / 2) ----
        mean_wo_phi = X @ m.T + log_s                                  # [N, T]
        var_now = psi_var()
        psi_mean = mean_wo_phi - log_phi.unsqueeze(0)
        c = torch.sqrt(psi_mean ** 2 + var_now)
        b = y + phi.unsqueeze(0)
        small = c < 1e-4
        E_omega = torch.where(small, b / 4.0, b / (2.0 * c.clamp_min(1e-12)) * torch.tanh(c / 2.0))

        # ---- q(theta): Gaussian, batched over features ----
        kappa = (y - phi.unsqueeze(0)) / 2.0
        offset = log_s - log_phi.unsqueeze(0)                          # [N, T]
        W = G.T @ E_omega                                               # [C, T]
        P = torch.einsum('cd,ct,ce->tde', Xg, W, Xg)                     # [T, C, C]
        prior_diag = torch.empty(T, C, device=device, dtype=dtype)
        prior_diag[:, 0] = prior_prec_mu
        if C > 1:
            prior_diag[:, 1:] = E_tau / s_alpha ** 2
        P = P + torch.diag_embed(prior_diag)
        rhs = X.T @ (kappa - E_omega * offset)                          # [C, T]
        rhs = rhs.T + prior_diag * m0                                   # [T, C]
        L = torch.linalg.cholesky(P)
        m = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        V = torch.cholesky_inverse(L)

        # ---- q(tau): Student-t scale mixture ----
        if C > 1:
            E_beta2 = m[:, 1:] ** 2 + torch.diagonal(V, dim1=-2, dim2=-1)[:, 1:]
            E_tau = (nu + 1.0) / (nu + E_beta2 / s_alpha ** 2)

        # ---- phi: Newton steps on log phi, separable across features ----
        mean_wo_phi = X @ m.T + log_s
        var_now = psi_var()
        for _ in range(phi_newton_steps if (it % phi_every == 1 or phi_every == 1) else 0):
            lp = log_phi.detach().clone().requires_grad_(True)
            f = expected_loglik(lp, mean_wo_phi, var_now) + log_prior_logphi(lp, beta_o)
            g, = torch.autograd.grad(f.sum(), lp, create_graph=True)
            h, = torch.autograd.grad(g.sum(), lp)
            g = g.detach()
            step = torch.where(h < 0, -g / h, 0.1 * torch.sign(g))
            log_phi = (log_phi + step.clamp(-1.0, 1.0)).clamp(-8.0, 12.0).detach()

        # ---- beta_o: conditional mode under Gamma(beta_o_alpha, beta_o_beta) ----
        o = torch.exp(-0.5 * log_phi)
        beta_o = (beta_o_alpha - 1.0 + T) / (beta_o_beta + o.sum())

        delta = (m - m_old).abs().max().item()
        if verbose and (it == 1 or it % 25 == 0):
            print(f"[CAVI] iter {it}: max |delta m| = {delta:.2e}")
        if delta < tol:
            converged = True
            break

    # ---- Laplace correction of the covariance ----
    # The PG bound at convergence is the Jaakkola-Jordan tangent bound, whose curvature
    # (y + phi) tanh(psi/2) / (2 psi) exceeds the true (y + phi) sigma(psi)(1 - sigma(psi))
    # away from psi = 0. The CAVI means are good, but V is too narrow. Replace it by the
    # inverse negative Hessian of the exact log posterior in (theta, log phi) at the mode.
    V_cavi = V
    m_cavi = m.clone()
    # A few Newton steps from the CAVI solution to the exact posterior mode, then Laplace there.
    for _ in range(newton_polish_steps):
        grad, H = _exact_newton_terms(y, groups, Xg, G, log_s, m, log_phi, prior_diag, m0, beta_o, C)
        step = torch.linalg.solve(H, grad.unsqueeze(-1)).squeeze(-1)
        step = step * (1.0 / step.abs().amax(-1, keepdim=True).clamp_min(1.0))  # cap at 1 per feature
        m = m + step[:, :C]
        log_phi = (log_phi + step[:, C]).clamp(-8.0, 12.0)
    grad, H = _exact_newton_terms(y, groups, Xg, G, log_s, m, log_phi, prior_diag, m0, beta_o, C)
    try:
        cov_full = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        cov_full = torch.linalg.inv(H)
    V = cov_full[:, :C, :C]
    log_phi_sd = torch.sqrt(cov_full[:, C, C].clamp_min(1e-12))

    ln2 = math.log(2.0)
    sd = torch.sqrt(torch.diagonal(V, dim1=-2, dim2=-1))
    out = {
        "theta_mean": m.cpu().numpy(),
        "theta_cov": V.cpu().numpy(),
        "theta_mean_cavi": m_cavi.cpu().numpy(),
        "theta_cov_cavi": V_cavi.cpu().numpy(),
        "max_abs_grad": float(grad.abs().max()),
        "log_mu_mean": m[:, 0].cpu().numpy(),
        "log_mu_sd": sd[:, 0].cpu().numpy(),
        "log2_alpha_mean": (m[:, 1:].T / ln2).cpu().numpy(),
        "log2_alpha_sd": (sd[:, 1:].T / ln2).cpu().numpy(),
        "log_phi_mean": log_phi.cpu().numpy(),
        "log_phi_sd": log_phi_sd.detach().cpu().numpy(),
        "phi": torch.exp(log_phi).cpu().numpy(),
        "o_y": torch.exp(-0.5 * log_phi).cpu().numpy(),
        "beta_o": float(beta_o),
        "n_iter": it,
        "converged": converged,
        "time_s": time.time() - t0,
    }
    if verbose:
        print(f"[CAVI] {'converged' if converged else 'stopped'} after {it} iterations, {out['time_s']:.1f} s")
    return out
