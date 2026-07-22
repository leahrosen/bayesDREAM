"""
Post-fitting diagnostics for bayesDREAM.

This module provides statistical tests for detecting systematic expression shifts
between NTC (non-targeting control) and targeted cells after cis fitting. The test
controls for the estimated cis gene expression (x_true) using a GAM smooth, so that
only residual shifts -- not those explained by the dose-response relationship -- are
detected.

Supported distributions
-----------------------
- negbinom    : Negative-Binomial GAM with log offset (size factors).
                Dispersion phi = 1/o_y^2, o_y taken from posterior_samples_ntc
                (preferred) or posterior_samples_trans.
- normal      : Gaussian GAM with fixed scale sigma = sigma_y taken from
                posterior_samples_ntc (preferred) or posterior_samples_trans.
                Using the model-estimated sigma makes the LRT calibrated; without it
                the Gaussian GLM estimates scale from the window data.
- studentt    : Proper Student-t GAM with fixed sigma = sigma_y (from
                posterior_samples_ntc, preferred, or posterior_samples_trans) and
                fixed nu = posterior median of nu_y (per-feature degrees of freedom,
                from posterior_samples_trans -- nu_y is only fit there).
                Optimised via scipy.optimize (L-BFGS-B) over the Student-t log-likelihood
                with a B-spline basis. Falls back to Gaussian if either parameter is
                unavailable.
- binomial    : Binomial GAM (logit link); denominator array required.
- multinomial : Per-category Binomial GAM + Fisher p-value combination.

Theta / sigma extraction priority
----------------------------------
1. User-supplied ``theta`` argument.
2. ``modality.posterior_samples_ntc['o_y']`` (negbinom) or ``['sigma_y']``
   (normal/studentt) -- the pre-fit technical estimate.
3. ``modality.posterior_samples_trans`` (same keys), used as fallback.

NTC is preferred over trans because fit_trans() itself anchors its likelihood to
the NTC-derived o_y/sigma_y rather than its own resampled value, which is only
weakly identified per feature (see fitting/trans.py). Using the same anchor here
keeps this test's dispersion consistent with what fit_trans actually relied on.

Requires: statsmodels >= 0.14
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from scipy.stats import chi2, combine_pvalues, t as t_dist
from scipy.optimize import minimize

try:
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.genmod.families import NegativeBinomial, Gaussian, Binomial
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_log(arr: np.ndarray, minval: float = 1e-12) -> np.ndarray:
    """Logarithm clipped to avoid log(0)."""
    return np.log(np.clip(np.asarray(arr, dtype=float), minval, None))


def _require_statsmodels() -> None:
    if not _HAS_STATSMODELS:
        raise ImportError(
            "statsmodels >= 0.14 is required for check_systematic_shift. "
            "Install with:  pip install 'statsmodels>=0.14'"
        )


def _fit_shift_nb_gam(
    dt: pd.DataFrame,
    theta: float,
    df_spline: int = 6,
    degree: int = 3,
) -> dict:
    """
    LRT for a mean shift in a Negative Binomial GAM.

    Null : y ~ s(x) + offset(log(sum_factor))
    Alt  : y ~ s(x) + targeted + offset(log(sum_factor))

    Parameters
    ----------
    dt : DataFrame with columns y, x, targeted (0/1), offset (log sum_factor).
    theta : float
        NB dispersion (total_count = phi = 1/o_y^2 from technical fit).
        statsmodels NegativeBinomial alpha = 1 / theta.
    """
    alpha = 1.0 / float(theta)
    family = NegativeBinomial(alpha=alpha)

    x_smooth = dt[["x"]].to_numpy(dtype=float)
    bs = BSplines(x_smooth, df=[df_spline], degree=[degree])

    y = dt["y"].to_numpy(dtype=float)
    offset = dt["offset"].to_numpy(dtype=float)

    exog_null = np.ones((len(dt), 1), dtype=float)
    exog_alt = np.column_stack([
        np.ones(len(dt), dtype=float),
        dt["targeted"].to_numpy(dtype=float),
    ])

    fit0 = GLMGam(endog=y, exog=exog_null, smoother=bs, offset=offset, family=family).fit(disp=False)
    fit1 = GLMGam(endog=y, exog=exog_alt, smoother=bs, offset=offset, family=family).fit(disp=False)

    lr = 2.0 * (fit1.llf - fit0.llf)
    pval = chi2.sf(max(lr, 0.0), df=1)

    return {
        "lrt_stat": lr,
        "pval": pval,
        "shift_est": fit1.params[1],
        "shift_se": fit1.bse[1],
        "shift_p": fit1.pvalues[1],
        "shift_fc": np.exp(fit1.params[1]),
    }


def _fit_shift_gaussian_gam(
    dt: pd.DataFrame,
    sigma: Optional[float] = None,
    df_spline: int = 6,
    degree: int = 3,
) -> dict:
    """
    LRT for a mean shift in a Gaussian GAM.

    Null : y ~ s(x)
    Alt  : y ~ s(x) + targeted

    No offset is used for continuous measurements.

    Parameters
    ----------
    dt : DataFrame with columns y, x, targeted (0/1).
    sigma : float or None
        If provided, used as the fixed residual standard deviation (scale = sigma^2).
        This is o_y from the trans (or technical) model, since sigma_y = o_y for
        normal/studentt distributions (phi_y = 1/o_y^2, sigma_y = 1/sqrt(phi_y) = o_y).
        When None, the scale is estimated from the window data by each GLM fit.
    """
    family = Gaussian()

    x_smooth = dt[["x"]].to_numpy(dtype=float)
    bs = BSplines(x_smooth, df=[df_spline], degree=[degree])

    y = dt["y"].to_numpy(dtype=float)

    exog_null = np.ones((len(dt), 1), dtype=float)
    exog_alt = np.column_stack([
        np.ones(len(dt), dtype=float),
        dt["targeted"].to_numpy(dtype=float),
    ])

    # Pass fixed scale if sigma is available; otherwise let GLM estimate it.
    fit_kwargs = {"disp": False}
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        fit_kwargs["scale"] = float(sigma) ** 2

    fit0 = GLMGam(endog=y, exog=exog_null, smoother=bs, family=family).fit(**fit_kwargs)
    fit1 = GLMGam(endog=y, exog=exog_alt, smoother=bs, family=family).fit(**fit_kwargs)

    lr = 2.0 * (fit1.llf - fit0.llf)
    pval = chi2.sf(max(lr, 0.0), df=1)

    return {
        "lrt_stat": lr,
        "pval": pval,
        "shift_est": fit1.params[1],
        "shift_se": fit1.bse[1],
        "shift_p": fit1.pvalues[1],
        "shift_fc": float("nan"),  # Not meaningful for continuous outcomes
    }


def _fit_shift_studentt_gam(
    dt: pd.DataFrame,
    sigma: float,
    nu: float,
    df_spline: int = 6,
    degree: int = 3,
) -> dict:
    """
    LRT for a mean shift using a Student-t GAM with fixed sigma and nu.

    Null : y ~ s(x)         (Student-t likelihood, fixed sigma and nu)
    Alt  : y ~ s(x) + targeted

    Since statsmodels does not provide a Student-t GLM/GAM family, parameters
    are optimised directly via scipy.optimize.minimize over the Student-t
    log-likelihood with the B-spline basis from statsmodels.

    Parameters
    ----------
    dt : DataFrame with columns y, x, targeted (0/1).
    sigma : float
        Fixed residual scale (= o_y from the trans/technical posterior).
    nu : float
        Fixed degrees of freedom (= posterior mean of nu_y from trans model).
    """
    x_smooth = dt[["x"]].to_numpy(dtype=float)
    bs = BSplines(x_smooth, df=[df_spline], degree=[degree])
    # basis: [N, df_spline]  (statsmodels BSplines includes intercept by default)
    spline_basis = bs.basis  # [N, df_spline]

    y = dt["y"].to_numpy(dtype=float)
    targeted = dt["targeted"].to_numpy(dtype=float)

    def neg_loglik(params, X):
        """Negative Student-t log-likelihood with fixed sigma and nu."""
        mu = X @ params
        return -t_dist.logpdf(y, df=nu, loc=mu, scale=sigma).sum()

    # Null model design matrix: spline basis only
    X_null = spline_basis  # [N, df_spline]
    # Alt model design matrix: spline basis + targeted indicator
    X_alt = np.column_stack([spline_basis, targeted])  # [N, df_spline + 1]

    p_null = X_null.shape[1]
    p_alt = X_alt.shape[1]

    # Initialise at zero; the spline columns should absorb the mean quickly
    res0 = minimize(neg_loglik, x0=np.zeros(p_null), args=(X_null,), method="L-BFGS-B")
    res1 = minimize(neg_loglik, x0=np.zeros(p_alt), args=(X_alt,), method="L-BFGS-B")

    llf0 = -res0.fun
    llf1 = -res1.fun

    lr = 2.0 * (llf1 - llf0)
    pval = chi2.sf(max(lr, 0.0), df=1)

    # Approximate SE for the shift coefficient via inverse Hessian (if available)
    shift_est = float(res1.x[-1])
    shift_se = float("nan")
    shift_p = float("nan")
    if hasattr(res1, "hess_inv"):
        try:
            # L-BFGS-B returns an LbfgsInvHessProduct; convert to dense for the diagonal
            hess_inv_diag = res1.hess_inv.todense().diagonal()
            shift_se = float(np.sqrt(hess_inv_diag[-1]))
            if shift_se > 0:
                shift_p = float(2.0 * t_dist.sf(abs(shift_est / shift_se), df=nu))
        except Exception:
            pass

    return {
        "lrt_stat": lr,
        "pval": pval,
        "shift_est": shift_est,
        "shift_se": shift_se,
        "shift_p": shift_p if np.isfinite(shift_p) else pval,
        "shift_fc": float("nan"),  # Not meaningful for continuous outcomes
    }


def _fit_shift_binomial_gam(
    dt: pd.DataFrame,
    df_spline: int = 6,
    degree: int = 3,
) -> dict:
    """
    LRT for a proportion shift in a Binomial GAM (logit link).

    Requires columns: y (successes), denom (total trials), x, targeted.

    Null : y/n ~ s(x)
    Alt  : y/n ~ s(x) + targeted
    """
    family = Binomial()

    x_smooth = dt[["x"]].to_numpy(dtype=float)
    bs = BSplines(x_smooth, df=[df_spline], degree=[degree])

    # Endog as [successes, failures] array
    successes = dt["y"].to_numpy(dtype=float)
    total = dt["denom"].to_numpy(dtype=float)
    failures = total - successes
    endog_2d = np.column_stack([successes, failures])

    exog_null = np.ones((len(dt), 1), dtype=float)
    exog_alt = np.column_stack([
        np.ones(len(dt), dtype=float),
        dt["targeted"].to_numpy(dtype=float),
    ])

    fit0 = GLMGam(endog=endog_2d, exog=exog_null, smoother=bs, family=family).fit(disp=False)
    fit1 = GLMGam(endog=endog_2d, exog=exog_alt, smoother=bs, family=family).fit(disp=False)

    lr = 2.0 * (fit1.llf - fit0.llf)
    pval = chi2.sf(max(lr, 0.0), df=1)

    return {
        "lrt_stat": lr,
        "pval": pval,
        "shift_est": fit1.params[1],  # on logit scale
        "shift_se": fit1.bse[1],
        "shift_p": fit1.pvalues[1],
        "shift_fc": np.exp(fit1.params[1]),  # odds ratio
    }


def _fit_shift_multinomial_gam(
    dt: pd.DataFrame,
    n_categories: int,
    df_spline: int = 6,
    degree: int = 3,
) -> dict:
    """
    Test for a composition shift in multinomial data.

    For each of the first (n_categories - 1) categories, tests:
        y_k / total ~ s(x) + targeted   (Binomial GAM, logit link)

    P-values across categories are combined using Fisher's method.
    The reported shift_est is the mean log-odds shift across categories.

    Requires columns: y_k_* (per-category counts), x, targeted.
    Category columns are named 'y_0', 'y_1', ..., 'y_{n_categories-1}'.
    Total is in column 'y_total'.
    """
    family = Binomial()

    x_smooth = dt[["x"]].to_numpy(dtype=float)
    bs = BSplines(x_smooth, df=[df_spline], degree=[degree])

    exog_null = np.ones((len(dt), 1), dtype=float)
    exog_alt = np.column_stack([
        np.ones(len(dt), dtype=float),
        dt["targeted"].to_numpy(dtype=float),
    ])

    total = dt["y_total"].to_numpy(dtype=float)
    per_cat_results = []

    # Test each non-trivial category
    for k in range(n_categories - 1):  # exclude residual last category
        col = f"y_{k}"
        if col not in dt.columns:
            continue
        successes = dt[col].to_numpy(dtype=float)
        failures = total - successes
        # Skip category if it has zero variance
        with np.errstate(divide='ignore', invalid='ignore'):
            props = np.where(total > 0, successes / total, 0.0)
        if np.std(props) < 1e-10:
            continue

        endog_2d = np.column_stack([successes, failures])

        try:
            fit0 = GLMGam(
                endog=endog_2d, exog=exog_null, smoother=bs, family=family
            ).fit(disp=False)
            fit1 = GLMGam(
                endog=endog_2d, exog=exog_alt, smoother=bs, family=family
            ).fit(disp=False)
            lr_k = 2.0 * (fit1.llf - fit0.llf)
            p_k = chi2.sf(max(lr_k, 0.0), df=1)
            per_cat_results.append({
                "k": k,
                "pval": p_k,
                "shift_est": fit1.params[1],
                "lrt_stat": lr_k,
            })
        except Exception:
            pass

    if not per_cat_results:
        raise ValueError("No categories could be tested")

    cat_pvals = [r["pval"] for r in per_cat_results]
    # Fisher's combined test
    if len(cat_pvals) == 1:
        combined_pval = cat_pvals[0]
        combined_stat = 0.0
    else:
        combined_stat, combined_pval = combine_pvalues(cat_pvals, method="fisher")

    mean_shift = float(np.mean([r["shift_est"] for r in per_cat_results]))

    return {
        "lrt_stat": combined_stat,
        "pval": combined_pval,
        "shift_est": mean_shift,
        "shift_se": float("nan"),
        "shift_p": combined_pval,
        "shift_fc": np.exp(mean_shift),
        "n_categories_tested": len(cat_pvals),
    }


# ---------------------------------------------------------------------------
# Loss-matrix helpers  (module-level so they can be tested independently)
# ---------------------------------------------------------------------------

def _reconstruct_full_alpha_y(alpha_y_raw, C, distribution):
    """
    Ensure alpha_y has shape [C, T] (or [C, T, K]) by prepending the
    reference-group row if only C-1 groups are stored.

    negbinom reference = 1.0 (multiplicative); others = 0.0 (additive).
    """
    import torch
    if alpha_y_raw.shape[0] == C:
        return alpha_y_raw  # already includes reference

    if distribution == "negbinom":
        ref = torch.ones((1,) + alpha_y_raw.shape[1:],
                         dtype=alpha_y_raw.dtype, device=alpha_y_raw.device)
    else:
        ref = torch.zeros((1,) + alpha_y_raw.shape[1:],
                          dtype=alpha_y_raw.dtype, device=alpha_y_raw.device)
    return torch.cat([ref, alpha_y_raw], dim=0)


def _reconstruct_mu_y(
    x_subset,
    function_type,
    distribution,
    A, alpha, beta,
    Vmax_a, Vmax_b,
    log_K_a, log_K_b, K_b,
    n_a, n_b,
    poly_coeffs, polynomial_degree,
    Hill_based_positive_logK,
    Hill_based_positive,
    Polynomial_function,
    epsilon=1e-6,
):
    """
    Reconstruct the dose-response μ_y for a single gene across N_sub cells.

    All parameter tensors are scalars or 1-D (per-gene already extracted).
    Returns a tensor of shape [N_sub] (or [N_sub, K] for multinomial).
    """
    import torch
    eps = torch.tensor(epsilon, dtype=x_subset.dtype, device=x_subset.device)
    x = x_subset  # [N_sub]

    if function_type in ("single_hill", "additive_hill", "nested_hill"):
        if distribution == "multinomial":
            # A, alpha, Vmax_a, log_K_a, n_a are [K] or [K-1] tensors here
            # This path is complex; fall back to simplified per-category Hills
            K_minus_1 = A.shape[-1] if A.dim() > 0 else 1
            hills_a = Hill_based_positive_logK(
                x.unsqueeze(-1),        # [N, 1]
                Vmax=Vmax_a,            # [K-1]
                A=torch.zeros_like(Vmax_a),
                logK=log_K_a,           # [K-1]
                n=n_a,                  # [K-1]
            )  # [N, K-1]
            y_kminus1 = A + alpha * hills_a  # [N, K-1]
            if function_type == "additive_hill" and Vmax_b is not None:
                hills_b = Hill_based_positive_logK(
                    x.unsqueeze(-1), Vmax=Vmax_b, A=torch.zeros_like(Vmax_b),
                    logK=log_K_b, n=n_b,
                )
                y_kminus1 = y_kminus1 + beta * hills_b
            y_kminus1 = torch.clamp(y_kminus1, eps, 1.0 - eps)
            sum_k = y_kminus1.sum(dim=-1, keepdim=True)
            y_kminus1 = torch.where(sum_k > 1.0 - eps,
                                    y_kminus1 * (1.0 - eps) / sum_k, y_kminus1)
            y_K = (1.0 - y_kminus1.sum(dim=-1, keepdim=True)).clamp(eps, 1.0 - eps)
            return torch.cat([y_kminus1, y_K], dim=-1)  # [N, K]

        x_u = x.unsqueeze(-1)  # [N, 1] to broadcast with scalar params
        Hilla = Hill_based_positive_logK(x_u, Vmax=Vmax_a, A=torch.zeros(1, device=x.device, dtype=x.dtype), logK=log_K_a, n=n_a).squeeze(-1)  # [N]

        if function_type == "single_hill":
            mu = A + alpha * Hilla
        elif function_type == "additive_hill":
            Hillb = Hill_based_positive_logK(x_u, Vmax=Vmax_b, A=torch.zeros(1, device=x.device, dtype=x.dtype), logK=log_K_b, n=n_b).squeeze(-1)
            mu = A + alpha * Hilla + beta * Hillb
        else:  # nested_hill
            Hillb = Hill_based_positive(Hilla.unsqueeze(-1), Vmax=Vmax_b, A=torch.zeros(1, device=x.device, dtype=x.dtype), K=K_b, n=n_b, epsilon=epsilon).squeeze(-1)
            mu = A + alpha * Hillb

        if distribution == "binomial":
            mu = torch.clamp(mu, eps, 1.0 - eps)
        elif distribution == "negbinom":
            mu = torch.clamp(mu, eps, None)
        return mu  # [N]

    elif function_type == "polynomial":
        if polynomial_degree is None or poly_coeffs is None:
            raise ValueError("poly_coeffs not found in posterior_samples_trans.")

        if distribution in ("normal", "studentt"):
            poly_val = Polynomial_function(x, poly_coeffs.unsqueeze(-1)).squeeze(-1)
            return A + alpha * poly_val

        elif distribution == "negbinom":
            log2_x = torch.log2(x.clamp_min(eps))
            poly_val = Polynomial_function(log2_x, poly_coeffs.unsqueeze(-1)).squeeze(-1)
            log2_mu = torch.log2(A.clamp_min(eps)) + alpha * poly_val
            return (2.0 ** log2_mu).clamp_min(eps)

        elif distribution == "binomial":
            A_clamped = A.clamp(eps, 1.0 - eps)
            logit_A = torch.log(A_clamped) - torch.log(1.0 - A_clamped)
            poly_val = Polynomial_function(x, poly_coeffs.unsqueeze(-1)).squeeze(-1)
            return torch.sigmoid(logit_A + alpha * poly_val)

        else:
            raise ValueError(f"Polynomial not implemented for distribution '{distribution}'.")
    else:
        raise ValueError(f"Unknown function_type: '{function_type}'.")


def _apply_alpha_y(mu_y, alpha_y_g, groups_tensor, distribution):
    """
    Apply technical-group effects to μ_y.

    alpha_y_g : [C] (scalar per group) or [C, K] for multinomial, or None.
    groups_tensor : [N_sub] int64 tensor of group codes, or None.
    """
    if alpha_y_g is None or groups_tensor is None:
        return mu_y

    alpha_cell = alpha_y_g[groups_tensor]  # [N] or [N, K]

    if distribution == "negbinom":
        return mu_y * alpha_cell           # multiplicative
    elif distribution in ("normal", "studentt"):
        return mu_y + alpha_cell           # additive
    elif distribution == "binomial":
        import torch
        eps = 1e-6
        mu_c = mu_y.clamp(eps, 1.0 - eps)
        logit = torch.log(mu_c) - torch.log(1.0 - mu_c) + alpha_cell
        return torch.sigmoid(logit)
    elif distribution == "multinomial":
        import torch
        # alpha_cell: [N, K]; mu_y: [N, K]
        eps = torch.tensor(1e-12, dtype=mu_y.dtype, device=mu_y.device)
        log_mu = torch.log(mu_y.clamp_min(eps))
        logits = log_mu + alpha_cell
        return torch.softmax(logits, dim=-1)
    else:
        return mu_y


def _compute_nll_per_cell(y_obs, mu_final, distribution, o_y, nu_y, denom):
    """
    Negative log-likelihood per cell for a single gene.

    Returns [N_sub] float tensor (non-negative values).
    """
    import torch
    import torch.nn.functional as F

    eps = 1e-8

    if distribution == "negbinom":
        phi = 1.0 / (o_y ** 2 + eps)   # total_count
        phi = phi.clamp_min(eps)
        mu_safe = mu_final.clamp_min(eps)
        logits = torch.log(mu_safe) - torch.log(phi)
        nll = -torch.distributions.NegativeBinomial(
            total_count=phi, logits=logits
        ).log_prob(y_obs)

    elif distribution == "normal":
        sigma = o_y.clamp_min(eps)
        nll = -torch.distributions.Normal(loc=mu_final, scale=sigma).log_prob(y_obs)
        nll = torch.where(torch.isfinite(y_obs), nll, torch.zeros_like(nll))

    elif distribution == "studentt":
        sigma = o_y.clamp_min(eps)
        nu = (nu_y if nu_y is not None else torch.tensor(5.0)).clamp_min(2.01)
        nll = -torch.distributions.StudentT(
            df=nu, loc=mu_final, scale=sigma
        ).log_prob(y_obs)
        nll = torch.where(torch.isfinite(y_obs), nll, torch.zeros_like(nll))

    elif distribution == "binomial":
        if denom is None:
            raise ValueError("denom required for binomial NLL.")
        valid = denom > 0
        mu_c = mu_final.clamp(eps, 1.0 - eps)
        logit = torch.log(mu_c) - torch.log(1.0 - mu_c)
        n_safe = torch.where(valid, denom, torch.ones_like(denom))
        y_safe = torch.where(valid, y_obs, torch.zeros_like(y_obs))
        bce = F.binary_cross_entropy_with_logits(
            logit, y_safe / n_safe, reduction="none"
        )
        nll = torch.where(valid, n_safe * bce, torch.zeros_like(bce))

    elif distribution == "multinomial":
        # y_obs: [N, K], mu_final: [N, K]
        total = y_obs.sum(dim=-1)  # [N]
        log_p = torch.log(mu_final.clamp_min(1e-12))
        ll = (
            torch.lgamma(total + 1.0)
            - torch.lgamma(y_obs + 1.0).sum(dim=-1)
            + (y_obs * log_p).sum(dim=-1)
        )
        nll = -ll
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return nll.clamp_min(0.0)


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------

class DiagnosticsMixin:
    """
    Mixin providing post-fitting diagnostic tests for bayesDREAM models.
    Mixed into _BayesDREAMCore.
    """

    def check_systematic_shift(
        self,
        modality_name: Optional[str] = None,
        sum_factor_col: str = "sum_factor",
        target_col: str = "target",
        tech_col: str = "technical_group_code",
        ntc_label: str = "ntc",
        targeted_label: Optional[str] = None,
        min_cells_per_group: int = 30,
        df_spline: int = 6,
        degree: int = 3,
        theta: Optional[Union[dict, np.ndarray, pd.Series]] = None,
        exclude_cells: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Test for systematic expression shifts between NTC and targeted cells,
        matched on cis expression (x_true).

        For each feature × group (grouped by ``tech_col``, cells are first
        restricted to a matched x_true window (see "Matching window" below),
        then two GAMs are fit:

          Null: response ~ s(log2 x_true) [+ offset]
          Alt : response ~ s(log2 x_true) + targeted [+ offset]

        and a likelihood-ratio test (1 df) is computed. The smooth absorbs
        any covariation driven by the cis perturbation itself, so only
        residual shifts (not explained by dose-response) are flagged.

        Requires ``fit_cis()`` (and ideally ``refit_sumfactor()``) to have
        been called so that ``self.x_true`` is set, and ``len(self.x_true)``
        must equal ``len(self.meta)``.

        Matching window
        ----------------
        Within each group, NTC cells' ``log2(x_true)`` mean and SD are
        computed, and *both* NTC and targeted cells are restricted to
        ``ntc_mean ± ntc_sd`` (a fixed ±1 SD window; not configurable)
        before fitting. This is what makes the comparison "matched" rather
        than a naive NTC-vs-targeted test across the full dose range. Groups
        with fewer than 2 NTC cells (can't compute an SD), a zero/invalid
        NTC SD, or fewer than ``min_cells_per_group`` cells of either type
        remaining after windowing are skipped (see ``reason`` below).

        Parameters
        ----------
        modality_name : str or None
            Name of the modality to test (default: primary modality, usually 'gene').
        sum_factor_col : str
            Column in ``self.meta`` to use as normalisation offset (negbinom only).
        target_col : str
            Column in ``self.meta`` with perturbation labels. Cells are first
            filtered to ``target_col`` in ``{ntc_label, targeted_label}``.
        tech_col : str
            Column in ``self.meta`` to group by (default: integer technical-group
            codes, but any categorical column works, e.g. ``'cell_line'``). The
            output DataFrame's group column is named after ``tech_col`` itself
            (not hardcoded to ``'technical_group_code'``). To pool all cells
            into a single group, pass a column with one unique value.
        ntc_label : str
            Value in ``target_col`` identifying NTC cells.
        targeted_label : str or None
            Value in ``target_col`` identifying targeted cells.
            Defaults to ``self.cis_gene``.
        min_cells_per_group : int
            Minimum number of NTC **and** targeted cells required within the
            matched x_true window (see "Matching window" above); groups with
            fewer are skipped.
        df_spline : int
            Degrees of freedom for the B-spline smooth on log2(x_true).
        degree : int
            Polynomial degree of the B-spline.
        theta : dict, array, or Series, optional
            Per-feature dispersion parameter override. Ignored for
            ``binomial``/``multinomial`` distributions (no dispersion
            parameter applies there), even if supplied.
            - negbinom: NB total_count phi = 1/o_y^2
            - normal/studentt: sigma_y (residual std dev)
            If None, extracted automatically from ``modality.posterior_samples_ntc``
            (preferred -- the same pre-fit estimate fit_trans() itself anchors to) or
            from ``modality.posterior_samples_trans`` as a fallback.
        exclude_cells : list of str, optional
            Cell names (matched against ``self.meta['cell']``) to drop before
            windowing/fitting, e.g. cells carrying a guide you want excluded
            from the comparison. Applied globally, before the per-group NTC
            mean/SD window is computed, so excluded cells also don't
            contribute to that window.

        Returns
        -------
        pd.DataFrame
            One row per (feature × group) with columns:
            ``feature``, ``<tech_col>`` (named after the ``tech_col`` argument,
            e.g. ``technical_group_code`` by default), ``ok``, ``reason``,
            ``n_ntc``, ``n_targeted``, ``theta``, ``pval``, ``p_lrt``,
            ``p_adj`` (BH-corrected), ``shift_est``, ``shift_se``,
            ``shift_p``, ``shift_p_adj``, ``shift_fc``, ``lrt_stat``,
            ``lrt_df``.
            For multinomial data an additional ``n_categories_tested`` column
            is included.

            When ``ok`` is False, most other columns are absent/NaN for that
            row and ``reason`` is one of: ``"missing_theta"`` (negbinom only,
            dispersion unavailable), ``"too_few_ntc_for_sd_window"`` (fewer
            than 2 NTC cells in the group), ``"too_few_cells_before_subsetting"``
            (fewer than ``min_cells_per_group`` NTC or targeted cells in the
            group *before* windowing), ``"all_denominators_zero"`` (binomial
            only), ``"invalid_ntc_sd"`` (NTC SD is zero/non-finite),
            ``"too_few_cells_after_subsetting"`` (fewer than
            ``min_cells_per_group`` NTC or targeted cells remain *after*
            windowing), or ``"fit_failed: <ExceptionType>: <message>"``.

        Raises
        ------
        RuntimeError
            If ``self.x_true`` is not set (``fit_cis()`` hasn't been run).
        ValueError
            If ``targeted_label`` is None and ``self.cis_gene`` is also None;
            if ``target_col``, ``tech_col``, or (for negbinom) ``sum_factor_col``
            are missing from ``self.meta``; or if ``len(self.meta) != len(self.x_true)``.

        Notes
        -----
        Distribution-specific behaviour:

        * **negbinom** – NB-GAM with log-offset = log(sum_factor).
          Requires theta (phi = 1/o_y^2 = NB total_count). Auto-extracted from
          ``posterior_samples_ntc['o_y']`` (preferred) or ``posterior_samples_trans['o_y']``.
        * **normal** – Gaussian GAM with fixed scale = sigma_y. Auto-extracted from
          ``posterior_samples_ntc['sigma_y']`` (preferred) or ``posterior_samples_trans['sigma_y']``.
          If sigma is not available the GAM estimates it from the window data.
          No offset.
        * **studentt** – Proper Student-t GAM optimised via scipy with fixed
          sigma = sigma_y (from ``posterior_samples_ntc``, preferred, or
          ``posterior_samples_trans``) and nu = posterior median of nu_y (from
          ``posterior_samples_trans`` only -- nu_y is not fit during ``fit_ntc()``).
          Falls back to Gaussian GAM if either parameter is unavailable.
        * **binomial** – Binomial GAM (logit link). Denominator from
          ``modality.denominator``.
        * **multinomial** – Per-category Binomial GAM; p-values combined
          via Fisher's method.

        **Single group**: if ``tech_col`` has a single unique value across
        all cells, the test runs as normal and returns one row per feature.

        **High-MOI**: not MOI-aware — cells are grouped purely by ``target_col``
        (collapsed to a single label per cell upstream in high-MOI cell
        classification), so a "targeted" cell that also carries guides for
        other genes is still counted as targeted here.
        """
        _require_statsmodels()

        if not hasattr(self, "x_true") or self.x_true is None:
            raise RuntimeError(
                "x_true is not set. Run fit_cis() before check_systematic_shift()."
            )

        if targeted_label is None:
            targeted_label = self.cis_gene
        if targeted_label is None:
            raise ValueError("targeted_label must be specified (or set self.cis_gene).")

        # Resolve modality
        if modality_name is None:
            modality_name = getattr(self, "primary_modality", "gene")
        modality = self.get_modality(modality_name)
        distribution = modality.distribution

        # ---- Extract dispersion parameter theta (and nu for studentt) ---------
        theta_array = self._extract_theta_array(modality, theta, distribution)
        nu_array = self._extract_nu_array(modality) if distribution == "studentt" else None

        # ---- Build cell-level base DataFrame -----------------------------------
        base = self._build_shift_base(
            modality=modality,
            sum_factor_col=sum_factor_col,
            target_col=target_col,
            tech_col=tech_col,
            ntc_label=ntc_label,
            targeted_label=targeted_label,
            distribution=distribution,
            exclude_cells=exclude_cells,
        )
        if base is None or len(base) == 0:
            warnings.warn("No valid cells found after filtering. Returning empty result.")
            return pd.DataFrame()

        # ---- Extract counts array ----------------------------------------------
        counts_arr = self._get_counts_array(modality)  # [T, N] or [T, N, K]

        # ---- Feature names / indices -------------------------------------------
        feature_names = modality.feature_names
        if feature_names is None:
            feature_names = list(range(counts_arr.shape[0]))

        T = counts_arr.shape[0]
        if len(feature_names) != T:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) != "
                f"counts first dimension ({T})"
            )

        # ---- Loop over features × technical groups -----------------------------
        results = []
        for g_idx, feature in enumerate(feature_names):
            # Get per-feature dispersion (sigma for studentt; phi for negbinom)
            t_val = self._lookup_theta(theta_array, feature, g_idx, distribution)
            # For studentt: also get per-feature degrees of freedom
            nu_val = self._lookup_theta(nu_array, feature, g_idx, "studentt") \
                if nu_array is not None else float("nan")

            # Attach feature-specific counts to base
            dt_feature = self._attach_feature_counts(
                base=base,
                counts_arr=counts_arr,
                g_idx=g_idx,
                modality=modality,
                distribution=distribution,
            )

            for tech, dt_sub in dt_feature.groupby(tech_col, sort=True):
                row_base = {
                    "feature": feature,
                    tech_col: tech,
                    "theta": t_val,
                }

                # --- Skip checks ---
                skip_reason = self._shift_skip_reason(
                    dt_sub=dt_sub,
                    target_col=target_col,
                    ntc_label=ntc_label,
                    targeted_label=targeted_label,
                    min_cells_per_group=min_cells_per_group,
                    t_val=t_val,
                    distribution=distribution,
                )
                if skip_reason is not None:
                    results.append({**row_base, "ok": False, "reason": skip_reason})
                    continue

                # --- Subset to matched x window ---
                x_ntc = dt_sub.loc[dt_sub[target_col] == ntc_label, "x"]
                if len(x_ntc) < 2:
                    results.append({**row_base, "ok": False,
                                    "reason": "too_few_ntc_for_sd_window"})
                    continue
                ntc_mean, ntc_sd = x_ntc.mean(), x_ntc.std(ddof=1)
                if not (np.isfinite(ntc_sd) and ntc_sd > 0):
                    results.append({**row_base, "ok": False, "reason": "invalid_ntc_sd"})
                    continue

                dt_win = dt_sub[
                    (dt_sub["x"] > ntc_mean - ntc_sd) &
                    (dt_sub["x"] < ntc_mean + ntc_sd)
                ].copy()

                n_ntc = int((dt_win[target_col] == ntc_label).sum())
                n_tgt = int((dt_win[target_col] == targeted_label).sum())

                if n_ntc < min_cells_per_group or n_tgt < min_cells_per_group:
                    results.append({
                        **row_base, "ok": False,
                        "reason": "too_few_cells_after_subsetting",
                        "n_ntc": n_ntc, "n_targeted": n_tgt,
                    })
                    continue

                # --- Fit GAM ---
                try:
                    out = self._run_shift_gam(
                        dt=dt_win,
                        distribution=distribution,
                        t_val=t_val,
                        nu_val=nu_val,
                        df_spline=df_spline,
                        degree=degree,
                    )
                    results.append({
                        **row_base,
                        "ok": True,
                        "reason": "",
                        "n_ntc": n_ntc,
                        "n_targeted": n_tgt,
                        "pval": out["pval"],
                        "p_lrt": out["pval"],
                        "shift_est": out["shift_est"],
                        "shift_se": out.get("shift_se", float("nan")),
                        "shift_p": out.get("shift_p", out["pval"]),
                        "shift_fc": out.get("shift_fc", float("nan")),
                        "lrt_stat": out["lrt_stat"],
                        "lrt_df": 1,
                        **({} if "n_categories_tested" not in out
                           else {"n_categories_tested": out["n_categories_tested"]}),
                    })
                except Exception as e:
                    results.append({
                        **row_base,
                        "ok": False,
                        "reason": f"fit_failed: {type(e).__name__}: {e}",
                        "n_ntc": n_ntc,
                        "n_targeted": n_tgt,
                    })

        res = pd.DataFrame(results)
        if res.empty:
            return res

        # ---- Multiple testing correction ----------------------------------------
        for p_col, adj_col in [("p_lrt", "p_adj"), ("shift_p", "shift_p_adj")]:
            if p_col in res.columns:
                mask = res["ok"].fillna(False) & res[p_col].notna()
                res[adj_col] = float("nan")
                if mask.any():
                    res.loc[mask, adj_col] = multipletests(
                        res.loc[mask, p_col], method="fdr_bh"
                    )[1]

        return res

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _extract_theta_array(self, modality, user_theta, distribution):
        """
        Build a feature-indexed array of dispersion parameters.

        For negbinom:        theta = phi_y = 1/o_y^2  (NB total_count; statsmodels alpha = 1/theta)
        For normal/studentt: theta = sigma_y           (posterior site 'sigma_y', not 'o_y')
        For binomial/multinomial: None (not needed)

        Extraction priority:
          1. User-supplied ``user_theta``.
          2. modality.posterior_samples_ntc  (pre-fit technical estimate).
          3. modality.posterior_samples_trans  (fallback if NTC posteriors unavailable).

        NTC is preferred over trans: fit_trans() itself anchors its likelihood to the
        NTC-derived o_y/sigma_y (see fitting/trans.py's o_y_ntc_tensor / sigma_hat_tensor
        construction) because the o_y/sigma_y *resampled inside* fit_trans is only weakly
        identified per feature (fitting/trans.py explicitly notes the sampled o_y posterior
        collapses toward the prior mean for every gene). Using the same NTC anchor here
        keeps this test's dispersion consistent with what fit_trans actually relied on.

        If user_theta is supplied it overrides the posterior estimate.
        """
        if user_theta is not None:
            return user_theta  # caller-supplied; may be dict, array, or Series

        if distribution in ("binomial", "multinomial"):
            return None

        site_key = "o_y" if distribution == "negbinom" else "sigma_y"

        posterior = None
        for attr in ("posterior_samples_ntc", "posterior_samples_trans"):
            cand = getattr(modality, attr, None)
            if cand is not None and site_key in cand:
                posterior = cand
                break

        if posterior is None:
            return None

        val = posterior[site_key]
        if hasattr(val, "cpu"):
            val = val.detach().cpu().numpy()
        else:
            val = np.asarray(val)
        # Collapse all leading sample/batch dims via median, keeping the last
        # (feature) axis -- mirrors fitting/trans.py's o_y_ntc_tensor/sigma_hat_tensor
        # extraction, which loops rather than collapsing a single axis (site tensors can
        # carry an extra size-1 batch dim, e.g. shape [S, 1, T]).
        while val.ndim > 1:
            val = np.median(val, axis=0)

        if distribution == "negbinom":
            # theta = phi_y = 1/o_y^2  (statsmodels NB total_count)
            theta_arr = 1.0 / (val ** 2)
        else:
            # normal / studentt: theta = sigma_y directly
            theta_arr = val

        feature_names = modality.feature_names
        if feature_names is not None and len(theta_arr) == len(feature_names):
            return pd.Series(theta_arr, index=feature_names)
        return theta_arr  # positional fallback

    def _extract_nu_array(self, modality):
        """
        Extract per-feature degrees-of-freedom (nu_y) from posterior samples.

        Returns a pd.Series indexed by feature name, or a numpy array, or None.
        Only applies to studentt distribution; always prefers trans posteriors.
        """
        for attr in ("posterior_samples_trans", "posterior_samples_ntc"):
            cand = getattr(modality, attr, None)
            if cand is not None and "nu_y" in cand:
                nu_y = cand["nu_y"]
                if hasattr(nu_y, "cpu"):
                    nu_y = nu_y.detach().cpu().numpy()
                else:
                    nu_y = np.asarray(nu_y)
                while nu_y.ndim > 1:
                    nu_y = np.median(nu_y, axis=0)  # collapse to [T]
                feature_names = modality.feature_names
                if feature_names is not None and len(nu_y) == len(feature_names):
                    return pd.Series(nu_y, index=feature_names)
                return nu_y
        return None

    def _lookup_theta(self, theta_array, feature, g_idx, distribution):
        """Retrieve scalar theta (or nu) for a single feature."""
        if distribution in ("binomial", "multinomial"):
            return float("nan")
        if theta_array is None:
            return float("nan")
        if isinstance(theta_array, pd.Series):
            return float(theta_array.get(feature, float("nan")))
        if isinstance(theta_array, dict):
            return float(theta_array.get(feature, float("nan")))
        # array-like: positional
        try:
            return float(theta_array[g_idx])
        except (IndexError, TypeError):
            return float("nan")

    def _build_shift_base(
        self,
        modality,
        sum_factor_col,
        target_col,
        tech_col,
        ntc_label,
        targeted_label,
        distribution,
        exclude_cells=None,
    ):
        """
        Build the cell-level DataFrame used for all GAM fits.

        Returns a DataFrame indexed by integer cell position (0 .. N-1)
        with columns: target_col, tech_col, x, offset (negbinom only),
        targeted (0/1), and for binomial: denom_col.

        Cells are filtered to those with finite x_true and valid sum factor.
        """
        import torch

        x_true = self.x_true
        if hasattr(x_true, "cpu"):
            x_true = x_true.detach().cpu().numpy()
        else:
            x_true = np.asarray(x_true, dtype=float)

        N = len(x_true)

        # Start with integer positional index
        base = self.meta.reset_index(drop=True).copy()
        if len(base) != N:
            raise ValueError(
                f"len(meta) ({len(base)}) != len(x_true) ({N}). "
                "meta and x_true must be aligned."
            )

        base["x"] = np.log2(np.clip(x_true, 1e-12, None))

        # Drop explicitly excluded cells
        if exclude_cells:
            if "cell" not in base.columns:
                raise ValueError(
                    "exclude_cells was provided but 'cell' column not found in meta."
                )
            exclude_set = set(exclude_cells)
            base = base[~base["cell"].isin(exclude_set)].copy()

        # Filter to NTC and targeted only
        if target_col not in base.columns:
            raise ValueError(f"Column '{target_col}' not found in meta.")
        base = base[base[target_col].isin([ntc_label, targeted_label])].copy()

        # Drop non-finite x
        base = base[np.isfinite(base["x"])].copy()

        # Negbinom: offset from sum factors
        if distribution == "negbinom":
            if sum_factor_col not in base.columns:
                raise ValueError(
                    f"Column '{sum_factor_col}' not in meta. "
                    "Provide the correct sum_factor_col or run refit_sumfactor() first."
                )
            base["offset"] = _safe_log(
                pd.to_numeric(base[sum_factor_col], errors="coerce").to_numpy()
            )
            base = base[np.isfinite(base["offset"])].copy()

        # Technical group column
        if tech_col not in base.columns:
            raise ValueError(
                f"Column '{tech_col}' not in meta. "
                "Run set_technical_groups() or fit_ntc() first."
            )

        base["targeted"] = (base[target_col] == targeted_label).astype(int)
        return base

    def _get_counts_array(self, modality) -> np.ndarray:
        """Return counts as a dense numpy array [T, N] or [T, N, K]."""
        from scipy import sparse
        counts = modality.counts
        if sparse.issparse(counts):
            counts = counts.toarray()
        return np.asarray(counts, dtype=float)

    def _attach_feature_counts(
        self,
        base: pd.DataFrame,
        counts_arr: np.ndarray,
        g_idx: int,
        modality,
        distribution: str,
    ) -> pd.DataFrame:
        """
        Add count column(s) for feature g_idx to a copy of base.

        The base DataFrame has integer RangeIndex 0..N-1 matching the cell
        axis of counts_arr.
        """
        dt = base.copy()
        cell_pos = dt.index.to_numpy()  # integer positions into counts

        if distribution == "multinomial":
            # counts_arr shape: [T, N, K]
            feat_counts = counts_arr[g_idx, cell_pos, :]  # [n_cells, K]
            dt["y_total"] = feat_counts.sum(axis=1)
            K = feat_counts.shape[1]
            for k in range(K):
                dt[f"y_{k}"] = feat_counts[:, k]
        elif distribution == "binomial":
            dt["y"] = counts_arr[g_idx, cell_pos]
            # Denominator
            denom = modality.denominator
            if denom is None:
                raise ValueError(
                    "modality.denominator is None but distribution is 'binomial'. "
                    "Denominator is required for binomial shift test."
                )
            from scipy import sparse as sp
            if sp.issparse(denom):
                denom = denom.toarray()
            denom_arr = np.asarray(denom, dtype=float)
            dt["denom"] = denom_arr[g_idx, cell_pos]
        else:
            dt["y"] = counts_arr[g_idx, cell_pos]

        return dt

    def _shift_skip_reason(
        self,
        dt_sub,
        target_col,
        ntc_label,
        targeted_label,
        min_cells_per_group,
        t_val,
        distribution,
    ):
        """Return a reason string if this (feature, tech_group) should be skipped, else None."""
        # Check theta for distributions that need it
        if distribution == "negbinom":
            if not (np.isfinite(t_val) and t_val > 0):
                return "missing_theta"

        # Minimum cell counts (before window subsetting)
        n_ntc = int((dt_sub[target_col] == ntc_label).sum())
        n_tgt = int((dt_sub[target_col] == targeted_label).sum())
        if n_ntc < 2:
            return "too_few_ntc_for_sd_window"
        if n_ntc < min_cells_per_group or n_tgt < min_cells_per_group:
            return "too_few_cells_before_subsetting"

        # Distribution-specific checks
        if distribution == "binomial":
            if "denom" in dt_sub.columns:
                if (dt_sub["denom"] <= 0).all():
                    return "all_denominators_zero"

        return None

    def _run_shift_gam(
        self,
        dt: pd.DataFrame,
        distribution: str,
        t_val: float,
        nu_val: float = float("nan"),
        df_spline: int = 6,
        degree: int = 3,
    ) -> dict:
        """
        Dispatch to the appropriate distribution-specific GAM fitter.

        Parameters
        ----------
        t_val : float
            Per-feature dispersion:
            - negbinom:     phi = 1/o_y^2 (NB total_count)
            - normal:       sigma = o_y
            - studentt:     sigma = o_y
            - binomial/multinomial: ignored (NaN)
        nu_val : float
            Degrees of freedom for studentt only; NaN for all other distributions
            or when not available.
        """
        if distribution == "negbinom":
            return _fit_shift_nb_gam(dt, theta=t_val, df_spline=df_spline, degree=degree)
        elif distribution == "normal":
            sigma = t_val if (np.isfinite(t_val) and t_val > 0) else None
            return _fit_shift_gaussian_gam(dt, sigma=sigma, df_spline=df_spline, degree=degree)
        elif distribution == "studentt":
            sigma = t_val if (np.isfinite(t_val) and t_val > 0) else None
            nu = nu_val if (np.isfinite(nu_val) and nu_val > 2) else None
            if sigma is not None and nu is not None:
                return _fit_shift_studentt_gam(
                    dt, sigma=sigma, nu=nu, df_spline=df_spline, degree=degree
                )
            else:
                # Fall back to Gaussian if sigma or nu unavailable
                warnings.warn(
                    f"studentt shift test: sigma={'N/A' if sigma is None else f'{sigma:.3f}'}, "
                    f"nu={'N/A' if nu is None else f'{nu:.1f}'}. "
                    "Falling back to Gaussian GAM.",
                    UserWarning,
                    stacklevel=4,
                )
                return _fit_shift_gaussian_gam(dt, sigma=sigma, df_spline=df_spline, degree=degree)
        elif distribution == "binomial":
            return _fit_shift_binomial_gam(dt, df_spline=df_spline, degree=degree)
        elif distribution == "multinomial":
            y_cols = [c for c in dt.columns if c.startswith("y_") and c != "y_total"]
            n_cats = len(y_cols)
            return _fit_shift_multinomial_gam(
                dt, n_categories=n_cats, df_spline=df_spline, degree=degree
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    # =========================================================================
    # Loss matrix computation
    # =========================================================================

    def compute_loss_matrix(
        self,
        genes: Union[str, List[str]],
        modality_name: Optional[str] = None,
        cells: Optional[Union[List[str], str]] = None,
        cell_meta_filter: Optional[dict] = None,
        sum_factor_col: str = "sum_factor",
    ) -> pd.DataFrame:
        """
        Compute the per-cell, per-gene negative log-likelihood (loss) under the
        fitted trans model.

        The model parameters used are the **posterior means** from
        ``modality.posterior_samples_trans``.  The loss for each (gene, cell)
        pair is the negative log-probability of the observed count/measurement
        under the dose-response + technical-group model:

            loss[g, i] = -log p(y_{g,i} | f(x_true_i; θ_g), technical_group_i)

        where f is the fitted Hill or polynomial function.

        Requires ``fit_trans()`` to have been called for the target modality.

        Parameters
        ----------
        genes : str or list of str
            Gene (feature) name(s) to compute the loss for.  This must be a
            subset of ``modality.feature_names``.
        modality_name : str or None
            Modality to use (default: primary modality).
        cells : list of str or str or None
            Cells to include.
            - ``None`` (default): all cells.
            - list of cell name strings: subset to those cells.
            - string: passed as a ``pd.DataFrame.query()`` expression on
              ``self.meta`` (e.g. ``"target == 'ntc'``).
        cell_meta_filter : dict or None
            Alternative cell subsetting via column-value pairs, e.g.
            ``{'target': 'ntc', 'cell_line': ['A', 'B']}``.  Applied after
            the ``cells`` filter.  Values may be scalars or lists.
        sum_factor_col : str
            Column in ``self.meta`` with size factors (negbinom only).

        Returns
        -------
        pd.DataFrame
            Shape (n_genes, n_cells).  Index = gene names, columns = cell names.
            Values = negative log-likelihood (non-negative; higher = worse fit).

        Notes
        -----
        For negbinom the NLL depends on library size (cells with larger
        sum_factor contribute more counts and therefore have larger raw NLL).
        Dividing by ``log(sum_factor)`` gives a per-log-count normalised loss
        if cross-cell comparability is needed.
        """
        import torch
        import torch.distributions as tdist
        from .utils import (
            Hill_based_positive_logK,
            Hill_based_positive,
            Polynomial_function,
        )

        if modality_name is None:
            modality_name = getattr(self, "primary_modality", "gene")
        modality = self.get_modality(modality_name)
        distribution = modality.distribution

        posterior = getattr(modality, "posterior_samples_trans", None)
        if posterior is None:
            raise RuntimeError(
                f"posterior_samples_trans not found on modality '{modality_name}'. "
                "Run fit_trans() first."
            )

        function_type = (
            modality.trans_prior_params.get("function_type")
            if getattr(modality, "trans_prior_params", None)
            else None
        )
        if function_type is None:
            raise RuntimeError(
                "function_type not stored on modality.trans_prior_params. "
                "Re-run fit_trans() with a recent version of bayesDREAM."
            )

        # ---- Resolve gene list ------------------------------------------------
        if isinstance(genes, str):
            genes = [genes]
        feature_names = modality.feature_names or list(range(modality.counts.shape[0]))
        gene_indices = []
        for g in genes:
            if g not in feature_names:
                raise ValueError(
                    f"Gene '{g}' not found in modality '{modality_name}'. "
                    f"Available features: {feature_names[:10]}..."
                )
            gene_indices.append(feature_names.index(g))

        # ---- Resolve cell list / mask ----------------------------------------
        meta_reset = self.meta.reset_index(drop=True)
        cell_mask = pd.Series(True, index=meta_reset.index)

        if isinstance(cells, str):
            cell_mask = cell_mask & meta_reset.index.isin(meta_reset.query(cells).index)
        elif cells is not None:
            cell_names_col = meta_reset["cell"].values
            valid = set(cells)
            cell_mask = cell_mask & pd.Series(
                [c in valid for c in cell_names_col], index=meta_reset.index
            )

        if cell_meta_filter is not None:
            for col, val in cell_meta_filter.items():
                if col not in meta_reset.columns:
                    raise ValueError(f"cell_meta_filter column '{col}' not in meta.")
                if isinstance(val, (list, tuple, set)):
                    cell_mask = cell_mask & meta_reset[col].isin(val)
                else:
                    cell_mask = cell_mask & (meta_reset[col] == val)

        cell_positions = meta_reset.index[cell_mask].to_numpy()  # integer positions
        cell_names_out = meta_reset.loc[cell_mask, "cell"].values

        if len(cell_positions) == 0:
            raise ValueError("Cell filter matched zero cells.")

        # ---- Cell-level tensors ---------------------------------------------
        dev = getattr(self, "device", torch.device("cpu"))

        x_true_all = self.x_true
        if hasattr(x_true_all, "cpu"):
            x_true_all = x_true_all.detach()
        else:
            x_true_all = torch.tensor(np.asarray(x_true_all, dtype=np.float32), device=dev)
        x_subset = x_true_all[cell_positions]  # [N_sub]

        if distribution == "negbinom":
            if sum_factor_col not in meta_reset.columns:
                raise ValueError(f"sum_factor column '{sum_factor_col}' not in meta.")
            sf_all = torch.tensor(
                meta_reset[sum_factor_col].values.astype(np.float32), device=dev
            )
            sf_subset = sf_all[cell_positions]  # [N_sub]
        else:
            sf_subset = None

        has_groups = "technical_group_code" in meta_reset.columns
        if has_groups:
            groups_subset = torch.tensor(
                meta_reset.loc[cell_mask, "technical_group_code"].values.astype(np.int64),
                device=dev,
            )  # [N_sub]
            C = int(meta_reset["technical_group_code"].max()) + 1
        else:
            groups_subset = None
            C = 1

        # ---- Posterior mean parameters (all features, then subsetted) -------
        def _pm(key):
            """Return posterior mean of key as a float32 CPU tensor or None."""
            if key not in posterior:
                return None
            v = posterior[key]
            if hasattr(v, "detach"):
                v = v.detach().float()
            else:
                v = torch.tensor(np.asarray(v, dtype=np.float32))
            if v.dim() == 0:
                return v
            # Average over sample dimension (dim 0) if present
            # Heuristic: if first dim >> second, it's the sample dim
            return v.mean(dim=0) if v.dim() > 1 else v

        A         = _pm("A")          # [T] or [T, K] for multinomial
        alpha_pm  = _pm("alpha")      # [T] or [T, K] for multinomial
        beta_pm   = _pm("beta")
        Vmax_a    = _pm("Vmax_a")
        Vmax_b    = _pm("Vmax_b")
        log_K_a   = _pm("log_K_a")
        log_K_b   = _pm("log_K_b")
        K_b       = _pm("K_b")
        n_a       = _pm("n_a")
        n_b       = _pm("n_b")
        o_y       = _pm("o_y")        # [T]
        nu_y      = _pm("nu_y")       # [T] or None
        alpha_y_raw = _pm("alpha_y")  # [C-1, T] or [C, T] or [C-1, T, K]

        # Polynomial coefficients: collect by degree
        poly_coeffs = []
        for d in range(1, 20):
            c = _pm(f"poly_coeff_{d}")
            if c is None:
                break
            poly_coeffs.append(c)
        polynomial_degree = len(poly_coeffs) if poly_coeffs else None

        # ---- Reconstruct full alpha_y [C, T] or [C, T, K] -------------------
        if alpha_y_raw is not None and has_groups:
            alpha_y_full = _reconstruct_full_alpha_y(alpha_y_raw, C, distribution)
            alpha_y_full = alpha_y_full.to(dev)
        else:
            alpha_y_full = None

        # ---- Build loss arrays ----------------------------------------------
        loss_rows = {}
        N_sub = len(cell_positions)

        for g, g_idx in zip(genes, gene_indices):
            # Subset posterior mean parameters to this gene
            def _g(t):
                """Index tensor t at gene dimension."""
                if t is None:
                    return None
                if t.dim() == 0:
                    return t
                if t.shape[0] == len(feature_names):
                    return t[g_idx]
                return t  # scalar or already subsetted

            A_g     = _g(A)
            alp_g   = _g(alpha_pm)
            bet_g   = _g(beta_pm)
            Va_g    = _g(Vmax_a)
            Vb_g    = _g(Vmax_b)
            lKa_g   = _g(log_K_a)
            lKb_g   = _g(log_K_b)
            Kb_g    = _g(K_b)
            na_g    = _g(n_a)
            nb_g    = _g(n_b)
            oy_g    = _g(o_y)        # scalar
            nuy_g   = _g(nu_y)       # scalar or None

            if poly_coeffs:
                coeffs_g = torch.stack([c[g_idx] for c in poly_coeffs], dim=0)  # [D]
            else:
                coeffs_g = None

            # alpha_y for this gene across technical groups: [C] or [C, K]
            if alpha_y_full is not None:
                if alpha_y_full.dim() == 2:
                    alpha_y_g = alpha_y_full[:, g_idx]  # [C]
                else:
                    alpha_y_g = alpha_y_full[:, g_idx, :]  # [C, K]
            else:
                alpha_y_g = None

            # Observed counts for this gene × subset cells: shape [N_sub]
            from scipy import sparse as sp_sparse
            counts_arr = modality.counts
            if sp_sparse.issparse(counts_arr):
                counts_arr = counts_arr.toarray()
            else:
                counts_arr = np.asarray(counts_arr)

            if distribution == "multinomial":
                y_np = counts_arr[g_idx, :, :][cell_positions, :]  # [N_sub, K]
                y_obs = torch.tensor(y_np, dtype=torch.float32, device=dev)
            else:
                y_np = counts_arr[g_idx, cell_positions]  # [N_sub]
                y_obs = torch.tensor(y_np, dtype=torch.float32, device=dev)

            # Denominator for binomial
            if distribution == "binomial" and modality.denominator is not None:
                denom_arr = modality.denominator
                if sp_sparse.issparse(denom_arr):
                    denom_arr = denom_arr.toarray()
                denom_g = torch.tensor(
                    np.asarray(denom_arr)[g_idx, cell_positions].astype(np.float32),
                    device=dev,
                )
            else:
                denom_g = None

            # ---- Reconstruct μ_y [N_sub] or [N_sub, K] ---------------------
            mu_y = _reconstruct_mu_y(
                x_subset=x_subset,
                function_type=function_type,
                distribution=distribution,
                A=A_g, alpha=alp_g, beta=bet_g,
                Vmax_a=Va_g, Vmax_b=Vb_g,
                log_K_a=lKa_g, log_K_b=lKb_g, K_b=Kb_g,
                n_a=na_g, n_b=nb_g,
                poly_coeffs=coeffs_g,
                polynomial_degree=polynomial_degree,
                Hill_based_positive_logK=Hill_based_positive_logK,
                Hill_based_positive=Hill_based_positive,
                Polynomial_function=Polynomial_function,
                epsilon=1e-6,
            )  # [N_sub] or [N_sub, K]

            # ---- Apply technical-group effects ------------------------------
            mu_final = _apply_alpha_y(mu_y, alpha_y_g, groups_subset, distribution)

            # ---- Apply sum factor (negbinom only) ---------------------------
            if distribution == "negbinom" and sf_subset is not None:
                mu_final = mu_final * sf_subset  # [N_sub]

            # ---- Negative log-likelihood per cell ---------------------------
            nll = _compute_nll_per_cell(
                y_obs=y_obs,
                mu_final=mu_final,
                distribution=distribution,
                o_y=oy_g,
                nu_y=nuy_g,
                denom=denom_g,
            )  # [N_sub]

            loss_rows[g] = nll.detach().cpu().numpy()

        # ---- Assemble output DataFrame --------------------------------------
        return pd.DataFrame(loss_rows, index=cell_names_out).T  # [genes, cells]

    # =========================================================================
    # Loss vs x_true plot
    # =========================================================================

    def plot_loss_vs_xtrue(
        self,
        gene: str,
        modality_name: Optional[str] = None,
        cells: Optional[Union[List[str], str]] = None,
        cell_meta_filter: Optional[dict] = None,
        target_col: str = "target",
        ntc_label: str = "ntc",
        targeted_label: Optional[str] = None,
        tech_col: Optional[str] = "technical_group_code",
        sum_factor_col: str = "sum_factor",
        summarize: str = "bins",
        n_bins: int = 20,
        bin_stat: str = "median",
        show_scatter: bool = True,
        show_ribbon: bool = True,
        lowess_frac: float = 0.4,
        log_loss: bool = False,
        alpha_scatter: float = 0.25,
        ax=None,
        figsize=None,
    ):
        """
        Plot per-cell loss as a function of log₂(x_true), coloured by NTC vs
        targeted.

        Two summary strategies are offered via ``summarize``:

        ``"bins"`` (default)
            Divide cells into equal-frequency bins along log₂(x_true).  Within
            each bin compute the chosen ``bin_stat`` (median or mean) and the
            inter-quartile range (IQR).  The line shows the per-bin statistic;
            the shaded ribbon shows the IQR.  This is robust to the heavy right
            skew typical of NLL values.

        ``"lowess"``
            Overlay a LOWESS curve (locally weighted regression), which
            estimates the conditional mean of the loss given x_true.  The mean
            is sensitive to extreme NLL values (e.g. cells with very low
            counts); use ``log_loss=True`` or switch to ``"bins"`` if the curve
            is dominated by outliers.

        In both modes the underlying per-cell scatter is shown when
        ``show_scatter=True``.

        Parameters
        ----------
        gene : str
            Single gene to plot.
        modality_name : str or None
            Modality to use (default: primary modality).
        cells : list of str, query string, or None
            Cell subset (see ``compute_loss_matrix``).
        cell_meta_filter : dict or None
            Additional cell filter (see ``compute_loss_matrix``).
        target_col : str
            Column distinguishing NTC from targeted cells.
        ntc_label : str
            Value of target_col identifying NTC cells.
        targeted_label : str or None
            Value of target_col identifying targeted cells (default: self.cis_gene).
        tech_col : str or None
            Facet column.  Each unique value gets its own panel.  Pass ``None``
            to merge all technical groups.
        sum_factor_col : str
            Size-factor column (negbinom only).
        summarize : {"bins", "lowess"}
            Summary strategy (see above).
        n_bins : int
            Number of equal-frequency x_true bins (``summarize="bins"`` only).
        bin_stat : {"median", "mean"}
            Central statistic within each bin.  ``"median"`` is more robust to
            outliers; ``"mean"`` matches the expected NLL interpretation.
        show_scatter : bool
            Draw the per-cell scatter underneath the summary.
        show_ribbon : bool
            Draw the IQR ribbon (``summarize="bins"`` only).
        lowess_frac : float
            LOWESS bandwidth fraction 0–1 (``summarize="lowess"`` only).
        log_loss : bool
            Plot log₁₀(loss + 1) on the y-axis instead of raw NLL.  Useful
            when the distribution is very right-skewed (e.g. negbinom with
            high-coverage cells).  The ribbon and smoother are then on the
            log scale too.
        alpha_scatter : float
            Opacity of scatter points.
        ax : matplotlib Axes or None
            Axes to plot into.  If provided, tech_col facetting is ignored.
        figsize : tuple or None
            Figure size (ignored when ax is provided).

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if summarize not in ("bins", "lowess"):
            raise ValueError(f"summarize must be 'bins' or 'lowess', got '{summarize}'.")
        if bin_stat not in ("median", "mean"):
            raise ValueError(f"bin_stat must be 'median' or 'mean', got '{bin_stat}'.")

        if summarize == "lowess":
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            except ImportError:
                raise ImportError(
                    "statsmodels is required for summarize='lowess'. "
                    "Install with: pip install statsmodels"
                )

        if targeted_label is None:
            targeted_label = self.cis_gene

        # ---- Compute loss ---------------------------------------------------
        loss_df = self.compute_loss_matrix(
            genes=[gene],
            modality_name=modality_name,
            cells=cells,
            cell_meta_filter=cell_meta_filter,
            sum_factor_col=sum_factor_col,
        )
        loss_series = loss_df.loc[gene]

        # ---- Merge with metadata -------------------------------------------
        x_true_np = (
            self.x_true.detach().cpu().numpy()
            if hasattr(self.x_true, "detach")
            else np.asarray(self.x_true)
        )
        full_meta = self.meta.reset_index(drop=True)
        cell_to_pos = {c: i for i, c in enumerate(full_meta["cell"].values)}

        meta_plot = full_meta[full_meta["cell"].isin(loss_series.index)].copy()
        meta_plot["_loss"] = meta_plot["cell"].map(loss_series).values
        meta_plot["_log2_x"] = [
            float(np.log2(max(x_true_np[cell_to_pos[c]], 1e-12)))
            for c in meta_plot["cell"]
        ]

        # Optional y transform
        if log_loss:
            meta_plot["_y"] = np.log10(meta_plot["_loss"].clip(lower=0) + 1)
            y_label = "log₁₀(NLL + 1)"
        else:
            meta_plot["_y"] = meta_plot["_loss"]
            y_label = "Negative log-likelihood (loss)"

        # Restrict to NTC / targeted for colouring
        if target_col in meta_plot.columns:
            meta_plot = meta_plot[
                meta_plot[target_col].isin([ntc_label, targeted_label])
            ].copy()
        else:
            target_col = None

        # ---- Facetting ------------------------------------------------------
        groups_for_facet = (
            sorted(meta_plot[tech_col].unique())
            if (tech_col and tech_col in meta_plot.columns and ax is None)
            else [None]
        )
        n_panels = len(groups_for_facet)

        if ax is not None:
            axes = [ax]
            fig = ax.figure
        else:
            fs = figsize or (5 * n_panels, 4)
            fig, axes_arr = plt.subplots(1, n_panels, figsize=fs, squeeze=False)
            axes = axes_arr[0]

        ntc_color = "#4878CF"
        tgt_color = "#D65F5F"
        labels_colors = [(ntc_label, ntc_color), (targeted_label, tgt_color)]

        # ---- Draw panels ---------------------------------------------------
        for panel_idx, grp_val in enumerate(groups_for_facet):
            panel_ax = axes[panel_idx]

            df_panel = (
                meta_plot[meta_plot[tech_col] == grp_val]
                if grp_val is not None
                else meta_plot
            )

            for label, color in labels_colors:
                sub = (
                    df_panel[df_panel[target_col] == label]
                    if target_col is not None
                    else df_panel
                )
                if len(sub) == 0:
                    continue

                x_vals = sub["_log2_x"].values
                y_vals = sub["_y"].values
                sort_idx = np.argsort(x_vals)
                x_sorted = x_vals[sort_idx]
                y_sorted = y_vals[sort_idx]

                # Scatter
                if show_scatter:
                    panel_ax.scatter(
                        x_sorted, y_sorted,
                        c=color, alpha=alpha_scatter, s=10,
                        linewidths=0, rasterized=True, zorder=2,
                    )

                # Summary line + ribbon
                if summarize == "bins" and len(sub) >= n_bins:
                    # Equal-frequency bins along x_true
                    bin_edges = np.percentile(
                        x_sorted, np.linspace(0, 100, n_bins + 1)
                    )
                    bin_edges[-1] += 1e-9  # ensure last cell included
                    bin_ids = np.digitize(x_sorted, bin_edges[1:-1])

                    bin_x, bin_mid, bin_lo, bin_hi = [], [], [], []
                    for b in range(n_bins):
                        mask_b = bin_ids == b
                        if mask_b.sum() < 2:
                            continue
                        y_b = y_sorted[mask_b]
                        bin_x.append(x_sorted[mask_b].mean())
                        if bin_stat == "median":
                            bin_mid.append(np.median(y_b))
                        else:
                            bin_mid.append(np.mean(y_b))
                        bin_lo.append(np.percentile(y_b, 25))
                        bin_hi.append(np.percentile(y_b, 75))

                    if bin_x:
                        bx = np.array(bin_x)
                        bm = np.array(bin_mid)
                        bl = np.array(bin_lo)
                        bh = np.array(bin_hi)
                        panel_ax.plot(bx, bm, color=color, lw=2.0, zorder=5,
                                      label=label)
                        if show_ribbon:
                            panel_ax.fill_between(
                                bx, bl, bh,
                                color=color, alpha=0.20, linewidth=0, zorder=3,
                            )

                elif summarize == "lowess" and len(sub) >= 10:
                    smoothed = _lowess(y_sorted, x_sorted,
                                       frac=lowess_frac, return_sorted=True)
                    panel_ax.plot(smoothed[:, 0], smoothed[:, 1],
                                  color=color, lw=2.0, zorder=5, label=label)

                else:
                    # Not enough points for summary — just label the scatter
                    panel_ax.scatter([], [], c=color, s=10, label=label)

            # Axes labels & title
            panel_ax.set_xlabel("log₂(x_true)")
            stat_str = f"per-bin {bin_stat}" if summarize == "bins" else "LOWESS mean"
            ribbon_str = " ± IQR" if (summarize == "bins" and show_ribbon) else ""
            panel_ax.set_ylabel(f"{y_label}\n({stat_str}{ribbon_str})")
            title = gene
            if grp_val is not None:
                title += f"  [{tech_col}={grp_val}]"
            panel_ax.set_title(title)
            if panel_idx == n_panels - 1:
                panel_ax.legend(frameon=False, fontsize=9)

        fig.tight_layout()
        return fig
