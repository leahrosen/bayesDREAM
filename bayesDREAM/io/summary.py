"""
Summary export for bayesDREAM results.

Exports model results as R-friendly CSV files with:
- Mean and 95% credible intervals for all parameters
- Cell-wise and feature-wise summaries
- Observed log2FC, predicted log2FC, inflection points for trans fits
- Derivative roots and function classification

Supports three function types:
- additive_hill: Two Hill components with independent weights
- single_hill: Single Hill function with optional weight
- polynomial: Polynomial in x with configurable degree
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, Tuple, List
from scipy.optimize import brentq


class ModelSummarizer:
    """
    Export bayesDREAM results as CSV files for plotting.

    Creates summary tables with:
    - Mean and 95% credible intervals for parameters
    - Cell-wise and feature-wise data
    - Compatible with R for downstream plotting
    """

    def __init__(self, model):
        """
        Initialize summarizer with bayesDREAM model.

        Parameters
        ----------
        model : bayesDREAM
            Fitted bayesDREAM model instance
        """
        self.model = model

    # ========================================================================
    # Grid and Range Building Helpers
    # ========================================================================

    def _build_x_range_from_x_true(self, x_true: np.ndarray, n_grid: int = 2000, eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
        """
        Build a dense x grid spanning observed x_true range, spaced evenly in log2(x).
        Returns x_range and (x_obs_min, x_obs_max).
        """
        x_true = np.asarray(x_true, dtype=float)
        x_obs_min = max(np.nanmin(x_true), eps)
        x_obs_max = np.nanmax(x_true)
        if not np.isfinite(x_obs_min) or not np.isfinite(x_obs_max) or x_obs_max <= x_obs_min:
            # fallback to a sane range if degenerate
            x_obs_min = eps
            x_obs_max = 1.0
    
        # Create evenly spaced points in log2 space for derivative root finding
        log2_min = np.log2(x_obs_min)
        log2_max = np.log2(x_obs_max)
        
        # widen slightly so bracketing doesn’t fail right at the edges
        pad = 0.25  # in log2 units (~19% each side); tweak if you want
        x_range = 2.0 ** np.linspace(log2_min - pad, log2_max + pad, 6000)
        return x_range, x_obs_min, x_obs_max

    def _single_hill_derivative_roots(
        self,
        alpha: float,
        Vmax: float,
        K: float,
        n: float,
        x_range: np.ndarray,
        compute_third: bool = True
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Find derivative roots for single Hill function: y(x) = A + alpha * Vmax * x^n / (K^n + x^n)

        Since dy/dx = alpha * dHill/dx, roots of dy/dx are roots of dHill/dx (when alpha != 0).

        Parameters
        ----------
        alpha : float
            Weight/scale factor
        Vmax : float
            Hill amplitude
        K : float
            Half-max concentration
        n : float
            Hill coefficient
        x_range : np.ndarray
            Grid of x values to search for roots
        compute_third : bool
            Whether to compute third derivative roots

        Returns
        -------
        roots_d1 : List[float]
            First derivative roots (where dy/dx = 0)
        roots_d2 : List[float]
            Second derivative roots (where d²y/dx² = 0, inflection points)
        roots_d3 : List[float]
            Third derivative roots (where d³y/dx³ = 0)
        """
        # if alpha is (numerically) zero, y is flat -> all derivatives are ~0; roots are ill-defined.
        if not np.isfinite(alpha) or abs(alpha) < 1e-12:
            return [], [], []
    
        def d1(x):
            return alpha * self._hill_first_derivative(x, Vmax, K, n)
    
        def d2(x):
            return alpha * self._hill_second_derivative(x, Vmax, K, n)
    
        roots1 = self._find_roots_empirical(d1, x_range)
        roots2 = self._find_roots_empirical(d2, x_range)
    
        roots3 = []
        if compute_third:
            def d3(x):
                return alpha * self._hill_third_derivative(x, Vmax, K, n)
            roots3 = self._find_roots_empirical(d3, x_range)
    
        return roots1, roots2, roots3

    # ========================================================================
    # Polynomial Derivative Helpers
    # ========================================================================

    def _poly_derivative_coefs(self, coefs_asc: np.ndarray, order: int) -> np.ndarray:
        """
        Given polynomial coefficients in ascending order:
            p(x) = sum_{i=0}^d c[i] x^i
        Return coefficients of the 'order'-th derivative, also ascending.
        """
        c = np.asarray(coefs_asc, dtype=float).copy()
        for _ in range(order):
            if c.size <= 1:
                return np.array([0.0], dtype=float)
            # derivative: new_c[i-1] = i * c[i] for i>=1
            c = np.array([i * c[i] for i in range(1, c.size)], dtype=float)
        return c
    
    
    def _poly_eval(self, x: np.ndarray, coefs_asc: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial with ascending coefs at x (vectorized).
        """
        x = np.asarray(x, dtype=float)
        c = np.asarray(coefs_asc, dtype=float)
        # Horner's method (ascending -> reverse for Horner)
        y = np.zeros_like(x, dtype=float)
        for a in c[::-1]:
            y = y * x + a
        return y
    
    
    def _poly_derivative_roots(
        self,
        coefs_asc: np.ndarray,
        x_range: np.ndarray,
        compute_third: bool = True
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Find derivative roots for polynomial: p(x) = sum_{i=0}^d coef_i * x^i

        Uses analytic derivative coefficients and empirical root finding.

        Parameters
        ----------
        coefs_asc : np.ndarray
            Coefficients in ascending order [c0, c1, ..., cd]
        x_range : np.ndarray
            Grid of x values to search for roots
        compute_third : bool
            Whether to compute third derivative roots

        Returns
        -------
        roots_d1 : List[float]
            First derivative roots (where p'(x) = 0, extrema)
        roots_d2 : List[float]
            Second derivative roots (where p''(x) = 0, inflection points)
        roots_d3 : List[float]
            Third derivative roots (where p'''(x) = 0)
        """
        c1 = self._poly_derivative_coefs(coefs_asc, 1)
        c2 = self._poly_derivative_coefs(coefs_asc, 2)
        c3 = self._poly_derivative_coefs(coefs_asc, 3) if compute_third else None
    
        def d1(x):
            return self._poly_eval(x, c1)
    
        def d2(x):
            return self._poly_eval(x, c2)
    
        roots1 = self._find_roots_empirical(d1, x_range)
        roots2 = self._find_roots_empirical(d2, x_range)
    
        roots3 = []
        if compute_third:
            def d3(x):
                return self._poly_eval(x, c3)
            roots3 = self._find_roots_empirical(d3, x_range)
    
        return roots1, roots2, roots3

    def _n_effective_from_ci(self, n_mean, n_lo, n_hi):
        """
        Feature-wise rule: if 0 is inside the 95% CI, treat n as exactly 0
        for *all* downstream calculations.
        """
        n_mean = np.asarray(n_mean, dtype=float)
        n_lo = np.asarray(n_lo, dtype=float)
        n_hi = np.asarray(n_hi, dtype=float)
        zero_in_ci = (n_lo <= 0.0) & (0.0 <= n_hi)
        n_eff = n_mean.copy()
        n_eff[zero_in_ci] = 0.0
        return n_eff, zero_in_ci
    
    def _maybe_zero_scalar(self, x, do_zero: bool):
        """Convenience for per-sample scalars."""
        return 0.0 if do_zero else float(x)

    def _full_log2fc_from_extrema_and_boundaries(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_range,
        eps=1e-10
    ):
        """
        full_log2fc = log2(y_max / y_min) over x in [0, inf] using:
          - boundary limits at x->0 and x->inf
          - interior extrema from y'(x)=0
    
        Fixes:
          - uses *per-feature widened* x grid that covers observed range AND EC50s
          - uses permissive root finder for y'
          - DOES NOT clamp negative y to eps (invalid for log2); ignores nonpositive candidates
        """
        # --- Hill boundary limits (handle n==0 -> constant 0.5*V) ---
        def hill_limit_at_0(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return V if n < 0 else 0.0
    
        def hill_limit_at_inf(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return 0.0 if n < 0 else V
    
        y0 = A + alpha * hill_limit_at_0(Vmax_a, n_a) + beta * hill_limit_at_0(Vmax_b, n_b)
        yinf = A + alpha * hill_limit_at_inf(Vmax_a, n_a) + beta * hill_limit_at_inf(Vmax_b, n_b)
    
        # --- Build a widened x grid for finding y' roots ---
        xr = np.asarray(x_range, dtype=float)
        xr = xr[np.isfinite(xr) & (xr > 0)]
        if xr.size < 2:
            # fallback wide grid
            xr = np.logspace(-8, 8, 6000)
    
        log2_min = np.log2(np.min(xr))
        log2_max = np.log2(np.max(xr))
    
        # include EC50s in the span
        if np.isfinite(K_a) and K_a > 0:
            log2_min = min(log2_min, np.log2(K_a))
            log2_max = max(log2_max, np.log2(K_a))
        if np.isfinite(K_b) and K_b > 0:
            log2_min = min(log2_min, np.log2(K_b))
            log2_max = max(log2_max, np.log2(K_b))
    
        # extra padding for safety (2^2 = 4x each side)
        pad = 2.0
        xgrid = 2.0 ** np.linspace(log2_min - pad, log2_max + pad, max(6000, xr.size))
    
        # --- y'(x) and roots ---
        def yprime(x):
            return self._additive_hill_first_derivative(
                x, alpha, Vmax_a, K_a, n_a,
                beta, Vmax_b, K_b, n_b
            )
    
        # IMPORTANT: use the permissive root finder
        roots = self._find_roots_empirical_loose(yprime, xgrid)
    
        # --- evaluate candidates ---
        ys = [y0, yinf]
    
        # always include EC50s as candidates (cheap + helps if root finding misses one)
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (xx > 0):
                Ha = self._hill_value(xx, Vmax_a, K_a, n_a)
                Hb = self._hill_value(xx, Vmax_b, K_b, n_b)
                ys.append(A + alpha * Ha + beta * Hb)
    
        # extrema candidates
        for r in (roots or []):
            if not (np.isfinite(r) and r > 0):
                continue
            Ha = self._hill_value(r, Vmax_a, K_a, n_a)
            Hb = self._hill_value(r, Vmax_b, K_b, n_b)
            ys.append(A + alpha * Ha + beta * Hb)
    
        ys = np.asarray(ys, dtype=float)
    
        # DO NOT clamp negative/zero values for log2; those are invalid for log2(y)
        ys = ys[np.isfinite(ys) & (ys > eps)]
        if ys.size < 2:
            return np.nan
    
        return float(np.log2(np.max(ys) / np.min(ys)))

    def _full_delta_p_from_extrema_and_boundaries(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_range,
        eps=1e-10
    ):
        """
        full_delta_p = y_max - y_min over x in [0, inf] using:
          - boundary limits at x->0 and x->inf
          - interior extrema from y'(x)=0

        For binomial distributions where delta_p = p - p_ntc.
        """
        # --- Hill boundary limits (handle n==0 -> constant 0.5*V) ---
        def hill_limit_at_0(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return V if n < 0 else 0.0

        def hill_limit_at_inf(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return 0.0 if n < 0 else V

        y0 = A + alpha * hill_limit_at_0(Vmax_a, n_a) + beta * hill_limit_at_0(Vmax_b, n_b)
        yinf = A + alpha * hill_limit_at_inf(Vmax_a, n_a) + beta * hill_limit_at_inf(Vmax_b, n_b)

        # --- Build a widened x grid for finding y' roots ---
        xr = np.asarray(x_range, dtype=float)
        xr = xr[np.isfinite(xr) & (xr > 0)]
        if xr.size < 2:
            # fallback wide grid
            xr = np.logspace(-8, 8, 6000)

        log2_min = np.log2(np.min(xr))
        log2_max = np.log2(np.max(xr))

        # include EC50s in the span
        if np.isfinite(K_a) and K_a > 0:
            log2_min = min(log2_min, np.log2(K_a))
            log2_max = max(log2_max, np.log2(K_a))
        if np.isfinite(K_b) and K_b > 0:
            log2_min = min(log2_min, np.log2(K_b))
            log2_max = max(log2_max, np.log2(K_b))

        # extra padding for safety (2^2 = 4x each side)
        pad = 2.0
        xgrid = 2.0 ** np.linspace(log2_min - pad, log2_max + pad, max(6000, xr.size))

        # --- y'(x) and roots ---
        def yprime(x):
            return self._additive_hill_first_derivative(
                x, alpha, Vmax_a, K_a, n_a,
                beta, Vmax_b, K_b, n_b
            )

        # IMPORTANT: use the permissive root finder
        roots = self._find_roots_empirical_loose(yprime, xgrid)

        # --- evaluate candidates ---
        ys = [y0, yinf]

        # always include EC50s as candidates (cheap + helps if root finding misses one)
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (xx > 0):
                Ha = self._hill_value(xx, Vmax_a, K_a, n_a)
                Hb = self._hill_value(xx, Vmax_b, K_b, n_b)
                ys.append(A + alpha * Ha + beta * Hb)

        # extrema candidates
        for r in (roots or []):
            if not (np.isfinite(r) and r > 0):
                continue
            Ha = self._hill_value(r, Vmax_a, K_a, n_a)
            Hb = self._hill_value(r, Vmax_b, K_b, n_b)
            ys.append(A + alpha * Ha + beta * Hb)

        ys = np.asarray(ys, dtype=float)

        # For delta_p, we don't require positive values (probabilities can be any value)
        ys = ys[np.isfinite(ys)]
        if ys.size < 2:
            return np.nan

        return float(np.max(ys) - np.min(ys))

    def _full_log2fc_candidates_no_roots(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_candidates,
        eps=1e-10
    ):
        """
        full_log2fc using only boundary limits + supplied interior candidate points.
        Fixes:
          - includes EC50s as candidates
          - ignores nonpositive y instead of clamping to eps
        """
        def hill_limit_at_0(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return V if n < 0 else 0.0
    
        def hill_limit_at_inf(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return 0.0 if n < 0 else V
    
        y0 = A + alpha * hill_limit_at_0(Vmax_a, n_a) + beta * hill_limit_at_0(Vmax_b, n_b)
        yinf = A + alpha * hill_limit_at_inf(Vmax_a, n_a) + beta * hill_limit_at_inf(Vmax_b, n_b)
    
        xs = []
        if x_candidates is not None:
            for x in x_candidates:
                if x is None:
                    continue
                x = float(x)
                if np.isfinite(x) and (x > 0):
                    xs.append(x)
    
        # add EC50s as cheap robust candidates
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (xx > 0):
                xs.append(float(xx))
    
        ys = [y0, yinf]
        for x in xs:
            Ha = self._hill_value(x, Vmax_a, K_a, n_a)
            Hb = self._hill_value(x, Vmax_b, K_b, n_b)
            ys.append(A + alpha * Ha + beta * Hb)
    
        ys = np.asarray(ys, dtype=float)
        ys = ys[np.isfinite(ys) & (ys > eps)]
        if ys.size < 2:
            return np.nan
    
        return float(np.log2(np.max(ys) / np.min(ys)))

    def _observed_log2fc_candidates_no_roots(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_obs_min, x_obs_max,
        x_candidates=None,
        eps=1e-10
    ):
        """
        observed_log2fc over [x_obs_min, x_obs_max] using endpoints + interior candidates.
        Fix: ignores nonpositive y instead of clamping.
        """
        x0 = float(max(x_obs_min, eps))
        x1 = float(max(x_obs_max, eps))
    
        def y_at(x):
            Ha = self._hill_value(x, Vmax_a, K_a, n_a)
            Hb = self._hill_value(x, Vmax_b, K_b, n_b)
            return A + alpha * Ha + beta * Hb
    
        xs = [x0, x1]
    
        if x_candidates is not None:
            for x in x_candidates:
                if x is None:
                    continue
                x = float(x)
                if np.isfinite(x) and (x0 <= x <= x1) and (x > 0):
                    xs.append(x)
    
        # also include EC50s if they fall in-window
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (x0 <= xx <= x1):
                xs.append(float(xx))
    
        ys = np.asarray([y_at(x) for x in xs], dtype=float)
        ys = ys[np.isfinite(ys) & (ys > eps)]
        if ys.size < 2:
            return np.nan
    
        return float(np.log2(np.max(ys) / np.min(ys)))

    def _full_delta_p_candidates_no_roots(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_candidates,
        eps=1e-10
    ):
        """
        full_delta_p using only boundary limits + supplied interior candidate points.
        Returns y_max - y_min (for binomial distributions).
        """
        def hill_limit_at_0(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return V if n < 0 else 0.0

        def hill_limit_at_inf(V, n):
            if abs(n) < 1e-15:
                return 0.5 * V
            return 0.0 if n < 0 else V

        y0 = A + alpha * hill_limit_at_0(Vmax_a, n_a) + beta * hill_limit_at_0(Vmax_b, n_b)
        yinf = A + alpha * hill_limit_at_inf(Vmax_a, n_a) + beta * hill_limit_at_inf(Vmax_b, n_b)

        xs = []
        if x_candidates is not None:
            for x in x_candidates:
                if x is None:
                    continue
                x = float(x)
                if np.isfinite(x) and (x > 0):
                    xs.append(x)

        # add EC50s as cheap robust candidates
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (xx > 0):
                xs.append(float(xx))

        ys = [y0, yinf]
        for x in xs:
            Ha = self._hill_value(x, Vmax_a, K_a, n_a)
            Hb = self._hill_value(x, Vmax_b, K_b, n_b)
            ys.append(A + alpha * Ha + beta * Hb)

        ys = np.asarray(ys, dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size < 2:
            return np.nan

        return float(np.max(ys) - np.min(ys))

    def _observed_delta_p_candidates_no_roots(
        self,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_obs_min, x_obs_max,
        x_candidates=None,
        eps=1e-10
    ):
        """
        observed_delta_p over [x_obs_min, x_obs_max] using endpoints + interior candidates.
        Returns y_max - y_min (for binomial distributions).
        """
        x0 = float(max(x_obs_min, eps))
        x1 = float(max(x_obs_max, eps))

        def y_at(x):
            Ha = self._hill_value(x, Vmax_a, K_a, n_a)
            Hb = self._hill_value(x, Vmax_b, K_b, n_b)
            return A + alpha * Ha + beta * Hb

        xs = [x0, x1]

        if x_candidates is not None:
            for x in x_candidates:
                if x is None:
                    continue
                x = float(x)
                if np.isfinite(x) and (x0 <= x <= x1) and (x > 0):
                    xs.append(x)

        # also include EC50s if they fall in-window
        for xx in (K_a, K_b):
            if np.isfinite(xx) and (xx is not None) and (x0 <= xx <= x1):
                xs.append(float(xx))

        ys = np.asarray([y_at(x) for x in xs], dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size < 2:
            return np.nan

        return float(np.max(ys) - np.min(ys))

    def _is_flat_feature(self,
                         alpha_lower, alpha_upper,
                         beta_lower, beta_upper,
                         n_a_lower, n_a_upper,
                         n_b_lower, n_b_upper):
        """
        A feature is flat if neither Hill component is active.
        """
        dep_a = not (alpha_lower <= 0 <= alpha_upper) and not (n_a_lower <= 0 <= n_a_upper)
        dep_b = not (beta_lower  <= 0 <= beta_upper) and not (n_b_lower <= 0 <= n_b_upper)
        return not (dep_a or dep_b)

    def _build_u_range_from_observed_x(
        self,
        x_obs_min: float,
        x_obs_max: float,
        x_ntc: float,
        n_grid: int = 2000,
        pad_u: float = 0.25,     # same idea as your pad in log2 units
        eps: float = 1e-10
    ) -> np.ndarray:
        x0 = max(float(x_obs_min), eps)
        x1 = max(float(x_obs_max), eps)
        xntc = max(float(x_ntc), eps)
    
        u_min = np.log2(x0) - np.log2(xntc)
        u_max = np.log2(x1) - np.log2(xntc)
    
        # widen slightly
        u = np.linspace(u_min - pad_u, u_max + pad_u, int(n_grid))
        return u
    
    def _g_derivs_at_u(
        self,
        u: np.ndarray,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_ntc: float,
        eps: float = 1e-10
    ):
        """
        Vectorized g'(u), g''(u), g'''(u) for:
          g(u) = log2(y(x)) - log2(y_ntc),  x = x_ntc * 2^u
          y(x) = A + alpha*Ha(x) + beta*Hb(x)
        Uses S=y, S',S'',S''' w.r.t x and chain-rule to u.
    
        Returns: g1, g2, g3 arrays (same shape as u).
        """
        u = np.asarray(u, dtype=float)
        x = max(float(x_ntc), eps) * (2.0 ** u)
        ln2 = np.log(2.0)
    
        # S(x)
        # (use your stable hill_value; vectorize via np.vectorize or do explicit log-space like you did elsewhere)
        # We'll do explicit log-space for speed/consistency:
        x_safe = np.maximum(x, eps)
    
        def hill_vec(Vmax, K, n):
            K_safe = max(float(K), eps)
            n = float(n)
            if abs(n) < 1e-15:
                return np.full_like(x_safe, 0.5 * float(Vmax), dtype=float)
            x_n = self._exp_clip(n * np.log(x_safe))
            K_n = self._exp_clip(n * np.log(K_safe))
            return float(Vmax) * x_n / (K_n + x_n)
    
        Ha = hill_vec(Vmax_a, K_a, n_a)
        Hb = hill_vec(Vmax_b, K_b, n_b)
        S = float(A) + float(alpha) * Ha + float(beta) * Hb
        S = np.where(np.isfinite(S), S, np.nan)
    
        # S', S'', S''' (w.r.t x)
        Sp  = self._additive_hill_first_derivative(x_safe,  alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b)
        Spp = self._additive_hill_second_derivative(x_safe, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b)
        Sppp= self._additive_hill_third_derivative(x_safe,  alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b)
    
        # Guard bad S
        bad = ~np.isfinite(S) | (S <= eps)
        if np.all(bad):
            nan = np.full_like(u, np.nan, dtype=float)
            return nan, nan, nan
    
        # ratios
        r1 = Sp / S
        r2 = Spp / S
        r3 = Sppp / S
    
        # g'(u) = x * S'/S
        g1 = x_safe * r1
    
        # g''(u) = ln2 * [ x*S'/S + x^2*S''/S - x^2*(S'/S)^2 ]
        x2 = x_safe * x_safe
        g2 = ln2 * (x_safe * r1 + x2 * r2 - x2 * (r1 ** 2))
    
        # g'''(u) = (ln2)^2 * [ x*r1 + 3x^2*r2 - 3x^2*r1^2 + x^3*r3 - 3x^3*r1*r2 + 2x^3*r1^3 ]
        x3 = x2 * x_safe
        ln2_sq = ln2 * ln2
        g3 = ln2_sq * (x_safe * r1 + 3.0 * x2 * r2 - 3.0 * x2 * (r1 ** 2)
                       + x3 * r3 - 3.0 * x3 * r1 * r2 + 2.0 * x3 * (r1 ** 3))
    
        # Apply bad-mask
        g1 = np.where(bad, np.nan, g1)
        g2 = np.where(bad, np.nan, g2)
        g3 = np.where(bad, np.nan, g3)
        return g1, g2, g3

    def _S_derivs_at_u(
        self,
        u: np.ndarray,
        A, alpha, Vmax_a, K_a, n_a,
        beta, Vmax_b, K_b, n_b,
        x_ntc: float,
        eps: float = 1e-12
    ):
        u = np.asarray(u, dtype=float)
        x = max(float(x_ntc), eps) * (2.0 ** u)
        x = np.maximum(x, eps)
    
        # S(x)
        Ha = np.vectorize(self._hill_value)(x, Vmax_a, K_a, n_a)
        Hb = np.vectorize(self._hill_value)(x, Vmax_b, K_b, n_b)
        S  = float(A) + float(alpha) * Ha + float(beta) * Hb
    
        # derivatives wrt x
        Sp   = self._additive_hill_first_derivative (x, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b, epsilon=1e-12)
        Spp  = self._additive_hill_second_derivative(x, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b, epsilon=1e-12)
        Sppp = self._additive_hill_third_derivative (x, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b, epsilon=1e-12)
    
        # log2FC is undefined if S<=0; mask these out so they don't create bracketing artifacts
        bad = ~np.isfinite(S) | (S <= eps)
        S    = np.where(bad, np.nan, S)
        Sp   = np.where(bad, np.nan, Sp)
        Spp  = np.where(bad, np.nan, Spp)
        Sppp = np.where(bad, np.nan, Sppp)
    
        return x, S, Sp, Spp, Sppp
    
    def _find_roots_on_grid(
        self,
        func_u,
        u_grid,
        *,
        tol_u_dedup=1e-3,
        flat_atol=1e-10,
        flat_q=0.95,
        include_tangent=True,
        tangent_atol=None,
        tangent_neighbor_mult=25.0,
        maxiter=200,
    ):
        """
        Robust 1D root finder on a fixed grid:
          - sign-change bracketing on adjacent grid points
          - optional tangent roots (touching 0)
          - flat-curve bailout to avoid spurious/infinite roots
    
        Parameters
        ----------
        func_u : callable
            Vectorized: takes u array -> y array (same length).
        u_grid : array-like
            Monotone grid in u.
        tol_u_dedup : float
            Deduplicate roots closer than this in u.
        flat_atol : float
            If the function is essentially zero everywhere (robustly), return [].
        flat_q : float
            Quantile of |y| used to estimate amplitude for flatness.
        include_tangent : bool
            Also detect "touching" roots without sign change.
        tangent_atol : float or None
            Absolute tolerance for tangent detection. If None, set from robust amplitude.
        tangent_neighbor_mult : float
            Neighbors must be this many times larger than tangent_atol to accept tangent.
        """
        u = np.asarray(u_grid, dtype=float).ravel()
        if u.size < 3:
            return []
    
        y = np.asarray(func_u(u), dtype=float).ravel()
        if y.size != u.size:
            raise ValueError(f"func_u(u_grid) returned shape {y.shape}, expected {(u.size,)}")
    
        finite = np.isfinite(u) & np.isfinite(y)
        if finite.sum() < 3:
            return []
    
        # ---- flat-curve bailout (prevents "infinite roots" in near-zero curves)
        amp = np.nanquantile(np.abs(y[finite]), flat_q)
        if (not np.isfinite(amp)) or (amp < flat_atol):
            return []
    
        # choose tangent threshold
        if tangent_atol is None:
            tangent_atol = max(1e-12, 1e-6 * amp)
    
        def f_scalar(z):
            return float(np.asarray(func_u(np.array([z], dtype=float))).ravel()[0])
    
        roots = []
    
        # ---- sign-change bracketing on ADJACENT points (no deadband / no snapping)
        for i in range(u.size - 1):
            if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
                continue
    
            ui, uj = u[i], u[i + 1]
            yi, yj = y[i], y[i + 1]
    
            # exact zeros on grid
            if yi == 0.0:
                roots.append(ui)
                continue
            if yj == 0.0:
                roots.append(uj)
                continue
    
            # strict sign change
            if yi * yj < 0.0:
                try:
                    r = brentq(f_scalar, ui, uj, maxiter=maxiter)
                    if np.isfinite(r):
                        roots.append(r)
                except Exception:
                    pass
    
        # ---- tangent roots (touching 0 without sign change)
        if include_tangent:
            ay = np.abs(y)
            for i in range(1, u.size - 1):
                if not (np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1])):
                    continue
    
                if ay[i] > tangent_atol:
                    continue
    
                # local minimum of |y|
                if not (ay[i] <= ay[i - 1] and ay[i] <= ay[i + 1]):
                    continue
    
                # neighbors must be clearly away from zero
                if (ay[i - 1] < tangent_neighbor_mult * tangent_atol) or (ay[i + 1] < tangent_neighbor_mult * tangent_atol):
                    continue
    
                # same sign on both sides = touching
                if np.sign(y[i - 1]) == np.sign(y[i + 1]):
                    roots.append(u[i])
    
        # ---- dedup & sort
        roots = sorted([float(r) for r in roots if np.isfinite(r)])
        if not roots:
            return []
    
        deduped = [roots[0]]
        for r in roots[1:]:
            if abs(r - deduped[-1]) > tol_u_dedup:
                deduped.append(r)
    
        return deduped

    def _find_roots_empirical_ugrid(
        self,
        func_u,
        u_range: np.ndarray,
        tol_u_dedup: float = 1e-3,
        zero_atol: float = 1e-12,
        zero_rtol: float = 1e-9,
        noise_mult: float = 8.0,
        hysteresis: float = 3.0,
        include_near_zero_minima: bool = True,
        brentq_maxiter: int = 100,
    ) -> List[float]:
        """
        Robust root finder on a u-grid that:
          - returns [] for truly flat curves (prevents 'infinite roots')
          - avoids spurious roots in long near-zero plateaus / jittery regions
          - still finds real roots (including sign-change roots) reliably
    
        Strategy:
          1) Compute a deadband threshold thr using scale + noise floor (MAD of diffs).
          2) Snap |y|<=thr to 0 for sign bookkeeping.
          3) Bracket roots only between successive NONZERO sign points (skips plateaus).
          4) Optionally add tangent roots via isolated minima of |y| near 0.
        """
        u = np.asarray(u_range, dtype=float)
        if u.size < 2:
            return []
    
        y = np.asarray(func_u(u), dtype=float)
    
        finite = np.isfinite(u) & np.isfinite(y)
        if finite.sum() < 3:
            return []
    
        uu = u[finite]
        yy = y[finite]
    
        # scale
        scale = np.nanmax(np.abs(yy))
        if not np.isfinite(scale) or scale == 0.0:
            return []
    
        # noise floor from finite first differences
        dy = np.diff(yy)
        dy = dy[np.isfinite(dy)]
        if dy.size > 0:
            mad = np.median(np.abs(dy - np.median(dy)))
            noise = 1.4826 * mad
        else:
            noise = 0.0
    
        thr = max(zero_atol, zero_rtol * scale, noise_mult * noise)
        thr_hi = hysteresis * thr
    
        # --- flat curve guard: if everything is inside deadband, treat as "no roots"
        if np.all(np.abs(yy) <= thr):
            return []
    
        # snap for sign bookkeeping
        y_snap = yy.copy()
        y_snap[np.abs(y_snap) <= thr] = 0.0
        s = np.sign(y_snap)
    
        roots: List[float] = []
    
        # --- 1) sign-change roots, bracketing ONLY between nonzero sign points
        nz = np.where(s != 0)[0]
        if nz.size >= 2:
            for a, b in zip(nz[:-1], nz[1:]):
                if s[a] == s[b]:
                    continue
    
                uL, uR = float(uu[a]), float(uu[b])
    
                # hysteresis: at least one endpoint must be meaningfully away from 0
                if (abs(yy[a]) < thr_hi) and (abs(yy[b]) < thr_hi):
                    continue
    
                try:
                    root = brentq(
                        lambda z: float(np.asarray(func_u(np.array([z]))).ravel()[0]),
                        uL, uR,
                        maxiter=brentq_maxiter
                    )
                except Exception:
                    # fallback: linear interpolation in u
                    yL = float(yy[a]); yR = float(yy[b])
                    t = abs(yL) / (abs(yL) + abs(yR))
                    root = uL + t * (uR - uL)
    
                if np.isfinite(root):
                    roots.append(float(root))
    
        # --- 2) tangent roots: isolated minima of |y| near 0
        if include_near_zero_minima and yy.size >= 3:
            ay = np.abs(yy)
            mins = np.where((ay[1:-1] <= ay[:-2]) & (ay[1:-1] <= ay[2:]))[0] + 1
            for i in mins:
                if ay[i] > thr:
                    continue
    
                # "isolated": neighbors should be clearly away from zero
                if (ay[i - 1] < thr_hi) and (ay[i + 1] < thr_hi):
                    continue
    
                # and it's a touch (no sign change across neighbors)
                if np.sign(yy[i - 1]) == 0 or np.sign(yy[i + 1]) == 0:
                    continue
                if np.sign(yy[i - 1]) != np.sign(yy[i + 1]):
                    continue
    
                roots.append(float(uu[i]))
    
        if not roots:
            return []
    
        # de-duplicate in u
        roots = np.array([r for r in roots if np.isfinite(r)], dtype=float)
        if roots.size == 0:
            return []
    
        roots.sort()
        out = [float(roots[0])]
        last = float(roots[0])
        for r in roots[1:]:
            r = float(r)
            if abs(r - last) > tol_u_dedup:
                out.append(r)
                last = r
    
        return out

    def _roots_x_to_u_str(self, roots_x: List[float], x_ntc: float, eps: float = 1e-10) -> str:
        if (roots_x is None) or (len(roots_x) == 0) or (x_ntc is None) or (not np.isfinite(x_ntc)) or (x_ntc <= 0):
            return np.nan
        log2_xntc = np.log2(max(float(x_ntc), eps))
        u = []
        for r in roots_x:
            if r is None:
                continue
            r = float(r)
            if np.isfinite(r) and (r > 0):
                u.append(np.log2(max(r, eps)) - log2_xntc)
        if len(u) == 0:
            return np.nan
        return ";".join([f"{val:.4f}" for val in u])


    # ========================================================================
    # Technical Fit Summary
    # ========================================================================

    def save_technical_summary(
        self,
        output_dir: Optional[str] = None,
        modality_name: Optional[str] = None
    ):
        """
        Save technical fit parameters as feature-wise CSV.

        Creates: technical_feature_summary_{modality}.csv

        Columns:
        - feature: Feature name
        - modality: Modality name
        - distribution: Distribution type
        - group_{i}_alpha_y_mean: Mean alpha_y for group i
        - group_{i}_alpha_y_lower: 2.5% quantile for group i
        - group_{i}_alpha_y_upper: 97.5% quantile for group i

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir/model.label)
        modality_name : str, optional
            Modality to export (default: primary modality)
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        if modality_name is None:
            modality_name = self.model.primary_modality

        # Get modality
        modality = self.model.get_modality(modality_name)

        # Check if technical fit has been run for this modality
        if not hasattr(modality, 'alpha_y_prefit') or modality.alpha_y_prefit is None:
            raise ValueError(f"Technical fit not found for modality '{modality_name}'. Run fit_technical() first.")

        # Get alpha_y for this modality (prefer distribution-specific versions)
        if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
            alpha_y = modality.alpha_y_prefit_mult  # [n_samples, n_groups, n_features]
        elif hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
            alpha_y = modality.alpha_y_prefit_add  # [n_samples, n_groups, n_features]
        else:
            alpha_y = modality.alpha_y_prefit  # [n_samples, n_groups, n_features]

        # Convert to numpy if tensor
        if isinstance(alpha_y, torch.Tensor):
            alpha_y = alpha_y.cpu().numpy()

        # Get feature names - prefer modality.feature_names (what users see)
        if modality.feature_names is not None:
            feature_names = modality.feature_names
        else:
            feature_names = modality.feature_meta.index.tolist()
        n_features = len(feature_names)
        n_groups = alpha_y.shape[1]

        # Compute mean and CI
        alpha_mean = alpha_y.mean(axis=0)  # [n_groups, n_features]
        alpha_lower = np.quantile(alpha_y, 0.025, axis=0)
        alpha_upper = np.quantile(alpha_y, 0.975, axis=0)

        # Build DataFrame
        data = {
            'feature': feature_names,
            'modality': modality_name,
            'distribution': modality.distribution
        }

        # Add columns for each group
        for g in range(n_groups):
            data[f'group_{g}_alpha_y_mean'] = alpha_mean[g, :]
            data[f'group_{g}_alpha_y_lower'] = alpha_lower[g, :]
            data[f'group_{g}_alpha_y_upper'] = alpha_upper[g, :]

        df = pd.DataFrame(data)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'technical_feature_summary_{modality_name}.csv')
        df.to_csv(output_file, index=False)

        print(f"[SAVE] Technical summary saved to {output_file}")
        print(f"       {n_features} features × {n_groups} groups")

        return df

    # ========================================================================
    # Cis Fit Summary
    # ========================================================================

    def save_cis_summary(
        self,
        output_dir: Optional[str] = None,
        include_cell_level: bool = True
    ):
        """
        Save cis fit parameters as guide-wise and cell-wise CSVs.

        Creates:
        - cis_guide_summary.csv: Guide-level x_true with CI
        - cis_cell_summary.csv: Cell-level data (if include_cell_level=True)

        Guide-level columns:
        - guide: Guide name
        - target: Target gene
        - n_cells: Number of cells
        - x_true_mean: Mean x_true
        - x_true_lower: 2.5% quantile
        - x_true_upper: 97.5% quantile
        - raw_counts_mean: Average raw counts

        Cell-level columns:
        - cell: Cell barcode
        - guide: Guide name
        - target: Target gene
        - cell_line: Cell line (if available)
        - x_true_mean: Mean x_true for this guide
        - x_true_lower: 2.5% quantile
        - x_true_upper: 97.5% quantile
        - raw_counts: Raw counts for this cell

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir/model.label)
        include_cell_level : bool
            Whether to save cell-level summary (default: True)
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        # Check if cis fit has been run
        if not hasattr(self.model, 'x_true') or self.model.x_true is None:
            raise ValueError("Cis fit not found. Run fit_cis() first.")

        # Get x_true samples — shape is [n_samples, n_cells] (cell-level)
        if hasattr(self.model, 'posterior_samples_cis') and 'x_true' in self.model.posterior_samples_cis:
            x_true_cell_samples = self.model.posterior_samples_cis['x_true']
            if isinstance(x_true_cell_samples, torch.Tensor):
                x_true_cell_samples = x_true_cell_samples.cpu().numpy()
        else:
            # Use point estimate
            x_true_pt = self.model.x_true
            if isinstance(x_true_pt, torch.Tensor):
                x_true_pt = x_true_pt.cpu().numpy()
            x_true_cell_samples = x_true_pt[np.newaxis, :]  # [1, n_cells]

        # Get guide-level metadata
        guide_meta = self.model.meta.groupby('guide').agg({
            'target': 'first',
            'cell': 'count'
        }).rename(columns={'cell': 'n_cells'})

        guides = guide_meta.index.tolist()

        # Aggregate x_true from cell-level to guide-level (mean over cells per guide)
        guide_to_cell_indices = {
            g: self.model.meta.index[self.model.meta['guide'] == g].tolist()
            for g in guides
        }

        # Build [n_samples, n_guides] by averaging cells within each guide
        n_samples = x_true_cell_samples.shape[0]
        n_guides = len(guides)
        x_true_guide_samples = np.zeros((n_samples, n_guides))
        for gi, g in enumerate(guides):
            cell_idx = guide_to_cell_indices[g]
            x_true_guide_samples[:, gi] = x_true_cell_samples[:, cell_idx].mean(axis=1)

        # Compute mean and CI per guide
        x_true_mean = x_true_guide_samples.mean(axis=0)
        x_true_lower = np.quantile(x_true_guide_samples, 0.025, axis=0)
        x_true_upper = np.quantile(x_true_guide_samples, 0.975, axis=0)

        # Get raw counts from cis modality
        cis_mod = self.model.get_modality('cis')
        cis_counts = cis_mod.counts[0, :]  # [n_cells]

        # Map cells to guides
        cell_to_guide = dict(zip(self.model.meta['cell'], self.model.meta['guide']))
        guide_to_cells = self.model.meta.groupby('guide')['cell'].apply(list).to_dict()

        # Compute average raw counts per guide
        raw_counts_mean = []
        for guide in guides:
            guide_cells = guide_to_cells.get(guide, [])
            guide_cell_indices = [i for i, cell in enumerate(self.model.meta['cell']) if cell in guide_cells]
            if len(guide_cell_indices) > 0:
                raw_counts_mean.append(cis_counts[guide_cell_indices].mean())
            else:
                raw_counts_mean.append(np.nan)

        # Build guide-level DataFrame
        guide_df = pd.DataFrame({
            'guide': guides,
            'target': guide_meta['target'].values,
            'n_cells': guide_meta['n_cells'].values,
            'x_true_mean': x_true_mean,
            'x_true_lower': x_true_lower,
            'x_true_upper': x_true_upper,
            'raw_counts_mean': raw_counts_mean
        })

        # Save guide-level
        os.makedirs(output_dir, exist_ok=True)
        guide_file = os.path.join(output_dir, 'cis_guide_summary.csv')
        guide_df.to_csv(guide_file, index=False)

        print(f"[SAVE] Cis guide summary saved to {guide_file}")
        print(f"       {len(guides)} guides")

        # Build cell-level DataFrame if requested
        if include_cell_level:
            cell_data = {
                'cell': self.model.meta['cell'].values,
                'guide': self.model.meta['guide'].values,
                'target': self.model.meta['target'].values,
            }

            # Add cell_line if available
            if 'cell_line' in self.model.meta.columns:
                cell_data['cell_line'] = self.model.meta['cell_line'].values

            # Use cell-level x_true directly from posterior samples
            cell_data['x_true_mean'] = x_true_cell_samples.mean(axis=0)
            cell_data['x_true_lower'] = np.quantile(x_true_cell_samples, 0.025, axis=0)
            cell_data['x_true_upper'] = np.quantile(x_true_cell_samples, 0.975, axis=0)
            cell_data['raw_counts'] = cis_counts

            cell_df = pd.DataFrame(cell_data)

            # Save cell-level
            cell_file = os.path.join(output_dir, 'cis_cell_summary.csv')
            cell_df.to_csv(cell_file, index=False)

            print(f"[SAVE] Cis cell summary saved to {cell_file}")
            print(f"       {len(cell_df)} cells")

            return guide_df, cell_df

        return guide_df

    # ========================================================================
    # Trans Fit Summary
    # ========================================================================

    def save_trans_summary(
        self,
        output_dir: Optional[str] = None,
        modality_name: Optional[str] = None,
        compute_inflection: bool = True,
        compute_full_log2fc: bool = True,
        compute_derivative_roots: bool = True,
        compute_log2fc_params: bool = True,
        x_ntc: Optional[float] = None,
        y_ntc=None,
    ):
        """
        Save trans fit parameters as feature-wise CSV.

        Creates: trans_feature_summary_{modality}.csv

        Columns (depend on function_type and distribution):
        - feature: Feature name
        - modality: Modality name
        - distribution: Distribution type
        - function_type: Function type (additive_hill, single_hill, polynomial)
        - x_obs_min, x_obs_max: Observed x range (min/max of guide-level x_true)

        For negbinom/normal/studentt distributions:
        - observed_log2fc: log2(y_max / y_min) over observed x range from fitted function
        - full_log2fc_mean/lower/upper: log2(y_max / y_min) over theoretical x range (0 to ∞)

        For binomial distributions (e.g., splicing_sj):
        - observed_delta_p: y_max - y_min over observed x range (probability difference)
        - full_delta_p_mean/lower/upper: y_max - y_min over theoretical x range

        FDR columns (all function types):
        - fdr_alpha: Bayesian q-value for component A (positive Hill, or the single Hill
                     component in single_hill).  The minimum expected FDR among all called
                     sets that include this feature's component A.  Analogous to a
                     Benjamini-Hochberg adjusted p-value: calling all features with
                     fdr_alpha <= q controls the expected false discovery rate at q.
                     For additive_hill, computed over the *pooled* family of 2*n_features
                     tests (alpha + beta), which accounts for the increased multiple-testing
                     burden relative to single_hill.
        - fdr_beta:  Bayesian q-value for component B (negative Hill).  NaN for
                     single_hill and polynomial.  Computed in the same pooled family as
                     fdr_alpha for additive_hill.

        For additive_hill (all parameters needed to recreate: y = A + alpha*Vmax_a*Hill(x;K_a,n_a) + beta*Vmax_b*Hill(x;K_b,n_b)):
        - A_mean, A_lower, A_upper: Baseline (intercept)
        - Vmax_a_mean, Vmax_a_lower, Vmax_a_upper: Component A magnitude
        - K_a_mean, K_a_lower, K_a_upper: Component A half-max (EC50)
        - EC50_a_mean, EC50_a_lower, EC50_a_upper: Same as K_a (alias)
        - n_a_mean, n_a_lower, n_a_upper: Component A Hill coefficient (cooperativity)
        - alpha_mean, alpha_lower, alpha_upper: Component A weight
        - Vmax_b_mean, Vmax_b_lower, Vmax_b_upper: Component B magnitude
        - K_b_mean, K_b_lower, K_b_upper: Component B half-max (EC50)
        - EC50_b_mean, EC50_b_lower, EC50_b_upper: Same as K_b (alias)
        - n_b_mean, n_b_lower, n_b_upper: Component B Hill coefficient (cooperativity)
        - beta_mean, beta_lower, beta_upper: Component B weight
        - pi_y_mean, pi_y_lower, pi_y_upper: Sparsity weight (optional)
        - inflection_a_mean, inflection_a_lower, inflection_a_upper: Component A inflection x
        - inflection_b_mean, inflection_b_lower, inflection_b_upper: Component B inflection x
        - classification: Function shape classification based on dependency masks:
          - 'single_positive': Only one Hill component active, with positive n (increasing)
          - 'single_negative': Only one Hill component active, with negative n (decreasing)
          - 'additive_positive': Both components active, both with positive n
          - 'additive_negative': Both components active, both with negative n
          - 'non_monotonic_min': Both active, opposite signs, concave up at x=x_ntc (local min)
          - 'non_monotonic_max': Both active, opposite signs, concave down at x=x_ntc (local max)
          - 'flat': No active components
        - first_deriv_roots_mean: Roots of first derivative (x values where dy/dx=0)
        - second_deriv_roots_mean: Roots of second derivative (x values where d²y/dx²=0, inflections)
        - third_deriv_roots_mean: Roots of third derivative (x values where d³y/dx³=0)
        - n_first_deriv_roots: Number of first derivative roots (0 for monotonic)
        - n_second_deriv_roots: Number of second derivative roots (inflection points)
        - n_third_deriv_roots: Number of third derivative roots

        If compute_log2fc_params=True (log2FC relative to NTC):
        - x_ntc: NTC mean for cis gene (same for all trans genes)
        - y_ntc: NTC mean for each trans gene
        - log2fc_at_u0: log2FC value at u=0 (x = x_ntc), i.e., g(0) = log2(y(x_ntc)) - log2(y_ntc)
        - EC50_a_log2fc, EC50_b_log2fc: EC50 in log2FC x-space (log2(K) - log2(x_ntc))
        - inflection_a_log2fc, inflection_b_log2fc: Inflection points in log2FC x-space

        For negbinom/normal/studentt (log2FC y-space):
        - dg_du_at_u0, dg_du_at_u0_lower, dg_du_at_u0_upper: First derivative of log2FC at u=0 with 95% CI
        - d2g_du2_at_u0, d2g_du2_at_u0_lower, d2g_du2_at_u0_upper: Second derivative at u=0 with 95% CI
        - d3g_du3_at_u0, d3g_du3_at_u0_lower, d3g_du3_at_u0_upper: Third derivative at u=0 with 95% CI
        - first_deriv_roots_log2fc_mean: Roots of dg/du=0 in u-space (g = log2(y) - log2(y_ntc))
        - second_deriv_roots_log2fc_mean: Roots of d²g/du²=0
        - third_deriv_roots_log2fc_mean: Roots of d³g/du³=0
        - A_log2fc: Baseline in log2FC y-space (log2(A) - log2(y_ntc))

        For binomial distributions (delta_p y-space):
        - delta_p_at_u0: Probability difference at u=0, i.e., p(x_ntc) - p_ntc
        - dp_du_at_u0, dp_du_at_u0_lower, dp_du_at_u0_upper: First derivative of delta_p at u=0 with 95% CI
        - d2p_du2_at_u0, d2p_du2_at_u0_lower, d2p_du2_at_u0_upper: Second derivative at u=0 with 95% CI
        - d3p_du3_at_u0, d3p_du3_at_u0_lower, d3p_du3_at_u0_upper: Third derivative at u=0 with 95% CI
        - first_deriv_roots_delta_p_mean: Roots of dp/du=0 in u-space
        - second_deriv_roots_delta_p_mean: Roots of d²p/du²=0
        - third_deriv_roots_delta_p_mean: Roots of d³p/du³=0
        - A_delta_p: Baseline in delta_p space (A - y_ntc)

        For single_hill:
        - B_mean, B_lower, B_upper: Hill magnitude
        - K_mean, K_lower, K_upper: Hill coefficient
        - xc_mean, xc_lower, xc_upper: Half-max point (EC50 or IC50)
        - inflection_mean, inflection_lower, inflection_upper: Inflection x
        - full_log2fc_mean, full_log2fc_lower, full_log2fc_upper: Full dynamic range

        For polynomial:
        - coef_{i}_mean, coef_{i}_lower, coef_{i}_upper: Coefficient i
        - full_log2fc_mean, full_log2fc_lower, full_log2fc_upper: Full dynamic range

        Overdispersion and technical group effects (for simulation):
        - phi_y_mean, phi_y_lower, phi_y_upper: NB overdispersion (phi_y = 1/o_y^2)
        - group_{g}_alpha_y_mean: Technical group effect for group g (from fit_technical)

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir/model.label)
        modality_name : str, optional
            Modality to export (default: primary modality)
        compute_inflection : bool
            Whether to compute inflection points for Hill functions (default: True)
        compute_full_log2fc : bool
            Whether to compute full log2FC range (default: True)
        compute_derivative_roots : bool
            Whether to compute roots of first and second derivatives (default: True).
            Roots are found empirically over the observed x_range.
        compute_log2fc_params : bool
            If True (default), compute parameters in log2FC space relative to NTC:
            - x-axis: log2(x) - log2(x_ntc) where x_ntc is cis gene NTC mean
            - y-axis: log2(y) - log2(y_ntc) where y_ntc is trans gene NTC mean
            Requires posterior_samples_technical to be available, or x_ntc/y_ntc to be
            provided manually.
        x_ntc : float, optional
            Manually provided NTC mean for the cis gene (x-axis reference point).
            If None (default), computed from posterior_samples_technical of the cis modality.
            Useful when fit_technical was not run (e.g., CRISPRa/CRISPRi-only subsets).
        y_ntc : array-like, optional
            Manually provided NTC means for each trans feature (y-axis reference points).
            Shape: (n_features,). If None (default), computed from posterior_samples_technical
            of the trans modality.
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        if modality_name is None:
            modality_name = self.model.primary_modality

        # Get modality
        modality = self.model.get_modality(modality_name)

        # Check if trans fit has been run for this modality
        # Primary modality: check model-level posterior (backward compatibility)
        # Non-primary modality: check modality-level posterior
        if modality_name == self.model.primary_modality:
            if not hasattr(self.model, 'posterior_samples_trans') or self.model.posterior_samples_trans is None:
                raise ValueError(f"Trans fit not found for primary modality '{modality_name}'. Run fit_trans() first.")
            posterior = self.model.posterior_samples_trans
        else:
            if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
                raise ValueError(f"Trans fit not found for modality '{modality_name}'. Run fit_trans(modality_name='{modality_name}') first.")
            posterior = modality.posterior_samples_trans

        # Get feature names - prefer modality.feature_names (what users see and use)
        feature_meta = modality.feature_meta
        if modality.feature_names is not None:
            # First priority: use modality.feature_names (this is what users see)
            feature_names = modality.feature_names
        elif feature_meta is not None and len(feature_meta) > 0:
            # Fallback: try feature_meta columns or index
            # Check if index is integer-based (RangeIndex or integer dtype)
            index_is_integer = (
                isinstance(feature_meta.index, pd.RangeIndex) or
                feature_meta.index.dtype in ['int64', 'int32', 'int']
            )

            if index_is_integer:
                # Prefer named columns for feature identifiers
                if 'gene' in feature_meta.columns:
                    feature_names = feature_meta['gene'].tolist()
                elif 'gene_name' in feature_meta.columns:
                    feature_names = feature_meta['gene_name'].tolist()
                elif 'feature' in feature_meta.columns:
                    feature_names = feature_meta['feature'].tolist()
                else:
                    # Fall back to index
                    feature_names = feature_meta.index.tolist()
            else:
                # Index is named, use it
                feature_names = feature_meta.index.tolist()
        else:
            # No feature_names or feature_meta, use range
            feature_names = list(range(modality.counts.shape[0]))

        n_features = len(feature_names)

        # Determine function type from posterior keys
        keys = set(posterior.keys())
        if {'poly_coefs'} <= keys:
            function_type = 'polynomial'
        elif {'Vmax_a','K_a','n_a','Vmax_b','K_b','n_b'} <= keys:
            # Could be additive or nested; decide by presence of an indicator you store,
            # OR fall back to model setting if available.
            function_type = getattr(self.model, 'function_type', 'additive_hill')
        elif {'Vmax_a','K_a','n_a'} <= keys:
            function_type = 'single_hill'
        else:
            raise ValueError(
                f"Cannot determine function_type from posterior_samples_trans keys. "
                f"Found: {list(posterior.keys())}"
            )

        # Initialize DataFrame with basic info
        data = {
            'feature': feature_names,
            'modality': modality_name,
            'distribution': modality.distribution,
            'function_type': function_type
        }

        # Get x_range from model (guide-level x_true values)
        # This defines the "observed" x range as min to max of x_eff_g (guide-level x_true)
        x_range = None
        x_obs_min = None
        x_obs_max = None
        if hasattr(self.model, 'x_true') and self.model.x_true is not None:
            x_true = self.model.x_true
            if isinstance(x_true, torch.Tensor):
                x_true = x_true.cpu().numpy()
            if x_true.ndim > 1:
                x_true = x_true.mean(axis=0)  # Average over posterior samples
            # Store observed x range (min to max of guide-level x_true)
            x_obs_min = max(x_true.min(), 1e-6)
            x_obs_max = x_true.max()
            # Create evenly spaced points in log2 space for derivative root finding
            log2_min = np.log2(x_obs_min)
            log2_max = np.log2(x_obs_max)
            x_range, x_obs_min, x_obs_max = self._build_x_range_from_x_true(x_true, n_grid=6000, eps=1e-6)
            data['x_obs_min'] = x_obs_min
            data['x_obs_max'] = x_obs_max


        # Get x_ntc and y_ntc for log2FC parameter computation
        # Use manually provided values if given, otherwise try to compute from technical fit
        _x_ntc = float(x_ntc) if x_ntc is not None else None
        _y_ntc = np.asarray(y_ntc) if y_ntc is not None else None

        if compute_log2fc_params:
            # Get x_ntc from cis modality's technical fit (if not manually provided)
            if _x_ntc is None:
                cis_mod = self.model.get_modality('cis')
                if hasattr(cis_mod, 'posterior_samples_technical') and cis_mod.posterior_samples_technical is not None:
                    if 'mu_ntc' in cis_mod.posterior_samples_technical:
                        mu_ntc_cis = cis_mod.posterior_samples_technical['mu_ntc']
                        if isinstance(mu_ntc_cis, torch.Tensor):
                            mu_ntc_cis = mu_ntc_cis.cpu().numpy()
                        # mu_ntc is [n_samples, n_groups, 1] for cis (single gene)
                        # Average over samples and groups to get scalar x_ntc
                        _x_ntc = float(mu_ntc_cis.mean())

            if _x_ntc is not None:
                data['x_ntc'] = _x_ntc

            # Get y_ntc from trans modality's technical fit (per-feature, if not manually provided)
            if _y_ntc is None:
                if hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
                    if 'mu_ntc' in modality.posterior_samples_technical:
                        mu_ntc_trans = modality.posterior_samples_technical['mu_ntc']
                        if isinstance(mu_ntc_trans, torch.Tensor):
                            mu_ntc_trans = mu_ntc_trans.cpu().numpy()
                        # mu_ntc is [n_samples, n_groups, n_features]
                        # Average over samples and groups to get [n_features] y_ntc
                        _y_ntc = mu_ntc_trans.mean(axis=(0, 1))

            if _y_ntc is not None:
                data['y_ntc'] = _y_ntc

            if _x_ntc is None:
                print("[WARNING] Cannot compute log2FC params: x_ntc not available from cis modality. "
                      "Provide x_ntc manually to save_trans_summary().")
                compute_log2fc_params = False
            if _y_ntc is None:
                print("[WARNING] Cannot compute log2FC params: y_ntc not available from trans modality. "
                      "Provide y_ntc manually to save_trans_summary().")
                compute_log2fc_params = False

        # Add function-specific parameters
        if function_type == 'additive_hill':
            data = self._add_additive_hill_params(
                data, posterior, n_features,
                compute_inflection, compute_full_log2fc,
                compute_derivative_roots, x_range,
                compute_log2fc_params, _x_ntc, _y_ntc,
                x_obs_min, x_obs_max
            )
        elif function_type == 'single_hill':
            data = self._add_single_hill_params(
                data, posterior, n_features,
                compute_inflection=compute_inflection,
                compute_full_log2fc=compute_full_log2fc,
                compute_derivative_roots=compute_derivative_roots,
                x_range=x_range,
                x_obs_min=x_obs_min,
                x_obs_max=x_obs_max,
                compute_log2fc_params=compute_log2fc_params,
                x_ntc=_x_ntc,
                y_ntc=_y_ntc
            )
        elif function_type == 'polynomial':
            data = self._add_polynomial_params(
                data, posterior, n_features,
                compute_full_log2fc=compute_full_log2fc,
                compute_derivative_roots=compute_derivative_roots,
                x_range=x_range,
                x_obs_min=x_obs_min,
                x_obs_max=x_obs_max
            )

        # Compute Bayesian q-values (FDR columns) for Hill components
        fdr_cols = self._compute_bayesian_fdr(posterior, function_type, n_features)
        data.update(fdr_cols)

        # Add phi_y (overdispersion) from posterior o_y: phi_y = 1 / o_y^2
        if 'o_y' in posterior:
            o_y = posterior['o_y']
            if isinstance(o_y, torch.Tensor):
                o_y = o_y.cpu().numpy()
            # Handle [n_samples, n_features] or [n_samples, 1, n_features]
            if o_y.ndim == 3 and o_y.shape[1] == 1:
                o_y = o_y.squeeze(1)
            phi_y = 1.0 / (o_y ** 2)
            if phi_y.ndim >= 2:
                data['phi_y_mean'] = phi_y.mean(axis=0)
                data['phi_y_lower'] = np.quantile(phi_y, 0.025, axis=0)
                data['phi_y_upper'] = np.quantile(phi_y, 0.975, axis=0)
            else:
                data['phi_y_mean'] = phi_y
                data['phi_y_lower'] = phi_y
                data['phi_y_upper'] = phi_y

        # Add alpha_y (technical group effects) from modality.alpha_y_prefit
        if hasattr(modality, 'alpha_y_prefit') and modality.alpha_y_prefit is not None:
            # Get alpha_y (prefer distribution-specific versions)
            if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
                alpha_y_tech = modality.alpha_y_prefit_mult
            elif hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
                alpha_y_tech = modality.alpha_y_prefit_add
            else:
                alpha_y_tech = modality.alpha_y_prefit
            if isinstance(alpha_y_tech, torch.Tensor):
                alpha_y_tech = alpha_y_tech.cpu().numpy()
            # Shape: [n_samples, n_groups, n_features] (posterior) or [n_groups, n_features] (point)
            if alpha_y_tech.ndim == 3:
                alpha_y_mean = alpha_y_tech.mean(axis=0)  # [n_groups, n_features]
            elif alpha_y_tech.ndim == 2:
                alpha_y_mean = alpha_y_tech  # already [n_groups, n_features]
            else:
                alpha_y_mean = None
            if alpha_y_mean is not None and alpha_y_mean.ndim == 2:
                n_groups = alpha_y_mean.shape[0]
                for g in range(n_groups):
                    data[f'group_{g}_alpha_y_mean'] = alpha_y_mean[g, :]

        df = pd.DataFrame(data)

        # Merge with feature_meta from modality
        if modality.feature_meta is not None and len(modality.feature_meta) > 0:
            # Reset index to get feature names as a column for merging
            feature_meta_df = modality.feature_meta.reset_index()
            # Rename index column to avoid conflicts
            if 'index' in feature_meta_df.columns:
                feature_meta_df = feature_meta_df.rename(columns={'index': 'feature_meta_idx'})

            # Ensure feature column is string type for consistent merging
            df['feature'] = df['feature'].astype(str)

            # Merge on feature names - try to match by index or common columns
            if 'feature' in feature_meta_df.columns:
                feature_meta_df['feature'] = feature_meta_df['feature'].astype(str)
                df = df.merge(feature_meta_df, on='feature', how='left')
            elif 'gene' in feature_meta_df.columns:
                feature_meta_df['gene'] = feature_meta_df['gene'].astype(str)
                df = df.merge(feature_meta_df, left_on='feature', right_on='gene', how='left')
            elif 'gene_name' in feature_meta_df.columns:
                feature_meta_df['gene_name'] = feature_meta_df['gene_name'].astype(str)
                df = df.merge(feature_meta_df, left_on='feature', right_on='gene_name', how='left')
            else:
                # Try direct concatenation if feature order matches
                if len(feature_meta_df) == len(df):
                    for col in feature_meta_df.columns:
                        if col not in df.columns:
                            df[col] = feature_meta_df[col].values

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'trans_feature_summary_{modality_name}.csv')
        df.to_csv(output_file, index=False)

        print(f"[SAVE] Trans summary saved to {output_file}")
        print(f"       {n_features} features, function_type={function_type}")

        return df

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _compute_empirical_log2fc(self, modality):
        """Compute empirical log2FC from raw counts (perturbed vs NTC means)."""
        counts = modality.counts  # [n_features, n_cells]

        # Get NTC and perturbed masks
        ntc_mask = (self.model.meta['target'] == 'ntc').values
        pert_mask = (self.model.meta['target'] == self.model.cis_gene).values

        if ntc_mask.sum() == 0 or pert_mask.sum() == 0:
            # Return NaN if no NTC or perturbed cells
            return np.full(counts.shape[0], np.nan), np.full(counts.shape[0], np.nan)

        # Compute mean expression
        ntc_mean = counts[:, ntc_mask].mean(axis=1)
        pert_mean = counts[:, pert_mask].mean(axis=1)

        # Log2 fold change
        log2fc = np.log2((pert_mean + 1) / (ntc_mean + 1))

        # Compute standard error (approximate)
        ntc_std = counts[:, ntc_mask].std(axis=1)
        pert_std = counts[:, pert_mask].std(axis=1)

        n_ntc = ntc_mask.sum()
        n_pert = pert_mask.sum()

        # SE on log scale (delta method approximation)
        se_ntc = ntc_std / (np.sqrt(n_ntc) * (ntc_mean + 1))
        se_pert = pert_std / (np.sqrt(n_pert) * (pert_mean + 1))
        log2fc_se = np.sqrt(se_ntc**2 + se_pert**2) / np.log(2)

        return log2fc, log2fc_se

    def _compute_bayesian_fdr(
        self,
        posterior: dict,
        function_type: str,
        n_features: int,
    ) -> dict:
        """
        Compute per-component Bayesian q-values (FDR-equivalents) for Hill components.

        The Bayesian q-value for component i is the minimum expected false discovery rate
        (FDR) of any called set that includes component i:

            q_i = min_{k' >= rank(i)}  BayesFDR(top-k'),

        where features are ranked by local FDR ascending (most-significant first),

            lfdr_i = 1 - E[alpha_i]   (posterior probability that component is inactive),

        and BayesFDR(top-k) = mean of lfdr over the k most-significant features.

        This is the Bayesian analogue of the Benjamini-Hochberg adjusted p-value: calling
        all features with q_i <= q controls the expected false discovery rate at level q.

        For additive_hill (two components per feature), alpha and beta are treated as a
        *single pooled family* of 2*n_features tests.  This directly accounts for the
        increased multiple-testing burden: a feature must clear a higher evidence bar to
        reach the same q-value as in single_hill.

        Parameters
        ----------
        posterior : dict
            Posterior samples dict from fit_trans (keys include 'alpha', 'beta', etc.).
        function_type : str
            'additive_hill', 'single_hill', or 'polynomial'.
        n_features : int
            Number of trans features.

        Returns
        -------
        dict with keys:
            fdr_alpha : ndarray (n_features,)
                Bayesian q-value for component A (positive Hill / only component).
                Interpret as: the minimum expected FDR at which this feature's positive
                component would be included in a called set.
            fdr_beta : ndarray (n_features,)
                Bayesian q-value for component B (negative Hill).
                NaN for single_hill and polynomial.

        Notes
        -----
        The q-value formula uses the "running minimum from the right" trick to ensure
        monotonicity: q_i <= q_j whenever lfdr_i <= lfdr_j.  Concretely:

            cumfdr     = cumulative mean of sorted lfdr array
            qvals      = reverse running minimum of cumfdr
                       = minimum.accumulate(cumfdr[::-1])[::-1]

        The posterior mean E[alpha] is computed as the sample mean across all MCMC/VI
        samples.  Values are clipped to [0, 1] to guard against numerical noise.
        """

        def _posterior_mean(key: str) -> Optional[np.ndarray]:
            """Extract posterior mean for a parameter → shape (n_features,)."""
            if key not in posterior:
                return None
            param = posterior[key]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            if param.ndim == 0:
                return np.full(n_features, float(param))
            if param.ndim == 1:
                if param.shape[0] == n_features:
                    return param.copy()
                return np.full(n_features, float(param.mean()))
            # ≥ 2D: average over sample axis 0
            val = param.mean(axis=0)
            val = np.squeeze(val)
            if val.shape == () or val.shape == (1,):
                return np.full(n_features, float(val))
            return val

        def _qvalues(lfdr: np.ndarray) -> np.ndarray:
            """
            Convert a local-FDR array to Bayesian q-values.

            q_i = min_{k' >= rank(i)} cumulative_mean(lfdr_sorted[:k'])

            Implemented as a right-to-left running minimum of the cumulative mean,
            which ensures q-values are non-decreasing as local FDR increases.
            """
            n = len(lfdr)
            if n == 0:
                return np.array([], dtype=float)
            order = np.argsort(lfdr)
            lfdr_sorted = lfdr[order]
            cumfdr = np.cumsum(lfdr_sorted) / (np.arange(n, dtype=float) + 1.0)
            # Running minimum from right guarantees monotonicity
            qvals_sorted = np.minimum.accumulate(cumfdr[::-1])[::-1]
            qvals = np.empty(n, dtype=float)
            qvals[order] = qvals_sorted
            return qvals

        nan_col = np.full(n_features, np.nan)

        if function_type == 'polynomial':
            # Polynomial has no sparsity indicator; FDR not applicable
            return {'fdr_alpha': nan_col.copy(), 'fdr_beta': nan_col.copy()}

        if function_type == 'single_hill':
            E_alpha = _posterior_mean('alpha')
            if E_alpha is None:
                # No sparsity prior fitted — all features considered active
                return {'fdr_alpha': np.zeros(n_features, dtype=float),
                        'fdr_beta': nan_col.copy()}
            E_alpha = np.clip(E_alpha, 0.0, 1.0)
            return {
                'fdr_alpha': _qvalues(1.0 - E_alpha),
                'fdr_beta':  nan_col.copy(),
            }

        if function_type in ('additive_hill', 'nested_hill'):
            E_alpha = _posterior_mean('alpha')
            E_beta  = _posterior_mean('beta')
            E_alpha = np.ones(n_features) if E_alpha is None else np.clip(E_alpha, 0.0, 1.0)
            E_beta  = np.ones(n_features) if E_beta  is None else np.clip(E_beta,  0.0, 1.0)

            # Pool both components into a single family of 2*n_features tests.
            # The first n_features entries correspond to alpha (component A),
            # the next n_features to beta (component B).
            lfdr_all  = np.concatenate([1.0 - E_alpha, 1.0 - E_beta])
            qvals_all = _qvalues(lfdr_all)
            return {
                'fdr_alpha': qvals_all[:n_features],
                'fdr_beta':  qvals_all[n_features:],
            }

        # Unknown function type — return NaN
        return {'fdr_alpha': nan_col.copy(), 'fdr_beta': nan_col.copy()}

    def _add_additive_hill_params(self, data, posterior, n_features, compute_inflection, compute_full_log2fc,
                                    compute_derivative_roots=True, x_range=None,
                                    compute_log2fc_params=False, x_ntc=None, y_ntc=None,
                                    x_obs_min=None, x_obs_max=None):
        """
        Add additive Hill parameters to data dict.

        Uses individual parameter architecture: Vmax_a, K_a (EC50), n_a, Vmax_b, K_b, n_b.

        For binomial/multinomial:
        - Vmax_a = Vmax_sum * alpha (effective magnitude for component A)
        - Vmax_b = Vmax_sum * beta (effective magnitude for component B)
        - Where Vmax_sum comes from Beta/Dirichlet prior
        - Model: y = A + Vmax_sum * (alpha * Hill_a + beta * Hill_b)

        For negbinom/normal/studentt:
        - Vmax_a and Vmax_b are independent magnitudes sampled from log-normal priors
        """
        # All additive Hill models use individual parameters (Vmax_a, K_a, n_a, etc.)
        if 'Vmax_a' not in posterior:
            raise ValueError(
                f"Cannot find additive Hill parameters in posterior. Expected 'Vmax_a', 'K_a', 'n_a', etc. "
                f"Found keys: {list(posterior.keys())}"
            )

        return self._add_additive_hill_params_individual(
            data, posterior, n_features, compute_inflection, compute_full_log2fc,
            compute_derivative_roots, x_range,
            compute_log2fc_params, x_ntc, y_ntc, x_obs_min, x_obs_max
        )

    # Helper function to compute y(x) given parameters
    def _hill_value(self, x, Vmax, K, n, epsilon=1e-12):
        x_safe = max(float(x), epsilon)
        K_safe = max(float(K), epsilon)
        Vmax = float(Vmax); n = float(n)
    
        logx = np.log(x_safe)
        logK = np.log(K_safe)
    
        x_n = self._exp_clip(n * logx)
        K_n = self._exp_clip(n * logK)
    
        denom = K_n + x_n
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            out = Vmax * x_n / denom
    
        return out if np.isfinite(out) else np.nan

    def _add_additive_hill_params_individual(self, data, posterior, n_features, compute_inflection, compute_full_log2fc,
                                              compute_derivative_roots=True, x_range=None,
                                              compute_log2fc_params=False, x_ntc=None, y_ntc=None,
                                              x_obs_min=None, x_obs_max=None):
        """
        Add additive Hill parameters from individual parameter architecture (Vmax_a, K_a, n_a, etc.).

        Includes:
        - Basic parameters with mean and 95% CI
        - Function classification based on dependency masks
        - Derivative roots (where dy/dx=0 and d²y/dx²=0)
        - full_log2fc: log2(y_max / y_min) over theoretical range (in log2 space)
        - observed_log2fc: log2(y_max / y_min) over observed x range (in log2 space)
        - Log2FC versions of parameters relative to NTC (if compute_log2fc_params=True)

        For binomial distributions:
        - Uses delta_p (y_max - y_min) instead of log2fc (log2(y_max/y_min))
        - Derivative roots are found in y-space (dy/dx=0), then converted to u-space
        """

        # Check distribution type for delta_p vs log2fc handling
        distribution = data.get('distribution', 'negbinom')
        is_binomial = (distribution == 'binomial')

        def extract_param(name):
            """Helper to extract parameter with consistent handling of samples vs point estimates."""
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()

            # Handle different dimensionalities
            # Common shapes:
            # - [n_samples, n_features]: standard 2D
            # - [n_samples, 1, n_features]: has cis gene dimension (size 1) to squeeze
            # - [n_samples, n_features, n_categories]: multinomial, average over categories
            if param.ndim >= 2:
                if param.ndim == 3:
                    # Check if middle dimension is size 1 (cis gene dim to squeeze)
                    if param.shape[1] == 1:
                        param = param.squeeze(1)  # [n_samples, 1, n_features] -> [n_samples, n_features]
                    else:
                        # Otherwise it's [n_samples, n_features, n_categories] - average over categories
                        param = param.mean(axis=2)
                elif param.ndim > 3:
                    # Higher dimensions: squeeze size-1 dims, then average remaining extras
                    # First try to squeeze the cis gene dimension (axis 1) if size 1
                    if param.shape[1] == 1:
                        param = param.squeeze(1)
                    # Average over any remaining dimensions beyond 2
                    if param.ndim > 2:
                        param = param.mean(axis=tuple(range(2, param.ndim)))

                mean_val = param.mean(axis=0)
                lower_val = np.quantile(param, 0.025, axis=0)
                upper_val = np.quantile(param, 0.975, axis=0)
            else:
                # Point estimate: [n_features]
                mean_val = param
                lower_val = param
                upper_val = param

            return mean_val, lower_val, upper_val

        def extract_param_full(name):
            """Extract full posterior samples for per-sample computations."""
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            # Handle [n_samples, 1, n_features] -> [n_samples, n_features]
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            elif param.ndim > 2:
                # For other 3D+ cases, average over extra dimensions
                if param.shape[1] == 1:
                    param = param.squeeze(1)
                if param.ndim > 2:
                    param = param.mean(axis=tuple(range(2, param.ndim)))
            return param  # [n_samples, n_features] or [n_features]

        def extract_param_abs(name):
            """
            Posterior summaries for abs(parameter). This is needed for inflection because it depends on |n|.
            Returns mean, lower(2.5%), upper(97.5%) of |param|.
            """
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
        
            # Match extract_param logic:
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            elif param.ndim > 2:
                if param.shape[1] == 1:
                    param = param.squeeze(1)
                if param.ndim > 2:
                    param = param.mean(axis=tuple(range(2, param.ndim)))
        
            param_abs = np.abs(param)
        
            if param_abs.ndim >= 2:
                mean_val = param_abs.mean(axis=0)
                lower_val = np.quantile(param_abs, 0.025, axis=0)
                upper_val = np.quantile(param_abs, 0.975, axis=0)
            else:
                mean_val = lower_val = upper_val = param_abs
        
            return mean_val, lower_val, upper_val
        
        # Component A (first Hill function)
        Vmax_a_mean, Vmax_a_lower, Vmax_a_upper = extract_param('Vmax_a')
        K_a_mean, K_a_lower, K_a_upper = extract_param('K_a')  # EC50
        n_a_mean, n_a_lower, n_a_upper = extract_param('n_a')  # Hill coefficient

        data['Vmax_a_mean'] = Vmax_a_mean
        data['Vmax_a_lower'] = Vmax_a_lower
        data['Vmax_a_upper'] = Vmax_a_upper
        data['EC50_a_mean'] = K_a_mean
        data['EC50_a_lower'] = K_a_lower
        data['EC50_a_upper'] = K_a_upper
        data['n_a_mean'] = n_a_mean
        data['n_a_lower'] = n_a_lower
        data['n_a_upper'] = n_a_upper

        # Component B (second Hill function)
        Vmax_b_mean, Vmax_b_lower, Vmax_b_upper = extract_param('Vmax_b')
        K_b_mean, K_b_lower, K_b_upper = extract_param('K_b')  # EC50
        n_b_mean, n_b_lower, n_b_upper = extract_param('n_b')  # Hill coefficient

        data['Vmax_b_mean'] = Vmax_b_mean
        data['Vmax_b_lower'] = Vmax_b_lower
        data['Vmax_b_upper'] = Vmax_b_upper
        data['EC50_b_mean'] = K_b_mean
        data['EC50_b_lower'] = K_b_lower
        data['EC50_b_upper'] = K_b_upper
        data['n_b_mean'] = n_b_mean
        data['n_b_lower'] = n_b_lower
        data['n_b_upper'] = n_b_upper

        # APPLY YOUR RULE: if 0 in 95% CI, set n=0 for all downstream computations
        n_a_eff_mean, n_a_zeroed = self._n_effective_from_ci(n_a_mean, n_a_lower, n_a_upper)
        n_b_eff_mean, n_b_zeroed = self._n_effective_from_ci(n_b_mean, n_b_lower, n_b_upper)
        
        # overwrite the n used for computations (but keep original summaries as exported)
        n_a_used = n_a_eff_mean
        n_b_used = n_b_eff_mean
        
        # Pi_y (sparsity weight) - optional
        if 'pi_y' in posterior:
            pi_y_mean, pi_y_lower, pi_y_upper = extract_param('pi_y')
            data['pi_y_mean'] = pi_y_mean
            data['pi_y_lower'] = pi_y_lower
            data['pi_y_upper'] = pi_y_upper

        # Alpha and beta (component weights)
        # Default to 1.0 if not present (no sparsity)
        # Note: alpha/beta may be scalars (shared across features) or per-feature
        def _broadcast_to_features(arr, n_features):
            """Broadcast scalar or size-1 array to n_features length."""
            arr = np.atleast_1d(arr)
            if arr.size == 1:
                return np.full(n_features, arr.flat[0])
            return arr

        if 'alpha' in posterior:
            alpha_mean, alpha_lower, alpha_upper = extract_param('alpha')
            alpha_full = extract_param_full('alpha')
            # Broadcast if scalar
            alpha_mean = _broadcast_to_features(alpha_mean, n_features)
            alpha_lower = _broadcast_to_features(alpha_lower, n_features)
            alpha_upper = _broadcast_to_features(alpha_upper, n_features)
            if alpha_full.ndim == 1 or alpha_full.shape[-1] == 1:
                alpha_full = np.broadcast_to(alpha_full.reshape(-1, 1), (alpha_full.shape[0] if alpha_full.ndim > 1 else 1, n_features)).copy()
        else:
            alpha_mean = np.ones(n_features)
            alpha_lower = np.ones(n_features)
            alpha_upper = np.ones(n_features)
            alpha_full = np.ones((1, n_features))

        if 'beta' in posterior:
            beta_mean, beta_lower, beta_upper = extract_param('beta')
            beta_full = extract_param_full('beta')
            # Broadcast if scalar
            beta_mean = _broadcast_to_features(beta_mean, n_features)
            beta_lower = _broadcast_to_features(beta_lower, n_features)
            beta_upper = _broadcast_to_features(beta_upper, n_features)
            if beta_full.ndim == 1 or beta_full.shape[-1] == 1:
                beta_full = np.broadcast_to(beta_full.reshape(-1, 1), (beta_full.shape[0] if beta_full.ndim > 1 else 1, n_features)).copy()
        else:
            beta_mean = np.ones(n_features)
            beta_lower = np.ones(n_features)
            beta_upper = np.ones(n_features)
            beta_full = np.ones((1, n_features))

        data['alpha_mean'] = alpha_mean
        data['alpha_lower'] = alpha_lower
        data['alpha_upper'] = alpha_upper
        data['beta_mean'] = beta_mean
        data['beta_lower'] = beta_lower
        data['beta_upper'] = beta_upper

        # Also save K_a and K_b explicitly (same as EC50 but clearer naming for function recreation)
        data['K_a_mean'] = K_a_mean
        data['K_a_lower'] = K_a_lower
        data['K_a_upper'] = K_a_upper
        data['K_b_mean'] = K_b_mean
        data['K_b_lower'] = K_b_lower
        data['K_b_upper'] = K_b_upper

        # Get A (baseline) - required to recreate the fitted function
        if 'A' in posterior:
            A_mean, A_lower, A_upper = extract_param('A')
            A_full = extract_param_full('A')
            # Broadcast if scalar
            A_mean = _broadcast_to_features(A_mean, n_features)
            A_lower = _broadcast_to_features(A_lower, n_features)
            A_upper = _broadcast_to_features(A_upper, n_features)
            if A_full.ndim == 1 or A_full.shape[-1] == 1:
                A_full = np.broadcast_to(A_full.reshape(-1, 1), (A_full.shape[0] if A_full.ndim > 1 else 1, n_features)).copy()
        else:
            A_mean = np.zeros(n_features)
            A_lower = np.zeros(n_features)
            A_upper = np.zeros(n_features)
            A_full = np.zeros((1, n_features))

        # Save A (baseline) to data dict
        data['A_mean'] = A_mean
        data['A_lower'] = A_lower
        data['A_upper'] = A_upper

        # Compute inflection points for individual Hill components
        if compute_inflection:
            # Use the n USED for computations (already zeroed if CI includes 0)
            nA = np.asarray(n_a_used, dtype=float)
            nB = np.asarray(n_b_used, dtype=float)
        
            # For CI, we should use |n| bounds. A conservative choice:
            #   nabs_lower = min(|n_lo|, |n_hi|)  (lower bound on magnitude, but can be too small if CI spans 0)
            #   nabs_upper = max(|n_lo|, |n_hi|)
            nAabs_lo = np.minimum(np.abs(n_a_lower), np.abs(n_a_upper))
            nAabs_hi = np.maximum(np.abs(n_a_lower), np.abs(n_a_upper))
            nBabs_lo = np.minimum(np.abs(n_b_lower), np.abs(n_b_upper))
            nBabs_hi = np.maximum(np.abs(n_b_lower), np.abs(n_b_upper))
        
            # If you zeroed n because CI contains 0, enforce magnitude bounds = 0 too
            nAabs_lo[n_a_zeroed] = 0.0
            nAabs_hi[n_a_zeroed] = 0.0
            nBabs_lo[n_b_zeroed] = 0.0
            nBabs_hi[n_b_zeroed] = 0.0
        
            inflection_a_mean = self._compute_hill_inflection(nA,      K_a_mean)
            inflection_a_lower = self._compute_hill_inflection(nAabs_lo, K_a_lower)  # magnitude-lower
            inflection_a_upper = self._compute_hill_inflection(nAabs_hi, K_a_upper)  # magnitude-upper
        
            inflection_b_mean = self._compute_hill_inflection(nB,      K_b_mean)
            inflection_b_lower = self._compute_hill_inflection(nBabs_lo, K_b_lower)
            inflection_b_upper = self._compute_hill_inflection(nBabs_hi, K_b_upper)
        
            data['inflection_a_mean'] = inflection_a_mean
            data['inflection_a_lower'] = inflection_a_lower
            data['inflection_a_upper'] = inflection_a_upper
            data['inflection_b_mean'] = inflection_b_mean
            data['inflection_b_lower'] = inflection_b_lower
            data['inflection_b_upper'] = inflection_b_upper

        # Get full posterior samples for derivative computation
        Vmax_a_full = extract_param_full('Vmax_a')
        K_a_full = extract_param_full('K_a')
        n_a_full = extract_param_full('n_a')
        Vmax_b_full = extract_param_full('Vmax_b')
        K_b_full = extract_param_full('K_b')
        n_b_full = extract_param_full('n_b')

        # Initialize arrays for per-feature results
        classifications = []
        first_deriv_roots_mean_list = []
        second_deriv_roots_mean_list = []
        third_deriv_roots_mean_list = []
        # For negbinom: roots of dg/du=0 in u-space (g = log2(y) - log2(y_ntc))
        # For binomial: roots of dp/du=0 in u-space (same as dy/dx=0 roots, converted to u)
        first_deriv_roots_log2fc_mean_list = []
        second_deriv_roots_log2fc_mean_list = []
        third_deriv_roots_log2fc_mean_list = []
        n_first_deriv_roots = []
        n_second_deriv_roots = []
        n_third_deriv_roots = []
        # For negbinom: log2(y_max / y_min)
        # For binomial: y_max - y_min (delta_p)
        full_log2fc_mean_list = []
        full_log2fc_lower_list = []
        full_log2fc_upper_list = []
        observed_log2fc_list = []
        observed_log2fc_lower_list = []
        observed_log2fc_upper_list = []

        # Process each feature
        for i in range(n_features):
            # Get mean parameters for this feature
            alpha_i = alpha_mean[i]
            beta_i = beta_mean[i]
            Vmax_a_i = Vmax_a_mean[i]
            Vmax_b_i = Vmax_b_mean[i]
            K_a_i = K_a_mean[i]
            K_b_i = K_b_mean[i]
            n_a_i = float(n_a_used[i])
            n_b_i = float(n_b_used[i])
            A_i = A_mean[i]

            # Find derivative roots for mean parameters
            first_roots_mean = []
            second_roots_mean = []
            third_roots_mean = []

            if compute_derivative_roots and x_range is not None:
                # First derivative roots (where dy/dx = 0)
                def first_deriv_func(x):
                    return self._additive_hill_first_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                first_roots_mean = self._find_roots_empirical_loose(first_deriv_func, x_range)

                # Second derivative roots (inflection points of combined function)
                def second_deriv_func(x):
                    return self._additive_hill_second_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                second_roots_mean = self._find_roots_empirical_loose(second_deriv_func, x_range)

                # Third derivative roots (where d³y/dx³ = 0)
                def third_deriv_func(x):
                    return self._additive_hill_third_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                third_roots_mean = self._find_roots_empirical_loose(third_deriv_func, x_range)
                
            # --- roots in log2FC x-space (u-space) ---
            # For negbinom: dg/du=0, d2g/du2=0, d3g/du3=0 where g = log2(y) - log2(y_ntc)
            # For binomial: dp/du=0, etc. where delta_p = p - p_ntc
            #   Since dp/du = dp/dx * x * ln(2), roots of dp/du=0 are same as dy/dx=0
            #   So for binomial, just convert x-space roots to u-space
            u_roots_g1 = []
            u_roots_g2 = []
            u_roots_g3 = []

            if compute_log2fc_params and (x_ntc is not None) and np.isfinite(x_ntc) and (x_ntc > 0) \
               and (x_obs_min is not None) and (x_obs_max is not None):

                eps = 1e-10
                log2_xntc = np.log2(max(float(x_ntc), eps))

                # Build u-range for root finding (used for both binomial and negbinom)
                u_range_local = self._build_u_range_from_observed_x(
                    x_obs_min=x_obs_min,
                    x_obs_max=x_obs_max,
                    x_ntc=x_ntc,
                    n_grid=6000,
                    pad_u=0.25
                )

                # WIDEN using EC50s so we don't miss roots outside observed x-range
                u_ec50 = []
                if np.isfinite(K_a_i) and K_a_i > 0:
                    u_ec50.append(np.log2(max(float(K_a_i), eps)) - log2_xntc)
                if np.isfinite(K_b_i) and K_b_i > 0:
                    u_ec50.append(np.log2(max(float(K_b_i), eps)) - log2_xntc)

                if len(u_ec50) > 0:
                    umin = min(float(np.nanmin(u_range_local)), float(np.min(u_ec50))) - 1.0
                    umax = max(float(np.nanmax(u_range_local)), float(np.max(u_ec50))) + 1.0
                    u_range_local = np.linspace(umin, umax, 6000)

                if is_binomial:
                    # For binomial (delta_p = S - S_ntc):
                    # dp/du = S' * x * ln(2) → roots when S' = 0
                    # d²p/du² ∝ S' + x*S'' → roots when S' + x*S'' = 0
                    # d³p/du³ ∝ S' + 3x*S'' + x²*S''' → roots when S' + 3x*S'' + x²*S''' = 0

                    def p1_num_u(u):
                        # dp/du = 0 when S' = 0
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        return Sp

                    def p2_num_u(u):
                        # d²p/du² = 0 when S' + x*S'' = 0
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        return Sp + x * Spp

                    def p3_num_u(u):
                        # d³p/du³ = 0 when S' + 3x*S'' + x²*S''' = 0
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        return Sp + 3.0 * x * Spp + (x ** 2) * Sppp

                    u_roots_g1 = self._find_roots_on_grid(p1_num_u, u_range_local, include_tangent=True)
                    u_roots_g2 = self._find_roots_on_grid(p2_num_u, u_range_local, include_tangent=True)
                    u_roots_g3 = self._find_roots_on_grid(p3_num_u, u_range_local, include_tangent=True)
                else:
                    # For negbinom: find roots of dg/du = 0 directly (involves log2 transformation)
                    # g = log2(S) - log2(S_ntc)
                    # dg/du = x * S' / S → roots when S' = 0
                    # d²g/du² has numerator: S*(S' + x*S'') - x*S'^2
                    # d³g/du³ has numerator: S²*(S' + 3x*S'' + x²*S''') - 3S*x*S'*(S' + x*S'') + 2x²*S'^3

                    def g1_num_u(u):
                        # root of g'(u) is root of Sp (since x>0 and ln2>0)
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        return Sp

                    def g2_num_u(u):
                        # numerator of g''(u):  N2 = S*(Sp + x*Spp) - x*Sp^2
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        return S * (Sp + x * Spp) - x * (Sp ** 2)

                    def g3_num_u(u):
                        # numerator of g'''(u):
                        # N3 = S^2*(Sp + 3x*Spp + x^2*Sppp) - 3S*(x*Sp*(Sp + x*Spp)) + 2*x^2*Sp^3
                        x, S, Sp, Spp, Sppp = self._S_derivs_at_u(
                            u,
                            A=A_i, alpha=alpha_i, Vmax_a=Vmax_a_i, K_a=K_a_i, n_a=n_a_i,
                            beta=beta_i, Vmax_b=Vmax_b_i, K_b=K_b_i, n_b=n_b_i,
                            x_ntc=x_ntc
                        )
                        t1 = (Sp + 3.0 * x * Spp + (x ** 2) * Sppp)
                        t2 = (x * Sp * (Sp + x * Spp))
                        t3 = ((x ** 2) * (Sp ** 3))
                        return (S ** 2) * t1 - 3.0 * S * t2 + 2.0 * t3

                    u_roots_g1 = self._find_roots_on_grid(g1_num_u, u_range_local, include_tangent=True)
                    u_roots_g2 = self._find_roots_on_grid(g2_num_u, u_range_local, include_tangent=True)
                    u_roots_g3 = self._find_roots_on_grid(g3_num_u, u_range_local, include_tangent=True)

            else:
                u_roots_g1 = []
                u_roots_g2 = []
                u_roots_g3 = []
            
            # store alongside x-roots (recommended)
            # These are genuinely dg/du roots in u-units:
            # (use NaN when empty to match your other columns)
            u1_str = ";".join([f"{v:.4f}" for v in u_roots_g1]) if u_roots_g1 else np.nan
            u2_str = ";".join([f"{v:.4f}" for v in u_roots_g2]) if u_roots_g2 else np.nan
            u3_str = ";".join([f"{v:.4f}" for v in u_roots_g3]) if u_roots_g3 else np.nan
            
            # append per-feature strings (make lists outside the loop like you do for x-roots)
            first_deriv_roots_log2fc_mean_list.append(u1_str)
            second_deriv_roots_log2fc_mean_list.append(u2_str)
            third_deriv_roots_log2fc_mean_list.append(u3_str)

            first_deriv_roots_mean_list.append(first_roots_mean)
            second_deriv_roots_mean_list.append(second_roots_mean)
            third_deriv_roots_mean_list.append(third_roots_mean)
            n_first_deriv_roots.append(len(first_roots_mean))
            n_second_deriv_roots.append(len(second_roots_mean))
            n_third_deriv_roots.append(len(third_roots_mean))

            # Classify function
            classification = self._classify_additive_hill(
                alpha_i, alpha_lower[i], alpha_upper[i],
                beta_i, beta_lower[i], beta_upper[i],
                n_a_i, n_a_lower[i], n_a_upper[i],
                n_b_i, n_b_lower[i], n_b_upper[i],
                first_roots_mean, second_roots_mean,
                x_range if x_range is not None else np.array([1.0]),
                Vmax_a_i, K_a_i, Vmax_b_i, K_b_i,
                x_ntc=x_ntc  # Pass x_ntc for non-monotonic classification
            )
            classifications.append(classification)

            # Compute full dynamic range
            # For negbinom: full_log2fc = log2(y_max / y_min)
            # For binomial: full_delta_p = y_max - y_min
            is_flat_i = bool(n_a_zeroed[i]) and bool(n_b_zeroed[i])
            if is_flat_i:
                full_log2fc_i = 0.0
            elif is_binomial:
                # For binomial: compute y_max - y_min
                full_log2fc_i = self._full_delta_p_from_extrema_and_boundaries(
                    A_i, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                    beta_i, Vmax_b_i, K_b_i, n_b_i,
                    x_range if x_range is not None else np.linspace(1e-3, 1e3, 6000)
                )
            else:
                # For negbinom: compute log2(y_max / y_min)
                full_log2fc_i = self._full_log2fc_from_extrema_and_boundaries(
                    A_i, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                    beta_i, Vmax_b_i, K_b_i, n_b_i,
                    x_range if x_range is not None else np.linspace(1e-3, 1e3, 6000)
                )

            full_log2fc_mean_list.append(full_log2fc_i)

            # Compute observed dynamic range over observed x range (min to max x_eff_g)
            # For negbinom: observed_log2fc = log2(y_max / y_min)
            # For binomial: observed_delta_p = y_max - y_min
            if x_obs_min is not None and x_obs_max is not None:
                if is_binomial:
                    obs_log2fc_i, (x_minloc, x_maxloc) = self._compute_observed_delta_p_fitted(
                        A_i, alpha_i, Vmax_a_i, beta_i, Vmax_b_i,
                        K_a_i, n_a_i, K_b_i, n_b_i,
                        x_obs_min, x_obs_max,
                        return_argextrema=True
                    )
                else:
                    obs_log2fc_i, (x_minloc, x_maxloc) = self._compute_observed_log2fc_fitted(
                        A_i, alpha_i, Vmax_a_i, beta_i, Vmax_b_i,
                        K_a_i, n_a_i, K_b_i, n_b_i,
                        x_obs_min, x_obs_max,
                        return_argextrema=True
                    )
                observed_log2fc_list.append(obs_log2fc_i)
            else:
                observed_log2fc_list.append(np.nan)
            
            # Compute CI for observed dynamic range from posterior samples (FAST: no per-sample root finding)
            if (x_obs_min is not None) and (x_obs_max is not None) and (Vmax_a_full.ndim > 1):
                S = Vmax_a_full.shape[0]
                n_samples = min(S, 500)

                # candidates = mean first-derivative roots inside observed range
                x0 = float(max(x_obs_min, 1e-10))
                x1 = float(max(x_obs_max, 1e-10))
                x_candidates_obs = [r for r in first_roots_mean if (x0 <= r <= x1)]
                # add grid extrema locations as candidates (if finite and inside window)
                for xx in (x_minloc, x_maxloc):
                    if np.isfinite(xx) and (x0 <= xx <= x1):
                        x_candidates_obs.append(float(xx))

                # optional: dedup candidates in log2 space
                x_candidates_obs = self._dedup_roots_log2(x_candidates_obs, tol_log2=1e-3)

                is_flat_i = bool(n_a_zeroed[i]) and bool(n_b_zeroed[i])
                if is_flat_i:
                    observed_log2fc_lower_list.append(0.0)
                    observed_log2fc_upper_list.append(0.0)
                else:
                    vals = []
                    for s in range(n_samples):
                        alpha_s = float(alpha_full[s, i]) if alpha_full.ndim > 1 else float(alpha_full[i])
                        beta_s  = float(beta_full[s, i])  if beta_full.ndim > 1 else float(beta_full[i])
                        A_s     = float(A_full[s, i])     if A_full.ndim > 1 else float(A_full[i])

                        Vmax_a_s = float(Vmax_a_full[s, i])
                        Vmax_b_s = float(Vmax_b_full[s, i])
                        K_a_s    = float(K_a_full[s, i])
                        K_b_s    = float(K_b_full[s, i])

                        n_a_s = self._maybe_zero_scalar(n_a_full[s, i], bool(n_a_zeroed[i]))
                        n_b_s = self._maybe_zero_scalar(n_b_full[s, i], bool(n_b_zeroed[i]))

                        if is_binomial:
                            v = self._observed_delta_p_candidates_no_roots(
                                A_s, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                                beta_s, Vmax_b_s, K_b_s, n_b_s,
                                x_obs_min=x0, x_obs_max=x1,
                                x_candidates=x_candidates_obs
                            )
                        else:
                            v = self._observed_log2fc_candidates_no_roots(
                                A_s, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                                beta_s, Vmax_b_s, K_b_s, n_b_s,
                                x_obs_min=x0, x_obs_max=x1,
                                x_candidates=x_candidates_obs
                            )
                        if np.isfinite(v):
                            vals.append(v)

                    if vals:
                        observed_log2fc_lower_list.append(np.quantile(vals, 0.025))
                        observed_log2fc_upper_list.append(np.quantile(vals, 0.975))
                    else:
                        observed_log2fc_lower_list.append(np.nan)
                        observed_log2fc_upper_list.append(np.nan)

            else:
                # no posterior samples or no x-range -> fall back to mean
                observed_log2fc_lower_list.append(obs_log2fc_i)
                observed_log2fc_upper_list.append(obs_log2fc_i)

            # Compute CI for full dynamic range from posterior samples (FAST: no per-sample root finding)
            if Vmax_a_full.ndim > 1:
                S = Vmax_a_full.shape[0]
                n_samples = min(S, 500)  # tune as you like

                # reuse mean-root locations as interior candidates for ALL samples
                # (this is the key speedup)
                x_candidates = first_roots_mean  # already computed above per feature

                sample_log2fcs = []

                # Optional: short-circuit truly flat (both n zeroed by your rule)
                # This matches what you do later for log2fc-derivatives.
                is_flat_i = bool(n_a_zeroed[i]) and bool(n_b_zeroed[i])
                if is_flat_i:
                    full_log2fc_lower_list.append(0.0)
                    full_log2fc_upper_list.append(0.0)
                else:
                    for s in range(n_samples):
                        alpha_s = float(alpha_full[s, i]) if alpha_full.ndim > 1 else float(alpha_full[i])
                        beta_s  = float(beta_full[s, i])  if beta_full.ndim > 1 else float(beta_full[i])
                        A_s     = float(A_full[s, i])     if A_full.ndim > 1 else float(A_full[i])

                        Vmax_a_s = float(Vmax_a_full[s, i])
                        Vmax_b_s = float(Vmax_b_full[s, i])
                        K_a_s    = float(K_a_full[s, i])
                        K_b_s    = float(K_b_full[s, i])

                        # FEATURE-WISE ZEROING RULE (keep exactly your semantics)
                        n_a_s = self._maybe_zero_scalar(n_a_full[s, i], bool(n_a_zeroed[i]))
                        n_b_s = self._maybe_zero_scalar(n_b_full[s, i], bool(n_b_zeroed[i]))

                        if is_binomial:
                            val = self._full_delta_p_candidates_no_roots(
                                A_s, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                                beta_s, Vmax_b_s, K_b_s, n_b_s,
                                x_candidates=x_candidates
                            )
                        else:
                            val = self._full_log2fc_candidates_no_roots(
                                A_s, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                                beta_s, Vmax_b_s, K_b_s, n_b_s,
                                x_candidates=x_candidates
                            )
                        if np.isfinite(val):
                            sample_log2fcs.append(val)

                    if sample_log2fcs:
                        full_log2fc_lower_list.append(np.quantile(sample_log2fcs, 0.025))
                        full_log2fc_upper_list.append(np.quantile(sample_log2fcs, 0.975))
                    else:
                        full_log2fc_lower_list.append(np.nan)
                        full_log2fc_upper_list.append(np.nan)
            else:
                full_log2fc_lower_list.append(full_log2fc_i)
                full_log2fc_upper_list.append(full_log2fc_i)


        # Add computed values to data dict
        data['classification'] = classifications
        data['n_first_deriv_roots'] = n_first_deriv_roots
        data['n_second_deriv_roots'] = n_second_deriv_roots
        data['n_third_deriv_roots'] = n_third_deriv_roots

        # Store derivative roots as strings (list of x values)
        # For mean parameters
        data['first_deriv_roots_mean'] = [
            ';'.join(f'{r:.4f}' for r in roots) if roots else np.nan
            for roots in first_deriv_roots_mean_list
        ]
        
        data['second_deriv_roots_mean'] = [
            ';'.join(f'{r:.4f}' for r in roots) if roots else np.nan
            for roots in second_deriv_roots_mean_list
        ]
        
        data['third_deriv_roots_mean'] = [
            ';'.join(f'{r:.4f}' for r in roots) if roots else np.nan
            for roots in third_deriv_roots_mean_list
        ]

        # Derivative roots in u-space (log2FC x-space)
        # For negbinom: roots of dg/du = 0 where g = log2(y) - log2(y_ntc)
        # For binomial: roots of dp/du = 0 where delta_p = p - p_ntc (same as dy/dx = 0 roots in u-space)
        if is_binomial:
            data['first_deriv_roots_delta_p_mean']  = first_deriv_roots_log2fc_mean_list
            data['second_deriv_roots_delta_p_mean'] = second_deriv_roots_log2fc_mean_list
            data['third_deriv_roots_delta_p_mean']  = third_deriv_roots_log2fc_mean_list
        else:
            data['first_deriv_roots_log2fc_mean']  = first_deriv_roots_log2fc_mean_list
            data['second_deriv_roots_log2fc_mean'] = second_deriv_roots_log2fc_mean_list
            data['third_deriv_roots_log2fc_mean']  = third_deriv_roots_log2fc_mean_list

        # Full dynamic range (theoretical range) and observed dynamic range (observed x range)
        # For negbinom: log2(y_max / y_min)
        # For binomial: y_max - y_min (delta_p)
        if compute_full_log2fc:
            if is_binomial:
                data['full_delta_p_mean'] = full_log2fc_mean_list
                data['full_delta_p_lower'] = full_log2fc_lower_list
                data['full_delta_p_upper'] = full_log2fc_upper_list
            else:
                data['full_log2fc_mean'] = full_log2fc_mean_list
                data['full_log2fc_lower'] = full_log2fc_lower_list
                data['full_log2fc_upper'] = full_log2fc_upper_list

        # Observed dynamic range over the observed x range (min to max x_eff_g)
        if is_binomial:
            data['observed_delta_p'] = observed_log2fc_list
            data['observed_delta_p_lower'] = observed_log2fc_lower_list
            data['observed_delta_p_upper'] = observed_log2fc_upper_list
        else:
            data['observed_log2fc'] = observed_log2fc_list
            data['observed_log2fc_lower'] = observed_log2fc_lower_list
            data['observed_log2fc_upper'] = observed_log2fc_upper_list

        # Compute log2FC versions of parameters relative to NTC
        if compute_log2fc_params and x_ntc is not None and y_ntc is not None:
            epsilon = 1e-12  # Avoid log(0)
            ln2 = np.log(2)

            # Log2FC x-transformation: u = log2(x) - log2(x_ntc)
            log2_x_ntc = np.log2(max(x_ntc, epsilon))

            # =====================================================================
            # Compute log2FC (g), dg/du, d²g/du² at u=0 (x = x_ntc)
            # =====================================================================
            # At u=0, x = x_ntc
            # g(u) = log2(y(x)) - log2(y_ntc)
            #
            # Chain rule derivation:
            # u = log2(x) - log2(x_ntc), so x = x_ntc * 2^u, dx/du = x * ln(2)
            # g = log2(y) - const = ln(y)/ln(2) - const
            # dg/du = (1/ln(2)) * (1/y) * dy/dx * dx/du
            #       = (1/ln(2)) * (1/y) * S'(x) * x * ln(2)
            #       = x * S'(x) / y
            #
            # d²g/du² = d/du [x * S'/S] = dx/du * d/dx [x * S'/S]
            #         = x*ln(2) * [S'/S + x*S''/S - x*(S'/S)²]
            #         = ln(2) * [x*S'/S + x²*S''/S - x²*(S'/S)²]
            #
            # d³g/du³ = ln(2) * (x²*S'''/S - 3x²*S'*S''/S² + 3x*S''/S
            #                    + 2x²*S'³/S³ - 3x*S'²/S² + S'/S)

            log2fc_at_0 = np.full(n_features, np.nan)
            delta_p_at_0 = np.full(n_features, np.nan)  # For binomial: p(x_ntc) - p_ntc
            dg_du_at_0 = np.full(n_features, np.nan)
            d2g_du2_at_0 = np.full(n_features, np.nan)
            d3g_du3_at_0 = np.full(n_features, np.nan)

            # CI arrays
            dg_du_at_0_lower = np.full(n_features, np.nan)
            dg_du_at_0_upper = np.full(n_features, np.nan)
            d2g_du2_at_0_lower = np.full(n_features, np.nan)
            d2g_du2_at_0_upper = np.full(n_features, np.nan)
            d3g_du3_at_0_lower = np.full(n_features, np.nan)
            d3g_du3_at_0_upper = np.full(n_features, np.nan)

            # Check if this is a binomial distribution (for delta_p computation)
            distribution = data.get('distribution', 'negbinom')
            is_binomial = (distribution == 'binomial')

            # Helper to compute all three derivatives at u=0 for given parameters
            # For negbinom: compute dg/du (log2FC derivatives)
            # For binomial: compute dp/du (delta_p derivatives)
            def _compute_derivs_at_u0(alpha_i, beta_i, Vmax_a_i, Vmax_b_i, K_a_i, K_b_i, n_a_i, n_b_i, A_i, y_ntc_i):
                H_a_at_ntc = self._hill_value(x_ntc, Vmax_a_i, K_a_i, n_a_i)
                H_b_at_ntc = self._hill_value(x_ntc, Vmax_b_i, K_b_i, n_b_i)
                y_at_ntc = A_i + alpha_i * H_a_at_ntc + beta_i * H_b_at_ntc

                # For negbinom, we need y_at_ntc > 0 and y_ntc_i > 0
                # For binomial, we only need y_at_ntc to be valid (y_ntc_i can be any probability)
                if not is_binomial and (y_at_ntc <= epsilon or y_ntc_i <= epsilon):
                    return np.nan, np.nan, np.nan

                S_prime = self._additive_hill_first_derivative(
                    x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                    beta_i, Vmax_b_i, K_b_i, n_b_i
                )
                S_double_prime = self._additive_hill_second_derivative(
                    x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                    beta_i, Vmax_b_i, K_b_i, n_b_i
                )
                S_triple_prime = self._additive_hill_third_derivative(
                    x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                    beta_i, Vmax_b_i, K_b_i, n_b_i
                )

                x2 = x_ntc ** 2
                x3 = x_ntc ** 3
                ln2_sq = ln2 ** 2

                if is_binomial:
                    # For binomial: compute dp/du (delta_p derivatives in u-space)
                    # where p = S(x) and u = log2(x) - log2(x_ntc), so dx/du = x * ln(2)
                    #
                    # dp/du = ln(2) · x · S'
                    dp_du = ln2 * x_ntc * S_prime

                    # d²p/du² = (ln2)² · x · (S' + x·S'')
                    d2p_du2 = ln2_sq * x_ntc * (S_prime + x_ntc * S_double_prime)

                    # d³p/du³ = (ln2)³ · x · (S' + 3x·S'' + x²·S''')
                    ln2_cubed = ln2_sq * ln2
                    d3p_du3 = ln2_cubed * x_ntc * (S_prime + 3 * x_ntc * S_double_prime + x2 * S_triple_prime)

                    return dp_du, d2p_du2, d3p_du3
                else:
                    # For negbinom: compute dg/du (log2FC derivatives)
                    # dg/du = x * S'(x) / S(x)
                    dg_du = x_ntc * S_prime / y_at_ntc

                    # d²g/du² = ln(2) * [x*S'/S + x²*S''/S - x²*(S'/S)²]
                    term1 = x_ntc * S_prime / y_at_ntc
                    term2 = x2 * S_double_prime / y_at_ntc
                    term3 = x2 * (S_prime / y_at_ntc) ** 2
                    d2g_du2 = ln2 * (term1 + term2 - term3)

                    # d³g/du³ = (ln(2))² * [x*S'/S + 3x²*S''/S - 3x²*(S'/S)²
                    #                      + x³*S'''/S - 3x³*(S'/S)*(S''/S) + 2x³*(S'/S)³]
                    S_ratio = S_prime / y_at_ntc      # S'/S
                    S2_ratio = S_double_prime / y_at_ntc  # S''/S
                    S3_ratio = S_triple_prime / y_at_ntc  # S'''/S

                    t1 = x_ntc * S_ratio              # x*S'/S
                    t2 = 3 * x2 * S2_ratio            # 3x²*S''/S
                    t3 = -3 * x2 * S_ratio ** 2       # -3x²*(S'/S)²
                    t4 = x3 * S3_ratio                # x³*S'''/S
                    t5 = -3 * x3 * S_ratio * S2_ratio # -3x³*(S'/S)*(S''/S)
                    t6 = 2 * x3 * S_ratio ** 3        # 2x³*(S'/S)³
                    d3g_du3 = ln2_sq * (t1 + t2 + t3 + t4 + t5 + t6)

                    return dg_du, d2g_du2, d3g_du3

            for i in range(n_features):
                is_flat_i = bool(n_a_zeroed[i]) and bool(n_b_zeroed[i])
                # (or stricter: both components inactive by your dep_mask definition)
                
                if is_flat_i:
                    y_ntc_i = y_ntc[i] if y_ntc is not None else np.nan
                    # if n==0 for both, Hill(x_ntc)=0.5*Vmax
                    H_a = 0.5 * Vmax_a_mean[i]
                    H_b = 0.5 * Vmax_b_mean[i]
                    y_at_ntc = A_mean[i] + alpha_mean[i] * H_a + beta_mean[i] * H_b
                    if (y_at_ntc > epsilon) and (y_ntc_i > epsilon):
                        log2fc_at_0[i] = np.log2(y_at_ntc) - np.log2(y_ntc_i)
                    else:
                        log2fc_at_0[i] = np.nan
                    # For binomial: compute delta_p = p(x_ntc) - p_ntc
                    if is_binomial and np.isfinite(y_ntc_i):
                        delta_p_at_0[i] = y_at_ntc - y_ntc_i
                    dg_du_at_0[i] = 0.0
                    d2g_du2_at_0[i] = 0.0
                    d3g_du3_at_0[i] = 0.0
                    dg_du_at_0_lower[i] = 0.0
                    dg_du_at_0_upper[i] = 0.0
                    d2g_du2_at_0_lower[i] = 0.0
                    d2g_du2_at_0_upper[i] = 0.0
                    d3g_du3_at_0_lower[i] = 0.0
                    d3g_du3_at_0_upper[i] = 0.0
                    continue

                # Get mean parameters for this feature
                alpha_i = alpha_mean[i]
                beta_i = beta_mean[i]
                Vmax_a_i = Vmax_a_mean[i]
                Vmax_b_i = Vmax_b_mean[i]
                K_a_i = K_a_mean[i]
                K_b_i = K_b_mean[i]
                n_a_i = float(n_a_used[i])
                n_b_i = float(n_b_used[i])
                A_i = A_mean[i]
                y_ntc_i = y_ntc[i] if y_ntc is not None else 1.0

                # Compute y(x_ntc) using Hill function
                H_a_at_ntc = self._hill_value(x_ntc, Vmax_a_i, K_a_i, n_a_i)
                H_b_at_ntc = self._hill_value(x_ntc, Vmax_b_i, K_b_i, n_b_i)
                y_at_ntc = A_i + alpha_i * H_a_at_ntc + beta_i * H_b_at_ntc

                # log2FC at u=0: g(0) = log2(y(x_ntc)) - log2(y_ntc)
                # For binomial: also compute delta_p = p(x_ntc) - p_ntc
                if is_binomial and np.isfinite(y_ntc_i):
                    delta_p_at_0[i] = y_at_ntc - y_ntc_i

                if y_at_ntc > epsilon and y_ntc_i > epsilon:
                    log2fc_at_0[i] = np.log2(y_at_ntc) - np.log2(y_ntc_i)

                    # Compute mean derivatives
                    dg_du, d2g_du2, d3g_du3 = _compute_derivs_at_u0(
                        alpha_i, beta_i, Vmax_a_i, Vmax_b_i, K_a_i, K_b_i, n_a_i, n_b_i, A_i, y_ntc_i
                    )
                    dg_du_at_0[i] = dg_du
                    d2g_du2_at_0[i] = d2g_du2
                    d3g_du3_at_0[i] = d3g_du3

                    # Compute CI from posterior samples
                    if Vmax_a_full.ndim > 1:
                        n_samples = min(Vmax_a_full.shape[0], 1000)
                        sample_dg_du = []
                        sample_d2g_du2 = []
                        sample_d3g_du3 = []

                        for s in range(n_samples):
                            alpha_s = alpha_full[s, i] if alpha_full.ndim > 1 else alpha_full[i]
                            beta_s = beta_full[s, i] if beta_full.ndim > 1 else beta_full[i]
                            Vmax_a_s = Vmax_a_full[s, i]
                            Vmax_b_s = Vmax_b_full[s, i]
                            K_a_s = K_a_full[s, i]
                            K_b_s = K_b_full[s, i]
                            n_a_s = self._maybe_zero_scalar(n_a_full[s, i], bool(n_a_zeroed[i]))
                            n_b_s = self._maybe_zero_scalar(n_b_full[s, i], bool(n_b_zeroed[i]))
                            A_s = A_full[s, i] if A_full.ndim > 1 else A_full[i]

                            dg, d2g, d3g = _compute_derivs_at_u0(
                                alpha_s, beta_s, Vmax_a_s, Vmax_b_s, K_a_s, K_b_s, n_a_s, n_b_s, A_s, y_ntc_i
                            )
                            if not np.isnan(dg):
                                sample_dg_du.append(dg)
                            if not np.isnan(d2g):
                                sample_d2g_du2.append(d2g)
                            if not np.isnan(d3g):
                                sample_d3g_du3.append(d3g)

                        if sample_dg_du:
                            dg_du_at_0_lower[i] = np.quantile(sample_dg_du, 0.025)
                            dg_du_at_0_upper[i] = np.quantile(sample_dg_du, 0.975)
                        if sample_d2g_du2:
                            d2g_du2_at_0_lower[i] = np.quantile(sample_d2g_du2, 0.025)
                            d2g_du2_at_0_upper[i] = np.quantile(sample_d2g_du2, 0.975)
                        if sample_d3g_du3:
                            d3g_du3_at_0_lower[i] = np.quantile(sample_d3g_du3, 0.025)
                            d3g_du3_at_0_upper[i] = np.quantile(sample_d3g_du3, 0.975)
                    else:
                        # No posterior samples, use mean as lower/upper
                        dg_du_at_0_lower[i] = dg_du
                        dg_du_at_0_upper[i] = dg_du
                        d2g_du2_at_0_lower[i] = d2g_du2
                        d2g_du2_at_0_upper[i] = d2g_du2
                        d3g_du3_at_0_lower[i] = d3g_du3
                        d3g_du3_at_0_upper[i] = d3g_du3

            data['log2fc_at_u0'] = log2fc_at_0
            # For binomial distributions: add delta_p_at_u0 and use dp_du column names
            # For negbinom: use dg_du column names (log2FC derivatives)
            if is_binomial:
                data['delta_p_at_u0'] = delta_p_at_0
                # dp/du derivatives (delta_p space)
                data['dp_du_at_u0'] = dg_du_at_0
                data['dp_du_at_u0_lower'] = dg_du_at_0_lower
                data['dp_du_at_u0_upper'] = dg_du_at_0_upper
                data['d2p_du2_at_u0'] = d2g_du2_at_0
                data['d2p_du2_at_u0_lower'] = d2g_du2_at_0_lower
                data['d2p_du2_at_u0_upper'] = d2g_du2_at_0_upper
                data['d3p_du3_at_u0'] = d3g_du3_at_0
                data['d3p_du3_at_u0_lower'] = d3g_du3_at_0_lower
                data['d3p_du3_at_u0_upper'] = d3g_du3_at_0_upper
            else:
                # dg/du derivatives (log2FC space)
                data['dg_du_at_u0'] = dg_du_at_0
                data['dg_du_at_u0_lower'] = dg_du_at_0_lower
                data['dg_du_at_u0_upper'] = dg_du_at_0_upper
                data['d2g_du2_at_u0'] = d2g_du2_at_0
                data['d2g_du2_at_u0_lower'] = d2g_du2_at_0_lower
                data['d2g_du2_at_u0_upper'] = d2g_du2_at_0_upper
                data['d3g_du3_at_u0'] = d3g_du3_at_0
                data['d3g_du3_at_u0_lower'] = d3g_du3_at_0_lower
                data['d3g_du3_at_u0_upper'] = d3g_du3_at_0_upper

            # EC50 in log2FC space: log2(K) - log2(x_ntc)
            data['EC50_a_log2fc'] = np.log2(np.maximum(K_a_mean, epsilon)) - log2_x_ntc
            data['EC50_b_log2fc'] = np.log2(np.maximum(K_b_mean, epsilon)) - log2_x_ntc

            # Inflection points in log2FC space
            if compute_inflection:
                inflection_a_mean = data.get('inflection_a_mean', np.full(n_features, np.nan))
                inflection_b_mean = data.get('inflection_b_mean', np.full(n_features, np.nan))

                # Handle NaN values properly for log transformation
                inflection_a_log2fc = np.full(n_features, np.nan)
                inflection_b_log2fc = np.full(n_features, np.nan)
                valid_a = ~np.isnan(inflection_a_mean) & (inflection_a_mean > epsilon)
                valid_b = ~np.isnan(inflection_b_mean) & (inflection_b_mean > epsilon)

                if np.any(valid_a):
                    inflection_a_log2fc[valid_a] = np.log2(inflection_a_mean[valid_a]) - log2_x_ntc
                if np.any(valid_b):
                    inflection_b_log2fc[valid_b] = np.log2(inflection_b_mean[valid_b]) - log2_x_ntc

                # Inflection points in log2FC space: log2(inflection) - log2(x_ntc)
                def _log2fc_transform(arr):
                    out = np.full(n_features, np.nan)
                    ok = np.isfinite(arr) & (arr > epsilon)
                    out[ok] = np.log2(arr[ok]) - log2_x_ntc
                    return out
                
                inf_a_mean = data.get('inflection_a_mean', np.full(n_features, np.nan))
                inf_a_lo   = data.get('inflection_a_lower', np.full(n_features, np.nan))
                inf_a_hi   = data.get('inflection_a_upper', np.full(n_features, np.nan))
                
                inf_b_mean = data.get('inflection_b_mean', np.full(n_features, np.nan))
                inf_b_lo   = data.get('inflection_b_lower', np.full(n_features, np.nan))
                inf_b_hi   = data.get('inflection_b_upper', np.full(n_features, np.nan))
                
                data['inflection_a_log2fc_mean']  = _log2fc_transform(inf_a_mean)
                data['inflection_a_log2fc_lower'] = _log2fc_transform(inf_a_lo)
                data['inflection_a_log2fc_upper'] = _log2fc_transform(inf_a_hi)
                
                data['inflection_b_log2fc_mean']  = _log2fc_transform(inf_b_mean)
                data['inflection_b_log2fc_lower'] = _log2fc_transform(inf_b_lo)
                data['inflection_b_log2fc_upper'] = _log2fc_transform(inf_b_hi)
                            
            # A (baseline) in transformed y-space
            # For negbinom: A_log2fc = log2(A) - log2(y_ntc)
            # For binomial: A_delta_p = A - y_ntc
            if is_binomial:
                A_delta_p = A_mean - y_ntc
                data['A_delta_p'] = A_delta_p
            else:
                A_log2fc = np.full(n_features, np.nan)
                valid_A = (A_mean > epsilon) & (y_ntc > epsilon)
                if np.any(valid_A):
                    A_log2fc[valid_A] = np.log2(A_mean[valid_A]) - np.log2(y_ntc[valid_A])
                data['A_log2fc'] = A_log2fc

        return data

    def _add_single_hill_params(
        self, data, posterior, n_features,
        compute_inflection: bool = True,
        compute_full_log2fc: bool = True,
        compute_derivative_roots: bool = True,
        x_range: Optional[np.ndarray] = None,
        x_obs_min: Optional[float] = None,
        x_obs_max: Optional[float] = None,
        compute_log2fc_params: bool = False,
        x_ntc=None,
        y_ntc=None,
    ):
        """
        Add single Hill parameters to data dict.

        Single Hill model: y = A + alpha * Vmax * x^n / (K^n + x^n)

        Parameters stored (with _mean, _lower, _upper suffixes):
        - Vmax_a: Hill amplitude (maximum effect)
        - K_a: Half-max concentration (EC50/IC50)
        - n_a: Hill coefficient (cooperativity, sign determines direction)
        - alpha: Weight/scale factor
        - A: Baseline offset

        Also computes:
        - inflection_a: Inflection point (exists when |n| > 1)
        - full_log2fc: log2(y_max / y_min) over theoretical range
        - observed_log2fc: log2(y_max / y_min) over observed x range
        - Derivative roots (first, second, third) where dy/dx=0, d²y/dx²=0, d³y/dx³=0
        - Log2FC transforms if x_ntc and y_ntc are provided
        """
        # --- shared extraction helpers ---
        def extract_param(name):
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            # squeeze [S,1,T] -> [S,T]
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            # average over extra dims if needed (e.g. multinomial categories)
            if param.ndim > 2:
                param = param.mean(axis=tuple(range(2, param.ndim)))
    
            if param.ndim == 2:  # [S, T]
                mean = param.mean(axis=0)
                lo = np.quantile(param, 0.025, axis=0)
                hi = np.quantile(param, 0.975, axis=0)
            else:                # [T] point
                mean = lo = hi = param
            return mean, lo, hi
    
        def extract_param_full(name):
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            if param.ndim > 2:
                param = param.mean(axis=tuple(range(2, param.ndim)))
            return param  # [S,T] or [T]

        def extract_param_abs(name):
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            if param.ndim > 2:
                param = param.mean(axis=tuple(range(2, param.ndim)))
        
            param_abs = np.abs(param)
            if param_abs.ndim == 2:
                mean = param_abs.mean(axis=0)
                lo = np.quantile(param_abs, 0.025, axis=0)
                hi = np.quantile(param_abs, 0.975, axis=0)
            else:
                mean = lo = hi = param_abs
            return mean, lo, hi
    
        def _broadcast(arr):
            arr = np.atleast_1d(arr)
            if arr.size == 1:
                return np.full(n_features, arr.item())
            return arr
    
        # --- core params ---
        Vmax_mean, Vmax_lo, Vmax_hi = extract_param('Vmax_a')
        K_mean,    K_lo,    K_hi    = extract_param('K_a')
        n_mean,    n_lo,    n_hi    = extract_param('n_a')

        # APPLY SAME RULE AS ADDITIVE:
        n_used, n_zeroed = self._n_effective_from_ci(n_mean, n_lo, n_hi)
        # use n_used for all downstream computations
    
        Vmax_full = extract_param_full('Vmax_a')
        K_full    = extract_param_full('K_a')
        n_full    = extract_param_full('n_a')
    
        if 'alpha' in posterior:
            alpha_mean, alpha_lo, alpha_hi = extract_param('alpha')
            alpha_full = extract_param_full('alpha')
            alpha_mean = _broadcast(alpha_mean); alpha_lo = _broadcast(alpha_lo); alpha_hi = _broadcast(alpha_hi)
            if alpha_full.ndim == 1:
                alpha_full = alpha_full.reshape(-1, 1)
            if alpha_full.shape[-1] == 1:
                alpha_full = np.broadcast_to(alpha_full, (alpha_full.shape[0], n_features)).copy()
        else:
            alpha_mean = np.ones(n_features); alpha_lo = np.ones(n_features); alpha_hi = np.ones(n_features)
            alpha_full = np.ones((1, n_features))
    
        if 'A' in posterior:
            A_mean, A_lo, A_hi = extract_param('A')
            A_full = extract_param_full('A')
            A_mean = _broadcast(A_mean); A_lo = _broadcast(A_lo); A_hi = _broadcast(A_hi)
            if A_full.ndim == 1:
                A_full = A_full.reshape(-1, 1)
            if A_full.shape[-1] == 1:
                A_full = np.broadcast_to(A_full, (A_full.shape[0], n_features)).copy()
        else:
            A_mean = np.zeros(n_features); A_lo = np.zeros(n_features); A_hi = np.zeros(n_features)
            A_full = np.zeros((1, n_features))

        
        data['Vmax_a_mean'] = Vmax_mean
        data['Vmax_a_lower'] = Vmax_lo
        data['Vmax_a_upper'] = Vmax_hi
        data['K_a_mean'] = K_mean
        data['K_a_lower'] = K_lo
        data['K_a_upper'] = K_hi
        data['n_a_mean'] = n_mean
        data['n_a_lower'] = n_lo
        data['n_a_upper'] = n_hi
        data['alpha_mean'] = alpha_mean
        data['alpha_lower'] = alpha_lo
        data['alpha_upper'] = alpha_hi
        data['A_mean'] = A_mean
        data['A_lower'] = A_lo
        data['A_upper'] = A_hi
    
        # --- inflection of the Hill component ---
        if compute_inflection:
            n_abs_mean, n_abs_lo, n_abs_hi = extract_param_abs('n_a')
            
            infl_mean = self._compute_hill_inflection(n_abs_mean, K_mean)
            infl_lo   = self._compute_hill_inflection(n_abs_lo,   K_lo)
            infl_hi   = self._compute_hill_inflection(n_abs_hi,   K_hi)
            
            # (optional) store |n| summaries too
            data['n_a_abs_mean'] = n_abs_mean
            data['n_a_abs_lower'] = n_abs_lo
            data['n_a_abs_upper'] = n_abs_hi
            data['inflection_a_mean'] = infl_mean
            data['inflection_a_lower'] = infl_lo
            data['inflection_a_upper'] = infl_hi
    
        # --- full log2FC range for y = A + alpha*Hill ---
        if compute_full_log2fc:
            eps = 1e-10
        
            # mean version using n_used, with correct n==0 limits (Hill == 0.5*V everywhere)
            n_u = np.asarray(n_used, dtype=float)

            hill0 = np.where(
                np.abs(n_u) < 1e-15,
                0.5 * Vmax_mean,
                np.where(n_u < 0, Vmax_mean, 0.0)
            )
            hillinf = np.where(
                np.abs(n_u) < 1e-15,
                0.5 * Vmax_mean,
                np.where(n_u < 0, 0.0, Vmax_mean)
            )

            y0 = np.maximum(A_mean + alpha_mean * hill0, eps)
            yinf = np.maximum(A_mean + alpha_mean * hillinf, eps)

            data['full_log2fc_mean'] = np.log2(np.maximum(y0, yinf) / np.minimum(y0, yinf))
        
            # CI from posterior samples (coherent)
            if Vmax_full.ndim == 2:
                S = Vmax_full.shape[0]
                nS = min(S, 500)
                lo = np.full(n_features, np.nan)
                hi = np.full(n_features, np.nan)
        
                for i in range(n_features):
                    vals = []
                    for s in range(nS):
                        V_s = float(Vmax_full[s, i])
                        K_s = float(K_full[s, i])
                        a_s = float(alpha_full[s, i]) if alpha_full.ndim == 2 else float(alpha_full[i])
                        # A isn't extracted full in your current function; easiest is to add A_full extraction.
                        # For now, if A was point/broadcast: use A_mean[i]
                        A_s = float(A_full[s, i]) if A_full.ndim == 2 else float(A_full[i])
        
                        n_s = self._maybe_zero_scalar(n_full[s, i], bool(n_zeroed[i]))
        
                        # boundaries for single hill
                        if abs(n_s) < 1e-15:
                            # constant hill = 0.5*V everywhere -> no boundary range
                            y0_s = max(A_s + a_s * 0.5 * V_s, eps)
                            yinf_s = y0_s
                        else:
                            h0 = V_s if n_s < 0 else 0.0
                            hinf = 0.0 if n_s < 0 else V_s
                            y0_s = max(A_s + a_s * h0, eps)
                            yinf_s = max(A_s + a_s * hinf, eps)
        
                        vals.append(np.log2(max(y0_s, yinf_s) / min(y0_s, yinf_s)))
        
                    if vals:
                        lo[i] = np.quantile(vals, 0.025)
                        hi[i] = np.quantile(vals, 0.975)
        
                data['full_log2fc_lower'] = lo
                data['full_log2fc_upper'] = hi
            else:
                data['full_log2fc_lower'] = data['full_log2fc_mean']
                data['full_log2fc_upper'] = data['full_log2fc_mean']

    
        # --- observed log2FC over observed x-range (optional; uses fitted function) ---
        if (x_obs_min is not None) and (x_obs_max is not None) and np.isfinite(x_obs_min) and np.isfinite(x_obs_max):
            obs = []
            for i in range(n_features):
                obs.append(
                    self._compute_observed_log2fc_fitted(
                        A_mean[i], alpha_mean[i], Vmax_mean[i], 0.0, 0.0,
                        K_mean[i], n_used[i], 1.0, 1.0,
                        float(x_obs_min), float(x_obs_max)
                    )
                )
            data['observed_log2fc'] = obs

        if (x_obs_min is not None) and (x_obs_max is not None) and (Vmax_full.ndim == 2):
            eps = 1e-10
            x0 = float(max(x_obs_min, eps))
            x1 = float(max(x_obs_max, eps))
        
            lo = np.full(n_features, np.nan)
            hi = np.full(n_features, np.nan)
        
            S = Vmax_full.shape[0]
            nS = min(S, 500)
        
            for i in range(n_features):
                if bool(n_zeroed[i]):
                    lo[i] = 0.0
                    hi[i] = 0.0
                    continue
        
                vals = []
                for s in range(nS):
                    V_s = float(Vmax_full[s, i])
                    K_s = float(K_full[s, i])
                    a_s = float(alpha_full[s, i])
                    A_s = float(A_full[s, i])
        
                    n_s = self._maybe_zero_scalar(n_full[s, i], bool(n_zeroed[i]))
        
                    def y_at(x):
                        H = self._hill_value(x, V_s, K_s, n_s)
                        return A_s + a_s * H
        
                    y0 = max(y_at(x0), eps)
                    y1 = max(y_at(x1), eps)
                    vals.append(np.log2(max(y0, y1) / min(y0, y1)))
        
                lo[i] = np.quantile(vals, 0.025)
                hi[i] = np.quantile(vals, 0.975)
        
            data['observed_log2fc_lower'] = lo
            data['observed_log2fc_upper'] = hi
        else:
            data['observed_log2fc_lower'] = data.get('observed_log2fc', np.full(n_features, np.nan))
            data['observed_log2fc_upper'] = data.get('observed_log2fc', np.full(n_features, np.nan))
    
        # --- derivative roots over x_range ---
        if compute_derivative_roots and (x_range is not None):
            first_roots = []
            second_roots = []
            third_roots = []
            n_first = []
            n_second = []
            n_third = []
    
            for i in range(n_features):
                r1, r2, r3 = self._single_hill_derivative_roots(
                    alpha=float(alpha_mean[i]),
                    Vmax=float(Vmax_mean[i]),
                    K=float(K_mean[i]),
                    n=float(n_used[i]),
                    x_range=x_range,
                    compute_third=True
                )
                first_roots.append(r1); second_roots.append(r2); third_roots.append(r3)
                n_first.append(len(r1)); n_second.append(len(r2)); n_third.append(len(r3))
    
            data['n_first_deriv_roots'] = n_first
            data['n_second_deriv_roots'] = n_second
            data['n_third_deriv_roots'] = n_third
            data['first_deriv_roots_mean'] = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in first_roots]
            data['second_deriv_roots_mean'] = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in second_roots]
            data['third_deriv_roots_mean']  = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in third_roots]
    
        # --- log2FC transforms (unchanged from your version) ---
        if compute_log2fc_params and (x_ntc is not None) and (y_ntc is not None):
            eps = 1e-10
            log2_x_ntc = np.log2(max(float(x_ntc), eps))
            data['K_a_log2fc'] = np.log2(np.maximum(K_mean, eps)) - log2_x_ntc
    
            if compute_inflection:
                eps = 1e-10
                infl_mean = data.get('inflection_a_mean', np.full(n_features, np.nan))
                infl_lo   = data.get('inflection_a_lower', np.full(n_features, np.nan))
                infl_hi   = data.get('inflection_a_upper', np.full(n_features, np.nan))
            
                def _log2fc(arr):
                    out = np.full(n_features, np.nan)
                    ok = np.isfinite(arr) & (arr > eps)
                    out[ok] = np.log2(arr[ok]) - log2_x_ntc
                    return out
            
                data['inflection_a_log2fc_mean']  = _log2fc(infl_mean)
                data['inflection_a_log2fc_lower'] = _log2fc(infl_lo)
                data['inflection_a_log2fc_upper'] = _log2fc(infl_hi)
            
                # Backwards compat
                data['inflection_a_log2fc'] = data['inflection_a_log2fc_mean']
    
            y_ntc_arr = np.asarray(y_ntc, dtype=float)
            outA = np.full(n_features, np.nan)
            okA = (A_mean > eps) & (y_ntc_arr > eps)
            outA[okA] = np.log2(A_mean[okA]) - np.log2(y_ntc_arr[okA])
            data['A_log2fc'] = outA
    
            # Also roots in log2FC x-space, if present:
            if compute_derivative_roots and (x_range is not None) and ('first_deriv_roots_mean' in data):
                first_log = []
                second_log = []
                third_log = []
                for i in range(n_features):
                    # Parse from stored strings to avoid recomputing
                    def _parse(s):
                        """Parse root string, handling np.nan and empty strings."""
                        if s is None or (isinstance(s, float) and np.isnan(s)):
                            return []
                        if isinstance(s, str) and s.strip() == '':
                            return []
                        return [float(x) for x in str(s).split(';') if x != '']

                    r1 = _parse(data['first_deriv_roots_mean'][i])
                    r2 = _parse(data['second_deriv_roots_mean'][i])
                    r3 = _parse(data['third_deriv_roots_mean'][i])

                    # Use np.nan for empty roots (consistent with additive_hill)
                    first_log.append(';'.join([f'{(np.log2(max(r, eps)) - log2_x_ntc):.4f}' for r in r1]) if r1 else np.nan)
                    second_log.append(';'.join([f'{(np.log2(max(r, eps)) - log2_x_ntc):.4f}' for r in r2]) if r2 else np.nan)
                    third_log.append(';'.join([f'{(np.log2(max(r, eps)) - log2_x_ntc):.4f}' for r in r3]) if r3 else np.nan)
                data['first_deriv_roots_log2fc_mean'] = first_log
                data['second_deriv_roots_log2fc_mean'] = second_log
                data['third_deriv_roots_log2fc_mean'] = third_log
    
        return data

    def _add_polynomial_params(
        self,
        data,
        posterior,
        n_features,
        compute_full_log2fc: bool = True,
        compute_derivative_roots: bool = True,
        x_range: Optional[np.ndarray] = None,
        x_obs_min: Optional[float] = None,
        x_obs_max: Optional[float] = None
    ):
        """
        Add polynomial parameters to data dict.

        IMPORTANT: The polynomial operates in different spaces depending on distribution:
        - negbinom: log2(y) = log2(A) + alpha * poly(log2(x))  [log2 space for x]
        - normal/studentt: y = A + alpha * poly(x)  [linear space]
        - binomial: logit(p) = logit(A) + alpha * poly(x)  [linear x, logit y]

        For negbinom, the polynomial coefficients multiply powers of log2(x), so:
        - x_range should be in LINEAR x-space (this method converts internally)
        - Derivative roots are found in log2(x) space, then converted back
        - full_log2fc represents the range of log2(y) values

        For normal/studentt/binomial, coefficients multiply powers of linear x.

        Parameters stored (with _mean, _lower, _upper suffixes):
        - coef_{i}: Coefficient for the i-th power term
        - For negbinom: coef_i multiplies log2(x)^i
        - For others: coef_i multiplies x^i

        Also computes:
        - full_log2fc: Dynamic range in appropriate space
        - observed_log2fc: Same as full_log2fc for polynomial
        - Derivative roots in the native polynomial space
        - Log2FC transforms of roots if x_ntc is available
        """
        coefs = posterior['poly_coefs']  # [S, T, d+1] or [T, d+1]
        if isinstance(coefs, torch.Tensor):
            coefs = coefs.cpu().numpy()
    
        degree = coefs.shape[-1] - 1
    
        if coefs.ndim == 3:
            # summary per coefficient
            for i in range(degree + 1):
                data[f'coef_{i}_mean']  = coefs[:, :, i].mean(axis=0)
                data[f'coef_{i}_lower'] = np.quantile(coefs[:, :, i], 0.025, axis=0)
                data[f'coef_{i}_upper'] = np.quantile(coefs[:, :, i], 0.975, axis=0)
            coefs_mean = coefs.mean(axis=0)  # [T, d+1]
        else:
            for i in range(degree + 1):
                data[f'coef_{i}_mean']  = coefs[:, i]
                data[f'coef_{i}_lower'] = coefs[:, i]
                data[f'coef_{i}_upper'] = coefs[:, i]
            coefs_mean = coefs  # [T, d+1]
    
        # Determine if polynomial is in log2 space (negbinom) or linear space (others)
        distribution = data.get('distribution', 'negbinom')
        is_log2_space = (distribution == 'negbinom')

        # full_log2fc and observed_log2fc: evaluate over observed x-range
        # For negbinom: poly(log2(x)), evaluate on log2 grid
        # For others: poly(x), evaluate on linear grid
        if compute_full_log2fc:
            if (x_obs_min is not None) and (x_obs_max is not None) and np.isfinite(x_obs_min) and np.isfinite(x_obs_max):
                x_min = float(x_obs_min)
                x_max = float(x_obs_max)
            else:
                # fallback: if no x info, pick a generic range
                x_min, x_max = 1.0, 2.0

            eps = 1e-10
            n_eval = 1000

            if is_log2_space:
                # For negbinom: polynomial is poly(log2(x))
                # Create grid in log2 space, evaluate polynomial there
                log2_min = np.log2(max(x_min, eps))
                log2_max = np.log2(max(x_max, eps))
                u_eval = np.linspace(log2_min, log2_max, n_eval)  # u = log2(x)
            else:
                # For normal/studentt/binomial: polynomial is poly(x)
                x_eval = np.linspace(x_min, x_max, n_eval)

            full_mean = []
            observed_log2fc = []
            for t in range(n_features):
                if is_log2_space:
                    # poly(log2(x)) gives log2(y) - log2(A) (scaled by alpha)
                    # The polynomial output IS the log2FC contribution
                    poly_vals = self._poly_eval(u_eval, coefs_mean[t, :])
                    # Dynamic range is max - min of poly values (already in log2 space)
                    poly_range = np.nanmax(poly_vals) - np.nanmin(poly_vals)
                    full_mean.append(abs(poly_range) if np.isfinite(poly_range) else np.nan)
                    observed_log2fc.append(abs(poly_range) if np.isfinite(poly_range) else np.nan)
                else:
                    # poly(x) gives y directly (for normal/studentt) or logit contribution (binomial)
                    y_eval = self._poly_eval(x_eval, coefs_mean[t, :])
                    y_min_val = np.nanmin(y_eval)
                    y_max_val = np.nanmax(y_eval)

                    # Log2 ratio only valid if both values are positive
                    if y_min_val > eps and y_max_val > eps:
                        full_mean.append(np.log2(y_max_val / y_min_val))
                        observed_log2fc.append(np.log2(y_max_val / y_min_val))
                    else:
                        # Fall back to NaN if y values are non-positive
                        full_mean.append(np.nan)
                        observed_log2fc.append(np.nan)

            data['full_log2fc_mean'] = full_mean
            data['full_log2fc_lower'] = full_mean
            data['full_log2fc_upper'] = full_mean
            data['observed_log2fc'] = observed_log2fc
            data['observed_log2fc_lower'] = observed_log2fc
            data['observed_log2fc_upper'] = observed_log2fc

        # derivative roots over x_range
        # For negbinom: find roots in log2(x) space, convert back to linear x
        # For others: find roots in linear x space directly
        if compute_derivative_roots and (x_range is not None):
            first_roots = []
            second_roots = []
            third_roots = []
            n_first = []
            n_second = []
            n_third = []

            eps = 1e-10

            if is_log2_space:
                # Convert x_range to log2 space for root finding
                x_range_positive = x_range[x_range > eps]
                if len(x_range_positive) > 0:
                    u_range = np.log2(x_range_positive)
                else:
                    u_range = np.linspace(-5, 5, 1000)  # fallback

            for t in range(n_features):
                if is_log2_space:
                    # Find roots in u = log2(x) space
                    r1_u, r2_u, r3_u = self._poly_derivative_roots(coefs_mean[t, :], u_range, compute_third=True)
                    # Convert roots back to linear x space: x = 2^u
                    r1 = [2**u for u in r1_u] if r1_u else []
                    r2 = [2**u for u in r2_u] if r2_u else []
                    r3 = [2**u for u in r3_u] if r3_u else []
                else:
                    # Find roots directly in linear x space
                    r1, r2, r3 = self._poly_derivative_roots(coefs_mean[t, :], x_range, compute_third=True)

                first_roots.append(r1)
                second_roots.append(r2)
                third_roots.append(r3)
                n_first.append(len(r1))
                n_second.append(len(r2))
                n_third.append(len(r3))

            data['n_first_deriv_roots'] = n_first
            data['n_second_deriv_roots'] = n_second
            data['n_third_deriv_roots'] = n_third
            # Roots are now in linear x space for all distributions
            data['first_deriv_roots_mean'] = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in first_roots]
            data['second_deriv_roots_mean'] = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in second_roots]
            data['third_deriv_roots_mean'] = [';'.join([f'{r:.4f}' for r in rr]) if rr else np.nan for rr in third_roots]

            if 'x_ntc' in data and data['x_ntc'] is not None and np.isfinite(data['x_ntc']):
                log2_x_ntc = np.log2(max(float(data['x_ntc']), eps))

                def _to_log2fc_str(root_str):
                    """Convert x-space root string to log2FC space (u = log2(x) - log2(x_ntc))."""
                    if root_str is None or (isinstance(root_str, float) and np.isnan(root_str)):
                        return np.nan
                    if isinstance(root_str, str) and root_str.strip() == '':
                        return np.nan
                    roots = [float(r) for r in str(root_str).split(';')]
                    u = [np.log2(max(r, eps)) - log2_x_ntc for r in roots]
                    return ';'.join([f'{val:.4f}' for val in u]) if u else np.nan

                data['first_deriv_roots_log2fc_mean'] = [_to_log2fc_str(s) for s in data['first_deriv_roots_mean']]
                data['second_deriv_roots_log2fc_mean'] = [_to_log2fc_str(s) for s in data['second_deriv_roots_mean']]
                data['third_deriv_roots_log2fc_mean'] = [_to_log2fc_str(s) for s in data['third_deriv_roots_mean']]

    
        return data


    def _compute_hill_inflection(self, n, K, epsilon=1e-12):
        """
        Inflection point of Hill: Vmax * x^n / (K^n + x^n) (+ A)
    
        Exists iff |n| > 1.
    
        x_inflection = K * ((|n| - 1)/(|n| + 1))^(1/|n|)
        """
        n = np.asarray(n, dtype=float)
        K = np.asarray(K, dtype=float)
    
        nabs = np.abs(n)
        out = np.full(np.broadcast(n, K).shape, np.nan, dtype=float)
    
        valid = (nabs > 1) & np.isfinite(nabs) & np.isfinite(K) & (K > epsilon)
        if np.any(valid):
            ratio = (nabs[valid] - 1.0) / (nabs[valid] + 1.0)   # in (0,1)
            out[valid] = K[valid] * (ratio ** (1.0 / nabs[valid]))
        return out

    # ========================================================================
    # Derivative and Root Finding Methods (for Additive Hill)
    # ========================================================================

    def _exp_clip(self, z):
        # double precision exp overflow ~ 709; underflow ~ -745 is safe to clip
        return np.exp(np.clip(z, -745.0, 709.0))
    
    def _hill_first_derivative(self, x, Vmax, K, n, epsilon=1e-12):
        """
        d/dx [V * x^n / (K^n + x^n)] = (K^n * V * n * x^(n-1)) / (K^n + x^n)^2
        Stable for tiny x and/or negative n.
        """
        n = float(n)
        if abs(n) < 1e-15:
            return np.zeros_like(np.asarray(x, dtype=float))

        x = np.asarray(x, dtype=float)
        K = float(K); Vmax = float(Vmax); n = float(n)
    
        x_safe = np.maximum(x, epsilon)
        K_safe = max(K, epsilon)
    
        logx = np.log(x_safe)
        logK = np.log(K_safe)
    
        K_n   = self._exp_clip(n * logK)
        x_n   = self._exp_clip(n * logx)
        x_nm1 = self._exp_clip((n - 1.0) * logx)
    
        denom = (K_n + x_n) ** 2
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            out = (K_n * Vmax * n * x_nm1) / denom
    
        return np.where(np.isfinite(out), out, np.nan)
    
    def _hill_second_derivative(self, x, Vmax, K, n, epsilon=1e-12):
        """
        d²/dx² [V * x^n / (K^n + x^n)]
        = -(K^n * V * n * x^(n-2) * ((n+1)*x^n - K^n*(n-1))) / (K^n + x^n)^3
        Stable for tiny x and/or negative n.
        """
        n = float(n)
        if abs(n) < 1e-15:
            return np.zeros_like(np.asarray(x, dtype=float))

        x = np.asarray(x, dtype=float)
        K = float(K); Vmax = float(Vmax); n = float(n)
    
        x_safe = np.maximum(x, epsilon)
        K_safe = max(K, epsilon)
    
        logx = np.log(x_safe)
        logK = np.log(K_safe)
    
        K_n   = self._exp_clip(n * logK)
        x_n   = self._exp_clip(n * logx)
        x_nm2 = self._exp_clip((n - 2.0) * logx)
    
        denom = (K_n + x_n) ** 3
        inner = (n + 1.0) * x_n - K_n * (n - 1.0)
    
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            out = -(K_n * Vmax * n * x_nm2 * inner) / denom
    
        return np.where(np.isfinite(out), out, np.nan)

    def _additive_hill_first_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                         beta, Vmax_b, K_b, n_b, epsilon=1e-12):
        """
        First derivative of additive Hill: dy/dx = alpha * dHill_a/dx + beta * dHill_b/dx
        """
        d_a = self._hill_first_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d_b = self._hill_first_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d_a + beta * d_b

    def _additive_hill_second_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                          beta, Vmax_b, K_b, n_b, epsilon=1e-6):
        """
        Second derivative of additive Hill: d²y/dx² = alpha * d²Hill_a/dx² + beta * d²Hill_b/dx²
        """
        d2_a = self._hill_second_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d2_b = self._hill_second_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d2_a + beta * d2_b

    def _hill_third_derivative(self, x, Vmax, K, n, epsilon=1e-12):
        """
        Numerically-stable third derivative of:
            f(x) = V * x^n / (K^n + x^n)
    
        Uses log-space exponentiation to avoid overflow/underflow.
        Returns NaN where the expression is numerically non-finite.
        """
        n = float(n)
        if abs(n) < 1e-15:
            return np.zeros_like(np.asarray(x, dtype=float))

        x = np.asarray(x, dtype=float)
        K = float(K)
        Vmax = float(Vmax)
        n = float(n)
    
        # Keep positivity, but use a *much smaller* epsilon so you don't flatten a wide region near 0.
        x_safe = np.maximum(x, epsilon)
        K_safe = max(K, epsilon)
    
        logx = np.log(x_safe)
        logK = np.log(K_safe)
    
        # helper: exp with clipping to avoid overflow
        # double precision exp overflows around ~709
        def exp_clip(z):
            return np.exp(np.clip(z, -745.0, 709.0))
    
        # powers in log-space
        x_n   = exp_clip(n * logx)
        x_2n  = exp_clip((2.0 * n) * logx)
        x_nm3 = exp_clip((n - 3.0) * logx)
    
        K_n   = exp_clip(n * logK)
        K_2n  = exp_clip((2.0 * n) * logK)
    
        denom = (K_n + x_n) ** 4
    
        term1 = (n**2 + 3.0*n + 2.0) * x_2n
        term2 = 4.0 * K_n * (1.0 - n**2) * x_n
        term3 = K_2n * (n**2 - 3.0*n + 2.0)
    
        numer = K_n * Vmax * n * x_nm3 * (term1 + term2 + term3)
    
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            out = numer / denom
    
        # Do NOT turn inf into 0; that creates fake roots.
        out = np.where(np.isfinite(out), out, np.nan)
        return out

    def _additive_hill_third_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                         beta, Vmax_b, K_b, n_b, epsilon=1e-12):
        """
        Third derivative of additive Hill: d³y/dx³ = alpha * d³Hill_a/dx³ + beta * d³Hill_b/dx³
        """
        d3_a = self._hill_third_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d3_b = self._hill_third_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d3_a + beta * d3_b

    def _dedup_roots_log2(self, roots, tol_log2=1e-3):
        """
        Deduplicate roots by proximity in log2(x).
        tol_log2=1e-3 means ~0.1% multiplicative tolerance.
        """
        if not roots:
            return []
        roots = np.array([r for r in roots if np.isfinite(r) and r > 0], dtype=float)
        if roots.size == 0:
            return []
        roots.sort()
        out = [roots[0]]
        last = roots[0]
        for r in roots[1:]:
            if abs(np.log2(r) - np.log2(last)) > tol_log2:
                out.append(r)
                last = r
        return out
        
    def _find_roots_empirical( 
        self, 
        func, 
        x_range: np.ndarray, 
        tol_log2_dedup: float = 1e-3, 
        # baseline “zero” settings 
        zero_atol: float = 1e-12, 
        zero_rtol: float = 1e-9, 
        # new: noise floor multiplier (helps for flat curves) 
        noise_mult: float = 5.0, 
        # new: require endpoints to exceed thr_hi (hysteresis) 
        hysteresis: float = 2.0, 
        # new: validate found root by probing around it in log2 space 
        validate: bool = True, 
        validate_dlog2: float = 2e-3, # ~0.14% multiplicative 
    ): 
        y_vals = np.asarray(func(x_range), dtype=float) 
        x_vals = np.asarray(x_range, dtype=float) 
        
        finite = np.isfinite(y_vals) & np.isfinite(x_vals) 
        if finite.sum() < 2: 
            return [] 
        
        y = y_vals[finite] 
        x = x_vals[finite] 
        
        # global scale 
        scale = np.nanmax(np.abs(y)) 
        if not np.isfinite(scale) or scale == 0.0: 
            return [] 
        
        # estimate a numerical noise floor from local variation 
        # (MAD of first differences is a decent proxy for “flat + jitter”) 
        dy = np.diff(y) 
        dy = dy[np.isfinite(dy)] 
        if dy.size > 0:
            mad = np.median(np.abs(dy - np.median(dy))) 
            noise = 1.4826 * mad 
        else: 
            noise = 0.0 
        
        thr = max(zero_atol, zero_rtol * scale, noise_mult * noise)
        thr_hi = hysteresis * thr 
        
        # If everything is within the deadband, treat as “no roots” 
        if np.all(np.abs(y) <= thr): 
            return [] 
        
        # deadband snap
        y0 = y.copy() 
        y0[np.abs(y0) <= thr] = 0.0 
        s = np.sign(y0) 
        
        roots = [] 
        
        # helper for validation 
        def _ok_root(r): 
            if not validate: 
                return True 
            if (not np.isfinite(r)) or (r <= 0): 
                return False 
            
            # probe multiplicatively around r: r * 2^(±dlog2) 
            rL = r * (2.0 ** (-validate_dlog2)) 
            rR = r * (2.0 ** ( validate_dlog2)) 
            
            try: 
                f0 = float(np.asarray(func(np.array([r]))).ravel()[0]) 
                fL = float(np.asarray(func(np.array([rL]))).ravel()[0]) 
                fR = float(np.asarray(func(np.array([rR]))).ravel()[0]) 
            except Exception: 
                return False 
            
            if not (np.isfinite(f0) and np.isfinite(fL) and np.isfinite(fR)): 
                return False 
            
            # require a *real* crossing: opposite signs on the two sides, 
            # and at least one side has magnitude above the stronger threshold 
            if fL == 0.0 or fR == 0.0: 
                return False 
            if (fL * fR) >= 0.0: 
                return False 
            if (abs(fL) < thr_hi) and (abs(fR) < thr_hi): 
                return False 
            
            # avoid “flat root” where everything is ~0
            if abs(f0) < thr and (abs(fL) < thr_hi) and (abs(fR) < thr_hi):
                return False 
            
            return True

        for i in range(len(s) - 1): 
            # skip if either endpoint is inside the deadband (prevents “zero-touch” roots) 
            if s[i] == 0 or s[i + 1] == 0: 
                continue 
            
            # hysteresis: insist endpoints are not just barely outside thr 
            if (abs(y[i]) < thr_hi) or (abs(y[i + 1]) < thr_hi): 
                continue 
            
            if s[i] != s[i + 1]: 
                xL = float(x[i]) 
                xR = float(x[i + 1]) 
                
                try: 
                    root = brentq(lambda z: float(np.asarray(func(np.array([z]))).ravel()[0]), xL, xR) 
                except Exception: 
                    # fallback linear interpolation 
                    yL = float(y[i]) 
                    yR = float(y[i + 1]) 
                    t = abs(yL) / (abs(yL) + abs(yR)) 
                    root = float(xL + t * (xR - xL)) 
                    
                if _ok_root(root): 
                    roots.append(float(root)) 
                
        return self._dedup_roots_log2(roots, tol_log2=tol_log2_dedup)

    def _find_roots_empirical_loose(
        self,
        func,
        x_range: np.ndarray,
        tol_log2_dedup: float = 1e-3,
        zero_atol: float = 1e-12,
        zero_rtol: float = 1e-9,
        include_near_zero_minima: bool = True,   # capture “touching” roots
    ):
        """
        Permissive empirical root finder:
          - finds sign-change roots (brentq)
          - optionally also returns local minima of |f| that are very close to zero
            (captures tangent roots where sign doesn't flip)
        """
        x = np.asarray(x_range, dtype=float)
        y = np.asarray(func(x), dtype=float)
    
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            return []
    
        x = x[finite]
        y = y[finite]
    
        scale = np.nanmax(np.abs(y))
        if not np.isfinite(scale) or scale == 0.0:
            return []
    
        thr = max(zero_atol, zero_rtol * scale)
    
        # ---------- 1) sign-change roots (DO NOT skip near-zero endpoints) ----------
        # snap tiny values to 0 only for sign bookkeeping
        y_snap = y.copy()
        y_snap[np.abs(y_snap) <= thr] = 0.0
        s = np.sign(y_snap)
    
        roots = []
    
        # to avoid missing crossings through long near-zero stretches, hop between
        # the nearest non-zero sign points.
        nz = np.where(s != 0)[0]
        if nz.size >= 2:
            for a, b in zip(nz[:-1], nz[1:]):
                if s[a] == s[b]:
                    continue
                xL, xR = float(x[a]), float(x[b])
                try:
                    root = brentq(lambda z: float(np.asarray(func(np.array([z]))).ravel()[0]), xL, xR)
                except Exception:
                    # fallback: linear interpolation
                    yL = float(y[a]); yR = float(y[b])
                    t = abs(yL) / (abs(yL) + abs(yR))
                    root = xL + t * (xR - xL)
                if np.isfinite(root) and root > 0:
                    roots.append(float(root))
    
        # ---------- 2) tangent “touching” roots (local minima of |y| near 0) ----------
        if include_near_zero_minima and y.size >= 3:
            ay = np.abs(y)
            # local minima indices of |y|
            mins = np.where((ay[1:-1] <= ay[:-2]) & (ay[1:-1] <= ay[2:]))[0] + 1
            for i in mins:
                if ay[i] <= thr:
                    roots.append(float(x[i]))
    
        return self._dedup_roots_log2(roots, tol_log2=tol_log2_dedup)

    
    def _classify_additive_hill(self, alpha, alpha_lower, alpha_upper,
                                 beta, beta_lower, beta_upper,
                                 n_a, n_a_lower, n_a_upper,
                                 n_b, n_b_lower, n_b_upper,
                                 first_deriv_roots: List[float],
                                 second_deriv_roots: List[float],
                                 x_range: np.ndarray,
                                 Vmax_a, K_a, Vmax_b, K_b,
                                 x_ntc: float = None) -> str:
        """
        Classify additive Hill function into categories based on dependency masks.

        Classification Logic:
        - dep_mask_a: Component A is active (0 NOT in CI of alpha AND 0 NOT in CI of n_a)
        - dep_mask_b: Component B is active (0 NOT in CI of beta AND 0 NOT in CI of n_b)
        - dep_mask: Either component is active

        Categories:
        - 'single_positive': Only one component active, with positive n (increasing)
            Formula: (~(dep_mask_a & dep_mask_b) & dep_mask) & (n >= 0 for active component)
        - 'single_negative': Only one component active, with negative n (decreasing)
            Formula: (~(dep_mask_a & dep_mask_b) & dep_mask) & (n <= 0 for active component)
        - 'additive_positive': Both components active, both with positive n
            Formula: (dep_mask_a & dep_mask_b) & (n_a >= 0) & (n_b >= 0)
        - 'additive_negative': Both components active, both with negative n
            Formula: (dep_mask_a & dep_mask_b) & (n_a <= 0) & (n_b <= 0)
        - 'non_monotonic_min': Both active, opposite signs, local minimum at nearest extremum to x_ntc
            (where S'=0, checked via S'' > 0 at that point)
        - 'non_monotonic_max': Both active, opposite signs, local maximum at nearest extremum to x_ntc
            (where S'=0, checked via S'' < 0 at that point)
        """
        # Define dependency masks
        # Component is "active" if its weight is non-zero AND its Hill coefficient is non-zero
        # Use CI to determine if 0 is in the credible interval
        alpha_active = not (alpha_lower <= 0 <= alpha_upper)  # 0 NOT in CI
        n_a_active = not (n_a_lower <= 0 <= n_a_upper)  # n_a is not ~0

        beta_active = not (beta_lower <= 0 <= beta_upper)  # 0 NOT in CI
        n_b_active = not (n_b_lower <= 0 <= n_b_upper)  # n_b is not ~0

        # Dependency masks
        dep_mask_a = alpha_active and n_a_active
        dep_mask_b = beta_active and n_b_active
        dep_mask = dep_mask_a or dep_mask_b  # At least one component active

        # Check if both components are active
        both_active = dep_mask_a and dep_mask_b
        single_active = dep_mask and not both_active

        # Determine signs of Hill coefficients for active components
        # Use mean values for classification
        if single_active:
            # Only one component is active
            if dep_mask_a:
                # Only component A is active
                if n_a >= 0:
                    return 'single_positive'
                else:
                    return 'single_negative'
            else:
                # Only component B is active
                if n_b >= 0:
                    return 'single_positive'
                else:
                    return 'single_negative'

        if both_active:
            # Both components are active - check if same or opposite signs
            same_sign = (n_a * n_b) >= 0  # Both positive or both negative

            if same_sign:
                # Additive: both increasing or both decreasing
                if n_a >= 0 and n_b >= 0:
                    return 'additive_positive'
                else:
                    return 'additive_negative'
            else:
                # Non-monotonic: opposite signs of n
                # Determine min vs max based on second derivative at the extremum
                # (where first derivative = 0)
                # S''(x_extremum) > 0 → concave up → local minimum
                # S''(x_extremum) < 0 → concave down → local maximum
                #
                # If multiple extrema, pick the one closest to x_ntc (u=0)

                if len(first_deriv_roots) > 0:
                    # Find the extremum closest to x_ntc
                    if x_ntc is not None and x_ntc > 0:
                        # Find root closest to x_ntc
                        distances = [abs(np.log2(r) - np.log2(x_ntc)) for r in first_deriv_roots if r > 0]
                        if distances:
                            closest_idx = np.argmin(distances)
                            x_extremum = first_deriv_roots[closest_idx]
                        else:
                            x_extremum = first_deriv_roots[0]
                    else:
                        x_extremum = first_deriv_roots[0]

                    # Check curvature at the extremum
                    d2_extremum = self._additive_hill_second_derivative(
                        x_extremum, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b
                    )
                    if d2_extremum > 0:
                        return 'non_monotonic_min'  # Concave up at extremum → local min
                    else:
                        return 'non_monotonic_max'  # Concave down at extremum → local max

                return 'non_monotonic'  # Generic non-monotonic fallback (no roots found)

        # No active components or indeterminate
        return 'flat'

    def _compute_full_log2fc_additive(self, A, alpha, Vmax_a, beta, Vmax_b,
                                       x_range: np.ndarray,
                                       K_a, n_a, K_b, n_b,
                                       classification: str,
                                       first_deriv_roots: List[float] = None) -> float:
        """
        Compute full log2FC for additive Hill based on classification.

        Returns log2(y_max / y_min) - the dynamic range in log2 space.

        Hill function limits depend on sign of n:
        - n > 0: Hill(0) = 0, Hill(∞) = Vmax
        - n < 0: Hill(0) = Vmax, Hill(∞) = 0
        """
        epsilon = 1e-12

        # Compute boundary values based on signs of n
        # At x→0: Hill = Vmax if n<0, else 0
        # At x→∞: Hill = 0 if n<0, else Vmax
        hill_a_at_0 = Vmax_a if n_a < 0 else 0
        hill_a_at_inf = 0 if n_a < 0 else Vmax_a
        hill_b_at_0 = Vmax_b if n_b < 0 else 0
        hill_b_at_inf = 0 if n_b < 0 else Vmax_b

        y_at_0 = A + alpha * hill_a_at_0 + beta * hill_b_at_0
        y_at_inf = A + alpha * hill_a_at_inf + beta * hill_b_at_inf

        # Ensure positive values for log
        y_at_0 = max(y_at_0, epsilon)
        y_at_inf = max(y_at_inf, epsilon)

        if classification in ['monotonic_positive', 'monotonic_negative',
                               'single_positive', 'single_negative',
                               'additive_positive', 'additive_negative',
                               'single_hill_a', 'single_hill_b']:
            # For monotonic, log2FC is log2(y_max / y_min)
            y_max = max(y_at_0, y_at_inf)
            y_min = min(y_at_0, y_at_inf)
            return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

        # For non-monotonic, find interior extrema
        if first_deriv_roots is None:
            def first_deriv(x):
                return self._additive_hill_first_derivative(
                    x, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b
                )
            first_deriv_roots = self._find_roots_empirical(first_deriv, x_range)

        if len(first_deriv_roots) == 0:
            # No interior extremum, use boundary difference
            y_max = max(y_at_0, y_at_inf)
            y_min = min(y_at_0, y_at_inf)
            return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

        # Evaluate y at the extrema
        def hill_func(x):
            x_safe = max(x, epsilon)
            K_a_safe = max(K_a, epsilon)
            K_b_safe = max(K_b, epsilon)

            # Handle potential numerical issues with negative n
            if n_a >= 0:
                H_a = Vmax_a * (x_safe ** n_a) / (K_a_safe ** n_a + x_safe ** n_a + epsilon)
            else:
                # For n < 0: x^n = 1/x^|n|, need careful handling
                x_n_a = x_safe ** n_a
                K_n_a = K_a_safe ** n_a
                H_a = Vmax_a * x_n_a / (K_n_a + x_n_a + epsilon)

            if n_b >= 0:
                H_b = Vmax_b * (x_safe ** n_b) / (K_b_safe ** n_b + x_safe ** n_b + epsilon)
            else:
                x_n_b = x_safe ** n_b
                K_n_b = K_b_safe ** n_b
                H_b = Vmax_b * x_n_b / (K_n_b + x_n_b + epsilon)

            return A + alpha * H_a + beta * H_b

        y_extrema = [hill_func(r) for r in first_deriv_roots]

        # Find the largest range (total dynamic range) in log2 space
        all_y = [y_at_0, y_at_inf] + y_extrema
        y_max = max(all_y)
        y_min = min(all_y)
        return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

    def _compute_observed_log2fc_fitted(
        self, A, alpha, Vmax_a, beta, Vmax_b,
        K_a, n_a, K_b, n_b,
        x_obs_min: float, x_obs_max: float,
        return_argextrema: bool = False
    ):
        eps = 1e-10
    
        n_points = 1000
        log2_min = np.log2(max(x_obs_min, eps))
        log2_max = np.log2(max(x_obs_max, eps))
        x_eval = 2.0 ** np.linspace(log2_min, log2_max, n_points)
    
        K_a_safe = max(K_a, eps)
        K_b_safe = max(K_b, eps)
    
        # Hill A/B in log-space
        x_n_a = self._exp_clip(n_a * np.log(x_eval))
        K_n_a = self._exp_clip(n_a * np.log(K_a_safe))
        H_a = Vmax_a * x_n_a / (K_n_a + x_n_a)
    
        x_n_b = self._exp_clip(n_b * np.log(x_eval))
        K_n_b = self._exp_clip(n_b * np.log(K_b_safe))
        H_b = Vmax_b * x_n_b / (K_n_b + x_n_b)
    
        y_vals = A + alpha * H_a + beta * H_b
    
        # IMPORTANT: don't overwrite invalids with A (that can create fake extrema)
        y_vals = np.asarray(y_vals, dtype=float)
        y_vals[~np.isfinite(y_vals)] = np.nan
    
        if np.all(np.isnan(y_vals)):
            if return_argextrema:
                return np.nan, (np.nan, np.nan)
            return np.nan
    
        # positivity for log-ratio
        y_clip = np.maximum(y_vals, eps)
    
        imax = int(np.nanargmax(y_clip))
        imin = int(np.nanargmin(y_clip))
        y_max = float(y_clip[imax])
        y_min = float(y_clip[imin])
    
        out = float(np.log2(y_max / y_min))
        if return_argextrema:
            return out, (float(x_eval[imin]), float(x_eval[imax]))
        return out

    def _compute_observed_delta_p_fitted(
        self, A, alpha, Vmax_a, beta, Vmax_b,
        K_a, n_a, K_b, n_b,
        x_obs_min: float, x_obs_max: float,
        return_argextrema: bool = False
    ):
        """
        Compute observed delta_p (y_max - y_min) over the observed x range.
        For binomial distributions where delta_p = p - p_ntc.
        """
        eps = 1e-10

        n_points = 1000
        log2_min = np.log2(max(x_obs_min, eps))
        log2_max = np.log2(max(x_obs_max, eps))
        x_eval = 2.0 ** np.linspace(log2_min, log2_max, n_points)

        K_a_safe = max(K_a, eps)
        K_b_safe = max(K_b, eps)

        # Hill A/B in log-space
        x_n_a = self._exp_clip(n_a * np.log(x_eval))
        K_n_a = self._exp_clip(n_a * np.log(K_a_safe))
        H_a = Vmax_a * x_n_a / (K_n_a + x_n_a)

        x_n_b = self._exp_clip(n_b * np.log(x_eval))
        K_n_b = self._exp_clip(n_b * np.log(K_b_safe))
        H_b = Vmax_b * x_n_b / (K_n_b + x_n_b)

        y_vals = A + alpha * H_a + beta * H_b

        # IMPORTANT: don't overwrite invalids with A (that can create fake extrema)
        y_vals = np.asarray(y_vals, dtype=float)
        y_vals[~np.isfinite(y_vals)] = np.nan

        if np.all(np.isnan(y_vals)):
            if return_argextrema:
                return np.nan, (np.nan, np.nan)
            return np.nan

        # For delta_p, we use the raw values (no positivity constraint needed)
        imax = int(np.nanargmax(y_vals))
        imin = int(np.nanargmin(y_vals))
        y_max = float(y_vals[imax])
        y_min = float(y_vals[imin])

        out = float(y_max - y_min)
        if return_argextrema:
            return out, (float(x_eval[imin]), float(x_eval[imax]))
        return out
