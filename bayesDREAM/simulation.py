"""
Simulate negative binomial count data from fitted trans summary parameters.

Given a trans summary CSV (from save_trans_summary), cell metadata, and per-cell
x_true values, this module reconstructs the dose-response and samples NegBin
observations to produce a synthetic count matrix suitable for re-fitting with
bayesDREAM.

Two parameterizations are supported:

Direct parameterization (A_median column present):
    Uses Hill/polynomial parameters from the trans summary directly.

Fold-change parameterization (y_ntc_median column present):
    Derives A and V from interpretable quantities — NTC expression (y_ntc),
    NTC x_true (x_ntc), K_log2FC, and full_log2FC — for each Hill component.
    Null genes are indicated by n=0 or full_log2FC=0.

IMPORTANT: Sum factors used for simulation are NOT the same as sum factors for
downstream fitting. After simulation, recalculate sum factors from the simulated
counts (e.g., via scran::calculateSumFactors) before fitting bayesDREAM.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union


def _hill(x, Vmax, K, n, eps=1e-12):
    """Evaluate Hill function: Vmax * x^n / (K^n + x^n)."""
    x_safe = np.maximum(x, eps)
    K_safe = np.maximum(K, eps)
    x_n = np.power(x_safe, n)
    K_n = np.power(K_safe, n)
    return Vmax * x_n / (K_n + x_n + eps)


def _compute_AV_from_fc(n, y_ntc, x_ntc, K_log2FC, full_log2FC, eps=1e-12):
    """Compute A, V, K from the fold-change parameterization.

    Given y(x) = A + V * x^n / (x^n + K^n) with y_ntc = y(x_ntc):
        K    = x_ntc * 2^K_log2FC
        s_0  = 1 / (1 + 2^(n * K_log2FC))
        F_eff = sign(n) * full_log2FC
        A    = y_ntc / (1 + (2^F_eff - 1) * s_0)
        V    = A * (2^F_eff - 1)

    Null genes (n == 0 or full_log2FC == 0) return A = y_ntc, V = 0.
    Non-finite K_log2FC is replaced with 0 (K = x_ntc) to avoid NaN propagation
    for null genes where K is irrelevant.
    """
    n = np.asarray(n, dtype=float)
    y_ntc = np.asarray(y_ntc, dtype=float)
    x_ntc = np.asarray(x_ntc, dtype=float)
    K_log2FC = np.asarray(K_log2FC, dtype=float)
    full_log2FC = np.asarray(full_log2FC, dtype=float)

    null_mask = (n == 0) | (full_log2FC == 0)

    # Replace non-finite K_log2FC and zero n to avoid NaN in masked genes
    K_log2FC_safe = np.where(np.isfinite(K_log2FC), K_log2FC, 0.0)
    n_safe = np.where(null_mask, 1.0, n)

    K = x_ntc * np.power(2.0, K_log2FC_safe)
    s_0 = 1.0 / (1.0 + np.power(2.0, n_safe * K_log2FC_safe))

    F_eff = np.sign(n) * full_log2FC
    scale = np.power(2.0, F_eff) - 1.0

    A_active = y_ntc / (1.0 + scale * s_0 + eps)
    V_active = A_active * scale

    A = np.where(null_mask, y_ntc, A_active)
    V = np.where(null_mask, 0.0, V_active)
    K = np.where(null_mask, x_ntc, K)

    return A, V, K


def simulate_from_trans_summary(
    trans_summary_df: pd.DataFrame,
    meta: pd.DataFrame,
    x_true: Union[np.ndarray, pd.Series],
    x_counts: Union[np.ndarray, pd.Series],
    cis_gene: str,
    sim_sum_factor: Union[np.ndarray, pd.Series, float] = 1.0,
    genes: Optional[list] = None,
    group_col: str = 'technical_group_code',
    seed: Optional[int] = None,
    fdr_threshold: Optional[float] = 0.05,
) -> pd.DataFrame:
    """
    Simulate NegBin count data from fitted trans summary parameters.

    The generative model is:
        y_pred  = A + V_a * Hill(x; K_a, n_a) [+ V_b * Hill(x; K_b, n_b)]
        mu      = y_pred * alpha_y[group] * sim_sum_factor
        y_obs  ~ NegBin(total_count=phi_y, prob=phi_y / (phi_y + mu))

    Parameterization is detected automatically from which columns are present in
    ``trans_summary_df`` (see below). All rows must share the same ``function_type``.

    Parameters
    ----------
    trans_summary_df : pd.DataFrame
        One row per gene. The following columns are recognised:

        **Always required**

        - ``feature``: gene name.
        - ``function_type``: one of ``'single_hill'``, ``'additive_hill'``,
          ``'polynomial'``.
        - Dispersion — one of (first found is used, the other is ignored):
          ``phi_y_median`` or ``o_y_median``.

        **Parameterization detection** (mutually exclusive; ``y_ntc_median``
        takes priority if both are present):

        *Direct parameterization* — present when ``A_median`` is a column.
        A, Vmax, and K come directly from posterior summary statistics.

        *Fold-change parameterization* — present when ``y_ntc_median`` is a
        column. A and V are derived per-gene from interpretable quantities
        (NTC expression, NTC x_true, K expressed as a log2 fold-change over
        x_ntc, and the full asymptotic log2 fold-change). See
        ``_compute_AV_from_fc`` for the exact formulae.

        **Direct parameterization columns by function_type**

        ``single_hill``:

        - Required: ``A_median``, ``Vmax_a_median``, ``K_a_median``,
          ``n_a_median``.
        - Optional: ``alpha_median`` (default 1.0).
        - Optional FDR gating: ``fdr_alpha`` — if ``fdr_alpha > fdr_threshold``,
          alpha is zeroed (component treated as absent).

        ``additive_hill``:

        - Required: ``A_median``, ``Vmax_a_median``, ``K_a_median``,
          ``n_a_median``, ``Vmax_b_median``, ``K_b_median``, ``n_b_median``.
        - Optional: ``alpha_median`` (default 1.0), ``beta_median`` (default 1.0).
        - Optional FDR gating: ``fdr_alpha``, ``fdr_beta`` — components whose
          FDR exceeds ``fdr_threshold`` have their mixing weight zeroed.

        ``polynomial``:

        - Required: ``A_median``, and one or more columns of the form
          ``coef_<i>_median`` (e.g. ``coef_0_median``, ``coef_1_median``, …),
          detected automatically by name.

        **Fold-change parameterization columns by function_type**

        ``single_hill``:

        - Required: ``y_ntc_median``, ``x_ntc_median``, ``n_a_median``,
          ``K_log2FC_a_median``, ``full_log2FC_a_median``.
        - Null genes: rows where ``n_a == 0`` or ``full_log2FC_a == 0`` (after
          FDR gating) simulate as a flat NegBin draw at baseline ``A = y_ntc``.
          ``K_log2FC_a`` may be NaN for null genes.
        - Optional FDR gating: ``fdr_alpha`` — if ``fdr_alpha > fdr_threshold``,
          ``full_log2FC_a`` is set to 0, making the gene null.

        ``additive_hill``:

        - Required: ``y_ntc_median``, ``x_ntc_median``, ``n_a_median``,
          ``K_log2FC_a_median``, ``full_log2FC_a_median``, ``n_b_median``,
          ``K_log2FC_b_median``, ``full_log2FC_b_median``.
        - Each Hill component is parameterized independently using the same
          ``y_ntc`` / ``x_ntc`` reference. A and V are computed separately for
          each component, then combined: ``A = A_a + A_b``.
        - Null component logic: a component is null if its ``n == 0`` or
          ``full_log2FC == 0`` (after FDR gating); null components contribute 0
          to A and 0 to y_pred. If both components are null, ``A = y_ntc``.
          ``K_log2FC`` may be NaN for null components.
        - Optional FDR gating: ``fdr_alpha``, ``fdr_beta``.

        ``polynomial``:

        - Not supported with fold-change parameterization (raise ValueError).
          Use direct parameterization instead.

        **Optional columns (all parameterizations)**

        - ``group_<g>_alpha_y_mean`` (e.g. ``group_0_alpha_y_mean``,
          ``group_1_alpha_y_mean``, …): per-group technical multiplicative
          corrections. Detected automatically by name. If absent, or if
          ``group_col`` is not in ``meta``, no group correction is applied.

    meta : pd.DataFrame
        Cell metadata. Required columns: ``cell``, ``guide``, ``target``.
        If ``group_<g>_alpha_y_mean`` columns are present in
        ``trans_summary_df``, must also contain ``group_col``.
    x_true : array-like, shape (n_cells,)
        Per-cell x_true values (cis gene expression), aligned to ``meta`` rows.
    x_counts : array-like, shape (n_cells,)
        Per-cell raw cis gene counts, included unchanged as the cis gene row
        in the output matrix.
    cis_gene : str
        Name of the cis gene (added as a row in the output count matrix).
    sim_sum_factor : array-like or float, optional
        Per-cell sum factors for simulation (default: 1.0). Controls the count
        scale of simulated data. For simulated data to align visually with the
        reference curve in ``plot_xy_data``, this must equal the per-cell sum
        factor column that ``plot_xy_data`` uses for correction (typically the
        column passed to ``fit_trans``, e.g. ``'sum_factor_new'``)::

            sim_sum_factor = model.get_modality(model.primary_modality).sum_factors['sum_factor_new'].values

        **This is not the sum factor for downstream fitting.** After simulation,
        recalculate sum factors from the simulated counts (e.g. via
        ``scran::calculateSumFactors``) before re-fitting bayesDREAM.
    genes : list of str, optional
        Subset of genes to simulate. Default: all genes in ``trans_summary_df``.
    group_col : str, optional
        Column in ``meta`` containing integer technical group codes
        (default: ``'technical_group_code'``).
    seed : int, optional
        Random seed for reproducibility.
    fdr_threshold : float or None, optional
        FDR threshold for gating Hill components (default: 0.05). Components
        with FDR above this value are treated as absent (alpha/beta zeroed in
        the direct parameterization; full_log2FC zeroed in the fold-change
        parameterization). Set to ``None`` to use raw posterior means for all
        components regardless of significance.

    Returns
    -------
    counts_df : pd.DataFrame
        Simulated count matrix with genes as rows and cells as columns.
        Includes both trans genes (NegBin draws) and the cis gene (raw counts).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x_true = np.asarray(x_true, dtype=float)
    x_counts = np.asarray(x_counts, dtype=float)
    n_cells = len(x_true)

    if len(meta) != n_cells:
        raise ValueError(
            f"meta has {len(meta)} rows but x_true has {n_cells} elements"
        )

    _sim_sum_factor_is_scalar = np.isscalar(sim_sum_factor)
    if _sim_sum_factor_is_scalar:
        sum_factors = np.full(n_cells, float(sim_sum_factor))
    else:
        sum_factors = np.asarray(sim_sum_factor, dtype=float)
        if len(sum_factors) != n_cells:
            raise ValueError(
                f"sim_sum_factor has {len(sum_factors)} elements but expected {n_cells}"
            )

    df = trans_summary_df.copy()
    if genes is not None:
        df = df[df['feature'].isin(genes)].reset_index(drop=True)
    gene_names = df['feature'].values
    n_genes = len(gene_names)

    if n_genes == 0:
        raise ValueError("No genes to simulate after filtering.")

    function_type = df['function_type'].iloc[0]

    # --- Dispersion: accept phi_y_median or o_y_median ---
    if 'phi_y_median' in df.columns:
        phi_y = df['phi_y_median'].values
    elif 'o_y_median' in df.columns:
        phi_y = df['o_y_median'].values
    else:
        raise ValueError(
            "trans_summary_df must contain 'phi_y_median' or 'o_y_median'"
        )

    # --- Detect parameterization ---
    use_fc_params = 'y_ntc_median' in df.columns
    if not use_fc_params and 'A_median' not in df.columns:
        raise ValueError(
            "trans_summary_df must contain either 'A_median' (direct parameterization) "
            "or 'y_ntc_median' (fold-change parameterization)"
        )

    x_exp = x_true[np.newaxis, :]  # (1, n_cells) for broadcasting

    # ------------------------------------------------------------------ #
    # Compute y_pred  (n_genes, n_cells)                                   #
    # ------------------------------------------------------------------ #

    if use_fc_params:
        # ---- Fold-change parameterization ----
        for col in ['y_ntc_median', 'x_ntc_median']:
            if col not in df.columns:
                raise ValueError(f"Missing required column for fc parameterization: {col}")

        y_ntc = df['y_ntc_median'].values
        x_ntc = df['x_ntc_median'].values

        if function_type in ('single_hill', 'additive_hill'):
            for col in ['n_a_median', 'K_log2FC_a_median', 'full_log2FC_a_median']:
                if col not in df.columns:
                    raise ValueError(f"Missing required column for fc parameterization: {col}")

            n_a = df['n_a_median'].values
            K_log2FC_a = df['K_log2FC_a_median'].values
            full_log2FC_a = df['full_log2FC_a_median'].values.copy()

            # FDR gating: zero out component a if not significant
            if fdr_threshold is not None and 'fdr_alpha' in df.columns:
                fdr_a = df['fdr_alpha'].values
                full_log2FC_a = np.where(
                    np.isfinite(fdr_a) & (fdr_a <= fdr_threshold), full_log2FC_a, 0.0
                )

            A_a, V_a, K_a = _compute_AV_from_fc(n_a, y_ntc, x_ntc, K_log2FC_a, full_log2FC_a)

            if function_type == 'additive_hill':
                for col in ['n_b_median', 'K_log2FC_b_median', 'full_log2FC_b_median']:
                    if col not in df.columns:
                        raise ValueError(
                            f"Missing required column for additive_hill fc parameterization: {col}"
                        )

                n_b = df['n_b_median'].values
                K_log2FC_b = df['K_log2FC_b_median'].values
                full_log2FC_b = df['full_log2FC_b_median'].values.copy()

                # FDR gating: zero out component b if not significant
                if fdr_threshold is not None and 'fdr_beta' in df.columns:
                    fdr_b = df['fdr_beta'].values
                    full_log2FC_b = np.where(
                        np.isfinite(fdr_b) & (fdr_b <= fdr_threshold), full_log2FC_b, 0.0
                    )

                A_b, V_b, K_b = _compute_AV_from_fc(n_b, y_ntc, x_ntc, K_log2FC_b, full_log2FC_b)

                # Each null component contributes 0 to A; if both null, fall back to y_ntc
                null_a = (n_a == 0) | (full_log2FC_a == 0)
                null_b = (n_b == 0) | (full_log2FC_b == 0)
                A_a_contrib = np.where(null_a, 0.0, A_a)
                A_b_contrib = np.where(null_b, 0.0, A_b)
                A = np.where(null_a & null_b, y_ntc, A_a_contrib + A_b_contrib)
            else:
                A = A_a

            hill_a = _hill(x_exp, 1.0, K_a[:, np.newaxis], n_a[:, np.newaxis])
            y_pred = A[:, np.newaxis] + V_a[:, np.newaxis] * hill_a

            if function_type == 'additive_hill':
                hill_b = _hill(x_exp, 1.0, K_b[:, np.newaxis], n_b[:, np.newaxis])
                y_pred = y_pred + V_b[:, np.newaxis] * hill_b

        elif function_type == 'polynomial':
            raise ValueError(
                "Fold-change parameterization is not supported for function_type='polynomial'. "
                "Use the direct parameterization (A_median column) instead."
            )
        else:
            raise ValueError(f"Unsupported function_type: {function_type}")

    else:
        # ---- Direct parameterization ----
        A = df['A_median'].values

        if function_type == 'additive_hill':
            for col in ['Vmax_a_median', 'K_a_median', 'n_a_median',
                         'Vmax_b_median', 'K_b_median', 'n_b_median']:
                if col not in df.columns:
                    raise ValueError(f"Missing required column for additive_hill: {col}")

            Vmax_a = df['Vmax_a_median'].values
            K_a = df['K_a_median'].values
            n_a = df['n_a_median'].values
            alpha = df['alpha_median'].values.copy() if 'alpha_median' in df.columns else np.ones(n_genes)
            Vmax_b = df['Vmax_b_median'].values
            K_b = df['K_b_median'].values
            n_b = df['n_b_median'].values
            beta = df['beta_median'].values.copy() if 'beta_median' in df.columns else np.ones(n_genes)

            # FDR gating: without gating, null alpha/beta ~ 0.4-0.5 (RelaxedBernoulli prior)
            # rather than 0, adding spurious Hill contributions.
            if fdr_threshold is not None:
                if 'fdr_alpha' in df.columns:
                    fdr_a = df['fdr_alpha'].values
                    alpha = np.where(np.isfinite(fdr_a) & (fdr_a <= fdr_threshold), alpha, 0.0)
                if 'fdr_beta' in df.columns:
                    fdr_b = df['fdr_beta'].values
                    beta = np.where(np.isfinite(fdr_b) & (fdr_b <= fdr_threshold), beta, 0.0)

            hill_a = _hill(x_exp, Vmax_a[:, np.newaxis], K_a[:, np.newaxis], n_a[:, np.newaxis])
            hill_b = _hill(x_exp, Vmax_b[:, np.newaxis], K_b[:, np.newaxis], n_b[:, np.newaxis])

            y_pred = (A[:, np.newaxis]
                      + alpha[:, np.newaxis] * hill_a
                      + beta[:, np.newaxis] * hill_b)

        elif function_type == 'single_hill':
            for col in ['Vmax_a_median', 'K_a_median', 'n_a_median']:
                if col not in df.columns:
                    raise ValueError(f"Missing required column for single_hill: {col}")

            Vmax_a = df['Vmax_a_median'].values
            K_a = df['K_a_median'].values
            n_a = df['n_a_median'].values
            alpha = df['alpha_median'].values.copy() if 'alpha_median' in df.columns else np.ones(n_genes)

            if fdr_threshold is not None and 'fdr_alpha' in df.columns:
                fdr_a = df['fdr_alpha'].values
                alpha = np.where(np.isfinite(fdr_a) & (fdr_a <= fdr_threshold), alpha, 0.0)

            hill_a = _hill(x_exp, Vmax_a[:, np.newaxis], K_a[:, np.newaxis], n_a[:, np.newaxis])
            y_pred = A[:, np.newaxis] + alpha[:, np.newaxis] * hill_a

        elif function_type == 'polynomial':
            coef_cols = sorted(
                [c for c in df.columns if c.startswith('coef_') and c.endswith('_median')],
                key=lambda c: int(c.split('_')[1])
            )
            if not coef_cols:
                raise ValueError("No polynomial coefficient columns found.")

            y_pred = np.zeros((n_genes, n_cells))
            for i, col in enumerate(coef_cols):
                y_pred += df[col].values[:, np.newaxis] * np.power(x_exp, i)

        else:
            raise ValueError(f"Unsupported function_type: {function_type}")

    # ------------------------------------------------------------------ #
    # Apply technical effects and sum factors                               #
    # ------------------------------------------------------------------ #
    y_pred = np.maximum(y_pred, 1e-6)

    alpha_y_cols = sorted(
        [c for c in df.columns if c.startswith('group_') and c.endswith('_alpha_y_mean')],
        key=lambda c: int(c.split('_')[1])
    )

    if _sim_sum_factor_is_scalar and float(sim_sum_factor) == 1.0:
        warnings.warn(
            "sim_sum_factor=1.0 (default scalar). For simulated data to visually "
            "align with the reference curve in plot_xy_data, sim_sum_factor must "
            "equal the sum factor column that plot_xy_data will use for correction "
            "(typically the same column passed to fit_trans, e.g. 'sum_factor_new'). "
            "If you see a systematic scale offset (e.g. A_true below A_lower), pass "
            "the per-cell sum factors that match your plotting correction:\n"
            "    sim_sum_factor=model.get_modality(model.primary_modality).sum_factors['sum_factor_new'].values",
            UserWarning,
            stacklevel=2,
        )

    if alpha_y_cols and group_col in meta.columns:
        groups = meta[group_col].values.astype(int)
        alpha_y_matrix = np.column_stack([df[col].values for col in alpha_y_cols])
        alpha_y_per_cell = alpha_y_matrix[:, groups]  # (n_genes, n_cells)
        mu_final = y_pred * alpha_y_per_cell * sum_factors[np.newaxis, :]
    else:
        mu_final = y_pred * sum_factors[np.newaxis, :]

    mu_final = np.maximum(mu_final, 1e-10)

    # ------------------------------------------------------------------ #
    # Sample from NegBin                                                    #
    # Sample: y ~ NegBin(total_count=phi_y, prob=phi_y/(phi_y+mu))         #
    # ------------------------------------------------------------------ #
    phi_y_exp = phi_y[:, np.newaxis]  # (n_genes, 1)
    prob = phi_y_exp / (phi_y_exp + mu_final)  # (n_genes, n_cells)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    y_obs = rng.negative_binomial(n=phi_y_exp, p=prob)  # (n_genes, n_cells)

    # ------------------------------------------------------------------ #
    # Build output DataFrame                                                #
    # ------------------------------------------------------------------ #
    cells = meta['cell'].values if 'cell' in meta.columns else [f'cell_{i}' for i in range(n_cells)]

    counts_dict = {}
    for i, gene in enumerate(gene_names):
        counts_dict[gene] = y_obs[i, :]
    counts_dict[cis_gene] = x_counts.astype(int)

    counts_df = pd.DataFrame(counts_dict, index=cells).T
    counts_df.index.name = None

    return counts_df
