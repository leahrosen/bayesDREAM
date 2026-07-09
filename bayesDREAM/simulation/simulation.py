"""
Simulate count / continuous data from fitted trans summary parameters.

Given a trans summary CSV (from save_trans_summary), cell metadata, and per-cell
x_true values, this module reconstructs the dose-response and draws observations
from the fitted distribution to produce a synthetic dataset suitable for re-fitting
with bayesDREAM.

Two parameterizations are supported:

Direct parameterization (A_median column present):
    Uses Hill/polynomial parameters from the trans summary directly.

Fold-change parameterization (y_ntc_median column present):
    Derives A and V from interpretable quantities — NTC expression (y_ntc),
    NTC x_true (x_ntc), K_log2FC, and full_log2FC — for each Hill component.
    Null genes are indicated by n=0 or full_log2FC=0.
    Only supported for negbinom (log2FC is not defined for binomial/normal).

Supported distributions
-----------------------
negbinom
    y ~ NegBin(phi_y, mu)  where mu = y_pred * alpha_y[group] * sum_factor
    phi_y = 1 / o_y^2.  Requires phi_y_median or o_y_median.
    Output: integer counts.

normal
    y ~ Normal(mu, sigma)  where mu = y_pred + alpha_y[group] (no sum factor)
    sigma = o_y_median.  Requires o_y_median.
    Output: continuous floats.

studentt
    y ~ StudentT(nu, mu, sigma)  where mu = y_pred + alpha_y[group]
    sigma = o_y_median; nu = nu_y_median if present, else nu_y_default.
    Output: continuous floats.

binomial
    y ~ Binomial(n, p)  where logit(p) = logit(y_pred) + alpha_y[group]
    Requires sim_denominator (n per cell).  No o_y/phi_y needed.
    Output: integer counts.

multinomial
    y ~ Multinomial(n, p)  where p_k is computed via per-category Hill curves.
    K-1 fitted categories each have their own Vmax/K/n Hill parameters; the Kth
    category is the residual (1 - sum of K-1 fitted probabilities). Alpha_y is
    applied log-additively across all categories before softmax normalization.
    Requires sim_total_counts (total reads per feature per cell).
    Output: integer counts per category; returns a dict (not a DataFrame) with:
    'counts' (n_features, n_cells, K_max), 'feature_names', 'feature_meta',
    'cells', 'cis_gene', 'cis_counts', 'K_max'. Pass to
    model.add_custom_modality(name, distribution='multinomial', ...).

Technical group effects (alpha_y)
----------------------------------
Applied differently by distribution:

- negbinom: multiplicative  (mu = y_pred * alpha_y[group])
  Columns: group_{g}_alpha_y_median per feature.
- normal/studentt: additive (mu = y_pred + alpha_y[group])
  Columns: group_{g}_alpha_y_median per feature.
- binomial: additive on logit scale (logit(p) = logit(y_pred) + alpha_y[group])
  Columns: group_{g}_alpha_y_median per feature.
- multinomial: log-additive per category (log(p_k) += alpha_y[group, k]), then softmax.
  Columns: group_{g}_alpha_y_cat{k}_median per feature (one per category k per group g).
  Detected via _get_multinomial_alpha_y(); median preferred over mean.

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


def _get_alpha_y_cols(df):
    """Return sorted alpha_y column names, preferring _median over _mean."""
    median_cols = sorted(
        [c for c in df.columns if c.startswith('group_') and c.endswith('_alpha_y_median')],
        key=lambda c: int(c.split('_')[1])
    )
    if median_cols:
        return median_cols
    return sorted(
        [c for c in df.columns if c.startswith('group_') and c.endswith('_alpha_y_mean')],
        key=lambda c: int(c.split('_')[1])
    )


def _get_multinomial_alpha_y(df, K_max):
    """Return [n_features, n_groups, K_max] alpha_y array for multinomial, or None.

    Reads columns named ``group_{g}_alpha_y_cat{k}_median`` (preferred) or
    ``group_{g}_alpha_y_cat{k}_mean``.  Missing category columns default to 0.0.
    """
    n_groups = 0
    suffix = '_median'
    while f'group_{n_groups}_alpha_y_cat0{suffix}' in df.columns:
        n_groups += 1
    if n_groups == 0:
        suffix = '_mean'
        while f'group_{n_groups}_alpha_y_cat0{suffix}' in df.columns:
            n_groups += 1
    if n_groups == 0:
        return None

    arr = np.zeros((len(df), n_groups, K_max), dtype=float)
    for g in range(n_groups):
        for k in range(K_max):
            col = f'group_{g}_alpha_y_cat{k}{suffix}'
            if col in df.columns:
                arr[:, g, k] = df[col].values
    return arr


def _apply_technical_effects_and_scale(
    distribution, y_pred, alpha_y_cols, df, meta, group_col, sum_factors,
    n_genes, n_cells
):
    """Apply per-group technical effects and (for negbinom) sum factors.

    Returns mu_final with shape (n_genes, n_cells).

    For negbinom: mu_final = y_pred * alpha_y[group] * sum_factor
    For normal/studentt: mu_final = y_pred + alpha_y[group]   (no sum factor)
    For binomial: returns logit(y_pred) + alpha_y[group]       (callers apply sigmoid)
    """
    if alpha_y_cols and group_col in meta.columns:
        groups = meta[group_col].values.astype(int)
        alpha_y_matrix = np.column_stack([df[col].values for col in alpha_y_cols])
        # alpha_y_matrix: (n_genes, n_groups); index by group per cell -> (n_genes, n_cells)
        alpha_y_per_cell = alpha_y_matrix[:, groups]
    else:
        alpha_y_per_cell = None

    if distribution == 'negbinom':
        mu = y_pred * sum_factors[np.newaxis, :]
        if alpha_y_per_cell is not None:
            mu = mu * alpha_y_per_cell
        return np.maximum(mu, 1e-10)

    elif distribution in ('normal', 'studentt'):
        mu = y_pred.copy()
        if alpha_y_per_cell is not None:
            mu = mu + alpha_y_per_cell
        return mu

    elif distribution == 'binomial':
        # Work in logit space; caller applies sigmoid to convert back to probability
        p_clipped = np.clip(y_pred, 1e-6, 1.0 - 1e-6)
        logit_p = np.log(p_clipped) - np.log(1.0 - p_clipped)
        if alpha_y_per_cell is not None:
            logit_p = logit_p + alpha_y_per_cell
        return logit_p  # caller: p = sigmoid(logit_p)

    else:
        raise ValueError(f"Unsupported distribution for technical effects: {distribution}")


def simulate_from_trans_summary(
    trans_summary_df: pd.DataFrame,
    meta: pd.DataFrame,
    x_true: Union[np.ndarray, pd.Series],
    x_counts: Union[np.ndarray, pd.Series],
    cis_gene: str,
    sim_sum_factor: Union[np.ndarray, pd.Series, float] = 1.0,
    sim_denominator: Optional[Union[np.ndarray, pd.Series, float]] = None,
    sim_total_counts: Optional[Union[np.ndarray, pd.Series, float]] = None,
    genes: Optional[list] = None,
    group_col: str = 'technical_group_code',
    seed: Optional[int] = None,
    fdr_threshold: Optional[float] = 0.05,
    nu_y_default: float = 5.0,
) -> Union[pd.DataFrame, dict]:
    """
    Simulate trans observations from fitted trans summary parameters.

    The generative model is distribution-specific (see module docstring).
    Parameterization (direct vs fold-change) is detected from the columns present
    in ``trans_summary_df``.

    Parameters
    ----------
    trans_summary_df : pd.DataFrame
        One row per feature (gene / junction / etc.).  Required columns:

        **Always required**

        - ``feature``: feature name.
        - ``function_type``: one of ``'single_hill'``, ``'additive_hill'``,
          ``'polynomial'``.
        - ``distribution`` *(optional)*: one of ``'negbinom'``, ``'normal'``,
          ``'studentt'``, ``'binomial'``.  Defaults to ``'negbinom'`` when absent.

        **Dispersion / noise (required by some distributions)**

        - negbinom: ``phi_y_median`` or ``o_y_median`` (phi = 1/o^2).
        - normal/studentt: ``o_y_median`` (used as sigma).
        - studentt: also uses ``nu_y_median`` if present, else ``nu_y_default``.
        - binomial: no dispersion column needed.

        **Parameterization detection** (``y_ntc_median`` takes priority):

        *Direct* — when ``A_median`` is a column.
        *Fold-change* — when ``y_ntc_median`` is a column.  Only for negbinom.

        Hill / polynomial parameter columns follow the same conventions as
        ``save_trans_summary`` output.  See existing docstring (inline below)
        for the full column listing by function_type and parameterization.

        **Technical group effects** *(optional)*

        - ``group_{g}_alpha_y_median`` or ``group_{g}_alpha_y_mean``: per-group
          corrections (median preferred).  Interpretation is distribution-specific.

    meta : pd.DataFrame
        Cell metadata.  Required columns: ``cell``, ``guide``, ``target``.
        When group columns are present in ``trans_summary_df``, must also contain
        ``group_col``.
    x_true : array-like, shape (n_cells,)
        Per-cell x_true values (cis gene expression), aligned to ``meta`` rows.
    x_counts : array-like, shape (n_cells,)
        Per-cell raw cis gene counts, included unchanged as the cis gene row
        in the output matrix.
    cis_gene : str
        Name of the cis gene (added as a row in the output count matrix).
    sim_sum_factor : array-like or float, optional
        Per-cell sum factors (default: 1.0).  **Only applied for negbinom.**
        For simulated data to align with ``plot_xy_data``, pass the same sum
        factor column used during ``fit_trans``::

            sim_sum_factor = model.get_modality(model.primary_modality).sum_factors[col].values

        This is NOT the sum factor for downstream fitting.
    sim_denominator : array-like or float, optional
        **Required for binomial.**  Total-count denominator per cell (and optionally
        per feature).  Shapes accepted:

        - scalar: same denominator for every cell and feature.
        - 1-D array ``(n_cells,)``: per-cell denominator, shared across features.
        - 2-D array ``(n_features, n_cells)``: independent per feature and cell.

        For splice-junction data, ``sim_denominator`` is typically the per-cell
        gene expression for the junction's gene.
    sim_total_counts : array-like or float, optional
        **Required for multinomial.**  Total read count per cell (and optionally per
        feature) from which to draw the multinomial sample.  Shapes accepted:

        - scalar: same total count for every cell and feature.
        - 1-D array ``(n_cells,)``: per-cell total, shared across features.
        - 2-D array ``(n_features, n_cells)``: independent per feature and cell.

        Typically set to the observed total counts per donor/acceptor site per cell
        (i.e. the sum of all category counts).
    genes : list of str, optional
        Subset of features to simulate.  Default: all in ``trans_summary_df``.
    group_col : str, optional
        Column in ``meta`` with integer technical group codes (default:
        ``'technical_group_code'``).
    seed : int, optional
        Random seed for reproducibility.
    fdr_threshold : float or None, optional
        FDR threshold for gating Hill components (default: 0.05).  Components
        above this FDR are treated as absent.  Set to ``None`` to use raw
        posterior means for all components.
    nu_y_default : float, optional
        Degrees of freedom for Student-t when ``nu_y_median`` is not in
        ``trans_summary_df`` (default: 5.0).

    Returns
    -------
    counts_df : pd.DataFrame or dict
        For ``negbinom``, ``normal``, ``studentt``, ``binomial``: a DataFrame
        with features as rows and cells as columns.  negbinom/binomial → integer
        counts; normal/studentt → continuous floats.  Always includes the cis gene
        as an extra row with raw ``x_counts``.

        For ``multinomial``: a dict with keys:

        - ``'counts'``: ``np.ndarray`` of shape ``(n_features, n_cells, K_max)``,
          integer category counts.
        - ``'feature_names'``: list of feature names.
        - ``'feature_meta'``: ``pd.DataFrame`` with per-feature metadata
          (same rows as ``trans_summary_df`` after any gene subsetting).
        - ``'cells'``: list of cell names.
        - ``'cis_gene'``: the cis gene name.
        - ``'cis_counts'``: ``np.ndarray`` of raw per-cell cis counts.
        - ``'K_max'``: int, total number of categories including the residual.

        Re-ingest into a model via::

            model.add_custom_modality(
                name='splicing_donor_sim',
                counts=result['counts'],
                feature_meta=result['feature_meta'],
                distribution='multinomial',
                cell_names=result['cells'],
            )
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

    # ------------------------------------------------------------------ #
    # Detect distribution                                                   #
    # ------------------------------------------------------------------ #
    distribution = df['distribution'].iloc[0] if 'distribution' in df.columns else 'negbinom'

    valid_distributions = {'negbinom', 'normal', 'studentt', 'binomial', 'multinomial'}
    if distribution not in valid_distributions:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Supported: {sorted(valid_distributions)}"
        )

    # ------------------------------------------------------------------ #
    # Validate distribution-specific inputs                                 #
    # ------------------------------------------------------------------ #
    if distribution == 'negbinom':
        if 'phi_y_median' in df.columns:
            phi_y = df['phi_y_median'].values
        elif 'o_y_median' in df.columns:
            phi_y = 1.0 / (df['o_y_median'].values ** 2)
        else:
            raise ValueError(
                "trans_summary_df must contain 'phi_y_median' or 'o_y_median' for negbinom"
            )

    elif distribution in ('normal', 'studentt'):
        if 'o_y_median' not in df.columns:
            raise ValueError(
                f"trans_summary_df must contain 'o_y_median' for {distribution} distribution"
            )
        sigma_y = df['o_y_median'].values  # (n_genes,)
        if distribution == 'studentt':
            nu_y = (df['nu_y_median'].values if 'nu_y_median' in df.columns
                    else np.full(n_genes, nu_y_default))

    elif distribution == 'binomial':
        if sim_denominator is None:
            raise ValueError(
                "sim_denominator is required for binomial distribution. "
                "For splice-junction data this is typically the per-cell gene "
                "expression for the junction's host gene."
            )
        if np.isscalar(sim_denominator):
            denom_arr = np.full((n_genes, n_cells), int(sim_denominator))
        else:
            denom_raw = np.asarray(sim_denominator)
            if denom_raw.ndim == 1:
                if len(denom_raw) != n_cells:
                    raise ValueError(
                        f"sim_denominator has {len(denom_raw)} elements but expected {n_cells}"
                    )
                denom_arr = np.broadcast_to(denom_raw[np.newaxis, :], (n_genes, n_cells)).copy()
            elif denom_raw.ndim == 2:
                if denom_raw.shape != (n_genes, n_cells):
                    raise ValueError(
                        f"sim_denominator shape {denom_raw.shape} does not match "
                        f"(n_genes={n_genes}, n_cells={n_cells})"
                    )
                denom_arr = denom_raw.copy()
            else:
                raise ValueError(
                    f"sim_denominator must be scalar, 1-D (n_cells,) or 2-D (n_genes, n_cells); "
                    f"got shape {denom_raw.shape}"
                )
        denom_arr = denom_arr.astype(int)

    elif distribution == 'multinomial':
        if sim_total_counts is None:
            raise ValueError(
                "sim_total_counts is required for multinomial distribution. "
                "Pass the total read count per cell (and optionally per feature), "
                "e.g. the sum of all category counts for each donor/acceptor site."
            )
        if np.isscalar(sim_total_counts):
            total_counts_arr = np.full((n_genes, n_cells), int(sim_total_counts))
        else:
            tc_raw = np.asarray(sim_total_counts)
            if tc_raw.ndim == 1:
                if len(tc_raw) != n_cells:
                    raise ValueError(
                        f"sim_total_counts has {len(tc_raw)} elements but expected {n_cells}"
                    )
                total_counts_arr = np.broadcast_to(
                    tc_raw[np.newaxis, :], (n_genes, n_cells)
                ).copy()
            elif tc_raw.ndim == 2:
                if tc_raw.shape != (n_genes, n_cells):
                    raise ValueError(
                        f"sim_total_counts shape {tc_raw.shape} does not match "
                        f"(n_genes={n_genes}, n_cells={n_cells})"
                    )
                total_counts_arr = tc_raw.copy()
            else:
                raise ValueError(
                    f"sim_total_counts must be scalar, 1-D (n_cells,) or 2-D "
                    f"(n_genes, n_cells); got shape {tc_raw.shape}"
                )
        total_counts_arr = total_counts_arr.astype(int)

    # ------------------------------------------------------------------ #
    # Fold-change parameterization restriction                              #
    # ------------------------------------------------------------------ #
    use_fc_params = 'y_ntc_median' in df.columns
    if use_fc_params and distribution != 'negbinom':
        raise ValueError(
            f"Fold-change parameterization (y_ntc_median column) is only supported "
            f"for negbinom, not for '{distribution}'. Use the direct parameterization "
            f"(A_median column) for other distributions."
        )
    if not use_fc_params and 'A_median' not in df.columns and distribution != 'multinomial':
        raise ValueError(
            "trans_summary_df must contain either 'A_median' (direct parameterization), "
            "'y_ntc_median' (fold-change parameterization, negbinom only), "
            "or per-category A columns (A_cat{k}_median, for multinomial)."
        )

    # ------------------------------------------------------------------ #
    # Warn about sum_factor for non-negbinom                               #
    # ------------------------------------------------------------------ #
    if distribution != 'negbinom' and not (_sim_sum_factor_is_scalar and float(sim_sum_factor) == 1.0):
        warnings.warn(
            f"sim_sum_factor is ignored for distribution='{distribution}'. "
            "Sum factors are only applied for negbinom.",
            UserWarning,
            stacklevel=2,
        )
    elif distribution == 'negbinom' and _sim_sum_factor_is_scalar and float(sim_sum_factor) == 1.0:
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

    x_exp = x_true[np.newaxis, :]  # (1, n_cells) for broadcasting

    # ------------------------------------------------------------------ #
    # Compute y_pred  (n_genes, n_cells)                                   #
    # ------------------------------------------------------------------ #

    if use_fc_params:
        # ---- Fold-change parameterization (negbinom only) ----
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

                if fdr_threshold is not None and 'fdr_beta' in df.columns:
                    fdr_b = df['fdr_beta'].values
                    full_log2FC_b = np.where(
                        np.isfinite(fdr_b) & (fdr_b <= fdr_threshold), full_log2FC_b, 0.0
                    )

                A_b, V_b, K_b = _compute_AV_from_fc(n_b, y_ntc, x_ntc, K_log2FC_b, full_log2FC_b)

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

        if distribution == 'multinomial':
            # ---- Per-category parameter detection ----
            cat_A_cols = sorted(
                [c for c in df.columns if c.startswith('A_cat') and c.endswith('_median')],
                key=lambda c: int(c.split('_cat')[1].split('_')[0])
            )
            K_minus_1 = len(cat_A_cols)
            K_max_mn = K_minus_1 + 1  # K-1 fitted categories + 1 residual

            if K_minus_1 == 0:
                raise ValueError(
                    "No per-category A columns found (A_cat{k}_median). "
                    "Ensure the trans summary was generated with multinomial data "
                    "and that save_trans_summary exported per-category parameters."
                )

            A_cats = np.column_stack([df[c].values for c in cat_A_cols])  # (n_genes, K-1)

            # ---- Compute K-1 category dose-responses (n_genes, n_cells, K-1) ----
            y_pred_cats = np.zeros((n_genes, n_cells, K_minus_1))

            for k in range(K_minus_1):
                A_k = A_cats[:, k]  # baseline probability for category k

                if function_type in ('single_hill', 'additive_hill'):
                    vmax_a_col = f'Vmax_a_cat{k}_median'
                    K_a_col = f'K_a_cat{k}_median'
                    n_a_col = f'n_a_cat{k}_median'
                    alpha_col = f'alpha_cat{k}_median'

                    # Missing columns → phantom category; use flat baseline
                    if any(c not in df.columns for c in [vmax_a_col, K_a_col, n_a_col]):
                        y_pred_cats[:, :, k] = A_k[:, np.newaxis]
                        continue

                    Vmax_a_k = df[vmax_a_col].values
                    K_a_k = df[K_a_col].values
                    n_a_k = df[n_a_col].values
                    alpha_k = (df[alpha_col].values.copy()
                               if alpha_col in df.columns else np.ones(n_genes))

                    if fdr_threshold is not None and f'fdr_alpha_cat{k}' in df.columns:
                        fdr_k = df[f'fdr_alpha_cat{k}'].values
                        alpha_k = np.where(
                            np.isfinite(fdr_k) & (fdr_k <= fdr_threshold), alpha_k, 0.0
                        )

                    # NaN params → phantom category; zero out Hill contribution
                    nan_mask = (~np.isfinite(Vmax_a_k) | ~np.isfinite(K_a_k)
                                | ~np.isfinite(n_a_k))
                    Vmax_a_k = np.where(nan_mask, 0.0, Vmax_a_k)
                    K_a_k = np.where(nan_mask, 1.0, K_a_k)
                    n_a_k = np.where(nan_mask, 1.0, n_a_k)
                    alpha_k = np.where(nan_mask, 0.0, alpha_k)

                    hill_a_k = _hill(x_exp, Vmax_a_k[:, np.newaxis],
                                     K_a_k[:, np.newaxis], n_a_k[:, np.newaxis])
                    y_k = A_k[:, np.newaxis] + alpha_k[:, np.newaxis] * hill_a_k

                    if function_type == 'additive_hill':
                        vmax_b_col = f'Vmax_b_cat{k}_median'
                        K_b_col = f'K_b_cat{k}_median'
                        n_b_col = f'n_b_cat{k}_median'
                        beta_col = f'beta_cat{k}_median'

                        if all(c in df.columns for c in [vmax_b_col, K_b_col, n_b_col]):
                            Vmax_b_k = df[vmax_b_col].values
                            K_b_k = df[K_b_col].values
                            n_b_k = df[n_b_col].values
                            beta_k = (df[beta_col].values.copy()
                                      if beta_col in df.columns else np.ones(n_genes))

                            if fdr_threshold is not None and f'fdr_beta_cat{k}' in df.columns:
                                fdr_k = df[f'fdr_beta_cat{k}'].values
                                beta_k = np.where(
                                    np.isfinite(fdr_k) & (fdr_k <= fdr_threshold), beta_k, 0.0
                                )

                            nan_mask_b = (~np.isfinite(Vmax_b_k) | ~np.isfinite(K_b_k)
                                          | ~np.isfinite(n_b_k))
                            Vmax_b_k = np.where(nan_mask_b, 0.0, Vmax_b_k)
                            K_b_k = np.where(nan_mask_b, 1.0, K_b_k)
                            n_b_k = np.where(nan_mask_b, 1.0, n_b_k)
                            beta_k = np.where(nan_mask_b, 0.0, beta_k)

                            hill_b_k = _hill(x_exp, Vmax_b_k[:, np.newaxis],
                                             K_b_k[:, np.newaxis], n_b_k[:, np.newaxis])
                            y_k = y_k + beta_k[:, np.newaxis] * hill_b_k

                    y_pred_cats[:, :, k] = y_k

                elif function_type == 'polynomial':
                    raise NotImplementedError(
                        "Polynomial multinomial simulation is not yet supported. "
                        "Use single_hill or additive_hill."
                    )
                else:
                    raise ValueError(
                        f"Unsupported function_type for multinomial: {function_type!r}"
                    )

            # ---- Clamp K-1 probabilities and compute residual ----
            y_pred_cats = np.clip(y_pred_cats, 1e-6, 1.0 - 1e-6)
            sum_kminus1 = y_pred_cats.sum(axis=-1, keepdims=True)  # (n_genes, n_cells, 1)
            over = sum_kminus1 > (1.0 - 1e-6)
            y_pred_cats = np.where(over, y_pred_cats * (1.0 - 1e-6) / sum_kminus1, y_pred_cats)
            sum_kminus1 = y_pred_cats.sum(axis=-1, keepdims=True)
            y_K_col = np.clip(1.0 - sum_kminus1, 1e-6, 1.0)
            probs_3d = np.concatenate([y_pred_cats, y_K_col], axis=-1)  # (n_genes, n_cells, K_max)

            # ---- Apply technical effects (log-additive per category, then softmax) ----
            # alpha_y for multinomial is [C, T, K]: each category has its own group offset.
            mn_alpha_y = _get_multinomial_alpha_y(df, K_max_mn)
            if mn_alpha_y is not None and group_col in meta.columns:
                groups = meta[group_col].values.astype(int)
                alpha_y_per_cell = mn_alpha_y[:, groups, :]  # (n_genes, n_cells, K_max)
                log_probs = np.log(np.maximum(probs_3d, 1e-10))
                log_probs = log_probs + alpha_y_per_cell      # (n_genes, n_cells, K_max)
                log_probs -= log_probs.max(axis=-1, keepdims=True)
                probs_3d = np.exp(log_probs)
                probs_3d /= probs_3d.sum(axis=-1, keepdims=True)

            # ---- Sample ----
            y_obs_3d = np.zeros((n_genes, n_cells, K_max_mn), dtype=int)
            for t in range(n_genes):
                for n_idx in range(n_cells):
                    n_total = total_counts_arr[t, n_idx]
                    if n_total <= 0:
                        continue
                    p = probs_3d[t, n_idx, :]
                    p_sum = p.sum()
                    if p_sum <= 0 or not np.isfinite(p_sum):
                        continue
                    y_obs_3d[t, n_idx, :] = rng.multinomial(n=int(n_total), pvals=p / p_sum)

            # ---- Build output dict (use add_custom_modality to re-ingest) ----
            cells = (meta['cell'].values if 'cell' in meta.columns
                     else [f'cell_{i}' for i in range(n_cells)])
            return {
                'counts': y_obs_3d,                    # (n_features, n_cells, K_max)
                'feature_names': list(gene_names),
                'feature_meta': df.reset_index(drop=True),
                'cells': list(cells),
                'cis_gene': cis_gene,
                'cis_counts': x_counts,
                'K_max': K_max_mn,
            }

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
                raise ValueError("No polynomial coefficient columns found (coef_N_median).")

            if distribution == 'negbinom':
                # Polynomial operates in log2(x) space: log2(y) = sum(coef_i * log2(x)^i)
                # where coef_0 = log2(A) and coef_1..d are the polynomial terms.
                log2_x_exp = np.log2(np.maximum(x_exp, 1e-12))
                log2_y_pred = np.zeros((n_genes, n_cells))
                for i, col in enumerate(coef_cols):
                    log2_y_pred += df[col].values[:, np.newaxis] * np.power(log2_x_exp, i)
                y_pred = np.power(2.0, log2_y_pred)

            elif distribution in ('normal', 'studentt'):
                # Polynomial in linear x space: y = sum(coef_i * x^i)
                # where coef_0 = A
                y_pred = np.zeros((n_genes, n_cells))
                for i, col in enumerate(coef_cols):
                    y_pred += df[col].values[:, np.newaxis] * np.power(x_exp, i)

            elif distribution == 'binomial':
                # Polynomial in linear x space on logit scale:
                # logit(p) = sum(coef_i * x^i) where coef_0 = logit(A)
                logit_p = np.zeros((n_genes, n_cells))
                for i, col in enumerate(coef_cols):
                    logit_p += df[col].values[:, np.newaxis] * np.power(x_exp, i)
                y_pred = 1.0 / (1.0 + np.exp(-logit_p))  # sigmoid

            else:
                raise ValueError(f"Unsupported distribution for polynomial: {distribution}")

        else:
            raise ValueError(f"Unsupported function_type: {function_type}")

    # ------------------------------------------------------------------ #
    # Apply technical effects and (for negbinom) sum factors               #
    # ------------------------------------------------------------------ #
    # Floor y_pred to avoid log(0) in negbinom / logit(0) in binomial
    if distribution == 'negbinom':
        y_pred = np.maximum(y_pred, 1e-6)
    elif distribution == 'binomial':
        # Already a probability; clamp before logit in technical-effects step
        y_pred = np.clip(y_pred, 1e-6, 1.0 - 1e-6)

    alpha_y_cols = _get_alpha_y_cols(df)

    mu_or_logit = _apply_technical_effects_and_scale(
        distribution, y_pred, alpha_y_cols, df, meta, group_col, sum_factors,
        n_genes, n_cells
    )

    # ------------------------------------------------------------------ #
    # Sample                                                                #
    # ------------------------------------------------------------------ #

    if distribution == 'negbinom':
        phi_y_exp = phi_y[:, np.newaxis]                      # (n_genes, 1)
        prob = phi_y_exp / (phi_y_exp + mu_or_logit)          # (n_genes, n_cells)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        y_obs = rng.negative_binomial(n=phi_y_exp, p=prob)    # (n_genes, n_cells)

    elif distribution == 'normal':
        sigma_y_exp = sigma_y[:, np.newaxis]                  # (n_genes, 1)
        y_obs = rng.normal(loc=mu_or_logit, scale=sigma_y_exp)

    elif distribution == 'studentt':
        # scipy.stats.t is used because numpy has no native Student-t RNG
        from scipy.stats import t as scipy_t
        sigma_y_exp = sigma_y[:, np.newaxis]                  # (n_genes, 1)
        nu_y_exp = nu_y[:, np.newaxis]                        # (n_genes, 1)
        y_obs = scipy_t.rvs(df=nu_y_exp, loc=mu_or_logit, scale=sigma_y_exp,
                             random_state=rng)
        y_obs = np.reshape(y_obs, (n_genes, n_cells))  # scipy may squeeze n_genes=1

    elif distribution == 'binomial':
        # mu_or_logit holds logit(p); convert to probability
        prob = 1.0 / (1.0 + np.exp(-mu_or_logit))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        y_obs = rng.binomial(n=denom_arr, p=prob)             # (n_genes, n_cells)

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
