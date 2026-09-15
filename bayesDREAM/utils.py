"""
Utility functions for bayesDREAM.

This module contains helper functions used throughout the bayesDREAM package:
- Thread configuration
- Numerical solvers
- Dose-response functions (Hill, polynomial)
- Tensor utilities
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import betainc
from scipy.optimize import brentq


########################################
# Thread Configuration
########################################

def set_max_threads(cores: int):
    """
    Set the number of threads for Pyro and backend computations.

    Parameters
    ----------
    cores : int
        Number of CPU cores to use
    """
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)


########################################
# Name Handling
########################################

def make_names_unique(names, join: str = "-"):
    """
    Disambiguate duplicate names, AnnData/scanpy style.

    The first occurrence of a name is kept unchanged; each subsequent
    occurrence gets ``{join}{n}`` appended (n = 1, 2, ...). Used to make
    modality ``feature_names`` safe for name-based lookup (e.g. gene
    symbols that are not unique in Ensembl annotation, such as some
    pseudogenes/readthrough transcripts).

    Parameters
    ----------
    names : list of str
    join : str, default="-"
        Separator between the original name and the disambiguating suffix.

    Returns
    -------
    list of str
        Same length and order as `names`, with all entries unique.
    """
    names = [str(n) for n in names]
    existing = set(names)
    seen_counts = {}
    out = []
    for name in names:
        if name not in seen_counts:
            seen_counts[name] = 0
            out.append(name)
        else:
            seen_counts[name] += 1
            candidate = f"{name}{join}{seen_counts[name]}"
            while candidate in existing:
                seen_counts[name] += 1
                candidate = f"{name}{join}{seen_counts[name]}"
            existing.add(candidate)
            out.append(candidate)
    return out


def _is_integer_like_index(index: pd.Index) -> bool:
    """True if `index` is a RangeIndex or has integer/unsigned-integer dtype."""
    return isinstance(index, pd.RangeIndex) or index.dtype.kind in ('i', 'u')


def _invalid_identifier_reason(values: list, label: str) -> Optional[str]:
    """
    Check whether `values` is usable as a feature identifier source: every
    value must be a genuine ``str`` (this also rejects ``None``/``NaN``/``NA``,
    none of which are ``str`` instances) and all values must be unique.

    Returns
    -------
    str or None
        A human-readable reason the candidate is invalid, or None if it's
        valid (non-null strings, no duplicates).
    """
    n = len(values)
    non_string = [v for v in values if not isinstance(v, str)]
    if non_string:
        return (f"{label} contains {len(non_string)} missing or non-string "
                f"value(s) (expected every entry to be a str)")
    if len(set(values)) != n:
        return f"{label} contains {n - len(set(values))} duplicated value(s)"
    return None


# Bonus identifier columns tried (in this order, after the universal ones)
# only for modalities where each row IS a gene (the primary 'gene' modality
# and the 'cis' modality) — never for modalities where genes are many-to-one
# with rows (transcripts, splice junctions, ATAC peaks, custom modalities).
_GENE_IDENTITY_PRE_COLS = ('ens_id', 'gene_id', 'gene')
_GENE_IDENTITY_POST_COLS = ('gene_name', 'gene_symbol')


def resolve_feature_ids(
    counts,
    feature_meta: Optional[pd.DataFrame],
    feature_name_col: Optional[str] = None,
    feature_names: Optional[list] = None,
    cells_axis: int = 1,
    n_features: Optional[int] = None,
    is_gene_identity: bool = False,
    context: Optional[str] = None,
) -> tuple:
    """
    Single source of truth for deciding each feature's ``feature_id`` and for
    stamping it consistently onto both ``feature_meta`` and (if it is a
    DataFrame) ``counts``.

    ``feature_name_col`` and ``feature_names`` are mutually exclusive explicit
    overrides. At most one may be given.

    Priority (first usable source wins; "usable" always means: every value is
    a genuine, non-null ``str``, and there are no duplicates):

    1. ``feature_name_col`` — a column name of ``feature_meta`` (which must
       therefore be provided). OR ``feature_names`` — an explicit list, one
       entry per feature. Either is validated (existence, non-null, str type,
       uniqueness) and, if invalid, **raises** (these were explicitly
       requested, so a silent fallback would hide a real problem).
    2. ``counts``'s own index (if ``counts`` is a DataFrame) along the feature
       axis (row index when ``cells_axis == 1``, else columns) — only if that
       index is not a plain integer/RangeIndex. If it exists but is invalid
       (non-str/null entries, duplicates), this **raises** rather than
       silently falling through to step 3 — an explicit non-integer index is
       as deliberate a signal as ``feature_name_col``.
    3. ``feature_meta``'s own index — same non-integer-index check, same
       raise-if-invalid-once-present semantics as step 2.
    4. A cascade of ``feature_meta`` columns, tried in order and *silently
       skipped* (no error) if missing, containing nulls/non-strings, or
       duplicated: ``'feature_id' > 'feature' > ['ens_id' > 'gene_id' >
       'gene', only if is_gene_identity] > 'feature_name' > ['gene_name' >
       'gene_symbol', only if is_gene_identity]``. If none of those match,
       falls back to the first column of ``feature_meta`` (in column order,
       skipping ones already tried above) that is fully non-null, all-``str``,
       and duplicate-free.
    5. If nothing above produced a result: **raises** — ``feature_meta`` has
       no usable string column and neither ``counts`` nor ``feature_meta`` has
       a usable string index.

    Once resolved, the ``feature_id`` list is written into
    ``feature_meta['feature_id']`` (warning first if this overwrites an
    existing, different-valued ``'feature_id'`` column) and becomes
    ``feature_meta``'s index (warning first if this overwrites an existing,
    different string index). If ``counts`` is a DataFrame, the same list also
    becomes its index/columns along the feature axis (again warning first if
    it overwrites a different existing string index/columns). Callers receive
    copies — the objects passed in are never mutated in place.

    Parameters
    ----------
    counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
        The modality's count data (only its shape/index matters here).
    feature_meta : pd.DataFrame or None
        Feature-level metadata. Required (non-None, non-empty) unless
        ``counts`` is a DataFrame supplying a usable string index (step 2).
    feature_name_col : str, optional
        Column of ``feature_meta`` to use as the explicit feature_id source.
        Mutually exclusive with ``feature_names``.
    feature_names : list of str, optional
        Explicit feature_id list, length must equal ``n_features``. Mutually
        exclusive with ``feature_name_col``.
    cells_axis : int, default=1
        Which axis of a DataFrame ``counts`` holds cells (0 or 1); the other
        axis holds features.
    n_features : int, optional
        Expected feature count, used to validate an explicit ``feature_names``
        list's length. Callers should always pass this when known.
    is_gene_identity : bool, default=False
        True for modalities where each row IS a gene (the primary 'gene'
        modality and the 'cis' modality) — enables the extra gene-specific
        bonus columns in step 4. False for modalities where genes are
        many-to-one with rows (transcripts, splicing, ATAC, custom).
    context : str, optional
        Label (e.g. ``"Modality 'gene'"``) used in warning/error/info
        messages.

    Returns
    -------
    (feature_ids, feature_meta, counts) : (list of str, pd.DataFrame, object)
        Resolved feature_id list; feature_meta with 'feature_id' column and
        matching string index; counts (a reindexed copy if it was a
        DataFrame and needed relabeling along the feature axis, else the
        original object unchanged).
    """
    if feature_name_col is not None and feature_names is not None:
        raise ValueError(
            "Only one of feature_name_col and feature_names may be provided, not both."
        )

    ctx = context or "resolve_feature_ids"
    feature_ids = None

    # ---- Step 1: explicit override ----
    if feature_name_col is not None:
        if feature_meta is None or len(feature_meta) == 0:
            raise ValueError(
                f"{ctx}: feature_name_col='{feature_name_col}' was given but "
                f"feature_meta was not provided (feature_name_col requires feature_meta)."
            )
        if feature_name_col not in feature_meta.columns:
            raise ValueError(
                f"{ctx}: feature_name_col='{feature_name_col}' is not a column of "
                f"feature_meta. Available columns: {list(feature_meta.columns)}"
            )
        candidate = feature_meta[feature_name_col].tolist()
        reason = _invalid_identifier_reason(candidate, f"feature_meta['{feature_name_col}']")
        if reason is not None:
            raise ValueError(f"{ctx}: {reason}.")
        feature_ids = candidate

    elif feature_names is not None:
        feature_names = list(feature_names)
        if n_features is not None and len(feature_names) != n_features:
            raise ValueError(
                f"{ctx}: feature_names has length {len(feature_names)} but counts has "
                f"{n_features} features."
            )
        reason = _invalid_identifier_reason(feature_names, "feature_names")
        if reason is not None:
            raise ValueError(f"{ctx}: {reason}.")
        feature_ids = feature_names

    # ---- Step 2: counts' own string index (DataFrame only) ----
    if feature_ids is None and isinstance(counts, pd.DataFrame):
        axis_index = counts.index if cells_axis == 1 else counts.columns
        if not _is_integer_like_index(axis_index):
            candidate = axis_index.tolist()
            reason = _invalid_identifier_reason(candidate, "counts' index")
            if reason is not None:
                raise ValueError(
                    f"{ctx}: counts has a non-integer index intended as feature "
                    f"identifiers, but {reason}."
                )
            feature_ids = candidate
            print(f"[INFO] {ctx}: no explicit identifier given; using counts' own "
                  f"index as feature_id.")

    # ---- Step 3: feature_meta's own string index ----
    if feature_ids is None and feature_meta is not None and len(feature_meta) > 0:
        if not _is_integer_like_index(feature_meta.index):
            candidate = feature_meta.index.tolist()
            reason = _invalid_identifier_reason(candidate, "feature_meta's index")
            if reason is not None:
                raise ValueError(
                    f"{ctx}: feature_meta has a non-integer index intended as feature "
                    f"identifiers, but {reason}."
                )
            feature_ids = candidate
            print(f"[INFO] {ctx}: no explicit identifier given; using feature_meta's "
                  f"own index as feature_id.")

    # ---- Step 4: feature_meta column cascade ----
    if feature_ids is None:
        if feature_meta is None or len(feature_meta) == 0:
            raise ValueError(
                f"{ctx}: no feature_name_col/feature_names given, and neither counts nor "
                f"feature_meta has a usable string index — feature_meta must be provided "
                f"with an identifier column."
            )
        priority_cols = ['feature_id', 'feature']
        if is_gene_identity:
            priority_cols += list(_GENE_IDENTITY_PRE_COLS)
        priority_cols += ['feature_name']
        if is_gene_identity:
            priority_cols += list(_GENE_IDENTITY_POST_COLS)

        tried = set(priority_cols)
        for col in priority_cols:
            if col in feature_meta.columns:
                candidate = feature_meta[col].tolist()
                if _invalid_identifier_reason(candidate, col) is None:
                    feature_ids = candidate
                    print(f"[INFO] {ctx}: no explicit identifier given; using "
                          f"feature_meta['{col}'] as feature_id.")
                    break

        if feature_ids is None:
            for col in feature_meta.columns:
                if col in tried:
                    continue
                candidate = feature_meta[col].tolist()
                if _invalid_identifier_reason(candidate, col) is None:
                    feature_ids = candidate
                    print(f"[INFO] {ctx}: no standard identifier column found; using "
                          f"feature_meta['{col}'] as feature_id (first non-duplicated "
                          f"string column).")
                    break

    # ---- Step 5: give up ----
    if feature_ids is None:
        raise ValueError(
            f"{ctx}: could not determine feature identifiers. feature_meta has no fully "
            f"populated, duplicate-free string column, and neither counts nor feature_meta "
            f"has a usable string index. Pass feature_name_col or feature_names explicitly, "
            f"or add a unique string identifier column to feature_meta."
        )

    # ---- Stamp feature_id onto feature_meta (column + index) ----
    feature_meta = (feature_meta.copy() if feature_meta is not None
                     else pd.DataFrame(index=range(len(feature_ids))))

    if 'feature_id' in feature_meta.columns and feature_meta['feature_id'].tolist() != feature_ids:
        warnings.warn(
            f"{ctx}: overwriting feature_meta's existing 'feature_id' column "
            f"(values differ from the resolved feature_id).",
            UserWarning
        )
    feature_meta['feature_id'] = feature_ids

    if not _is_integer_like_index(feature_meta.index) and feature_meta.index.tolist() != feature_ids:
        warnings.warn(
            f"{ctx}: overwriting feature_meta's existing string index with the "
            f"resolved feature_id (values differ).",
            UserWarning
        )
    feature_meta.index = pd.Index(feature_ids, name='feature_id')

    # ---- Stamp feature_id onto counts (DataFrame only) ----
    if isinstance(counts, pd.DataFrame):
        counts = counts.copy()
        axis_index = counts.index if cells_axis == 1 else counts.columns
        axis_label = 'index' if cells_axis == 1 else 'columns'
        if not _is_integer_like_index(axis_index) and axis_index.tolist() != feature_ids:
            warnings.warn(
                f"{ctx}: overwriting counts' existing string {axis_label} with the "
                f"resolved feature_id (values differ).",
                UserWarning
            )
        if cells_axis == 1:
            counts.index = pd.Index(feature_ids, name='feature_id')
        else:
            counts.columns = pd.Index(feature_ids, name='feature_id')

    return feature_ids, feature_meta, counts


def _find_unique_match(values: list, name: str, label: str, ctx: str, required: bool) -> Optional[int]:
    """
    Find the position of `name` within `values` (by equality, not identifier-column
    validity — `values` may contain nulls/non-strings/duplicates elsewhere).

    Returns
    -------
    int or None
        The single matching position. None if `name` isn't in `values` and
        `required` is False (caller should try the next candidate source).

    Raises
    ------
    ValueError
        If `name` matches more than one entry in `values` (ambiguous — cannot
        tell which row is meant), or if `name` isn't found and `required` is True.
    """
    matches = [i for i, v in enumerate(values) if v == name]
    if len(matches) > 1:
        raise ValueError(
            f"{ctx}: '{name}' is duplicated {len(matches)} times in {label} — "
            f"cannot determine which row is meant."
        )
    if len(matches) == 1:
        return matches[0]
    if required:
        raise ValueError(f"{ctx}: '{name}' not found in {label}.")
    return None


def locate_feature(
    name: str,
    counts=None,
    feature_meta: Optional[pd.DataFrame] = None,
    feature_name_col: Optional[str] = None,
    feature_names: Optional[list] = None,
    cells_axis: int = 1,
    is_gene_identity: bool = False,
    context: Optional[str] = None,
) -> int:
    """
    Find the row position (0-based, `.iloc`-style) of a single feature named `name`.

    Tries the same candidate sources, in the same priority order, as
    `resolve_feature_ids` — but scoped to finding one specific value rather than
    establishing a whole-panel identifier column, so the rules are relaxed in two
    ways: (a) a source doesn't need to be a clean, fully-populated, duplicate-free
    identifier space to be searched — irrelevant nulls/non-strings/duplicates
    elsewhere in it don't block a search; (b) **no source is exclusive/terminal**
    — every step, including an explicit `feature_name_col`/`feature_names`, is
    just the first place searched, not the only place. At each source, in turn:
    zero matches for `name` -> move to the next source; exactly one match ->
    return it immediately (search stops); more than one match -> raise
    immediately (ambiguous: `name` is duplicated *within that source*). Only
    once every source has been tried with zero matches does this raise "not
    found".

    1. `feature_name_col` (a column of `feature_meta`) or `feature_names` (an
       explicit list) — mutually exclusive with each other. Tried first, but if
       `name` isn't found there, the search continues to steps 2-4 below rather
       than stopping.
    2. `counts`'s own non-integer index along the feature axis (if `counts` is
       a DataFrame).
    3. `feature_meta`'s own non-integer index.
    4. `feature_meta` columns, in the same order as `resolve_feature_ids` step 4
       (`'feature_id' > 'feature' > ['ens_id' > 'gene_id' > 'gene' if
       is_gene_identity] > 'feature_name' > ['gene_name' > 'gene_symbol' if
       is_gene_identity]`, then every remaining column in column order).

    Parameters mirror `resolve_feature_ids` (see its docstring for the shared
    priority rationale); `context` labels error messages.
    """
    if feature_name_col is not None and feature_names is not None:
        raise ValueError("Only one of feature_name_col and feature_names may be provided, not both.")

    ctx = context or "locate_feature"

    # ---- Step 1: feature_name_col / feature_names — tried first, not exclusive ----
    if feature_name_col is not None:
        if feature_meta is not None and feature_name_col in feature_meta.columns:
            pos = _find_unique_match(
                feature_meta[feature_name_col].tolist(), name,
                f"feature_meta['{feature_name_col}']", ctx, required=False,
            )
            if pos is not None:
                return pos
    elif feature_names is not None:
        pos = _find_unique_match(list(feature_names), name, "feature_names", ctx, required=False)
        if pos is not None:
            return pos

    # ---- Step 2: counts' own string index ----
    if isinstance(counts, pd.DataFrame):
        axis_index = counts.index if cells_axis == 1 else counts.columns
        if not _is_integer_like_index(axis_index):
            pos = _find_unique_match(axis_index.tolist(), name, "counts' index", ctx, required=False)
            if pos is not None:
                return pos

    # ---- Step 3: feature_meta's own string index ----
    if feature_meta is not None and len(feature_meta) > 0:
        if not _is_integer_like_index(feature_meta.index):
            pos = _find_unique_match(feature_meta.index.tolist(), name, "feature_meta's index", ctx, required=False)
            if pos is not None:
                return pos

    # ---- Step 4: feature_meta column cascade ----
    if feature_meta is not None and len(feature_meta) > 0:
        priority_cols = ['feature_id', 'feature']
        if is_gene_identity:
            priority_cols += list(_GENE_IDENTITY_PRE_COLS)
        priority_cols += ['feature_name']
        if is_gene_identity:
            priority_cols += list(_GENE_IDENTITY_POST_COLS)

        tried = set(priority_cols)
        for col in priority_cols:
            if col in feature_meta.columns:
                pos = _find_unique_match(feature_meta[col].tolist(), name, f"feature_meta['{col}']", ctx, required=False)
                if pos is not None:
                    return pos

        for col in feature_meta.columns:
            if col in tried:
                continue
            pos = _find_unique_match(feature_meta[col].tolist(), name, f"feature_meta['{col}']", ctx, required=False)
            if pos is not None:
                return pos

    raise ValueError(
        f"{ctx}: '{name}' not found — checked counts' index, feature_meta's index, "
        f"and every column of feature_meta."
    )


########################################
# Numerical Solvers
########################################

def find_beta(alpha: float, gamma_threshold: float, p0: float, epsilon: float = 1e-6) -> float:
    """
    Numerically solve for beta given Betainc(alpha, beta, gamma_threshold) = p0.

    Parameters
    ----------
    alpha : float
        Alpha parameter for incomplete beta function
    gamma_threshold : float
        Threshold value
    p0 : float
        Target probability
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    float
        Solved beta value
    """
    def equation(bet):
        return betainc(alpha, bet, gamma_threshold) - p0
    beta_lower = epsilon
    beta_upper = 1
    return brentq(equation, beta_lower, beta_upper)


def calculate_mu_x_guide(guide, x_obs_ntc_factored, guides_ntc):
    """
    Helper function to calculate mean per guide for mu_x_sd.

    Parameters
    ----------
    guide : str or int
        Guide identifier
    x_obs_ntc_factored : torch.Tensor
        Factored observations for NTC cells
    guides_ntc : torch.Tensor
        Guide assignments for NTC cells

    Returns
    -------
    torch.Tensor
        Mean value for the specified guide
    """
    mask = guides_ntc == guide
    return torch.mean(x_obs_ntc_factored[mask])


########################################
# Dose-Response Functions
########################################

def default_sigmoid_clamp(dtype):
    # log(max_float) is ~88.7 for float32, ~709.8 for float64
    log_max = torch.log(torch.tensor(torch.finfo(dtype).max))
    return float(0.75 * log_max)  # 0.75 is conservative

def Hill_based_positive_logK(x, Vmax, A, logK, n):
    tiny = torch.finfo(x.dtype).tiny
    x_safe = x.clamp_min(tiny)
    logit = n * (torch.log(x_safe) - logK)  # logK broadcasts across cells
    clamp_logit = default_sigmoid_clamp(logit.dtype)
    # Soft (tanh) squeeze instead of hard clamp: gradient is sech²(logit/c) ≥ sech²(1) ≈ 0.42
    # at the "boundary" |logit| = clamp_logit, never zero — fixes dead gradient for extreme n.
    # Behaviour is identical to hard clamp for |logit| << clamp_logit (typical operation).
    logit = clamp_logit * torch.tanh(logit / clamp_logit)
    return Vmax * torch.sigmoid(logit) + A


def Hill_based_positive(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Positive Hill equation: Vmax * (x^n / (K^n + x^n)) + A

    Used for modeling activation dose-response curves.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum response
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal response (EC50)
    n : torch.Tensor or float
        Hill coefficient (cooperativity)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon  # Ensure K is positive
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(n * x_log)
    K_n = torch.exp(n * K_log)
    denominator = K_n + x_n
    return Vmax * (x_n / denominator) + A


def Hill_based_negative(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Negative Hill equation: Vmax * (K^n / (K^n + x^n)) + A

    Used for modeling repression dose-response curves.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum repression
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal inhibition (IC50)
    n : torch.Tensor or float
        Hill coefficient (cooperativity)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(n * x_log)
    K_n = torch.exp(n * K_log)
    fraction = K_n / (K_n + x_n)
    return Vmax * fraction + A


def Hill_based_piecewise(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Piecewise Hill equation: switches between positive and negative based on sign of n.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum response
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal response
    n : torch.Tensor or float
        Hill coefficient (sign determines direction)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon  # Ensure K is positive
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(torch.abs(n) * x_log)
    K_n = torch.exp(torch.abs(n) * K_log)
    denominator = K_n + x_n
    fraction = torch.where(n < 0, K_n / denominator, x_n / denominator)
    return Vmax * fraction + A


def Polynomial_function(x, coeffs):
    """
    Polynomial dose-response function.

    Computes: sum_{i=1}^{degree} coeffs[i] * x^i

    Parameters
    ----------
    x : torch.Tensor
        Input values, shape [N] or [N, 1]
    coeffs : torch.Tensor
        Polynomial coefficients, shape [degree, T] or [S, degree, T]
        where T is number of features and S is number of samples

    Returns
    -------
    torch.Tensor
        Predicted response values
        - Shape [N, T] if coeffs is [degree, T]
        - Shape [S, N, T] if coeffs is [S, degree, T]
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    powers = torch.arange(1, coeffs.shape[-2] + 1, device=x.device).view(1, -1)  # [1, degree]
    x_powers = x ** powers  # [N, degree]

    if coeffs.dim() == 2:
        return x_powers @ coeffs  # [N, T]
    elif coeffs.dim() == 3:
        x_powers = x_powers.unsqueeze(0)  # [1, N, degree]
        return torch.matmul(x_powers, coeffs).squeeze(-2)  # [S, N, T]
    else:
        raise ValueError(f"Expected coeffs to have 2 or 3 dims, got shape {coeffs.shape}")


########################################
# Sigmoid and Cutoff Functions
########################################

def cutoff_sigmoid(x, threshold=0.1, slope=50.0):
    """
    Smooth cutoff function using sigmoid.

    Returns x * sigmoid(slope * (threshold - x))
    - Near threshold, the factor is ~0.5
    - For x << threshold, the factor approaches 1
    - For x >> threshold, the factor approaches 0

    Parameters
    ----------
    x : torch.Tensor
        Input values
    threshold : float
        Cutoff threshold
    slope : float
        Steepness of sigmoid transition

    Returns
    -------
    torch.Tensor
        Smoothly cutoff values
    """
    return x * torch.sigmoid(slope * (threshold - x))


########################################
# Tensor Utilities
########################################

def sample_or_use_point(name, value, device):
    """
    Handles whether `value` is a point estimate (float/tensor) or needs conversion.

    Parameters
    ----------
    name : str
        Name of the variable (for error messages)
    value : torch.Tensor, float, int, or np.ndarray
        Value to process
    device : torch.device
        Computation device (CPU/GPU)

    Returns
    -------
    torch.Tensor
        Processed tensor on specified device

    Raises
    ------
    TypeError
        If value is not a supported type
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)  # Treat as a fixed tensor (point estimate)
    elif isinstance(value, (int, float, np.ndarray)):  # Scalars or numpy arrays
        return torch.tensor(value, dtype=torch.float32, device=device)
    else:
        raise TypeError(f"Expected a tensor, float, or numpy array for {name}, but got {type(value)}.")


class SplitNormal(torch.distributions.Distribution):
    """
    Asymmetric (two-piece) Normal distribution.

    Mode at `loc`. Uses sigma_left for x < loc and sigma_right for x >= loc.
    Integrates to 1 because each half of a Normal centred at loc integrates to 0.5.

    Useful as a prior anchored at y_ntc with a longer left tail (down to Amean/2)
    and a short right tail (A rarely exceeds NTC mean).

    log_prob(x < loc)  = Normal(loc, sigma_left).log_prob(x)
    log_prob(x >= loc) = Normal(loc, sigma_right).log_prob(x)
    rsample: z ~ N(0,1), x = loc + (sigma_right if z>=0 else sigma_left) * z
    """
    arg_constraints = {
        'loc': torch.distributions.constraints.real,
        'sigma_left': torch.distributions.constraints.positive,
        'sigma_right': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, loc, sigma_left, sigma_right, validate_args=None):
        self.loc = loc
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        def _shape(x):
            return x.shape if isinstance(x, torch.Tensor) else torch.Size([])
        batch_shape = torch.broadcast_shapes(_shape(loc), _shape(sigma_left), _shape(sigma_right))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SplitNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.sigma_left = self.sigma_left.expand(batch_shape)
        new.sigma_right = self.sigma_right.expand(batch_shape)
        super(SplitNormal, new).__init__(batch_shape, validate_args=False)
        return new

    def log_prob(self, x):
        lp_left = torch.distributions.Normal(self.loc, self.sigma_left).log_prob(x)
        lp_right = torch.distributions.Normal(self.loc, self.sigma_right).log_prob(x)
        return torch.where(x < self.loc, lp_left, lp_right)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        sigma = torch.where(z >= 0, self.sigma_right, self.sigma_left)
        return self.loc + sigma * z

    def __call__(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


def check_tensor(name, tensor):
    """
    Debug utility to check tensor properties.

    Prints shape, min/max values, and checks for NaN/Inf.

    Parameters
    ----------
    name : str
        Name of the tensor (for display)
    tensor : torch.Tensor
        Tensor to check
    """
    print(f"--- {name} ---")
    print(f"  shape: {tensor.shape}")
    print(f"  min: {tensor.min().item()}, max: {tensor.max().item()}")
    print(f"  has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  has Inf: {torch.isinf(tensor).any().item()}")


########################################
# Lean-loaded posterior detection
########################################

LEAN_POSTERIOR_KEY = '__lean__'


def is_lean_posterior(posterior) -> bool:
    """
    Check whether a posterior_samples dict is lean-loaded.

    Lean loading (`load_ntc_fit(lean=True)` / `load_cis_fit(lean=True)`, see
    `bayesDREAM.io.load._reduce_posterior_samples`) collapses every tensor's
    sample axis to a point estimate (median, kept as a singleton leading dim)
    plus `<key>_lower`/`<key>_upper` (2.5%/97.5%) sibling keys, and discards
    the raw per-draw samples. Code that needs genuine multi-sample structure
    (histograms, KDE, prior/posterior comparisons, joint per-draw correlation
    across parameters) must check this first and refuse to run — see
    `require_full_posterior`. Code that only needs a point estimate + CI band
    can proceed either way.

    Parameters
    ----------
    posterior : dict or None
        A posterior_samples_ntc / posterior_samples_cis dict (or None).

    Returns
    -------
    bool
        True if lean-loaded, False for None, non-dict, or full posteriors.
    """
    return bool(isinstance(posterior, dict) and posterior.get(LEAN_POSTERIOR_KEY, False))


def require_full_posterior(posterior, context: str) -> None:
    """
    Raise a clear error if `posterior` is lean-loaded.

    Use this at the top of any plotting/analysis function that reads raw
    per-draw posterior samples for something other than a point estimate +
    95% CI band (e.g. histograms, KDE, prior/posterior overlays, joint
    per-draw correlation across parameters) — those silently produce a
    degenerate or misleading result on a lean-loaded posterior (only one
    "sample" remains) instead of erroring, which is worse than failing loudly.

    Parameters
    ----------
    posterior : dict or None
        A posterior_samples_ntc / posterior_samples_cis dict (or None).
    context : str
        Short description of what's being plotted/computed, used in the
        error message (e.g. "plot_prior_posterior_comparison").

    Raises
    ------
    ValueError
        If `posterior` is lean-loaded.
    """
    if is_lean_posterior(posterior):
        raise ValueError(
            f"{context} needs the full per-draw posterior samples, but this "
            f"model was loaded with lean=True (load_ntc_fit/load_cis_fit), "
            f"which collapses posterior_samples_ntc/posterior_samples_cis to "
            f"point estimates (median + 95% CI only) and discards the raw "
            f"draws. Reload with lean=False to use this plot."
        )
