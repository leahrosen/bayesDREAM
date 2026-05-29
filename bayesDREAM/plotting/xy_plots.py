"""
Raw x-y data plotting for bayesDREAM.

Plot relationships between cis gene expression (x_true) and modality values
with k-NN smoothing, technical group coloring, and optional technical correction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import scipy.sparse as _sp
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import torch

# Import dependency filtering from utils
from .utils import dependency_mask_from_n


# ============================================================================
# Helper Functions
# ============================================================================

# ============================================================================
# Cell Alignment Utilities
# ============================================================================

def _align_cells_to_modality(
    model,
    modality,
    x_true: np.ndarray,
    y_data: np.ndarray,
    subset_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align model.meta cells with modality cells, handling the case where
    the modality has fewer cells than model.meta.

    Parameters
    ----------
    model : bayesDREAM
        The model instance
    modality : Modality
        The modality instance
    x_true : np.ndarray
        x_true values for ALL cells in model.meta
    y_data : np.ndarray
        y values for cells in the modality (may be fewer than model.meta)
    subset_mask : Optional[np.ndarray]
        Optional boolean mask for subsetting (length = len(model.meta))

    Returns
    -------
    x_true_aligned : np.ndarray
        x_true values aligned to final cell set
    y_data_aligned : np.ndarray
        y values aligned to final cell set
    meta_aligned : pd.DataFrame
        Metadata aligned to final cell set
    """
    # Get cell identifiers
    if 'cell' in model.meta.columns:
        model_cells = model.meta['cell'].values
    else:
        model_cells = model.meta.index.values

    modality_cells = modality.cell_names
    if modality_cells is None:
        raise ValueError(f"Modality '{modality.name}' does not have cell_names set. Cannot align cells.")

    # Create mask for cells in model.meta that are also in modality
    modality_mask = np.isin(model_cells, modality_cells)

    # Combine with subset_mask if provided
    if subset_mask is not None:
        final_mask = modality_mask & subset_mask
    else:
        final_mask = modality_mask

    # Get final set of cells
    final_cells = model_cells[final_mask]

    # Subset x_true and metadata using final_mask
    x_true_aligned = x_true[final_mask]
    meta_aligned = model.meta[final_mask].copy()

    # Map final_cells to indices in modality
    # Create a lookup dictionary for efficiency
    modality_cell_to_idx = {cell: idx for idx, cell in enumerate(modality_cells)}
    y_indices = np.array([modality_cell_to_idx[cell] for cell in final_cells])

    # Subset y_data using these indices.
    # Scipy sparse matrix slices are always 2D, so indexing must match the cell axis:
    #   - shape (1, n_cells): row slice from a features×cells matrix → select columns
    #   - shape (n_cells, 1): col slice from a cells×features matrix → select rows, flatten
    if _sp.issparse(y_data):
        # np.asarray() on a sparse matrix gives a 0-d object array, not dense —
        # must use .toarray() to get a proper 2D ndarray before flattening.
        if y_data.ndim == 2 and y_data.shape[0] == 1:
            # features×cells orientation: cells are along axis 1
            y_data_aligned = y_data[:, y_indices].toarray().flatten()
        else:
            # cells×features orientation: cells are along axis 0
            y_data_aligned = y_data[y_indices].toarray().flatten()
    else:
        y_data_aligned = y_data[y_indices]

    return x_true_aligned, y_data_aligned, meta_aligned


# ============================================================================
# k-NN Smoothing Utilities
# ============================================================================

def _knn_k(n: int, window: Union[int, float]) -> int:
    """
    Compute k for k-NN smoothing.

    Parameters
    ----------
    n : int
        Number of data points
    window : int or float
        Window size (if int: exact k, if float: proportion of n)

    Returns
    -------
    int
        k value for k-NN (at least 1, at most n)
    """
    if n <= 0:
        return 1
    k = int(window) if isinstance(window, (int, np.integer)) else int(np.ceil(float(window) * n))
    return max(1, min(k, n))


def _smooth_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    is_ntc: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Centered sliding-window smoothing along x-axis.

    Uses a symmetric window of k cells (±k//2 around each point in sorted order).
    Points near the edges where a full symmetric window is not available are trimmed.

    Parameters
    ----------
    x : np.ndarray
        X values (1D)
    y : np.ndarray
        Y values (1D)
    k : int
        Total window size (k//2 cells on each side)
    is_ntc : np.ndarray, optional
        Boolean array indicating NTC cells (1D, same length as x)
        If provided, returns NTC proportions in windows

    Returns
    -------
    x_trimmed : np.ndarray
        Sorted x values (edges trimmed)
    y_hat : np.ndarray
        Smoothed y values
    ntc_prop : np.ndarray, optional
        If is_ntc provided: proportion of NTC cells in each window
    """
    if len(x) == 0:
        if is_ntc is not None:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    n = len(x)
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]

    k = max(1, min(k, n))
    half = k // 2

    # Trim edges where symmetric window is not possible
    lo = half
    hi = n - half  # exclusive
    if lo >= hi:
        # Not enough data for even one full window — fall back to full average
        if is_ntc is not None:
            return x_sorted, np.full(n, np.nanmean(y_sorted)), np.full(n, np.mean(np.asarray(is_ntc)[order]))
        return x_sorted, np.full(n, np.nanmean(y_sorted))

    n_out = hi - lo
    y_hat = np.empty(n_out, dtype=float)

    if is_ntc is not None:
        is_ntc_sorted = np.asarray(is_ntc)[order]
        ntc_prop = np.empty(n_out, dtype=float)

        for j, i in enumerate(range(lo, hi)):
            win = slice(i - half, i + half + 1)
            y_hat[j] = np.nanmean(y_sorted[win])
            ntc_prop[j] = np.mean(is_ntc_sorted[win])

        return x_sorted[lo:hi], y_hat, ntc_prop
    else:
        for j, i in enumerate(range(lo, hi)):
            win = slice(i - half, i + half + 1)
            y_hat[j] = np.nanmean(y_sorted[win])

        return x_sorted[lo:hi], y_hat


def _smooth_knn_counts(
    x: np.ndarray,
    numerators: np.ndarray,
    denominators: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centered sliding-window smoothing for count data by binning.

    Aggregates numerators and denominators within symmetric ±k//2 windows.
    Points near the edges where a full symmetric window is not available are trimmed.

    Parameters
    ----------
    x : np.ndarray
        X values (1D, length n)
    numerators : np.ndarray
        Numerator counts (1D for binomial: length n, or 2D for multinomial: n × K)
    denominators : np.ndarray
        Denominator totals (1D, length n)
    k : int
        Total window size (k//2 cells on each side)

    Returns
    -------
    x_trimmed : np.ndarray
        Sorted x values (edges trimmed)
    aggregated_num : np.ndarray
        Aggregated numerators (1D or 2D matching input shape)
    aggregated_denom : np.ndarray
        Aggregated denominators (1D)
    """
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    n = len(x)
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]

    k = max(1, min(k, n))
    half = k // 2

    # Trim edges where symmetric window is not possible
    lo = half
    hi = n - half  # exclusive
    if lo >= hi:
        # Not enough data — return full sums
        is_2d = numerators.ndim == 2
        if is_2d:
            return x_sorted, np.sum(np.asarray(numerators)[order, :], axis=0, keepdims=True).repeat(n, axis=0), np.full(n, np.sum(denominators))
        return x_sorted, np.full(n, np.sum(np.asarray(numerators)[order])), np.full(n, np.sum(denominators))

    n_out = hi - lo

    # Handle 1D (binomial) vs 2D (multinomial) numerators
    is_2d = numerators.ndim == 2
    if is_2d:
        num_sorted = np.asarray(numerators)[order, :]  # (n, K)
        K_dim = num_sorted.shape[1]
        aggregated_num = np.zeros((n_out, K_dim), dtype=float)
    else:
        num_sorted = np.asarray(numerators)[order]  # (n,)
        aggregated_num = np.zeros(n_out, dtype=float)

    denom_sorted = np.asarray(denominators)[order]
    aggregated_denom = np.zeros(n_out, dtype=float)

    for j, i in enumerate(range(lo, hi)):
        win = slice(i - half, i + half + 1)

        if is_2d:
            aggregated_num[j, :] = np.sum(num_sorted[win, :], axis=0)
        else:
            aggregated_num[j] = np.sum(num_sorted[win])

        aggregated_denom[j] = np.sum(denom_sorted[win])

    return x_sorted[lo:hi], aggregated_num, aggregated_denom


def _to_2d(a):
    """
    Convert input to 2D numpy array.

    Handles torch tensors and reshapes 1D arrays.
    """
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(float)
    except Exception:
        pass
    a = np.asarray(a)
    return a.reshape(1, -1).astype(float) if a.ndim == 1 else a.astype(float)


def reorder_xtrue_by_barcode(
    x_true: Union[np.ndarray, torch.Tensor],
    src_barcodes: np.ndarray,
    dst_barcodes: np.ndarray
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder x_true from source barcode order to destination barcode order.

    Parameters
    ----------
    x_true : array or tensor
        Values to reorder
    src_barcodes : np.ndarray
        Source barcode order
    dst_barcodes : np.ndarray
        Destination barcode order

    Returns
    -------
    array or tensor
        Reordered x_true
    """
    is_torch = False
    try:
        import torch
        if isinstance(x_true, torch.Tensor):
            is_torch = True
            x_np = x_true.detach().cpu().numpy()
            torch_dtype, torch_device = x_true.dtype, x_true.device
        else:
            x_np = np.asarray(x_true)
    except Exception:
        x_np = np.asarray(x_true)

    src = np.asarray(src_barcodes).astype(str)
    dst = np.asarray(dst_barcodes).astype(str)

    pos = {b: i for i, b in enumerate(src)}
    perm = np.fromiter((pos[b] for b in dst), dtype=int, count=len(dst))

    x_np = x_np if x_np.ndim == 1 else x_np[:, :]
    x_np = x_np[perm] if x_np.ndim == 1 else x_np[:, perm]

    if is_torch:
        import torch
        return torch.as_tensor(x_np, dtype=torch_dtype, device=torch_device)
    return x_np


# ============================================================================
# Feature Lookup Utilities
# ============================================================================

def _resolve_features(feature_or_gene: str, modality) -> Tuple[List[int], List[str], bool]:
    """
    Resolve input to feature index(es) and name(s).

    Returns
    -------
    feature_indices : List[int]
        Integer positions of features
    feature_names : List[str]
        Feature names
    is_gene : bool
        True if input was a gene name (multiple features), False if single feature

    Raises
    ------
    ValueError
        If feature/gene not found or modality doesn't have gene information
    """
    # First, try as a direct feature match
    feature_idx = _get_feature_index(feature_or_gene, modality)
    if feature_idx is not None:
        # Found as a feature - get the actual feature name
        if hasattr(modality, 'feature_names') and modality.feature_names is not None:
            feature_name = modality.feature_names[feature_idx]
        else:
            feature_name = feature_or_gene
        return [feature_idx], [feature_name], False

    # Not found as feature - try as gene
    # Check if modality has gene information
    gene_cols = ['gene', 'gene_name', 'gene_id']
    available_gene_cols = [col for col in gene_cols if col in modality.feature_meta.columns]

    if not available_gene_cols:
        raise ValueError(
            f"'{feature_or_gene}' not found as a feature, and modality '{modality.name}' "
            f"has no gene information (no 'gene', 'gene_name', or 'gene_id' columns in feature_meta). "
            f"Cannot search by gene name."
        )

    # Search for gene across all gene columns
    matching_features = []
    for gene_col in available_gene_cols:
        mask = modality.feature_meta[gene_col] == feature_or_gene
        if mask.sum() > 0:
            # Found matches
            indices = mask.values.nonzero()[0].tolist()

            # Get feature names for matched indices
            if hasattr(modality, 'feature_names') and modality.feature_names is not None:
                # Use explicit feature_names attribute if available
                names = [modality.feature_names[i] for i in indices]
            else:
                # Extract from feature_meta columns
                # Priority: feature_id > feature > gene_name > gene > junction coordinates > fallback to index
                name_cols = ['feature_id', 'feature', 'gene_name', 'gene', 'coord.intron', 'junction_id']
                name_col_found = None
                for col in name_cols:
                    if col in modality.feature_meta.columns:
                        name_col_found = col
                        break

                if name_col_found:
                    # Extract names from the identified column
                    names = modality.feature_meta.iloc[indices][name_col_found].tolist()
                elif modality.feature_meta.index.name:
                    # Use index if it has a name
                    names = modality.feature_meta.iloc[indices].index.tolist()
                else:
                    # Last resort: use str(index)
                    names = [str(i) for i in indices]

            return indices, names, True

    # Not found as feature or gene
    raise ValueError(
        f"'{feature_or_gene}' not found as a feature or gene in modality '{modality.name}'. "
        f"Checked: feature names/index and gene columns {available_gene_cols}"
    )


def _get_feature_index(feature: str, modality) -> Optional[int]:
    """
    Robustly find feature index, checking multiple locations.

    Checks feature_meta.index, feature_meta columns, and feature_names attribute.

    IMPORTANT: Does NOT check 'gene', 'gene_name', or 'gene_id' columns -
    those are reserved for gene-level searches in _resolve_features().

    Parameters
    ----------
    feature : str
        Feature name to find
    modality : Modality
        Modality object to search in

    Returns
    -------
    int or None
        Feature index if found, None otherwise
    """
    # Check index first
    if feature in modality.feature_meta.index:
        return modality.feature_meta.index.get_loc(feature)

    # Check common column names (EXCLUDING gene columns)
    for col in ['junction', 'feature', 'name', 'coord.intron', 'SJ']:
        if col in modality.feature_meta.columns:
            mask = modality.feature_meta[col] == feature
            if mask.sum() > 0:
                # Use argmax to get integer position, not index label
                return mask.values.argmax()

    # Check feature_names attribute
    if hasattr(modality, 'feature_names'):
        feature_names = modality.feature_names
        if isinstance(feature_names, (list, np.ndarray)):
            feature_names_list = list(feature_names)
            if feature in feature_names_list:
                return feature_names_list.index(feature)

    return None


def _to_scalar(val):
    """
    Convert value to Python scalar.

    Handles PyTorch tensors, numpy arrays, and Python scalars.

    Parameters
    ----------
    val : tensor, array, or scalar
        Value to convert

    Returns
    -------
    float
        Python scalar
    """
    # PyTorch tensor
    if hasattr(val, 'item'):
        return val.item()
    # Numpy array
    if isinstance(val, np.ndarray):
        return float(val)
    # Already scalar
    return float(val)

def _multinomial_correct_binned_probs(
    props_binned: np.ndarray,         # (n_bins, K) proportions from aggregated counts
    alpha_y_add,                      # [S, C, T, K] or [C, T, K]
    group_code: int,
    feature_idx: int,
    zero_cat_mask: np.ndarray         # (K,) True where category is globally zero
) -> np.ndarray:
    """
    Apply multinomial technical correction to binned proportions.

    For each bin: softmax(log(P) - alpha_s) for each posterior sample s,
    then average across s.

    Parameters
    ----------
    props_binned : np.ndarray
        Proportions computed from aggregated counts in k-NN bins (n_bins, K)
    alpha_y_add : torch.Tensor or np.ndarray
        Technical correction parameters [S, C, T, K] or [C, T, K]
    group_code : int
        Technical group code for this data
    feature_idx : int
        Feature index in alpha array
    zero_cat_mask : np.ndarray
        Boolean mask (K,) indicating globally absent categories

    Returns
    -------
    np.ndarray
        Corrected proportions (n_bins, K)
    """
    epsilon = 1e-6
    # Clip and log
    P = np.clip(props_binned, epsilon, 1 - epsilon)   # (n_bins, K)
    logP = np.log(P)                                  # (n_bins, K)

    # Pull alpha with explicit samples dim
    if alpha_y_add.ndim == 4:        # [S, C, T, K]
        A = alpha_y_add[:, group_code, feature_idx, :]          # (S, K)
    elif alpha_y_add.ndim == 3:      # [C, T, K] -> add S=1
        A = alpha_y_add[None, group_code, feature_idx, :]       # (1, K)
    else:
        raise ValueError(f"Unexpected alpha_y_add shape: {alpha_y_add.shape}")

    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()

    # Broadcast to (S, n_bins, K)
    logP_S = logP[None, :, :]
    A_S    = A[:, None, :]

    logits = logP_S - A_S                                 # (S, n_bins, K)

    # Hard-mask zero categories
    logits[:, :, zero_cat_mask] = -np.inf

    # Stable softmax with mask
    m = np.nanmax(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - m)
    exps[:, :, zero_cat_mask] = 0.0
    Z = exps.sum(axis=-1, keepdims=True)
    Z[Z == 0] = 1.0
    P_corr_S = exps / Z                                    # (S, n_bins, K)

    # Posterior mean in probability space
    return P_corr_S.mean(axis=0)                           # (n_bins, K)



# ============================================================================
# Technical Group Utilities
# ============================================================================

def get_technical_group_labels(model) -> List[str]:
    """
    Get informative labels for technical groups.

    Returns human-readable labels like "K562", "TF1:lane1" instead of
    generic "technical_group_0", "technical_group_1".

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model

    Returns
    -------
    List[str]
        Informative labels for each technical group code
    """
    if 'technical_group_code' not in model.meta.columns:
        return ['All']  # Single group when no technical groups

    # Get unique technical groups
    group_codes = sorted(model.meta['technical_group_code'].unique())

    # Determine which covariates were used
    # We need to reverse engineer this from the grouping
    # Try common covariate combinations
    potential_covariates = ['cell_line', 'lane', 'batch', 'replicate']
    available_covariates = [c for c in potential_covariates if c in model.meta.columns]

    if not available_covariates:
        # Fall back to generic labels
        return [f'Group_{i}' for i in group_codes]

    # Find which covariates were used by checking if they distinguish groups
    used_covariates = []
    for cov in available_covariates:
        test_grouping = model.meta.groupby(used_covariates + [cov]).ngroup()
        if test_grouping.nunique() == len(group_codes) and (test_grouping == model.meta['technical_group_code']).all():
            # This covariate set matches
            used_covariates.append(cov)
            break
        elif len(used_covariates) == 0:
            # Try single covariate
            test_grouping = model.meta.groupby([cov]).ngroup()
            if test_grouping.nunique() == len(group_codes) and (test_grouping == model.meta['technical_group_code']).all():
                used_covariates = [cov]
                break

    if not used_covariates:
        # Couldn't determine covariates, use generic labels
        return [f'Group_{i}' for i in group_codes]

    # Create labels
    labels = []
    for code in group_codes:
        mask = model.meta['technical_group_code'] == code
        group_data = model.meta.loc[mask, used_covariates].iloc[0]

        if len(used_covariates) == 1:
            # Single covariate: just use the value
            labels.append(str(group_data[used_covariates[0]]))
        else:
            # Multiple covariates: join with ":"
            parts = [str(group_data[cov]) for cov in used_covariates]
            labels.append(':'.join(parts))

    return labels


def get_default_color_palette(labels: List[str]) -> Dict[str, str]:
    """
    Get default color palette for technical groups.

    Parameters
    ----------
    labels : List[str]
        Technical group labels

    Returns
    -------
    Dict[str, str]
        Mapping from label to color
    """
    return {label: f'C{i}' for i, label in enumerate(labels)}


def _labels_by_code_for_df(model, df) -> dict[int, str]:
    """
    Return {technical_group_code -> human-readable label} *for codes present in df*.
    Prefers 'cell_line' if available, else falls back to a stable generic label.

    If technical_group_code doesn't exist, returns {0: 'All'} (single group).
    """
    # Handle case where technical groups don't exist
    if 'technical_group_code' not in df.columns:
        return {0: 'All'}

    present = np.sort(df['technical_group_code'].unique())
    label_col = 'cell_line' if 'cell_line' in model.meta.columns else None

    mapping = {}
    for code in present:
        if label_col is not None:
            vals = model.meta.loc[model.meta['technical_group_code'] == code, label_col]
            mapping[int(code)] = str(vals.iloc[0]) if len(vals) else f'Group_{int(code)}'
        else:
            mapping[int(code)] = f'Group_{int(code)}'
    return mapping


def _color_for_label(label: str, fallback_idx: int = 0, palette: dict | None = None) -> str:
    """
    Return colour for a technical group label, using user-supplied palette if provided.
    """
    palette = palette or {}
    if label in palette:
        return palette[label]
    return f'C{fallback_idx}'


def plot_colored_line(
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray,
    cmap,
    ax: plt.Axes,
    linewidth: float = 2
):
    """
    Plot a line with color gradient based on values.

    Parameters
    ----------
    x, y : np.ndarray
        Line coordinates
    color_values : np.ndarray
        Values for coloring (same length as x, y)
    cmap : colormap
        Matplotlib colormap
    ax : plt.Axes
        Axes to plot on
    linewidth : float
        Line width

    Returns
    -------
    LineCollection
        The created line collection
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=0, vmax=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_values)
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)

    return lc


# ============================================================================
# Trans Function Prediction Utilities (for overlaying fitted functions)
# ============================================================================

def Hill_based_positive(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Positive Hill function.

    Parameters
    ----------
    x : array
        Input values
    Vmax, A, K, n : float or array
        Hill parameters
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Hill function values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon
    x_log = np.log(x_safe)
    K_log = np.log(K_safe)
    x_n = np.exp(n * x_log)
    K_n = np.exp(n * K_log)
    denominator = K_n + x_n
    return Vmax * (x_n / denominator) + A


def Hill_first_derivative(x, Vmax, K, n, epsilon=1e-6):
    """
    First derivative of Hill function: d/dx [V * x^n / (K^n + x^n)]

    Formula: (K^n * V * n * x^(n-1)) / (K^n + x^n)^2

    Parameters
    ----------
    x : array
        Input values
    Vmax : float or array
        Maximum effect
    K : float or array
        Half-maximal concentration
    n : float or array
        Hill coefficient
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        First derivative values
    """
    x_safe = np.maximum(x, epsilon)
    K_safe = np.maximum(K, epsilon)

    x_n = np.power(x_safe, n)
    K_n = np.power(K_safe, n)
    x_nm1 = np.power(x_safe, n - 1)

    denom = np.power(K_n + x_n, 2)

    return (K_n * Vmax * n * x_nm1) / denom


def Hill_second_derivative(x, Vmax, K, n, epsilon=1e-6):
    """
    Second derivative of Hill function: d²/dx² [V * x^n / (K^n + x^n)]

    Formula: -(K^n * V * n * x^(n-2) * ((n+1)*x^n - K^n*(n-1))) / (K^n + x^n)^3

    Parameters
    ----------
    x : array
        Input values
    Vmax : float or array
        Maximum effect
    K : float or array
        Half-maximal concentration
    n : float or array
        Hill coefficient
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Second derivative values
    """
    x_safe = np.maximum(x, epsilon)
    K_safe = np.maximum(K, epsilon)

    x_n = np.power(x_safe, n)
    K_n = np.power(K_safe, n)
    x_nm2 = np.power(x_safe, n - 2)

    denom = np.power(K_n + x_n, 3)

    # Numerator: -K^n * V * n * x^(n-2) * ((n+1)*x^n - K^n*(n-1))
    inner_term = (n + 1) * x_n - K_n * (n - 1)
    numer = -K_n * Vmax * n * x_nm2 * inner_term

    return numer / denom


def log2fc_first_derivative(x, S, dS_dx, epsilon=1e-10):
    """
    First derivative in log2FC space: dg/du where g(u) = log2(S(x(u))) - y_ntc.

    With u = log2(x) - x_ntc, we have x = x_ntc * 2^u, so dx/du = x * ln(2).

    Formula: dg/du = x * S'(x) / S(x)

    Parameters
    ----------
    x : array
        Input x values (not log-transformed)
    S : array
        Function values S(x)
    dS_dx : array
        First derivative S'(x)
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        First derivative in log2FC space
    """
    S_safe = np.maximum(np.abs(S), epsilon)
    return x * dS_dx / S_safe


def log2fc_second_derivative(x, S, dS_dx, d2S_dx2, epsilon=1e-10):
    """
    Second derivative in log2FC space: d²g/du².

    Formula: d²g/du² = ln(2) * (x * S'/S + x² * S''/S - x² * (S'/S)²)

    Parameters
    ----------
    x : array
        Input x values (not log-transformed)
    S : array
        Function values S(x)
    dS_dx : array
        First derivative S'(x)
    d2S_dx2 : array
        Second derivative S''(x)
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Second derivative in log2FC space
    """
    S_safe = np.maximum(np.abs(S), epsilon)
    ln2 = np.log(2)

    term1 = x * dS_dx / S_safe
    term2 = x**2 * d2S_dx2 / S_safe
    term3 = x**2 * (dS_dx / S_safe)**2

    return ln2 * (term1 + term2 - term3)


def Hill_third_derivative(x, Vmax, K, n, epsilon=1e-6):
    """
    Third derivative of Hill function: d³/dx³ [V * x^n / (K^n + x^n)]

    Formula: (K^n * V * n * x^(n-3) * ((n²+3n+2)*x^(2n) + (4K^n - 4K^n*n²)*x^n
             + K^(2n)*n² - 3K^(2n)*n + 2K^(2n))) / (K^n + x^n)^4

    Parameters
    ----------
    x : array
        Input values
    Vmax : float or array
        Maximum effect
    K : float or array
        Half-maximal concentration
    n : float or array
        Hill coefficient
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Third derivative values
    """
    x_safe = np.maximum(x, epsilon)
    K_safe = np.maximum(K, epsilon)

    x_n = np.power(x_safe, n)
    x_2n = np.power(x_safe, 2 * n)
    K_n = np.power(K_safe, n)
    K_2n = np.power(K_safe, 2 * n)
    x_nm3 = np.power(x_safe, n - 3)

    denom = np.power(K_n + x_n, 4)

    # Numerator terms
    # (n² + 3n + 2) * x^(2n)
    term1 = (n**2 + 3*n + 2) * x_2n
    # (4K^n - 4K^n*n²) * x^n = 4K^n * (1 - n²) * x^n
    term2 = 4 * K_n * (1 - n**2) * x_n
    # K^(2n) * n² - 3K^(2n) * n + 2K^(2n) = K^(2n) * (n² - 3n + 2)
    term3 = K_2n * (n**2 - 3*n + 2)

    numer = K_n * Vmax * n * x_nm3 * (term1 + term2 + term3)

    return numer / denom


def log2fc_third_derivative(x, S, dS_dx, d2S_dx2, d3S_dx3, epsilon=1e-10):
    """
    Third derivative in log2FC space: d³g/du³.

    Derived by differentiating d²g/du² with respect to u:
    d³g/du³ = (ln(2))² * [x*S'/S + 3x²*S''/S - 3x²*(S'/S)²
                         + x³*S'''/S - 3x³*(S'/S)*(S''/S) + 2x³*(S'/S)³]

    Parameters
    ----------
    x : array
        Input x values (not log-transformed)
    S : array
        Function values S(x)
    dS_dx : array
        First derivative S'(x)
    d2S_dx2 : array
        Second derivative S''(x)
    d3S_dx3 : array
        Third derivative S'''(x)
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Third derivative in log2FC space
    """
    S_safe = np.maximum(np.abs(S), epsilon)
    ln2 = np.log(2)
    ln2_sq = ln2 ** 2

    S_ratio = dS_dx / S_safe      # S'/S
    S2_ratio = d2S_dx2 / S_safe   # S''/S
    S3_ratio = d3S_dx3 / S_safe   # S'''/S

    x2 = x ** 2
    x3 = x ** 3

    # x*S'/S
    term1 = x * S_ratio
    # 3x²*S''/S
    term2 = 3 * x2 * S2_ratio
    # -3x²*(S'/S)²
    term3 = -3 * x2 * S_ratio**2
    # x³*S'''/S
    term4 = x3 * S3_ratio
    # -3x³*(S'/S)*(S''/S)
    term5 = -3 * x3 * S_ratio * S2_ratio
    # 2x³*(S'/S)³
    term6 = 2 * x3 * S_ratio**3

    return ln2_sq * (term1 + term2 + term3 + term4 + term5 + term6)


def predict_trans_derivatives(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute first, second, and third derivatives of trans effect function.

    Only supports single_hill and additive_hill function types.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name to predict
    x_range : np.ndarray
        X values to compute derivatives at
    modality_name : str, optional
        Modality name (default: primary modality)

    Returns
    -------
    Tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None]
        (first_derivative, second_derivative, third_derivative) at x_range values,
        or (None, None, None) if trans model not fitted or unsupported function type

    Raises
    ------
    ValueError
        If function type is not single_hill or additive_hill
    """
    # Determine which modality to use
    if modality_name is None:
        modality_name = model.primary_modality

    # Check if trans model fitted for this modality
    if modality_name == model.primary_modality:
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            return None, None, None
        posterior = model.posterior_samples_trans
    else:
        modality = model.get_modality(modality_name)
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            return None, None, None
        posterior = modality.posterior_samples_trans

    # Get baseline A (present in all function types)
    if 'A' not in posterior:
        return None, None, None

    A_samples = posterior['A']

    # Get feature list — always from the modality, never from model.trans_genes
    modality = model.get_modality(modality_name)
    if modality.feature_names is not None:
        feature_list = modality.feature_names
    elif modality.feature_meta is not None:
        feature_list = None
        for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
            if col in modality.feature_meta.columns:
                feature_list = modality.feature_meta[col].tolist()
                break
        if feature_list is None:
            feature_list = modality.feature_meta.index.tolist()
    else:
        return None, None, None

    # Get dimensions
    if hasattr(A_samples, 'mean'):
        A_mean = A_samples.mean(dim=0)
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]
    else:
        A_mean = A_samples.mean(axis=0)
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]

    if n_genes_posterior != len(feature_list):
        return None, None, None

    if feature not in feature_list:
        return None, None, None

    feature_idx = feature_list.index(feature)

    # Helper function to extract parameter value
    def _extract_param(param_samples, feature_idx):
        if hasattr(param_samples, 'mean'):
            param_mean = param_samples.mean(dim=0)
        else:
            param_mean = param_samples.mean(axis=0)
        if param_mean.ndim > 1:
            param_mean = param_mean.squeeze(0)
        val = param_mean[feature_idx]
        return val.item() if hasattr(val, 'item') else val

    # Check function type and compute derivatives
    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL =====
        try:
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            Vmax_b = _extract_param(posterior['Vmax_b'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            # First derivatives
            dHill_a = Hill_first_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            dHill_b = Hill_first_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            first_deriv = alpha * dHill_a + beta * dHill_b

            # Second derivatives
            d2Hill_a = Hill_second_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            d2Hill_b = Hill_second_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            second_deriv = alpha * d2Hill_a + beta * d2Hill_b

            # Third derivatives
            d3Hill_a = Hill_third_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            d3Hill_b = Hill_third_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            third_deriv = alpha * d3Hill_a + beta * d3Hill_b

            return first_deriv, second_deriv, third_deriv

        except (KeyError, IndexError, AttributeError):
            return None, None, None

    elif 'upper_limit' in posterior and 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL (binomial/multinomial) =====
        try:
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            Vmax_b = _extract_param(posterior['Vmax_b'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            # First derivatives
            dHill_a = Hill_first_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            dHill_b = Hill_first_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            first_deriv = alpha * dHill_a + beta * dHill_b

            # Second derivatives
            d2Hill_a = Hill_second_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            d2Hill_b = Hill_second_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            second_deriv = alpha * d2Hill_a + beta * d2Hill_b

            # Third derivatives
            d3Hill_a = Hill_third_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)
            d3Hill_b = Hill_third_derivative(x_range, Vmax=Vmax_b, K=K_b, n=n_b)
            third_deriv = alpha * d3Hill_a + beta * d3Hill_b

            return first_deriv, second_deriv, third_deriv

        except (KeyError, IndexError, AttributeError):
            return None, None, None

    elif 'Vmax_a' in posterior:
        # ===== SINGLE HILL =====
        try:
            alpha = _extract_param(posterior['alpha'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)

            # First derivative
            first_deriv = alpha * Hill_first_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)

            # Second derivative
            second_deriv = alpha * Hill_second_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)

            # Third derivative
            third_deriv = alpha * Hill_third_derivative(x_range, Vmax=Vmax_a, K=K_a, n=n_a)

            return first_deriv, second_deriv, third_deriv

        except (KeyError, IndexError, AttributeError):
            return None, None, None

    elif 'theta' in posterior:
        # ===== POLYNOMIAL - NOT SUPPORTED =====
        raise ValueError(
            "Derivative plotting not supported for polynomial function type. "
            "Only single_hill and additive_hill are supported."
        )

    else:
        # Unknown function type
        return None, None, None


def predict_trans_log2fc(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    return_derivatives: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute trans effect function and derivatives in log2FC space.

    Transforms:
    - x-axis: u = log2(x) - log2(x_ntc)  (cis gene log2FC)
    - y-axis: g(u) = log2(S(x(u))) - log2(y_ntc)  (trans gene log2FC)

    Where x_ntc is the NTC mean for the cis gene, and y_ntc is the NTC mean for the trans gene.

    Derivative formulas:
    - dg/du = x * S'(x) / S(x)
    - d²g/du² = ln(2) * (x * S'/S + x² * S''/S - x² * (S'/S)²)
    - d³g/du³ = ln(2) * (x²*S'''/S - 3x²*S'*S''/S² + 3x*S''/S + 2x²*S'³/S³ - 3x*S'²/S² + S'/S)

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed and posterior_samples_technical available
    feature : str
        Feature name (trans gene) to predict
    x_range : np.ndarray
        X values (cis expression, NOT log-transformed)
    modality_name : str, optional
        Modality name (default: primary modality)
    return_derivatives : bool
        Whether to compute and return derivatives (default: True)

    Returns
    -------
    Tuple of (y_log2fc, u_range, first_deriv_log2fc, second_deriv_log2fc, third_deriv_log2fc)
        - y_log2fc: log2FC of trans gene relative to NTC
        - u_range: log2FC of cis gene relative to NTC
        - first_deriv_log2fc: dg/du (None if return_derivatives=False)
        - second_deriv_log2fc: d²g/du² (None if return_derivatives=False)
        - third_deriv_log2fc: d³g/du³ (None if return_derivatives=False)
        All are None if computation fails
    """
    # Get S(x) - the function values
    y_pred = predict_trans_function(model, feature, x_range, modality_name=modality_name)
    if y_pred is None:
        return None, None, None, None, None

    # Get NTC means
    # Cis NTC (for x-axis transformation)
    cis_mod = model.get_modality('cis')
    if cis_mod is None or not hasattr(cis_mod, 'posterior_samples_technical'):
        return None, None, None, None, None

    cis_mu_ntc = cis_mod.posterior_samples_technical.get('mu_ntc', None)
    if cis_mu_ntc is None:
        return None, None, None, None, None

    if hasattr(cis_mu_ntc, 'mean'):
        x_ntc = cis_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        x_ntc = np.mean(cis_mu_ntc, axis=0).squeeze()

    # Handle case where x_ntc is a scalar or 1-element array
    if np.ndim(x_ntc) == 0:
        x_ntc = float(x_ntc)
    elif len(x_ntc) == 1:
        x_ntc = float(x_ntc[0])
    else:
        x_ntc = float(x_ntc[0])  # Take first if multiple (shouldn't happen for cis)

    # Trans NTC (for y-axis transformation)
    if modality_name is None:
        modality_name = model.primary_modality
    trans_mod = model.get_modality(modality_name)

    if not hasattr(trans_mod, 'posterior_samples_technical'):
        return None, None, None, None, None

    trans_mu_ntc = trans_mod.posterior_samples_technical.get('mu_ntc', None)
    if trans_mu_ntc is None:
        return None, None, None, None, None

    if hasattr(trans_mu_ntc, 'mean'):
        y_ntc_all = trans_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        y_ntc_all = np.mean(trans_mu_ntc, axis=0).squeeze()

    # Find the feature index to get the right NTC
    # First priority: use modality.feature_names (this is what users see and should use)
    feature_list = None
    if trans_mod.feature_names is not None:
        feature_list = trans_mod.feature_names

    # Fallback: try feature_meta columns/index
    if feature_list is None or feature not in feature_list:
        feature_list = trans_mod.feature_meta.index.tolist() if trans_mod.feature_meta is not None else []
        if feature not in feature_list:
            # Try other column names
            for col in ['feature_id', 'feature', 'gene_name', 'gene']:
                if trans_mod.feature_meta is not None and col in trans_mod.feature_meta.columns:
                    feature_list = trans_mod.feature_meta[col].tolist()
                    if feature in feature_list:
                        break

    if feature not in feature_list:
        return None, None, None, None, None

    feature_idx = feature_list.index(feature)
    y_ntc = float(y_ntc_all[feature_idx]) if np.ndim(y_ntc_all) > 0 else float(y_ntc_all)

    # Compute log2FC transformations
    epsilon = 1e-10
    u_range = np.log2(np.maximum(x_range, epsilon)) - np.log2(max(x_ntc, epsilon))
    y_log2fc = np.log2(np.maximum(y_pred, epsilon)) - np.log2(max(y_ntc, epsilon))

    if not return_derivatives:
        return y_log2fc, u_range, None, None, None

    # Get S'(x), S''(x), and S'''(x)
    first_deriv, second_deriv, third_deriv = predict_trans_derivatives(model, feature, x_range, modality_name=modality_name)
    if first_deriv is None:
        return y_log2fc, u_range, None, None, None

    # Compute log2FC derivatives
    first_deriv_log2fc = log2fc_first_derivative(x_range, y_pred, first_deriv, epsilon)
    second_deriv_log2fc = log2fc_second_derivative(x_range, y_pred, first_deriv, second_deriv, epsilon)
    third_deriv_log2fc = log2fc_third_derivative(x_range, y_pred, first_deriv, second_deriv, third_deriv, epsilon)

    return y_log2fc, u_range, first_deriv_log2fc, second_deriv_log2fc, third_deriv_log2fc


def predict_trans_log2fc_samples(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute trans effect function in log2FC space for all posterior samples.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name (trans gene) to predict
    x_range : np.ndarray
        X values (cis expression, NOT log-transformed)
    modality_name : str, optional
        Modality name (default: primary modality)
    max_samples : int, optional
        Maximum number of samples to return

    Returns
    -------
    Tuple of (y_log2fc_samples, u_range)
        - y_log2fc_samples: [n_samples, n_points] log2FC predictions for each posterior sample
        - u_range: [n_points] log2FC of cis gene relative to NTC
        Both are None if computation fails
    """
    # Get all posterior samples for S(x)
    y_samples = predict_trans_function_samples(
        model, feature, x_range, modality_name=modality_name, max_samples=max_samples
    )
    if y_samples is None:
        return None, None

    # Get NTC means (same logic as predict_trans_log2fc)
    cis_mod = model.get_modality('cis')
    if cis_mod is None or not hasattr(cis_mod, 'posterior_samples_technical'):
        return None, None

    cis_mu_ntc = cis_mod.posterior_samples_technical.get('mu_ntc', None)
    if cis_mu_ntc is None:
        return None, None

    if hasattr(cis_mu_ntc, 'mean'):
        x_ntc = cis_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        x_ntc = np.mean(cis_mu_ntc, axis=0).squeeze()

    if np.ndim(x_ntc) == 0:
        x_ntc = float(x_ntc)
    elif len(x_ntc) == 1:
        x_ntc = float(x_ntc[0])
    else:
        x_ntc = float(x_ntc[0])

    # Trans NTC
    if modality_name is None:
        modality_name = model.primary_modality
    trans_mod = model.get_modality(modality_name)

    if not hasattr(trans_mod, 'posterior_samples_technical'):
        return None, None

    trans_mu_ntc = trans_mod.posterior_samples_technical.get('mu_ntc', None)
    if trans_mu_ntc is None:
        return None, None

    if hasattr(trans_mu_ntc, 'mean'):
        y_ntc_all = trans_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        y_ntc_all = np.mean(trans_mu_ntc, axis=0).squeeze()

    # Find feature index
    # First priority: use modality.feature_names (this is what users see and should use)
    feature_list = None
    if trans_mod.feature_names is not None:
        feature_list = trans_mod.feature_names

    # Fallback: try feature_meta columns/index
    if feature_list is None or feature not in feature_list:
        feature_list = trans_mod.feature_meta.index.tolist() if trans_mod.feature_meta is not None else []
        if feature not in feature_list:
            for col in ['feature_id', 'feature', 'gene_name', 'gene']:
                if trans_mod.feature_meta is not None and col in trans_mod.feature_meta.columns:
                    feature_list = trans_mod.feature_meta[col].tolist()
                    if feature in feature_list:
                        break

    if feature not in feature_list:
        return None, None

    feature_idx = feature_list.index(feature)
    y_ntc = float(y_ntc_all[feature_idx]) if np.ndim(y_ntc_all) > 0 else float(y_ntc_all)

    # Compute log2FC transformations for all samples
    epsilon = 1e-10
    u_range = np.log2(np.maximum(x_range, epsilon)) - np.log2(max(x_ntc, epsilon))

    # Transform each sample: log2(y_sample) - log2(y_ntc)
    y_log2fc_samples = np.log2(np.maximum(y_samples, epsilon)) - np.log2(max(y_ntc, epsilon))

    return y_log2fc_samples, u_range


def predict_trans_delta_p(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    return_derivatives: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute trans effect function in delta_p (probability difference) space for binomial modalities.

    Transforms:
    - x-axis: u = log2(x) - log2(x_ntc)  (cis gene log2FC)
    - y-axis: delta_p(u) = p(x(u)) - p_ntc  (probability difference from NTC)

    Where x_ntc is the NTC mean for the cis gene, and p_ntc is the NTC probability for the trans feature.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed and posterior_samples_technical available
    feature : str
        Feature name (trans gene/SJ) to predict
    x_range : np.ndarray
        X values (cis expression, NOT log-transformed)
    modality_name : str, optional
        Modality name (default: primary modality)
    return_derivatives : bool
        Whether to compute and return derivatives (default: True)
        Note: derivatives are dp/du, d²p/du², d³p/du³ (probability space)

    Returns
    -------
    Tuple of (delta_p, u_range, first_deriv, second_deriv, third_deriv)
        - delta_p: probability difference from NTC (p - p_ntc)
        - u_range: log2FC of cis gene relative to NTC
        - first_deriv: dp/du (None if return_derivatives=False)
        - second_deriv: d²p/du² (None if return_derivatives=False)
        - third_deriv: d³p/du³ (None if return_derivatives=False)
        All are None if computation fails
    """
    # Get p(x) - the probability values from fitted function
    y_pred = predict_trans_function(model, feature, x_range, modality_name=modality_name)
    if y_pred is None:
        return None, None, None, None, None

    # Get NTC means
    # Cis NTC (for x-axis transformation)
    cis_mod = model.get_modality('cis')
    if cis_mod is None or not hasattr(cis_mod, 'posterior_samples_technical'):
        return None, None, None, None, None

    cis_mu_ntc = cis_mod.posterior_samples_technical.get('mu_ntc', None)
    if cis_mu_ntc is None:
        return None, None, None, None, None

    if hasattr(cis_mu_ntc, 'mean'):
        x_ntc = cis_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        x_ntc = np.mean(cis_mu_ntc, axis=0).squeeze()

    # Handle case where x_ntc is a scalar or 1-element array
    if np.ndim(x_ntc) == 0:
        x_ntc = float(x_ntc)
    elif len(x_ntc) == 1:
        x_ntc = float(x_ntc[0])
    else:
        x_ntc = float(x_ntc[0])  # Take first if multiple

    # Trans NTC (for y-axis transformation)
    if modality_name is None:
        modality_name = model.primary_modality
    trans_mod = model.get_modality(modality_name)

    if not hasattr(trans_mod, 'posterior_samples_technical'):
        return None, None, None, None, None

    trans_mu_ntc = trans_mod.posterior_samples_technical.get('mu_ntc', None)
    if trans_mu_ntc is None:
        return None, None, None, None, None

    if hasattr(trans_mu_ntc, 'mean'):
        y_ntc_all = trans_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        y_ntc_all = np.mean(trans_mu_ntc, axis=0).squeeze()

    # Find the feature index to get the right NTC
    # First priority: use modality.feature_names (this is what users see and should use)
    feature_list = None
    if trans_mod.feature_names is not None:
        feature_list = trans_mod.feature_names

    # Fallback: try feature_meta columns/index
    if feature_list is None or feature not in feature_list:
        feature_list = trans_mod.feature_meta.index.tolist() if trans_mod.feature_meta is not None else []
        if feature not in feature_list:
            for col in ['feature_id', 'feature', 'gene_name', 'gene']:
                if trans_mod.feature_meta is not None and col in trans_mod.feature_meta.columns:
                    feature_list = trans_mod.feature_meta[col].tolist()
                    if feature in feature_list:
                        break

    if feature not in feature_list:
        return None, None, None, None, None

    feature_idx = feature_list.index(feature)
    p_ntc = float(y_ntc_all[feature_idx]) if np.ndim(y_ntc_all) > 0 else float(y_ntc_all)

    # Compute delta_p transformation
    epsilon = 1e-10
    u_range = np.log2(np.maximum(x_range, epsilon)) - np.log2(max(x_ntc, epsilon))
    delta_p = y_pred - p_ntc  # Simple difference for probability space

    if not return_derivatives:
        return delta_p, u_range, None, None, None

    # Get dp/dx derivatives
    first_deriv_dx, second_deriv_dx, third_deriv_dx = predict_trans_derivatives(
        model, feature, x_range, modality_name=modality_name
    )
    if first_deriv_dx is None:
        return delta_p, u_range, None, None, None

    # Convert dp/dx to dp/du using chain rule:
    # u = log2(x) - log2(x_ntc)
    # du/dx = 1/(x * ln(2))
    # dx/du = x * ln(2)
    # dp/du = dp/dx * dx/du = dp/dx * x * ln(2)
    ln2 = np.log(2.0)
    x_safe = np.maximum(x_range, epsilon)

    # First derivative: dp/du = ln(2) * x * dp/dx
    first_deriv = ln2 * x_safe * first_deriv_dx

    # Second derivative: d²p/du² = (ln2)² * x * (dp/dx + x * d²p/dx²)
    second_deriv = ln2**2 * x_safe * (first_deriv_dx + x_safe * second_deriv_dx)

    # Third derivative: d³p/du³ = (ln2)³ * x * (dp/dx + 3x * d²p/dx² + x² * d³p/dx³)
    third_deriv = ln2**3 * x_safe * (first_deriv_dx + 3 * x_safe * second_deriv_dx
                                      + x_safe**2 * third_deriv_dx)

    return delta_p, u_range, first_deriv, second_deriv, third_deriv


def predict_trans_delta_p_samples(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute trans effect function in delta_p space for all posterior samples.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name (trans gene/SJ) to predict
    x_range : np.ndarray
        X values (cis expression, NOT log-transformed)
    modality_name : str, optional
        Modality name (default: primary modality)
    max_samples : int, optional
        Maximum number of samples to return

    Returns
    -------
    Tuple of (delta_p_samples, u_range)
        - delta_p_samples: [n_samples, n_points] delta_p predictions for each posterior sample
        - u_range: [n_points] log2FC of cis gene relative to NTC
        Both are None if computation fails
    """
    # Get all posterior samples for p(x)
    y_samples = predict_trans_function_samples(
        model, feature, x_range, modality_name=modality_name, max_samples=max_samples
    )
    if y_samples is None:
        return None, None

    # Get NTC means (same logic as predict_trans_delta_p)
    cis_mod = model.get_modality('cis')
    if cis_mod is None or not hasattr(cis_mod, 'posterior_samples_technical'):
        return None, None

    cis_mu_ntc = cis_mod.posterior_samples_technical.get('mu_ntc', None)
    if cis_mu_ntc is None:
        return None, None

    if hasattr(cis_mu_ntc, 'mean'):
        x_ntc = cis_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        x_ntc = np.mean(cis_mu_ntc, axis=0).squeeze()

    if np.ndim(x_ntc) == 0:
        x_ntc = float(x_ntc)
    elif len(x_ntc) == 1:
        x_ntc = float(x_ntc[0])
    else:
        x_ntc = float(x_ntc[0])

    # Trans NTC
    if modality_name is None:
        modality_name = model.primary_modality
    trans_mod = model.get_modality(modality_name)

    if not hasattr(trans_mod, 'posterior_samples_technical'):
        return None, None

    trans_mu_ntc = trans_mod.posterior_samples_technical.get('mu_ntc', None)
    if trans_mu_ntc is None:
        return None, None

    if hasattr(trans_mu_ntc, 'mean'):
        y_ntc_all = trans_mu_ntc.mean(dim=0).squeeze().cpu().numpy()
    else:
        y_ntc_all = np.mean(trans_mu_ntc, axis=0).squeeze()

    # Find feature index
    # First priority: use modality.feature_names (this is what users see and should use)
    feature_list = None
    if trans_mod.feature_names is not None:
        feature_list = trans_mod.feature_names

    # Fallback: try feature_meta columns/index
    if feature_list is None or feature not in feature_list:
        feature_list = trans_mod.feature_meta.index.tolist() if trans_mod.feature_meta is not None else []
        if feature not in feature_list:
            for col in ['feature_id', 'feature', 'gene_name', 'gene']:
                if trans_mod.feature_meta is not None and col in trans_mod.feature_meta.columns:
                    feature_list = trans_mod.feature_meta[col].tolist()
                    if feature in feature_list:
                        break

    if feature not in feature_list:
        return None, None

    feature_idx = feature_list.index(feature)
    p_ntc = float(y_ntc_all[feature_idx]) if np.ndim(y_ntc_all) > 0 else float(y_ntc_all)

    # Compute delta_p transformations for all samples
    epsilon = 1e-10
    u_range = np.log2(np.maximum(x_range, epsilon)) - np.log2(max(x_ntc, epsilon))

    # Transform each sample: p_sample - p_ntc
    delta_p_samples = y_samples - p_ntc

    return delta_p_samples, u_range


def plot_trans_functions(
    model,
    features: Union[str, List[str]],
    modality_name: Optional[str] = None,
    show_function: bool = True,
    show_first_derivative: bool = False,
    show_second_derivative: bool = False,
    show_third_derivative: bool = False,
    x_range: Optional[np.ndarray] = None,
    n_points: int = 2000,
    use_log2_x: bool = True,
    use_log2fc: bool = False,
    use_delta_p: bool = False,
    show_posterior_samples: bool = False,
    show_ci: bool = False,
    posterior_alpha: float = 0.1,
    ci_alpha: float = 0.3,
    max_posterior_samples: int = 1000,
    colors: Optional[Union[str, List[str], Dict[str, str]]] = None,
    alpha: float = 0.8,
    linewidth: float = 1.5,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    overlay_roots=None,
    overlay_roots_lw: float = 1.0,
    overlay_roots_alpha: float = 0.8,
    overlay_roots_also_on_function: bool = True,
) -> plt.Figure:
    """
    Plot fitted trans functions and/or their derivatives.

    Simple plot showing just the fitted Hill functions (no smoothed data).
    Useful for comparing multiple genes or viewing function shape with derivatives.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    features : str or list of str
        Single feature name or list of feature names to plot
    modality_name : str, optional
        Modality name (default: primary modality)
    show_function : bool
        Show the fitted function y(x) (default: True)
    show_first_derivative : bool
        Show first derivative dy/dx (default: False)
    show_second_derivative : bool
        Show second derivative d²y/dx² (default: False)
    show_third_derivative : bool
        Show second derivative d3y/dx3 (default: False)
    x_range : np.ndarray, optional
        X values to plot at. If None, generates evenly spaced points in log2 space
        from model's x_true range.
    n_points : int
        Number of points for x_range if auto-generated (default: 2000).
        Points are evenly spaced in log2 space for smooth curves on log-log plots.
    use_log2_x : bool
        Use log2(x) for x-axis (default: True). Ignored if use_log2fc=True.
    use_log2fc : bool
        If True, plot in log2 fold-change space relative to NTC (default: False).
        - x-axis: log2FC = log2(x) - log2(x_ntc) where x_ntc is cis gene NTC mean
        - y-axis: log2FC = log2(y) - log2(y_ntc) where y_ntc is trans gene NTC mean
        - Derivatives: dg/du and d²g/du² (chain rule transformed)
        Requires posterior_samples_technical to be available for both cis and trans modalities.
        Not recommended for binomial modalities (use use_delta_p instead).
    use_delta_p : bool
        If True, plot in probability difference space relative to NTC (default: False).
        Designed for binomial modalities (e.g., splicing_sj).
        - x-axis: log2FC = log2(x) - log2(x_ntc) where x_ntc is cis gene NTC mean
        - y-axis: Δp = p - p_ntc where p is probability and p_ntc is NTC probability
        - Derivatives: dp/du and d²p/du² (chain rule transformed)
        Requires posterior_samples_technical to be available for both cis and trans modalities.
        Mutually exclusive with use_log2fc.
    show_posterior_samples : bool
        If True, plot individual posterior fits behind the mean line (default: False).
        Each posterior sample is plotted with transparency set by `posterior_alpha`.
    show_ci : bool
        If True, show 95% credible interval band around the mean line (default: False).
        The CI is computed at each x point and shown as a shaded region.
    posterior_alpha : float
        Transparency for individual posterior sample lines (default: 0.1).
        Only used when show_posterior_samples=True.
    ci_alpha : float
        Transparency for the 95% CI shaded region (default: 0.3).
        Only used when show_ci=True.
    max_posterior_samples : int
        Maximum number of posterior samples to plot (default: 1000).
        Only used when show_posterior_samples=True.
    colors : str, list, or dict, optional
        Colors for each feature. Can be:
        - Single color string (all features same color)
        - List of colors (one per feature)
        - Dict mapping feature names to colors
        If None, uses default color cycle.
    alpha : float
        Line transparency (default: 0.8)
    linewidth : float
        Line width (default: 1.5)
    figsize : tuple, optional
        Figure size (width, height). Auto-sized if None.
    title : str, optional
        Plot title. If None, auto-generated.
    legend : bool
        Show legend (default: True)
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    overlay_roots : pd.DataFrame, pd.Series, dict, or None
        If provided, draws dashed vertical lines at derivative roots on each
        subplot.  Accepted forms:

        - **pd.DataFrame** (full summary from ``save_trans_summary``): for each
          feature being plotted, the matching row is looked up by the ``feature``
          column.
        - **pd.Series / dict** (a single row): used as-is for every feature.
        - **None** (default): no overlay.

        Root columns are selected automatically to match the plot's x-axis space:
        ``*_log2fc_mean`` when ``use_log2fc=True``, ``*_delta_p_mean`` when
        ``use_delta_p=True``, and ``*_mean`` (x-space, converted to log2 if
        ``use_log2_x=True``) otherwise.
    overlay_roots_lw : float
        Line width for root vlines (default 1.0).
    overlay_roots_alpha : float
        Transparency for root vlines (default 0.8).
    overlay_roots_also_on_function : bool
        If True (default), draw all root sets also on the function subplot.

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Raises
    ------
    ValueError
        If function type is polynomial (derivatives not supported)
        If no features could be plotted
        If use_log2fc=True but NTC means not available
        If use_delta_p=True but NTC means not available
        If both use_log2fc and use_delta_p are True (mutually exclusive)

    Examples
    --------
    >>> # Plot function and derivatives for one gene
    >>> model.plot_trans_functions('TET2', show_first_derivative=True,
    ...                            show_second_derivative=True)

    >>> # Plot with derivative roots overlaid from summary df
    >>> df = model.save_trans_summary(compute_derivative_roots=True)
    >>> model.plot_trans_functions('TET2', use_log2fc=True,
    ...                            show_function=True,
    ...                            show_first_derivative=True,
    ...                            show_second_derivative=True,
    ...                            show_third_derivative=True,
    ...                            overlay_roots=df)

    >>> # Plot first derivative of multiple genes
    >>> model.plot_trans_functions(['TET2', 'MYB', 'GFI1B'],
    ...                            show_function=False,
    ...                            show_first_derivative=True)

    >>> # Plot all non-monotonic genes (genes where derivative changes sign)
    >>> feature_names = model.get_modality(model.primary_modality).feature_names
    >>> non_monotonic = [g for g in feature_names if is_non_monotonic(model, g)]
    >>> model.plot_trans_functions(non_monotonic, show_first_derivative=True)
    """
    import matplotlib.pyplot as plt

    # Ensure at least one thing to plot
    if not (show_function or show_first_derivative or show_second_derivative or show_third_derivative):
        raise ValueError("At least one of show_function, show_first_derivative, "
                        "show_second_derivative, or show_third_derivative must be True")

    # Validate mutually exclusive options
    if use_log2fc and use_delta_p:
        raise ValueError("use_log2fc and use_delta_p are mutually exclusive. "
                        "Use use_log2fc for negbinom modalities, use_delta_p for binomial modalities.")

    # Convert single feature to list
    if isinstance(features, str):
        features = [features]

    # Determine modality
    if modality_name is None:
        modality_name = model.primary_modality

    # Get x_range from model if not provided
    if x_range is None:
        if hasattr(model, 'x_true') and model.x_true is not None:
            x_true_np = model.x_true.cpu().numpy() if hasattr(model.x_true, 'cpu') else np.array(model.x_true)
            if x_true_np.ndim > 1:
                x_true_np = x_true_np.mean(axis=0)  # Average over posterior samples
            x_min = max(x_true_np.min(), 1e-6)  # Avoid log(0)
            x_max = x_true_np.max()
            # Evenly spaced points in log2 space for smooth curve on log-log plot
            log2_min = np.log2(x_min)
            log2_max = np.log2(x_max)
            x_range = 2 ** np.linspace(log2_min, log2_max, n_points)
        else:
            raise ValueError("x_range must be provided if model.x_true is not set")

    # Check if NTC data is available for log2FC/delta_p; fall back to log2 if not
    if use_log2fc or use_delta_p:
        cis_mod = model.get_modality('cis')
        ntc_available = (
            cis_mod is not None
            and hasattr(cis_mod, 'posterior_samples_technical')
            and cis_mod.posterior_samples_technical is not None
            and 'mu_ntc' in cis_mod.posterior_samples_technical
        )
        if not ntc_available:
            import warnings
            flag = 'use_log2fc' if use_log2fc else 'use_delta_p'
            warnings.warn(
                f"{flag}=True requested but NTC technical fit (posterior_samples_technical) "
                f"is not available. Falling back to log2(x) x-axis.",
                UserWarning
            )
            use_log2fc = False
            use_delta_p = False
            use_log2_x = True

    # Setup colors
    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_list = [color_cycle[i % len(color_cycle)] for i in range(len(features))]
    elif isinstance(colors, str):
        colors_list = [colors] * len(features)
    elif isinstance(colors, dict):
        colors_list = [colors.get(f, 'gray') for f in features]
    else:
        colors_list = list(colors)

    # Count how many subplots we need
    n_plots = sum([show_function, show_first_derivative, show_second_derivative, show_third_derivative])

    # Create figure if no axes provided
    if ax is None:
        if figsize is None:
            figsize = (5 * n_plots, 4)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
        axes = axes[0]  # Flatten to 1D
    else:
        if n_plots > 1:
            raise ValueError("Cannot use single ax with multiple subplots. "
                           "Pass ax=None to create new figure.")
        fig = ax.figure
        axes = [ax]

    # Track which plot index we're on
    plot_idx = 0
    plot_types = []

    # Set up plot types with appropriate labels
    if use_log2fc:
        if show_function:
            plot_types.append(('function', 'log₂FC (y)'))
        if show_first_derivative:
            plot_types.append(('first_deriv', "dg/du"))
        if show_second_derivative:
            plot_types.append(('second_deriv', "d²g/du²"))
        if show_third_derivative:
            plot_types.append(('third_deriv', "d³g/du³"))
    elif use_delta_p:
        if show_function:
            plot_types.append(('function', 'Δp'))
        if show_first_derivative:
            plot_types.append(('first_deriv', "dp/du"))
        if show_second_derivative:
            plot_types.append(('second_deriv', "d²p/du²"))
        if show_third_derivative:
            plot_types.append(('third_deriv', "d³p/du³"))
    else:
        if show_function:
            plot_types.append(('function', 'y'))
        if show_first_derivative:
            plot_types.append(('first_deriv', "dy/dx"))
        if show_second_derivative:
            plot_types.append(('second_deriv', "d²y/dx²"))
        if show_third_derivative:
            plot_types.append(('third_deriv', "d³y/dx³"))

    # Track successful plots
    successful_features = []

    for feat_idx, feature in enumerate(features):
        color = colors_list[feat_idx % len(colors_list)]

        # Get function and derivatives (mean values)
        try:
            if use_log2fc:
                # Get log2FC transformed values
                y_log2fc, u_range, first_deriv, second_deriv, third_deriv = predict_trans_log2fc(
                    model, feature, x_range, modality_name=modality_name, return_derivatives=True
                )
                if y_log2fc is None:
                    continue
                y_pred = y_log2fc
                x_plot_feat = u_range  # Use u (log2FC of x) for this feature
            elif use_delta_p:
                # Get delta_p transformed values (probability difference from NTC)
                delta_p, u_range, first_deriv, second_deriv, third_deriv = predict_trans_delta_p(
                    model, feature, x_range, modality_name=modality_name, return_derivatives=True
                )
                if delta_p is None:
                    continue
                y_pred = delta_p
                x_plot_feat = u_range  # Use u (log2FC of x) for this feature
            else:
                y_pred = predict_trans_function(model, feature, x_range, modality_name=modality_name)
                first_deriv, second_deriv, third_deriv = predict_trans_derivatives(model, feature, x_range, modality_name=modality_name)
                # X values for plotting (same for all features when not using log2fc/delta_p)
                x_plot_feat = np.log2(x_range) if use_log2_x else x_range
        except ValueError as e:
            # Polynomial not supported
            raise e

        if y_pred is None and first_deriv is None:
            continue  # Skip features that couldn't be computed

        successful_features.append(feature)

        # Get posterior samples if needed for uncertainty visualization
        y_samples = None
        first_deriv_samples = None
        second_deriv_samples = None
        third_deriv_samples = None

        if show_posterior_samples or show_ci:
            # Get derivative samples if we need them
            need_derivs = show_first_derivative or show_second_derivative or show_third_derivative

            if use_delta_p:
                # Get function samples in delta_p space
                if show_function:
                    delta_p_samples, _ = predict_trans_delta_p_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )
                    y_samples = delta_p_samples

                # For derivatives in delta_p mode, we need p(x), dp/dx samples
                # then transform using chain rule: dp/du = x * ln(2) * dp/dx
                if need_derivs:
                    S_samples, dS_samples, d2S_samples, d3S_samples = predict_trans_derivatives_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )
                    if S_samples is not None and dS_samples is not None:
                        # Transform derivatives to delta_p u-space using chain rule
                        # dp/du = x * ln(2) * dp/dx
                        # d²p/du² = ln(2) * (x * dp/dx + x² * ln(2) * d²p/dx²)
                        epsilon = 1e-10
                        ln2 = np.log(2)

                        # First derivative: dp/du = ln(2) * x * dp/dx
                        first_deriv_samples = ln2 * x_range[np.newaxis, :] * dS_samples

                        # Second derivative: d²p/du² = (ln2)² * x * (dp/dx + x * d²p/dx²)
                        if d2S_samples is not None:
                            x_arr = x_range[np.newaxis, :]
                            second_deriv_samples = (ln2**2) * x_arr * (dS_samples + x_arr * d2S_samples)

                        # Third derivative: d³p/du³ = (ln2)³ * x * (dp/dx + 3x*d²p/dx² + x²*d³p/dx³)
                        if d3S_samples is not None and d2S_samples is not None:
                            x_arr = x_range[np.newaxis, :]
                            third_deriv_samples = (ln2**3) * x_arr * (dS_samples + 3 * x_arr * d2S_samples
                                                                       + x_arr**2 * d3S_samples)

            elif use_log2fc:
                # Get function samples in log2FC space
                if show_function:
                    y_log2fc_samples, _ = predict_trans_log2fc_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )
                    y_samples = y_log2fc_samples

                # For derivatives in log2FC mode, we need S(x), S'(x), S''(x), S'''(x) samples
                # then transform using chain rule
                if need_derivs:
                    S_samples, dS_samples, d2S_samples, d3S_samples = predict_trans_derivatives_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )
                    if S_samples is not None and dS_samples is not None:
                        # Transform derivatives to log2FC space using chain rule
                        # dg/du = x * S'(x) / S(x)
                        # d²g/du² = ln(2) * (x * S'/S + x² * S''/S - x² * (S'/S)²)
                        epsilon = 1e-10
                        S_safe = np.maximum(np.abs(S_samples), epsilon)
                        ln2 = np.log(2)

                        # First derivative: dg/du = x * S'(x) / S(x)
                        first_deriv_samples = x_range[np.newaxis, :] * dS_samples / S_safe

                        # Second derivative: d²g/du² = ln(2) * (x*S'/S + x²*S''/S - x²*(S'/S)²)
                        if d2S_samples is not None:
                            term1 = x_range[np.newaxis, :] * dS_samples / S_safe
                            term2 = (x_range[np.newaxis, :] ** 2) * d2S_samples / S_safe
                            term3 = (x_range[np.newaxis, :] ** 2) * (dS_samples / S_safe) ** 2
                            second_deriv_samples = ln2 * (term1 + term2 - term3)

                        # Third derivative: d³g/du³ = (ln(2))² * [x*S'/S + 3x²*S''/S - 3x²*(S'/S)²
                        #                                         + x³*S'''/S - 3x³*(S'/S)*(S''/S) + 2x³*(S'/S)³]
                        if d3S_samples is not None and d2S_samples is not None:
                            term1 = x_range[np.newaxis, :] * (dS_samples / S_safe)
                            term2 = 3 * (x_range[np.newaxis, :] ** 2) * d2S_samples / S_safe
                            term3 = -3 * (x_range[np.newaxis, :] ** 2) * (dS_samples / S_safe) ** 2
                            term4 = (x_range[np.newaxis, :] ** 3) * (d3S_samples / S_safe)
                            term5 = -3 * (x_range[np.newaxis, :] ** 3) * (dS_samples / S_safe) * d2S_samples / S_safe
                            term6 = 2 * (x_range[np.newaxis, :] ** 3) * (dS_samples / S_safe) ** 3
                            third_deriv_samples = (ln2**2) * (term1 + term2 + term3 + term4 + term5 + term6)
            else:
                # Non-log2FC mode
                if show_function:
                    y_samples = predict_trans_function_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )

                if need_derivs:
                    _, first_deriv_samples, second_deriv_samples, third_deriv_samples = predict_trans_derivatives_samples(
                        model, feature, x_range,
                        modality_name=modality_name,
                        max_samples=max_posterior_samples
                    )

        # Plot each requested type
        for plot_i, (plot_type, ylabel) in enumerate(plot_types):
            ax_curr = axes[plot_i]

            if plot_type == 'function' and y_pred is not None:
                # Plot uncertainty visualization first (so mean line is on top)
                if y_samples is not None:
                    if show_posterior_samples:
                        for s in range(y_samples.shape[0]):
                            ax_curr.plot(x_plot_feat, y_samples[s, :], color=color,
                                       alpha=posterior_alpha, linewidth=0.5)
                    if show_ci:
                        y_lower = np.percentile(y_samples, 2.5, axis=0)
                        y_upper = np.percentile(y_samples, 97.5, axis=0)
                        ax_curr.fill_between(x_plot_feat, y_lower, y_upper,
                                           color=color, alpha=ci_alpha, linewidth=0)

                # Plot mean line on top
                ax_curr.plot(x_plot_feat, y_pred, color=color, alpha=alpha,
                           linewidth=linewidth, label=feature if feat_idx < 20 else None)

            elif plot_type == 'first_deriv' and first_deriv is not None:
                # Plot uncertainty for first derivative
                if first_deriv_samples is not None:
                    if show_posterior_samples:
                        for s in range(first_deriv_samples.shape[0]):
                            ax_curr.plot(x_plot_feat, first_deriv_samples[s, :], color=color,
                                       alpha=posterior_alpha, linewidth=0.5)
                    if show_ci:
                        d1_lower = np.percentile(first_deriv_samples, 2.5, axis=0)
                        d1_upper = np.percentile(first_deriv_samples, 97.5, axis=0)
                        ax_curr.fill_between(x_plot_feat, d1_lower, d1_upper,
                                           color=color, alpha=ci_alpha, linewidth=0)

                ax_curr.plot(x_plot_feat, first_deriv, color=color, alpha=alpha,
                           linewidth=linewidth, label=feature if feat_idx < 20 else None)

            elif plot_type == 'second_deriv' and second_deriv is not None:
                # Plot uncertainty for second derivative
                if second_deriv_samples is not None:
                    if show_posterior_samples:
                        for s in range(second_deriv_samples.shape[0]):
                            ax_curr.plot(x_plot_feat, second_deriv_samples[s, :], color=color,
                                       alpha=posterior_alpha, linewidth=0.5)
                    if show_ci:
                        d2_lower = np.percentile(second_deriv_samples, 2.5, axis=0)
                        d2_upper = np.percentile(second_deriv_samples, 97.5, axis=0)
                        ax_curr.fill_between(x_plot_feat, d2_lower, d2_upper,
                                           color=color, alpha=ci_alpha, linewidth=0)

                ax_curr.plot(x_plot_feat, second_deriv, color=color, alpha=alpha,
                           linewidth=linewidth, label=feature if feat_idx < 20 else None)

            elif plot_type == 'third_deriv' and third_deriv is not None:
                # Plot uncertainty for third derivative
                if third_deriv_samples is not None:
                    if show_posterior_samples:
                        for s in range(third_deriv_samples.shape[0]):
                            ax_curr.plot(x_plot_feat, third_deriv_samples[s, :], color=color,
                                       alpha=posterior_alpha, linewidth=0.5)
                    if show_ci:
                        d3_lower = np.percentile(third_deriv_samples, 2.5, axis=0)
                        d3_upper = np.percentile(third_deriv_samples, 97.5, axis=0)
                        ax_curr.fill_between(x_plot_feat, d3_lower, d3_upper,
                                           color=color, alpha=ci_alpha, linewidth=0)

                ax_curr.plot(x_plot_feat, third_deriv, color=color, alpha=alpha,
                           linewidth=linewidth, label=feature if feat_idx < 20 else None)

        # ---- Root overlay ------------------------------------------------
        if overlay_roots is not None:
            import pandas as _pd
            from bayesDREAM.io.summary import ModelSummarizer as _MS

            # Resolve the row for this feature
            if isinstance(overlay_roots, _pd.DataFrame):
                if 'feature' in overlay_roots.columns:
                    _matches = overlay_roots[overlay_roots['feature'] == feature]
                    _row = _matches.iloc[0] if len(_matches) > 0 else None
                else:
                    _row = None
            else:
                # Series or dict — use directly for every feature
                _row = overlay_roots

            if _row is not None:
                _parse = _MS._parse_semicolon_roots

                if use_log2fc:
                    _r1 = _parse(_row.get('first_deriv_roots_log2fc_mean',  None))
                    _r2 = _parse(_row.get('second_deriv_roots_log2fc_mean', None))
                    _r3 = _parse(_row.get('third_deriv_roots_log2fc_mean',  None))
                    _to_plot_x = lambda v: v  # already u-space
                elif use_delta_p:
                    _r1 = _parse(_row.get('first_deriv_roots_delta_p_mean',  None))
                    _r2 = _parse(_row.get('second_deriv_roots_delta_p_mean', None))
                    _r3 = _parse(_row.get('third_deriv_roots_delta_p_mean',  None))
                    _to_plot_x = lambda v: v  # already u-space
                else:
                    _r1 = _parse(_row.get('first_deriv_roots_mean',  None))
                    _r2 = _parse(_row.get('second_deriv_roots_mean', None))
                    _r3 = _parse(_row.get('third_deriv_roots_mean',  None))
                    _to_plot_x = (lambda v: np.log2(max(v, 1e-300))) if use_log2_x else (lambda v: v)

                # Build a map: plot_type → (ax, roots, label)
                _ax_for = {pt: axes[i] for i, (pt, _) in enumerate(plot_types)}
                _root_sets = [
                    ('first_deriv',  _r1, 'roots: 1st deriv'),
                    ('second_deriv', _r2, 'roots: 2nd deriv'),
                    ('third_deriv',  _r3, 'roots: 3rd deriv'),
                ]

                def _draw_vlines(ax, roots, label):
                    first = True
                    for v in roots:
                        xv = _to_plot_x(v)
                        if np.isfinite(xv):
                            ax.axvline(xv, linestyle='--',
                                       linewidth=overlay_roots_lw,
                                       alpha=overlay_roots_alpha,
                                       label=(label if first else None))
                            first = False

                for _pt, _roots, _lbl in _root_sets:
                    if _pt in _ax_for:
                        _draw_vlines(_ax_for[_pt], _roots, _lbl)

                if overlay_roots_also_on_function and 'function' in _ax_for:
                    for _pt, _roots, _lbl in _root_sets:
                        _draw_vlines(_ax_for['function'], _roots, _lbl)
        # ------------------------------------------------------------------

    if not successful_features:
        raise ValueError(f"Could not plot any of the requested features: {features}")

    # Set labels and formatting
    if use_log2fc or use_delta_p:
        xlabel = "log₂FC (x)"
    else:
        xlabel = "log₂(x)" if use_log2_x else "x"

    for plot_i, (plot_type, ylabel) in enumerate(plot_types):
        ax_curr = axes[plot_i]
        ax_curr.set_xlabel(xlabel)
        ax_curr.set_ylabel(ylabel)
        ax_curr.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add vertical line at u=0 (x = x_ntc) in log2FC or delta_p mode
        if use_log2fc or use_delta_p:
            ax_curr.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        if legend and len(successful_features) <= 20:
            ax_curr.legend(frameon=False, fontsize=8)

    # Title
    if title is None:
        if use_log2fc:
            suffix = " (log₂FC)"
        elif use_delta_p:
            suffix = " (Δp)"
        else:
            suffix = ""
        if len(successful_features) == 1:
            title = f"Trans function: {successful_features[0]}{suffix}"
        else:
            title = f"Trans functions ({len(successful_features)} features){suffix}"
    fig.suptitle(title)

    plt.tight_layout()
    return fig


def predict_trans_function(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Predict trans effect function for a feature given x_true values.

    Automatically detects function type (additive_hill, single_hill, polynomial)
    from posterior_samples_trans and computes predictions.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name to predict
    x_range : np.ndarray
        X values to predict at (cis expression levels)
    modality_name : str, optional
        Modality name (default: primary modality)

    Returns
    -------
    np.ndarray or None
        Predicted y values at x_range, or None if:
        - Trans model not fitted
        - Feature not in trans_genes
        - Cannot determine function type

    Notes
    -----
    Works with all function types:
    - additive_hill: A + alpha*Hill_a + beta*Hill_b
    - single_hill: A + Vmax*Hill(x, K, n)
    - polynomial: sum(coeffs[i] * x^i)
    """
    # Determine which modality to use
    if modality_name is None:
        modality_name = model.primary_modality

    # Check if trans model fitted for this modality
    if modality_name == model.primary_modality:
        # Primary modality: check model-level posterior
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            if debug: print(f"[predict_trans_function] returning None: no posterior_samples_trans on model")
            return None
        posterior = model.posterior_samples_trans
    else:
        # Non-primary modality: check modality-level posterior
        modality = model.get_modality(modality_name)
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            if debug: print(f"[predict_trans_function] returning None: no posterior_samples_trans on modality '{modality_name}'")
            return None
        posterior = modality.posterior_samples_trans

    if debug:
        print(f"[predict_trans_function] posterior keys: {list(posterior.keys())}")

    # Get baseline A (present in all function types)
    if 'A' not in posterior:
        if debug: print(f"[predict_trans_function] returning None: 'A' not in posterior keys {list(posterior.keys())}")
        return None

    A_samples = posterior['A']

    # Get posterior dimensions
    # For primary modality: A_samples is (S, T) where S=samples, T=trans_genes
    #   After mean(dim=0): (T,)
    # For non-primary modality: A_samples is (S, C, T) where C=cis_genes, T=trans_features
    #   After mean(dim=0): (C, T)
    # We want the LAST dimension (T) in both cases
    if hasattr(A_samples, 'mean'):
        A_mean = A_samples.mean(dim=0)
        # Squeeze out cis gene dimension if present (should be size 1 for non-primary modalities)
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]
    else:
        A_mean = A_samples.mean(axis=0)
        # Squeeze out cis gene dimension if present
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]

    # Get feature list from modality — always the ground truth
    if modality_name == model.primary_modality:
        # Use modality.feature_names — the modality is always the ground truth,
        # since it matches the fitted posterior dimensions exactly.
        _mod_obj = model.get_modality(modality_name)
        feature_list = _mod_obj.feature_names if _mod_obj.feature_names is not None else []
    else:
        # Non-primary modality: get feature names from modality
        modality = model.get_modality(modality_name)

        # First priority: use modality.feature_names (this is what users see and should use)
        if modality.feature_names is not None:
            feature_list = modality.feature_names
        elif modality.feature_meta is not None:
            # Fallback: Try common identifier columns in order of preference
            # For splicing: prioritize coordinate-based identifiers (coord.intron, junction_id)
            # For others: prioritize feature_id, feature, then fall back to gene names
            feature_list = None
            for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
                if col in modality.feature_meta.columns:
                    feature_list = modality.feature_meta[col].tolist()
                    break

            # If no column worked, try the index
            if feature_list is None:
                feature_list = modality.feature_meta.index.tolist()
        else:
            # No feature metadata - try to use feature count from posterior
            # Assume features are indexed 0, 1, 2, ... and cannot be matched by name
            if debug: print(f"[predict_trans_function] returning None: non-primary modality has no feature_names or feature_meta")
            return None

    n_features_list = len(feature_list)

    if debug:
        print(f"[predict_trans_function] n_genes_posterior={n_genes_posterior}, n_features_list={n_features_list}")
        print(f"[predict_trans_function] feature_list[:5]={feature_list[:5]}, ..., feature_list[-5:]={feature_list[-5:]}")

    # Check dimension consistency BEFORE using feature_list for indexing
    if n_genes_posterior != n_features_list:
        # Mismatch - cannot use feature_list for indexing
        # This happens when posterior was fitted on a subset of features
        if debug: print(f"[predict_trans_function] returning None: dimension mismatch n_genes_posterior={n_genes_posterior} != n_features_list={n_features_list}")
        return None

    # Now safe to check if feature is in feature_list and get its index
    if feature not in feature_list:
        if debug: print(f"[predict_trans_function] returning None: '{feature}' not in feature_list (len={n_features_list})")
        return None

    feature_idx = feature_list.index(feature)
    A = A_mean[feature_idx].item() if hasattr(A_mean, 'item') else A_mean[feature_idx]

    # Helper function to extract parameter value for a specific feature
    # Handles both primary modality (S, T) and non-primary modality (S, C, T) shapes
    def _extract_param(param_samples, feature_idx):
        """Extract mean parameter value for a specific feature, handling dimension squeezing."""
        if hasattr(param_samples, 'mean'):
            param_mean = param_samples.mean(dim=0)
        else:
            param_mean = param_samples.mean(axis=0)

        # Squeeze out cis gene dimension if present
        if param_mean.ndim > 1:
            param_mean = param_mean.squeeze(0)

        # Extract value for this feature
        val = param_mean[feature_idx]
        return val.item() if hasattr(val, 'item') else val

    # Determine function type from available parameters
    if debug:
        print(f"[predict_trans_function] A={A_mean[feature_idx]:.4g}, feature_idx={feature_idx}")
        print(f"[predict_trans_function] branch check: Vmax_a={'Vmax_a' in posterior}, Vmax_b={'Vmax_b' in posterior}, upper_limit={'upper_limit' in posterior}, theta={'theta' in posterior}")

    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL (negbinom/normal/studentt) =====
        try:
            # Extract parameters using helper function
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            Vmax_b = _extract_param(posterior['Vmax_b'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            if debug:
                print(f"[predict_trans_function] additive_hill: A={A:.4g}, alpha={alpha:.4g}, beta={beta:.4g}, Vmax_a={Vmax_a:.4g}, Vmax_b={Vmax_b:.4g}, K_a={K_a:.4g}, K_b={K_b:.4g}, n_a={n_a:.4g}, n_b={n_b:.4g}")

            # Compute Hill functions
            Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
            Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b, A=0, K=K_b, n=n_b)

            # Combined prediction
            y_pred = A + alpha * Hill_a + beta * Hill_b
            if debug:
                print(f"[predict_trans_function] y_pred range: [{y_pred.min():.4g}, {y_pred.max():.4g}], any>0: {(y_pred > 0).any()}")
            return y_pred

        except (KeyError, IndexError, AttributeError) as e:
            if debug: print(f"[predict_trans_function] returning None: additive_hill extraction failed: {e}")
            return None

    elif 'upper_limit' in posterior and 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL (binomial/multinomial with upper_limit and Vmax_a/b) =====
        try:
            # Extract parameters using helper function
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            Vmax_b = _extract_param(posterior['Vmax_b'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            # IMPORTANT: Match the actual model computation in fit_trans!
            # Model uses: y = A + (alpha * Hill_a(Vmax=Vmax_a)) + (beta * Hill_b(Vmax=Vmax_b))
            # where Hill functions return values in [0, Vmax]
            Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
            Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b, A=0, K=K_b, n=n_b)

            # Combined prediction
            y_pred = A + alpha * Hill_a + beta * Hill_b
            return y_pred

        except (KeyError, IndexError, AttributeError) as e:
            if debug: print(f"[predict_trans_function] returning None: upper_limit additive_hill extraction failed: {e}")
            return None

    elif 'Vmax_a' in posterior:
        # ===== SINGLE HILL =====
        try:
            alpha = _extract_param(posterior['alpha'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)

            # Compute Hill function: y = A + alpha * Hill(x, Vmax=Vmax_a, K=K_a, n=n_a)
            Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
            y_pred = A + alpha * Hill_a
            return y_pred

        except (KeyError, IndexError, AttributeError) as e:
            if debug: print(f"[predict_trans_function] returning None: single_hill extraction failed: {e}")
            return None

    elif 'theta' in posterior:
        # ===== POLYNOMIAL OR THETA-BASED =====
        try:
            theta_samples = posterior['theta']
            # theta shape for primary: (samples, features, n_params)
            # theta shape for non-primary: (samples, cis_genes, features, n_params)

            # Average over samples and squeeze cis gene dimension if present
            if hasattr(theta_samples, 'mean'):
                theta_mean = theta_samples.mean(dim=0)
                # Squeeze out cis gene dimension if present
                if theta_mean.ndim > 2:  # (cis_genes, features, n_params)
                    theta_mean = theta_mean.squeeze(0)  # (features, n_params)
                theta_mean = theta_mean[feature_idx, :]  # (n_params,)
                theta_np = theta_mean.cpu().numpy() if hasattr(theta_mean, 'cpu') else np.array(theta_mean)
            else:
                theta_mean = theta_samples.mean(axis=0)
                if theta_mean.ndim > 2:
                    theta_mean = theta_mean.squeeze(0)
                theta_mean = theta_mean[feature_idx, :]
                theta_np = np.array(theta_mean)

            # Polynomial: y = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
            # First coefficient is baseline (like A)
            baseline = theta_np[0]
            poly_coeffs = theta_np[1:]  # Remaining coefficients

            # Compute polynomial
            y_pred = np.full_like(x_range, baseline)
            for i, coeff in enumerate(poly_coeffs, start=1):
                y_pred += coeff * (x_range ** i)

            return y_pred

        except (KeyError, IndexError, AttributeError) as e:
            if debug: print(f"[predict_trans_function] returning None: polynomial extraction failed: {e}")
            return None

    else:
        # Unknown function type
        if debug: print(f"[predict_trans_function] returning None: unrecognized function type (posterior keys: {list(posterior.keys())})")
        return None


def predict_hill_from_summary_row(row, x_range: np.ndarray, fdr_threshold: float = 0.05) -> Optional[np.ndarray]:
    """
    Compute a Hill curve from a trans_summary DataFrame row.

    Supports both additive_hill (has Vmax_b_mean) and single_hill rows.
    Parameters are read from columns named A_mean, alpha_mean, Vmax_a_mean,
    K_a_mean, n_a_mean, beta_mean, Vmax_b_mean, K_b_mean, n_b_mean.

    FDR gating: components with fdr_alpha > fdr_threshold (or fdr_beta) are zeroed
    out so the curve reflects only statistically significant effects. Without gating,
    alpha/beta for null components are ~0.4-0.5 (RelaxedBernoulli prior) rather than 0,
    producing a spurious partial Hill contribution.

    Returns y values at x_range, or None if required columns are missing.
    """
    def _g(col, default=None):
        v = row.get(col, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    A       = _g('A_mean', 0.0)
    alpha   = _g('alpha_mean', 1.0)
    Vmax_a  = _g('Vmax_a_mean')
    K_a     = _g('K_a_mean')
    n_a     = _g('n_a_mean')
    if Vmax_a is None or K_a is None or n_a is None:
        return None

    # Apply FDR gating to alpha component
    fdr_alpha = _g('fdr_alpha')
    if fdr_alpha is not None and fdr_alpha > fdr_threshold:
        alpha = 0.0

    Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
    y_pred = A + alpha * Hill_a

    beta   = _g('beta_mean', 0.0)
    Vmax_b = _g('Vmax_b_mean')
    K_b    = _g('K_b_mean')
    n_b    = _g('n_b_mean')

    # Apply FDR gating to beta component
    fdr_beta = _g('fdr_beta')
    if fdr_beta is not None and fdr_beta > fdr_threshold:
        beta = 0.0

    if Vmax_b is not None and K_b is not None and n_b is not None and beta != 0.0:
        Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b, A=0, K=K_b, n=n_b)
        y_pred = y_pred + beta * Hill_b

    return y_pred


def predict_trans_function_samples(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Predict trans effect function for all posterior samples.

    Like predict_trans_function but returns predictions for each posterior sample
    instead of just the mean.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name to predict
    x_range : np.ndarray
        X values to predict at (cis expression levels)
    modality_name : str, optional
        Modality name (default: primary modality)
    max_samples : int, optional
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    np.ndarray or None
        Predicted y values with shape [n_samples, n_points], or None if prediction fails.
    """
    # Determine which modality to use
    if modality_name is None:
        modality_name = model.primary_modality

    # Check if trans model fitted for this modality
    if modality_name == model.primary_modality:
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            return None
        posterior = model.posterior_samples_trans
    else:
        modality = model.get_modality(modality_name)
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            return None
        posterior = modality.posterior_samples_trans

    # Get baseline A
    if 'A' not in posterior:
        return None

    A_samples = posterior['A']
    if hasattr(A_samples, 'cpu'):
        A_samples = A_samples.cpu().numpy()
    else:
        A_samples = np.array(A_samples)

    # Get feature list — always from the modality, never from model.trans_genes
    modality = model.get_modality(modality_name)
    if modality.feature_names is not None:
        feature_list = modality.feature_names
    elif modality.feature_meta is not None:
        feature_list = None
        for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
            if col in modality.feature_meta.columns:
                feature_list = modality.feature_meta[col].tolist()
                break
        if feature_list is None:
            feature_list = modality.feature_meta.index.tolist()
    else:
        return None

    # Handle dimension squeezing for non-primary modalities
    # A_samples shape: (S, T) for primary, (S, C, T) for non-primary
    if A_samples.ndim > 2:
        A_samples = A_samples.squeeze(1)  # Remove cis gene dimension

    # Get number of features in posterior
    n_features_posterior = A_samples.shape[1]
    n_features_list = len(feature_list)

    # Check dimension consistency
    if n_features_posterior != n_features_list:
        # Mismatch - likely cis gene excluded from posterior
        # Try to find correct index by checking if cis gene is in list
        if hasattr(model, 'cis_gene') and model.cis_gene in feature_list:
            # Remove cis gene from feature list for indexing
            feature_list_adjusted = [f for f in feature_list if f != model.cis_gene]
            if feature not in feature_list_adjusted:
                return None
            if len(feature_list_adjusted) != n_features_posterior:
                return None  # Still mismatched
            feature_idx = feature_list_adjusted.index(feature)
        else:
            return None
    else:
        if feature not in feature_list:
            return None
        feature_idx = feature_list.index(feature)

    # Helper to extract all samples for a parameter
    def _extract_samples(param_name, feature_idx):
        """Extract all posterior samples for a specific feature."""
        samples = posterior[param_name]
        if hasattr(samples, 'cpu'):
            samples = samples.cpu().numpy()
        else:
            samples = np.array(samples)
        # Squeeze cis gene dimension if present
        if samples.ndim > 2:
            samples = samples.squeeze(1)
        return samples[:, feature_idx]  # [n_samples]

    # Determine function type and compute predictions
    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL =====
        try:
            A = A_samples[:, feature_idx]  # [n_samples]
            alpha = _extract_samples('alpha', feature_idx)
            beta = _extract_samples('beta', feature_idx)
            Vmax_a = _extract_samples('Vmax_a', feature_idx)
            Vmax_b = _extract_samples('Vmax_b', feature_idx)
            K_a = _extract_samples('K_a', feature_idx)
            K_b = _extract_samples('K_b', feature_idx)
            n_a = _extract_samples('n_a', feature_idx)
            n_b = _extract_samples('n_b', feature_idx)

            n_samples = A.shape[0]
            if max_samples is not None and max_samples < n_samples:
                # Subsample
                indices = np.random.choice(n_samples, max_samples, replace=False)
                A = A[indices]
                alpha = alpha[indices]
                beta = beta[indices]
                Vmax_a = Vmax_a[indices]
                Vmax_b = Vmax_b[indices]
                K_a = K_a[indices]
                K_b = K_b[indices]
                n_a = n_a[indices]
                n_b = n_b[indices]
                n_samples = max_samples

            # Compute predictions for each sample
            # x_range is [n_points], we need to broadcast to [n_samples, n_points]
            n_points = len(x_range)
            y_pred = np.zeros((n_samples, n_points))

            for s in range(n_samples):
                Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a[s], A=0, K=K_a[s], n=n_a[s])
                Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b[s], A=0, K=K_b[s], n=n_b[s])
                y_pred[s, :] = A[s] + alpha[s] * Hill_a + beta[s] * Hill_b

            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'Vmax_a' in posterior:
        # ===== SINGLE HILL =====
        try:
            A = A_samples[:, feature_idx]
            alpha = _extract_samples('alpha', feature_idx)
            Vmax_a = _extract_samples('Vmax_a', feature_idx)
            K_a = _extract_samples('K_a', feature_idx)
            n_a = _extract_samples('n_a', feature_idx)

            n_samples = A.shape[0]
            if max_samples is not None and max_samples < n_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                A = A[indices]
                alpha = alpha[indices]
                Vmax_a = Vmax_a[indices]
                K_a = K_a[indices]
                n_a = n_a[indices]
                n_samples = max_samples

            n_points = len(x_range)
            y_pred = np.zeros((n_samples, n_points))

            for s in range(n_samples):
                Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a[s], A=0, K=K_a[s], n=n_a[s])
                y_pred[s, :] = A[s] + alpha[s] * Hill_a

            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    else:
        # Polynomial or unknown - not supported for samples
        return None


def predict_trans_derivatives_samples(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute trans function and derivatives for all posterior samples.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name to predict
    x_range : np.ndarray
        X values to compute at
    modality_name : str, optional
        Modality name (default: primary modality)
    max_samples : int, optional
        Maximum number of samples to return

    Returns
    -------
    Tuple of (y_samples, first_deriv_samples, second_deriv_samples, third_deriv_samples)
        Each has shape [n_samples, n_points], or None if computation fails
    """
    # Determine which modality to use
    if modality_name is None:
        modality_name = model.primary_modality

    # Check if trans model fitted
    if modality_name == model.primary_modality:
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            return None, None, None, None
        posterior = model.posterior_samples_trans
    else:
        modality = model.get_modality(modality_name)
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            return None, None, None, None
        posterior = modality.posterior_samples_trans

    # Get baseline A
    if 'A' not in posterior:
        return None, None, None, None

    A_samples = posterior['A']
    if hasattr(A_samples, 'cpu'):
        A_samples = A_samples.cpu().numpy()
    else:
        A_samples = np.array(A_samples)

    # Get feature list — always from the modality, never from model.trans_genes
    modality = model.get_modality(modality_name)
    if modality.feature_names is not None:
        feature_list = modality.feature_names
    elif modality.feature_meta is not None:
        feature_list = None
        for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
            if col in modality.feature_meta.columns:
                feature_list = modality.feature_meta[col].tolist()
                break
        if feature_list is None:
            feature_list = modality.feature_meta.index.tolist()
    else:
        return None, None, None, None

    # Handle dimension squeezing
    if A_samples.ndim > 2:
        A_samples = A_samples.squeeze(1)

    # Check dimension consistency
    n_features_posterior = A_samples.shape[1]
    n_features_list = len(feature_list)

    if n_features_posterior != n_features_list:
        if hasattr(model, 'cis_gene') and model.cis_gene in feature_list:
            feature_list_adjusted = [f for f in feature_list if f != model.cis_gene]
            if feature not in feature_list_adjusted:
                return None, None, None, None
            if len(feature_list_adjusted) != n_features_posterior:
                return None, None, None, None
            feature_idx = feature_list_adjusted.index(feature)
        else:
            return None, None, None, None
    else:
        if feature not in feature_list:
            return None, None, None, None
        feature_idx = feature_list.index(feature)

    # Helper to extract all samples for a parameter
    def _extract_samples(param_name, feature_idx):
        samples = posterior[param_name]
        if hasattr(samples, 'cpu'):
            samples = samples.cpu().numpy()
        else:
            samples = np.array(samples)
        if samples.ndim > 2:
            samples = samples.squeeze(1)
        return samples[:, feature_idx]

    # Compute based on function type
    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL =====
        try:
            A = A_samples[:, feature_idx]
            alpha = _extract_samples('alpha', feature_idx)
            beta = _extract_samples('beta', feature_idx)
            Vmax_a = _extract_samples('Vmax_a', feature_idx)
            Vmax_b = _extract_samples('Vmax_b', feature_idx)
            K_a = _extract_samples('K_a', feature_idx)
            K_b = _extract_samples('K_b', feature_idx)
            n_a = _extract_samples('n_a', feature_idx)
            n_b = _extract_samples('n_b', feature_idx)

            n_samples = A.shape[0]
            if max_samples is not None and max_samples < n_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                A = A[indices]
                alpha = alpha[indices]
                beta = beta[indices]
                Vmax_a = Vmax_a[indices]
                Vmax_b = Vmax_b[indices]
                K_a = K_a[indices]
                K_b = K_b[indices]
                n_a = n_a[indices]
                n_b = n_b[indices]
                n_samples = max_samples

            n_points = len(x_range)
            y_samples = np.zeros((n_samples, n_points))
            first_deriv_samples = np.zeros((n_samples, n_points))
            second_deriv_samples = np.zeros((n_samples, n_points))
            third_deriv_samples = np.zeros((n_samples, n_points))

            for s in range(n_samples):
                # Function values
                Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a[s], A=0, K=K_a[s], n=n_a[s])
                Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b[s], A=0, K=K_b[s], n=n_b[s])
                y_samples[s, :] = A[s] + alpha[s] * Hill_a + beta[s] * Hill_b

                # First derivatives
                dHill_a = Hill_first_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])
                dHill_b = Hill_first_derivative(x_range, Vmax=Vmax_b[s], K=K_b[s], n=n_b[s])
                first_deriv_samples[s, :] = alpha[s] * dHill_a + beta[s] * dHill_b

                # Second derivatives
                d2Hill_a = Hill_second_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])
                d2Hill_b = Hill_second_derivative(x_range, Vmax=Vmax_b[s], K=K_b[s], n=n_b[s])
                second_deriv_samples[s, :] = alpha[s] * d2Hill_a + beta[s] * d2Hill_b

                # Third derivatives
                d3Hill_a = Hill_third_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])
                d3Hill_b = Hill_third_derivative(x_range, Vmax=Vmax_b[s], K=K_b[s], n=n_b[s])
                third_deriv_samples[s, :] = alpha[s] * d3Hill_a + beta[s] * d3Hill_b

            return y_samples, first_deriv_samples, second_deriv_samples, third_deriv_samples

        except (KeyError, IndexError, AttributeError):
            return None, None, None, None

    elif 'Vmax_a' in posterior:
        # ===== SINGLE HILL =====
        try:
            A = A_samples[:, feature_idx]
            alpha = _extract_samples('alpha', feature_idx)
            Vmax_a = _extract_samples('Vmax_a', feature_idx)
            K_a = _extract_samples('K_a', feature_idx)
            n_a = _extract_samples('n_a', feature_idx)

            n_samples = A.shape[0]
            if max_samples is not None and max_samples < n_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                A = A[indices]
                alpha = alpha[indices]
                Vmax_a = Vmax_a[indices]
                K_a = K_a[indices]
                n_a = n_a[indices]
                n_samples = max_samples

            n_points = len(x_range)
            y_samples = np.zeros((n_samples, n_points))
            first_deriv_samples = np.zeros((n_samples, n_points))
            second_deriv_samples = np.zeros((n_samples, n_points))
            third_deriv_samples = np.zeros((n_samples, n_points))

            for s in range(n_samples):
                Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a[s], A=0, K=K_a[s], n=n_a[s])
                y_samples[s, :] = A[s] + alpha[s] * Hill_a
                first_deriv_samples[s, :] = alpha[s] * Hill_first_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])
                second_deriv_samples[s, :] = alpha[s] * Hill_second_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])
                third_deriv_samples[s, :] = alpha[s] * Hill_third_derivative(x_range, Vmax=Vmax_a[s], K=K_a[s], n=n_a[s])

            return y_samples, first_deriv_samples, second_deriv_samples, third_deriv_samples

        except (KeyError, IndexError, AttributeError):
            return None, None, None, None

    else:
        # Polynomial or unknown - not supported
        return None, None, None, None


# ============================================================================
# Hill Parameter Marker Helpers
# ============================================================================

def _extract_hill_param_samples(posterior, feature_idx):
    """Extract per-feature posterior samples (shape S,) for all Hill parameters."""
    def _get(key):
        if key not in posterior:
            return None
        arr = posterior[key]
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        elif not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.ndim == 2:        # (S, T)
            return arr[:, feature_idx]
        elif arr.ndim == 3:      # (S, C, T) - squeeze cis dim
            return arr[:, 0, feature_idx]
        return None

    return {k: _get(k) for k in ['A', 'alpha', 'beta', 'Vmax_a', 'Vmax_b', 'K_a', 'K_b', 'n_a', 'n_b']
            if _get(k) is not None}


def _build_hill_markers_from_params(
    A, alpha, Vmax_a, K_a, n_a,
    beta=None, Vmax_b=None, K_b=None, n_b=None,
    a_null=False, b_null=False,
    is_additive=False,
    log2_space=True, y_scale=1.0,
) -> list:
    """
    Core Hill marker building logic shared by both posterior-based and summary-row-based callers.

    Parameters
    ----------
    A, alpha, Vmax_a, K_a, n_a : float
        Posterior/summary mean values for component A parameters.
    beta, Vmax_b, K_b, n_b : float or None
        Component B parameters (additive_hill only).
    a_null, b_null : bool
        Whether each component is FDR-inactive (True = skip that component).
    is_additive : bool
        Whether this is an additive_hill model with both components.
    log2_space : bool
        True for negbinom (y-axis in log2). False for linear distributions.
    y_scale : float
        Multiplicative scale applied to linear y-values before plotting.
    """
    def yv(v):
        if v is None:
            return None
        if log2_space:
            return float(np.log2(v)) if v > 0 else None
        return float(v * y_scale)

    def xv(k):
        if k is None or k <= 0:
            return None
        return float(np.log2(k))

    markers = []

    if not is_additive:
        # ── Single Hill ────────────────────────────────────────────────────
        lv = yv(A)
        hv = yv(A + alpha * Vmax_a)
        ev = xv(K_a)
        if lv is not None and np.isfinite(lv):
            markers.append({'axis': 'h', 'value': lv,
                            'label': 'log2(A)' if log2_space else 'A',
                            'linestyle': ':', 'color': 'darkred', 'alpha': 0.85})
        if hv is not None and np.isfinite(hv):
            markers.append({'axis': 'h', 'value': hv,
                            'label': 'log2(A+α·Vmax)' if log2_space else 'A+α·Vmax',
                            'linestyle': ':', 'color': 'darkblue', 'alpha': 0.85})
        if ev is not None and np.isfinite(ev):
            markers.append({'axis': 'v', 'value': ev, 'label': 'log2(EC50)',
                            'linestyle': '--', 'color': 'darkgreen', 'alpha': 0.85})
        return markers

    # ── Additive Hill ──────────────────────────────────────────────────────
    if a_null or b_null:
        # ── Effectively single Hill ────────────────────────────────────────
        if a_null and not b_null:
            act_alpha, act_Vmax, act_K = beta, Vmax_b, K_b
        elif b_null and not a_null:
            act_alpha, act_Vmax, act_K = alpha, Vmax_a, K_a
        else:
            # Both null — just mark A
            lv = yv(A)
            if lv is not None and np.isfinite(lv):
                markers.append({'axis': 'h', 'value': lv,
                                'label': 'log2(A)' if log2_space else 'A',
                                'linestyle': ':', 'color': 'darkred', 'alpha': 0.85})
            return markers

        lv = yv(A)
        hv = yv(A + act_alpha * act_Vmax)
        ev = xv(act_K)
        if lv is not None and np.isfinite(lv):
            markers.append({'axis': 'h', 'value': lv,
                            'label': 'log2(A)' if log2_space else 'A',
                            'linestyle': ':', 'color': 'darkred', 'alpha': 0.85})
        if hv is not None and np.isfinite(hv):
            markers.append({'axis': 'h', 'value': hv,
                            'label': 'log2(A+α·Vmax)' if log2_space else 'A+α·Vmax',
                            'linestyle': ':', 'color': 'darkblue', 'alpha': 0.85})
        if ev is not None and np.isfinite(ev):
            markers.append({'axis': 'v', 'value': ev, 'label': 'log2(EC50)',
                            'linestyle': '--', 'color': 'darkgreen', 'alpha': 0.85})

    elif (n_a > 0) == (n_b > 0):
        # ── Same-sign monotone sum ─────────────────────────────────────────
        lv  = yv(A)
        hv  = yv(A + alpha * Vmax_a + beta * Vmax_b)
        e1  = xv(min(K_a, K_b))
        e2  = xv(max(K_a, K_b))
        if lv is not None and np.isfinite(lv):
            markers.append({'axis': 'h', 'value': lv,
                            'label': 'log2(A)' if log2_space else 'A',
                            'linestyle': ':', 'color': 'darkred', 'alpha': 0.85})
        if hv is not None and np.isfinite(hv):
            markers.append({'axis': 'h', 'value': hv,
                            'label': ('log2(A+α·Vmax_a+β·Vmax_b)' if log2_space
                                      else 'A+α·Vmax_a+β·Vmax_b'),
                            'linestyle': ':', 'color': 'darkblue', 'alpha': 0.85})
        if e1 is not None and np.isfinite(e1):
            markers.append({'axis': 'v', 'value': e1, 'label': 'log2(EC50_lower)',
                            'linestyle': '--', 'color': 'darkgreen', 'alpha': 0.85})
        if e2 is not None and np.isfinite(e2) and abs(e2 - e1) > 0.05:
            markers.append({'axis': 'v', 'value': e2, 'label': 'log2(EC50_upper)',
                            'linestyle': ':', 'color': 'darkgreen', 'alpha': 0.85})

    else:
        # ── Non-monotonic (opposite-sign Hills) ───────────────────────────
        if n_a > 0:   # a activates, b inhibits
            asym_low_x  = A + beta  * Vmax_b
            asym_high_x = A + alpha * Vmax_a
            K_act, K_inh = K_a, K_b
        else:         # b activates, a inhibits
            asym_low_x  = A + alpha * Vmax_a
            asym_high_x = A + beta  * Vmax_b
            K_act, K_inh = K_b, K_a

        has_peak  = (K_act < K_inh)
        peak_val  = A + alpha * Vmax_a + beta * Vmax_b
        a1 = yv(asym_low_x)
        a2 = yv(asym_high_x)
        e1 = xv(min(K_a, K_b))
        e2 = xv(max(K_a, K_b))

        if a1 is not None and np.isfinite(a1):
            markers.append({'axis': 'h', 'value': a1,
                            'label': 'log2(y: x→0)' if log2_space else 'y(x→0)',
                            'linestyle': ':', 'color': 'darkred', 'alpha': 0.85})
        if a2 is not None and np.isfinite(a2) and (a1 is None or abs(a2 - a1) > 0.05):
            markers.append({'axis': 'h', 'value': a2,
                            'label': 'log2(y: x→∞)' if log2_space else 'y(x→∞)',
                            'linestyle': ':', 'color': 'darkblue', 'alpha': 0.85})
        if has_peak:
            pv = yv(peak_val)
            if pv is not None and np.isfinite(pv):
                markers.append({'axis': 'h', 'value': pv,
                                'label': ('log2(A+α·Vmax_a+β·Vmax_b) [peak]' if log2_space
                                          else 'A+α·Vmax_a+β·Vmax_b [peak]'),
                                'linestyle': '--', 'color': 'darkorange', 'alpha': 0.85})
        if e1 is not None and np.isfinite(e1):
            markers.append({'axis': 'v', 'value': e1, 'label': 'log2(EC50_lower)',
                            'linestyle': '--', 'color': 'darkgreen', 'alpha': 0.85})
        if e2 is not None and np.isfinite(e2) and abs(e2 - e1) > 0.05:
            markers.append({'axis': 'v', 'value': e2, 'label': 'log2(EC50_upper)',
                            'linestyle': ':', 'color': 'darkgreen', 'alpha': 0.85})

    return markers


def _compute_hill_markers_from_summary_row(row, log2_space=True, y_scale=1.0,
                                           fdr_threshold=0.05) -> list:
    """
    Compute Hill parameter markers from a trans_summary DataFrame row.

    Reads *_mean columns for parameter values and fdr_alpha / fdr_beta for
    activity classification.  Returns the same list-of-dicts format as
    _compute_hill_markers / _build_hill_markers_from_params.
    """
    def _g(col, default=None):
        v = row.get(col, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    A      = _g('A_mean', 0.0);  alpha = _g('alpha_mean', 1.0)
    Vmax_a = _g('Vmax_a_mean');  K_a   = _g('K_a_mean');  n_a = _g('n_a_mean')
    if Vmax_a is None or K_a is None or n_a is None:
        return []

    beta   = _g('beta_mean', 0.0)
    Vmax_b = _g('Vmax_b_mean');  K_b   = _g('K_b_mean');  n_b = _g('n_b_mean')
    is_additive = (Vmax_b is not None and K_b is not None and n_b is not None)

    fdr_alpha_val = _g('fdr_alpha', 1.0)
    fdr_beta_val  = _g('fdr_beta',  1.0)
    a_null = fdr_alpha_val >= fdr_threshold
    b_null = fdr_beta_val  >= fdr_threshold

    return _build_hill_markers_from_params(
        A=A, alpha=alpha, Vmax_a=Vmax_a, K_a=K_a, n_a=n_a,
        beta=beta, Vmax_b=Vmax_b, K_b=K_b, n_b=n_b,
        a_null=a_null, b_null=b_null,
        is_additive=is_additive,
        log2_space=log2_space, y_scale=y_scale,
    )


def _compute_hill_markers(model, feature, modality, ci_level=95.0, log2_space=True, y_scale=1.0,
                          fdr_df=None, fdr_threshold=0.05):
    """
    Compute meaningful parameter markers for single_hill and additive_hill trans functions.

    Returns a list of marker dicts with keys:
        'axis' ('h' or 'v'), 'value', 'label', 'linestyle', 'color', 'alpha'

    Parameters
    ----------
    log2_space : bool
        True for negbinom (y-axis is log2). False for binomial/normal (linear y-axis).
    y_scale : float
        Multiplicative scale applied to linear y-values before plotting (e.g. 100 for PSI%).
    ci_level : float
        Credible interval level used to classify additive_hill regime (default 95.0).
    """
    # ── Resolve posterior and feature list ─────────────────────────────────
    if modality.name == model.primary_modality:
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            return []
        posterior = model.posterior_samples_trans
        # Use modality.feature_names — always matches the fitted posterior dimensions.
        feature_list = list(modality.feature_names) if modality.feature_names is not None else []
    else:
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            return []
        posterior = modality.posterior_samples_trans
        if modality.feature_names is not None:
            feature_list = list(modality.feature_names)
        elif modality.feature_meta is not None:
            feature_list = None
            for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
                if col in modality.feature_meta.columns:
                    feature_list = modality.feature_meta[col].tolist()
                    break
            if feature_list is None:
                return []
        else:
            return []

    if feature not in feature_list:
        return []
    if 'Vmax_a' not in posterior:   # polynomial or unknown
        return []

    feature_idx = feature_list.index(feature)
    params = _extract_hill_param_samples(posterior, feature_idx)

    required = ['A', 'alpha', 'Vmax_a', 'K_a', 'n_a']
    if not all(k in params for k in required):
        return []

    def pmean(key):
        return float(params[key].mean()) if key in params else None

    def pci(key):
        if key not in params:
            return None, None
        lo = float(np.percentile(params[key], (100 - ci_level) / 2))
        hi = float(np.percentile(params[key], 100 - (100 - ci_level) / 2))
        return lo, hi

    A      = pmean('A');     alpha = pmean('alpha');  Vmax_a = pmean('Vmax_a')
    K_a    = pmean('K_a');   n_a   = pmean('n_a')

    is_additive = ('Vmax_b' in posterior and 'Vmax_b' in params and
                   'n_b' in params and 'beta' in params and 'K_b' in params)

    beta = Vmax_b = K_b = n_b = None
    if is_additive:
        beta   = pmean('beta');  Vmax_b = pmean('Vmax_b')
        K_b    = pmean('K_b');   n_b    = pmean('n_b')

        n_a_lo, n_a_hi = pci('n_a')
        n_b_lo, n_b_hi = pci('n_b')

        # Prefer FDR-based null classification; fall back to CI criterion
        _fdr_row = None
        if fdr_df is not None:
            _name_col = next((c for c in ['gene_name', 'gene'] if c in fdr_df.columns), None)
            if _name_col:
                _match = fdr_df[fdr_df[_name_col] == feature]
                if not _match.empty:
                    _fdr_row = _match.iloc[0]
        if _fdr_row is not None:
            a_null = float(_fdr_row.get('fdr_alpha', 1.0)) >= fdr_threshold
            b_null = float(_fdr_row.get('fdr_beta',  1.0)) >= fdr_threshold
        else:
            a_null = (n_a_lo is not None) and (n_a_lo <= 0 <= n_a_hi)
            b_null = (n_b_lo is not None) and (n_b_lo <= 0 <= n_b_hi)
    else:
        a_null = b_null = False

    return _build_hill_markers_from_params(
        A=A, alpha=alpha, Vmax_a=Vmax_a, K_a=K_a, n_a=n_a,
        beta=beta, Vmax_b=Vmax_b, K_b=K_b, n_b=n_b,
        a_null=a_null, b_null=b_null,
        is_additive=is_additive,
        log2_space=log2_space, y_scale=y_scale,
    )


def _draw_hill_markers(ax, markers, x_offset=0.0, y_offset=0.0):
    """Draw Hill parameter marker lines (axhline / axvline) on ax.

    Parameters
    ----------
    x_offset, y_offset : float
        Subtract these from x/y marker values before drawing.  Pass the
        log2FC offsets (log2(x_ntc), log2(y_ntc)) when the plot axes are in
        log2FC space so that markers align with the shifted data.  Labels are
        automatically updated to say ``log2FC(...)`` when offsets are non-zero.
    """
    in_log2fc = (x_offset != 0.0 or y_offset != 0.0)
    for m in markers:
        label = m['label']
        if in_log2fc:
            label = label.replace('log2(', 'log2FC(')
        kw = dict(linestyle=m['linestyle'], color=m['color'],
                  alpha=m['alpha'], linewidth=1, label=label)
        if m['axis'] == 'h':
            ax.axhline(m['value'] - y_offset, **kw)
        elif m['axis'] == 'v':
            ax.axvline(m['value'] - x_offset, **kw)


# ============================================================================
# Distribution-Specific Plotting Functions
# ============================================================================

def _save_figure(fig, model, filename):
    """Save fig to model.output_dir/filename (adds .png if no extension)."""
    if filename is None:
        return
    import os
    if not os.path.splitext(filename)[1]:
        filename = filename + '.png'
    if os.path.isabs(filename):
        save_path = filename
    else:
        save_path = os.path.join(model.output_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"[PLOT] Saved to {save_path}")


def _compute_global_log2fc_offsets(
    model, modality, feature_name: str, x_true: np.ndarray,
    subset_mask: Optional[np.ndarray], sum_factor_col: str
) -> Tuple[float, float]:
    """
    Compute the NTC log2 x_offset and y_offset from the GLOBAL (non-faceted) data.

    This must be called with only the top-level ``subset_mask`` (no facet mask),
    so that all panels for the same feature share an identical NTC reference —
    even panels that contain no NTC cells (e.g. a "Targeting" facet column).

    y_offset priority (consistent with ``save_trans_summary``):
    1. ``mu_ntc`` from ``modality.posterior_samples_technical`` (model-smoothed reference-
       group NTC expression rate from ``fit_technical``).  Shape [n_samples, T]; we take
       the posterior mean for the requested feature.
    2. Fallback (``fit_technical`` not run): empirical mean of ``y_obs / sum_factor`` for
       NTC reference-group cells (group_code == 0), or all NTC cells if no reference group
       is tagged.

    Returns
    -------
    (x_offset, y_offset) : (float, float)
        Both in log2 space; 0.0 if no valid NTC cells found.
    """
    feature_idx = _get_feature_index(feature_name, modality)
    if feature_idx is None:
        return 0.0, 0.0

    # Get counts for this feature
    if modality.cells_axis == 1:
        y_obs = modality.counts[feature_idx, :]
    else:
        y_obs = modality.counts[:, feature_idx]

    # Align cells using the global subset_mask (no facet restriction)
    x_true_aligned, y_obs_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, y_obs, subset_mask
    )
    sf_aligned = modality.sum_factors.loc[
        meta_aligned['cell'].values, sum_factor_col
    ].values

    is_ntc = (meta_aligned['target'].str.lower() == 'ntc').values

    # x_offset: mean log2 NTC x_true (pooled, group-independent)
    x_ntc = x_true_aligned[is_ntc & (x_true_aligned > 0)]
    x_offset = np.log2(float(x_ntc.mean())) if len(x_ntc) > 0 else 0.0

    # y_offset: use mu_ntc from technical fit when available (matches save_trans_summary).
    # mu_ntc shape in posterior_samples_technical: [n_samples, T].
    # It represents the reference-group NTC expression rate — the same quantity that
    # save_trans_summary uses as y_ntc for log2FC parameter computation.
    y_offset = None
    if (hasattr(modality, 'posterior_samples_technical')
            and modality.posterior_samples_technical is not None
            and 'mu_ntc' in modality.posterior_samples_technical):
        mu_ntc = modality.posterior_samples_technical['mu_ntc']
        if hasattr(mu_ntc, 'cpu'):          # torch.Tensor
            mu_ntc = mu_ntc.cpu().numpy()
        mu_ntc = np.asarray(mu_ntc, dtype=float)
        # Collapse any unexpected leading dims beyond (samples, features)
        while mu_ntc.ndim > 2:
            mu_ntc = mu_ntc.mean(axis=0)
        # Average over posterior samples → [T], then pick this feature
        mu_ntc_mean = mu_ntc.mean(axis=0)   # [T]
        if feature_idx < len(mu_ntc_mean):
            y_ntc_val = float(mu_ntc_mean[feature_idx])
            if np.isfinite(y_ntc_val) and y_ntc_val > 0:
                y_offset = np.log2(y_ntc_val)

    if y_offset is None:
        # Fallback: empirical NTC reference-group mean (when fit_technical not run)
        if 'technical_group_code' in meta_aligned.columns:
            ntc_ref_mask = is_ntc & (meta_aligned['technical_group_code'].values == 0)
        else:
            ntc_ref_mask = is_ntc
        y_ntc_expr = y_obs_aligned[ntc_ref_mask] / sf_aligned[ntc_ref_mask]
        y_ntc_valid = y_ntc_expr[(y_ntc_expr > 0) & np.isfinite(y_ntc_expr)]
        if len(y_ntc_valid) == 0:           # fallback: pooled NTC (all groups)
            y_ntc_expr = y_obs_aligned[is_ntc] / sf_aligned[is_ntc]
            y_ntc_valid = y_ntc_expr[(y_ntc_expr > 0) & np.isfinite(y_ntc_expr)]
        y_offset = np.log2(float(y_ntc_valid.mean())) if len(y_ntc_valid) > 0 else 0.0

    return x_offset, y_offset


def _collect_legend_handles(axes):
    """Collect deduplicated (handle, label) pairs from a list of axes."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    return handles, labels


def plot_negbinom_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_hill_function: bool,
    show_ntc_gradient: bool = False,
    sum_factor_col: str = 'sum_factor',
    min_counts: int = 0,
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    mark_params: bool = False,
    ci_level: float = 95.0,
    legend_outside: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    log2fc: bool = False,
    reference_df=None,
    fdr_df=None,
    fdr_threshold: float = 0.05,
    color_by: Union[str, List[str]] = 'technical_group',
    ntc_x_offset: Optional[float] = None,
    ntc_y_offset: Optional[float] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot negbinom (gene counts) with optional Hill function overlay.

    Y-axis: log2(counts / sum_factor), optionally alpha_y-corrected.

    Parameters (selected)
    ---------------------
    color_by : str or list of two str
        What to color smoothed lines by. Options:
        - ``'technical_group'`` (default): one line per cell-line / technical group
        - ``'targeting'``: two lines — NTC vs all targeting cells
        - Any column name in ``model.meta``: one line per unique value
        - A **list of two** values (e.g. ``['technical_group', 'targeting']``): cross-product
          of two groupings. The first determines hue; the second determines shade — NTC gets
          a lighter tint of the group colour, Targeting gets the full colour.

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
    sum_factor_col : str
        Column name in model.meta for sum factors (default: 'sum_factor')
        Can be 'sum_factor', 'sum_factor_adj', or any other sum factor column
    min_counts : int
        Minimum raw counts to include a cell (default: 0, i.e., include all cells)
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts for this feature
    if modality.cells_axis == 1:
        y_obs = modality.counts[feature_idx, :]
    else:
        y_obs = modality.counts[:, feature_idx]

    # Check if sum_factor_col exists on modality sum_factors
    if modality.sum_factors is None or sum_factor_col not in modality.sum_factors.columns:
        raise ValueError(
            f"Sum factor column '{sum_factor_col}' not found in modality sum_factors. "
            f"Available columns: {list(modality.sum_factors.columns) if modality.sum_factors is not None else '(none)'}"
        )

    # Align cells between model.meta and modality
    x_true_aligned, y_obs_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, y_obs, subset_mask
    )

    # Build dataframe — read sum factors from modality, not from meta_aligned
    df_data = {
        'x_true': x_true_aligned,
        'y_obs': y_obs_aligned,
        'target': meta_aligned['target'].values,
        'sum_factor': modality.sum_factors.loc[meta_aligned['cell'].values, sum_factor_col].values
    }

    # Conditionally add technical_group_code if it exists
    if 'technical_group_code' in meta_aligned.columns:
        df_data['technical_group_code'] = meta_aligned['technical_group_code'].values

    # Forward any extra meta columns referenced in color_by into df_data now,
    # before the DataFrame is built, so that the min_counts filter applies to them too.
    _cb_items = [color_by] if isinstance(color_by, str) else list(color_by)
    _SPECIAL_CB = {'technical_group', 'targeting'}
    for _cb in _cb_items:
        if _cb in _SPECIAL_CB or _cb in df_data:
            continue
        if _cb in meta_aligned.columns:
            df_data[_cb] = meta_aligned[_cb].values
        # If not found anywhere, _resolve_grouping will warn later.

    df = pd.DataFrame(df_data)

    # Filter by min_counts (exclude cells with raw counts below threshold)
    if min_counts > 0:
        df = df[df['y_obs'] >= min_counts].copy()

    # Technical correction
    has_technical_fit = modality.alpha_y_prefit is not None

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))
            axes = [ax]
    else:
        # Single axis provided - in multifeature mode, function is called twice
        # (once with 'uncorrected' and once with 'corrected'), so don't override show_correction
        axes = [ax]

    # Map technical_group_code → label using only codes present in this df
    # If no technical groups, treat as single group
    code_to_label = _labels_by_code_for_df(model, df)

    if 'technical_group_code' in df.columns:
        group_codes = np.sort(df['technical_group_code'].unique())
    else:
        group_codes = np.array([0])  # Single group

    group_labels = [code_to_label[int(c)] for c in group_codes]

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Compute NTC reference offsets for log2FC mode.
    # x_offset: pooled NTC x_true (x_true is group-independent).
    # y_offset: reference group (group_code=0, alpha_y=1.0) NTC uncorrected expression.
    #   Because alpha_y=1.0 for the reference group, its uncorrected NTC equals the corrected
    #   NTC for any group — so this is the natural "true NTC" baseline.
    #
    # ntc_x_offset / ntc_y_offset: pre-computed overrides supplied by the caller
    # (plot_xy_data passes these when faceting, so that panels without NTC cells —
    # e.g. the "Targeting" facet — still use the correct global NTC baseline).
    if log2fc:
        if ntc_x_offset is not None and ntc_y_offset is not None:
            # Use caller-supplied global offsets (correct even when df has no NTC cells)
            x_offset = ntc_x_offset
            y_offset = ntc_y_offset
        else:
            ntc_df = df[is_ntc]
            x_ntc_valid = ntc_df['x_true'][ntc_df['x_true'] > 0]
            if len(x_ntc_valid) > 0:
                x_offset = np.log2(float(x_ntc_valid.mean()))
            else:
                warnings.warn("log2fc=True: no valid NTC cells found for x; plotting raw log2 instead.")
                x_offset = 0.0
            # y_offset: prefer mu_ntc from technical fit (same source as save_trans_summary).
            # mu_ntc is the model-inferred reference-group NTC expression rate [n_samples, T].
            # Fall back to empirical NTC reference-group mean when fit_technical not run.
            _feature_idx = _get_feature_index(feature, modality)
            _y_offset_from_technical = None
            if (_feature_idx is not None
                    and hasattr(modality, 'posterior_samples_technical')
                    and modality.posterior_samples_technical is not None
                    and 'mu_ntc' in modality.posterior_samples_technical):
                _mu_ntc = modality.posterior_samples_technical['mu_ntc']
                if hasattr(_mu_ntc, 'cpu'):
                    _mu_ntc = _mu_ntc.cpu().numpy()
                _mu_ntc = np.asarray(_mu_ntc, dtype=float)
                while _mu_ntc.ndim > 2:
                    _mu_ntc = _mu_ntc.mean(axis=0)
                _mu_ntc_mean = _mu_ntc.mean(axis=0)   # [T]
                if _feature_idx < len(_mu_ntc_mean):
                    _val = float(_mu_ntc_mean[_feature_idx])
                    if np.isfinite(_val) and _val > 0:
                        _y_offset_from_technical = np.log2(_val)
            if _y_offset_from_technical is not None:
                y_offset = _y_offset_from_technical
            else:
                # Empirical fallback: reference-group NTC mean
                if 'technical_group_code' in ntc_df.columns:
                    ntc_ref_df = ntc_df[ntc_df['technical_group_code'] == 0]
                else:
                    ntc_ref_df = ntc_df
                y_ntc_expr = ntc_ref_df['y_obs'] / ntc_ref_df['sum_factor']
                y_ntc_valid = y_ntc_expr[(y_ntc_expr > 0) & np.isfinite(y_ntc_expr)]
                if len(y_ntc_valid) == 0:  # fallback to pooled if reference group has no NTC
                    y_ntc_expr = ntc_df['y_obs'] / ntc_df['sum_factor']
                    y_ntc_valid = y_ntc_expr[(y_ntc_expr > 0) & np.isfinite(y_ntc_expr)]
                y_offset = np.log2(float(y_ntc_valid.mean())) if len(y_ntc_valid) > 0 else 0.0
    else:
        x_offset = 0.0
        y_offset = 0.0

    # Build color groups based on color_by.
    # color_by can be a str (single grouping) or a list of two str (cross-product).
    # Alpha_y correction is always per technical_group_code regardless of color_by.

    def _resolve_grouping(cb):
        """Return list of (label, boolean_mask) for one color_by value, or None if not found."""
        if cb == 'technical_group':
            if 'technical_group_code' in df.columns:
                return [(code_to_label[int(gc)], df['technical_group_code'] == gc)
                        for gc in group_codes]
            return [('All', pd.Series(True, index=df.index))]
        if cb == 'targeting':
            ntc_m = df['target'].str.lower() == 'ntc'
            return [('NTC', ntc_m), ('Targeting', ~ntc_m)]
        if cb in df.columns:
            n_unique = df[cb].nunique()
            if n_unique == 0:
                warnings.warn(f"color_by='{cb}': column has no values after filtering.")
                return [('All', pd.Series(True, index=df.index))]
            if pd.api.types.is_float_dtype(df[cb]) and n_unique > 10:
                warnings.warn(
                    f"color_by='{cb}' looks continuous (float dtype, {n_unique} unique values). "
                    f"Coloring by a continuous column will produce many overlapping lines. "
                    f"Consider binning or using a categorical column instead."
                )
            elif n_unique > 30:
                warnings.warn(
                    f"color_by='{cb}' has {n_unique} unique values — this will produce "
                    f"{n_unique} smoothed lines and may be hard to read. "
                    f"Consider a coarser grouping."
                )
            uvals = sorted(df[cb].dropna().unique(), key=str)
            return [(str(v), df[cb] == v) for v in uvals]
        return None  # not found

    _cb_list = [color_by] if isinstance(color_by, str) else list(color_by)

    if len(_cb_list) == 1:
        # ── Single grouping ──────────────────────────────────────────────────
        groups = _resolve_grouping(_cb_list[0])
        if groups is None:
            warnings.warn(f"color_by='{_cb_list[0]}' not found in data; "
                          "falling back to 'technical_group'.")
            groups = _resolve_grouping('technical_group')
        color_groups_list = groups

    else:
        # ── Cross-product of two groupings, with auto shade variation ────────
        # Primary grouping determines hue; secondary determines shade.
        # For 'targeting' as secondary: NTC → light shade, Targeting → full color.
        # For any other secondary: shades spread evenly from light to full.
        primary_cb, secondary_cb = _cb_list[0], _cb_list[1]
        primary_groups   = _resolve_grouping(primary_cb)   or _resolve_grouping('technical_group')
        secondary_groups = _resolve_grouping(secondary_cb) or [('All', pd.Series(True, index=df.index))]

        if secondary_cb == 'targeting':
            _shade_for = {'NTC': 0.35, 'Targeting': 1.0}
        else:
            _n = len(secondary_groups)
            _shade_for = {lbl: 0.3 + 0.7 * i / max(_n - 1, 1)
                          for i, (lbl, _) in enumerate(secondary_groups)}

        # Build cross-product; auto-populate color_palette with shade colours.
        # shade=1.0 → full base colour; shade<1.0 → mixed with white (lighter).
        from matplotlib.colors import to_rgb, to_hex
        color_palette = dict(color_palette)  # local copy — do not mutate caller's dict
        color_groups_list = []
        for pidx, (plabel, pmask) in enumerate(primary_groups):
            base_color = _color_for_label(plabel, fallback_idx=pidx, palette=color_palette)
            try:
                base_rgb = np.array(to_rgb(base_color))
            except ValueError:
                base_rgb = np.array([0.5, 0.5, 0.5])
            for slabel, smask in secondary_groups:
                combined_mask = pmask & smask
                if not combined_mask.any():
                    continue
                combined_label = f"{plabel} / {slabel}"
                if combined_label not in color_palette:
                    shade = _shade_for.get(slabel, 1.0)
                    shaded = np.clip(1 - shade * (1 - base_rgb), 0, 1)
                    color_palette[combined_label] = to_hex(shaded)
                color_groups_list.append((combined_label, combined_mask))

    # Create colormaps for NTC gradient — one per color group (white → group color)
    group_cmaps = {}
    if show_ntc_gradient:
        for idx, (group_label, _) in enumerate(color_groups_list):
            base_color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}", ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        # Compute per-cell y_expr with alpha_y correction applied per technical group.
        # This is independent of color_by — correction always uses technical_group_code.
        if corrected and has_technical_fit:
            alpha_y_full = modality.alpha_y_prefit  # [C, T] or [S, C, T]
            y_expr_vals = np.empty(len(df))
            if 'technical_group_code' in df.columns:
                for gc in df['technical_group_code'].unique():
                    gc = int(gc)
                    if alpha_y_full.ndim == 3:
                        a = _to_scalar(alpha_y_full[:, gc, feature_idx].mean())
                    else:
                        a = _to_scalar(alpha_y_full[gc, feature_idx])
                    mask = df['technical_group_code'].values == gc
                    y_expr_vals[mask] = (df['y_obs'].values[mask] /
                                         (df['sum_factor'].values[mask] * a))
            else:
                y_expr_vals = (df['y_obs'] / df['sum_factor']).values
        else:
            y_expr_vals = (df['y_obs'] / df['sum_factor']).values
        y_expr_all = pd.Series(y_expr_vals, index=df.index)

        colorbar_added = False
        for idx, (group_label, group_mask) in enumerate(color_groups_list):
            df_group = df[group_mask].copy()
            y_expr = y_expr_all[group_mask]

            # Filter valid cells for kNN smoothing.
            # x_true must be positive (needed for log2-axis and Hill function alignment).
            #
            # Zero-count cells (y_expr=0) treatment depends on mode:
            #
            # log2fc=True (axes shifted by NTC offsets):
            #   Exclude zeros → y_expr > 0.
            #   Reason: the y_offset is the model/empirical NTC mean *excluding zeros*.
            #   If zeros are included in the smooth they pull it below y_offset, causing
            #   NTC cells to appear at log2FC < 0 (spuriously below the NTC baseline).
            #
            # log2fc=False (absolute log2 axes, y_offset = 0):
            #   Include zeros → isfinite(y_expr).
            #   Reason: the Hill function (black dashed line) is the model's *unconditional*
            #   expected value E[y_obs/sum_factor] = A + α·Hill_a + β·Hill_b, which
            #   averages over both expressed and zero-count cells.  Excluding zeros raises
            #   the kNN smooth to E[y_obs/sum_factor | y_obs > 0] > E[y_obs/sum_factor],
            #   making the smooth appear above the Hill curve — misleadingly suggesting the
            #   model under-predicts the data.  Including zeros gives the arithmetic mean
            #   over all cells, which matches the Hill function in expectation.
            #   Windows where all cells have zero counts are removed by the subsequent
            #   valid_smooth = y_smooth_linear > 0 filter (no log2(0) is taken).
            if log2fc:
                valid = (df_group['x_true'] > 0) & (y_expr > 0)
            else:
                valid = (df_group['x_true'] > 0) & np.isfinite(y_expr)
            df_group = df_group[valid].copy()
            y_expr = y_expr[valid]

            if len(df_group) == 0:
                continue

            is_ntc_group = is_ntc[df_group.index].values
            k = _knn_k(len(df_group), window)

            # k-NN smoothing in LINEAR space first (log2(mean(y)) ≥ mean(log2(y)))
            if show_ntc_gradient:
                x_smooth, y_smooth_linear, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_expr.values if hasattr(y_expr, 'values') else np.asarray(y_expr),
                    k, is_ntc=is_ntc_group
                )
                valid_smooth = y_smooth_linear > 0
                if not valid_smooth.all():
                    x_smooth = x_smooth[valid_smooth]
                    y_smooth_linear = y_smooth_linear[valid_smooth]
                    ntc_prop = ntc_prop[valid_smooth]
                y_smooth_log = np.log2(y_smooth_linear)
                group_cmap = group_cmaps.get(group_label, plt.cm.gray)
                plot_colored_line(
                    x=np.log2(x_smooth) - x_offset,
                    y=y_smooth_log - y_offset,
                    color_values=1 - ntc_prop,
                    cmap=group_cmap, ax=ax_plot, linewidth=2
                )
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                ax_plot.plot([], [], color=color, linewidth=2, label=group_label)
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    cmap_gray = LinearSegmentedColormap.from_list("gray_gradient", ["white", "black"])
                    sm = ScalarMappable(cmap=cmap_gray, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                x_smooth, y_smooth_linear = _smooth_knn(
                    df_group['x_true'].values,
                    y_expr.values if hasattr(y_expr, 'values') else np.asarray(y_expr),
                    k
                )
                valid_smooth = y_smooth_linear > 0
                if not valid_smooth.all():
                    x_smooth = x_smooth[valid_smooth]
                    y_smooth_linear = y_smooth_linear[valid_smooth]
                y_smooth_log = np.log2(y_smooth_linear)
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                ax_plot.plot(np.log2(x_smooth) - x_offset, y_smooth_log - y_offset,
                             color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot if available, otherwise show on uncorrected plot
        if show_hill_function and (corrected or not has_technical_fit):
            # Evenly spaced points in log2 space for smooth curve on log-log plot
            log2_min = np.log2(max(x_true.min(), 1e-6))
            log2_max = np.log2(x_true.max())
            x_range = 2 ** np.linspace(log2_min, log2_max, 2000)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=None)

            if y_pred is not None:
                # Transform prediction to log2(y) space to match data
                # Filter out zero/negative predictions
                valid_pred = y_pred > 0
                if valid_pred.any():
                    ax_plot.plot(np.log2(x_range[valid_pred]) - x_offset,
                                np.log2(y_pred[valid_pred]) - y_offset,
                                color='black', linestyle='--', linewidth=2,
                                label='Fitted Trans Function')


            # Full parameter markers (replaces simple A baseline)
            # Use reference_df as FDR source if no explicit fdr_df provided
            _effective_fdr_df = fdr_df if fdr_df is not None else reference_df
            if mark_params and (corrected or not has_technical_fit):
                _markers = _compute_hill_markers(
                    model, feature, modality,
                    ci_level=ci_level, log2_space=True, y_scale=1.0,
                    fdr_df=_effective_fdr_df, fdr_threshold=fdr_threshold,
                )
                # In log2FC mode the axes are shifted; pass the same offsets so
                # that parameter lines (A, Vmax ceiling, EC50) land at the correct
                # positions — consistent with A_log2fc / EC50_*_log2fc in
                # save_trans_summary.
                _draw_hill_markers(ax_plot, _markers,
                                   x_offset=x_offset if log2fc else 0.0,
                                   y_offset=y_offset if log2fc else 0.0)

            # Reference curve overlay (from trans_summary DataFrame)
            if reference_df is not None and (corrected or not has_technical_fit):
                _name_col = next((c for c in ['gene_name', 'gene'] if c in reference_df.columns), None)
                if _name_col:
                    _ref_match = reference_df[reference_df[_name_col] == feature]
                    if not _ref_match.empty:
                        _ref_row = _ref_match.iloc[0]
                        _x_ref = x_range if 'x_range' in dir() else (
                            2 ** np.linspace(np.log2(max(x_true.min(), 1e-6)),
                                             np.log2(x_true.max()), 2000))
                        _y_ref = predict_hill_from_summary_row(_ref_row, _x_ref, fdr_threshold=fdr_threshold)
                        if _y_ref is not None:
                            _valid_ref = _y_ref > 0
                            if _valid_ref.any():
                                ax_plot.plot(
                                    np.log2(_x_ref[_valid_ref]) - x_offset,
                                    np.log2(_y_ref[_valid_ref]) - y_offset,
                                    color='red', linestyle='--', linewidth=2,
                                    alpha=0.8, label='Reference Function')
                        # Reference mark_params: full set of Hill markers (A, Vmax, K)
                        if mark_params:
                            _ref_markers = _compute_hill_markers_from_summary_row(
                                _ref_row, log2_space=True, y_scale=1.0,
                                fdr_threshold=fdr_threshold,
                            )
                            # Offset reference markers visually (thinner, semi-transparent)
                            for _m in _ref_markers:
                                _kw = dict(linestyle=_m['linestyle'], color='red',
                                           alpha=0.5, linewidth=1.2,
                                           label=f'Ref {_m["label"]}')
                                if _m['axis'] == 'h':
                                    ax_plot.axhline(_m['value'] - y_offset, **_kw)
                                elif _m['axis'] == 'v':
                                    ax_plot.axvline(_m['value'] - x_offset, **_kw)

        ax_plot.set_xlabel("log2FC(x_true)" if log2fc else xlabel)
        ax_plot.set_ylabel("log2FC(counts)" if log2fc else "log2(counts)")
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature}{title_suffix}")
        if log2fc:
            ax_plot.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
            ax_plot.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        if not legend_outside:
            ax_plot.legend(frameon=False)

    # Plot
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:
        _plot_one(axes[0], corrected=False)

    if legend_outside:
        handles, labels = _collect_legend_handles(axes)
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.03, 0.5), loc='center left', frameon=False)

    return axes[0] if len(axes) == 1 else axes


def plot_binomial_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    show_correction: str = 'both',
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    mark_params: bool = False,
    ci_level: float = 95.0,
    legend_outside: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot binomial (PSI - percent spliced in).

    Y-axis: PSI (%) = (counts / denominator) * 100  (percentage scale: 0-100)
    Filter: min_counts on denominator

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y_add additive correction (PSI - alpha_y_add)
        'both': show both side-by-side

    Notes
    -----
    Technical correction for binomial uses alpha_y_add (additive correction on LOGIT scale).
    logit(PSI) = log(PSI / (1 - PSI))
    logit_corrected = logit(PSI) - alpha_y_add
    PSI_corrected = 1 / (1 + exp(-logit_corrected))
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts and denominator
    if modality.cells_axis == 1:
        counts = modality.counts[feature_idx, :]
        denom = modality.denominator[feature_idx, :]
    else:
        counts = modality.counts[:, feature_idx]
        denom = modality.denominator[:, feature_idx]

    # Align cells between model.meta and modality
    x_true_aligned, counts_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, counts, subset_mask
    )
    # Also align denominator (same cell alignment)
    _, denom_aligned, _ = _align_cells_to_modality(
        model, modality, x_true, denom, subset_mask
    )

    # Filter by min_counts
    valid_mask = denom_aligned >= min_counts

    # Build dataframe
    df_data = {
        'x_true': x_true_aligned[valid_mask],
        'counts': counts_aligned[valid_mask],
        'denominator': denom_aligned[valid_mask],
        'target': meta_aligned['target'].values[valid_mask]
    }

    # Conditionally add technical_group_code if it exists
    if 'technical_group_code' in meta_aligned.columns:
        df_data['technical_group_code'] = meta_aligned['technical_group_code'].values[valid_mask]

    df = pd.DataFrame(df_data)

    # Keep counts and denominators for binning (don't compute PSI per-cell)
    # Filter valid
    valid = (df['x_true'] > 0) & (df['denominator'] > 0)
    df = df[valid].copy()

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Check technical correction availability
    # First check attribute, then fall back to posterior_samples_technical dict
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))
            axes = [ax]
    else:
        # Single axis provided - in multifeature mode, function is called twice
        # (once with 'uncorrected' and once with 'corrected'), so don't override show_correction
        axes = [ax]

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)

    if 'technical_group_code' in df.columns:
        group_codes = np.sort(df['technical_group_code'].unique())
    else:
        group_codes = np.array([0])  # Single group

    group_labels = [code_to_label[int(c)] for c in group_codes]

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormaps for NTC gradient (per technical group)
    # Each group gets white → group_color gradient
    group_cmaps = {}
    if show_ntc_gradient:
        for group_label in group_labels:
            base_color = color_palette.get(group_label, f'C0')
            # Create colormap: white (low color value) → group color (high color value)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}",
                ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added
        
        for idx, group_code in enumerate(group_codes):
            group_code = int(group_code)
            group_label = code_to_label[group_code]
            color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

            # Get data for this group
            if 'technical_group_code' in df.columns:
                group_mask = df['technical_group_code'] == group_code
            else:
                group_mask = pd.Series([True] * len(df), index=df.index)

            if group_mask.sum() == 0:
                continue

            x_group = df.loc[group_mask, 'x_true'].values
            counts_group = df.loc[group_mask, 'counts'].values       # (n_cells_group,)
            denom_group = df.loc[group_mask, 'denominator'].values   # (n_cells_group,)

            # Bin counts using k-NN
            k = _knn_k(len(x_group), window)
            x_smooth, counts_binned, denom_binned = _smooth_knn_counts(
                x_group, counts_group, denom_group, k
            )  # counts_binned: (n_bins,), denom_binned: (n_bins,)

            # Compute PSI from binned counts (as proportion [0, 1])
            epsilon = 1e-10
            with np.errstate(divide='ignore', invalid='ignore'):
                p_binned = np.where(denom_binned > epsilon,
                                    counts_binned / denom_binned,
                                    0.5)  # Default to 0.5 if no data

            if corrected and has_technical_fit:
                # Apply additive correction on LOGIT scale to binned proportions
                # 1. Clip proportions to avoid log(0) or log(1)
                epsilon_clip = 1e-6
                p_clipped = np.clip(p_binned, epsilon_clip, 1 - epsilon_clip)

                # 2. Calculate logit
                logit_p = np.log(p_clipped / (1 - p_clipped))

                # 3. Apply correction on logit scale
                alpha_y_add = modality.alpha_y_prefit_add
                if alpha_y_add.ndim == 3:
                    correction = _to_scalar(alpha_y_add[:, group_code, feature_idx].mean())
                else:
                    correction = _to_scalar(alpha_y_add[group_code, feature_idx])
                logit_corrected = logit_p - correction

                # 4. Convert back to proportion
                p_corrected = 1.0 / (1.0 + np.exp(-logit_corrected))

                # 5. Convert to percentage
                y_smooth = p_corrected * 100.0
            else:
                # Convert uncorrected proportion to percentage
                y_smooth = p_binned * 100.0

            # Plot (no additional smoothing needed - already binned)
            # Note: We don't currently support NTC gradient for binned data
            ax_plot.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot if available, otherwise show on uncorrected plot
        if show_trans_function and (corrected or not has_technical_fit):
            # Evenly spaced points in log2 space for smooth curve on log-log plot
            log2_min = np.log2(max(x_true.min(), 1e-6))
            log2_max = np.log2(x_true.max())
            x_range = 2 ** np.linspace(log2_min, log2_max, 2000)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

            if y_pred is not None:
                # For binomial, PSI is in percentage scale [0, 100], so scale and clip predictions
                y_pred_pct = y_pred * 100.0
                y_pred_clipped = np.clip(y_pred_pct, 0, 100)
                ax_plot.plot(np.log2(x_range), y_pred_clipped,
                           color='black', linestyle='--', linewidth=2,
                           label='Fitted Trans Function')

        # Full parameter markers (PSI%, linear y-axis)
        if mark_params and (corrected or not has_technical_fit):
            _markers = _compute_hill_markers(
                model, feature, modality,
                ci_level=ci_level, log2_space=False, y_scale=100.0
            )
            _draw_hill_markers(ax_plot, _markers)

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('PSI (%)')
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature} (min_counts={min_counts}{title_suffix})")
        if not legend_outside:
            ax_plot.legend(frameon=False)

    # Plot
    # NOTE: When ax is provided (multifeature mode), show_correction will be 'uncorrected' or 'corrected'
    # (forced by warning above), so we respect the show_correction value
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:  # show_correction == 'uncorrected'
        _plot_one(axes[0], corrected=False)

    if legend_outside:
        handles, labels = _collect_legend_handles(axes)
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.03, 0.5), loc='center left', frameon=False)

    return axes[0] if len(axes) == 1 else axes


def plot_multinomial_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> Union[plt.Figure, List[plt.Axes]]:
    """
    Plot multinomial (e.g., donor/acceptor usage) - one subplot per category.

    Y-axis: Proportion for each category
    Layout:
    - If show_correction='uncorrected' or 'corrected': K columns (one per category), 1 row
    - If show_correction='both': 2 columns (uncorrected left, corrected right), K rows (one per category)

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        **Note**: Not yet fully implemented for multinomial - will issue warning

    Notes
    -----
    Technical correction for multinomial uses alpha_y_add (additive on LOGIT scale).
    logits_corrected = logits - alpha_y_add
    proportions_corrected = softmax(logits_corrected)
    """
    if show_ntc_gradient:
        warnings.warn("NTC gradient coloring not yet implemented for multinomial distributions - using standard colors")

    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts: shape (cells, K) for this feature
    if modality.counts.ndim == 3:
        counts_3d = modality.counts[feature_idx, :, :]  # (cells, K)
    else:
        raise ValueError(f"Expected 3D counts for multinomial, got {modality.counts.ndim}D")

    K = counts_3d.shape[1]

    # Align cells between model.meta and modality
    # For multinomial, we need to align the first dimension (cells) of the 3D array
    # We'll use the total counts across categories for alignment
    total_counts_all = counts_3d.sum(axis=1)
    x_true_aligned, total_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, total_counts_all, subset_mask
    )

    # Now we need to align the full 3D counts array using the same cell mapping
    # Get the cell mapping
    if 'cell' in model.meta.columns:
        model_cells = model.meta['cell'].values
    else:
        model_cells = model.meta.index.values
    modality_cells = modality.cell_names
    modality_mask = np.isin(model_cells, modality_cells)
    if subset_mask is not None:
        final_mask = modality_mask & subset_mask
    else:
        final_mask = modality_mask
    final_cells = model_cells[final_mask]
    modality_cell_to_idx = {cell: idx for idx, cell in enumerate(modality_cells)}
    y_indices = np.array([modality_cell_to_idx[cell] for cell in final_cells])
    counts_3d = counts_3d[y_indices, :]  # Align 3D array

    # Get category labels if available in feature_meta
    category_labels = None
    if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
        if 'category_labels' in modality.feature_meta.columns:
            category_labels = modality.feature_meta.loc[feature_idx, 'category_labels']
            if category_labels is not None and len(category_labels) != K:
                warnings.warn(f"category_labels length ({len(category_labels)}) doesn't match K ({K}) - ignoring labels")
                category_labels = None

    # Identify non-zero categories (skip padded zeros and empty labels)
    # A category is non-zero if it has any counts across all cells AND has a non-empty label
    non_zero_cats = []
    for k in range(K):
        has_counts = counts_3d[:, k].sum() > 0
        has_label = True if category_labels is None else (k < len(category_labels) and category_labels[k] != "")
        if has_counts and has_label:
            non_zero_cats.append(k)

    if len(non_zero_cats) == 0:
        raise ValueError(f"No non-zero categories found for feature '{feature}'")

    K_plot = len(non_zero_cats)  # Only plot non-zero categories

    # Filter by min_counts (total across categories)
    total_counts = counts_3d.sum(axis=1)
    valid_mask = total_counts >= min_counts

    # Build dataframe
    df_data = {
        'x_true': x_true_aligned[valid_mask],
        'target': meta_aligned['target'].values[valid_mask]
    }

    # Conditionally add technical_group_code if it exists
    if 'technical_group_code' in meta_aligned.columns:
        df_data['technical_group_code'] = meta_aligned['technical_group_code'].values[valid_mask]

    df = pd.DataFrame(df_data)

    # Store raw counts for each category (needed for binning)
    counts_filtered = counts_3d[valid_mask, :]  # (n_cells_filtered, K)
    total_filtered = total_counts[valid_mask]

    # Filter valid x_true
    valid = df['x_true'] > 0
    df = df[valid].copy()
    counts_filtered = counts_filtered[valid, :]
    total_filtered = total_filtered[valid]

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Check technical correction availability
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create figure
    # Layout: if show_correction='both', then K_plot rows × 2 columns, else 1 row × K_plot columns
    if show_correction == 'both':
        if figsize is None:
            figsize = (12, 3 * K_plot)  # 2 columns, K_plot rows
        fig, axes = plt.subplots(K_plot, 2, figsize=figsize, squeeze=False,
                                 constrained_layout=True)
    else:
        if figsize is None:
            figsize = (4 * K_plot, 5)  # K_plot columns, 1 row
        fig, axes_row = plt.subplots(1, K_plot, figsize=figsize, squeeze=False,
                                     constrained_layout=True)
        axes = axes_row  # Shape: (1, K_plot)

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)

    if 'technical_group_code' in df.columns:
        group_codes = np.sort(df['technical_group_code'].unique())
    else:
        group_codes = np.array([0])  # Single group

    group_labels = [code_to_label[int(c)] for c in group_codes]

    # Plot each non-zero category
    for plot_idx, k in enumerate(non_zero_cats):
        # Get label for this category
        if category_labels is not None:
            cat_label = category_labels[k]
        else:
            cat_label = f"Category {k}"

        if show_correction == 'both':
            # Row plot_idx, columns 0 (uncorrected) and 1 (corrected)
            for col_idx, corrected in enumerate([False, True]):
                ax = axes[plot_idx, col_idx]

                for idx, group_code in enumerate(group_codes):
                    group_code = int(group_code)
                    group_label = code_to_label[group_code]
                    color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                    # Get data for this group
                    if 'technical_group_code' in df.columns:
                        group_mask = df['technical_group_code'] == group_code
                    else:
                        group_mask = pd.Series([True] * len(df), index=df.index)

                    if group_mask.sum() == 0:
                        continue

                    x_group = df.loc[group_mask, 'x_true'].values
                    counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                    totals_group = total_filtered[group_mask]      # (n_cells_group,)

                    # Bin counts using k-NN
                    knn = _knn_k(len(x_group), window)
                    x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                        x_group, counts_group, totals_group, knn
                    )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                    # Compute proportions from binned counts
                    epsilon = 1e-10
                    with np.errstate(divide='ignore', invalid='ignore'):
                        props_binned = np.where(totals_binned[:, None] > epsilon,
                                                counts_binned / totals_binned[:, None],
                                                1.0 / K)

                    if corrected and has_technical_fit:
                        # Compute zero-category mask
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                        # Apply correction to binned proportions
                        alpha_y_add = modality.alpha_y_prefit_add
                        props_corrected = _multinomial_correct_binned_probs(
                            props_binned=props_binned,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feature_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_bins, K)
                        y_smooth = props_corrected[:, k]
                    else:
                        y_smooth = props_binned[:, k]

                    # Plot (no additional smoothing needed - already binned)
                    ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                           label=group_label if plot_idx == 0 else None)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(f'Proportion')
                title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                ax.set_title(f"{cat_label}{title_suffix}")
                if plot_idx == 0:
                    ax.legend(frameon=False, loc='upper right')
        else:
            # Single row, column plot_idx
            ax = axes[0, plot_idx]
            corrected = (show_correction == 'corrected')

            for idx, group_code in enumerate(group_codes):
                group_code = int(group_code)
                group_label = code_to_label[group_code]
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                # Get data for this group
                if 'technical_group_code' in df.columns:
                    group_mask = df['technical_group_code'] == group_code
                else:
                    group_mask = pd.Series([True] * len(df), index=df.index)

                if group_mask.sum() == 0:
                    continue

                x_group = df.loc[group_mask, 'x_true'].values
                counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                totals_group = total_filtered[group_mask]      # (n_cells_group,)

                # Bin counts using k-NN
                knn = _knn_k(len(x_group), window)
                x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                    x_group, counts_group, totals_group, knn
                )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                # Compute proportions from binned counts
                epsilon = 1e-10
                with np.errstate(divide='ignore', invalid='ignore'):
                    props_binned = np.where(totals_binned[:, None] > epsilon,
                                            counts_binned / totals_binned[:, None],
                                            1.0 / K)

                if corrected and has_technical_fit:
                    # Compute zero-category mask
                    zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                    # Apply correction to binned proportions
                    alpha_y_add = modality.alpha_y_prefit_add
                    props_corrected = _multinomial_correct_binned_probs(
                        props_binned=props_binned,
                        alpha_y_add=alpha_y_add,
                        group_code=group_code,
                        feature_idx=feature_idx,
                        zero_cat_mask=zero_cat_mask
                    )  # (n_bins, K)
                    y_smooth = props_corrected[:, k]
                else:
                    y_smooth = props_binned[:, k]

                # Plot (no additional smoothing needed - already binned)
                ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                       label=group_label if plot_idx == 0 else None)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Proportion')
            title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
            ax.set_title(f"{cat_label}{title_suffix}")
            if plot_idx == 0:
                ax.legend(frameon=False, loc='upper right')

    plt.suptitle(f"{model.cis_gene} → {feature} (min_counts={min_counts})")
    return fig


def plot_normal_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    mark_params: bool = False,
    ci_level: float = 95.0,
    legend_outside: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot normal or studentT distribution (continuous scores like SpliZ).

    Y-axis: Raw value

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get values
    if modality.cells_axis == 1:
        y_vals = modality.counts[feature_idx, :]
    else:
        y_vals = modality.counts[:, feature_idx]

    # Align cells between model.meta and modality
    x_true_aligned, y_vals_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, y_vals, subset_mask
    )

    # Build dataframe
    df_data = {
        'x_true': x_true_aligned,
        'y_val': y_vals_aligned,
        'target': meta_aligned['target'].values
    }

    # Conditionally add technical_group_code if it exists
    if 'technical_group_code' in meta_aligned.columns:
        df_data['technical_group_code'] = meta_aligned['technical_group_code'].values

    df = pd.DataFrame(df_data)

    # Filter valid
    valid = (df['x_true'] > 0) & np.isfinite(df['y_val'])
    df = df[valid].copy()

    # Check technical correction
    # For normal distribution, check alpha_y_prefit_add (additive correction)
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from poster ior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))
            axes = [ax]
    else:
        # Single axis provided - in multifeature mode, function is called twice
        # (once with 'uncorrected' and once with 'corrected'), so don't override show_correction
        axes = [ax]

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)

    if 'technical_group_code' in df.columns:
        group_codes = np.sort(df['technical_group_code'].unique())
    else:
        group_codes = np.array([0])  # Single group

    group_labels = [code_to_label[int(c)] for c in group_codes]

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormaps for NTC gradient (per technical group)
    # Each group gets white → group_color gradient
    group_cmaps = {}
    if show_ntc_gradient:
        for group_label in group_labels:
            base_color = color_palette.get(group_label, f'C0')
            # Create colormap: white (low color value) → group color (high color value)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}",
                ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added
        
        for idx, group_code in enumerate(group_codes):
            group_code = int(group_code)
            group_label = code_to_label[group_code]
            color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
            df_group = df[df['technical_group_code'] == group_code].copy()

            if len(df_group) == 0:
                continue

            y_plot = df_group['y_val'].values

            if corrected and has_technical_fit:
                # Apply additive correction (alpha_y_add for normal)
                if hasattr(modality, 'alpha_y_prefit_add'):
                    alpha_y_add = modality.alpha_y_prefit_add
                    if alpha_y_add.ndim == 3:
                        correction = _to_scalar(alpha_y_add[:, group_code, feature_idx].mean())
                    else:
                        correction = _to_scalar(alpha_y_add[group_code, feature_idx])
                    y_plot = y_plot - correction

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index].values

            # k-NN smoothing
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient and not corrected:
                # Smoothing with NTC tracking
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_plot,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use per-group gradient coloring (white → group color)
                # Color value = 1 - ntc_prop: high NTC → 0 → white, low NTC → 1 → group color
                group_cmap = group_cmaps.get(group_label, plt.cm.gray)
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=y_smooth,
                    color_values=1 - ntc_prop,  # Darker (group color) = fewer NTCs
                    cmap=group_cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add dummy invisible line for legend label (using base group color)
                ax_plot.plot([], [], color=color, linewidth=2, label=group_label)

                # Add colorbar (once per axis) - use grayscale to show NTC gradient
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    cmap_gray = LinearSegmentedColormap.from_list("gray_gradient", ["white", "black"])
                    sm = ScalarMappable(cmap=cmap_gray, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_plot, k)

                # Use standard coloring
                ax_plot.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot if available, otherwise show on uncorrected plot
        if show_trans_function and (corrected or not has_technical_fit):
            # Evenly spaced points in log2 space for smooth curve on log-log plot
            log2_min = np.log2(max(x_true.min(), 1e-6))
            log2_max = np.log2(x_true.max())
            x_range = 2 ** np.linspace(log2_min, log2_max, 2000)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

            if y_pred is not None:
                ax_plot.plot(np.log2(x_range), y_pred,
                           color='black', linestyle='--', linewidth=2,
                           label='Fitted Trans Function')

        # Full parameter markers (linear y-axis)
        if mark_params and (corrected or not has_technical_fit):
            _markers = _compute_hill_markers(
                model, feature, modality,
                ci_level=ci_level, log2_space=False, y_scale=1.0
            )
            _draw_hill_markers(ax_plot, _markers)

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('Value')
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature}{title_suffix}")
        if not legend_outside:
            ax_plot.legend(frameon=False)

    # Plot
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:
        _plot_one(axes[0], corrected=False)

    if legend_outside:
        handles, labels = _collect_legend_handles(axes)
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.03, 0.5), loc='center left', frameon=False)

    return axes[0] if len(axes) == 1 else axes


# ============================================================================
# Multi-feature multinomial plotting
# ============================================================================

def _plot_multinomial_multifeature(
    model,
    feature_indices: List[int],
    feature_names: List[str],
    gene_name: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    show_correction: str,
    color_palette: Dict[str, str],
    xlabel: str,
    figsize: Optional[Tuple[int, int]] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot multiple multinomial features (e.g., multiple donor sites for one gene).

    Layout:
    - Single correction: n_features rows × K columns (dimensions as columns, features as rows)
    - Both corrections: (n_features × K) rows × 2 columns (dimensions in rows, corrections as columns)

    Parameters
    ----------
    model : bayesDREAM
        Fitted model
    feature_indices : List[int]
        Integer indices of features to plot
    feature_names : List[str]
        Names of features to plot
    gene_name : str
        Gene name (for title)
    modality : Modality
        Multinomial modality
    x_true : np.ndarray
        Cis expression values
    window : int
        k-NN window size
    min_counts : int
        Minimum total counts filter
    show_correction : str
        'uncorrected', 'corrected', or 'both'
    color_palette : Dict[str, str]
        Technical group colors
    xlabel : str
        X-axis label
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_features = len(feature_indices)

    # Compute proper cell alignment once for all features.
    # modality may cover fewer cells than model.meta, and subset_mask (if any) is on model.meta.
    # We combine both into y_indices (into modality cell order) and pre-aligned x_true / meta.
    _model_cells_arr = (model.meta['cell'].values if 'cell' in model.meta.columns
                        else model.meta.index.values)
    _modality_cells = modality.cell_names
    if _modality_cells is None:
        raise ValueError(f"Modality '{modality.name}' does not have cell_names set.")
    _modality_mask = np.isin(_model_cells_arr, _modality_cells)
    _final_mask = _modality_mask & subset_mask if subset_mask is not None else _modality_mask
    _final_cells = _model_cells_arr[_final_mask]
    _mod_cell_to_idx = {c: i for i, c in enumerate(_modality_cells)}
    y_indices = np.array([_mod_cell_to_idx[c] for c in _final_cells])
    x_true_aligned = x_true[_final_mask]   # x_true is full model x_true
    meta_aligned = model.meta[_final_mask].reset_index(drop=True)

    # Get number of categories (K) for each feature and identify non-zero categories
    # IMPORTANT: Must compute non-zero categories AFTER subsetting to get correct max_non_zero
    Ks = []
    non_zero_cats_per_feature = []

    for feat_idx in feature_indices:
        if modality.counts.ndim == 3:
            counts_3d = modality.counts[feat_idx, y_indices, :]  # (N_final, K) - cell-aligned

            K = counts_3d.shape[1]
            Ks.append(K)

            # Get category labels if available for this feature
            category_labels = None
            if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
                if 'category_labels' in modality.feature_meta.columns:
                    category_labels = modality.feature_meta.loc[feat_idx, 'category_labels']
                    if category_labels is not None and len(category_labels) != K:
                        category_labels = None

            # Identify non-zero categories AFTER subsetting (skip padded zeros and empty labels)
            non_zero_cats = []
            for k in range(K):
                has_counts = counts_3d[:, k].sum() > 0
                has_label = True if category_labels is None else (k < len(category_labels) and category_labels[k] != "")
                if has_counts and has_label:
                    non_zero_cats.append(k)

            non_zero_cats_per_feature.append(non_zero_cats)
        else:
            raise ValueError(f"Expected 3D counts for multinomial, got {modality.counts.ndim}D")

    # Check technical correction
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create figure layout using GridSpec to allocate rows based on actual categories per feature
    # This avoids blank space when features have different numbers of categories
    if show_correction == 'both':
        # Layout: each feature gets rows for its actual non-zero categories × 2 columns
        total_rows = sum(len(nz) for nz in non_zero_cats_per_feature)
        n_cols = 2
        if figsize is None:
            figsize = (12, 3 * total_rows)  # 3 inches per row
    else:
        # Layout: each feature gets 1 row × columns for its actual non-zero categories
        # Use GridSpec with variable column widths
        max_cats = max(len(nz) for nz in non_zero_cats_per_feature) if non_zero_cats_per_feature else 1
        total_rows = n_features
        n_cols = max_cats
        if figsize is None:
            figsize = (4 * max_cats, 3 * n_features)  # 4 inches per column, 3 inches per row

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(total_rows, n_cols)

    # Create axes dict to store subplot positions
    axes_dict = {}

    # Get technical group labels
    if 'technical_group_code' in model.meta.columns:
        group_labels = get_technical_group_labels(model)
        group_codes = sorted(model.meta['technical_group_code'].unique())
    else:
        group_labels = ['All']
        group_codes = [0]

    # Track current row position for GridSpec
    current_row = 0

    # Plot each feature
    for feat_i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names)):
        K = Ks[feat_i]
        non_zero_cats = non_zero_cats_per_feature[feat_i]

        if len(non_zero_cats) == 0:
            continue  # Skip features with no non-zero categories

        # Get counts for this feature, already cell-aligned via y_indices
        counts_3d = modality.counts[feat_idx, y_indices, :]  # (N_final, K)

        # Get category labels if available for this feature
        category_labels = None
        if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
            if 'category_labels' in modality.feature_meta.columns:
                category_labels = modality.feature_meta.loc[feat_idx, 'category_labels']
                if category_labels is not None and len(category_labels) != K:
                    warnings.warn(f"category_labels length ({len(category_labels)}) doesn't match K ({K}) for feature {feat_name} - ignoring labels")
                    category_labels = None

        # Use pre-computed non_zero_cats (already computed from subsetted data)
        # non_zero_cats was set at line 1897
        if len(non_zero_cats) == 0:
            continue  # Skip if no non-zero categories after subsetting

        # Filter by min_counts
        total_counts = counts_3d.sum(axis=1)
        valid_mask = total_counts >= min_counts

        # Build dataframe using pre-aligned x_true and meta
        df_data = {
            'x_true': x_true_aligned[valid_mask],
            'target': meta_aligned['target'].values[valid_mask]
        }
        if 'technical_group_code' in meta_aligned.columns:
            df_data['technical_group_code'] = meta_aligned['technical_group_code'].values[valid_mask]
        df = pd.DataFrame(df_data)

        # Keep raw counts for binning (don't compute proportions per-cell)
        counts_filtered = counts_3d[valid_mask, :]
        total_filtered = total_counts[valid_mask]

        # Filter valid x_true
        valid = df['x_true'] > 0
        df = df[valid].copy()
        counts_filtered = counts_filtered[valid, :]
        total_filtered = total_filtered[valid]

        if len(df) == 0:
            # Skip to next feature if no data after filtering
            # Note: we still need to advance current_row for 'both' mode
            if show_correction == 'both':
                current_row += len(non_zero_cats)
            continue

        # Map technical_group_code → label for THIS feature (after all filters)
        code_to_label = _labels_by_code_for_df(model, df)

        if 'technical_group_code' in df.columns:
            group_codes = np.sort(df['technical_group_code'].unique())
        else:
            group_codes = np.array([0])  # Single group

        group_labels = [code_to_label[int(c)] for c in group_codes]

        # Plot each non-zero category
        for cat_plot_idx, k in enumerate(non_zero_cats):
            # Get label for this category
            if category_labels is not None:
                cat_label = category_labels[k]
            else:
                cat_label = f"Cat{k}"

            if show_correction == 'both':
                # Create axes for this category row
                row = current_row + cat_plot_idx
                for col_idx, corrected in enumerate([False, True]):
                    ax = fig.add_subplot(gs[row, col_idx])

        
                    for idx, group_code in enumerate(group_codes):
                        group_code = int(group_code)
                        group_label = code_to_label[group_code]
                        color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                        # Get data for this group
                        group_mask = df['technical_group_code'] == group_code
                        if group_mask.sum() == 0:
                            continue

                        x_group = df.loc[group_mask, 'x_true'].values
                        counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                        totals_group = total_filtered[group_mask]      # (n_cells_group,)

                        # Bin counts using k-NN
                        knn = _knn_k(len(x_group), window)
                        x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                            x_group, counts_group, totals_group, knn
                        )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                        # Compute proportions from binned counts
                        epsilon = 1e-10
                        with np.errstate(divide='ignore', invalid='ignore'):
                            props_binned = np.where(totals_binned[:, None] > epsilon,
                                                    counts_binned / totals_binned[:, None],
                                                    1.0 / K)

                        if corrected and has_technical_fit:
                            # Compute zero-category mask
                            zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                            # Apply correction to binned proportions
                            alpha_y_add = modality.alpha_y_prefit_add
                            props_corrected = _multinomial_correct_binned_probs(
                                props_binned=props_binned,
                                alpha_y_add=alpha_y_add,
                                group_code=group_code,
                                feature_idx=feat_idx,
                                zero_cat_mask=zero_cat_mask
                            )  # (n_bins, K)
                            y_smooth = props_corrected[:, k]
                        else:
                            y_smooth = props_binned[:, k]

                        # Plot (no additional smoothing needed - already binned)
                        ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                               label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(f'Proportion')
                    title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                    ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                    if cat_plot_idx == 0 and feat_i == 0:
                        ax.legend(frameon=False, loc='upper right', fontsize=8)
            else:
                # Create ax for this category column
                ax = fig.add_subplot(gs[current_row, cat_plot_idx])
                corrected = (show_correction == 'corrected')
        
                for idx, group_code in enumerate(group_codes):
                    group_code = int(group_code)
                    group_label = code_to_label[group_code]
                    color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                    # Get data for this group
                    if 'technical_group_code' in df.columns:
                        group_mask = df['technical_group_code'] == group_code
                    else:
                        group_mask = pd.Series([True] * len(df), index=df.index)

                    if group_mask.sum() == 0:
                        continue

                    x_group = df.loc[group_mask, 'x_true'].values
                    counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                    totals_group = total_filtered[group_mask]      # (n_cells_group,)

                    # Bin counts using k-NN
                    knn = _knn_k(len(x_group), window)
                    x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                        x_group, counts_group, totals_group, knn
                    )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                    # Compute proportions from binned counts
                    epsilon = 1e-10
                    with np.errstate(divide='ignore', invalid='ignore'):
                        props_binned = np.where(totals_binned[:, None] > epsilon,
                                                counts_binned / totals_binned[:, None],
                                                1.0 / K)

                    if corrected and has_technical_fit:
                        # Compute zero-category mask
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                        # Apply correction to binned proportions
                        alpha_y_add = modality.alpha_y_prefit_add
                        props_corrected = _multinomial_correct_binned_probs(
                            props_binned=props_binned,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feat_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_bins, K)
                        y_smooth = props_corrected[:, k]
                    else:
                        y_smooth = props_binned[:, k]

                    # Plot (no additional smoothing needed - already binned)
                    ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                           label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(f'Proportion')
                title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                if cat_plot_idx == 0 and feat_i == 0:
                    ax.legend(frameon=False, loc='upper right', fontsize=8)

        # Update current_row for next feature (only for 'both' mode)
        if show_correction == 'both':
            current_row += len(non_zero_cats)
        else:
            current_row += 1

    plt.suptitle(f"{model.cis_gene} → {gene_name} (gene, n={n_features} features, min_counts={min_counts})")
    return fig


# ============================================================================
# Unified plot_xy_data function
# ============================================================================

def plot_xy_data(
    model,
    feature: str,
    modality_name: Optional[str] = None,
    window: int = 100,
    show_correction: str = 'both',
    min_counts: Optional[int] = None,
    color_palette: Optional[Dict[str, str]] = None,
    show_hill_function: bool = True,
    show_ntc_gradient: bool = False,
    sum_factor_col: str = 'sum_factor',
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    src_barcodes: Optional[np.ndarray] = None,
    subset_meta: Optional[Dict[str, Any]] = None,
    only_dependent: bool = False,
    ci_level: float = 95.0,
    mark_params: bool = False,
    legend_outside: bool = False,
    filename: Optional[str] = None,
    log2fc: bool = False,
    reference_df=None,
    fdr_df=None,
    fdr_threshold: float = 0.05,
    color_by: Union[str, List[str]] = 'technical_group',
    facet_by: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot raw x-y data showing relationship between cis gene expression and modality values.

    Requires x_true to be set (must run fit_cis() first).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model with x_true set
    feature : str
        Feature name (junction, donor, etc.) OR gene name.
        - If a specific feature name (e.g., 'chr1:999788:999865'), plots that feature
        - If a gene name (e.g., 'HES4'), plots all features for that gene in subplots
        - Requires modality to have gene information ('gene', 'gene_name', or 'gene_id' columns)
    modality_name : str, optional
        Modality name (default: primary modality)
    window : int
        k-NN window size for smoothing (default: 100 cells)
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y technical correction
        'both': show both side-by-side (default)
    min_counts : int, optional
        Minimum raw counts to include cells.
        Default depends on distribution: 0 for negbinom, 3 for binomial/multinomial.
        For negbinom: excludes cells with y_obs < min_counts
        For binomial: minimum denominator
        For multinomial: minimum total counts
    color_palette : dict, optional
        Custom colors for technical groups, keyed by the group label string
        Example: {'K562': 'crimson', 'TF1': 'dodgerblue'}
    show_hill_function : bool
        Overlay fitted trans function if trans model fitted (all distributions, default: True)
        Works with all function types: additive_hill, single_hill, polynomial
        Automatically detects function type from posterior_samples_trans
    show_ntc_gradient : bool
        Color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
        Fully implemented for: negbinom, binomial, normal, studentt
        Not yet implemented for: multinomial (will issue warning)
    sum_factor_col : str
        Column name in model.meta for sum factors (default: 'sum_factor')
        Can be 'sum_factor', 'sum_factor_adj', or any other sum factor column
        Only used for negbinom distribution (gene expression)
    xlabel : str
        X-axis label (default: "log2(x_true)")
    figsize : tuple, optional
        Figure size (auto-sized if None)
    src_barcodes : np.ndarray, optional
        Source barcode order if x_true not in model.meta order
    subset_meta : dict, optional
        Subset cells by metadata columns. Dictionary of {column: value} pairs.
        Example: {'target': 'ntc'} - plot only NTC cells
        Example: {'cell_line': 'K562'} - plot only K562 cells
        Example: {'lane': 'L1', 'cell_line': 'K562'} - plot L1 lane K562 cells
        Multiple conditions are combined with AND logic.
    only_dependent : bool
        If True and plotting multiple features (gene name), filter to only "dependent" features
        where the Hill coefficient (n_a or n_b) credible interval excludes 0 (default: False).
        Requires fit_trans() to have been run with function_type='additive_hill'.
        Ignored for single-feature plots.
    ci_level : float
        Credible interval level for dependency filtering and parameter marker
        classification (default: 95.0).
        Used by only_dependent=True and mark_params=True.
    mark_params : bool
        Overlay meaningful parameter markers on the Hill function (default: False).
        Requires show_hill_function=True and fit_trans() completed with
        function_type='single_hill' or 'additive_hill'.
        Markers drawn depend on the fitted regime:
        - **Single Hill / effectively single**: log2(A), log2(A+α·Vmax), log2(EC50)
        - **Same-sign additive** (both Hills in same direction):
          log2(A), log2(A+α·Vmax_a+β·Vmax_b), log2(EC50_lower), log2(EC50_upper)
        - **Non-monotonic** (opposite-sign Hills):
          log2(y: x→0), log2(y: x→∞), log2(EC50_lower), log2(EC50_upper);
          additionally log2(A+α·Vmax_a+β·Vmax_b) [peak] if activation EC50 < inhibition EC50
        For non-negbinom distributions, all y-axis markers are in linear space.
        Note: asymptotes include the α/β weight factors (e.g. A+α·Vmax, not A+Vmax).
    log2fc : bool
        If True, plot log2FC relative to NTC instead of raw log2 counts
        (default: False). Only applies to negbinom modalities.
        x-axis: log2(x_true) - log2(mean NTC x_true)
        y-axis: log2(counts) - log2(mean NTC counts, reference group)
        A grey dotted crosshair is drawn at (0, 0) to mark the NTC reference.
    color_by : str or list of two str
        What to color smoothed lines by (default: ``'technical_group'``).
        - ``'technical_group'``: one line per cell-line / technical group (default)
        - ``'targeting'``: two lines — ``NTC`` vs ``Targeting``
        - **Any column name in ``model.meta``**: one line per unique value.
          A warning is issued if the column looks continuous (float dtype with many
          unique values) or has more than 30 unique values.
        - **List of two** (e.g. ``['technical_group', 'targeting']``): cross-product of
          two groupings — the first sets hue, the second sets shade. When the secondary
          is ``'targeting'``, NTC lines are drawn as a light tint of the group colour
          and targeting lines as the full colour (matching the classic "shades" style).
          Override any auto-generated colour via ``color_palette`` using the combined
          label ``"K562 / NTC"``, ``"K562 / Targeting"``, etc.
        Alpha-y technical correction is always applied per technical group regardless
        of this setting.
    facet_by : str or list of str, optional
        Column name(s) to facet by (default: ``None``).  Accepts either a
        single string or a list of **at most two** strings.  Each value may be
        a real column name in ``model.meta`` **or** one of the same special
        keywords supported by ``color_by``:

        - ``'targeting'`` — two panels: ``NTC`` / ``Targeting``
          (derived from ``model.meta['target']``)
        - ``'technical_group'`` — one panel per technical group code

        Analogous to ``facet_grid`` in ggplot2.

        **1 column** (``facet_by='cell_line'``) — all unique values become
        side-by-side column groups::

            rows = n_features,  cols = n_facets × n_corrections

        Example: 3 cell lines + ``show_correction='both'`` →  1 × 6 grid
        ``[K562 uncorr | K562 corr | TF1 uncorr | TF1 corr | HL60 uncorr | HL60 corr]``

        **2 columns** (``facet_by=['cell_line', 'perturbation']``) — first
        column maps to grid *rows*, second column maps to grid *columns*::

            rows = n_facet1_values × n_features
            cols = n_facet2_values × n_corrections

        Example: 3 cell lines × 2 perturbation types + ``show_correction='corrected'``
        → 3 × 2 grid (cell lines as rows, perturbation types as columns).

        **Combining with other parameters:**

        - ``subset_meta``: each facet mask is AND-combined with the global
          ``subset_meta`` filter, so you can first narrow to CRISPRi cells and
          then facet by cell line within that subset.
        - ``color_by``: works independently within each panel — e.g.
          ``facet_by='cell_line'`` splits cell lines into columns while
          ``color_by='targeting'`` still draws NTC / Targeting lines *inside*
          each panel.
        - ``legend_outside=True``: recommended when faceting produces many
          panels, to keep the shared legend out of the plot area.

        Facet labels appear as the first line of each panel title, e.g.
        ``cell_line=K562`` or ``cell_line=K562  perturbation=CRISPRi``.
        A warning is issued for columns with more than 20 unique values.
        Not yet supported for ``multinomial`` distributions (ignored with a
        warning in that case).
    legend_outside : bool
        Place the legend outside the panel to the right, shared across all panels
        (default: False). Useful when many lines clutter the plot area.
    filename : str, optional
        If provided, save the figure to ``model.output_dir / filename``.
        A ``.png`` extension is added automatically if none is given.
        Use a full path to save elsewhere.
    figsize : tuple, optional
        Figure size as ``(width, height)`` in inches.  When None (default) the
        size is chosen automatically: 6 inches per column × 3 inches per row
        for multi-panel plots, or (8, 5) / (14, 5) for single-panel plots.
    reference_df : pd.DataFrame, optional
        A ``trans_summary`` DataFrame whose rows contain fitted Hill parameters
        (columns ending in ``_mean``, e.g. ``A_mean``, ``Vmax_a_mean``, etc.).
        When provided, a reference Hill curve is overlaid in red dashed on each
        negbinom panel.  The feature name is matched via a ``gene_name`` or
        ``gene`` column.  Only applied to negbinom modalities.
    fdr_df : pd.DataFrame, optional
        A ``trans_summary`` DataFrame with ``fdr_alpha`` and ``fdr_beta`` columns
        (e.g. the output of ``save_trans_summary``).  Used for two purposes:
        (1) greying out FDR-inactive parameter markers in ``mark_params`` mode,
        (2) classifying the Hill regime for ``_compute_hill_markers``.
        Feature names are matched via a ``gene_name`` or ``gene`` column.
    fdr_threshold : float
        FDR threshold for classifying components as active (default: 0.05).
        Components with ``fdr_alpha`` or ``fdr_beta`` >= this threshold are
        treated as inactive.  Only used when ``fdr_df`` is provided.
    **kwargs
        Additional plotting arguments

    Returns
    -------
    plt.Figure or plt.Axes
        Matplotlib figure or axes object

    Raises
    ------
    ValueError
        If x_true not set (must run fit_cis first)
        If feature not found in modality
        If show_correction='corrected' but fit_technical not run

    Warnings
    --------
    If fit_technical not run for modality and show_correction='corrected',
    warns and plots uncorrected only.

    Examples
    --------
    >>> # Plot single gene with Hill function
    >>> model.plot_xy_data('TET2', window=100, show_hill_function=True)
    >>>
    >>> # Plot specific splice junction with min_counts filter
    >>> model.plot_xy_data('chr1:12345:67890:+', modality_name='splicing_sj',
    ...                     min_counts=5)
    >>>
    >>> # Plot all splice junctions for a gene (creates multi-panel figure)
    >>> model.plot_xy_data('HES4', modality_name='splicing_sj')
    >>>
    >>> # Plot with custom colors
    >>> model.plot_xy_data('GFI1B', color_palette={'K562': 'red', 'TF1': 'blue'})
    >>>
    >>> # Show both corrected and uncorrected (default)
    >>> model.plot_xy_data('TET2', show_correction='both')
    >>>
    >>> # Plot with NTC gradient coloring
    >>> model.plot_xy_data('TET2', show_ntc_gradient=True, show_correction='uncorrected')
    >>>
    >>> # Plot only NTC cells
    >>> model.plot_xy_data('TET2', subset_meta={'target': 'ntc'})
    >>>
    >>> # Plot only cells from one group
    >>> model.plot_xy_data('GFI1B', subset_meta={'cell_line': 'K562'})
    >>>
    >>> # Plot all junctions for a gene, but only show dependent ones (n_a or n_b CI excludes 0)
    >>> model.plot_xy_data('HES4', modality_name='splicing_sj', only_dependent=True, ci_level=95.0)
    >>>
    >>> # ── color_by examples ─────────────────────────────────────────────────
    >>> # Color by NTC vs targeting instead of technical group
    >>> model.plot_xy_data('GFI1B', color_by='targeting')
    >>>
    >>> # Color by any column in model.meta (warns if continuous or >30 unique values)
    >>> model.plot_xy_data('GFI1B', color_by='cell_line')
    >>>
    >>> # Cross-product: hue = technical group, shade = NTC vs targeting
    >>> # NTC cells are drawn as a light tint of the group colour;
    >>> # targeting cells as the full colour.
    >>> model.plot_xy_data('GFI1B', color_by=['technical_group', 'targeting'])
    >>>
    >>> # Cross-product with a custom secondary grouping
    >>> model.plot_xy_data('GFI1B', color_by=['cell_line', 'perturbation_type'])
    >>>
    >>> # ── facet_by: 1-column (values → columns) ────────────────────────────
    >>> # One column-group of panels per cell line
    >>> # Grid: 1 row × (n_cell_lines × 2) cols  (default show_correction='both')
    >>> model.plot_xy_data('GFI1B', facet_by='cell_line')
    >>>
    >>> # Facet by cell line, show only corrected, log2FC mode
    >>> model.plot_xy_data('GFI1B',
    ...     facet_by='cell_line',
    ...     show_correction='corrected',
    ...     log2fc=True)
    >>>
    >>> # Facet by cell line AND color by NTC vs targeting within each facet panel
    >>> # Grid: 1 row × n_cell_lines cols; two lines per panel (NTC / Targeting)
    >>> model.plot_xy_data('GFI1B',
    ...     facet_by='cell_line',
    ...     color_by='targeting',
    ...     show_correction='corrected',
    ...     legend_outside=True)   # recommended when many panels
    >>>
    >>> # First restrict to CRISPRi guides, then facet by cell line within that subset
    >>> model.plot_xy_data('GFI1B',
    ...     subset_meta={'guide_type': 'CRISPRi'},
    ...     facet_by='cell_line')
    >>>
    >>> # ── facet_by: 2-column (proper row × col grid) ────────────────────────
    >>> # First column → row groups, second column → column groups
    >>> # Grid: n_cell_lines rows × n_perturbation_types cols (each corrected only)
    >>> model.plot_xy_data('GFI1B',
    ...     facet_by=['cell_line', 'perturbation_type'],
    ...     show_correction='corrected')
    >>>
    >>> # 2-D facet with color_by and log2FC; legend outside the grid
    >>> model.plot_xy_data('GFI1B',
    ...     facet_by=['cell_line', 'perturbation_type'],
    ...     color_by='targeting',
    ...     show_correction='corrected',
    ...     log2fc=True,
    ...     legend_outside=True)
    >>>
    >>> # Multi-feature + 1-D facet: rows = features, cols = facets × corrections
    >>> model.plot_xy_data('HES4', modality_name='splicing_sj',
    ...     facet_by='cell_line',
    ...     show_correction='corrected',
    ...     only_dependent=True)
    >>>
    >>> # Multi-feature + 2-D facet:
    >>> # rows = n_cell_lines × n_features,  cols = n_perturb_types × n_corrections
    >>> model.plot_xy_data('HES4', modality_name='splicing_sj',
    ...     facet_by=['cell_line', 'perturbation_type'],
    ...     show_correction='corrected',
    ...     only_dependent=True)
    """
    # Check x_true is set
    if not hasattr(model, 'x_true') or model.x_true is None:
        raise ValueError(
            "x_true not set. Must run fit_cis() before plotting x-y data.\n"
            "Example: model.fit_cis(sum_factor_col='sum_factor')"
        )

    # Check technical_group_code is set (only required if showing correction)
    # Auto-fallback to uncorrected if technical groups not available
    has_technical_groups = 'technical_group_code' in model.meta.columns

    if show_correction in ['corrected', 'both']:
        if not has_technical_groups:
            import warnings
            warnings.warn(
                f"technical_group_code not set, falling back to show_correction='uncorrected'. "
                f"To show correction, run model.set_technical_groups(['cell_line']) first.",
                UserWarning
            )
            show_correction = 'uncorrected'

    # Get x_true
    x2d = _to_2d(model.x_true)
    x_true = x2d.mean(axis=0) if x2d.ndim == 2 else x2d.ravel()

    # Reorder if needed
    if src_barcodes is not None:
        cell_barcodes = model.meta['cell'].values if 'cell' in model.meta.columns else model.meta.index.values
        x_true = reorder_xtrue_by_barcode(x_true, src_barcodes, cell_barcodes)

    # Apply metadata subsetting
    subset_mask = None
    if subset_meta is not None:
        # Create mask based on subset_meta conditions (AND logic)
        subset_mask = np.ones(len(model.meta), dtype=bool)
        for col, value in subset_meta.items():
            if col not in model.meta.columns:
                raise ValueError(f"Column '{col}' not found in model.meta. Available columns: {list(model.meta.columns)}")
            subset_mask &= (model.meta[col] == value).values

        n_cells_before = len(model.meta)
        n_cells_after = subset_mask.sum()
        if n_cells_after == 0:
            raise ValueError(f"No cells match subset_meta criteria: {subset_meta}")

        print(f"[SUBSET] Filtering {n_cells_before} → {n_cells_after} cells based on {subset_meta}")

        # NOTE: Don't subset x_true here - let _align_cells_to_modality() handle it
        # (subsetting here causes IndexError when mask is applied again in alignment)

    # Resolve facet groups (if facet_by is set).
    # facet_by can be a single column name (str) or a list of 1-2 column names.
    #   1 column  → all unique values become column groups (existing behaviour)
    #   2 columns → first column → row groups, second column → column groups (2-D grid)
    # Each group is a (label, boolean_mask on model.meta rows) pair.
    facet_col_groups = None   # unique values of the *column* facet dimension
    facet_row_groups = None   # unique values of the *row* facet dimension (2-column only)
    _facet_by_list: List[str] = []  # normalised list, used for title-building later

    if facet_by is not None:
        _facet_by_list = [facet_by] if isinstance(facet_by, str) else list(facet_by)
        if len(_facet_by_list) > 2:
            raise ValueError(
                f"facet_by supports at most 2 columns (got {len(_facet_by_list)}). "
                f"Pass a string for a 1-D column facet, or a list of two strings for a "
                f"2-D row × column facet grid."
            )

        def _resolve_facet_col(col: str, role: str):
            """Return [(label, mask), ...] for one facet column; warn on edge cases.

            Supports the same special keywords as color_by:
              'targeting'       → two groups: NTC / Targeting
              'technical_group' → one group per technical_group_code
            Any other value is looked up as a column in model.meta.
            """
            # ── Special keyword: 'targeting' ──────────────────────────────────
            if col == 'targeting':
                if 'target' not in model.meta.columns:
                    raise ValueError(
                        "facet_by='targeting' requires a 'target' column in model.meta."
                    )
                ntc_mask = (model.meta['target'].str.lower() == 'ntc').values
                return [('NTC', ntc_mask), ('Targeting', ~ntc_mask)]

            # ── Special keyword: 'technical_group' ────────────────────────────
            if col == 'technical_group':
                if 'technical_group_code' not in model.meta.columns:
                    warnings.warn(
                        "facet_by='technical_group' requires technical_group_code in "
                        "model.meta. Using a single 'All' group.",
                        UserWarning
                    )
                    return [('All', np.ones(len(model.meta), dtype=bool))]
                grp_labels = get_technical_group_labels(model)
                codes = np.sort(model.meta['technical_group_code'].unique())
                return [
                    (grp_labels[int(c)] if int(c) < len(grp_labels) else str(int(c)),
                     (model.meta['technical_group_code'] == c).values)
                    for c in codes
                ]

            # ── Regular model.meta column ─────────────────────────────────────
            if col not in model.meta.columns:
                raise ValueError(
                    f"facet_by column '{col}' not found in model.meta. "
                    f"Available columns: {list(model.meta.columns)}\n"
                    f"Special keywords also accepted: 'targeting', 'technical_group'."
                )
            fvals = sorted(model.meta[col].dropna().unique(), key=str)
            n_uniq = len(fvals)
            if n_uniq > 20:
                warnings.warn(
                    f"facet_by='{col}' ({role}) has {n_uniq} unique values — this will "
                    f"create {n_uniq} panel {role}s. Consider a coarser grouping.",
                    UserWarning
                )
            elif n_uniq <= 1:
                warnings.warn(
                    f"facet_by='{col}' has {n_uniq} unique value(s) — faceting has no effect.",
                    UserWarning
                )
            return [(str(v), (model.meta[col] == v).values) for v in fvals]

        if len(_facet_by_list) == 1:
            facet_col_groups = _resolve_facet_col(_facet_by_list[0], 'column')
        else:
            # Two columns: first → row groups, second → column groups
            facet_row_groups = _resolve_facet_col(_facet_by_list[0], 'row')
            facet_col_groups = _resolve_facet_col(_facet_by_list[1], 'column')

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)
    distribution = modality.distribution

    # Set distribution-specific default for min_counts
    if min_counts is None:
        if modality.distribution == 'negbinom':
            min_counts = 0  # Include all cells for gene expression
        else:
            min_counts = 3  # Require min counts for binomial/multinomial

    # Get color palette
    if color_palette is None:
        group_labels = get_technical_group_labels(model)
        color_palette = get_default_color_palette(group_labels)

    # Resolve feature(s) - could be single feature or gene name
    feature_indices, feature_names_resolved, is_gene = _resolve_features(feature, modality)

    # Filter to only dependent features if requested (multifeature mode only)
    if only_dependent and is_gene and len(feature_indices) > 1:
        n_before = len(feature_indices)

        if fdr_df is not None:
            # FDR-based filtering: use min(fdr_alpha, fdr_beta) per feature.
            # fdr_beta is NaN for single_hill/polynomial — nanmin treats NaN as inf.
            fdr_lookup = {}
            if 'feature' not in fdr_df.columns:
                warnings.warn(
                    "fdr_df must have a 'feature' column matching modality feature names. "
                    "Showing all features instead.",
                    UserWarning
                )
            else:
                for _, row in fdr_df.iterrows():
                    fa = row.get('fdr_alpha', np.nan)
                    fb = row.get('fdr_beta', np.nan)
                    fdr_lookup[row['feature']] = float(np.nanmin([fa, fb]))

                kept_indices = []
                kept_names = []
                for idx, name in zip(feature_indices, feature_names_resolved):
                    fdr_val = fdr_lookup.get(name, np.nan)
                    if np.isfinite(fdr_val) and fdr_val <= fdr_threshold:
                        kept_indices.append(idx)
                        kept_names.append(name)

                if len(kept_indices) == 0:
                    warnings.warn(
                        f"No features passed FDR <= {fdr_threshold} for '{feature}'. "
                        f"Showing all {n_before} features instead.",
                        UserWarning
                    )
                else:
                    feature_indices = kept_indices
                    feature_names_resolved = kept_names
                    print(f"[DEPENDENCY FILTER] {feature}: {n_before} → {len(feature_indices)} features "
                          f"(FDR <= {fdr_threshold})")

        else:
            # Fallback: CI-based filtering on n_a / n_b from posterior samples.
            # Requires fit_trans() with function_type='additive_hill'.
            if modality_name == model.primary_modality:
                posterior = getattr(model, 'posterior_samples_trans', None)
            else:
                posterior = getattr(modality, 'posterior_samples_trans', None)

            if posterior is None:
                warnings.warn(
                    "only_dependent=True requires fit_trans() to have been run "
                    f"(modality='{modality_name}'). Showing all features instead.",
                    UserWarning
                )
            elif 'n_a' not in posterior or 'n_b' not in posterior:
                warnings.warn(
                    "only_dependent=True without fdr_df requires fit_trans() with "
                    "function_type='additive_hill' (needs n_a/n_b). "
                    "Showing all features instead.",
                    UserWarning
                )
            else:
                n_a_samps = posterior['n_a'].detach().cpu().numpy()
                n_b_samps = posterior['n_b'].detach().cpu().numpy()
                if n_a_samps.ndim == 3:
                    n_a_samps = n_a_samps.squeeze(1)
                    n_b_samps = n_b_samps.squeeze(1)

                dep_mask_a = dependency_mask_from_n(n_a_samps, ci=ci_level)
                dep_mask_b = dependency_mask_from_n(n_b_samps, ci=ci_level)
                dep_mask = dep_mask_a | dep_mask_b

                filtered_indices = [idx for idx in feature_indices if dep_mask[idx]]
                filtered_names = [name for idx, name in zip(feature_indices, feature_names_resolved)
                                  if dep_mask[idx]]

                if len(filtered_indices) == 0:
                    warnings.warn(
                        f"No dependent features found for '{feature}' (all n CIs include 0). "
                        f"Showing all {n_before} features instead.",
                        UserWarning
                    )
                else:
                    feature_indices = filtered_indices
                    feature_names_resolved = filtered_names
                    print(f"[DEPENDENCY FILTER] {feature}: {n_before} → {len(feature_indices)} features "
                          f"(CI={ci_level}%, {dep_mask_a.sum()} positive, {dep_mask_b.sum()} negative)")

    # ── Grid path: multi-feature (gene input) OR single feature with facet_by ──
    # Multinomial is handled separately (has its own K-category subplot layout).
    use_grid = (is_gene and len(feature_indices) > 1) or (facet_by is not None)

    if use_grid:
        # Special handling for multinomial - needs K subplots per feature
        if distribution == 'multinomial':
            if facet_by is not None:
                warnings.warn(
                    "facet_by is not yet supported for multinomial distribution. "
                    "Ignoring facet_by and plotting without faceting.",
                    UserWarning
                )
            return _plot_multinomial_multifeature(
                model=model,
                feature_indices=feature_indices,
                feature_names=feature_names_resolved,
                gene_name=feature,
                modality=modality,
                x_true=x_true,  # full model x_true; alignment to modality cells done inside
                window=window,
                min_counts=min_counts,
                show_correction=show_correction,
                color_palette=color_palette,
                xlabel=xlabel,
                figsize=figsize,
                subset_mask=subset_mask,
                **kwargs
            )

        # Standard grid for 2D distributions.
        # Layout:
        #   rows = n_facet_rows × n_features   (facet-row groups nest features)
        #   cols = n_facet_cols × n_corrections (facet-col groups nest correction mode)
        # When facet_by is a single column, facet_row_groups is None (n_facet_rows=1),
        # so the layout reduces to rows=features, cols=n_facets×n_corrections as before.
        n_features = len(feature_indices)
        corrections_list = ['uncorrected', 'corrected'] if show_correction == 'both' else [show_correction]
        n_corrections = len(corrections_list)
        n_facet_rows = len(facet_row_groups) if facet_row_groups else 1
        n_facet_cols = len(facet_col_groups) if facet_col_groups else 1
        n_rows = n_facet_rows * n_features
        n_cols = n_facet_cols * n_corrections

        if figsize is None:
            figsize = (6 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False,
                                 constrained_layout=True)

        def _plot_one_grid(feat_name, ax, correction, mask,
                           ntc_x_offset=None, ntc_y_offset=None):
            """Plot one panel: one feature × one correction × one (optional) facet."""
            if distribution == 'negbinom':
                plot_negbinom_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction=correction,
                    color_palette=color_palette, show_hill_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, sum_factor_col=sum_factor_col,
                    min_counts=min_counts, xlabel=xlabel, ax=ax,
                    subset_mask=mask, mark_params=mark_params, ci_level=ci_level,
                    log2fc=log2fc, reference_df=reference_df,
                    fdr_df=fdr_df, fdr_threshold=fdr_threshold,
                    color_by=color_by,
                    ntc_x_offset=ntc_x_offset, ntc_y_offset=ntc_y_offset,
                    **kwargs
                )
            elif distribution == 'binomial':
                plot_binomial_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction=correction,
                    min_counts=min_counts, color_palette=color_palette,
                    show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=ax,
                    subset_mask=mask, mark_params=mark_params, ci_level=ci_level,
                    **kwargs
                )
            elif distribution in ('normal', 'studentt'):
                plot_normal_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction=correction,
                    color_palette=color_palette, show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=ax,
                    subset_mask=mask, mark_params=mark_params, ci_level=ci_level,
                    **kwargs
                )
            else:
                ax.text(0.5, 0.5, f"Grid not supported for {distribution}",
                        ha='center', va='center', transform=ax.transAxes)

        # Iterate: features (outer) → row-facets → col-facets → corrections.
        # Features are outer so we can compute the global NTC log2FC offsets once per
        # feature (using the global subset_mask, not any facet mask) and reuse them
        # across all facet panels — ensuring the NTC reference is consistent even when
        # a facet panel contains no NTC cells (e.g. a "Targeting" column facet).
        facet_col_iter = facet_col_groups if facet_col_groups else [(None, None)]
        facet_row_iter = facet_row_groups if facet_row_groups else [(None, None)]

        for feat_i, feat_name in enumerate(feature_names_resolved):
            # Pre-compute global NTC log2FC offsets for this feature (negbinom only).
            if log2fc and distribution == 'negbinom':
                _ntc_x_off, _ntc_y_off = _compute_global_log2fc_offsets(
                    model, modality, feat_name, x_true, subset_mask, sum_factor_col
                )
            else:
                _ntc_x_off = _ntc_y_off = None

            for frow_i, (frow_label, frow_mask) in enumerate(facet_row_iter):
                grid_row = frow_i * n_features + feat_i
                for fcol_i, (fcol_label, fcol_mask) in enumerate(facet_col_iter):
                    # Combine all masks (global subset_mask AND row-facet AND col-facet)
                    combined_mask = subset_mask
                    for m in [frow_mask, fcol_mask]:
                        if m is not None:
                            combined_mask = (combined_mask & m
                                             if combined_mask is not None else m)

                    for corr_i, correction in enumerate(corrections_list):
                        grid_col = fcol_i * n_corrections + corr_i
                        ax = axes[grid_row, grid_col]
                        _plot_one_grid(feat_name, ax, correction, combined_mask,
                                       ntc_x_offset=_ntc_x_off, ntc_y_offset=_ntc_y_off)
                        # Strip the per-panel legend — one shared legend is drawn below
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()
                        # Build title prefix from active facet labels
                        title_parts = []
                        if frow_label is not None:
                            title_parts.append(f"{_facet_by_list[0]}={frow_label}")
                        if fcol_label is not None:
                            col_key = (_facet_by_list[1] if len(_facet_by_list) > 1
                                       else _facet_by_list[0])
                            title_parts.append(f"{col_key}={fcol_label}")
                        if title_parts:
                            current_title = ax.get_title()
                            ax.set_title("  ".join(title_parts) + "\n" + current_title)

        # Overall suptitle for multi-feature plots
        if is_gene and len(feature_indices) > 1:
            plt.suptitle(
                f"{model.cis_gene} → {feature} (gene, n={len(feature_indices)} features)"
            )

        # One shared legend for the whole grid (no duplicates across panels).
        handles, labels = _collect_legend_handles(axes.ravel())
        if handles:
            if legend_outside:
                # Anchor to the right edge of the figure (transFigure coords)
                fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5),
                           loc='center left', frameon=False)
            else:
                axes[-1, -1].legend(handles, labels, frameon=False)

        _save_figure(fig, model, filename)
        return fig

    # ── Single-feature path (no faceting) ─────────────────────────────────────
    feature = feature_names_resolved[0]  # Use resolved feature name

    if distribution == 'negbinom':
        # Pre-compute global NTC offsets for log2FC mode (same approach as grid path).
        # This ensures the NTC reference is computed from ALL NTC cells (ignoring
        # min_counts and color_by grouping), making y_ntc independent of which colour
        # or subset is shown.
        _sf_ntc_x_off = _sf_ntc_y_off = None
        if log2fc:
            _sf_ntc_x_off, _sf_ntc_y_off = _compute_global_log2fc_offsets(
                model, modality, feature, x_true, subset_mask, sum_factor_col
            )

        result = plot_negbinom_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            color_palette=color_palette,
            show_hill_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            sum_factor_col=sum_factor_col,
            min_counts=min_counts,
            xlabel=xlabel,
            subset_mask=subset_mask,
            mark_params=mark_params,
            ci_level=ci_level,
            legend_outside=legend_outside,
            figsize=figsize,
            log2fc=log2fc,
            reference_df=reference_df,
            fdr_df=fdr_df,
            fdr_threshold=fdr_threshold,
            color_by=color_by,
            ntc_x_offset=_sf_ntc_x_off,
            ntc_y_offset=_sf_ntc_y_off,
            **kwargs
        )

    elif distribution == 'binomial':
        result = plot_binomial_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            min_counts=min_counts,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            subset_mask=subset_mask,
            mark_params=mark_params,
            ci_level=ci_level,
            legend_outside=legend_outside,
            figsize=figsize,
            **kwargs
        )

    elif distribution == 'multinomial':
        result = plot_multinomial_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            min_counts=min_counts,
            show_correction=show_correction,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            figsize=figsize,
            subset_mask=subset_mask,
            **kwargs
        )

    elif distribution in ('normal', 'studentt'):
        result = plot_normal_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            subset_mask=subset_mask,
            mark_params=mark_params,
            ci_level=ci_level,
            legend_outside=legend_outside,
            figsize=figsize,
            **kwargs
        )

    else:
        raise ValueError(f"Plotting not implemented for distribution '{distribution}'")

    # Save single-feature figure
    if filename is not None:
        ax_result = result if isinstance(result, plt.Axes) else result[0]
        _save_figure(ax_result.get_figure(), model, filename)
    return result
