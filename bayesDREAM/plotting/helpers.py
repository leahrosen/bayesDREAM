"""
Helper utility functions for bayesDREAM plotting.
"""

import numpy as np

# NTC target name variants recognised across the codebase
_NTC_VARIANTS = frozenset({
    'ntc', 'NTC', 'non-targeting', 'non-targeting-control',
    'Non-Targeting', 'non_targeting',
})


def to_np(a):
    """
    Safely convert torch/array-like to numpy.

    Parameters
    ----------
    a : torch.Tensor or array-like
        Input to convert

    Returns
    -------
    np.ndarray
        Numpy array
    """
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def resolve_guide_labels(model, single_guide_cells_only=False):
    """
    Return effective guide labels and a keep-mask for every cell in the model.

    Low-MOI (default)
    -----------------
    Reads ``model.meta['guide']`` directly.  All cells are kept.

    High-MOI
    --------
    Requires ``single_guide_cells_only=True``; raises ``ValueError`` otherwise.

    When enabled, the following rules are applied per cell:

    * Multiple targeting guides → cell is **excluded**.
    * Exactly 1 targeting guide (+ any NTC guides) → labelled by the targeting guide;
      NTC co-assignments are ignored.
    * Only NTC guides:
        - 1 NTC guide → labelled by that guide's name.
        - >1 NTC guides → labelled ``'multiple_NTC'``.

    Parameters
    ----------
    model : bayesDREAM
        Fitted model.
    single_guide_cells_only : bool, default False
        Must be ``True`` for high-MOI models.

    Returns
    -------
    guide_labels : np.ndarray of str, shape (N_cells,)
        Effective guide label for each cell.
    cell_mask : np.ndarray of bool, shape (N_cells,)
        ``True`` for cells to include in the plot.

    Raises
    ------
    ValueError
        If the model is in high-MOI mode and *single_guide_cells_only* is ``False``.
    """
    if not getattr(model, 'is_high_moi', False):
        if 'guide' not in model.meta.columns:
            raise ValueError("model.meta does not have a 'guide' column.")
        guide_labels = model.meta['guide'].astype(str).to_numpy()
        cell_mask = np.ones(len(guide_labels), dtype=bool)
        return guide_labels, cell_mask

    # ------------------------------------------------------------------ #
    # High-MOI path                                                        #
    # ------------------------------------------------------------------ #
    if not single_guide_cells_only:
        raise ValueError(
            "This model is in high-MOI mode (cells may carry multiple guides). "
            "Guide-level plots cannot be produced without subsetting. "
            "Pass single_guide_cells_only=True to restrict to cells that have "
            "at most one targeting guide."
        )

    guide_assignment = model.guide_assignment            # [N_cells, G_guides]
    guide_names = model.guide_meta['guide'].values       # [G_guides]

    # Identify which guide columns are NTC
    guide_targets_dict = getattr(model, 'guide_targets_dict', {})
    if 'target' in model.guide_meta.columns:
        is_ntc = model.guide_meta['target'].isin(_NTC_VARIANTS).values
    elif guide_targets_dict:
        is_ntc = np.array([
            any(t in _NTC_VARIANTS for t in guide_targets_dict.get(gn, []))
            for gn in guide_names
        ])
    else:
        is_ntc = np.zeros(len(guide_names), dtype=bool)

    assigned = guide_assignment > 0                               # [N_cells, G_guides]
    targeting_counts = assigned[:, ~is_ntc].sum(axis=1)          # [N_cells]

    N_cells = guide_assignment.shape[0]
    guide_labels = np.empty(N_cells, dtype=object)
    cell_mask = np.ones(N_cells, dtype=bool)

    # Cells with >1 targeting guide: exclude
    cell_mask[targeting_counts > 1] = False
    guide_labels[targeting_counts > 1] = '__excluded__'

    # Cells with exactly 1 targeting guide: label by that guide
    for i in np.where(targeting_counts == 1)[0]:
        j = int(np.where(assigned[i] & ~is_ntc)[0][0])
        guide_labels[i] = guide_names[j]

    # NTC-only cells
    for i in np.where(targeting_counts == 0)[0]:
        ntc_js = np.where(assigned[i] & is_ntc)[0]
        if len(ntc_js) == 1:
            guide_labels[i] = guide_names[int(ntc_js[0])]
        else:
            guide_labels[i] = 'multiple_NTC'

    return guide_labels.astype(str), cell_mask


def per_cell_mean_std(x):
    """
    Compute per-cell mean and std along axis 0 (samples x cells).

    Parameters
    ----------
    x : array-like, shape (n_samples, n_cells)
        Posterior samples

    Returns
    -------
    mean : np.ndarray, shape (n_cells,)
        Per-cell mean across samples
    std : np.ndarray, shape (n_cells,)
        Per-cell std across samples
    """
    x_np = to_np(x)
    return x_np.mean(axis=0), x_np.std(axis=0)
