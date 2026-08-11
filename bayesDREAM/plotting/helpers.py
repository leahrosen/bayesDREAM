"""
Helper utility functions for bayesDREAM plotting.
"""

import numpy as np

from ..utils import is_lean_posterior

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


def _guide_ntc_mask(guide_labels, model):
    """
    Boolean array of length ``len(guide_labels)``, True where the label is NTC.

    Works for both low-MOI (reads model.meta target column) and high-MOI
    (reads guide_targets_dict / guide_meta).
    """
    if not getattr(model, 'is_high_moi', False):
        if 'guide' in model.meta.columns and 'target' in model.meta.columns:
            gtmap = (model.meta.drop_duplicates('guide')
                     .set_index('guide')['target'].to_dict())
            ntc_guides = frozenset(g for g, t in gtmap.items()
                                   if str(t) in _NTC_VARIANTS)
        else:
            ntc_guides = _NTC_VARIANTS
    else:
        guide_names = model.guide_meta['guide'].values
        guide_targets_dict = getattr(model, 'guide_targets_dict', {})
        if 'target' in model.guide_meta.columns:
            is_ntc_arr = model.guide_meta['target'].isin(_NTC_VARIANTS).values
            ntc_guides = frozenset(guide_names[is_ntc_arr])
        elif guide_targets_dict:
            ntc_guides = frozenset(
                gn for gn in guide_names
                if any(t in _NTC_VARIANTS for t in guide_targets_dict.get(gn, []))
            )
        else:
            ntc_guides = frozenset()

    ntc_all = ntc_guides | frozenset({'multiple_NTC'})
    return np.array([g in ntc_all for g in guide_labels])


def _xtrue_posterior(model):
    """
    Return posterior samples of x_true as ``np.ndarray`` shape ``[S, N_cells]``,
    or ``None`` if not available.

    Reads ``model.posterior_samples_cis['x_true']``. If lean-loaded, this is
    the ``[1, N_cells]`` median singleton (see
    ``bayesDREAM.io.load._reduce_posterior_samples``) — callers that need a
    real distribution (histograms, KDE, std) rather than a point estimate +
    CI should use ``_xtrue_posterior_stats`` instead, or guard with
    ``bayesDREAM.utils.require_full_posterior``.
    """
    psc = getattr(model, 'posterior_samples_cis', None)
    if psc is None or 'x_true' not in psc:
        return None
    return to_np(psc['x_true'])


def _log2_safe(x):
    """log2 of x, returning NaN for non-positive values."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log2(np.maximum(x, 1e-300)), np.nan)


_Z_975 = 1.959963984540054  # standard normal 97.5th percentile


def _xtrue_posterior_stats(model, log2=False):
    """
    Return per-cell ``(point, std, lower, upper)`` arrays for
    ``model.posterior_samples_cis['x_true']``, shape ``[N_cells]`` each (no
    cell subsetting — callers index the result themselves). All four are
    ``None`` if no x_true posterior is available.

    Full (non-lean) posterior: computed directly from the raw ``[S, N_cells]``
    samples — ``point`` is the posterior mean, ``std`` the sample std,
    ``lower``/``upper`` the 2.5%/97.5% percentiles.

    Lean-loaded posterior (see ``bayesDREAM.io.load._reduce_posterior_samples``):
    only a point estimate + CI survive, so:
      - ``point`` is the stored per-cell posterior MEDIAN (not mean) — the
        median/mean substitution already used throughout lean-mode summary
        export (matches how alpha_x_prefit/alpha_y_prefit use the median as
        their point estimate at fit time).
      - ``lower``/``upper`` are the precomputed 2.5%/97.5% quantiles — exact,
        not approximated (quantiles commute with the monotonic log2
        transform, so log2(lower)/log2(upper) are still exact quantiles of
        log2(x_true)).
      - ``std`` is APPROXIMATED from the CI half-width assuming approximate
        normality: ``(upper - lower) / (2 * 1.96)``. This is a standard but
        inexact substitution — flag it in any docstring/label that surfaces it.

    Parameters
    ----------
    model : bayesDREAM
    log2 : bool, default False
        Transform to log2 space (non-positive values become NaN).
    """
    psc = getattr(model, 'posterior_samples_cis', None)
    if psc is None or 'x_true' not in psc:
        return None, None, None, None

    if is_lean_posterior(psc):
        lower = psc.get('x_true_lower')
        upper = psc.get('x_true_upper')
        if lower is None or upper is None:
            return None, None, None, None
        point = to_np(psc['x_true'])[0]  # [1, N] singleton -> [N]
        lower = to_np(lower)
        upper = to_np(upper)
        if log2:
            point = _log2_safe(point)
            lower = _log2_safe(lower)
            upper = _log2_safe(upper)
        std = (upper - lower) / (2 * _Z_975)
        return point, std, lower, upper

    post = to_np(psc['x_true'])  # [S, N]
    if log2:
        post = _log2_safe(post)
    point = np.nanmean(post, axis=0)
    std = np.nanstd(post, axis=0)
    lower = np.nanpercentile(post, 2.5, axis=0)
    upper = np.nanpercentile(post, 97.5, axis=0)
    return point, std, lower, upper


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
