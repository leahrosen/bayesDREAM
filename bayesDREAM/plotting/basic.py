"""
Basic plotting functions for x_true distributions.

Provides scatter, violin, and density plots colored by guide.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .helpers import to_np, resolve_guide_labels, _guide_ntc_mask, _xtrue_posterior
from .colors import ColorScheme


def _log2_safe(x):
    """log2 of x, returning NaN for non-positive values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log2(np.maximum(x, 1e-300)), np.nan)


def _guide_sort_key(g):
    """Sort guides: primary key = name before last underscore; secondary = trailing number."""
    parts = g.rsplit('_', 1)
    root = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (root, idx)


def scatter_by_guide(model, cis_gene=None, log2=False, log2fc=False,
                     color_scheme=None,
                     single_guide_cells_only=False, facet_ntc=False,
                     ax=None, show=True):
    """
    Scatter of per-cell x_true posterior mean vs std, one point per cell.

    Uses ``model.posterior_samples_cis['x_true']`` (shape ``[S, N_cells]``) when
    available so that x and y reflect genuine posterior uncertainty per cell.
    Falls back to the ``model.x_true`` point estimate (std = 0) if no posterior
    is stored.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    cis_gene : str, optional
        Cis gene name (title only; defaults to model.cis_gene).
    log2 : bool, default False
        Apply log2 transform before computing statistics.
    log2fc : bool, default False
        Express x-axis as log2 fold-change relative to the NTC mean.
        Implies ``log2=True``; subtracts the mean log2 value across all NTC
        cells so that NTC cells are centred at x = 0.
    color_scheme : ColorScheme, optional
        Custom color scheme.
    single_guide_cells_only : bool, default False
        Required to be ``True`` for high-MOI models.
    facet_ntc : bool, default False
        If ``True``, split into two side-by-side panels: NTC guides (left) and
        targeting guides (right).  Returns ``(fig, [ax_ntc, ax_targeting])``.
    ax : matplotlib axes, optional
        Used only when ``facet_ntc=False``.
    show : bool, default True

    Returns
    -------
    ax : matplotlib axes  (when ``facet_ntc=False``)
    (fig, [ax_ntc, ax_targeting]) : tuple  (when ``facet_ntc=True``)
    """
    if log2fc:
        log2 = True

    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)
    color_scheme = color_scheme.connect(model)
    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    guide_labels, cell_mask = resolve_guide_labels(model, single_guide_cells_only)

    post = _xtrue_posterior(model)
    if post is not None:
        post = post[:, cell_mask]                          # [S, N_kept]
        if log2:
            post = np.where(post > 0, np.log2(np.maximum(post, 1e-300)), np.nan)
        x_mean = np.nanmean(post, axis=0)                  # [N_kept]
        x_std  = np.nanstd(post,  axis=0)                  # [N_kept]
    else:
        x_vals = to_np(model.x_true)[cell_mask]
        if log2:
            x_vals = _log2_safe(x_vals)
        x_mean = x_vals
        x_std  = np.zeros_like(x_vals)

    guide_labels = guide_labels[cell_mask]

    if log2fc:
        ntc_mask = _guide_ntc_mask(guide_labels, model)
        ntc_mean = float(np.nanmean(x_mean[ntc_mask])) if ntc_mask.any() else 0.0
        x_mean = x_mean - ntc_mean
        xlabel = 'log2FC x_true (vs NTC)'
    else:
        xlabel = f'mean x_true{" (log2)" if log2 else ""}'

    suffix = ' (log2FC)' if log2fc else (' (log2)' if log2 else '')

    def _draw(ax_, gl, xm, xs, title_):
        for guide in sorted(np.unique(gl), key=_guide_sort_key):
            gmask = gl == guide
            xm_g = xm[gmask];  xs_g = xs[gmask]
            valid = ~(np.isnan(xm_g) | np.isnan(xs_g))
            if not valid.any():
                continue
            color = color_scheme.get_guide_color(guide, 'black')
            ax_.scatter(xm_g[valid], xs_g[valid], s=14, alpha=0.8,
                        color=color, label=guide)
        if log2fc:
            ax_.axvline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(f'std x_true{" (log2)" if log2 else ""}')
        ax_.set_title(title_)
        ax_.grid(True, linewidth=0.5, alpha=0.4)
        n_guides_ax = len(np.unique(gl))
        n_cols = max(1, n_guides_ax // 15 + 1)
        ax_.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False,
                   loc='upper left', bbox_to_anchor=(1.01, 1.0),
                   ncol=n_cols, borderaxespad=0.)

    if facet_ntc:
        ntc = _guide_ntc_mask(guide_labels, model)
        fig, (ax_ntc, ax_tgt) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        _draw(ax_ntc, guide_labels[ntc],  x_mean[ntc],  x_std[ntc],
              f'{cis_gene}: NTC{suffix}')
        _draw(ax_tgt, guide_labels[~ntc], x_mean[~ntc], x_std[~ntc],
              f'{cis_gene}: targeting{suffix}')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, [ax_ntc, ax_tgt]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    _draw(ax, guide_labels, x_mean, x_std,
          f'{cis_gene}: mean vs std of x_true{suffix}')
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def scatter_ci95_by_guide(model, cis_gene=None, log2=False, log2fc=False,
                          full_width=False,
                          color_scheme=None, single_guide_cells_only=False,
                          facet_ntc=False, ax=None, show=True):
    """
    Scatter of per-cell x_true posterior mean vs 95 % CI width, one point per cell.

    Uses ``model.posterior_samples_cis['x_true']`` when available so that the CI
    reflects genuine posterior uncertainty.  Falls back to ``model.x_true`` (CI = 0).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    cis_gene : str, optional
        Cis gene name (title only).
    log2 : bool, default False
        Apply log2 transform before computing statistics.
    log2fc : bool, default False
        Express x-axis as log2 fold-change relative to the NTC mean.
        Implies ``log2=True``.
    full_width : bool, default False
        If ``True``, show full (p97.5 − p2.5) CI; otherwise half-width.
    color_scheme : ColorScheme, optional
        Custom color scheme.
    single_guide_cells_only : bool, default False
        Required to be ``True`` for high-MOI models.
    facet_ntc : bool, default False
        If ``True``, split into NTC / targeting panels.
        Returns ``(fig, [ax_ntc, ax_targeting])``.
    ax : matplotlib axes, optional
        Used only when ``facet_ntc=False``.
    show : bool, default True

    Returns
    -------
    ax : matplotlib axes  (when ``facet_ntc=False``)
    (fig, [ax_ntc, ax_targeting]) : tuple  (when ``facet_ntc=True``)
    """
    if log2fc:
        log2 = True

    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)
    color_scheme = color_scheme.connect(model)
    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    guide_labels, cell_mask = resolve_guide_labels(model, single_guide_cells_only)

    post = _xtrue_posterior(model)
    if post is not None:
        post = post[:, cell_mask]                          # [S, N_kept]
        if log2:
            post = np.where(post > 0, np.log2(np.maximum(post, 1e-300)), np.nan)
        x_mean = np.nanmean(post, axis=0)
        q_lo   = np.nanpercentile(post, 2.5,  axis=0)
        q_hi   = np.nanpercentile(post, 97.5, axis=0)
    else:
        x_vals = to_np(model.x_true)[cell_mask]
        if log2:
            x_vals = _log2_safe(x_vals)
        x_mean = x_vals
        q_lo = q_hi = x_vals

    y_val = (q_hi - q_lo) if full_width else 0.5 * (q_hi - q_lo)
    guide_labels = guide_labels[cell_mask]

    if log2fc:
        ntc_mask = _guide_ntc_mask(guide_labels, model)
        ntc_mean = float(np.nanmean(x_mean[ntc_mask])) if ntc_mask.any() else 0.0
        x_mean = x_mean - ntc_mean
        xlabel = 'log2FC x_true (vs NTC)'
    else:
        xlabel = f'mean x_true{" (log2)" if log2 else ""}'

    suffix = ' (log2FC)' if log2fc else (' (log2)' if log2 else '')
    ylabel = '95% CI ' + ('width' if full_width else 'half-width') + f' x_true{" (log2)" if log2 else ""}'

    def _draw(ax_, gl, xm, yv, title_):
        for guide in sorted(np.unique(gl), key=_guide_sort_key):
            gmask = gl == guide
            xm_g = xm[gmask];  yv_g = yv[gmask]
            valid = ~(np.isnan(xm_g) | np.isnan(yv_g))
            if not valid.any():
                continue
            color = color_scheme.get_guide_color(guide, 'black')
            ax_.scatter(xm_g[valid], yv_g[valid], s=14, alpha=0.85,
                        color=color, label=guide)
        if log2fc:
            ax_.axvline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(ylabel)
        ax_.set_title(title_)
        ax_.grid(True, linewidth=0.5, alpha=0.4)
        n_guides_ax = len(np.unique(gl))
        n_cols = max(1, n_guides_ax // 15 + 1)
        ax_.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False,
                   loc='upper left', bbox_to_anchor=(1.01, 1.0),
                   ncol=n_cols, borderaxespad=0.)

    if facet_ntc:
        ntc = _guide_ntc_mask(guide_labels, model)
        fig, (ax_ntc, ax_tgt) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        _draw(ax_ntc, guide_labels[ntc],  x_mean[ntc],  y_val[ntc],
              f'{cis_gene}: NTC{suffix}')
        _draw(ax_tgt, guide_labels[~ntc], x_mean[~ntc], y_val[~ntc],
              f'{cis_gene}: targeting{suffix}')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, [ax_ntc, ax_tgt]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    _draw(ax, guide_labels, x_mean, y_val,
          f'{cis_gene}: mean vs 95% CI of x_true{suffix}')
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def violin_by_guide_log2(model, cis_gene=None, color_scheme=None,
                         single_guide_cells_only=False,
                         log2fc=False, sort_by_mean=False,
                         color_by='target', width_per_guide=0.7,
                         ax=None, show=True):
    """
    Violin plot of x_true (log2) grouped by guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    cis_gene : str, optional
        Cis gene name (title only).
    color_scheme : ColorScheme, optional
        Custom color scheme.  Defaults to ``model.color_scheme``.
    single_guide_cells_only : bool, default False
        Required to be ``True`` for high-MOI models.
    log2fc : bool, default False
        If ``True``, subtract the NTC guide mean so the y-axis shows log2 FC
        relative to NTC.
    sort_by_mean : bool, default False
        If ``True``, sort guides by their per-guide mean x_true (ascending).
        NTC guides are always placed first regardless.
    color_by : str, default 'target'
        How to color violins.  Options:

        * ``'target'`` – one color per target (default)
        * ``'guide'``  – one color per guide
        * any column name in ``model.meta`` – color by that metadata column
          (e.g. ``'cell_line'``, ``'batch'``)
    width_per_guide : float, default 0.7
        Figure width allocated per guide (inches).  Total width is
        ``max(6, n_guides * width_per_guide)``.
    ax : matplotlib axes, optional
    show : bool, default True

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)
    color_scheme = color_scheme.connect(model)

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    x_vals = to_np(model.x_true)                          # [N_cells]
    guide_labels, cell_mask = resolve_guide_labels(model, single_guide_cells_only)
    x_vals = x_vals[cell_mask]
    guide_labels = guide_labels[cell_mask]

    x_log = _log2_safe(x_vals)
    pos_mask = ~np.isnan(x_log)

    # --- log2FC: subtract NTC mean ---
    if log2fc:
        ntc_mask = _guide_ntc_mask(guide_labels, model)
        ntc_vals = x_log[ntc_mask & pos_mask]
        ntc_mean = float(np.nanmean(ntc_vals)) if len(ntc_vals) > 0 else 0.0
        x_log = x_log - ntc_mean
        ylabel = 'log₂FC (relative to NTC)'
    else:
        ylabel = 'x_true (log₂)'

    # --- guide ordering ---
    unique_guides = np.unique(guide_labels)
    ntc_guides = sorted([g for g in unique_guides if _guide_ntc_mask([g], model)[0]],
                        key=_guide_sort_key)
    tgt_guides = sorted([g for g in unique_guides if not _guide_ntc_mask([g], model)[0]],
                        key=_guide_sort_key)

    if sort_by_mean:
        tgt_guides = sorted(
            tgt_guides,
            key=lambda g: float(np.nanmean(x_log[(guide_labels == g) & pos_mask]))
                          if np.any((guide_labels == g) & pos_mask) else -np.inf
        )

    guide_order = ntc_guides + tgt_guides
    data = [x_log[(guide_labels == g) & pos_mask] for g in guide_order]

    # --- build color list ---
    # Determine color_by mode
    meta_col = None
    if color_by not in ('target', 'guide') and hasattr(model, 'meta'):
        if color_by in model.meta.columns:
            meta_col = color_by
        else:
            import warnings as _warn
            _warn.warn(f"color_by='{color_by}' not found in model.meta columns; "
                       f"falling back to 'target'")
            color_by = 'target'

    if meta_col is not None:
        # Build guide → metadata value map
        if 'guide' in model.meta.columns:
            gmap = model.meta.drop_duplicates('guide').set_index('guide')[meta_col].to_dict()
        else:
            gmap = {}
        # Build unique-value → color map (cycle through a tab20 palette)
        from matplotlib import cm as _cm
        unique_vals = sorted({str(gmap.get(g, '?')) for g in guide_order})
        tab20 = _cm.get_cmap('tab20', max(len(unique_vals), 1))
        val_color = {v: tab20(i / max(len(unique_vals), 1)) for i, v in enumerate(unique_vals)}
        colors = [val_color.get(str(gmap.get(g, '?')), 'gray') for g in guide_order]
        # Legend entries: one per unique value
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=val_color[v], alpha=0.85, label=v)
            for v in unique_vals
        ]
        legend_title = meta_col
    elif color_by == 'guide':
        colors = [color_scheme.get_guide_color(g, 'gray') for g in guide_order]
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=color_scheme.get_guide_color(g, 'gray'),
                          alpha=0.85, label=g)
            for g in guide_order
        ]
        legend_title = 'guide'
    else:  # 'target'
        colors = []
        for g in guide_order:
            target = color_scheme.guide_target(g)
            if target is None:
                target = g  # last resort
            colors.append(color_scheme.get_target_color(target, 'gray'))
        # Legend: unique target → color (preserve guide_order ordering)
        seen_tgt = {}
        for g, c in zip(guide_order, colors):
            tgt = color_scheme.guide_target(g) or g
            if tgt not in seen_tgt:
                seen_tgt[tgt] = c
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, label=t)
            for t, c in seen_tgt.items()
        ]
        legend_title = 'target'

    # --- figure size ---
    n_guides = len(guide_order)
    fig_w = max(6, n_guides * width_per_guide)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, 5))

    vp = ax.violinplot(data, positions=np.arange(1, n_guides + 1),
                       showmeans=True, showextrema=True, widths=0.7)
    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(0.85)

    for pc_key in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
        if pc_key in vp:
            vp[pc_key].set_edgecolor('black')
            vp[pc_key].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, n_guides + 1))
    ax.set_xticklabels(guide_order, rotation=90, ha='center', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    title_suffix = ' (log2FC)' if log2fc else ' (log₂)'
    ax.set_title(f'{cis_gene}: x_true distribution by guide{title_suffix}', fontsize=12)

    # Horizontal + vertical grid
    ax.grid(axis='y', linewidth=0.5, alpha=0.3)
    ax.grid(axis='x', linewidth=0.5, alpha=0.2)

    # Legend outside (multiple columns so it doesn't exceed plot height)
    n_cols = max(1, len(legend_handles) // 15 + 1)
    ax.legend(handles=legend_handles, title=legend_title, fontsize=8, frameon=False,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              ncol=n_cols, borderaxespad=0.)

    plt.tight_layout()

    if show:
        plt.show()

    return ax


def filled_density_by_guide_log2(model, cis_gene=None, bw=None, log2fc=False,
                                 color_scheme=None,
                                 single_guide_cells_only=False, facet_ntc=False,
                                 ax=None, show=True):
    """
    Filled KDE density plot of x_true (log2), one curve per guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    cis_gene : str, optional
        Cis gene name (title only).
    bw : float, optional
        KDE bandwidth.  If ``None``, uses Scott's rule.
    log2fc : bool, default False
        Express x-axis as log2 fold-change relative to the NTC mean.
        Subtracts the mean log2 value across all NTC cells so densities
        are shown centred on 0 for NTC.
    color_scheme : ColorScheme, optional
        Custom color scheme.
    single_guide_cells_only : bool, default False
        Required to be ``True`` for high-MOI models.
    facet_ntc : bool, default False
        If ``True``, split into NTC / targeting panels.
        Returns ``(fig, [ax_ntc, ax_targeting])``.
    ax : matplotlib axes, optional
        Used only when ``facet_ntc=False``.
    show : bool, default True

    Returns
    -------
    ax : matplotlib axes  (when ``facet_ntc=False``)
    (fig, [ax_ntc, ax_targeting]) : tuple  (when ``facet_ntc=True``)
    """
    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)
    color_scheme = color_scheme.connect(model)

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    x_vals = to_np(model.x_true)                          # [N_cells]
    guide_labels, cell_mask = resolve_guide_labels(model, single_guide_cells_only)
    x_vals = x_vals[cell_mask]
    guide_labels = guide_labels[cell_mask]

    x_log = _log2_safe(x_vals)
    valid = ~np.isnan(x_log)

    if log2fc:
        ntc_mask = _guide_ntc_mask(guide_labels, model)
        ntc_mean = float(np.nanmean(x_log[ntc_mask & valid])) if (ntc_mask & valid).any() else 0.0
        x_log = x_log - ntc_mean
        xlabel = 'log2FC x_true (vs NTC)'
        title_base = f'{cis_gene}: x_true density (log2FC)'
    else:
        xlabel = 'x_true (log₂)'
        title_base = f'{cis_gene}: x_true density by guide'

    xmin, xmax = np.nanpercentile(x_log[valid], [0.5, 99.5])
    x_grid = np.linspace(xmin, xmax, 500)

    def _draw(ax_, gl, xl, vl, title_):
        guide_order = sorted(np.unique(gl), key=_guide_sort_key)
        for g in guide_order:
            sub = xl[(gl == g) & vl]
            if len(sub) < 2:
                continue
            kde = gaussian_kde(sub, bw_method=bw)
            density = kde(x_grid)
            color = color_scheme.get_guide_color(g, 'black')
            ax_.fill_between(x_grid, density, alpha=0.4, color=color, label=g)
            ax_.plot(x_grid, density, color=color, linewidth=1.5)
        if log2fc:
            ax_.axvline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_.set_xlabel(xlabel, fontsize=11)
        ax_.set_ylabel('Density', fontsize=11)
        ax_.set_title(title_, fontsize=12)
        n_guides_ax = len(np.unique(gl))
        n_cols = max(1, n_guides_ax // 15 + 1)
        ax_.legend(title='guide', fontsize=8, frameon=False,
                   loc='upper left', bbox_to_anchor=(1.01, 1.0),
                   ncol=n_cols, borderaxespad=0.)
        ax_.grid(True, linewidth=0.5, alpha=0.3)

    if facet_ntc:
        ntc = _guide_ntc_mask(guide_labels, model)
        fig, (ax_ntc, ax_tgt) = plt.subplots(1, 2, figsize=(16, 5))
        _draw(ax_ntc, guide_labels[ntc],  x_log[ntc],  valid[ntc],  f'{cis_gene}: NTC')
        _draw(ax_tgt, guide_labels[~ntc], x_log[~ntc], valid[~ntc], f'{cis_gene}: targeting')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, [ax_ntc, ax_tgt]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    _draw(ax, guide_labels, x_log, valid, title_base)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def scatter_param_mean_vs_ci(
    param_samps,
    param_name='parameter',
    subset_mask=None,
    color_by=None,
    color_label='color metric',
    cmap='Blues_r',
    vmin=None,
    vmax=None,
    log2=False,
    ax=None,
    show=True,
    title=None,
    figsize=(7, 5),
):
    """
    Scatter plot of parameter mean vs 95% CI width, with optional color coding.

    This is useful for visualizing parameter uncertainty vs magnitude, with
    optional coloring by dependency masks, NaN fractions, or other metrics.

    Parameters
    ----------
    param_samps : np.ndarray
        Parameter samples, shape (n_samples, n_features)
    param_name : str
        Parameter name for axis labels (default: 'parameter')
    subset_mask : np.ndarray, optional
        Boolean mask to subset features. If provided, plots two groups:
        masked (colored) and unmasked (grey).
    color_by : np.ndarray, optional
        Values to color points by (length n_features). Requires subset_mask.
        Common uses:
        - NaN fraction: color by how many samples are NaN
        - Dependency metric: color by strength of effect
    color_label : str
        Label for colorbar (default: 'color metric')
    cmap : str
        Colormap name (default: 'Blues_r' for darker = fewer NaNs)
    vmin, vmax : float, optional
        Color scale limits. If None, uses data range.
    log2 : bool
        Whether param_samps are on log2 scale (affects axis label only)
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool
        Whether to display the plot (default: True)
    title : str, optional
        Plot title. If None, auto-generates from param_name.
    figsize : tuple
        Figure size (default: (7, 5))

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    >>> # Example 1: Inflection point with NaN fraction coloring
    >>> xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> frac_nan = np.mean(np.isnan(xinf_samps), axis=0)
    >>> fig = scatter_param_mean_vs_ci(
    ...     xinf_samps,
    ...     param_name='x_infl',
    ...     subset_mask=dep_mask,
    ...     color_by=frac_nan,
    ...     color_label='fraction NaN (lighter = more NaN)',
    ...     cmap='Blues_r',
    ...     log2=True
    ... )

    >>> # Example 2: Hill coefficient n with dependency coloring
    >>> n_samps = model.posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> fig = scatter_param_mean_vs_ci(
    ...     n_samps,
    ...     param_name='n (Hill coefficient)',
    ...     subset_mask=dep_mask,
    ...     log2=False
    ... )
    """
    param_samps = np.asarray(param_samps)

    # Compute mean and CI width
    param_mean = np.nanmean(param_samps, axis=0)
    param_lo = np.nanpercentile(param_samps, 2.5, axis=0)
    param_hi = np.nanpercentile(param_samps, 97.5, axis=0)
    param_ci_width = param_hi - param_lo

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Case 1: No subsetting - plot all points in one color
    if subset_mask is None:
        ax.scatter(param_mean, param_ci_width, s=8, alpha=0.6, color='blue')

    # Case 2: Subsetting without color coding
    elif color_by is None:
        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey', label='not selected'
            )

        # Plot masked points in blue
        ax.scatter(
            param_mean[subset_mask],
            param_ci_width[subset_mask],
            s=5, alpha=0.2, color='blue', label='selected'
        )
        ax.legend(frameon=False, loc='best')

    # Case 3: Subsetting with color coding
    else:
        color_by = np.asarray(color_by)
        if len(color_by) != len(subset_mask):
            raise ValueError(f"color_by length ({len(color_by)}) must match subset_mask length ({len(subset_mask)})")

        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey'
            )

        # Plot masked points with color coding
        valid_mask = subset_mask & np.isfinite(param_mean) & np.isfinite(param_ci_width)

        if vmin is None:
            vmin = np.nanmin(color_by[valid_mask])
        if vmax is None:
            vmax = np.nanmax(color_by[valid_mask])

        sc = ax.scatter(
            param_mean[valid_mask],
            param_ci_width[valid_mask],
            c=color_by[valid_mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=8,
            alpha=0.9
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_label)

    # Labels and formatting
    xlabel = f'Mean {param_name}' + (' (log₂)' if log2 else '')
    ylabel = f'95% CI width of {param_name}' + (' (log₂)' if log2 else '')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f'{param_name}: mean vs uncertainty'
    ax.set_title(title)

    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def _sample_prior_for_param(param, gene_idx, prior_params, n_samples=400, rng=None):
    """
    Generate samples from the analytic prior distribution for one parameter × gene.

    Returns a 1-D numpy array of length n_samples, or None if the parameter's
    prior is not stored in prior_params.

    Prior distributions
    -------------------
    n_a, n_b  : marginal over sigma_n ~ Exp(rate) of Normal(n_mu_raw, sigma_n),
                then soft-clamped to [nmin, nmax]
    Vmax_a/b  : LogNormal(Vmax_log_mu[gene], Vmax_log_sigma)
    K_a/b     : LogNormal(K_log_mu, K_log_sigma)
    A         : Exponential(rate=1/Amean[gene])
    alpha/beta: RelaxedBernoulli(logits=p_n_logits, temperature=1)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if param in ('n_a', 'n_b'):
        nmin      = prior_params.get('nmin')
        nmax      = prior_params.get('nmax')
        n_mu_raw  = prior_params.get('n_mu_raw', 0.0)
        rate      = prior_params.get('sigma_n_prior_rate', 0.5)
        if nmin is None or nmax is None:
            return None
        sigma_n = rng.exponential(1.0 / rate, n_samples)       # marginalise sigma_n
        n_raw   = rng.normal(n_mu_raw, sigma_n)
        half    = 0.5 * (nmax - nmin)
        center  = 0.5 * (nmax + nmin)
        return center + half * np.tanh(n_raw / half)            # soft_clamp

    elif param in ('Vmax_a', 'Vmax_b'):
        Vmax_log_mu    = prior_params.get('Vmax_log_mu')
        Vmax_log_sigma = prior_params.get('Vmax_log_sigma')
        if Vmax_log_mu is None or Vmax_log_sigma is None:
            return None
        mu = (float(Vmax_log_mu[gene_idx])
              if hasattr(Vmax_log_mu, '__len__') else float(Vmax_log_mu))
        return np.exp(rng.normal(mu, float(Vmax_log_sigma), n_samples))

    elif param in ('K_a', 'K_b'):
        K_log_mu    = prior_params.get('K_log_mu')
        K_log_sigma = prior_params.get('K_log_sigma')
        if K_log_mu is None or K_log_sigma is None:
            return None
        mu = (float(K_log_mu[gene_idx])
              if hasattr(K_log_mu, '__len__') else float(K_log_mu))
        return np.exp(rng.normal(mu, float(K_log_sigma), n_samples))

    elif param == 'A':
        Amean = prior_params.get('Amean')
        if Amean is None:
            return None
        amean_val = (float(Amean[gene_idx])
                     if hasattr(Amean, '__len__') else float(Amean))
        return rng.exponential(amean_val, n_samples)   # Exp(rate=1/amean)

    elif param in ('alpha', 'beta'):
        logits = prior_params.get('p_n_logits', -13.8)
        temp   = prior_params.get('temperature_prior', 1.0)
        # RelaxedBernoulli via logistic-noise reparameterisation
        u        = rng.uniform(1e-8, 1 - 1e-8, n_samples)
        log_odds = np.log(u / (1.0 - u))
        return 1.0 / (1.0 + np.exp(-(logits + log_odds) / temp))

    elif param in ('alpha_y', 'log2_alpha_y', 'alpha_y_mult', 'alpha_y_add'):
        # Prior on log2(alpha_y) for non-reference technical groups: StudentT(df=3, loc=0, scale=20)
        # alpha_y_mult and alpha_y are displayed in log2 space; alpha_y_add is already additive
        return rng.standard_t(3, n_samples) * 20.0

    return None


# Parameters that come from posterior_samples_technical rather than posterior_samples_trans.
# NOTE: o_y is sampled in BOTH technical and trans posteriors.
#   'o_y'      → posterior_samples_trans  (trans fit overdispersion, the usual plotting target)
#   'o_y_tech' → posterior_samples_technical  (NTC-only technical fit overdispersion)
_TECHNICAL_PARAMS = frozenset({'alpha_y', 'alpha_y_mult', 'alpha_y_add', 'log2_alpha_y', 'mu_ntc', 'o_y_tech'})


def _get_technical_param_samples(param, tech_posterior, technical_group):
    """
    Extract samples for a technical parameter from posterior_samples_technical.

    Handles the C (technical group) dimension and log2 conversion for multiplicative params.

    Parameters
    ----------
    param : str
        Parameter name ('alpha_y', 'log2_alpha_y', 'alpha_y_mult', 'alpha_y_add', 'mu_ntc', 'o_y_tech')
    tech_posterior : dict
        posterior_samples_technical dict
    technical_group : int
        Which technical group index to display (1 = first non-reference group).
        Index 0 is always the reference group (alpha=1 for mult, 0 for add).

    Returns
    -------
    np.ndarray of shape [S, T]
    """
    if param == 'alpha_y':
        # Try multiplicative first (negbinom), then additive
        if 'alpha_y_mult' in tech_posterior:
            raw = to_np(tech_posterior['alpha_y_mult'])
            if raw.ndim == 3:
                raw = raw[:, technical_group, :]
            return np.log2(np.maximum(raw, 1e-10))
        elif 'alpha_y_add' in tech_posterior:
            raw = to_np(tech_posterior['alpha_y_add'])
            if raw.ndim == 3:
                raw = raw[:, technical_group, :]
            return raw
        elif 'alpha_y' in tech_posterior:
            raw = to_np(tech_posterior['alpha_y'])
            if raw.ndim == 3:
                raw = raw[:, technical_group, :]
            return raw
        else:
            raise KeyError(
                f"'alpha_y' not found in posterior_samples_technical. "
                f"Available: {list(tech_posterior.keys())}"
            )

    elif param == 'alpha_y_mult':
        raw = to_np(tech_posterior['alpha_y_mult'])
        if raw.ndim == 3:
            raw = raw[:, technical_group, :]
        return np.log2(np.maximum(raw, 1e-10))

    elif param == 'alpha_y_add':
        raw = to_np(tech_posterior['alpha_y_add'])
        if raw.ndim == 3:
            raw = raw[:, technical_group, :]
        return raw

    elif param == 'log2_alpha_y':
        # Directly sampled in log2 space, shape [S, C-1, T] (no reference group)
        raw = to_np(tech_posterior['log2_alpha_y'])
        if raw.ndim == 3:
            g_idx = technical_group - 1  # 1-based group → 0-based index into C-1 groups
            if g_idx < 0 or g_idx >= raw.shape[1]:
                raise ValueError(
                    f"technical_group={technical_group} out of range for log2_alpha_y with "
                    f"{raw.shape[1]} non-reference group(s). "
                    f"Use technical_group between 1 and {raw.shape[1]}."
                )
            raw = raw[:, g_idx, :]
        return raw

    elif param == 'mu_ntc':
        # mu_ntc has no C dimension ([S, T]), but handle 3D defensively
        raw = to_np(tech_posterior['mu_ntc'])
        if raw.ndim == 3:
            raw = raw[:, technical_group, :]
        return raw

    elif param == 'o_y_tech':
        # o_y from the technical fit (NTC-only). Use 'o_y' for the trans fit version.
        # Shape [S, T] — not group-specific. If stored as [S, 1, T], use index 0.
        raw = to_np(tech_posterior['o_y'])
        if raw.ndim == 3:
            raw = raw[:, 0, :]
        return raw

    else:
        # Generic fallback for any other key in technical posterior
        raw = to_np(tech_posterior[param])
        if raw.ndim == 3:
            raw = raw[:, technical_group, :]
        return raw


def plot_parameter_ci_panel(
    model,
    params: list,
    modality_name: str = None,
    genes: list = None,
    ci_level: float = 95.0,
    sort_by: str = 'none',
    filter_dependent: bool = False,
    dependency_params: list = None,
    max_genes: int = 100,
    ymin: float = None,
    ymax: float = None,
    title: str = None,
    ylabel: str = 'value',
    figsize: tuple = None,
    color_palette: dict = None,
    marker_size: int = 18,
    capsize: int = 3,
    show_zero_line: bool = True,
    show_gene_separators: bool = True,
    ax=None,
    show: bool = True,
    fdr_df=None,
    fdr_threshold: float = 0.05,
    hide_inactive: bool = False,
    show_prior: bool = False,
    technical_group: int = 1,
):
    """
    Forest plot (dot + whisker CI) for posterior parameters across trans genes.

    Creates a plot with genes on the x-axis and parameter values (median + CI) on
    the y-axis. Multiple parameters are dodged side-by-side for comparison.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model with posterior_samples_trans
    params : list of str
        Parameter names to plot (e.g., ['n_a', 'n_b'] or ['alpha', 'beta']).
        These must exist in posterior_samples_trans.
    modality_name : str, optional
        Modality name. If None, uses primary modality.
    genes : list of str, optional
        Specific genes to plot. If None, plots all genes (subject to max_genes).
        Gene names must match feature names in the modality.
    ci_level : float
        Credible interval level (default: 95.0 for 95% CI)
    sort_by : str
        How to sort genes on x-axis:
        - 'none': Keep original order
        - 'alphabetical': Sort alphabetically by gene name
        - 'median': Sort by median of first parameter (ascending)
        - 'abs_median': Sort by absolute median of first parameter (descending)
        - 'effect': Sort by max absolute effect across all params (descending)
    filter_dependent : bool
        If True, only show genes where CI excludes 0 for any param in
        dependency_params (default: False)
    dependency_params : list, optional
        Parameters to use for dependency filtering. If None, uses all params.
        Common: ['n_a', 'n_b'] for Hill coefficients.
    max_genes : int
        Maximum number of genes to plot (default: 100). If more genes would be
        plotted, raises ValueError with suggestions. Set to None to disable limit.
    ymin, ymax : float, optional
        Y-axis limits. If None, auto-scaled.
    title : str, optional
        Plot title. If None, auto-generated.
    ylabel : str
        Y-axis label (default: 'value')
    figsize : tuple, optional
        Figure size. If None, auto-scaled based on number of genes.
    color_palette : dict, optional
        Custom colors for parameters. Keys are param names, values are colors.
        If None, uses seaborn color palette.
    marker_size : int
        Size of median markers (default: 18)
    capsize : int
        Size of error bar caps (default: 3)
    show_zero_line : bool
        Whether to draw horizontal line at y=0 (default: True)
    show_gene_separators : bool
        Whether to draw vertical lines between genes (default: True).
        Helps visually distinguish which parameters belong to which gene.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool
        Whether to display the plot (default: True)
    fdr_df : pd.DataFrame, optional
        trans_summary DataFrame (output of ``save_trans_summary()``).  The
        DataFrame must contain a gene name column (``gene_name`` or ``gene``)
        and the FDR columns ``fdr_alpha`` and ``fdr_beta``.  When provided,
        parameters belonging to FDR-inactive components (fdr_alpha or fdr_beta
        >= fdr_threshold) are either rendered in light grey (default) or
        omitted entirely (when ``hide_inactive=True``).
        Component mapping: alpha/n_a/K_a/Vmax_a → fdr_alpha;
        beta/n_b/K_b/Vmax_b → fdr_beta.
    fdr_threshold : float
        FDR threshold for inactivity (default: 0.05). Used with fdr_df.
    hide_inactive : bool
        If True and fdr_df is provided, FDR-inactive parameters are completely
        hidden (not plotted at all) rather than shown in grey (default: False).
        Useful to avoid visual clutter from wandering posteriors of "off"
        components.
    show_prior : bool
        If True, underlay each posterior CI with a light-grey violin drawn from
        the analytic prior distribution (default: False).  Requires
        ``model.trans_prior_params`` to be set (automatically set by
        ``fit_trans()``).  Useful for assessing how much the posterior has moved
        away from the prior.
    technical_group : int
        Which technical group to display for technical parameters (``alpha_y``,
        ``log2_alpha_y``, ``mu_ntc``, ``o_y``). Index 0 is the reference group
        (always 0 in log2 space), so typically use 1 for the first non-reference
        group (default: 1). For ``log2_alpha_y`` specifically, this is 1-based
        into the C-1 non-reference groups.

    Returns
    -------
    fig : matplotlib Figure (if ax was None)
    ax : matplotlib Axes

    Examples
    --------
    >>> # Plot n_a and n_b for all genes
    >>> fig, ax = model.plot_parameter_ci_panel(['n_a', 'n_b'])

    >>> # Plot only dependent genes, sorted by effect size
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['n_a', 'n_b'],
    ...     filter_dependent=True,
    ...     sort_by='effect'
    ... )

    >>> # Plot alpha and beta with custom colors
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['alpha', 'beta'],
    ...     color_palette={'alpha': 'crimson', 'beta': 'dodgerblue'}
    ... )

    >>> # Plot for a specific modality
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['n_a', 'n_b'],
    ...     modality_name='splicing_sj'
    ... )
    """
    import seaborn as sns

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Separate trans params from technical params (alpha_y, mu_ntc, o_y, etc.)
    trans_params = [p for p in params if p not in _TECHNICAL_PARAMS]
    tech_params  = [p for p in params if p in _TECHNICAL_PARAMS]

    # Get trans posterior (only required when trans params are requested)
    if trans_params:
        if modality_name == model.primary_modality:
            posterior = model.posterior_samples_trans
        else:
            posterior = modality.posterior_samples_trans

        if posterior is None:
            raise ValueError(
                f"No posterior_samples_trans found for modality '{modality_name}'. "
                "Must run fit_trans() first."
            )

        missing = [p for p in trans_params if p not in posterior]
        if missing:
            available = list(posterior.keys())
            raise ValueError(
                f"Parameters {missing} not found in posterior_samples_trans. "
                f"Available: {available}"
            )
    else:
        posterior = None

    # Get technical posterior (required when technical params are requested)
    tech_posterior = None
    if tech_params:
        tech_posterior = modality.posterior_samples_technical
        if tech_posterior is None:
            raise ValueError(
                f"Parameters {tech_params} require posterior_samples_technical. "
                "Must run fit_technical() first."
            )
        # Validate that needed keys are present
        missing_tech = []
        for p in tech_params:
            if p == 'alpha_y':
                if not any(k in tech_posterior for k in ('alpha_y_mult', 'alpha_y_add', 'alpha_y')):
                    missing_tech.append(p)
            elif p == 'o_y_tech':
                if 'o_y' not in tech_posterior:
                    missing_tech.append(p)
            elif p not in tech_posterior:
                missing_tech.append(p)
        if missing_tech:
            raise ValueError(
                f"Technical parameters {missing_tech} not found in posterior_samples_technical. "
                f"Available: {list(tech_posterior.keys())}"
            )

    # Get gene names from modality
    gene_names = modality.feature_names
    if gene_names is None:
        # Fallback to feature_meta
        for col in ['gene_name', 'gene', 'feature_id', 'feature']:
            if col in modality.feature_meta.columns:
                gene_names = modality.feature_meta[col].tolist()
                break
    if gene_names is None:
        gene_names = [str(i) for i in range(modality.dims['n_features'])]

    # Extract samples for each parameter
    # Trans params: posterior[param] is typically (S, n_cis, T) where n_cis=1
    # Technical params: posterior_samples_technical[param] may be (S, C, T) or (S, T)
    samples_dict = {}
    for param in params:
        if param in _TECHNICAL_PARAMS:
            samps = _get_technical_param_samples(param, tech_posterior, technical_group)
        else:
            samps = to_np(posterior[param])
            if samps.ndim == 3:
                samps = samps[:, 0, :]  # (S, 1, T) -> (S, T)
        if samps.ndim == 1:
            samps = samps.reshape(-1, 1)  # (S,) -> (S, 1)
        samples_dict[param] = samps

    T = samples_dict[params[0]].shape[1]

    # Ensure gene_names matches T
    if len(gene_names) != T:
        gene_names = gene_names[:T]

    # Build per-param FDR inactive masks from fdr_df (if provided)
    _PARAM_TO_FDR_COL = {
        'alpha': 'fdr_alpha', 'n_a': 'fdr_alpha', 'K_a': 'fdr_alpha', 'Vmax_a': 'fdr_alpha',
        'beta':  'fdr_beta',  'n_b': 'fdr_beta',  'K_b': 'fdr_beta',  'Vmax_b': 'fdr_beta',
    }
    inactive_masks = {}  # {param: bool array of length T}
    if fdr_df is not None:
        _gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        _name_col = next((c for c in ['gene_name', 'gene', 'feature'] if c in fdr_df.columns), None)
        for param in params:
            fdr_col = _PARAM_TO_FDR_COL.get(param)
            if fdr_col and fdr_col in fdr_df.columns and _name_col:
                mask = np.zeros(T, dtype=bool)
                for _, row in fdr_df.iterrows():
                    gname = row[_name_col]
                    if gname in _gene_to_idx and np.isfinite(row[fdr_col]):
                        mask[_gene_to_idx[gname]] = row[fdr_col] >= fdr_threshold
                inactive_masks[param] = mask

    # Compute CI bounds
    lo_q = (100 - ci_level) / 2.0
    hi_q = 100 - lo_q

    stats = {}  # {param: {'median': array, 'lo': array, 'hi': array}}
    for param, samps in samples_dict.items():
        stats[param] = {
            'median': np.nanmedian(samps, axis=0),
            'lo': np.nanpercentile(samps, lo_q, axis=0),
            'hi': np.nanpercentile(samps, hi_q, axis=0),
        }

    # Filter to dependent genes if requested
    gene_mask = np.ones(T, dtype=bool)
    if filter_dependent:
        dep_params = dependency_params if dependency_params else params
        # Start with all False, then OR with each param's dependency
        gene_mask = np.zeros(T, dtype=bool)
        for param in dep_params:
            if param in samples_dict:
                samps = samples_dict[param]
                lo = np.nanpercentile(samps, lo_q, axis=0)
                hi = np.nanpercentile(samps, hi_q, axis=0)
                param_dep = (lo > 0) | (hi < 0)
                gene_mask = gene_mask | param_dep

        n_dep = gene_mask.sum()
        print(f"[FILTER] {n_dep}/{T} genes pass dependency filter (CI excludes 0)")

    # Get indices of genes to plot
    gene_indices = np.where(gene_mask)[0]

    # Filter to user-specified genes if provided
    if genes is not None:
        # Map gene names to indices
        name_to_idx = {name: i for i, name in enumerate(gene_names)}
        user_indices = []
        missing_genes = []
        for g in genes:
            if g in name_to_idx:
                idx = name_to_idx[g]
                if idx in gene_indices:  # Respect dependency filter
                    user_indices.append(idx)
            else:
                missing_genes.append(g)
        if missing_genes:
            import warnings
            warnings.warn(f"Genes not found in modality: {missing_genes[:5]}{'...' if len(missing_genes) > 5 else ''}")
        gene_indices = np.array(user_indices)

    n_genes = len(gene_indices)

    if n_genes == 0:
        print("No genes to plot after filtering.")
        return None, None

    # Check max_genes limit
    if max_genes is not None and n_genes > max_genes:
        raise ValueError(
            f"Too many genes to plot ({n_genes} > {max_genes}). Options:\n"
            f"  1. Use filter_dependent=True to show only dependent genes\n"
            f"  2. Use genes=['gene1', 'gene2', ...] to specify specific genes\n"
            f"  3. Use sort_by='effect' with filter_dependent=True for top effects\n"
            f"  4. Set max_genes=None to disable this limit (not recommended)"
        )
    elif n_genes > 100:
        import warnings
        warnings.warn(
            f"Plotting {n_genes} genes. Consider using filter_dependent=True "
            f"or genes=[...] to reduce the number of genes."
        )

    # Sort genes
    if sort_by == 'alphabetical':
        # Get gene names for current indices, then sort alphabetically
        names_for_sort = [gene_names[i] for i in gene_indices]
        order = np.argsort(names_for_sort)
        gene_indices = gene_indices[order]
    elif sort_by == 'median':
        sort_vals = stats[params[0]]['median'][gene_indices]
        order = np.argsort(sort_vals)
        gene_indices = gene_indices[order]
    elif sort_by == 'abs_median':
        sort_vals = np.abs(stats[params[0]]['median'][gene_indices])
        order = np.argsort(sort_vals)[::-1]  # Descending
        gene_indices = gene_indices[order]
    elif sort_by == 'effect':
        # Max absolute median across all params
        max_effect = np.zeros(n_genes)
        for param in params:
            max_effect = np.maximum(max_effect, np.abs(stats[param]['median'][gene_indices]))
        order = np.argsort(max_effect)[::-1]  # Descending
        gene_indices = gene_indices[order]

    # Get sorted gene names
    sorted_gene_names = [gene_names[i] for i in gene_indices]

    # Create figure
    if ax is None:
        if figsize is None:
            fig_w = min(max(0.5 * n_genes, 12), 28)
            fig_h = 5.5
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set up colors
    if color_palette is None:
        colors = sns.color_palette(n_colors=len(params))
        color_palette = dict(zip(params, colors))

    # Set up x positions with dodging
    # Use smaller width to keep params for same gene close together
    x_base = np.arange(n_genes)
    n_params = len(params)
    width = 0.3  # Reduced from 0.7 to keep params closer within gene
    if n_params > 1:
        offsets = np.linspace(-width/2, width/2, n_params)
    else:
        offsets = np.array([0.0])

    # --- Prior violin underlay ---
    if show_prior:
        prior_params = None
        if hasattr(modality, 'trans_prior_params') and modality.trans_prior_params is not None:
            prior_params = modality.trans_prior_params
        elif (modality_name == model.primary_modality
              and hasattr(model, 'trans_prior_params')
              and model.trans_prior_params is not None):
            prior_params = model.trans_prior_params

        # Technical params (alpha_y, log2_alpha_y, etc.) have analytic priors that don't
        # need trans_prior_params. Use an empty dict so _sample_prior_for_param can handle them.
        if prior_params is None and all(p in _TECHNICAL_PARAMS for p in params):
            prior_params = {}

        if prior_params is not None:
            _rng = np.random.default_rng(0)
            for j, param in enumerate(params):
                violin_data = []
                violin_positions = []
                for k, gene_idx in enumerate(gene_indices):
                    samps_prior = _sample_prior_for_param(
                        param, int(gene_idx), prior_params, n_samples=400, rng=_rng)
                    if samps_prior is not None and np.isfinite(samps_prior).any():
                        violin_data.append(samps_prior[np.isfinite(samps_prior)])
                        violin_positions.append(x_base[k] + offsets[j])

                if violin_data:
                    parts = ax.violinplot(
                        violin_data, positions=violin_positions,
                        widths=0.18, showmeans=False, showmedians=False,
                        showextrema=False)
                    for body in parts['bodies']:
                        body.set_facecolor('lightgray')
                        body.set_edgecolor('gray')
                        body.set_alpha(0.35)
                        body.set_zorder(0)
        else:
            import warnings
            warnings.warn(
                "show_prior=True but no trans_prior_params found on modality or model. "
                "Run fit_trans() first."
            )

    # Plot each parameter
    for j, param in enumerate(params):
        medians = stats[param]['median'][gene_indices]
        los = stats[param]['lo'][gene_indices]
        his = stats[param]['hi'][gene_indices]

        x = x_base + offsets[j]
        color = color_palette.get(param, 'blue')

        # Determine FDR inactive subset (greyed out)
        raw_inactive = inactive_masks.get(param)
        inactive_plot = raw_inactive[gene_indices] if raw_inactive is not None else np.zeros(len(gene_indices), dtype=bool)
        active_plot = ~inactive_plot

        # Plot active genes with full color
        if active_plot.any():
            ax.scatter(x[active_plot], medians[active_plot], label=param,
                       s=marker_size, zorder=3, color=color)
            yerr_act = np.vstack([medians[active_plot] - los[active_plot],
                                   his[active_plot] - medians[active_plot]])
            ax.errorbar(x[active_plot], medians[active_plot], yerr=yerr_act,
                        fmt='none', elinewidth=1.5, capsize=capsize, color=color, zorder=2)
        else:
            # No active genes — add phantom entry for legend
            ax.scatter([], [], label=param, s=marker_size, color=color)

        # Plot inactive genes: grey (default) or hidden (hide_inactive=True)
        if inactive_plot.any() and not hide_inactive:
            ax.scatter(x[inactive_plot], medians[inactive_plot],
                       s=marker_size, zorder=3, color='lightgray', alpha=0.5)
            yerr_inact = np.vstack([medians[inactive_plot] - los[inactive_plot],
                                     his[inactive_plot] - medians[inactive_plot]])
            ax.errorbar(x[inactive_plot], medians[inactive_plot], yerr=yerr_inact,
                        fmt='none', elinewidth=1.5, capsize=capsize,
                        color='lightgray', alpha=0.5, zorder=2)

    # Styling
    if title is None:
        param_str = ', '.join(params)
        title = f"{model.cis_gene} → trans genes: {param_str}"
        if filter_dependent:
            title += f" (n={n_genes} dependent)"

    ax.set_title(title)
    ax.set_xlabel("Trans gene")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(sorted_gene_names, rotation=90, ha="center")

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(False)

    if show_zero_line:
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Draw vertical separators between genes
    if show_gene_separators and n_genes > 1:
        for i in range(1, n_genes):
            ax.axvline(i - 0.5, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)

    # Tighten x-axis margins
    ax.set_xlim(-0.5, n_genes - 0.5)

    if ymin is not None or ymax is not None:
        cur = ax.get_ylim()
        ax.set_ylim(
            ymin if ymin is not None else cur[0],
            ymax if ymax is not None else cur[1]
        )

    ax.legend(title='parameter', bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def extract_posterior_dataframe(
    model,
    params: list,
    modality_name: str = None,
    include_samples: bool = False,
):
    """
    Extract posterior parameters into a long-format DataFrame.

    This is useful for custom analysis or plotting with seaborn/plotnine.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    params : list of str
        Parameter names to extract (e.g., ['n_a', 'n_b', 'K_a', 'K_b'])
    modality_name : str, optional
        Modality name. If None, uses primary modality.
    include_samples : bool
        If True, includes all posterior samples (can be large).
        If False (default), only includes summary statistics.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - gene: Gene name
        - gene_idx: Gene index
        - param: Parameter name
        - median: Median value
        - lo: Lower CI bound (2.5%)
        - hi: Upper CI bound (97.5%)
        - mean: Mean value
        - std: Standard deviation
        - ci_excludes_zero: Boolean, True if CI excludes 0
        If include_samples=True, also includes:
        - sample_idx: Sample index
        - value: Sample value

    Examples
    --------
    >>> # Get summary statistics
    >>> df = extract_posterior_dataframe(model, ['n_a', 'n_b', 'K_a', 'K_b'])
    >>> df_dependent = df[df['ci_excludes_zero']]

    >>> # Get all samples for custom analysis
    >>> df_samples = extract_posterior_dataframe(model, ['n_a'], include_samples=True)
    >>> sns.violinplot(data=df_samples, x='gene', y='value')
    """
    import pandas as pd

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Get posterior samples
    if modality_name == model.primary_modality:
        posterior = model.posterior_samples_trans
    else:
        posterior = modality.posterior_samples_trans

    if posterior is None:
        raise ValueError(
            f"No posterior_samples_trans found for modality '{modality_name}'. "
            "Must run fit_trans() first."
        )

    # Get gene names
    gene_names = modality.feature_names
    if gene_names is None:
        for col in ['gene_name', 'gene', 'feature_id', 'feature']:
            if col in modality.feature_meta.columns:
                gene_names = modality.feature_meta[col].tolist()
                break
    if gene_names is None:
        gene_names = [str(i) for i in range(modality.dims['n_features'])]

    rows = []

    for param in params:
        if param not in posterior:
            print(f"[WARNING] Parameter '{param}' not found in posterior, skipping.")
            continue

        samps = to_np(posterior[param])

        # Handle different shapes
        if samps.ndim == 3:
            samps = samps[:, 0, :]  # (S, 1, T) -> (S, T)
        elif samps.ndim == 1:
            samps = samps.reshape(-1, 1)

        S, T = samps.shape

        # Ensure gene_names matches T
        gene_names_use = gene_names[:T] if len(gene_names) >= T else gene_names + [f'gene_{i}' for i in range(len(gene_names), T)]

        for i in range(T):
            gene_samps = samps[:, i]

            # Compute statistics
            median_val = np.nanmedian(gene_samps)
            lo_val = np.nanpercentile(gene_samps, 2.5)
            hi_val = np.nanpercentile(gene_samps, 97.5)
            mean_val = np.nanmean(gene_samps)
            std_val = np.nanstd(gene_samps)
            ci_excludes_zero = (lo_val > 0) or (hi_val < 0)

            if include_samples:
                # Add one row per sample
                for s_idx, val in enumerate(gene_samps):
                    rows.append({
                        'gene': gene_names_use[i],
                        'gene_idx': i,
                        'param': param,
                        'sample_idx': s_idx,
                        'value': float(val),
                        'median': median_val,
                        'lo': lo_val,
                        'hi': hi_val,
                        'mean': mean_val,
                        'std': std_val,
                        'ci_excludes_zero': ci_excludes_zero,
                    })
            else:
                # Add one summary row per gene
                rows.append({
                    'gene': gene_names_use[i],
                    'gene_idx': i,
                    'param': param,
                    'median': median_val,
                    'lo': lo_val,
                    'hi': hi_val,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_excludes_zero': ci_excludes_zero,
                })

    return pd.DataFrame(rows)
