"""
Posterior density visualization functions.

Provides vertical density line plots for parameter posteriors and x_true distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from .helpers import to_np, resolve_guide_labels, _guide_ntc_mask, _xtrue_posterior
from .colors import ColorScheme


def _guide_sort_key(g):
    """Sort guides by root name then trailing number."""
    parts = g.rsplit('_', 1)
    root = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (root, idx)


def plot_posterior_density_lines(
    samples,
    title="Posterior density lines",
    sort_by="median",
    subset_mask=None,
    cmap="viridis",
    alpha_overall=0.5,
    density_gamma=0.7,
    norm_global=True,
    y_quantiles=(0.5, 99.5),
    grid_points=350,
    linewidth=0.8,
    add_median_lines=True,
    y_label=r"$\theta$",
    ax=None,
    show=True,
    y_range=None,
):
    """
    Plot per-feature posterior densities as vertical color lines.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_features)
        Posterior samples
    title : str
        Plot title
    sort_by : {'median', 'mean', None}
        How to sort features
    subset_mask : array-like, optional
        Boolean mask to subset features
    cmap : str or Colormap
        Colormap for density visualization
    alpha_overall : float
        Overall alpha for density colors
    density_gamma : float
        Gamma correction for density intensity
    norm_global : bool
        If True, normalize density across all features
    y_quantiles : tuple
        Quantiles for y-axis range
    grid_points : int
        Number of grid points for KDE
    linewidth : float
        Width of median lines
    add_median_lines : bool
        Whether to add horizontal median lines
    y_label : str
        Y-axis label
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool
        Whether to display the plot
    y_range : tuple, optional
        Explicit (y_min, y_max) range

    Returns
    -------
    ax : matplotlib axes
    """
    samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)

    S, T = samples.shape

    if subset_mask is not None:
        subset_mask = np.asarray(subset_mask, dtype=bool)
        samples = samples[:, subset_mask]
        S, T = samples.shape

    if sort_by == "median":
        order = np.argsort(np.nanmedian(samples, axis=0))
    elif sort_by == "mean":
        order = np.argsort(np.nanmean(samples, axis=0))
    else:
        order = np.arange(T)
    samples_sorted = samples[:, order]

    # --- y-range: either from samples, or overridden explicitly ---
    if y_range is None:
        y_min, y_max = np.nanpercentile(samples_sorted, y_quantiles)
    else:
        y_min, y_max = y_range

    y_grid = np.linspace(y_min, y_max, grid_points)

    # KDE per feature
    dens_list = []
    for t in range(T):
        vals = samples_sorted[:, t]
        vals = vals[~np.isnan(vals)]
        if vals.size < 2:
            dens = np.zeros_like(y_grid)
        else:
            kde = gaussian_kde(vals)
            dens = kde(y_grid)
        dens_list.append(dens)
    dens_mat = np.stack(dens_list, axis=0)  # (T, grid_points)

    # Normalize density
    if norm_global:
        dens_max = dens_mat.max()
    else:
        dens_max = dens_mat.max(axis=1, keepdims=True)
    dens_max = np.maximum(dens_max, 1e-12)
    dens_norm = dens_mat / dens_max

    # Apply gamma correction
    dens_norm = dens_norm ** density_gamma

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Get colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Draw vertical density lines
    for t in range(T):
        x_pos = t + 1
        for i in range(grid_points):
            alpha = dens_norm[t, i] * alpha_overall
            if alpha > 0.01:  # Skip very faint lines
                color = cmap_obj(dens_norm[t, i])
                ax.plot([x_pos, x_pos], [y_grid[i], y_grid[i]],
                       color=color, alpha=alpha, linewidth=linewidth)

    # Add median lines
    if add_median_lines:
        medians = np.nanmedian(samples_sorted, axis=0)
        for t in range(T):
            ax.plot([t+0.7, t+1.3], [medians[t], medians[t]],
                   color='red', linewidth=1.2, alpha=0.8)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature index (sorted)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def plot_xtrue_density_by_guide(
    model,
    cis_gene=None,
    log2=False,
    cmap="viridis",
    alpha_overall=0.5,
    density_gamma=0.7,
    norm_global=True,
    y_quantiles=(0.5, 99.5),
    grid_points=350,
    linewidth=0.8,
    group_by_guide=True,
    single_guide_cells_only=False,
    targeted_only=False,
    color_scheme=None,
    show=True,
):
    """
    Per-cell posterior density plot for x_true, one density strip per cell.

    Each cell's strip is a KDE computed from its ``S`` posterior samples of
    ``x_true`` (loaded from ``model.posterior_samples_cis['x_true']``, shape
    ``[S, N_cells]``).  Cells are grouped and sorted by guide, with a colour
    bar above the plot annotating guide membership.

    Falls back to ``model.x_true`` point estimates (one value per cell, no KDE
    width) when no posterior is stored.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    cis_gene : str, optional
        Cis gene name (title only).
    log2 : bool, default False
        Apply log2 transform.
    cmap : str or Colormap
        Colormap for density intensity.
    alpha_overall : float
        Overall alpha for density colours.
    density_gamma : float
        Gamma correction for density intensity.
    norm_global : bool
        Normalise density globally across all cells.
    y_quantiles : tuple
        Quantiles (lo, hi) for y-axis range.
    grid_points : int
        KDE grid resolution.
    linewidth : float
        Line width for median ticks.
    group_by_guide : bool
        Sort cells by guide then by within-guide median.  If ``False``,
        sort globally by median.
    single_guide_cells_only : bool, default False
        Required to be ``True`` for high-MOI models.
    targeted_only : bool, default False
        If ``True``, exclude NTC cells (show only cells from targeting guides).
    color_scheme : ColorScheme, optional
        Custom colour scheme for guide annotations.
    show : bool, default True

    Returns
    -------
    fig : matplotlib figure
    """
    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)
    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    guide_labels, cell_mask = resolve_guide_labels(model, single_guide_cells_only)

    # ---- load posterior or fall back to point estimates ----
    post = _xtrue_posterior(model)          # [S, N_cells] or None
    if post is not None:
        samples = post[:, cell_mask]        # [S, N_kept]
    else:
        x_vals = to_np(model.x_true)[cell_mask]
        samples = x_vals[np.newaxis, :]     # [1, N_kept]

    guide_labels = guide_labels[cell_mask]

    if log2:
        with np.errstate(divide='ignore', invalid='ignore'):
            samples = np.where(samples > 0,
                               np.log2(np.maximum(samples, 1e-300)), np.nan)

    # ---- optional NTC filter ----
    if targeted_only:
        ntc = _guide_ntc_mask(guide_labels, model)
        guide_labels = guide_labels[~ntc]
        samples = samples[:, ~ntc]

    S, N = samples.shape
    guides = guide_labels            # [N]

    # ---- cell ordering ----
    med_per_cell = np.nanmedian(samples, axis=0)   # [N]

    if group_by_guide:
        unique_guides = sorted(np.unique(guides), key=_guide_sort_key)
        block_rank    = {g: i for i, g in enumerate(unique_guides)}
        guide_ranks   = np.array([block_rank[g] for g in guides])
        order         = np.lexsort((med_per_cell, guide_ranks))
    else:
        unique_guides = sorted(np.unique(guides), key=_guide_sort_key)
        order         = np.argsort(med_per_cell)

    samples_sorted = samples[:, order]       # [S, N]
    guides_sorted  = guides[order]           # [N]

    ylabel = "x_true" + (" (log₂)" if log2 else "")

    fig = plt.figure(figsize=(max(10, N * 0.03 + 2), 6))
    ax  = plt.subplot2grid((20, 1), (2, 0), rowspan=18)

    plot_posterior_density_lines(
        samples_sorted,          # [S, N] — each column is one cell
        title="",
        sort_by=None,
        cmap=cmap,
        alpha_overall=alpha_overall,
        density_gamma=density_gamma,
        norm_global=norm_global,
        y_quantiles=y_quantiles,
        grid_points=grid_points,
        linewidth=linewidth,
        add_median_lines=False,
        y_label=ylabel,
        ax=ax,
        show=False,
    )

    # ---- coloured median tick per cell ----
    medians_sorted = np.nanmedian(samples_sorted, axis=0)
    for i, g in enumerate(guides_sorted):
        color = color_scheme.get_guide_color(g, 'black')
        ax.plot([i + 0.7, i + 1.3], [medians_sorted[i], medians_sorted[i]],
                color=color, linewidth=1.5, alpha=0.9)

    # ---- guide colour bar above axes ----
    ax_bar = plt.subplot2grid((20, 1), (0, 0), rowspan=1)
    ax_bar.set_xlim(0.5, N + 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    for i, g in enumerate(guides_sorted):
        color = color_scheme.get_guide_color(g, 'black')
        rect = Rectangle((i + 0.5, 0), 1, 1, facecolor=color, edgecolor='none')
        ax_bar.add_patch(rect)

    # ---- legend ----
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_scheme.get_guide_color(g, 'black'),
               markersize=8, label=g)
        for g in unique_guides
    ]
    ax.legend(handles=legend_handles, title='guide', fontsize=9,
              loc='upper left', frameon=True, framealpha=0.9)

    title_suffix = ' (targeted only)' if targeted_only else ''
    fig.suptitle(f'{cis_gene}: x_true posterior density by cell{title_suffix}',
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()

    return fig


def plot_parameter_density_with_xtrue(
    param_samps,
    model,
    cis_gene=None,
    param_name='x_infl',
    subset_mask=None,
    log2=True,
    cmap="viridis",
    alpha_overall=0.45,
    density_gamma=0.7,
    norm_global=True,
    grid_points=350,
    linewidth=0.8,
    show_xtrue=True,
    color_scheme=None,
    show=True,
):
    """
    Two-panel density plot: parameter samples (left) + x_true distribution (right).

    This is useful for comparing trans parameter distributions (e.g., x_infl, K_a, n_a)
    with the observed x_true range to contextualize the parameter values.

    Parameters
    ----------
    param_samps : np.ndarray
        Parameter samples, shape (n_samples, n_features)
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    param_name : str
        Parameter name for labeling (default: 'x_infl')
    subset_mask : np.ndarray, optional
        Boolean mask to subset features (e.g., dependent genes only)
    log2 : bool
        Whether to plot on log2 scale (default: True)
    cmap : str
        Colormap for density visualization (default: 'viridis')
    alpha_overall : float
        Overall alpha for density colors (default: 0.45)
    density_gamma : float
        Gamma correction for density intensity (default: 0.7)
    norm_global : bool
        If True, normalize density across all features (default: True)
    grid_points : int
        Number of grid points for KDE (default: 350)
    linewidth : float
        Width of density lines (default: 0.8)
    show_xtrue : bool
        Whether to show x_true panel on right (default: True)
    color_scheme : ColorScheme, optional
        Custom color scheme for x_true target colors
    show : bool
        Whether to display the plot (default: True)

    Returns
    -------
    fig : matplotlib figure

    Examples
    --------
    >>> from bayesDREAM.plotting import (plot_parameter_density_with_xtrue,
    ...                                   hill_xinf_samples, dependency_mask_from_n)
    >>> K_samps = model.posterior_samples_trans['K_a'][:, 0, :].detach().cpu().numpy()
    >>> n_samps = model.posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()
    >>> xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
    >>> mask = dependency_mask_from_n(n_samps)
    >>> fig = plot_parameter_density_with_xtrue(
    ...     xinf_samps, model, 'GFI1B',
    ...     param_name='x_infl',
    ...     subset_mask=mask,
    ...     log2=True
    ... )
    """
    from scipy.stats import gaussian_kde

    if color_scheme is None:
        color_scheme = getattr(model, 'color_scheme', None) or ColorScheme.from_model(model)

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    # Convert to numpy and apply log2 if requested
    param_samps = np.asarray(param_samps)
    if log2:
        from .utils import log2_pos
        param_samps = log2_pos(param_samps)

    # Get x_true (point estimate per cell)
    df_meta = model.meta.copy()
    xtrue_mean_per_cell = to_np(model.x_true)  # [N_cells]

    if log2:
        from .utils import log2_pos
        xtrue_mean_per_cell = log2_pos(xtrue_mean_per_cell)

    # Compute y-range from x_true distribution (central 99% of cells)
    vals_all = xtrue_mean_per_cell[~np.isnan(xtrue_mean_per_cell)]
    if vals_all.size > 0:
        y_min = np.percentile(vals_all, 0.5)
        y_max = np.percentile(vals_all, 99.5)
    else:
        y_min, y_max = 0, 1
    y_range = (y_min, y_max)

    # Global 95% CI of parameter (for reference lines)
    param_vals_all = param_samps.flatten()
    param_vals_all = param_vals_all[~np.isnan(param_vals_all)]
    if param_vals_all.size > 0:
        ci_lo, ci_hi = np.percentile(param_vals_all, [2.5, 97.5])
    else:
        ci_lo, ci_hi = y_min, y_max

    # Create figure layout
    if show_xtrue:
        fig, (ax_main, ax_side) = plt.subplots(
            1, 2,
            figsize=(8, 5),
            gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
            sharey=True,
        )
    else:
        fig, ax_main = plt.subplots(figsize=(10, 5))
        ax_side = None

    # ----- Main panel: parameter posterior density -----
    ylabel = f'$\\log_2$ {param_name}' if log2 else param_name
    title = f"{cis_gene} — posterior of {ylabel}"
    if subset_mask is not None:
        n_total = param_samps.shape[1]
        n_subset = np.sum(subset_mask)
        title += f" ({n_subset}/{n_total} features)"

    ax_main = plot_posterior_density_lines(
        param_samps,
        title=title,
        subset_mask=subset_mask,
        cmap=cmap,
        alpha_overall=alpha_overall,
        density_gamma=density_gamma,
        norm_global=norm_global,
        add_median_lines=True,
        y_label=ylabel,
        ax=ax_main,
        show=False,
        y_range=y_range,
        grid_points=grid_points,
        linewidth=linewidth,
    )

    ax_main.set_xlabel("Features (ordered by median)")
    ax_main.set_ylim(y_min, y_max)

    # Left y ticks
    ax_main.set_yticks(np.linspace(y_min, y_max, 5))
    ax_main.yaxis.set_ticks_position('left')
    ax_main.tick_params(axis='y', which='both', length=4)

    # Indicate global 95% CI region
    ax_main.axhline(ci_lo, color='white', linestyle=':', linewidth=0.7, alpha=0.7)
    ax_main.axhline(ci_hi, color='white', linestyle=':', linewidth=0.7, alpha=0.7)

    # ----- Side panel: x_true density by target -----
    if show_xtrue and ax_side is not None:
        ax_side.set_xlabel(f'density of\n$\\log_2$ x_true' if log2 else 'density of\nx_true',
                          fontsize=9)
        ax_side.xaxis.set_label_position('top')

        targets = df_meta['target'].astype(str).to_numpy()
        uniq_targets = sorted(np.unique(targets))

        y_grid = np.linspace(y_min, y_max, 400)

        for t in uniq_targets:
            mask_t = targets == t
            vals_t = xtrue_mean_per_cell[mask_t]
            vals_t = vals_t[~np.isnan(vals_t)]
            if vals_t.size == 0:
                continue

            color = color_scheme.get_target_color(t, 'grey')

            if vals_t.size < 2:
                # Tiny bump if no variance
                y0 = vals_t[0]
                bump = np.exp(-0.5 * ((y_grid - y0) / 0.05) ** 2)
                bump /= bump.max() + 1e-12
                ax_side.fill_betweenx(y_grid, 0, bump, color=color, alpha=0.45)
                ax_side.plot(bump, y_grid, color=color, linewidth=1.0)
            else:
                kde = gaussian_kde(vals_t)
                dens_t = kde(y_grid)
                dens_t /= dens_t.max() + 1e-12
                ax_side.fill_betweenx(y_grid, 0, dens_t, color=color, alpha=0.45)
                ax_side.plot(dens_t, y_grid, color=color, linewidth=1.0, label=t)

        ax_side.set_xlim(0, 1.05)

        # Right y-axis
        ax_side.yaxis.set_ticks_position('right')
        ax_side.yaxis.set_label_position('right')
        ax_side.set_yticks(np.linspace(y_min, y_max, 5))
        ax_side.tick_params(axis='y', which='both', length=4)
        ax_side.set_ylabel("")  # Keep only left label

        # Legend for targets
        ax_side.legend(title='target', fontsize=8, loc='upper right', frameon=False)

    fig.tight_layout()

    if show:
        plt.show()

    return fig
