"""
Diagnostic plotting functions.

Provides quality control and diagnostic plots for bayesDREAM fits.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from statsmodels.nonparametric.smoothers_lowess import lowess

from .colors import ColorScheme


def scatter_with_smooth_by_group(
    x,
    y,
    group,
    frac=0.2,
    s=1,
    alpha=0.3,
    xlabel=None,
    ylabel=None,
    title=None,
):
    """
    Scatter plot with per-group lowess smoothed lines, drawn on the current axes.

    Parameters
    ----------
    x, y : array-like or torch.Tensor
    group : array-like
        Group labels; one line per unique value.
    frac : float
        Lowess smoothing fraction.
    s, alpha : float
        Scatter marker size and transparency.
    xlabel, ylabel, title : str, optional
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    group = np.asarray(group)

    x = np.ravel(x)
    y = np.ravel(y)
    group = np.ravel(group)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    group = group[mask]

    for g in np.unique(group):
        idx = group == g
        x_g = x[idx]
        y_g = y[idx]

        order = np.argsort(x_g)
        x_g = x_g[order]
        y_g = y_g[order]

        plt.scatter(x_g, y_g, s=s, alpha=alpha, label=str(g))
        smoothed = lowess(y_g, x_g, frac=frac, return_sorted=True)
        (line,) = plt.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.5, zorder=10)
        line.set_path_effects([
            pe.Stroke(linewidth=4, foreground="white"),
            pe.Normal(),
        ])

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    plt.legend()
    plt.show()


def plot_x_true_residuals_vs_sumfactor(
    model,
    sum_factor_col="sum_factor",
    group_col="lane",
    facet_col="target",
    frac=0.2,
    s=1,
    alpha=0.3,
    figsize=(12, 5),
    min_cells_for_smooth=10,
):
    """
    Diagnostic: log2(sum_factor) vs normalised log2_x_true, faceted by a
    user-specified column, with a lowess line per group.

    For each cell assigned to guide g, the normalised x_true is::

        (log2_x_true_cell - log2_x_eff_g_guide) / sigma_eff_guide

    where log2_x_eff_g and sigma_eff are posterior-mean guide-level parameters
    from the cis fit.  Under the model this should be N(0, 1) with no dependence
    on sum_factor; a systematic trend reveals residual confounding between
    sequencing depth and inferred expression.

    Only single-guide cells are included (all cells in low-MOI; a subset in
    high-MOI mode).

    Parameters
    ----------
    model : bayesDREAM
        A fitted model (fit_cis must have been called).
    sum_factor_col : str
        Column in the primary modality's sum_factors to use as y-axis.
    group_col : str
        Column in model.meta to colour / smooth separately (e.g. 'lane').
    facet_col : str
        Column in model.meta to facet by; one panel per unique value
        (e.g. 'target', 'cell_line').
    frac : float
        Lowess smoothing fraction.
    s, alpha : float
        Scatter marker size and transparency.
    figsize : tuple
    min_cells_for_smooth : int
        Minimum cells in a (facet × group) slice to draw a smoothed line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import torch

    if not hasattr(model, "posterior_samples_cis") or model.posterior_samples_cis is None:
        raise RuntimeError("fit_cis() must be called before this plot.")

    # --- posterior-mean guide-level quantities ---
    x_eff_g_mean = model.posterior_samples_cis["x_eff_g"].mean(dim=0)     # [G]
    sigma_eff_mean = model.posterior_samples_cis["sigma_eff"].mean(dim=0)  # [G]

    log2_x_eff_g_np = torch.log2(x_eff_g_mean.clamp(min=1e-12)).cpu().numpy()
    sigma_eff_np = sigma_eff_mean.cpu().numpy()

    # --- cell-level arrays (positionally aligned with model.meta rows) ---
    meta = model.meta
    cell_col = meta["cell"].values  # cell barcodes for sum-factor lookup

    log2_x_true = model.log2_x_true
    if hasattr(log2_x_true, "detach"):
        log2_x_true = log2_x_true.detach().cpu().numpy()
    else:
        log2_x_true = np.asarray(log2_x_true, dtype=float)

    sf_series = model.get_modality(model.primary_modality).sum_factors[sum_factor_col]
    sf_values = sf_series.loc[cell_col].values.astype(float)
    log2_sf = np.log2(sf_values)

    groups = meta[group_col].values
    facets_raw = meta[facet_col].values

    # --- single-guide mask + guide index per cell ---
    is_high_moi = getattr(model, "is_high_moi", False)
    if is_high_moi:
        ga = model.guide_assignment_tensor  # [N, G]
        n_guides_per_cell = ga.sum(dim=1).cpu().numpy()
        single_mask = n_guides_per_cell == 1
        guide_idx = ga.argmax(dim=1).cpu().numpy()
    else:
        single_mask = np.ones(len(meta), dtype=bool)
        guide_idx = meta["guide_code"].values.astype(int)

    # --- normalised residual ---
    log2_x_eff_per_cell = log2_x_eff_g_np[guide_idx]
    sigma_per_cell = sigma_eff_np[guide_idx]
    normalized = (log2_x_true - log2_x_eff_per_cell) / np.maximum(sigma_per_cell, 1e-6)

    # apply single-guide mask
    sel = single_mask
    normalized = normalized[sel]
    log2_sf = log2_sf[sel]
    groups = groups[sel]
    facets_raw = facets_raw[sel]
    n_single = int(sel.sum())

    # --- facet panels ---
    facet_values = np.unique(facets_raw)
    n_facets = len(facet_values)

    all_groups = np.unique(groups)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(all_groups), 1)))
    color_map = {g: cmap[i] for i, g in enumerate(all_groups)}

    fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True)
    if n_facets == 1:
        axes = [axes]

    for ax, fval in zip(axes, facet_values):
        facet_mask = facets_raw == fval
        n_facet = int(facet_mask.sum())

        for g in all_groups:
            g_mask = facet_mask & (groups == g)
            if g_mask.sum() < 2:
                continue

            x_g = normalized[g_mask]
            y_g = log2_sf[g_mask]
            order = np.argsort(x_g)
            x_g = x_g[order]
            y_g = y_g[order]

            color = color_map[g]
            ax.scatter(x_g, y_g, s=s, alpha=alpha, color=color, label=str(g), rasterized=True)

            if g_mask.sum() >= min_cells_for_smooth:
                smoothed = lowess(y_g, x_g, frac=frac, return_sorted=True)
                (line,) = ax.plot(
                    smoothed[:, 0], smoothed[:, 1],
                    linewidth=2.5, color=color, zorder=10,
                )
                line.set_path_effects([
                    pe.Stroke(linewidth=4, foreground="white"),
                    pe.Normal(),
                ])

        ax.axvline(0, color="gray", linewidth=1, linestyle="--", zorder=5, alpha=0.7)
        ax.set_xlabel(
            "Normalised log2_x_true\n"
            r"$(log_2 x_{true} - log_2 x_{eff,g}) \,/\, \sigma_{eff,g}$"
        )
        ax.set_title(f"{facet_col}={fval}  (n={n_facet:,})")
        ax.legend(title=group_col, markerscale=4, fontsize=7)

    axes[0].set_ylabel(f"log2({sum_factor_col})")
    fig.suptitle(
        f"log2({sum_factor_col}) vs normalised log2_x_true\n"
        f"single-guide cells only  (n={n_single:,})"
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_sum_factor_comparison(model, cis_gene=None, sf_col1='clustered.sum.factor',
                               sf_col2='sum_factor_adj', color_scheme=None, show=True):
    """
    Plot pairwise comparison of sum factors (e.g., original vs adjusted).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str, optional
        Cis gene name for title (defaults to model.cis_gene)
    sf_col1 : str
        First sum factor column name
    sf_col2 : str
        Second sum factor column name
    color_scheme : ColorScheme, optional
        Custom color scheme
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib figure
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    fig, ax = plt.subplots(figsize=(5, 4))

    df = model.meta[['cell', 'guide']].copy() if 'guide' in model.meta.columns \
        else model.meta[['cell']].copy()

    # Resolve each sum factor column: check model.meta first, then primary modality
    def _resolve_sf_col(col):
        if col in model.meta.columns:
            return model.meta.set_index('cell')[col]
        primary_mod = model.get_modality(model.primary_modality)
        if (primary_mod.sum_factors is not None
                and col in primary_mod.sum_factors.columns):
            return primary_mod.sum_factors[col]
        return None

    s1 = _resolve_sf_col(sf_col1)
    s2 = _resolve_sf_col(sf_col2)

    if s1 is None or s2 is None:
        missing = [c for c, s in [(sf_col1, s1), (sf_col2, s2)] if s is None]
        primary_mod = model.get_modality(model.primary_modality)
        available_meta = list(model.meta.columns)
        available_mod  = (list(primary_mod.sum_factors.columns)
                          if primary_mod.sum_factors is not None else [])
        print(f"Missing sum factor column(s): {missing}. "
              f"Available in meta: {available_meta}. "
              f"Available in modality sum_factors: {available_mod}")
        return fig

    df[sf_col1] = s1.loc[model.meta['cell'].values].values
    df[sf_col2] = s2.loc[model.meta['cell'].values].values

    # Filter to positive values
    df = df[(df[sf_col1] > 0) & (df[sf_col2] > 0)]

    df = df[(df[sf_col1] > 0) & (df[sf_col2] > 0)]

    # Plot by guide
    for guide, sub in df.groupby('guide'):
        color = color_scheme.get_guide_color(guide, 'black')
        ax.scatter(
            sub[sf_col1],
            sub[sf_col2],
            s=12,
            alpha=0.8,
            color=color,
            label=guide
        )

    # Identity line
    all_sf = np.concatenate([df[sf_col1].values, df[sf_col2].values])
    sf_min, sf_max = all_sf.min(), all_sf.max()
    ax.plot([sf_min, sf_max], [sf_min, sf_max], 'k--', linewidth=1, alpha=0.6)

    ax.set_xlabel(sf_col1, fontsize=10)
    ax.set_ylabel(sf_col2, fontsize=10)
    ax.set_title(f'{cis_gene}: sum factor comparison', fontsize=11)
    ax.legend(fontsize=8, markerscale=1.2, frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig
