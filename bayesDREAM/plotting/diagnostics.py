"""
Diagnostic plotting functions.

Provides quality control and diagnostic plots for bayesDREAM fits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr, spearmanr

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
    ax=None,
    show=True,
):
    """
    Scatter plot with per-group lowess smoothed lines, drawn on the current axes.

    Parameters
    ----------
    x : array-like or torch.Tensor
        X-axis values.
    y : array-like or torch.Tensor
        Y-axis values.
    group : array-like
        Group labels; one smoothed line is drawn per unique value.
    frac : float, default 0.2
        Lowess smoothing fraction.
    s : float, default 1
        Scatter marker size.
    alpha : float, default 0.3
        Scatter marker transparency.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    ax : matplotlib axes, optional
        Axes to plot on. If None, uses current axes.
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    ax : matplotlib axes
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

    if ax is None:
        ax = plt.gca()

    for g in np.unique(group):
        idx = group == g
        x_g = x[idx]
        y_g = y[idx]

        order = np.argsort(x_g)
        x_g = x_g[order]
        y_g = y_g[order]

        ax.scatter(x_g, y_g, s=s, alpha=alpha, label=str(g))
        smoothed = lowess(y_g, x_g, frac=frac, return_sorted=True)
        (line,) = ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.5, zorder=10)
        line.set_path_effects([
            pe.Stroke(linewidth=4, foreground="white"),
            pe.Normal(),
        ])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.legend()
    if show:
        plt.show()
    return ax


def plot_x_true_residuals_vs_sumfactor(
    model,
    group_col,
    facet_col,
    sum_factor_col="sum_factor",
    frac=0.2,
    s=1,
    alpha=0.3,
    figsize=(12, 5),
    min_cells_for_smooth=10,
    show=True,
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
    group_col : str
        Column in model.meta to colour and smooth separately (e.g. ``'lane'``,
        ``'batch'``).
    facet_col : str
        Column in model.meta to facet by; one panel per unique value
        (e.g. ``'target'``, ``'cell_line'``).
    sum_factor_col : str, default ``'sum_factor'``
        Column in the primary modality's sum_factors to use as y-axis.
    frac : float, default 0.2
        Lowess smoothing fraction.
    s : float, default 1
        Scatter marker size.
    alpha : float, default 0.3
        Scatter marker transparency.
    figsize : tuple, default (12, 5)
        Figure size.
    min_cells_for_smooth : int, default 10
        Minimum cells in a (facet × group) slice to draw a smoothed line.
    show : bool, default True
        Whether to call ``plt.show()``.

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
    if show:
        plt.show()
    return fig


def plot_sum_factor_comparison(model, sf_col1, sf_col2, cis_gene=None,
                               color_scheme=None, show=True):
    """
    Plot pairwise comparison of sum factors (e.g., original vs adjusted).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model.
    sf_col1 : str
        First sum factor column name (checked in model.meta then primary modality
        sum_factors).
    sf_col2 : str
        Second sum factor column name (checked in model.meta then primary modality
        sum_factors).
    cis_gene : str, optional
        Cis gene name for plot title. Defaults to ``model.cis_gene``.
    color_scheme : ColorScheme, optional
        Custom color scheme. Defaults to a plain ``ColorScheme()``.
    show : bool, default True
        Whether to call ``plt.show()``.

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


# Columns produced by check_systematic_shift() that are never the group/facet
# column -- used to auto-detect the group column (named after whatever
# tech_col was passed to check_systematic_shift).
_SHIFT_RESULT_COLS = frozenset({
    'feature', 'ok', 'reason', 'n_ntc', 'n_targeted', 'theta',
    'pval', 'p_lrt', 'p_adj', 'shift_est', 'shift_se', 'shift_p',
    'shift_p_adj', 'shift_fc', 'lrt_stat', 'lrt_df', 'n_categories_tested',
})

# Display labels for the two standard corrected-p-value columns; falls back
# to the raw column name for anything else (e.g. a user-supplied p_col).
_P_COL_LABELS = {
    'p_adj': r'$p_{adj}$',
    'shift_p_adj': r'$p_{adj}^{\mathrm{shift}}$',
}


def _detect_group_col(res, group_col):
    """
    Resolve the grouping column for check_systematic_shift() results.

    If ``group_col`` is given, returns it unchanged. Otherwise prefers
    ``'technical_group_code'`` (check_systematic_shift's default ``tech_col``
    name) if present; else falls back to the single non-boolean column in
    ``res`` that isn't one of the standard result columns, raising
    ``ValueError`` if that's ambiguous (e.g. user-added derived columns).
    """
    if group_col is not None:
        return group_col
    if "technical_group_code" in res.columns:
        return "technical_group_code"
    extra = [
        c for c in res.columns
        if c not in _SHIFT_RESULT_COLS and res[c].dtype != bool
    ]
    if len(extra) != 1:
        raise ValueError(
            f"Could not auto-detect group_col (candidates: {extra}). "
            "Pass group_col explicitly."
        )
    return extra[0]


def _declutter_texts(fig, ax, texts, iterations=200, pad_px=1.5):
    """
    Iteratively nudge overlapping ``Text`` labels apart in display-pixel
    space, converting shifts back to data coordinates for repositioning.

    Lightweight fallback for when ``adjustText`` isn't installed: not as
    good, but resolves direct overlaps within a panel.
    """
    if len(texts) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent(renderer=renderer)
    px_per_xdata = bbox.width / (x1 - x0)
    px_per_ydata = bbox.height / (y1 - y0)
    if px_per_xdata <= 0 or px_per_ydata <= 0:
        return

    for _ in range(iterations):
        boxes = [t.get_window_extent(renderer=renderer) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if not bi.overlaps(bj):
                    continue
                moved = True
                overlap_x = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                overlap_y = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                cix, cjx = (bi.x0 + bi.x1) / 2, (bj.x0 + bj.x1) / 2
                ciy, cjy = (bi.y0 + bi.y1) / 2, (bj.y0 + bj.y1) / 2
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                if overlap_x < overlap_y:
                    shift = (overlap_x / 2 + pad_px) * (1.0 if cix >= cjx else -1.0)
                    dx = shift / px_per_xdata
                    texts[i].set_position((xi + dx, yi))
                    texts[j].set_position((xj - dx, yj))
                else:
                    shift = (overlap_y / 2 + pad_px) * (1.0 if ciy >= cjy else -1.0)
                    dy = shift / px_per_ydata
                    texts[i].set_position((xi, yi + dy))
                    texts[j].set_position((xj, yj - dy))
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _draw_leader_lines(ax, texts, orig_xy, min_px=3.0, color="gray", lw=0.5):
    """Draw a thin line from each label's original point to its (possibly
    decluttered) final position, when it moved more than ``min_px`` pixels."""
    fig = ax.figure
    fig.canvas.draw()
    for txt, (x0, y0) in zip(texts, orig_xy):
        x1, y1 = txt.get_position()
        d0 = ax.transData.transform((x0, y0))
        d1 = ax.transData.transform((x1, y1))
        if np.hypot(*(d1 - d0)) > min_px:
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=3.5)


def plot_systematic_shift_volcano(
    res,
    group_col=None,
    p_col="p_adj",
    effect_col="shift_est",
    label_col="feature",
    fdr_threshold=0.1,
    top_n=20,
    sig_color="firebrick",
    nonsig_color="#999999",
    label_fontsize=7,
    figsize=None,
    show=True,
):
    """
    Volcano plot of ``check_systematic_shift()`` results, faceted by group.

    One panel per unique value of ``group_col`` (the grouping column produced
    by ``check_systematic_shift(tech_col=...)`` -- named after whatever
    ``tech_col`` was passed, e.g. ``'technical_group_code'`` by default).
    Each panel plots ``effect_col`` (x) against ``-log10(p_col)`` (y),
    highlights points below ``fdr_threshold``, and labels the ``top_n`` most
    significant features per panel.

    Parameters
    ----------
    res : pd.DataFrame
        Output of ``model.check_systematic_shift()``.
    group_col : str, optional
        Column to facet by. If None: uses ``'technical_group_code'`` if present
        (``check_systematic_shift``'s default ``tech_col`` name), otherwise
        auto-detected as the single non-boolean column in ``res`` that isn't
        one of the standard result columns (raises ``ValueError`` if that's
        ambiguous, e.g. if you added your own derived columns to ``res`` --
        pass it explicitly in that case).
    p_col : str, default ``'p_adj'``
        Which BH-corrected p-value column to plot: ``'p_adj'`` (likelihood-ratio
        test) or ``'shift_p_adj'`` (Wald test on the ``targeted`` coefficient).
        See ``check_systematic_shift``'s docstring for the difference.
    effect_col : str, default ``'shift_est'``
        Column for the x-axis (effect size).
    label_col : str, default ``'feature'``
        Column supplying the text used to label top hits.
    fdr_threshold : float, default 0.1
        Significance cutoff; also drawn as a horizontal dashed line.
    top_n : int, default 20
        Number of most-significant (lowest ``p_col``) features to label,
        per panel, among rows below ``fdr_threshold``.
    sig_color, nonsig_color : str
        Colors for significant / non-significant points.
    label_fontsize : int, default 7
        Font size for point labels.
    figsize : tuple, optional
        Figure size. Defaults to ``(5 * n_panels, 5)``.
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``group_col`` is None and cannot be unambiguously auto-detected
        (see ``group_col`` above), or if no rows have ``ok == True`` with
        finite ``effect_col``/``p_col`` values to plot.

    Notes
    -----
    Only rows with ``ok == True`` and finite ``effect_col``/``p_col`` are
    plotted; skipped (``ok == False``) rows are silently dropped.

    If the `adjustText <https://github.com/Phlya/adjustText>`_ package is
    installed, labels are auto-repelled using it (best quality). Otherwise a
    built-in pixel-space decluttering pass nudges overlapping labels apart
    and draws thin leader lines back to their points.
    """
    group_col = _detect_group_col(res, group_col)

    df = res[res["ok"] & np.isfinite(res[effect_col]) & np.isfinite(res[p_col])].copy()
    if df.empty:
        raise ValueError("No rows with ok=True and finite values to plot.")

    # Avoid -log10(0) = inf for exact-zero p-values
    tiny = np.finfo(float).tiny
    df["_neg_log10_p"] = -np.log10(df[p_col].clip(lower=tiny))
    df["_sig"] = df[p_col] < fdr_threshold

    group_values = sorted(df[group_col].unique())
    n_panels = len(group_values)
    if figsize is None:
        figsize = (5 * n_panels, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    hline_y = -np.log10(fdr_threshold)

    for ax, gval in zip(axes, group_values):
        sub = df[df[group_col] == gval]
        nonsig = sub[~sub["_sig"]]
        sig = sub[sub["_sig"]]

        ax.axvline(0, linestyle="--", color="gray", linewidth=1, zorder=1)
        ax.axhline(hline_y, linestyle="--", color=sig_color, linewidth=1, zorder=1)

        ax.scatter(
            nonsig[effect_col], nonsig["_neg_log10_p"],
            s=10, alpha=0.4, color=nonsig_color, label="Not significant", zorder=2,
        )
        ax.scatter(
            sig[effect_col], sig["_neg_log10_p"],
            s=14, alpha=0.8, color=sig_color,
            label=f"{_P_COL_LABELS.get(p_col, p_col)} < {fdr_threshold}", zorder=3,
        )

        top_hits = sig.nsmallest(top_n, p_col)
        texts = []
        orig_xy = []
        x_span = sub[effect_col].max() - sub[effect_col].min()
        y_span = sub["_neg_log10_p"].max() - sub["_neg_log10_p"].min()
        dx0 = 0.01 * x_span if x_span > 0 else 0.01
        dy0 = 0.02 * y_span if y_span > 0 else 0.02
        for _, row in top_hits.iterrows():
            x, y = row[effect_col], row["_neg_log10_p"]
            txt = ax.text(
                x + dx0, y + dy0, str(row[label_col]),
                fontsize=label_fontsize, zorder=4,
            )
            txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
            texts.append(txt)
            orig_xy.append((x, y))

        if texts:
            try:
                from adjustText import adjust_text
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )
            except ImportError:
                # Built-in fallback: nudge overlapping labels apart, then
                # draw thin leader lines back to their points.
                _declutter_texts(fig, ax, texts)
                _draw_leader_lines(ax, texts, orig_xy)

        ax.set_xlabel(effect_col)
        ax.set_title(f"{group_col}={gval}  (n={len(sub):,}, sig={len(sig):,})")
        ax.legend(fontsize=7, frameon=False, markerscale=1.5)

    axes[0].set_ylabel(f"$-\\log_{{10}}$({p_col})")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_shift_est_group_correlation(
    res,
    group_col=None,
    groups=None,
    effect_col="shift_est",
    p_col="p_adj",
    label_col="feature",
    fdr_threshold=0.1,
    highlight="either",
    top_n=0,
    sig_color="firebrick",
    nonsig_color="#999999",
    label_fontsize=7,
    figsize=(5, 5),
    show=True,
):
    """
    Scatter ``effect_col`` for one group against another, matched by feature.

    Checks whether the residual shift detected by ``check_systematic_shift()``
    is consistent across groups (e.g. two technical groups, or CRISPRi vs
    CRISPRa) rather than being an artefact of one group's fit -- a real
    off-target/guide effect should show up with a similar ``shift_est`` in
    both groups; a group-specific artefact will not.

    Parameters
    ----------
    res : pd.DataFrame
        Output of ``model.check_systematic_shift()``.
    group_col : str, optional
        Column identifying groups. See ``plot_systematic_shift_volcano`` for
        auto-detection rules (same logic, shared via ``_detect_group_col``).
    groups : tuple of 2, optional
        Which two group values to compare. Defaults to the two (sorted)
        unique values in ``group_col`` if exactly two are present; raises
        ``ValueError`` if there are more than 2 and this isn't specified.
    effect_col : str, default ``'shift_est'``
        Column to correlate between the two groups.
    p_col : str, default ``'p_adj'``
        Column used for significance highlighting/labeling.
    label_col : str, default ``'feature'``
        Column used to match rows across groups (inner join), and to label
        points.
    fdr_threshold : float, default 0.1
        Significance cutoff for highlighting.
    highlight : {'either', 'both', 'none'}, default ``'either'``
        Highlight points significant in either group, in both groups, or
        skip highlighting.
    top_n : int, default 0
        Label the top N highlighted points by ``min(p_col)`` across the two
        groups. 0 = no labels.
    sig_color, nonsig_color : str
        Colors for significant / non-significant points.
    label_fontsize : int, default 7
        Font size for point labels.
    figsize : tuple, default (5, 5)
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``groups`` is None and ``group_col`` doesn't have exactly 2
        unique values among ``ok == True`` rows; if the inner join on
        ``label_col`` between the two groups produces no shared, valid
        features; or if ``highlight`` isn't one of ``'either'``, ``'both'``,
        ``'none'``.

    Notes
    -----
    Only features with ``ok == True`` and finite ``effect_col``/``p_col`` in
    *both* groups are plotted (inner join on ``label_col``) -- features that
    were skipped or untested in one group are silently dropped. Pearson r
    and Spearman rho (computed on the matched pairs) are annotated on the
    plot. Labels use the same adjustText-or-built-in decluttering as
    ``plot_systematic_shift_volcano``.
    """
    group_col = _detect_group_col(res, group_col)

    ok = res[res["ok"] & np.isfinite(res[effect_col]) & np.isfinite(res[p_col])]
    unique_groups = sorted(ok[group_col].unique())

    if groups is None:
        if len(unique_groups) != 2:
            raise ValueError(
                f"Expected exactly 2 groups, found {unique_groups}. "
                "Pass groups=(a, b) explicitly."
            )
        groups = tuple(unique_groups)
    ga, gb = groups

    da = ok[ok[group_col] == ga][[label_col, effect_col, p_col]]
    db = ok[ok[group_col] == gb][[label_col, effect_col, p_col]]
    merged = pd.merge(da, db, on=label_col, suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(
            f"No shared features with ok=True between groups {ga!r} and {gb!r}."
        )

    x = merged[f"{effect_col}_a"].to_numpy()
    y = merged[f"{effect_col}_b"].to_numpy()
    pa = merged[f"{p_col}_a"].to_numpy()
    pb = merged[f"{p_col}_b"].to_numpy()

    if highlight == "either":
        sig_mask = (pa < fdr_threshold) | (pb < fdr_threshold)
    elif highlight == "both":
        sig_mask = (pa < fdr_threshold) & (pb < fdr_threshold)
    elif highlight == "none":
        sig_mask = np.zeros(len(merged), dtype=bool)
    else:
        raise ValueError("highlight must be 'either', 'both', or 'none'.")

    r, r_p = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    n = len(merged)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x[~sig_mask], y[~sig_mask],
        s=10, alpha=0.4, color=nonsig_color, label="Not significant", zorder=2,
    )
    if sig_mask.any():
        ax.scatter(
            x[sig_mask], y[sig_mask],
            s=14, alpha=0.8, color=sig_color,
            label=f"{_P_COL_LABELS.get(p_col, p_col)} < {fdr_threshold} ({highlight})", zorder=3,
        )

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            linestyle="--", color="gray", linewidth=1, zorder=1)
    ax.axhline(0, linestyle=":", color="gray", linewidth=0.7, zorder=1)
    ax.axvline(0, linestyle=":", color="gray", linewidth=0.7, zorder=1)

    ax.text(
        0.03, 0.97,
        f"Pearson r = {r:.2f} (p={r_p:.1e})\nSpearman ρ = {rho:.2f}\nn = {n:,}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )

    if top_n > 0 and sig_mask.any():
        min_p = np.minimum(pa, pb)
        sig_idx = np.flatnonzero(sig_mask)
        top_idx = sig_idx[np.argsort(min_p[sig_idx])][:top_n]

        texts, orig_xy = [], []
        x_span, y_span = x.max() - x.min(), y.max() - y.min()
        dx0 = 0.01 * x_span if x_span > 0 else 0.01
        dy0 = 0.02 * y_span if y_span > 0 else 0.02
        for i in top_idx:
            xi, yi = x[i], y[i]
            txt = ax.text(
                xi + dx0, yi + dy0, str(merged[label_col].iloc[i]),
                fontsize=label_fontsize, zorder=4,
            )
            txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
            texts.append(txt)
            orig_xy.append((xi, yi))

        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except ImportError:
            _declutter_texts(fig, ax, texts)
            _draw_leader_lines(ax, texts, orig_xy)

    ax.set_xlabel(f"{effect_col} ({group_col}={ga})")
    ax.set_ylabel(f"{effect_col} ({group_col}={gb})")
    ax.set_title(f"{effect_col} correlation across {group_col}")
    ax.legend(fontsize=7, frameon=False, markerscale=1.5, loc="lower right")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_systematic_shift_hits_xy(
    model,
    res,
    p_col="p_adj",
    fdr_threshold=0.1,
    group_col=None,
    label_col="feature",
    features=None,
    tech_col=None,
    target_col="target",
    ntc_label="ntc",
    targeted_label=None,
    exclude_cells=None,
    window_col="_shift_window",
    **plot_xy_kwargs,
):
    """
    Call ``model.plot_xy_data()`` for each significant ``check_systematic_shift()``
    hit, restricted to the cells the test actually used.

    ``check_systematic_shift()`` matches NTC and targeted cells to a ±1 SD
    window around the NTC mean of ``log2(x_true)`` before testing (see its
    "Matching window" docstring section) -- a much narrower slice than the
    full dose-response range ``plot_xy_data()`` shows by default. This
    reconstructs that same window (via ``model.get_shift_window_cells()``)
    and applies it as a ``subset_meta`` filter, so the plot shows exactly
    the cells that drove the test's result.

    Loops over the significant features in ``res`` (``ok`` and
    ``p_col < fdr_threshold``, unless ``features`` is given explicitly) and
    calls::

        model.plot_xy_data(feature, subset_meta={window_col: True, **your_subset_meta}, **plot_xy_kwargs)

    for each, after temporarily adding a boolean ``window_col`` column to
    ``model.meta`` marking cells within *their own group's* matched window
    (each cell only belongs to one ``tech_col`` group, so a single combined
    column is enough even when ``plot_xy_kwargs['facet_by']`` differs from
    ``tech_col``). The column is removed again afterwards, even on error.

    Parameters
    ----------
    model : bayesDREAM
        The fitted model ``res`` was produced from.
    res : pd.DataFrame
        Output of ``model.check_systematic_shift()``.
    p_col : str, default ``'p_adj'``
        Which corrected p-value column selects significant hits.
    fdr_threshold : float, default 0.1
        Significance cutoff for selecting hits (ignored if ``features`` is given).
    group_col : str, optional
        Grouping column in ``res`` (auto-detected as in
        ``plot_systematic_shift_volcano`` if None).
    label_col : str, default ``'feature'``
        Column in ``res`` naming features.
    features : list of str, optional
        Explicit feature list to plot, bypassing the significance filter.
    tech_col, target_col, ntc_label, targeted_label, exclude_cells
        Passed to ``model.get_shift_window_cells()`` to reconstruct the same
        matched window ``check_systematic_shift()`` used -- pass the same
        values you used there, or the window won't match what the test
        actually saw. ``tech_col`` defaults to ``group_col`` (the results
        column is named after whatever ``tech_col`` was used, so this is
        usually correct without passing it explicitly).
    window_col : str, default ``'_shift_window'``
        Name of the temporary boolean column added to ``model.meta``.
        Restored to its previous value (or removed) when done.
    **plot_xy_kwargs
        Forwarded to ``model.plot_xy_data()`` (e.g. ``facet_by``, ``color_by``,
        ``color_palette``, ``log2fc``, ``show_correction``, ``legend_outside``,
        ``figsize``, ``mark_params``, ``show_hill_function``, ``sum_factor_col``).
        A ``subset_meta`` dict, if given, is merged with the window filter
        rather than overridden by it.

    Returns
    -------
    dict[str, plt.Figure or plt.Axes]
        ``feature -> plot_xy_data()`` return value, in plotting order.

    Raises
    ------
    ValueError
        If there are no features to plot (``features=[]``, or no rows in
        ``res`` pass the ``ok``/``p_col < fdr_threshold`` filter), or if no
        cells fall within any group's matched window (a strong sign
        ``tech_col``/``target_col``/``ntc_label``/``targeted_label`` don't
        match what ``check_systematic_shift()`` was actually called with).
    RuntimeError, ValueError
        Propagated from ``model.get_shift_window_cells()`` if ``x_true``
        isn't set or ``targeted_label``/``self.cis_gene`` are both None.
    """
    group_col = _detect_group_col(res, group_col)
    if tech_col is None:
        tech_col = group_col

    if features is None:
        hits = res[res["ok"] & (res[p_col] < fdr_threshold)]
        features = list(pd.unique(hits[label_col]))
    if not features:
        raise ValueError("No features to plot (empty `features` / no rows passed the significance filter).")

    windows = model.get_shift_window_cells(
        tech_col=tech_col,
        target_col=target_col,
        ntc_label=ntc_label,
        targeted_label=targeted_label,
        exclude_cells=exclude_cells,
    )
    all_window_cells = set()
    for cells in windows.values():
        all_window_cells.update(cells)
    if not all_window_cells:
        raise ValueError(
            "No cells fall within any group's matched window; check tech_col/"
            "target_col/ntc_label/targeted_label match what check_systematic_shift() used."
        )

    user_subset_meta = dict(plot_xy_kwargs.pop("subset_meta", {}) or {})

    had_col = window_col in model.meta.columns
    prev_values = model.meta[window_col].copy() if had_col else None
    model.meta[window_col] = model.meta["cell"].isin(all_window_cells)

    figs = {}
    try:
        for feature in features:
            subset_meta = dict(user_subset_meta)
            subset_meta[window_col] = True
            figs[feature] = model.plot_xy_data(feature, subset_meta=subset_meta, **plot_xy_kwargs)
    finally:
        if had_col:
            model.meta[window_col] = prev_values
        else:
            del model.meta[window_col]

    return figs
