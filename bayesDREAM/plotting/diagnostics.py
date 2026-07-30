"""
Diagnostic plotting functions.

Provides quality control and diagnostic plots for bayesDREAM fits.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr, spearmanr

from .colors import ColorScheme


# Collapse the additive-Hill `classification` column (see
# `ModelSummarizer.export_trans_summary` / `_classify_additive_hill` in
# io/summary.py) into the 3 directions of effect a viewer typically wants to
# facet/color trans hits by. Includes 'non_monotonic' -- the generic fallback
# _classify_additive_hill returns (summary.py's non_monotonic branch) when it
# detects opposite-sign active components but the numeric root-finder couldn't
# locate the extremum over the observed x-range, so no _min/_max could be
# assigned. Missing this key previously caused those rows to map to NaN and
# get silently dropped from the counts.
_TRANS_CLASSIFICATION_TO_DIRECTION = {
    'single_positive':   'positive',
    'additive_positive': 'positive',
    'single_negative':   'negative',
    'additive_negative': 'negative',
    'non_monotonic_min': 'non_monotonic',
    'non_monotonic_max': 'non_monotonic',
    'non_monotonic':     'non_monotonic',
    'flat':              None,  # not a dependent hit -- dropped
}
_TRANS_DIRECTION_ORDER = ['positive', 'negative', 'non_monotonic']
_TRANS_DIRECTION_COLORS = {
    'positive':      '#2a78d6',
    'negative':      '#eb6834',
    'non_monotonic': '#1baf7a',
}

# Full classification (all non-'flat' values of `classification`), for callers
# who want single_hill vs additive_hill distinguished rather than collapsed
# into direction. Ordered/colored so same-direction pairs (single vs additive)
# share a warm/cool family: blue/aqua = positive, orange/yellow = negative,
# magenta/green/violet = non-monotonic (incl. the unresolved-extremum fallback).
_TRANS_CLASSIFICATION_ORDER = [
    'single_positive', 'additive_positive',
    'single_negative', 'additive_negative',
    'non_monotonic_min', 'non_monotonic_max', 'non_monotonic',
]
_TRANS_CLASSIFICATION_COLORS = {
    'single_positive':   '#2a78d6',
    'additive_positive': '#1baf7a',
    'single_negative':   '#eb6834',
    'additive_negative': '#eda100',
    'non_monotonic_min': '#e87ba4',
    'non_monotonic_max': '#008300',
    'non_monotonic':     '#4a3aa7',
}

# Base column names (before the _median/_lower/_upper/_log2fc suffix) that are
# specific to one Hill component of an additive_hill fit -- see the "For
# additive_hill" column list in ModelSummarizer.export_trans_summary's
# docstring (io/summary.py). A feature classified 'single_positive'/
# 'single_negative' has only ONE of these two components active (the other's
# alpha/beta absorbed a constant offset while its Hill exponent is
# unidentified -- the exact degenerate case `is_dependent`/`classification`
# guard against, see the fdr_threshold docstring fix). which_active ('a', 'b',
# or 'both') says which. single_hill fit_type columns (B, K, xc, inflection --
# no _a/_b suffix) aren't in this registry since they have only one component
# and no analogous ambiguity.
_COMPONENT_A_BASE_COLS = ['alpha', 'Vmax_a', 'K_a', 'EC50_a', 'n_a', 'inflection_a']
_COMPONENT_B_BASE_COLS = ['beta', 'Vmax_b', 'K_b', 'EC50_b', 'n_b', 'inflection_b']


def _component_for_value_col(value_col):
    """Return 'a'/'b' if value_col is a component-specific additive_hill
    parameter (see _COMPONENT_*_BASE_COLS), else None."""
    for base in _COMPONENT_A_BASE_COLS:
        if value_col == base or value_col.startswith(base + '_'):
            return 'a'
    for base in _COMPONENT_B_BASE_COLS:
        if value_col == base or value_col.startswith(base + '_'):
            return 'b'
    return None


def _prepare_trans_group_data(fdr_df, gene_col, classification_col, dependent_col,
                                color_by, require_dependent, colors):
    """
    Shared setup for plot_trans_hits_by_gene / plot_trans_values_by_gene:
    validate columns, resolve the direction-vs-classification grouping, and
    apply the is_dependent / classification filters.

    Returns
    -------
    d : pd.DataFrame
        Filtered rows (subset of fdr_df, index preserved).
    group : pd.Series
        Per-row group label (direction or classification), aligned to d.index.
    group_order : list of str
    color_map : dict
    legend_title : str
    """
    if classification_col not in fdr_df.columns:
        raise ValueError(
            f"'{classification_col}' not found in fdr_df. This column is only "
            "populated for function_type='additive_hill' trans fits."
        )
    if gene_col not in fdr_df.columns:
        raise ValueError(f"'{gene_col}' not found in fdr_df; pass gene_col= explicitly.")

    if color_by == "direction":
        group_map = _TRANS_CLASSIFICATION_TO_DIRECTION
        group_order = _TRANS_DIRECTION_ORDER
        color_map = dict(_TRANS_DIRECTION_COLORS)
        legend_title = "Direction"
    elif color_by == "classification":
        group_map = {c: c for c in _TRANS_CLASSIFICATION_ORDER}
        group_order = _TRANS_CLASSIFICATION_ORDER
        color_map = dict(_TRANS_CLASSIFICATION_COLORS)
        legend_title = "Classification"
    else:
        raise ValueError(f"color_by must be 'direction' or 'classification', got {color_by!r}")
    if colors:
        color_map.update(colors)

    d = fdr_df
    if require_dependent:
        if dependent_col not in d.columns:
            raise ValueError(
                f"'{dependent_col}' not found in fdr_df; pass require_dependent=False "
                "to skip the significance filter."
            )
        d = d[d[dependent_col] == True]  # noqa: E712 -- may be object/NaN dtype

    group = d[classification_col].map(group_map)
    d = d.loc[group.notna() & d[gene_col].notna()]
    group = group.loc[d.index]

    return d, group, group_order, color_map, legend_title


def plot_trans_hits_by_gene(
    fdr_df,
    gene_col="gene",
    classification_col="classification",
    dependent_col="is_dependent",
    color_by="direction",
    require_dependent=True,
    top_n=None,
    colors=None,
    cis_gene=None,
    figsize=None,
    show=True,
):
    """
    Stacked bar chart of the number of trans-dependent features per gene,
    colored by shape of the dose-response curve.

    Meant for feature-level modalities where several features can map to the
    same gene (e.g. ``splicing_sj``/``splicing_donor`` -- several SJs per
    gene, or ``transcript`` -- several transcripts per gene). Groups
    ``fdr_df`` by ``gene_col`` and draws one stacked bar per gene.

    Parameters
    ----------
    fdr_df : pd.DataFrame
        Trans summary dataframe, e.g. the output of
        ``model.export_trans_summary()`` / ``save_trans_summary``. Must have
        ``classification_col`` (only produced for ``function_type='additive_hill'``)
        and ``gene_col``.
    gene_col : str, default ``'gene'``
        Column giving the gene each row (feature) belongs to. For splicing
        modalities this is aliased from ``gene_for_denominator`` by the
        splicing loader (see ``docs/SPLICING_LOADER_GUIDE.md``).
    classification_col : str, default ``'classification'``
        Column with the per-feature shape classification (``'single_positive'``,
        ``'additive_negative'``, ``'non_monotonic_min'``, ``'flat'``, ...).
    dependent_col : str, default ``'is_dependent'``
        Column flagging FDR-significant, non-degenerate dose-response features.
    color_by : {'direction', 'classification'}, default ``'direction'``
        ``'direction'`` collapses ``classification_col`` into 3 groups --
        ``positive`` (``single_positive``/``additive_positive``), ``negative``
        (``single_negative``/``additive_negative``), ``non_monotonic``
        (``non_monotonic_min``/``non_monotonic_max``).
        ``'classification'`` keeps all 6 non-``'flat'`` values distinct, so
        single-Hill and additive-Hill features are shown separately within
        each direction.
    require_dependent : bool, default True
        If True, restrict to rows with ``dependent_col == True`` before
        counting (dropping ``'flat'``/non-significant rows in addition to the
        ``classification_col`` mapping already excluding ``'flat'``).
    top_n : int, optional
        Only show the ``top_n`` genes with the most trans-dependent features
        (sorted descending by total). Default: show all genes.
    colors : dict, optional
        Override the default color mapping. Keys are ``'positive'``,
        ``'negative'``, ``'non_monotonic'`` when ``color_by='direction'``, or
        the classification values when ``color_by='classification'``.
    cis_gene : str, optional
        Name of the cis gene these trans features were fit against (e.g.
        ``model.cis_gene``). Used only to make the y-axis label explicit
        about what "dependent" means here -- every row in a trans summary is
        a feature whose response was modeled as a function of this one cis
        gene, so a bar's height is "number of features dependent on
        ``cis_gene``", not some property intrinsic to the x-axis gene itself.
        If omitted, the label falls back to generic wording.
    figsize : tuple, optional
        Figure size. Defaults to ``(max(6, 0.4 * n_genes), 5)``.
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``classification_col`` or ``gene_col`` is missing from ``fdr_df``
        (e.g. ``fdr_df`` was exported with ``function_type='polynomial'``,
        which has no Hill components to classify), if ``color_by`` is not one
        of ``'direction'``/``'classification'``, or if no rows remain after
        filtering.
    """
    d, group, group_order, color_map, legend_title = _prepare_trans_group_data(
        fdr_df, gene_col, classification_col, dependent_col,
        color_by, require_dependent, colors
    )

    if len(d) == 0:
        raise ValueError("No rows left to plot after filtering (require_dependent / classification).")

    counts = (
        pd.crosstab(d[gene_col], group)
        .reindex(columns=group_order, fill_value=0)
    )
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    if top_n is not None:
        counts = counts.iloc[:top_n]

    if figsize is None:
        figsize = (max(6, 0.4 * len(counts)), 5)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for group_name in group_order:
        vals = counts[group_name].values
        ax.bar(x, vals, bottom=bottom, width=0.7, color=color_map[group_name],
               label=group_name.replace('_', ' '), edgecolor='white', linewidth=1)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=90, fontsize=8)
    ylabel = (f"Number of features dependent on {cis_gene}" if cis_gene
              else "Number of features dependent on cis gene")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(gene_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, title=legend_title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_trans_values_by_gene(
    fdr_df,
    value_col,
    gene_col="gene",
    classification_col="classification",
    dependent_col="is_dependent",
    color_by="direction",
    require_dependent=True,
    top_n=None,
    gene_order=None,
    colors=None,
    exclude_inactive_component=True,
    active_col="which_active",
    jitter=0.3,
    point_size=20,
    alpha=0.8,
    show_zero_line=True,
    show_gene_bands=True,
    gene_band_color="#000000",
    gene_band_alpha=0.06,
    figsize=None,
    show=True,
):
    """
    Dot (strip) plot of a per-feature value column, grouped by gene along the
    x-axis and colored by direction/classification.

    Same gene grouping and direction/classification coloring as
    ``plot_trans_hits_by_gene``, but instead of counting hits, plots one point
    per feature at its actual value of ``value_col`` -- e.g.
    ``'observed_delta_p_median'``, ``'full_delta_p_median'``, ``'n_a_median'``,
    ``'n_b_median'``, ``'EC50_a_median'``, ``'EC50_b_median'``.

    Parameters
    ----------
    fdr_df : pd.DataFrame
        Trans summary dataframe, e.g. the output of
        ``model.export_trans_summary()`` / ``save_trans_summary``. Must have
        ``classification_col``, ``gene_col``, and ``value_col``.
    value_col : str
        Column in ``fdr_df`` to plot on the y-axis. Rows with a missing/NaN
        value are dropped.
    gene_col, classification_col, dependent_col, color_by, require_dependent, colors :
        See ``plot_trans_hits_by_gene``.
    exclude_inactive_component : bool, default True
        ``value_col`` columns specific to one Hill component (``alpha``,
        ``Vmax_a``/``Vmax_b``, ``K_a``/``K_b``, ``EC50_a``/``EC50_b``,
        ``n_a``/``n_b``, ``inflection_a``/``inflection_b``, and their
        ``_median``/``_lower``/``_upper``/``_log2fc`` variants) are meaningless
        for a feature whose ``which_active`` doesn't include that component --
        e.g. ``n_a_median`` for a feature classified ``single_negative`` via
        component B, where component A's alpha absorbed a constant offset
        while ``n_a`` is completely unidentified. When True (default) and
        ``value_col`` is recognized as component-specific, rows where the
        corresponding component isn't active (per ``active_col``) are dropped
        before plotting. Has no effect on component-agnostic columns (e.g.
        ``observed_delta_p_median``, ``A_median``, ``full_delta_p_median``).
    active_col : str, default ``'which_active'``
        Column naming which component(s) are active per feature (``'a'``,
        ``'b'``, or ``'both'``). Only consulted when
        ``exclude_inactive_component=True`` and ``value_col`` is
        component-specific; if missing from ``fdr_df`` in that case, a warning
        is issued and the extra filtering is skipped.
    top_n : int, optional
        Only show the ``top_n`` genes with the most dependent features
        (sorted descending by count, same criterion as
        ``plot_trans_hits_by_gene``). Ignored if ``gene_order`` is given.
    gene_order : list of str, optional
        Explicit x-axis gene order -- e.g. reuse ``counts.index.tolist()``
        from a prior ``plot_trans_hits_by_gene()`` call so the two plots line
        up side by side. Genes not in ``gene_order`` are dropped from the data;
        genes in ``gene_order`` with no surviving rows appear as empty columns.
    jitter : float, default 0.3
        Half-width of horizontal jitter applied within each gene's column
        (uniform, seeded for reproducibility) to separate overlapping points.
    point_size : float, default 20
        Marker size (``ax.scatter``'s ``s``).
    alpha : float, default 0.8
        Marker transparency.
    show_zero_line : bool, default True
        Draw a grey dotted horizontal line at y=0. Meaningful for signed
        columns (``*_delta_p_*``, ``n_a``/``n_b``, log2fc columns); harmless
        otherwise -- set False if it's not a useful reference for your column.
    show_gene_bands : bool, default True
        Shade every other gene's column with a faint background band (zebra
        striping), so it's clear which gene a jittered point belongs to even
        with many genes packed tightly on the x-axis. Purely visual --
        doesn't affect ``gene_order``/alternation, just alternates starting
        from the first gene.
    gene_band_color : str, default ``'#000000'``
        Fill color for the shaded bands (only visible via ``gene_band_alpha``).
    gene_band_alpha : float, default 0.06
        Opacity of the shaded bands. Keep low -- these are a background
        reference, not data ink.
    figsize : tuple, optional
        Figure size. Defaults to ``(max(6, 0.4 * n_genes), 5)``.
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``classification_col``/``gene_col``/``value_col`` is missing from
        ``fdr_df``, if ``color_by`` is invalid, or if no rows remain after
        filtering.
    """
    if value_col not in fdr_df.columns:
        raise ValueError(f"'{value_col}' not found in fdr_df.")

    d, group, group_order, color_map, legend_title = _prepare_trans_group_data(
        fdr_df, gene_col, classification_col, dependent_col,
        color_by, require_dependent, colors
    )

    if exclude_inactive_component:
        component = _component_for_value_col(value_col)
        if component is not None:
            if active_col in d.columns:
                d = d.loc[d[active_col].isin([component, 'both'])]
                group = group.loc[d.index]
            else:
                warnings.warn(
                    f"'{value_col}' looks like a component-{component} parameter, but "
                    f"'{active_col}' is not in fdr_df -- cannot exclude rows where "
                    "that component is inactive (values may include degenerate/"
                    "meaningless fits for the inactive component). Add "
                    f"'{active_col}' (from save_trans_summary) or pass "
                    "exclude_inactive_component=False to silence this warning."
                )

    d = d.loc[d[value_col].notna()]
    group = group.loc[d.index]

    if len(d) == 0:
        raise ValueError("No rows left to plot after filtering (require_dependent / classification / value_col).")

    if gene_order is None:
        gene_order_list = (
            d[gene_col].value_counts().sort_values(ascending=False).index.tolist()
        )
        if top_n is not None:
            gene_order_list = gene_order_list[:top_n]
    else:
        gene_order_list = list(gene_order)

    d = d.loc[d[gene_col].isin(gene_order_list)]
    group = group.loc[d.index]

    if len(d) == 0:
        raise ValueError("No rows left to plot for the requested genes (check gene_order/top_n).")

    gene_pos = {g: i for i, g in enumerate(gene_order_list)}

    if figsize is None:
        figsize = (max(6, 0.4 * len(gene_order_list)), 5)
    fig, ax = plt.subplots(figsize=figsize)

    if show_gene_bands:
        for i in range(1, len(gene_order_list), 2):
            ax.axvspan(i - 0.5, i + 0.5, color=gene_band_color,
                       alpha=gene_band_alpha, zorder=0, linewidth=0)

    rng = np.random.RandomState(0)
    base_x = d[gene_col].map(gene_pos).values.astype(float)
    jittered_x = base_x + rng.uniform(-jitter, jitter, size=len(d))
    values = d[value_col].values

    for group_name in group_order:
        mask = (group == group_name).values
        if not mask.any():
            continue
        ax.scatter(jittered_x[mask], values[mask], s=point_size, alpha=alpha,
                   color=color_map[group_name], label=group_name.replace('_', ' '),
                   edgecolor='none', zorder=3)

    if show_zero_line:
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    ax.set_xticks(range(len(gene_order_list)))
    ax.set_xticklabels(gene_order_list, rotation=90, fontsize=8)
    ax.set_xlim(-0.5, len(gene_order_list) - 0.5)
    ax.set_ylabel(value_col)
    ax.set_xlabel(gene_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, title=legend_title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


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
    highlight_features=None,
    highlight_color="tab:blue",
    highlight_label_all=False,
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
        Number of most-significant (lowest ``p_col``) features to label per
        panel. Without ``highlight_features``, this is the top_n among rows
        below ``fdr_threshold``. With ``highlight_features``, this is instead
        the top_n most significant *of the highlighted matches* (regardless
        of ``fdr_threshold``) -- e.g. ``top_n=5`` labels only the 5 most
        significant heat shock proteins. Ignored if ``highlight_label_all=True``.
    highlight_features : list of str, optional
        Feature names (matched against ``label_col``) to highlight, e.g. a
        curated gene set such as heat shock proteins. When given, every
        matching row in each panel is drawn with an open ``highlight_color``
        ring (regardless of significance, so you can see the whole set), and
        the ``top_n`` most significant of them are text-labeled (see
        ``top_n`` and ``highlight_label_all``).
    highlight_color : str, default ``'tab:blue'``
        Ring color used to mark ``highlight_features`` points.
    highlight_label_all : bool, default ``False``
        If True, text-label every ``highlight_features`` match instead of
        just the ``top_n`` most significant. Only meaningful when
        ``highlight_features`` is given -- expect heavy overlap if the set
        is large and clustered.
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

        if highlight_features is not None:
            highlighted = sub[sub[label_col].isin(set(highlight_features))]
            if not highlighted.empty:
                ax.scatter(
                    highlighted[effect_col], highlighted["_neg_log10_p"],
                    s=50, facecolors="none", edgecolors=highlight_color,
                    linewidths=1.5, zorder=5, label="Highlighted",
                )
            # By default, only text-label the top_n most significant (lowest
            # p_col) of the highlighted matches -- labeling all of them
            # (often dozens clustered near p_adj~1) produces an unreadable
            # pile of overlapping text. Set highlight_label_all=True to
            # label every match regardless of significance.
            to_label = highlighted if highlight_label_all else highlighted.nsmallest(top_n, p_col)
        else:
            to_label = sig.nsmallest(top_n, p_col)

        texts = []
        orig_xy = []
        x_span = sub[effect_col].max() - sub[effect_col].min()
        y_span = sub["_neg_log10_p"].max() - sub["_neg_log10_p"].min()
        dx0 = 0.01 * x_span if x_span > 0 else 0.01
        dy0 = 0.02 * y_span if y_span > 0 else 0.02
        for _, row in to_label.iterrows():
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
        unique values among ``ok == True`` rows -- if this is because one or
        more groups have zero usable rows (e.g. every feature was skipped
        for ``too_few_cells_after_subsetting``), the error message includes
        a breakdown of skip reasons for the empty group(s); if the inner
        join on ``label_col`` between the two groups produces no shared,
        valid features; or if ``highlight`` isn't one of ``'either'``,
        ``'both'``, ``'none'``.

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
            all_groups = sorted(res[group_col].unique())
            empty_groups = sorted(set(all_groups) - set(unique_groups))
            if empty_groups:
                lines = [
                    f"Group(s) {empty_groups} have zero rows with ok=True in `res` -- "
                    f"nothing to correlate for {'it' if len(empty_groups) == 1 else 'them'}. "
                    f"Groups with usable rows: {unique_groups}."
                ]
                for g in empty_groups:
                    reason_counts = res.loc[res[group_col] == g, "reason"].value_counts()
                    if not reason_counts.empty:
                        lines.append(f"  Skip reasons for group {g!r}:")
                        lines.extend(f"    {r}: {c}" for r, c in reason_counts.items())
                lines.append(
                    "This usually means min_cells_per_group (in check_systematic_shift()) "
                    "was too high for this group's cell counts after x_true-window matching. "
                    "Consider lowering min_cells_per_group and re-running, or pass "
                    f"groups={tuple(unique_groups)} explicitly if you only meant to plot those."
                )
                raise ValueError("\n".join(lines))
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


def plot_cross_dataset_correlation(
    res_a,
    res_b,
    value_col="shift_est",
    group_col_a=None,
    group_col_b=None,
    label_col="feature",
    p_col="p_adj",
    transform=None,
    fdr_threshold=0.1,
    highlight="either",
    top_n=0,
    dataset_names=("Dataset A", "Dataset B"),
    sig_color="firebrick",
    sig_color_a="tab:blue",
    sig_color_b="tab:orange",
    nonsig_color="#999999",
    label_fontsize=7,
    figsize=None,
    show=True,
):
    """
    Grid of scatter plots correlating ``value_col`` between two independently
    produced ``check_systematic_shift()`` result tables (e.g. from different
    datasets/experiments), matched by feature.

    Columns = technical groups in ``res_a``; rows = technical groups in
    ``res_b`` -- i.e. every (group_a, group_b) combination gets its own
    panel (a full cross-tabulation, not a 1:1 group match, since group codes
    from two independently fit models aren't assumed to correspond). Each
    panel inner-joins ``res_a`` and ``res_b`` on ``label_col`` for that pair
    of groups and scatters ``res_a[value_col]`` (x) against
    ``res_b[value_col]`` (y).

    Call this once with ``value_col='shift_est'`` and once with
    ``value_col='p_adj'`` (or ``'shift_p_adj'``) to get the shift-estimate
    and p-value correlation grids.

    Parameters
    ----------
    res_a, res_b : pd.DataFrame
        Two ``check_systematic_shift()`` outputs to compare. Must share at
        least some ``label_col`` values within a given (group_a, group_b)
        pair for that panel to be non-empty.
    value_col : str, default ``'shift_est'``
        Column to correlate. Known p-value columns (``'p_adj'``,
        ``'shift_p_adj'``, ``'pval'``, ``'p_lrt'``, ``'shift_p'``) are
        ``-log10``-transformed automatically unless ``transform`` overrides
        this.
    group_col_a, group_col_b : str, optional
        Grouping column in ``res_a`` / ``res_b`` respectively. Auto-detected
        independently for each dataframe (same logic as
        ``plot_systematic_shift_volcano``), so they need not share a name.
    label_col : str, default ``'feature'``
        Column used to match rows across the two dataframes within a panel.
    p_col : str, default ``'p_adj'``
        Column name (assumed shared by both dataframes) used for
        significance highlighting, regardless of what ``value_col`` is.
    transform : {None, '-log10'}, optional
        Force the axis transform. If None (default), auto-detected from
        ``value_col`` (see above).
    fdr_threshold : float, default 0.1
        Significance cutoff applied to ``p_col`` in both dataframes.
    highlight : {'either', 'both', 'none'}, default ``'either'``
        - ``'either'``: color points significant in dataset A only, dataset B
          only, and both, each with a distinct color (``sig_color_a``,
          ``sig_color_b``, ``sig_color``) -- so you can see whether hits
          replicate or are dataset-specific.
        - ``'both'``: single-color highlight (``sig_color``) for points
          significant in both datasets; everything else grey.
        - ``'none'``: no significance-based coloring.
    top_n : int, default 0
        Label the top N highlighted points per panel (any of the
        significant categories under ``highlight='either'``) by ``min(p_col)``
        across the two datasets. 0 = no labels.
    dataset_names : tuple of 2 str, default ``('Dataset A', 'Dataset B')``
        Names used in the shared x/y axis labels and in the "significant in
        <name> only" legend entries.
    sig_color : str, default ``'firebrick'``
        Color for points significant in both datasets.
    sig_color_a, sig_color_b : str
        Colors for points significant in dataset A only / dataset B only
        (only used when ``highlight='either'``).
    nonsig_color : str
        Color for non-significant (or, if ``highlight='none'``, all) points.
    label_fontsize : int, default 7
    figsize : tuple, optional
        Defaults to ``(4 * n_groups_a, 4 * n_groups_b)``.
    show : bool, default True

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``highlight`` isn't one of ``'either'``, ``'both'``, ``'none'``;
        or if either dataframe has zero groups with usable (``ok == True``,
        finite ``value_col``/``p_col``) rows.

    Notes
    -----
    Panels with no shared features between the two datasets for that
    (group_a, group_b) pair are left blank with an annotation rather than
    raising -- expected when the two group grids don't align 1:1.

    Each panel draws a dashed reference crosshair (both axes) at the null
    value for whatever's plotted: 0 for effect-size columns (gray dotted),
    or the FDR threshold on the -log10 scale for p-value columns
    (``sig_color`` dashed, matching ``plot_systematic_shift_volcano``'s
    threshold line).

    Examples
    --------
    >>> fig_shift = plot_cross_dataset_correlation(
    ...     res, res_Domingo, value_col='shift_est',
    ...     dataset_names=('This run', 'Domingo'))
    >>> fig_p = plot_cross_dataset_correlation(
    ...     res, res_Domingo, value_col='p_adj',
    ...     dataset_names=('This run', 'Domingo'))
    """
    if highlight not in ("either", "both", "none"):
        raise ValueError("highlight must be 'either', 'both', or 'none'.")

    group_col_a = _detect_group_col(res_a, group_col_a)
    group_col_b = _detect_group_col(res_b, group_col_b)

    p_known = {"p_adj", "shift_p_adj", "pval", "p_lrt", "shift_p"}
    do_log = (value_col in p_known) if transform is None else (transform == "-log10")

    ok_a = res_a[res_a["ok"] & np.isfinite(res_a[value_col]) & np.isfinite(res_a[p_col])]
    ok_b = res_b[res_b["ok"] & np.isfinite(res_b[value_col]) & np.isfinite(res_b[p_col])]

    groups_a = sorted(ok_a[group_col_a].unique())
    groups_b = sorted(ok_b[group_col_b].unique())
    if not groups_a or not groups_b:
        raise ValueError(
            f"No usable rows (ok=True, finite {value_col}/{p_col}) in "
            f"{'res_a' if not groups_a else 'res_b'}."
        )

    n_cols, n_rows = len(groups_a), len(groups_b)
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    axis_label = f"$-\\log_{{10}}$({value_col})" if do_log else value_col
    tiny = np.finfo(float).tiny

    # Build legend proxies up front (rather than harvesting handles from
    # whichever panel happens to be plotted first) so the legend always
    # shows every category ``highlight`` can produce, even if some category
    # is empty in every panel.
    def _dot(color, alpha, label):
        return mlines.Line2D([], [], marker="o", linestyle="None",
                              color=color, alpha=alpha, markersize=6, label=label)
    if highlight == "none":
        legend_elements = [_dot(nonsig_color, 0.4, "Data")]
    elif highlight == "both":
        legend_elements = [
            _dot(nonsig_color, 0.4, "Not significant"),
            _dot(sig_color, 0.9, f"{_P_COL_LABELS.get(p_col, p_col)} < {fdr_threshold} (both)"),
        ]
    else:  # either
        legend_elements = [
            _dot(nonsig_color, 0.4, "Not significant"),
            _dot(sig_color_a, 0.8, f"Significant in {dataset_names[0]} only"),
            _dot(sig_color_b, 0.8, f"Significant in {dataset_names[1]} only"),
            _dot(sig_color, 0.9, "Significant in both"),
        ]

    # Avoid selecting the same column twice (e.g. value_col == p_col == 'p_adj'
    # for the p-value correlation call) -- that would duplicate the column
    # name within a single dataframe, before merge suffixing even applies.
    cols = [label_col, value_col] + ([p_col] if p_col != value_col else [])

    for i, gb in enumerate(groups_b):
        db_full = ok_b[ok_b[group_col_b] == gb][cols]
        for j, ga in enumerate(groups_a):
            ax = axes[i, j]
            da_full = ok_a[ok_a[group_col_a] == ga][cols]
            merged = pd.merge(da_full, db_full, on=label_col, suffixes=("_a", "_b"))

            if merged.empty:
                ax.text(
                    0.5, 0.5, "no shared\nfeatures", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if p_col == value_col:
                    # Both columns are the same underlying values (merge only
                    # suffixed them once) -- no separate "_a"/"_b" p_col pair.
                    x_raw = merged[f"{value_col}_a"].to_numpy()
                    y_raw = merged[f"{value_col}_b"].to_numpy()
                    pa, pb = x_raw, y_raw
                else:
                    x_raw = merged[f"{value_col}_a"].to_numpy()
                    y_raw = merged[f"{value_col}_b"].to_numpy()
                    pa = merged[f"{p_col}_a"].to_numpy()
                    pb = merged[f"{p_col}_b"].to_numpy()
                if do_log:
                    x = -np.log10(np.clip(x_raw, tiny, None))
                    y = -np.log10(np.clip(y_raw, tiny, None))
                else:
                    x, y = x_raw, y_raw

                sig_a = pa < fdr_threshold
                sig_b = pb < fdr_threshold

                if highlight == "either":
                    cat_both = sig_a & sig_b
                    cat_a_only = sig_a & ~sig_b
                    cat_b_only = ~sig_a & sig_b
                elif highlight == "both":
                    cat_both = sig_a & sig_b
                    cat_a_only = np.zeros(len(merged), dtype=bool)
                    cat_b_only = np.zeros(len(merged), dtype=bool)
                else:
                    cat_both = np.zeros(len(merged), dtype=bool)
                    cat_a_only = np.zeros(len(merged), dtype=bool)
                    cat_b_only = np.zeros(len(merged), dtype=bool)
                sig_mask = cat_both | cat_a_only | cat_b_only
                cat_none = ~sig_mask

                # Reference line: null effect (0) for effect-size columns,
                # or the FDR threshold (on the -log10 scale) for p-value columns.
                if do_log:
                    ref = -np.log10(fdr_threshold)
                    ref_color, ref_style = sig_color, "--"
                else:
                    ref = 0.0
                    ref_color, ref_style = "gray", ":"
                ax.axhline(ref, linestyle=ref_style, color=ref_color, linewidth=1, zorder=1)
                ax.axvline(ref, linestyle=ref_style, color=ref_color, linewidth=1, zorder=1)

                ax.scatter(x[cat_none], y[cat_none], s=10, alpha=0.4,
                           color=nonsig_color, zorder=2)
                if cat_a_only.any():
                    ax.scatter(x[cat_a_only], y[cat_a_only], s=14, alpha=0.8,
                               color=sig_color_a, zorder=3)
                if cat_b_only.any():
                    ax.scatter(x[cat_b_only], y[cat_b_only], s=14, alpha=0.8,
                               color=sig_color_b, zorder=3)
                if cat_both.any():
                    ax.scatter(x[cat_both], y[cat_both], s=16, alpha=0.9,
                               color=sig_color, zorder=4)

                if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                    r, _ = pearsonr(x, y)
                    rho, _ = spearmanr(x, y)
                    stats_txt = f"r={r:.2f}\nρ={rho:.2f}\nn={len(merged):,}"
                else:
                    stats_txt = f"n={len(merged):,}"
                ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                        va="top", ha="left", fontsize=7)

                if top_n > 0 and sig_mask.any():
                    min_p = np.minimum(pa, pb)
                    sig_idx = np.flatnonzero(sig_mask)
                    top_idx = sig_idx[np.argsort(min_p[sig_idx])][:top_n]
                    texts, orig_xy = [], []
                    x_span = x.max() - x.min()
                    y_span = y.max() - y.min()
                    dx0 = 0.01 * x_span if x_span > 0 else 0.01
                    dy0 = 0.02 * y_span if y_span > 0 else 0.02
                    for k in top_idx:
                        xi, yi = x[k], y[k]
                        txt = ax.text(
                            xi + dx0, yi + dy0, str(merged[label_col].iloc[k]),
                            fontsize=label_fontsize, zorder=4,
                        )
                        txt.set_path_effects(
                            [pe.Stroke(linewidth=2, foreground="white"), pe.Normal()]
                        )
                        texts.append(txt)
                        orig_xy.append((xi, yi))
                    try:
                        from adjustText import adjust_text
                        adjust_text(
                            texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                        )
                    except ImportError:
                        _declutter_texts(fig, ax, texts)
                        _draw_leader_lines(ax, texts, orig_xy)

            if i == 0:
                ax.set_title(f"{group_col_a}={ga}")
            if i == n_rows - 1:
                ax.set_xlabel(f"{dataset_names[0]}\n{axis_label}")
            if j == 0:
                ax.set_ylabel(f"{dataset_names[1]}\n{axis_label}")
            if j == n_cols - 1:
                ax.annotate(
                    f"{group_col_b}={gb}", xy=(1.05, 0.5), xycoords="axes fraction",
                    rotation=270, va="center", ha="left", fontsize=10, fontweight="bold",
                )

    fig.suptitle(
        f"{value_col} correlation: {dataset_names[0]} (cols) vs {dataset_names[1]} (rows)",
        y=0.99,
    )
    fig.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=len(legend_elements), frameon=False, fontsize=8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
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
