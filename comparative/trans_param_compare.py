"""
Transcriptome-wide fitted-parameter comparisons between two bayesDREAM runs.

Unlike dose_response_panels.py, this works directly off each run's
``trans_feature_summary_{modality}.csv`` (already written by
``save_trans_summary()`` during the normal fit_trans stage -- see
``publication_runs/common/run_trans.py``). No model reload is needed, so
this scales to Morris/Replogle's transcriptome-wide trans gene sets.

Quick start
-----------
    from comparative.datasets import MORRIS, REPLOGLE
    from comparative.trans_param_compare import compare_cis_gene

    merged, fig_obs, fig_grid = compare_cis_gene(MORRIS, REPLOGLE, 'GFI1B')

Or, for every cis gene shared between two datasets::

    from comparative.trans_param_compare import compare_all_shared_cis_genes
    results = compare_all_shared_cis_genes(MORRIS, REPLOGLE, out_dir='./param_comparison_plots')

Design notes
------------
- The join key is the dataset's own unique identifier whenever both sides
  have one -- 'gene_id' (Ensembl), preferred over 'gene_symbol' -- and
  falls back to 'gene_symbol' only when at least one side lacks a 'gene_id'
  (currently: any comparison involving Domingo, which carries no Ensembl
  mapping at all -- see merge_pair()). Comparisons against Domingo will
  therefore have noticeably fewer points than Morris-vs-Replogle, since
  Domingo's own trans panel is only ~91 genes to begin with.
- Parameter names differ slightly by function_type (additive_hill vs.
  single_hill) and across library versions (``_median`` vs ``_mean``
  suffixes). ``PARAM_ALIASES`` maps a stable "logical" parameter name to a
  list of candidate raw column names tried in order; whichever is present
  gets copied to a column named after the logical name, so downstream code
  never has to special-case function_type.
- Cross-dataset EC50/inflection comparisons use the "_log2fc" (relative to
  each dataset's own NTC baseline) variants, not raw x_true units -- the
  two datasets' cis gene expression scales are not directly comparable.
"""

import os
import re
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as scipy_stats

from .datasets import DatasetSpec


# ── Logical parameter name -> candidate raw column names (first match wins) ──
PARAM_ALIASES: Dict[str, List[str]] = {
    'observed_log2fc':        ['observed_log2fc'],
    'full_log2fc':             ['full_log2fc_median', 'full_log2fc_mean'],
    'EC50_a_log2fc':           ['EC50_a_log2fc', 'K_a_log2fc'],
    'EC50_b_log2fc':           ['EC50_b_log2fc', 'K_b_log2fc'],
    'n_a':                     ['n_a_median', 'n_a_mean'],
    'n_b':                     ['n_b_median', 'n_b_mean'],
    'Vmax_a':                  ['Vmax_a_median', 'Vmax_a_mean', 'B_median', 'B_mean'],
    'Vmax_b':                  ['Vmax_b_median', 'Vmax_b_mean'],
    'inflection_a_log2fc':     ['inflection_a_log2fc_median'],
    'y_ntc':                   ['y_ntc'],
    # From comparative/hill_eval.py's add_log2fc_at_columns() (x_log2fc=-1.0,
    # i.e. the fitted curve's y-log2FC at 50% cis-gene knockdown) -- only
    # present once comparative/reconstruct_export.py has backfilled a given
    # dataset's trans_feature_summary CSV. Prefers the "_allgenes" variant
    # (every gene, not NaN-masked by that dataset's OWN significance call)
    # since datasets differ a lot in power -- use is_dependent/highlight_col
    # to mark which points are independently significant instead of
    # dropping the rest.
    'y_log2fc_at_xm1':         ['y_at_x_log2fcm1_log2fc_median_allgenes', 'y_at_x_log2fcm1_log2fc_median'],
}

# Sensible default grid for "everything besides observed_log2fc" -- edit
# freely per call via the `params=` argument.
DEFAULT_GRID_PARAMS = ['full_log2fc', 'y_log2fc_at_xm1', 'EC50_a_log2fc', 'n_a', 'Vmax_a']


def _standardize_params(df: pd.DataFrame, aliases: Dict[str, List[str]] = PARAM_ALIASES) -> pd.DataFrame:
    """Add a canonical column for each logical parameter name that has a
    match among its candidate raw column names (first match wins). No-op
    (keeps existing column) if the logical name is already a column.
    """
    for logical, candidates in aliases.items():
        if logical in df.columns:
            continue
        for c in candidates:
            if c in df.columns:
                df[logical] = df[c]
                break
    return df


def load_trans_summary(spec: DatasetSpec, cis_gene: str, modality_name: Optional[str] = None) -> pd.DataFrame:
    """Load and standardize one dataset's trans_feature_summary_{modality}.csv
    for a given cis gene.

    Adds:
      - 'gene_symbol': a display/fallback join key (from spec.symbol_col)
      - 'gene_id': a stable Ensembl-style identifier, when available (see
        below) -- the PREFERRED cross-dataset join key (see merge_pair()),
        since 'gene_symbol' can be many-to-one within a single dataset.
      - canonical PARAM_ALIASES columns (see module docstring)
    Drops the cis gene's own row if present (is_cis_gene == True) -- it's
    not a trans feature and would otherwise show up as a spurious point.

    Ambiguous symbols (see below) aside, low_memory=False avoids pandas'
    chunked dtype inference spuriously flagging mixed-type columns on a
    dataset with this many columns/rows (harmless but noisy DtypeWarning).
    """
    path = spec.trans_summary_path(cis_gene, modality_name)
    df = pd.read_csv(path, low_memory=False)

    if spec.symbol_col not in df.columns:
        raise KeyError(
            f"[{spec.name}] symbol_col={spec.symbol_col!r} not found in {path!r} "
            f"(columns: {list(df.columns)[:15]}...)"
        )
    df['gene_symbol'] = df[spec.symbol_col].astype(str)

    # 'gene_id': a stable Ensembl-style id, if this dataset has one.
    #   - Replogle: spec.symbol_col != 'feature' means 'feature' ITSELF is
    #     that id (it's the Ensembl gene ID there, distinct from the symbol).
    #   - Morris (and any dataset saved after comparative/reconstruct_export.py
    #     backfills it): save_trans_summary() positionally attaches every
    #     column from modality.feature_meta, so a 'gene_id' column shows up
    #     for free whenever feature_meta carried one (Morris's does --
    #     see CLAUDE.md's High-MOI section) -- just look for it directly,
    #     no separate lookup file needed.
    #   - Domingo: neither applies. No Ensembl mapping exists for this
    #     dataset at all -- 'gene_id' stays null, and any merge involving
    #     it falls back to 'gene_symbol' (see merge_pair()).
    if spec.symbol_col != 'feature':
        df['gene_id'] = df['feature'].astype(str)
    elif 'gene_id' in df.columns:
        df['gene_id'] = df['gene_id'].astype(str)
    else:
        df['gene_id'] = np.nan

    # 'feature' is guaranteed unique within a dataset (save_trans_summary()
    # enforces this at fit time -- see bayesDREAM/io/summary.py). 'gene_symbol'
    # is NOT, whenever it's derived rather than being 'feature' itself: e.g.
    # Replogle is indexed by Ensembl gene ID ('feature'), and a handful of
    # genes (TBCE, HSPA14, ...) are annotated as two separate Ensembl IDs
    # sharing one symbol -- a known segmental-duplication artifact, not a
    # bug here. This only matters for a SYMBOL-based merge (Domingo, which
    # has no 'gene_id' to merge on instead) -- Morris/Replogle-vs-Replogle
    # comparisons merge on 'gene_id' and are unaffected by this at all (see
    # merge_pair()). Rather than silently pick one of the ambiguous rows
    # (wrong half the time) or let merge_pair's duplicate check crash the
    # whole comparison, drop just the ambiguous symbols here -- using the
    # dataset's OWN unique 'feature' id to detect them -- and keep everything
    # else.
    if spec.symbol_col != 'feature' and 'feature' in df.columns:
        dupe_symbols = df.loc[df['gene_symbol'].duplicated(keep=False), 'gene_symbol'].unique()
        if len(dupe_symbols):
            n_ambig = df['gene_symbol'].isin(dupe_symbols).sum()
            # Null out (not drop the row) -- gene_id-based merges still need
            # these rows; only a symbol-based merge (merge_pair's fallback
            # for datasets with no gene_id, e.g. Domingo) can't safely use them.
            df.loc[df['gene_symbol'].isin(dupe_symbols), 'gene_symbol'] = np.nan
            print(f"[{spec.name}] {len(dupe_symbols)} ambiguous gene_symbol value(s) map to >1 distinct "
                  f"'feature' id in this dataset (e.g. {sorted(dupe_symbols)[:5]}) -- {n_ambig} row(s) "
                  f"excluded from symbol-based cross-dataset comparison only (their gene_id, if any, "
                  f"is unaffected).")

    if 'is_cis_gene' in df.columns:
        df = df.loc[~df['is_cis_gene'].fillna(False).astype(bool)].copy()

    if 'is_dependent' in df.columns:
        df['is_dependent'] = df['is_dependent'].fillna(False).astype(bool)
    else:
        df['is_dependent'] = False

    df = _standardize_params(df)
    df['dataset'] = spec.name
    df['cis_gene'] = cis_gene
    return df


def merge_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str,
                on: Optional[str] = None) -> pd.DataFrame:
    """Inner-join two standardized trans summaries.

    on : {'gene_id', 'gene_symbol'}, optional
        Join key. If None (default), uses 'gene_id' (Ensembl) when BOTH
        sides have at least one non-null value there, else falls back to
        'gene_symbol'. 'gene_id' is preferred because 'gene_symbol' can be
        many-to-one within a single dataset (see load_trans_summary());
        'gene_id' avoids that entirely for any pair where both sides have
        one (today: any pair not involving Domingo).

    Rows with a null value in the chosen `on` column are excluded from
    THIS merge only (e.g. Domingo's rows, when merging on 'gene_id'; or an
    ambiguous-symbol row, when merging on 'gene_symbol' -- see
    load_trans_summary()) -- not from the original DataFrames.

    Shared column names (parameters, 'is_dependent', etc.) get suffixed
    ``_{name_a}`` / ``_{name_b}``; the join key itself is not suffixed. If
    joined on 'gene_id', a single display 'gene_symbol' column is attached
    afterward (preferring whichever side's symbol doesn't look
    scanpy-deduplicated, e.g. 'TBCE-1' -- see bayesDREAM/modality.py).
    """
    if on is None:
        has_id_a = 'gene_id' in df_a.columns and df_a['gene_id'].notna().any()
        has_id_b = 'gene_id' in df_b.columns and df_b['gene_id'].notna().any()
        on = 'gene_id' if (has_id_a and has_id_b) else 'gene_symbol'

    work_a = df_a.dropna(subset=[on])
    work_b = df_b.dropna(subset=[on])

    if work_a[on].duplicated().any():
        dupes = work_a.loc[work_a[on].duplicated(), on].unique()[:5]
        raise ValueError(f"{name_a}: duplicated {on} values, e.g. {list(dupes)} -- cannot merge safely.")
    if work_b[on].duplicated().any():
        dupes = work_b.loc[work_b[on].duplicated(), on].unique()[:5]
        raise ValueError(f"{name_b}: duplicated {on} values, e.g. {list(dupes)} -- cannot merge safely.")

    merged = work_a.merge(work_b, on=on, how='inner', suffixes=(f'_{name_a}', f'_{name_b}'))

    if on != 'gene_symbol':
        sym_a, sym_b = f'gene_symbol_{name_a}', f'gene_symbol_{name_b}'
        _clean = lambda s: not re.search(r'-\d+$', str(s))  # noqa: E731
        if sym_a in merged.columns and sym_b in merged.columns:
            merged['gene_symbol'] = [a if _clean(a) else b for a, b in zip(merged[sym_a], merged[sym_b])]
        elif sym_a in merged.columns:
            merged['gene_symbol'] = merged[sym_a]
        elif sym_b in merged.columns:
            merged['gene_symbol'] = merged[sym_b]

    return merged


def _col(param: str, dataset_name: str) -> str:
    return f'{param}_{dataset_name}'


def scatter_param(
    merged: pd.DataFrame,
    param: str,
    name_a: str,
    name_b: str,
    *,
    ax: Optional[plt.Axes] = None,
    color_by: Optional[str] = None,
    cmap: str = 'RdBu_r',
    vcenter: float = 0.0,
    highlight_col: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    s: float = 16,
    alpha: float = 0.7,
    diag: bool = True,
    annotate_corr: bool = True,
) -> plt.Axes:
    """Scatter one fitted parameter, dataset A (x) vs dataset B (y), across
    genes shared between the two.

    Parameters
    ----------
    param : str
        A logical parameter name from PARAM_ALIASES (or any column present
        in both, un-suffixed, e.g. 'observed_log2fc').
    color_by : str, optional
        A column in `merged` (already suffixed, e.g. 'observed_log2fc_Morris')
        to color points by. Points are drawn in order of increasing
        |color_by - vcenter|, so the strongest-effect genes are drawn last
        (on top) rather than buried under a sea of near-zero points.
    highlight_col : str, optional
        A boolean column in `merged` (e.g. 'is_dependent_Morris') -- matching
        points get a black outline, to show where a well-powered dataset's
        hits land in the other (typically under-powered) dataset.
    """
    colA, colB = _col(param, name_a), _col(param, name_b)
    for c in (colA, colB):
        if c not in merged.columns:
            raise KeyError(
                f"Parameter {param!r} not available for both datasets "
                f"(looked for {colA!r} and {colB!r}; have columns like "
                f"{[c for c in merged.columns if param.split('_')[0] in c][:8]})"
            )

    keep_cols = [colA, colB]
    # color_by can legitimately equal colA/colB (e.g. coloring the
    # observed_log2fc panel itself by observed_log2fc) -- don't add it twice,
    # or merged[keep_cols] silently returns a DataFrame instead of a Series
    # for that column (duplicate column name), breaking everything downstream.
    if color_by and color_by in merged.columns and color_by not in keep_cols:
        keep_cols.append(color_by)
    sub = merged[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[colA, colB]).copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    fig = ax.figure

    if color_by and color_by in sub.columns and sub[color_by].notna().any():
        c = sub[color_by].values
        order = np.argsort(np.abs(np.nan_to_num(c - vcenter, nan=0.0)))
        sub = sub.iloc[order]
        c = sub[color_by].values
        vmax = np.nanmax(np.abs(c))
        vmax = vmax if (vmax and np.isfinite(vmax) and vmax > 0) else 1.0
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=-vmax, vmax=vmax)
        sc = ax.scatter(sub[colA], sub[colB], c=c, cmap=cmap, norm=norm,
                         s=s, alpha=alpha, edgecolor='none')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(color_by, fontsize=8)
    else:
        ax.scatter(sub[colA], sub[colB], s=s, alpha=alpha, color='#444444', edgecolor='none')

    if highlight_col and highlight_col in merged.columns:
        hl = merged.loc[sub.index, highlight_col].fillna(False).astype(bool).values
        if hl.any():
            ax.scatter(sub.loc[hl, colA], sub.loc[hl, colB],
                       s=s * 1.8, facecolor='none', edgecolor='black', linewidth=0.8,
                       zorder=5, label=f'{highlight_col} (n={int(hl.sum())})')
            # Fixed corner (not 'best') -- matplotlib's auto-placement doesn't
            # know about the ax.text() correlation annotation below and will
            # happily stack the legend right on top of it.
            ax.legend(fontsize=7, frameon=False, loc='lower right')

    if len(sub):
        lo = min(sub[colA].min(), sub[colB].min())
        hi = max(sub[colA].max(), sub[colB].max())
        pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        lims = (lo - pad, hi + pad)
        if diag:
            ax.plot(lims, lims, ls='--', color='#999999', lw=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)

    if annotate_corr and len(sub) >= 3:
        r, _ = scipy_stats.pearsonr(sub[colA], sub[colB])
        rho, _ = scipy_stats.spearmanr(sub[colA], sub[colB])
        ax.text(0.03, 0.97, f"n={len(sub)}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))

    ax.set_xlabel(xlabel or f"{param} ({name_a})")
    ax.set_ylabel(ylabel or f"{param} ({name_b})")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_obs_log2fc(
    merged: pd.DataFrame, name_a: str, name_b: str, cis_gene: str,
    *, ax: Optional[plt.Axes] = None, highlight_col: Optional[str] = None,
) -> plt.Axes:
    """The starting-point plot: observed_log2FC, dataset A (x) vs dataset B (y)."""
    if highlight_col is None:
        cand = f'is_dependent_{name_a}'
        highlight_col = cand if cand in merged.columns else None
    return scatter_param(
        merged, 'observed_log2fc', name_a, name_b, ax=ax,
        color_by=None, highlight_col=highlight_col,
        xlabel=f'observed log2FC ({name_a})',
        ylabel=f'observed log2FC ({name_b})',
        title=f'{cis_gene}: observed log2FC',
    )


def plot_param_grid(
    merged: pd.DataFrame, name_a: str, name_b: str, cis_gene: str,
    *, params: Optional[Iterable[str]] = None,
    color_by_param: Optional[str] = 'observed_log2fc',
    color_dataset: Optional[str] = None,
    highlight_col: Optional[str] = None,
    ncols: int = 2,
    figsize_per: Tuple[float, float] = (3.8, 3.8),
) -> plt.Figure:
    """Grid of parameter-comparison scatter plots (one per parameter in
    `params`), each colored+sorted by `color_by_param` from `color_dataset`
    (default: dataset A's observed_log2fc) so large-effect genes stand out.
    """
    params = list(params) if params is not None else DEFAULT_GRID_PARAMS
    color_dataset = color_dataset or name_a
    color_col = _col(color_by_param, color_dataset) if color_by_param else None
    if color_col and color_col not in merged.columns:
        print(f"[plot_param_grid] color_by column {color_col!r} not found -- plotting uncolored.")
        color_col = None

    if highlight_col is None:
        cand = f'is_dependent_{name_a}'
        highlight_col = cand if cand in merged.columns else None

    avail = [p for p in params if _col(p, name_a) in merged.columns and _col(p, name_b) in merged.columns]
    missing = [p for p in params if p not in avail]
    if missing:
        print(f"[plot_param_grid] {cis_gene}: skipping unavailable params: {missing}")

    n = max(len(avail), 1)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                              constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    for ax, p in zip(axes, avail):
        try:
            scatter_param(merged, p, name_a, name_b, ax=ax, color_by=color_col,
                          highlight_col=highlight_col, title=p)
        except KeyError as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center', wrap=True, fontsize=7)
            ax.axis('off')

    for ax in axes[len(avail):]:
        ax.axis('off')

    subtitle = f'colored by {color_by_param} ({color_dataset})' if color_col else 'uncolored'
    fig.suptitle(f'{cis_gene}: {name_a} vs {name_b} trans-fit parameters ({subtitle})', fontsize=11)
    return fig


def compare_cis_gene(
    spec_a: DatasetSpec, spec_b: DatasetSpec, cis_gene: str,
    *, out_dir: Optional[str] = None, params: Optional[Iterable[str]] = None,
    color_by_param: str = 'observed_log2fc', save: bool = True,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """Load both datasets' trans summaries for `cis_gene`, merge on gene
    symbol, and produce (1) the observed_log2FC scatter and (2) a grid of
    other parameters colored/sorted by observed_log2FC.

    Returns (merged_df, fig_obs_log2fc, fig_param_grid).
    """
    df_a = load_trans_summary(spec_a, cis_gene)
    df_b = load_trans_summary(spec_b, cis_gene)
    merged = merge_pair(df_a, df_b, spec_a.name, spec_b.name)
    print(f"[{cis_gene}] {spec_a.name}: {len(df_a)} trans genes, {spec_b.name}: {len(df_b)} trans genes, "
          f"shared: {len(merged)}")

    highlight_col = f'is_dependent_{spec_a.name}' if f'is_dependent_{spec_a.name}' in merged.columns else None

    fig_obs, ax_obs = plt.subplots(figsize=(4.4, 4.4))
    plot_obs_log2fc(merged, spec_a.name, spec_b.name, cis_gene, ax=ax_obs, highlight_col=highlight_col)

    fig_grid = plot_param_grid(merged, spec_a.name, spec_b.name, cis_gene, params=params,
                                color_by_param=color_by_param, color_dataset=spec_a.name,
                                highlight_col=highlight_col)

    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig_obs.savefig(os.path.join(out_dir, f'{cis_gene}_{spec_a.name}_vs_{spec_b.name}_obs_log2fc.png'),
                         dpi=150, bbox_inches='tight')
        fig_grid.savefig(os.path.join(out_dir, f'{cis_gene}_{spec_a.name}_vs_{spec_b.name}_param_grid.png'),
                          dpi=150, bbox_inches='tight')

    return merged, fig_obs, fig_grid


def compare_all_shared_cis_genes(
    spec_a: DatasetSpec, spec_b: DatasetSpec,
    *, out_dir: Optional[str] = None, close_figs: bool = True, **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Run compare_cis_gene() for every cis gene with a completed fit_trans
    run in *both* datasets (DatasetSpec.cis_genes).

    Raises immediately (does not skip) if a listed cis gene's summary CSV is
    actually missing -- DatasetSpec.cis_genes means "has a completed
    fit_trans run", so that would be a real inconsistency to fix, not an
    expected gap.

    Returns {cis_gene: merged_df}.
    """
    shared = sorted(set(spec_a.cis_genes) & set(spec_b.cis_genes))
    print(f"Shared cis genes between {spec_a.name} and {spec_b.name}: {shared}")
    results = {}
    for g in shared:
        merged, fig_obs, fig_grid = compare_cis_gene(spec_a, spec_b, g, out_dir=out_dir, **kwargs)
        results[g] = merged
        if close_figs:
            plt.close(fig_obs)
            plt.close(fig_grid)
    return results


# ── N-way ("grid of pairwise comparisons") ───────────────────────────────────

def load_all(specs: List[DatasetSpec], cis_gene: str) -> Dict[str, pd.DataFrame]:
    """load_trans_summary() for every spec, keyed by dataset name. A missing
    export for one dataset raises FileNotFoundError immediately (not
    silently dropped) -- pass a smaller `specs` list if you expect that."""
    return {s.name: load_trans_summary(s, cis_gene) for s in specs}


def plot_pairwise_grid(
    dfs: Dict[str, pd.DataFrame], param: str, cis_gene: str = '',
    *, color_by_param: Optional[str] = 'observed_log2fc',
    color_dataset: Optional[str] = None,
    highlight_by: Optional[str] = 'is_dependent',
    ncols: Optional[int] = None,
    figsize_per: Tuple[float, float] = (3.8, 3.8),
) -> plt.Figure:
    """One scatter_param() subplot per unique pair of datasets in `dfs`
    (e.g. 3 datasets -> 3 subplots: A-B, A-C, B-C), all for the same `param`.
    Each pair's `on` key (gene_id vs gene_symbol) is chosen independently by
    merge_pair() -- a pair not involving Domingo will merge on Ensembl ID,
    so don't expect the same point count in every panel.

    color_by_param/color_dataset/highlight_by are the same idea as
    plot_param_grid(), applied per-pair -- `color_dataset` defaults to
    whichever dataset in the pair comes first (alphabetically among the
    pair, not globally), so it's always one of the two actually being
    plotted in that panel.
    """
    names = sorted(dfs)
    pairs = list(combinations(names, 2))
    n = max(len(pairs), 1)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                              constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    for ax, (a, b) in zip(axes, pairs):
        try:
            merged = merge_pair(dfs[a], dfs[b], a, b)
            cd = color_dataset or a
            color_col = _col(color_by_param, cd) if color_by_param else None
            if color_col and color_col not in merged.columns:
                color_col = None
            highlight_col = f'{highlight_by}_{a}' if highlight_by else None
            if highlight_col and highlight_col not in merged.columns:
                highlight_col = None
            scatter_param(merged, param, a, b, ax=ax, color_by=color_col,
                          highlight_col=highlight_col, title=f'{a} vs {b}')
        except (KeyError, ValueError) as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center', wrap=True, fontsize=7)
            ax.axis('off')

    for ax in axes[len(pairs):]:
        ax.axis('off')

    fig.suptitle(f'{cis_gene}: {param} (pairwise)'.strip(': '), fontsize=11)
    return fig


def compare_cis_gene_grid(
    specs: List[DatasetSpec], cis_gene: str,
    *, params: Optional[Iterable[str]] = None, out_dir: Optional[str] = None,
    color_by_param: str = 'observed_log2fc', save: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, plt.Figure]]:
    """N-way version of compare_cis_gene(): load every spec's summary once,
    then produce one pairwise-grid figure per parameter in `params`
    (default: observed_log2fc + DEFAULT_GRID_PARAMS).

    Returns (dfs_by_name, {param: figure}).
    """
    dfs = load_all(specs, cis_gene)
    params = list(params) if params is not None else ['observed_log2fc'] + DEFAULT_GRID_PARAMS

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    figs = {}
    for param in params:
        fig = plot_pairwise_grid(dfs, param, cis_gene, color_by_param=color_by_param)
        figs[param] = fig
        if save and out_dir:
            tag = '_'.join(sorted(dfs))
            fig.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_{param}_pairwise_grid.png'),
                        dpi=150, bbox_inches='tight')
    return dfs, figs


def compare_all_cis_genes_grid(
    datasets: Optional[List[DatasetSpec]] = None, bounding_dataset: Optional[DatasetSpec] = None,
    *, params: Optional[Iterable[str]] = None, out_dir: Optional[str] = None,
    color_by_param: str = 'observed_log2fc', save: bool = True, close_figs: bool = True,
) -> Dict[str, Tuple[Dict[str, pd.DataFrame], Dict[str, plt.Figure]]]:
    """Automates compare_cis_gene_grid() across every cis gene in
    `bounding_dataset.cis_genes` (default: the first of `datasets`, i.e.
    Domingo). For each cis gene, uses whichever subset of `datasets`
    actually lists it in their own cis_genes -- e.g. Morris drops out for
    MYB/TET2, which it never fit -- and produces a 2-dataset pairwise grid
    for those instead of a 3-way one. That drop is expected/structural, not
    an error.

    Raises immediately (does not skip) if a dataset that DOES list a given
    cis gene turns out to be missing its summary CSV -- see
    compare_all_shared_cis_genes()'s docstring for why.

    Writes into `out_dir/<cis_gene>/`. Returns {cis_gene: (dfs_by_name, {param: figure})}.
    """
    from .datasets import DOMINGO, MORRIS, REPLOGLE
    datasets = datasets or [DOMINGO, MORRIS, REPLOGLE]
    bounding_dataset = bounding_dataset or datasets[0]

    results = {}
    for cis_gene in bounding_dataset.cis_genes:
        participating = [s for s in datasets if cis_gene in s.cis_genes]
        if len(participating) < 2:
            print(f"=== {cis_gene}: only {[s.name for s in participating]} has a completed fit -- skipping ===")
            continue
        print(f"=== {cis_gene}: {[s.name for s in participating]} ===")
        gene_out = os.path.join(out_dir, cis_gene) if out_dir else None
        dfs, figs = compare_cis_gene_grid(participating, cis_gene, params=params,
                                          out_dir=gene_out, color_by_param=color_by_param, save=save)
        results[cis_gene] = (dfs, figs)
        if close_figs:
            for fig in figs.values():
                plt.close(fig)
    return results
