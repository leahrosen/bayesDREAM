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
- The join key is always a gene *symbol* (``DatasetSpec.symbol_col`` tells
  each dataset which raw column holds that), never the raw ``feature``
  column -- Replogle's ``feature`` is an Ensembl gene ID.
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
}

# Sensible default grid for "everything besides observed_log2fc" -- edit
# freely per call via the `params=` argument.
DEFAULT_GRID_PARAMS = ['full_log2fc', 'EC50_a_log2fc', 'n_a', 'Vmax_a']


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
      - 'gene_symbol': the cross-dataset join key (from spec.symbol_col)
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

    # 'feature' is guaranteed unique within a dataset (save_trans_summary()
    # enforces this at fit time -- see bayesDREAM/io/summary.py). 'gene_symbol'
    # is NOT, whenever it's derived rather than being 'feature' itself: e.g.
    # Replogle is indexed by Ensembl gene ID ('feature'), and a handful of
    # genes (TBCE, HSPA14, ...) are annotated as two separate Ensembl IDs
    # sharing one symbol -- a known segmental-duplication artifact, not a
    # bug here. A symbol-only dataset (Domingo/Morris) has no Ensembl ID to
    # disambiguate which locus it means, so there is no safe way to decide
    # which of the ambiguous rows a cross-dataset match should use. Rather
    # than silently pick one (wrong half the time) or let merge_pair's
    # duplicate check crash the whole comparison, drop just the ambiguous
    # symbols here -- using the dataset's OWN unique 'feature' id to detect
    # them -- and keep everything else.
    if spec.symbol_col != 'feature' and 'feature' in df.columns:
        dupe_symbols = df.loc[df['gene_symbol'].duplicated(keep=False), 'gene_symbol'].unique()
        if len(dupe_symbols):
            n_before = len(df)
            df = df[~df['gene_symbol'].isin(dupe_symbols)].copy()
            print(f"[{spec.name}] dropped {len(dupe_symbols)} ambiguous gene_symbol value(s) that map "
                  f"to >1 distinct 'feature' id in this dataset (e.g. {sorted(dupe_symbols)[:5]}) -- "
                  f"{n_before - len(df)} row(s) excluded from cross-dataset comparison.")

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
                on: str = 'gene_symbol') -> pd.DataFrame:
    """Inner-join two standardized trans summaries on gene symbol.

    Shared column names (parameters, 'is_dependent', etc.) get suffixed
    ``_{name_a}`` / ``_{name_b}``; the join key itself is not suffixed.
    """
    if df_a[on].duplicated().any():
        dupes = df_a.loc[df_a[on].duplicated(), on].unique()[:5]
        raise ValueError(f"{name_a}: duplicated {on} values, e.g. {list(dupes)} -- cannot merge safely.")
    if df_b[on].duplicated().any():
        dupes = df_b.loc[df_b[on].duplicated(), on].unique()[:5]
        raise ValueError(f"{name_b}: duplicated {on} values, e.g. {list(dupes)} -- cannot merge safely.")

    merged = df_a.merge(df_b, on=on, how='inner', suffixes=(f'_{name_a}', f'_{name_b}'))
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
    if color_by and color_by in merged.columns:
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
    run in *both* datasets (DatasetSpec.cis_genes). Returns {cis_gene: merged_df}.
    """
    shared = sorted(set(spec_a.cis_genes) & set(spec_b.cis_genes))
    print(f"Shared cis genes between {spec_a.name} and {spec_b.name}: {shared}")
    results = {}
    for g in shared:
        try:
            merged, fig_obs, fig_grid = compare_cis_gene(spec_a, spec_b, g, out_dir=out_dir, **kwargs)
            results[g] = merged
            if close_figs:
                plt.close(fig_obs)
                plt.close(fig_grid)
        except FileNotFoundError as e:
            print(f"[skip {g}] {e}")
    return results
