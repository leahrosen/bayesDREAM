"""
Per-gene dose-response curve panels, comparing two OR MORE fitted bayesDREAM
models -- a generalized, dataset-agnostic version of the original
GEX_comp_Doming_Morris.ipynb / compare_models.py (Domingo vs Morris, GFI1B
only, hardcoded paths).

Panel shape auto-adapts to however many datasets have a completed fit for a
given cis gene: 2 datasets -> 2x2 panel, 3 datasets -> 2x3 panel, etc. Row 0
is each dataset standalone (its own data + its own curve); row 1 is each
dataset's own data + own curve again, with every *other* available dataset's
curve overlaid on top (so for 3 datasets, row 1 shows up to 2 extra curves
per subplot). See `make_panel`.

Prerequisites
-------------
For each (dataset, cis_gene) pair you want a panel for, `save_model_for_plotting()`
(see save_for_plotting.py at the repo root) must have been run once, in the
original fitting session, and its output directory registered as that
dataset's `save_for_plotting_dir_fn` in comparative/datasets.py. This is a
full-model reload, so it only scales to a bounded number of genes (Domingo's
~91 shared trans genes is fine; doing this transcriptome-wide for Morris/
Replogle is not the intended use -- see trans_param_compare.py for that).

Quick start
-----------
    from comparative.datasets import DOMINGO, MORRIS
    from comparative.dose_response_panels import compare_datasets

    compare_datasets([DOMINGO, MORRIS], cis_gene='GFI1B', out_dir='./dose_response_plots')

Or all three datasets in one panel per gene, for a cis gene all three have exported::

    from comparative.datasets import DOMINGO, MORRIS, REPLOGLE
    compare_datasets([DOMINGO, MORRIS, REPLOGLE], cis_gene='GFI1B', out_dir='./dose_response_plots')

To automate this across every one of Domingo's cis genes -- using whichever
subset of {Domingo, Morris, Replogle} actually has a completed fit for each
one (see `DatasetSpec.cis_genes`) -- see `compare_all_domingo_cis_genes()`.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from bayesDREAM import bayesDREAM
from bayesDREAM.plotting.xy_plots import predict_hill_from_summary_row
from .datasets import DatasetSpec


# ── Loading ────────────────────────────────────────────────────────────────

def load_model_for_plotting(spec: DatasetSpec, cis_gene: str, device: Optional[str] = None) -> "bayesDREAM":
    """Re-initialise a bayesDREAM model from files written by
    save_model_for_plotting() (save_for_plotting.py) and load the fitted
    NTC/cis/trans parameters.
    """
    save_dir = spec.plotting_save_dir(cis_gene)

    meta = pd.read_csv(os.path.join(save_dir, 'meta_plot.csv'))

    data = np.load(os.path.join(save_dir, 'counts_plot.npz'))
    counts = pd.DataFrame(
        data['counts'],
        index=data['feature_names'].tolist(),
        columns=data['cell_names'].tolist(),
    )

    gene_meta_path = os.path.join(save_dir, 'gene_meta_plot.csv')
    gene_meta = pd.read_csv(gene_meta_path) if os.path.exists(gene_meta_path) else None

    ga_path = os.path.join(save_dir, 'guide_assignment_plot.npz')
    gm_path = os.path.join(save_dir, 'guide_meta_plot.csv')
    gt_path = os.path.join(save_dir, 'guide_target_plot.csv')
    if os.path.exists(ga_path) and os.path.exists(gm_path):
        guide_assignment = np.load(ga_path)['guide_assignment']
        guide_meta = pd.read_csv(gm_path)
        guide_target = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    else:
        guide_assignment = None
        guide_meta = None
        guide_target = None

    # Translate symbol -> feature-index identifier where they differ (Replogle:
    # Ensembl gene ID -- counts.index above comes straight from counts_plot.npz's
    # feature_names, which for Replogle IS the Ensembl ID, not the symbol; see
    # DatasetSpec.cis_gene_id_fn's docstring). None for Domingo/Morris, where
    # the feature index already is the symbol.
    feature_cis_gene = cis_gene
    if spec.cis_gene_id_fn is not None:
        feature_cis_gene = spec.cis_gene_id_fn(cis_gene)
        if feature_cis_gene is None:
            raise KeyError(
                f"[{spec.name}] cis_gene_id_fn has no mapping for symbol {cis_gene!r} -- "
                f"add one (e.g. to REPLOGLE_GENE_TO_ID in comparative/datasets.py)."
            )

    model_kwargs = dict(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene=feature_cis_gene,
        output_dir=spec.name,
        label=cis_gene,
        sum_factor_col=spec.init_sum_factor_col,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        guide_target=guide_target,
        require_ntc=False,
    )
    if device is not None:
        model_kwargs['device'] = device

    model = bayesDREAM(**model_kwargs)

    model.load_ntc_fit(input_dir=save_dir)
    model.load_cis_fit(input_dir=save_dir)
    model.load_trans_fit(input_dir=save_dir)

    sf_path = os.path.join(save_dir, 'sum_factors_plot.csv')
    if os.path.exists(sf_path):
        sf = pd.read_csv(sf_path, index_col=0)
        primary_mod = model.get_modality(model.primary_modality)
        primary_mod.sum_factors = sf
        if 'cis' in model.modalities:
            model.modalities['cis'].sum_factors = sf

    if spec.force_single_cell_line:
        model.meta['cell_line'] = spec.force_single_cell_line

    return model


def pick_sum_factor_col(model: "bayesDREAM", preference: Sequence[str] = (
    'sum_factor_new', 'sum_factor_refit', 'sum_factor_adj', 'sum_factor',
)) -> str:
    """Pick the most-adjusted sum_factor column actually available on this
    model's primary modality, trying `preference` in order. Which column a
    given export has depends on which of compute_scran/adjust_ntc_sum_factor/
    refit_sumfactor were run for that dataset -- rather than hardcode a
    per-dataset assumption that can go stale, just use whatever's furthest
    along the chain.
    """
    mod = model.get_modality(model.primary_modality)
    cols = set(mod.sum_factors.columns) if mod.sum_factors is not None else set()
    for c in preference:
        if c in cols:
            return c
    if 'sum_factor' in model.meta.columns:
        return 'sum_factor'
    raise ValueError(f"No usable sum_factor column found among {preference} or 'sum_factor' in meta.")


def resolve_sum_factor_col(spec: DatasetSpec, model: "bayesDREAM") -> str:
    """DatasetSpec.plot_sum_factor_col if set (known-correct per dataset --
    Domingo's own refit_sumfactor() writes 'sum_factor_new'; Morris/Replogle
    stop at adjust_ntc_sum_factor() and use 'sum_factor_adj'), else fall back
    to probing the reloaded model via pick_sum_factor_col().
    """
    if spec.plot_sum_factor_col:
        mod = model.get_modality(model.primary_modality)
        cols = set(mod.sum_factors.columns) if mod.sum_factors is not None else set()
        if spec.plot_sum_factor_col not in cols and spec.plot_sum_factor_col not in model.meta.columns:
            raise ValueError(
                f"[{spec.name}] configured plot_sum_factor_col={spec.plot_sum_factor_col!r} "
                f"not found on the reloaded model (available: {sorted(cols)}). Update "
                "this DatasetSpec's plot_sum_factor_col in comparative/datasets.py."
            )
        return spec.plot_sum_factor_col
    return pick_sum_factor_col(model)


def allsig_copy(summary: pd.DataFrame) -> pd.DataFrame:
    """Copy of a trans summary with FDR columns zeroed (so the Hill shape
    always draws, not just FDR-significant features) and gene_name forced to
    equal feature (feature_meta merges can otherwise populate gene_name with
    a non-matching identifier, e.g. an Ensembl ID for Replogle, breaking the
    reference_df lookup used by plot_xy_data's overlay).
    """
    out = summary.copy()
    for col in [c for c in out.columns if c.startswith('fdr_')]:
        out[col] = 0.0
    out['gene_name'] = out['feature']
    return out


# ── Panel plotting ────────────────────────────────────────────────────────

def _plot_into(model, goi, ax, spec: DatasetSpec, sum_factor_col: str,
                mark_params, fdr_df: Optional[pd.DataFrame] = None,
                show_hill_function: bool = True):
    kw = dict(
        show_hill_function=show_hill_function,
        sum_factor_col=sum_factor_col,
        log2fc=True,
        show_correction='corrected',
        legend_outside=False,
        color_by='cell_line',
        color_palette=spec.cell_line_palette,
        hill_color=spec.color,
        hill_label=spec.name,
    )
    if fdr_df is not None:
        # fdr_df (as opposed to reference_df) only overrides which FDR values
        # gate THIS dataset's own curve -- it does NOT trigger plot_xy_data's
        # built-in single reference-curve overlay (that's handled separately,
        # for any number of extra datasets, by _overlay_extra_curve below).
        kw['fdr_df'] = fdr_df
    model.plot_xy_data(goi, mark_params=mark_params, ax=ax, **kw)


def _overlay_extra_curve(ax: plt.Axes, ref_row: pd.Series, spec: DatasetSpec,
                          fdr_threshold: float = 0.05) -> bool:
    """Draw one additional dataset's fitted Hill curve on `ax`, in log2FC
    space, replicating plot_negbinom_xy's own 'reference curve overlay' math
    (bayesDREAM/plotting/xy_plots.py) -- but done externally so more than one
    extra curve can be layered onto the same ax (plot_xy_data's own
    `reference_df` argument only accepts a single overlay per call, which
    isn't enough once >2 datasets are being compared at once).

    `ref_row` should come from an "allsig" copy of that dataset's trans
    summary (see allsig_copy()) so the curve renders regardless of that
    dataset's own significance -- matching the original 2-way panels' intent
    ("show the shape even where a formal FDR call is not (yet) significant").

    Returns True if a curve was actually drawn (False if the row lacked
    x_ntc/y_ntc or all predicted y were non-positive).
    """
    y_ntc = float(ref_row.get('y_ntc', np.nan)) if hasattr(ref_row, 'get') else float('nan')
    x_ntc = float(ref_row.get('x_ntc', np.nan)) if hasattr(ref_row, 'get') else float('nan')
    if not (np.isfinite(y_ntc) and y_ntc > 0 and np.isfinite(x_ntc) and x_ntc > 0):
        return False

    x_off, y_off = np.log2(x_ntc), np.log2(y_ntc)
    xlim = ax.get_xlim()  # current visible log2FC(x) range for this panel
    x_abs = 2 ** (np.linspace(xlim[0], xlim[1], 2000) + x_off)
    y_pred = predict_hill_from_summary_row(ref_row, x_abs, fdr_threshold=fdr_threshold)
    if y_pred is None:
        return False
    valid = y_pred > 0
    if not valid.any():
        return False

    ax.plot(np.log2(x_abs[valid]) - x_off, np.log2(y_pred[valid]) - y_off,
            color=spec.color, linestyle='--', linewidth=2, alpha=0.8, label=spec.name)
    return True


def _lookup_row(summary_allsig: pd.DataFrame, goi: str) -> Optional[pd.Series]:
    match = summary_allsig.loc[summary_allsig['gene_name'] == goi]
    return match.iloc[0] if not match.empty else None


def make_panel(
    goi: str,
    specs: List[DatasetSpec], models: list, summaries_allsig: List[pd.DataFrame], sfcols: List[str],
    *, cis_gene: str, show_param_markers: bool = True, fdr_threshold: float = 0.05,
    figsize_per: Tuple[float, float] = (3.6, 3.0),
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """2xN panel (N = len(specs)): row 0 is each dataset standalone (own data
    + own curve, with fitted-parameter markers if show_param_markers); row 1
    is each dataset's own data + own curve again, with every *other*
    dataset's curve overlaid on top (so for N=3, up to 2 extra curves per
    row-1 subplot). For N=2 this reduces to the original 2x2 design.

    show_param_markers controls whether row-0 standalone panels draw the
    fitted parameter markers (EC50/inflection/etc as small annotated lines)
    -- set False if they read as confusing next to the dataset-colour curves.
    """
    n = len(specs)
    assert n >= 2, "make_panel needs at least 2 datasets to compare"
    fig, axes = plt.subplots(2, n, figsize=(figsize_per[0] * n, figsize_per[1] * 2),
                              constrained_layout=True, squeeze=False)

    mark = 'fit' if show_param_markers else False

    # Row 0: standalone (real per-dataset FDR gating, as usual).
    for j in range(n):
        _plot_into(models[j], goi, axes[0][j], specs[j], sfcols[j], mark_params=mark)
        axes[0][j].set_title(specs[j].name)

    # Row 1: dataset j's own data + own curve (force-rendered regardless of
    # dataset j's own significance, via fdr_df=its own allsig summary -- same
    # "always show the shape" intent as the original 2-way overlay panels),
    # plus every other dataset's curve manually overlaid on top.
    for j in range(n):
        ax = axes[1][j]
        _plot_into(models[j], goi, ax, specs[j], sfcols[j], mark_params=False,
                   fdr_df=summaries_allsig[j])
        overlaid_names = []
        for k in range(n):
            if k == j:
                continue
            row = _lookup_row(summaries_allsig[k], goi)
            if row is not None and _overlay_extra_curve(ax, row, specs[k], fdr_threshold=fdr_threshold):
                overlaid_names.append(specs[k].name)
        title = specs[j].name if not overlaid_names else f"{specs[j].name} + {' + '.join(overlaid_names)}"
        ax.set_title(title)
        # Rebuild this ax's own legend from scratch: plot_xy_data already drew
        # one internally, but the manually-added extra curves above were
        # plotted afterwards and aren't in it yet. get_legend_handles_labels()
        # picks up every labeled artist on the ax regardless, old and new.
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
        if h2:
            ax.legend(h2, l2, fontsize=7, frameon=False)

    # Unified axis limits, derived from the row-1 (fully-overlaid) panels so
    # row-0's parameter markers get clipped rather than expanding the view.
    xlims = [axes[1][j].get_xlim() for j in range(n)]
    ylims = [axes[1][j].get_ylim() for j in range(n)]
    unified_x = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    unified_y = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    for ax in axes.ravel():
        ax.set_xlim(unified_x)
        ax.set_ylim(unified_y)

    fig.suptitle(f'{cis_gene} → {goi}', fontsize=11, fontweight='bold')

    # Single shared legend (dedup across all 2N subplots' own legends), then
    # drop the per-subplot ones so it isn't shown twice.
    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is None:
            continue
        for h, t in zip(leg.legend_handles, [t.get_text() for t in leg.get_texts()]):
            if t not in seen:
                seen.add(t)
                handles.append(h)
                labels.append(t)
        leg.remove()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left',
                   frameon=False, fontsize=8)

    return fig, unified_x


# ── Cis-side guide-density panel ─────────────────────────────────────────────

def _get_x_ntc_log2(model) -> float:
    try:
        cis_mod = model.get_modality('cis')
        psn = getattr(cis_mod, 'posterior_samples_ntc', None)
        if psn is not None and 'mu_ntc' in psn:
            mu = np.asarray(psn['mu_ntc']).mean()
            if np.isfinite(mu) and mu > 0:
                return float(np.log2(mu))
    except Exception:
        pass
    ntc = model.meta.loc[model.meta['target'] == 'ntc', 'x_true']
    ntc = ntc[ntc > 0]
    return float(np.log2(ntc.mean())) if len(ntc) > 0 else 0.0


def _expand_cell_guide_data(model, meta_filtered: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, targeting guide) for per-guide KDE plotting.
    High-MOI: reads model.guide_assignment directly. Low-MOI: parses the
    'guide' string column.
    """
    if getattr(model, 'is_high_moi', False) and model.guide_assignment is not None:
        guide_names = model.guide_meta['guide'].tolist()
        gtd = getattr(model, 'guide_targets_dict', {})
        ntc_guides = {g for g, targets in gtd.items() if all(str(t).lower() == 'ntc' for t in targets)}

        ga = model.guide_assignment
        cell_to_idx = {cell: i for i, cell in enumerate(model.meta['cell'])}

        expanded = []
        for _, row in meta_filtered.iterrows():
            ci = cell_to_idx.get(row['cell'])
            if ci is None:
                continue
            active = [guide_names[j] for j in np.where(ga[ci] > 0)[0]]
            targeting = [g for g in active if g not in ntc_guides]
            if not targeting:
                r = row.to_dict()
                r['eff_guide'] = 'ntc'
                expanded.append(r)
            else:
                for g in targeting:
                    r = row.to_dict()
                    r['eff_guide'] = g
                    expanded.append(r)

        if not expanded:
            return meta_filtered.assign(eff_guide='unknown')
        return pd.DataFrame(expanded)

    gtd = getattr(model, 'guide_targets_dict', None)
    ntc_names = set()
    if gtd:
        ntc_names = {g for g, targets in gtd.items() if all(str(t).lower() == 'ntc' for t in targets)}

    def _parse(g):
        if pd.isna(g):
            return 'unknown'
        g = str(g)
        for sep in ('+', ',', '|', ';'):
            if sep in g:
                parts = [p.strip() for p in g.split(sep)]
                tgt = [p for p in parts if p not in ntc_names and 'ntc' not in p.lower()]
                if len(tgt) == 1:
                    return tgt[0]
                if len(tgt) > 1:
                    return '+'.join(sorted(tgt))
                return parts[0]
        return g

    out = meta_filtered.copy()
    out['eff_guide'] = out.get('guide', pd.Series('unknown', index=out.index)).map(_parse)
    return out


def make_density_panel(cis_gene: str, specs_and_models: List[Tuple[DatasetSpec, "bayesDREAM"]],
                        unified_x: Tuple[float, float], figsize_per=(4.5, 3)) -> plt.Figure:
    """1xN panel of log2FC(x_true) KDE density curves, one column per model
    (x_true is the cis gene's own expression -- constant across trans genes,
    so this panel is produced once per (cis_gene, dataset set), not per gene).
    """
    n = len(specs_and_models)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), constrained_layout=True)
    axes = np.atleast_1d(axes)
    x_grid = np.linspace(unified_x[0], unified_x[1], 500)

    def _kde_plot(ax, vals, color, alpha_fill=0.15, lw=1.2):
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            return
        y = gaussian_kde(vals, bw_method='scott')(x_grid)
        ax.fill_between(x_grid, 0, y, alpha=alpha_fill, color=color)
        ax.plot(x_grid, y, color=color, linewidth=lw)

    for ax, (spec, model) in zip(axes, specs_and_models):
        x_ntc_log2 = _get_x_ntc_log2(model)
        meta = model.meta.copy()
        if 'x_true' not in meta.columns:
            xt = getattr(model, 'x_true', None)
            if xt is None:
                raise ValueError(f"x_true not found for {spec.name}")
            meta['x_true'] = np.asarray(xt.cpu() if hasattr(xt, 'cpu') else xt)
        meta = meta[meta['x_true'] > 0].copy()
        meta['log2fc_x'] = np.log2(meta['x_true']) - x_ntc_log2

        expanded = _expand_cell_guide_data(model, meta)
        legend_patches = []

        ntc_vals = expanded.loc[expanded['target'] == 'ntc', 'log2fc_x'].values
        if len(ntc_vals[np.isfinite(ntc_vals)]) >= 2:
            _kde_plot(ax, ntc_vals, color='gray')
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.5, label='NTC'))

        tgt = expanded[expanded['target'] != 'ntc']
        crispr_i = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRi', 'eff_guide'].unique())
        crispr_a = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRa', 'eff_guide'].unique()) if 'CRISPRa' in spec.cell_line_palette else []

        if crispr_i:
            blues = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(crispr_i), 1)))
            for j, guide in enumerate(crispr_i):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRi'), 'log2fc_x'].values
                _kde_plot(ax, vals, color=blues[j])
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Blues(0.65), alpha=0.6, label='CRISPRi'))

        if crispr_a:
            reds = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(crispr_a), 1)))
            for j, guide in enumerate(crispr_a):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRa'), 'log2fc_x'].values
                _kde_plot(ax, vals, color=reds[j])
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Reds(0.65), alpha=0.6, label='CRISPRa'))

        ax.set_xlim(unified_x)
        ax.set_xlabel('log2FC(x_true)')
        ax.set_ylabel('Density')
        ax.set_title(spec.name)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.legend(handles=legend_patches, frameon=False, fontsize=8)

    fig.suptitle(f'{cis_gene}  |  guide distributions', fontsize=10)
    return fig


# ── Top-level orchestrator ───────────────────────────────────────────────────

def _tag(specs: List[DatasetSpec]) -> str:
    return '_vs_'.join(s.name for s in specs)


def _missing_exports(specs: List[DatasetSpec], cis_gene: str) -> List[Tuple[DatasetSpec, str]]:
    """Cheap upfront check (no model loading) of which `specs` are missing
    their save_model_for_plotting() export for `cis_gene`. Returns
    [(spec, error_message), ...] for each one that's missing (empty if all
    present). Used to fail fast / fall back gracefully instead of fully
    reloading N-1 (potentially large) models before discovering the Nth is
    missing.
    """
    missing = []
    for spec in specs:
        try:
            spec.plotting_save_dir(cis_gene)
        except (FileNotFoundError, ValueError) as e:
            missing.append((spec, str(e)))
    return missing


def compare_datasets(
    specs: List[DatasetSpec], cis_gene: str,
    *, out_dir: str = './dose_response_plots', genes: Optional[List[str]] = None,
    show_param_markers: bool = True, device: Optional[str] = None,
    panel_figsize_per: Tuple[float, float] = (3.6, 3.0),
) -> List[str]:
    """Full pipeline for N (>=2) datasets: load all N models for `cis_gene`,
    summarise each, find trans genes present in *every* dataset's summary
    (or use `genes` if given), and write one 2xN panel PNG per gene plus one
    guide-density panel to `out_dir`. Panel width auto-scales with N.

    Checks that every spec's save_model_for_plotting() export actually
    exists BEFORE loading any model -- so a missing dataset fails immediately
    rather than after wastefully reloading the others (which, for a
    transcriptome-wide dataset like Morris, can take a while). If you'd
    rather silently drop whatever's missing and continue with the rest, use
    compare_all_domingo_cis_genes() (or pre-filter `specs` yourself via
    _missing_exports()).

    Returns the list of trans genes actually plotted.
    """
    assert len(specs) >= 2, "compare_datasets needs at least 2 datasets"
    names = [s.name for s in specs]
    tag = _tag(specs)

    missing = _missing_exports(specs, cis_gene)
    if missing:
        lines = "\n".join(f"  - {msg}" for _, msg in missing)
        raise FileNotFoundError(
            f"Cannot compare {names} for {cis_gene!r}: {len(missing)}/{len(specs)} dataset(s) "
            f"missing their save_model_for_plotting() export (checked before loading any "
            f"models, to avoid wasting time loading the others):\n{lines}"
        )

    models = []
    for spec in specs:
        print(f"[{spec.name}] loading model for {cis_gene}...")
        models.append(load_model_for_plotting(spec, cis_gene, device=device))

    sfcols = [resolve_sum_factor_col(s, m) for s, m in zip(specs, models)]
    for s, sf in zip(specs, sfcols):
        print(f"[{s.name}] plotting sum_factor_col={sf!r}")

    # compute_derivative_roots isn't needed for dose-response curve plotting
    # (only for finding local optima elsewhere) -- skip it everywhere for speed.
    summaries = []
    for spec, model in zip(specs, models):
        print(f"[{spec.name}] summarising trans fit...")
        summaries.append(model.save_trans_summary(compute_lfc_ci=False, compute_derivative_roots=False))
    summaries_allsig = [allsig_copy(s) for s in summaries]

    if genes is None:
        gene_sets = [set(s['feature']) for s in summaries]
        genes = sorted(set.intersection(*gene_sets))
    print(f"\n{' vs '.join(names)} ({cis_gene}): {len(genes)} trans genes to plot")

    os.makedirs(out_dir, exist_ok=True)

    unified_x_all = None
    plotted = []
    for i, goi in enumerate(genes, 1):
        print(f"  [{i}/{len(genes)}] {goi}", end='', flush=True)
        fig, unified_x = make_panel(
            goi, specs, models, summaries_allsig, sfcols,
            cis_gene=cis_gene, show_param_markers=show_param_markers, figsize_per=panel_figsize_per,
        )
        fig.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_{goi}_panel.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        plotted.append(goi)
        unified_x_all = unified_x if unified_x_all is None else (
            min(unified_x_all[0], unified_x[0]), max(unified_x_all[1], unified_x[1]))
        print("  done")

    if unified_x_all is not None:
        print("Plotting guide density panel...", end='', flush=True)
        fig_density = make_density_panel(cis_gene, list(zip(specs, models)), unified_x_all)
        fig_density.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_guide_density.png'),
                             dpi=150, bbox_inches='tight')
        plt.close(fig_density)
        print("  done")

    print(f"\nDone. {len(plotted)} panels + density plot written to {out_dir}/")
    return plotted


def compare_pair(
    spec_a: DatasetSpec, spec_b: DatasetSpec, cis_gene: str, **kwargs,
) -> List[str]:
    """Thin 2-dataset convenience wrapper around compare_datasets()."""
    return compare_datasets([spec_a, spec_b], cis_gene, **kwargs)


def compare_all_domingo_cis_genes(
    *, out_dir: str = './dose_response_plots', datasets: Optional[List[DatasetSpec]] = None,
    bounding_dataset: Optional[DatasetSpec] = None, require_all_exports: bool = True, **kwargs,
) -> Dict[str, List[str]]:
    """Automate compare_datasets() across every cis gene, using whichever
    subset of `datasets` (default: [DOMINGO, MORRIS, REPLOGLE]) structurally
    has a completed fit_trans run for that gene (per each DatasetSpec.cis_genes)
    -- so GFI1B/NFE2 (all 3 fit) get a 2x3 panel, while TET2/MYB (Morris
    never fit these; see publication_runs/morris/config.yaml's primary_genes)
    are Domingo-vs-Replogle from the start. That drop is expected/structural,
    not an error.

    require_all_exports : bool
        If True (default), raise immediately if a dataset that DOES list a
        given cis gene in its own cis_genes is still missing its
        save_model_for_plotting() export -- run
        comparative.reconstruct_export(_replogle).reconstruct_and_export_all()
        first. Pass False to instead silently drop that dataset from the
        gene's panel (the old, pre-automation behavior, for partial/manual
        adoption before every gene has been exported).

    The cis gene list iterated is `bounding_dataset.cis_genes` (default: the
    first dataset in `datasets`, i.e. Domingo) -- the dataset with the
    smallest/most tractable trans gene panel, since this is a full-model-reload
    per (dataset, gene) operation (see module docstring).

    Writes into `out_dir/<cis_gene>/`. Returns {cis_gene: [genes plotted]}.
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

        missing = _missing_exports(participating, cis_gene)
        if missing:
            lines = "\n".join(f"  - {msg}" for _, msg in missing)
            if require_all_exports:
                raise FileNotFoundError(
                    f"{cis_gene}: {len(missing)} dataset(s) have a completed fit_trans but no "
                    f"save_model_for_plotting() export yet -- run "
                    f"comparative.reconstruct_export(_replogle).reconstruct_and_export_all() "
                    f"first, or pass require_all_exports=False to drop them instead:\n{lines}"
                )
            missing_names = {spec.name for spec, _ in missing}
            participating = [s for s in participating if s.name not in missing_names]
            print(f"    ({cis_gene}: {sorted(missing_names)} dropped -- no export yet)")
            if len(participating) < 2:
                print(f"=== {cis_gene}: only {[s.name for s in participating]} usable -- skipping ===")
                continue

        print(f"=== {cis_gene}: {[s.name for s in participating]} ===")
        results[cis_gene] = compare_datasets(
            participating, cis_gene, out_dir=os.path.join(out_dir, cis_gene), **kwargs,
        )
    return results
