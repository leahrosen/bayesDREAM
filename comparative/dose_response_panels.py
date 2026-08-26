"""
Per-gene dose-response curve panels, comparing two (or more) fitted
bayesDREAM models -- a generalized, dataset-agnostic version of the original
GEX_comp_Doming_Morris.ipynb / compare_models.py (Domingo vs Morris, GFI1B
only, hardcoded paths).

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
    from comparative.dose_response_panels import compare_pair

    compare_pair(DOMINGO, MORRIS, cis_gene='GFI1B', out_dir='./dose_response_plots')

To compare all three datasets pairwise for a gene all three have exported::

    from comparative.datasets import DOMINGO, MORRIS, REPLOGLE
    from comparative.dose_response_panels import compare_pair
    from itertools import combinations
    for a, b in combinations([DOMINGO, MORRIS, REPLOGLE], 2):
        compare_pair(a, b, cis_gene='GFI1B', out_dir='./dose_response_plots')
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from bayesDREAM import bayesDREAM
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

    model_kwargs = dict(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene=cis_gene,
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
                mark_params, ref_df=None, ref_spec: Optional[DatasetSpec] = None,
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
    ref_kwargs = {}
    if ref_df is not None and ref_spec is not None:
        ref_kwargs = dict(reference_df=ref_df, ref_color=ref_spec.color, ref_label=ref_spec.name)
    model.plot_xy_data(goi, mark_params=mark_params, ax=ax, **kw, **ref_kwargs)


def make_panel(
    goi: str,
    spec_a: DatasetSpec, model_a, summary_a_allsig: pd.DataFrame, sfcol_a: str,
    spec_b: DatasetSpec, model_b, summary_b_allsig: pd.DataFrame, sfcol_b: str,
    *, cis_gene: str, show_param_markers: bool = True, figsize=(9, 6),
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """2x2 panel: [A standalone, B standalone] / [A + B curve, B + A curve].
    show_param_markers controls whether standalone panels draw the fitted
    parameter markers (EC50/inflection/etc as small annotated lines) -- set
    False if the marker lines get visually confused with the dataset-colour
    curves (e.g. when using unfamiliar new dataset colours).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    (ax_a, ax_b), (ax_ab, ax_ba) = axes

    mark = 'fit' if show_param_markers else False
    _plot_into(model_a, goi, ax_a, spec_a, sfcol_a, mark_params=mark)
    _plot_into(model_b, goi, ax_b, spec_b, sfcol_b, mark_params=mark)
    _plot_into(model_a, goi, ax_ab, spec_a, sfcol_a, mark_params=False,
               ref_df=summary_b_allsig, ref_spec=spec_b)
    _plot_into(model_b, goi, ax_ba, spec_b, sfcol_b, mark_params=False,
               ref_df=summary_a_allsig, ref_spec=spec_a)

    # Unified axis limits, derived from the marker-free overlay panels so
    # standalone-panel markers get clipped rather than expanding the view.
    xlims = [ax_ab.get_xlim(), ax_ba.get_xlim()]
    ylims = [ax_ab.get_ylim(), ax_ba.get_ylim()]
    unified_x = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    unified_y = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    for ax in axes.ravel():
        ax.set_xlim(unified_x)
        ax.set_ylim(unified_y)

    ax_a.set_title(spec_a.name)
    ax_b.set_title(spec_b.name)
    ax_ab.set_title(f'{spec_a.name} + {spec_b.name} curve')
    ax_ba.set_title(f'{spec_b.name} + {spec_a.name} curve')
    fig.suptitle(f'{cis_gene} → {goi}', fontsize=11, fontweight='bold')

    # Single shared legend (dedup across all four subplots' own legends).
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

def compare_pair(
    spec_a: DatasetSpec, spec_b: DatasetSpec, cis_gene: str,
    *, out_dir: str = './dose_response_plots', genes: Optional[List[str]] = None,
    show_param_markers: bool = True, device: Optional[str] = None,
    panel_figsize=(9, 6),
) -> List[str]:
    """Full pipeline: load both models for `cis_gene`, summarise, find
    overlapping trans genes (or use `genes` if given), and write one
    2x2 panel PNG per gene plus one guide-density panel to `out_dir`.

    Returns the list of trans genes actually plotted.
    """
    print(f"[{spec_a.name}] loading model for {cis_gene}...")
    model_a = load_model_for_plotting(spec_a, cis_gene, device=device)
    print(f"[{spec_b.name}] loading model for {cis_gene}...")
    model_b = load_model_for_plotting(spec_b, cis_gene, device=device)

    sfcol_a = resolve_sum_factor_col(spec_a, model_a)
    sfcol_b = resolve_sum_factor_col(spec_b, model_b)
    print(f"[{spec_a.name}] plotting sum_factor_col={sfcol_a!r}; [{spec_b.name}] sum_factor_col={sfcol_b!r}")

    print(f"[{spec_a.name}] summarising trans fit...")
    summary_a = model_a.save_trans_summary(compute_lfc_ci=False)
    print(f"[{spec_b.name}] summarising trans fit...")
    summary_b = model_b.save_trans_summary(compute_lfc_ci=False, compute_derivative_roots=False)

    summary_a_allsig = allsig_copy(summary_a)
    summary_b_allsig = allsig_copy(summary_b)

    if genes is None:
        genes_a, genes_b = set(summary_a['feature']), set(summary_b['feature'])
        genes = sorted(genes_a & genes_b)
    print(f"\n{spec_a.name} vs {spec_b.name} ({cis_gene}): {len(genes)} trans genes to plot")

    os.makedirs(out_dir, exist_ok=True)

    unified_x_all = None
    plotted = []
    for i, goi in enumerate(genes, 1):
        print(f"  [{i}/{len(genes)}] {goi}", end='', flush=True)
        fig, unified_x = make_panel(
            goi, spec_a, model_a, summary_a_allsig, sfcol_a,
            spec_b, model_b, summary_b_allsig, sfcol_b,
            cis_gene=cis_gene, show_param_markers=show_param_markers, figsize=panel_figsize,
        )
        fig.savefig(os.path.join(out_dir, f'{cis_gene}_{spec_a.name}_vs_{spec_b.name}_{goi}_panel.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        plotted.append(goi)
        unified_x_all = unified_x if unified_x_all is None else (
            min(unified_x_all[0], unified_x[0]), max(unified_x_all[1], unified_x[1]))
        print("  done")

    if unified_x_all is not None:
        print("Plotting guide density panel...", end='', flush=True)
        fig_density = make_density_panel(cis_gene, [(spec_a, model_a), (spec_b, model_b)], unified_x_all)
        fig_density.savefig(os.path.join(out_dir, f'{cis_gene}_{spec_a.name}_vs_{spec_b.name}_guide_density.png'),
                             dpi=150, bbox_inches='tight')
        plt.close(fig_density)
        print("  done")

    print(f"\nDone. {len(plotted)} panels + density plot written to {out_dir}/")
    return plotted
