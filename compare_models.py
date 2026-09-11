"""
Compare two fitted bayesDREAM models across all overlapping trans genes.

For each overlapping gene, produces one panel saved to PLOT_DIR/:
  {gene}_panel.png  - 2×2 grid:
      top-left:     Model A data + Model A curve (standalone, with param markers)
      top-right:    Model B data + Model B curve (standalone, with param markers)
      bottom-left:  Model A data + Model A curve + Model B curve overlaid
      bottom-right: Model B data + Model B curve + Model A curve overlaid

  All four subplots share the same x/y axis limits (derived from the data
  range only — param markers do not expand the limits).

Prerequisites
-------------
Run save_for_plotting.py in your fitting session first.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(__file__))
from bayesDREAM import bayesDREAM


# ── Configure these ───────────────────────────────────────────────────────────

cis_gene = 'YOUR_CIS_GENE'

# These must match the output_dir / label used when the models were fitted
# (save_for_plotting.py writes files into <OUTDIR>/<LABEL>/).
OUTDIR_A  = 'path/to/output_dir_a'
FSLABEL_A = 'label_a'   # filesystem label (used when the model was fitted)

OUTDIR_B  = 'path/to/output_dir_b'
FSLABEL_B = 'label_b'

# sum_factor_col used at model init time (usually 'sum_factor', the default).
# This is NOT the same as the sum_factor_col you pass to plot_xy_data.
INIT_SUM_FACTOR_COL = 'sum_factor'

PLOT_DIR = './model_comparison_plots'   # where to write the PNGs

# Display labels and colours for the two models' fitted curves.
LABEL_A = 'Domingo'
LABEL_B = 'Morris'
COLOR_A = '#E07B39'   # warm orange
COLOR_B = '#6A5ACD'   # slate purple

# Data colours (by cell_line).
PALETTE_A = {'CRISPRi': 'steelblue', 'CRISPRa': 'tomato'}
PALETTE_B = {'CRISPRi': 'steelblue'}

PANEL_FIGSIZE = (9, 6)   # width × height of the 2×2 panel


# ── Loading helper ────────────────────────────────────────────────────────────

def load_model_for_plotting(outdir, fslabel, cis_gene,
                            sum_factor_col=INIT_SUM_FACTOR_COL):
    """
    Re-initialise a bayesDREAM model from files written by save_for_plotting.py
    and load the fitted NTC, cis, and trans parameters.
    """
    save_dir = os.path.join(outdir, fslabel)

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
        guide_meta       = pd.read_csv(gm_path)
        guide_target     = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    else:
        guide_assignment = None
        guide_meta       = None
        guide_target     = None

    model = bayesDREAM(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene=cis_gene,
        output_dir=outdir,
        label=fslabel,
        sum_factor_col=sum_factor_col,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        guide_target=guide_target,
        require_ntc=False,
    )

    model.load_ntc_fit()
    model.load_cis_fit()
    model.load_trans_fit()

    sf_path = os.path.join(save_dir, 'sum_factors_plot.csv')
    if os.path.exists(sf_path):
        sf = pd.read_csv(sf_path, index_col=0)
        primary_mod = model.get_modality(model.primary_modality)
        primary_mod.sum_factors = sf
        if 'cis' in model.modalities:
            model.modalities['cis'].sum_factors = sf

    return model


# ── Load models ───────────────────────────────────────────────────────────────

print("Loading model A...")
model_a = load_model_for_plotting(OUTDIR_A, FSLABEL_A, cis_gene)

print("Loading model B...")
model_b = load_model_for_plotting(OUTDIR_B, FSLABEL_B, cis_gene)
model_b.meta['cell_line'] = 'CRISPRi'

# Compute trans summaries for the cross-overlay.
# compute_lfc_ci=False skips the ~1.5h CI loops — reference_df only needs
# the *_mean parameter columns.
print("Summarising model A (for overlay on model B)...")
summary_a = model_a.save_trans_summary(compute_lfc_ci=False)

print("Summarising model B (for overlay on model A)...")
summary_b = model_b.save_trans_summary(compute_lfc_ci=False)

# Copies used for overlays: FDR zeroed (so Hill shape always draws, not just
# significant components) and gene_name forced to equal feature (feature_meta
# merges in save_trans_summary can populate gene_name with non-matching values
# like Ensembl IDs, breaking the reference_df lookup in plot_negbinom_xy).
_fdr_cols = [c for c in summary_a.columns if c.startswith('fdr_')]
summary_a_allsig = summary_a.copy()
summary_b_allsig = summary_b.copy()
for col in _fdr_cols:
    summary_a_allsig[col] = 0.0
    summary_b_allsig[col] = 0.0
summary_a_allsig['gene_name'] = summary_a_allsig['feature']
summary_b_allsig['gene_name'] = summary_b_allsig['feature']


# ── Find overlapping genes ────────────────────────────────────────────────────

genes_a = set(summary_a['feature'])
genes_b = set(summary_b['feature'])
overlap  = sorted(genes_a & genes_b)

print(f"\nOverlap: {len(overlap)} genes ({len(genes_a)} in A, {len(genes_b)} in B)")

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Panel-making function ─────────────────────────────────────────────────────

# kwargs shared by every plot_xy_data call (no figsize — we supply axes directly)
_base = dict(
    show_hill_function=True,
    sum_factor_col='sum_factor_new',
    log2fc=True,
    show_correction='corrected',
    legend_outside=False,
)
_kw_a = dict(**_base, color_by='cell_line', color_palette=PALETTE_A,
             hill_color=COLOR_A, hill_label=LABEL_A)
_kw_b = dict(**_base, color_by='cell_line', color_palette=PALETTE_B,
             hill_color=COLOR_B, hill_label=LABEL_B)


def _plot_into(model, goi, ax, kw, ref_df=None, mark_params=False):
    """Call plot_xy_data into a pre-existing axis."""
    ref_kwargs = {}
    if ref_df is not None:
        # Determine which model is the reference so we can label it correctly.
        is_a_ref = ref_df is summary_a_allsig
        ref_kwargs = dict(
            reference_df=ref_df,
            ref_color=COLOR_A if is_a_ref else COLOR_B,
            ref_label=LABEL_A if is_a_ref else LABEL_B,
        )
    model.plot_xy_data(goi, mark_params=mark_params, ax=ax, **kw, **ref_kwargs)


def make_panel(goi):
    fig, axes = plt.subplots(2, 2, figsize=PANEL_FIGSIZE, constrained_layout=True)
    (ax_a, ax_b), (ax_ab, ax_ba) = axes

    # ── Draw all 4 subplots ───────────────────────────────────────────────────
    # Standalone plots get mark_params='fit'; limits may be expanded by markers.
    # Overlay plots get no markers; their limits reflect pure data range.
    _plot_into(model_a, goi, ax_a,  _kw_a, mark_params='fit')
    _plot_into(model_b, goi, ax_b,  _kw_b, mark_params='fit')
    _plot_into(model_a, goi, ax_ab, _kw_a, ref_df=summary_b_allsig, mark_params=False)
    _plot_into(model_b, goi, ax_ba, _kw_b, ref_df=summary_a_allsig, mark_params=False)

    # ── Unified axis limits ───────────────────────────────────────────────────
    # Derive limits from the overlay subplots (marker-free) and take the union,
    # so all four panels see the full data range of both models.  Applying these
    # limits to the standalone panels clips any markers that fall outside the
    # data range without expanding the view.
    xlims = [ax_ab.get_xlim(), ax_ba.get_xlim()]
    ylims = [ax_ab.get_ylim(), ax_ba.get_ylim()]
    unified_x = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    unified_y = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    for ax in axes.ravel():
        ax.set_xlim(unified_x)
        ax.set_ylim(unified_y)

    # ── Titles ────────────────────────────────────────────────────────────────
    ax_a.set_title(LABEL_A)
    ax_b.set_title(LABEL_B)
    ax_ab.set_title(f'{LABEL_A} + {LABEL_B} curve')
    ax_ba.set_title(f'{LABEL_B} + {LABEL_A} curve')
    fig.suptitle(f'{cis_gene} → {goi}', fontsize=11, fontweight='bold')

    # ── Single shared legend ──────────────────────────────────────────────────
    # Collect handles from all subplots, deduplicate by label, then replace
    # per-subplot legends with one figure-level legend.
    seen_labels = set()
    legend_handles = []
    legend_labels  = []
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is None:
            continue
        for handle, text in zip(leg.legend_handles,
                                [t.get_text() for t in leg.get_texts()]):
            if text not in seen_labels:
                seen_labels.add(text)
                legend_handles.append(handle)
                legend_labels.append(text)
        leg.remove()

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False,
                   fontsize=8)

    return fig, unified_x


# ── Density-panel helpers ─────────────────────────────────────────────────────

def _get_x_ntc_log2(model):
    """Return log2(x_ntc) using the same source as plot_negbinom_xy."""
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


def _expand_cell_guide_data(model, meta_filtered):
    """
    Return a DataFrame with an 'eff_guide' column for per-guide KDE plotting.

    High-MOI path (uses guide_assignment matrix directly):
      - NTC-only cells  → eff_guide = 'ntc'  (one row)
      - 1 targeting guide + any NTC → eff_guide = targeting guide  (one row)
      - 2+ targeting guides → one row per targeting guide (cell duplicated)

    Low-MOI path: parses the 'guide' string in meta.
    """
    if getattr(model, 'is_high_moi', False) and model.guide_assignment is not None:
        guide_names = model.guide_meta['guide'].tolist()
        gtd = getattr(model, 'guide_targets_dict', {})
        ntc_guides = {g for g, targets in gtd.items()
                      if all(str(t).lower() == 'ntc' for t in targets)}

        ga = model.guide_assignment          # (n_cells_total × n_guides)
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

    else:
        # Low-MOI: one guide string per cell, parse NTC + combo strings.
        gtd = getattr(model, 'guide_targets_dict', None)
        ntc_names = set()
        if gtd:
            ntc_names = {g for g, targets in gtd.items()
                         if all(str(t).lower() == 'ntc' for t in targets)}

        def _parse(g):
            if pd.isna(g):
                return 'unknown'
            g = str(g)
            for sep in ('+', ',', '|', ';'):
                if sep in g:
                    parts = [p.strip() for p in g.split(sep)]
                    tgt = [p for p in parts
                           if p not in ntc_names and 'ntc' not in p.lower()]
                    if len(tgt) == 1:
                        return tgt[0]
                    if len(tgt) > 1:
                        return '+'.join(sorted(tgt))
                    return parts[0]
            return g

        out = meta_filtered.copy()
        out['eff_guide'] = out.get('guide', pd.Series('unknown', index=out.index)).map(_parse)
        return out


def make_density_panel(label, unified_x):
    """
    1×2 panel of log2FC(x_true) KDE density curves, one column per model.
    Guides in CRISPRi cells → shades of blue; CRISPRa → shades of red; NTC → gray.
    x_true is the cis gene expression — constant across trans genes — so this
    panel is produced once per run, not once per gene.
    unified_x: (lo, hi) in log2FC space, covering the range of all gene panels.
    """
    # Same total width as the main panel; height = half (density needs less space)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(PANEL_FIGSIZE[0], PANEL_FIGSIZE[1] / 2),
        constrained_layout=True,
    )
    x_grid = np.linspace(unified_x[0], unified_x[1], 500)

    def _plot_one(ax, model, title):
        x_ntc_log2 = _get_x_ntc_log2(model)

        meta = model.meta.copy()
        # x_true may not be in meta for high-MOI models; fall back to model.x_true
        if 'x_true' not in meta.columns:
            xt = getattr(model, 'x_true', None)
            if xt is None:
                raise ValueError(f"x_true not found for {title}")
            meta['x_true'] = np.asarray(xt.cpu() if hasattr(xt, 'cpu') else xt)

        meta = meta[meta['x_true'] > 0].copy()
        meta['log2fc_x'] = np.log2(meta['x_true']) - x_ntc_log2

        # Expand to one row per (cell × targeting guide); high-MOI uses the
        # guide_assignment matrix directly rather than parsing the guide string.
        expanded = _expand_cell_guide_data(model, meta)

        legend_patches = []

        def _kde_plot(vals, color, alpha_fill=0.15, lw=1.2):
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                return
            y = gaussian_kde(vals, bw_method='scott')(x_grid)
            ax.fill_between(x_grid, 0, y, alpha=alpha_fill, color=color)
            ax.plot(x_grid, y, color=color, linewidth=lw)

        # NTC cells — pooled, gray
        ntc_vals = expanded.loc[expanded['target'] == 'ntc', 'log2fc_x'].values
        if len(ntc_vals[np.isfinite(ntc_vals)]) >= 2:
            _kde_plot(ntc_vals, color='gray')
            legend_patches.append(
                plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.5, label='NTC'))

        # Targeting cells grouped by (eff_guide, cell_line)
        tgt = expanded[expanded['target'] != 'ntc']

        crispr_i = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRi', 'eff_guide'].unique())
        crispr_a = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRa', 'eff_guide'].unique())

        if crispr_i:
            blues = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(crispr_i), 1)))
            for j, guide in enumerate(crispr_i):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRi'),
                               'log2fc_x'].values
                _kde_plot(vals, color=blues[j])
            legend_patches.append(
                plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Blues(0.65), alpha=0.6, label='CRISPRi'))

        if crispr_a:
            reds = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(crispr_a), 1)))
            for j, guide in enumerate(crispr_a):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRa'),
                               'log2fc_x'].values
                _kde_plot(vals, color=reds[j])
            legend_patches.append(
                plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Reds(0.65), alpha=0.6, label='CRISPRa'))

        ax.set_xlim(unified_x)
        ax.set_xlabel('log2FC(x_true)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.legend(handles=legend_patches, frameon=False, fontsize=8)

    _plot_one(ax_a, model_a, LABEL_A)
    _plot_one(ax_b, model_b, LABEL_B)
    fig.suptitle(f'{cis_gene}  |  guide distributions', fontsize=10)
    return fig


# ── Loop over all overlapping genes ──────────────────────────────────────────

print(f"Plotting panels into {PLOT_DIR}/...\n")

# x_true is cis gene expression — the same for every trans gene.
# Collect the unified x range across all genes first, then make the density
# plot once.  We use a two-pass approach: first pass builds all panels and
# records their unified x ranges; second pass is just saving (already done).

unified_x_all = None  # will expand to cover all genes

for i, goi in enumerate(overlap, 1):
    print(f"  [{i}/{len(overlap)}] {goi}", end='', flush=True)

    fig_panel, unified_x = make_panel(goi)
    fig_panel.savefig(os.path.join(PLOT_DIR, f'{goi}_panel.png'),
                      dpi=150, bbox_inches='tight')
    plt.close(fig_panel)

    # Expand the running x range.
    if unified_x_all is None:
        unified_x_all = unified_x
    else:
        unified_x_all = (min(unified_x_all[0], unified_x[0]),
                         max(unified_x_all[1], unified_x[1]))

    print("  done")

# One density panel for the whole run (x_true doesn't vary with trans gene).
print("Plotting guide density panel...", end='', flush=True)
fig_density = make_density_panel(cis_gene, unified_x_all)
fig_density.savefig(os.path.join(PLOT_DIR, f'{cis_gene}_guide_density.png'),
                    dpi=150, bbox_inches='tight')
plt.close(fig_density)
print("  done")

print(f"\nDone. {len(overlap)} panels + 1 density plot written to {PLOT_DIR}/")
