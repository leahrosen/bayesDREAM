"""
Within-dataset comparison of the three Replogle publication-run fit variants
(publication_runs/replogle/, label prefix replogle_20260908):

    bm_indmu    batch-matched NTC subset, independent_mu_sigma=True
    bm_noindmu  batch-matched NTC subset, independent_mu_sigma=False
    all_indmu   all-NTC subset,           independent_mu_sigma=True

Unlike the rest of comparative/ (Domingo vs Morris vs Replogle), this
compares the SAME dataset/cis gene across fit configurations, so every
variant here is treated as its own "dataset" by trans_param_compare.py's
existing pairwise machinery (load_trans_summary/plot_pairwise_grid) -- see
variant_specs().

Two halves, matching the two questions asked of these fits:

- cis: per-cell x_true posterior (median + tightness) straight from each
  variant's posterior_samples_cis.pt -- no model rebuild needed. Cells are
  joined across variants by barcode: the cis gene's own target cells are
  shared by all three; bm's NTC cells are a subset of all_indmu's.
- trans: n_a / observed_log2fc / full_log2fc read from each variant's own
  production trans_feature_summary_gene.csv, plus the fitted curve's
  y-log2FC at x log2FC = -1 (comparative/hill_eval.py), which needs the
  full trans posterior and therefore a real model reload -- see
  backfill_variant().

Production output directories are never written to: backfilled summaries go
to CACHE_DIR/<gene>/<variant>/trans_feature_summary_gene.csv.

Usage
-----
    from comparative.replogle_variants import (
        load_cis_cells, backfill_variant, variant_specs)

    cells = load_cis_cells('GFI1B')                 # long per-cell table, all 3 variants
    for v in VARIANTS:
        backfill_variant('GFI1B', v)                # once per (gene, variant); cached
    dfs = load_all(variant_specs(), 'GFI1B')        # trans_param_compare.load_all
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import bayesDREAM as _bayesdream_pkg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_bayesdream_pkg.__file__)))
_COMMON_DIR = os.path.join(REPO_ROOT, 'publication_runs', 'common')
for _p in (REPO_ROOT, _COMMON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_utils import (  # noqa: E402
    build_model_from_config,
    load_bayesdream_yaml,
    normalize_stage_args,
    is_enabled,
    apply_sum_factor_adjustments,
    apply_device_override,
)

from .datasets import DatasetSpec, PALETTE_B, REPLOGLE_GENE_TO_ID  # noqa: E402
from .hill_eval import add_log2fc_at_columns, get_x_ntc, tag_for_log2fc, HILL_LOG2FC_TARGETS  # noqa: E402


# ── Paths / naming (publication_runs/replogle/config.yaml) ───────────────────
LABEL_PREFIX = 'replogle_20260908'
OUTPUT_DIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Replogle2022/'
    'processed_Leah/BayesianModel_outs'
)
CONFIGS_DIR = os.path.join(REPO_ROOT, 'publication_runs', 'replogle', 'slurm', 'configs')
CACHE_DIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/CRISPRmodelling/'
    'BayesianModelling/Comparative/replogle_variants'
)

VARIANTS = ['bm_indmu', 'bm_noindmu', 'all_indmu']
VARIANT_DATA = {'bm_indmu': 'bm', 'bm_noindmu': 'bm', 'all_indmu': 'all'}
VARIANT_COLORS = {'bm_indmu': '#1b9e77', 'bm_noindmu': '#d95f02', 'all_indmu': '#7570b3'}
CIS_GENES = sorted(REPLOGLE_GENE_TO_ID)


def run_dir(gene: str, variant: str) -> str:
    """Production output dir shared by this (gene, variant)'s cis and trans fits."""
    return os.path.join(OUTPUT_DIR, f'{LABEL_PREFIX}_{gene}_{variant}')


def subset_dir(gene: str, variant: str, mode: str = 'cis_only') -> str:
    return os.path.join(OUTPUT_DIR, f'{LABEL_PREFIX}_{gene}_{VARIANT_DATA[variant]}_subset', mode)


def cache_dir(gene: str, variant: str) -> str:
    return os.path.join(CACHE_DIR, gene, variant)


# ── cis: per-cell x_true posterior ───────────────────────────────────────────

def _as_samples_by_cells(v, n_cells: int) -> np.ndarray:
    """[S, N] numpy array from a posterior tensor that may carry extra
    singleton axes (e.g. [S, 1, N] from a plate)."""
    a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    a = a.reshape(a.shape[0], -1)
    if a.shape[1] != n_cells:
        raise ValueError(f"expected {n_cells} cells along the last axis, got shape {a.shape}")
    return a


def load_cis_posterior(gene: str, variant: str) -> dict:
    """Raw posterior_samples_cis.pt dict (full samples, not the lean file)."""
    path = os.path.join(run_dir(gene, variant), 'posterior_samples_cis.pt')
    return torch.load(path, map_location='cpu', weights_only=False)


def cis_cell_summary(gene: str, variant: str) -> pd.DataFrame:
    """One row per cell for one variant's cis fit.

    log2_x_true_* are quantiles of the per-sample log2(x_true) posterior;
    tightness is reported two ways, both in log2 units: the 95% CI width
    (q97.5 - q2.5) and the posterior SD. log2fc_x_true is relative to this
    variant's OWN x_ntc (median x_true over its NTC cells, the same
    reference hill_eval.get_x_ntc() and EC50_a_log2fc use), so it stays
    comparable across variants even if the absolute x scale shifts.
    """
    post = load_cis_posterior(gene, variant)
    cells = post.get('cell_names')
    if cells is None:
        raise KeyError(f"[{gene}/{variant}] posterior_samples_cis.pt has no cell_names")
    ps = post['posterior_samples']
    x = _as_samples_by_cells(ps['x_true'], len(cells))
    lx = np.log2(np.clip(x, 1e-12, None))

    df = pd.DataFrame({
        'cell': cells,
        'log2_x_true_median': np.median(lx, axis=0),
        'log2_x_true_lower': np.quantile(lx, 0.025, axis=0),
        'log2_x_true_upper': np.quantile(lx, 0.975, axis=0),
        'log2_x_true_sd': lx.std(axis=0),
    })
    df['ci_width'] = df['log2_x_true_upper'] - df['log2_x_true_lower']

    meta = pd.read_csv(os.path.join(subset_dir(gene, variant), 'meta.csv'),
                       usecols=lambda c: c in {'cell', 'guide', 'target'})
    df = df.merge(meta, on='cell', how='left')
    if df['target'].isna().any():
        raise ValueError(f"[{gene}/{variant}] {int(df['target'].isna().sum())} cell(s) in the cis "
                         f"posterior not found in {subset_dir(gene, variant)}/meta.csv")
    df['is_ntc'] = df['target'].astype(str).str.lower().eq('ntc')

    log2_x_ntc = float(np.log2(np.median(np.median(x[:, df['is_ntc'].values], axis=0))))
    df['log2fc_x_true'] = df['log2_x_true_median'] - log2_x_ntc
    df['log2_x_ntc'] = log2_x_ntc
    df['variant'] = variant
    return df


def load_cis_cells(gene: str, variants: List[str] = VARIANTS) -> pd.DataFrame:
    """Long table: cis_cell_summary() for every variant, stacked."""
    return pd.concat([cis_cell_summary(gene, v) for v in variants], ignore_index=True)


def cis_cells_wide(cells: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot one column of load_cis_cells() to cell x variant, keeping
    guide/target/is_ntc. Cells missing from a variant (all_indmu's extra NTC
    cells, absent from bm) are NaN there."""
    wide = cells.pivot(index='cell', columns='variant', values=value)
    ann = cells.drop_duplicates('cell').set_index('cell')[['guide', 'target', 'is_ntc']]
    return ann.join(wide)


def cis_guide_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, guide): n cells, median of per-cell log2 x_true
    medians, median per-cell log2FC vs own NTC, and median per-cell CI
    width / SD."""
    return (cells.groupby(['variant', 'target', 'guide'], observed=True)
                 .agg(n_cells=('cell', 'size'),
                      log2_x_true_median=('log2_x_true_median', 'median'),
                      log2fc_x_true=('log2fc_x_true', 'median'),
                      ci_width_median=('ci_width', 'median'),
                      sd_median=('log2_x_true_sd', 'median'))
                 .reset_index())


def cis_guide_posterior(gene: str, variant: str) -> Optional[pd.DataFrame]:
    """Guide-level x_eff_g posterior (log2 median + 95% CI) per guide, using
    the guide_axis_labels saved with the fit. None if the fit has no
    x_eff_g or no labels (legacy files).

    The saved labels are bayesDREAM's guide_used values,
    f"{guide}_{'_'.join(covariate values)}" (core.py). Replogle has no
    guide covariates, so each is the guide name plus a trailing '_';
    'guide' strips it so this joins against meta['guide'], and
    'guide_used' keeps the raw label."""
    post = load_cis_posterior(gene, variant)
    labels = post.get('guide_axis_labels')
    ps = post['posterior_samples']
    if labels is None or 'x_eff_g' not in ps:
        return None
    lx = np.log2(np.clip(_as_samples_by_cells(ps['x_eff_g'], len(labels)), 1e-12, None))
    return pd.DataFrame({
        'variant': variant, 'guide_used': labels,
        'guide': [lab[:-1] if lab.endswith('_') else lab for lab in labels],
        'log2_x_eff_median': np.median(lx, axis=0),
        'log2_x_eff_lower': np.quantile(lx, 0.025, axis=0),
        'log2_x_eff_upper': np.quantile(lx, 0.975, axis=0),
    })


def cis_variant_overview(cells: pd.DataFrame) -> pd.DataFrame:
    """Per variant: cell counts, the NTC reference, and median x_true / CI
    width / SD separately for target and NTC cells."""
    g = cells.assign(group=np.where(cells['is_ntc'], 'ntc', 'target'))
    out = (g.groupby(['variant', 'group'])
             .agg(n_cells=('cell', 'size'),
                  log2fc_x_true=('log2fc_x_true', 'median'),
                  ci_width=('ci_width', 'median'),
                  sd=('log2_x_true_sd', 'median'))
             .unstack('group'))
    out.columns = [f'{a}_{b}' for a, b in out.columns]
    out['log2_x_ntc'] = cells.groupby('variant')['log2_x_ntc'].first()
    return out.loc[[v for v in VARIANTS if v in out.index]]


def plot_cis_pairwise(cells: pd.DataFrame, value: str = 'log2_x_true_median', *,
                      title: str = '', target_only: bool = False):
    """One scatter per variant pair of a per-cell column (default: log2
    x_true posterior median), on cells present in both. Target cells are
    coloured by guide; NTC cells grey underneath. Identity line dashed."""
    import matplotlib.pyplot as plt
    from itertools import combinations

    wide = cis_cells_wide(cells, value)
    if target_only:
        wide = wide[~wide['is_ntc']]
    variants = [v for v in VARIANTS if v in wide.columns]
    pairs = list(combinations(variants, 2))
    vals = wide[variants].to_numpy(dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = 0.03 * (hi - lo)
    lims = (lo - pad, hi + pad)

    guides = sorted(wide.loc[~wide['is_ntc'], 'guide'].dropna().unique())
    cmap = plt.get_cmap('tab10' if len(guides) <= 10 else 'tab20')
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.2), constrained_layout=True,
                             squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        sub = wide.dropna(subset=[a, b])
        ntc = sub[sub['is_ntc']]
        if len(ntc):
            ax.scatter(ntc[a], ntc[b], s=4, c='lightgrey', alpha=0.5, label=f'NTC (n={len(ntc)})',
                       rasterized=True)
        for i, gd in enumerate(guides):
            s = sub[(~sub['is_ntc']) & (sub['guide'] == gd)]
            if len(s):
                ax.scatter(s[a], s[b], s=8, color=cmap(i % cmap.N), alpha=0.8, label=f'{gd} (n={len(s)})')
        ax.plot(lims, lims, ls='--', c='k', lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(a); ax.set_ylabel(b)
        r = sub[[a, b]].corr().iloc[0, 1]
        ax.set_title(f'{a} vs {b}  (r={r:.3f}, n={len(sub)})', fontsize=9)
    axes[0][0].legend(fontsize=6, markerscale=2, loc='upper left')
    fig.suptitle(f'{title}{value}' + (' (targeting cells only)' if target_only else ''), fontsize=11)
    return fig


def plot_cis_tightness(cells: pd.DataFrame, *, title: str = ''):
    """Distributions of per-cell posterior tightness (95% CI width and SD
    of log2 x_true) per variant, split target vs NTC."""
    import matplotlib.pyplot as plt

    variants = [v for v in VARIANTS if v in set(cells['variant'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, col, lab in zip(axes, ['ci_width', 'log2_x_true_sd'],
                            ['95% CI width of log2 x_true', 'posterior SD of log2 x_true']):
        data, labels, colors = [], [], []
        for v in variants:
            for grp, is_ntc in (('target', False), ('NTC', True)):
                s = cells.loc[(cells['variant'] == v) & (cells['is_ntc'] == is_ntc), col].dropna()
                data.append(s.values); labels.append(f'{v}\n{grp}'); colors.append(VARIANT_COLORS[v])
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, c, lbl in zip(bp['boxes'], colors, labels):
            patch.set_facecolor(c); patch.set_alpha(0.35 if 'NTC' in lbl else 0.8)
        ax.set_ylabel(lab); ax.tick_params(axis='x', labelsize=7)
    fig.suptitle(f'{title}x_true posterior tightness', fontsize=11)
    return fig


def plot_guide_posteriors(guide_post: pd.DataFrame, *, title: str = ''):
    """Point-range plot of guide-level log2 x_eff_g (median + 95% CI) per
    variant, guides on the y axis. Only guides present in EVERY variant are
    shown (the cis gene's own guides + bm's curated NTC guides) --
    all_indmu's full NTC guide panel would otherwise swamp the plot."""
    import matplotlib.pyplot as plt

    variants = [v for v in VARIANTS if v in set(guide_post['variant'])]
    per_variant = [set(guide_post.loc[guide_post['variant'] == v, 'guide']) for v in variants]
    guides = sorted(set.intersection(*per_variant))
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(guides) + 1.5), constrained_layout=True)
    offs = np.linspace(-0.25, 0.25, len(variants))
    for off, v in zip(offs, variants):
        s = guide_post[guide_post['variant'] == v].set_index('guide').reindex(guides)
        y = np.arange(len(guides)) + off
        ax.errorbar(s['log2_x_eff_median'], y,
                    xerr=[s['log2_x_eff_median'] - s['log2_x_eff_lower'],
                          s['log2_x_eff_upper'] - s['log2_x_eff_median']],
                    fmt='o', ms=3, lw=1, color=VARIANT_COLORS[v], label=v)
    ax.set_yticks(np.arange(len(guides))); ax.set_yticklabels(guides, fontsize=7)
    ax.set_xlabel('log2 x_eff_g (median, 95% CI)')
    ax.legend(fontsize=8)
    ax.set_title(f'{title}guide-level cis expression', fontsize=11)
    return fig


# ── trans: backfill y-log2FC at x log2FC = -1, then reuse trans_param_compare ─

def _trans_config(gene: str, variant: str) -> dict:
    path = os.path.join(CONFIGS_DIR, f'{LABEL_PREFIX}_{gene}_{variant}_trans.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"no rendered trans config at {path!r} -- run generate_slurm.py first")
    return load_bayesdream_yaml(Path(path))


def reconstruct_variant_model(gene: str, variant: str, *, device: Optional[str] = None):
    """Rebuild the fitted trans model for (gene, variant) by replaying its
    real <label>_trans.yaml exactly as publication_runs/common/run_trans.py
    ran it (eager cis_gene, load_ntc lean, load_cis, exclude_trans_genes),
    then load_trans_fit() instead of fit_trans(). Nothing is re-fit.
    Returns (model, modality_name)."""
    cfg = _trans_config(gene, variant)
    apply_device_override(cfg, device)
    model = build_model_from_config(cfg)

    trans_cfg = cfg.get('trans') or {}
    ntc_cfg = cfg.get('ntc') or cfg.get('technical') or {}
    if is_enabled(trans_cfg.get('load_ntc'), default=True):
        model.load_ntc_fit(**normalize_stage_args(trans_cfg.get('load_ntc')))
    if is_enabled(trans_cfg.get('load_cis'), default=True):
        model.load_cis_fit(**normalize_stage_args(trans_cfg.get('load_cis')))
    if 'technical_group_code' not in model.meta.columns:
        covariates = ntc_cfg.get('set_technical_groups')
        if covariates:
            model.set_technical_groups(covariates)
    apply_sum_factor_adjustments(model, cfg.get('sum_factor') or {})
    excl_cfg = cfg.get('exclude_trans_genes') or {}
    if is_enabled(excl_cfg, default=False):
        model.exclude_trans_genes(**normalize_stage_args(excl_cfg))

    fit_args = normalize_stage_args(trans_cfg.get('fit'))
    modality_name = fit_args.get('modality_name') or model.primary_modality
    model.load_trans_fit(modalities=[modality_name], subset_features=True)
    return model, modality_name


def _backfilled_path(gene: str, variant: str, modality_name: str = 'gene') -> str:
    return os.path.join(cache_dir(gene, variant), f'trans_feature_summary_{modality_name}.csv')


def backfill_variant(gene: str, variant: str, *, targets=HILL_LOG2FC_TARGETS,
                     device: Optional[str] = None, force: bool = False) -> str:
    """Write CACHE_DIR/<gene>/<variant>/trans_feature_summary_gene.csv: the
    variant's PRODUCTION summary (unchanged columns, incl. observed/full
    log2FC CIs) plus hill_eval's y_at_x_log2fc{...} columns, joined on
    'feature'. Skips if the cached file already has those columns, unless
    force=True. Needs a full model reload (minutes, several GB).
    """
    out_path = _backfilled_path(gene, variant)
    needed = {f'y_at_{tag_for_log2fc(t)}_log2fc_median_allgenes' for t in targets}
    if not force and os.path.exists(out_path) and needed.issubset(pd.read_csv(out_path, nrows=0).columns):
        print(f"[{gene}/{variant}] already backfilled -> {out_path}")
        return out_path

    prod_path = os.path.join(run_dir(gene, variant), 'trans_feature_summary_gene.csv')
    prod = pd.read_csv(prod_path, low_memory=False)

    print(f"[{gene}/{variant}] reconstructing model from {LABEL_PREFIX}_{gene}_{variant}_trans.yaml ...")
    model, modality_name = reconstruct_variant_model(gene, variant, device=device)

    # Fast summary only to get is_dependent/y_ntc aligned with the model's own
    # feature order for add_log2fc_at_columns(); written to the cache dir, then
    # overwritten below with the merged production version.
    os.makedirs(cache_dir(gene, variant), exist_ok=True)
    fast = model.save_trans_summary(
        output_dir=cache_dir(gene, variant), modality_name=modality_name, x_ntc=get_x_ntc(model),
        compute_derivative_roots=False, compute_inflection=False, compute_lfc_ci=False,
    )
    fast = add_log2fc_at_columns(model, fast, modality_name=modality_name, targets=targets)
    del model
    gc.collect()

    y_cols = [c for c in fast.columns if c.startswith('y_at_x_log2fc')]
    merged = prod.drop(columns=[c for c in y_cols if c in prod.columns]).merge(
        fast[['feature'] + y_cols], on='feature', how='left', validate='one_to_one')
    n_missing = int(merged[y_cols[0]].isna().sum()) if y_cols else 0
    if n_missing:
        print(f"[{gene}/{variant}] note: {n_missing} production row(s) got no y_at_* value "
              f"(e.g. the cis gene's own row)")
    merged.to_csv(out_path, index=False)
    print(f"[{gene}/{variant}] wrote {out_path}")
    return out_path


def variant_specs(variants: List[str] = VARIANTS) -> List[DatasetSpec]:
    """One DatasetSpec per variant, reading the backfilled cache summaries,
    so trans_param_compare.load_all()/plot_pairwise_grid() treat each
    variant as a dataset. 'feature' is the Ensembl ID (feature_name_col:
    gene_id); 'gene_name' is the symbol."""
    return [
        DatasetSpec(
            name=v, color=VARIANT_COLORS[v], cell_line_palette=PALETTE_B, cis_genes=CIS_GENES,
            symbol_col='gene_name', run_dir_fn=(lambda g, v=v: cache_dir(g, v)),
            cis_gene_id_fn=REPLOGLE_GENE_TO_ID.get, plot_sum_factor_col='sum_factor_adj',
        )
        for v in variants
    ]


def dependency_overlap(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Number of is_dependent trans genes per variant (diagonal) and shared
    between each pair (off-diagonal), keyed on gene_id."""
    sets = {k: set(d.loc[d['is_dependent'], 'gene_id']) for k, d in dfs.items()}
    names = list(sets)
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in names] for a in names], index=names, columns=names)
