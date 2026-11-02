"""
Replogle (each fit in replogle_variants.COMPARED) against Domingo and Morris,
for one cis gene:

- trans: the same fitted parameters as the within-Replogle comparison, one
  scatter per (external dataset, Replogle fit), via trans_param_compare's
  merge_pair()/scatter_param() (gene_id join for Morris, symbol join for
  Domingo -- see merge_pair()).
- guides: guide-level knockdown (log2FC of x_eff_g vs the fit's own NTC
  guides, bayesDREAM's cis_guide_summary.csv log2fc_x_eff_g_* definition)
  per guide, and side by side for guides whose SEQUENCE is shared between
  Replogle and Domingo/Morris. Same matching rule as
  Panten2026/cis_guide_comparison_export.py and
  bayesDREAM_analysis_scratch/cis_guide_fit_comparison.ipynb: exact,
  uppercased sequence, no reverse complement. Replogle is dual-guide: one fit
  per guide pair ('sgID_A|sgID_B'), matched if either sequence is shared.

Domingo/Morris guide tables are their runs' own cis_guide_summary.csv
(written by bayesDREAM_analysis_scratch/generate_cis_summaries_publication.py).
Domingo is restricted to its CRISPRi arm (Replogle is CRISPRi only).

Usage
-----
    from comparative.replogle_cross import (
        external_specs, plot_cross_grid, cross_dependency_table,
        guide_effects, shared_guide_table, plot_guide_effects, plot_shared_guides)
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .datasets import DOMINGO, MORRIS, DatasetSpec
from .replogle_variants import COMPARED, VARIANT_COLORS, _ordered, _wrap_guide_label, cis_guide_posterior
from .trans_param_compare import _global_param_lims, merge_pair, scatter_param

LEAH = '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah'

# guide name -> sequence per study (same sources as cis_guide_fit_comparison.ipynb).
# seq_len: keep only the first N letters (Domingo's GUIDE_SEQ carries extra bases).
GUIDE_SEQ_SOURCES = {
    'Domingo': dict(path=os.path.join(LEAH, 'data/Domingo2024/processed_Leah/preprocessed/cell_meta.csv'),
                    id_col='guide', seq_col='GUIDE_SEQ', seq_len=20),
    'Morris': dict(path=os.path.join(LEAH, 'data/Morris2023/processed_Leah/guidemeta.csv'),
                   id_col='gRNA ID new', seq_col='sgRNA guide sequence', seq_len=None),
}
# Replogle dual-guide libraries: sgID_A/B + 'targeting sequence A/B' per pair.
REPLOGLE_GUIDE_LIBRARIES = [
    os.path.join(LEAH, 'data/Replogle2022/input', f)
    for f in ('K562_essential_gRNA_library.csv', 'K562_gwps_gRNA_library.csv')
]
EXTERNAL = [DOMINGO, MORRIS]
SOURCE_COLORS = {'Domingo': DOMINGO.color, 'Morris': MORRIS.color, **VARIANT_COLORS}


def external_specs(gene: str) -> List[DatasetSpec]:
    """Domingo/Morris specs that have a completed fit for `gene`."""
    return [s for s in EXTERNAL if gene in s.cis_genes]


# ── trans ─────────────────────────────────────────────────────────────────────

def cross_dependency_table(rep_dfs: Dict[str, pd.DataFrame], ext_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (external dataset, Replogle fit): join key, genes in both
    panels, dependent genes in each (within the shared panel), dependent in
    both."""
    rows = []
    for e, de in ext_dfs.items():
        for r in _ordered(rep_dfs):
            m = merge_pair(de, rep_dfs[r], e, r)
            de_dep, dr_dep = m[f'is_dependent_{e}'], m[f'is_dependent_{r}']
            on = 'gene_id' if (de['gene_id'].notna().any() and rep_dfs[r]['gene_id'].notna().any()) else 'gene_symbol'
            rows.append(dict(external=e, replogle_fit=r, join_on=on, n_shared_genes=len(m),
                             n_dep_external=int(de_dep.sum()), n_dep_replogle=int(dr_dep.sum()),
                             n_dep_both=int((de_dep & dr_dep).sum())))
    return pd.DataFrame(rows)


def plot_cross_grid(rep_dfs: Dict[str, pd.DataFrame], ext_dfs: Dict[str, pd.DataFrame], param: str,
                    cis_gene: str = '', *, exclude_categories=('neither',), figsize_per=(3.8, 3.8)):
    """Rows = external datasets (x axis), columns = Replogle fits (y axis);
    one scatter_param() per cell, all on one shared axis range, coloured
    by which side calls the gene dependent ('neither' dropped by default)."""
    import matplotlib.pyplot as plt

    reps, exts = _ordered(rep_dfs), list(ext_dfs)
    fig, axes = plt.subplots(len(exts), len(reps), figsize=(figsize_per[0] * len(reps), figsize_per[1] * len(exts)),
                             constrained_layout=True, squeeze=False)
    lims = _global_param_lims({**ext_dfs, **rep_dfs}, param)
    for i, e in enumerate(exts):
        for j, r in enumerate(reps):
            ax = axes[i][j]
            try:
                merged = merge_pair(ext_dfs[e], rep_dfs[r], e, r)
                scatter_param(merged, param, e, r, ax=ax, color_by='dependency_category',
                              title=f'{e} vs {r}', lims=lims, exclude_categories=exclude_categories)
            except (KeyError, ValueError) as err:
                ax.text(0.5, 0.5, str(err), ha='center', va='center', wrap=True, fontsize=7)
                ax.axis('off')
    fig.suptitle(f'{cis_gene}: {param}, Replogle vs Domingo/Morris'.strip(': '), fontsize=11)
    return fig


# ── guides ────────────────────────────────────────────────────────────────────

def _norm_seq(s: pd.Series, seq_len: Optional[int] = None) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    return s.str[:seq_len] if seq_len else s


@lru_cache(maxsize=None)
def _seq_lookup(study: str) -> pd.DataFrame:
    """guide -> guide_seq for Domingo/Morris; one row per guide."""
    src = GUIDE_SEQ_SOURCES[study]
    d = pd.read_csv(src['path'], usecols=[src['id_col'], src['seq_col']])
    d = d.rename(columns={src['id_col']: 'guide', src['seq_col']: 'guide_seq'}).dropna()
    d['guide'] = d['guide'].astype(str)
    d['guide_seq'] = _norm_seq(d['guide_seq'], src['seq_len'])
    d = d.drop_duplicates()
    amb = d['guide'][d['guide'].duplicated()]
    if len(amb):
        raise ValueError(f"{src['path']}: {amb.nunique()} guide name(s) map to >1 sequence, e.g. {amb.iloc[0]}")
    return d


@lru_cache(maxsize=None)
def _replogle_library() -> pd.DataFrame:
    """sgID -> guide_seq, union of both Replogle libraries."""
    parts = []
    for path in REPLOGLE_GUIDE_LIBRARIES:
        lib = pd.read_csv(path)
        for ab in 'AB':
            parts.append(pd.DataFrame({'sgID': lib[f'sgID_{ab}'].astype(str),
                                       'guide_seq': _norm_seq(lib[f'targeting sequence {ab}'])}))
    d = pd.concat(parts).drop_duplicates()
    amb = d['sgID'][d['sgID'].duplicated()]
    if len(amb):
        raise ValueError(f"Replogle libraries: {amb.nunique()} sgID(s) map to >1 sequence, e.g. {amb.iloc[0]}")
    return d


def _external_guides(spec: DatasetSpec, gene: str) -> pd.DataFrame:
    path = os.path.join(spec.run_dir_fn(gene), 'cis_guide_summary.csv')
    if not os.path.exists(path):
        print(f"[{spec.name}/{gene}] no cis_guide_summary.csv at {path} -- run "
              f"generate_cis_summaries_publication.py for it")
        return pd.DataFrame()
    g = pd.read_csv(path)
    target = g['target'] if 'target' in g.columns else g.get('targets', pd.Series('', index=g.index))
    target = target.astype(str)
    g = g[~(target.str.lower().eq('ntc') | target.str.contains('non-targeting', case=False))]
    if 'cell_line' in g.columns:
        g = g[g['cell_line'].astype(str).eq('CRISPRi')]
    out = pd.DataFrame({
        'source': spec.name, 'guide': g['guide'].astype(str),
        'n_cells': g['n_cells'] if 'n_cells' in g.columns else np.nan,
        'lfc_median': g['log2fc_x_eff_g_median'], 'lfc_lower': g['log2fc_x_eff_g_lower'],
        'lfc_upper': g['log2fc_x_eff_g_upper'],
    })
    return out.merge(_seq_lookup(spec.name), on='guide', how='left')


def _replogle_guides(gene: str, variants: List[str]) -> pd.DataFrame:
    frames = []
    for v in variants:
        gp = cis_guide_posterior(gene, v)
        if gp is None or 'log2fc_x_eff_median' not in gp.columns:
            print(f"[Replogle/{gene}/{v}] no guide-level x_eff_g -- skipped")
            continue
        gp = gp[~gp['is_ntc']]
        frames.append(pd.DataFrame({
            'source': v, 'guide': gp['guide'], 'n_cells': gp['n_cells'],
            'lfc_median': gp['log2fc_x_eff_median'], 'lfc_lower': gp['log2fc_x_eff_lower'],
            'lfc_upper': gp['log2fc_x_eff_upper'],
        }))
    rep = pd.concat(frames, ignore_index=True)
    # one row per constituent sgID of each guide pair
    sg = rep[['guide']].drop_duplicates().assign(sgID=lambda d: d['guide'].str.split('|')).explode('sgID')
    sg = sg.merge(_replogle_library(), on='sgID', how='left')
    missing = sg.loc[sg['guide_seq'].isna(), 'sgID'].unique()
    if len(missing):
        print(f"[Replogle/{gene}] {len(missing)} sgID(s) not in the libraries, e.g. {missing[0]}")
    return rep.merge(sg[['guide', 'guide_seq']], on='guide', how='left')


def guide_effects(gene: str, variants: List[str] = COMPARED) -> pd.DataFrame:
    """Long table, one row per (source, guide[, sequence]): targeting
    guides' log2FC x_eff_g (median, 95% CI) vs each fit's own NTC guides.
    source = 'Domingo' (CRISPRi), 'Morris', or a Replogle fit name.
    Replogle guide pairs appear once per constituent sequence; 'shared'
    marks sequences also present in another study."""
    parts = [_external_guides(s, gene) for s in external_specs(gene)] + [_replogle_guides(gene, variants)]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df['study'] = np.where(df['source'].isin(['Domingo', 'Morris']), df['source'], 'Replogle')
    n_studies = df.dropna(subset=['guide_seq']).groupby('guide_seq')['study'].nunique()
    df['shared'] = df['guide_seq'].map(n_studies).fillna(0).ge(2)
    return df


def shared_guide_table(effects: pd.DataFrame) -> pd.DataFrame:
    """Sequences shared between Replogle and Domingo/Morris: one row per
    sequence, one log2FC column per source, plus each source's guide name."""
    sh = effects[effects['shared'] & effects['study'].eq('Replogle')]['guide_seq'].unique()
    d = effects[effects['guide_seq'].isin(sh)]
    if d.empty:
        return pd.DataFrame()
    lfc = d.pivot_table(index='guide_seq', columns='source', values='lfc_median', aggfunc='first')
    names = d.drop_duplicates(['guide_seq', 'study']).pivot(index='guide_seq', columns='study', values='guide')
    cols = [c for c in ['Domingo', 'Morris'] if c in lfc.columns] + _ordered(lfc.columns)
    return names.add_prefix('guide_').join(lfc[cols].round(3))


def plot_guide_effects(effects: pd.DataFrame, *, title: str = ''):
    """Every targeting guide's knockdown, one row per source; guides whose
    sequence is shared across studies are outlined black (Replogle's guide
    pair is drawn once, outlined if either sequence is shared)."""
    import matplotlib.pyplot as plt

    d = (effects.sort_values('shared', ascending=False)
                .drop_duplicates(['source', 'guide']))
    sources = [s for s in ['Domingo', 'Morris'] if s in set(d['source'])] + _ordered(d['source'])
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(sources) + 1.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, s in enumerate(sources):
        sd = d[d['source'] == s]
        y = i + rng.uniform(-0.15, 0.15, len(sd))
        ax.scatter(sd['lfc_median'], y, s=25, color=SOURCE_COLORS.get(s, 'grey'),
                   edgecolors=np.where(sd['shared'], 'black', 'none'), linewidths=1.0, zorder=3)
        ax.text(1.01, i, f'{len(sd)} guide(s)', transform=ax.get_yaxis_transform(), va='center', fontsize=7)
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_yticks(range(len(sources))); ax.set_yticklabels(sources)
    ax.set_ylim(len(sources) - 0.5, -0.5)
    ax.set_xlabel('log2FC x_eff_g vs own NTC guides (median)')
    ax.set_title(f'{title}targeting-guide knockdown per dataset (black outline: sequence shared across studies)',
                 fontsize=10)
    return fig


def plot_shared_guides(effects: pd.DataFrame, *, title: str = ''):
    """Point-range (median, 95% CI) per shared sequence, one point per source."""
    import matplotlib.pyplot as plt

    sh = effects[effects['shared'] & effects['study'].eq('Replogle')]['guide_seq'].unique()
    d = effects[effects['guide_seq'].isin(sh)]
    if d.empty:
        return None
    sources = [s for s in ['Domingo', 'Morris'] if s in set(d['source'])] + _ordered(d['source'])
    seqs = sorted(sh)
    # label: sequence + each study's guide name
    names = d.drop_duplicates(['guide_seq', 'study']).groupby('guide_seq').apply(
        lambda g: '\n'.join(f"{r.study}: {_wrap_guide_label(r.guide, 40)}" for r in g.itertuples()))
    labels = [f'{s}\n{names[s]}' for s in seqs]
    heights = np.array([max(lab.count('\n') + 1, len(sources)) * 0.2 for lab in labels])
    ypos = np.cumsum(heights) - heights / 2
    fig, ax = plt.subplots(figsize=(8, heights.sum() + 1.5), constrained_layout=True)
    offs = np.linspace(-0.3, 0.3, len(sources)) * heights.min()
    for off, s in zip(offs, sources):
        sd = d[d['source'] == s].drop_duplicates('guide_seq').set_index('guide_seq').reindex(seqs)
        ax.errorbar(sd['lfc_median'], ypos + off,
                    xerr=[sd['lfc_median'] - sd['lfc_lower'], sd['lfc_upper'] - sd['lfc_median']],
                    fmt='o', ms=4, lw=1, color=SOURCE_COLORS.get(s, 'grey'), label=s)
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(heights.sum(), 0)
    ax.set_xlabel('log2FC x_eff_g vs own NTC guides (median, 95% CI)')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_title(f'{title}guides shared by sequence', fontsize=11)
    return fig
