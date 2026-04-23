"""
Diagnostic plotting functions.

Provides quality control and diagnostic plots for bayesDREAM fits.
"""

import numpy as np
import matplotlib.pyplot as plt

from .colors import ColorScheme


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
