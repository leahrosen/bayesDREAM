"""
Differential expression comparison functions.

Compares bayesDREAM results with external DE methods (e.g., edgeR).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .helpers import to_np
from .colors import lighten, darken, ColorScheme
from .utils import dependency_mask_from_n


def compute_log2fc_metrics(A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
                           x_true_samps, eps=1e-6):
    """
    Compute directional log2 fold-change metrics for the Hill-based model:

        y(x) = A + alpha * Vmax * x^n / (K^n + x^n)

    Parameters
    ----------
    A_samps, alpha_samps, Vmax_samps, K_samps, n_samps : array-like, shape (n_samples, n_features)
        Posterior samples for each parameter
    x_true_samps : array-like, shape (n_samples, n_cells)
        Posterior samples of x_true for this cis gene
    eps : float
        Small constant for numerical stability

    Returns
    -------
    log2fc_full : np.ndarray, shape (n_samples, n_features)
        Full-range log2FC: y(x→∞) vs y(x→0), directional w.r.t. x increasing
        - n > 0: log2( (A+αVmax) / A )
        - n < 0: log2( A / (A+αVmax) )
    log2fc_obs : np.ndarray, shape (n_samples, n_features)
        Observed-range log2FC: y(x_max_obs) vs y(x_min_obs), directional in x
    x_min_obs, x_max_obs : float
        Observed min/max of mean x_true across cells
    """
    # ensure arrays
    A_samps     = np.asarray(A_samps)
    alpha_samps = np.asarray(alpha_samps)
    Vmax_samps  = np.asarray(Vmax_samps)
    K_samps     = np.asarray(K_samps)
    n_samps     = np.asarray(n_samps)

    # --- observed x range from mean x_true across samples per cell ---
    X = np.asarray(x_true_samps)              # [S, N_cells]
    x_means_per_cell = X.mean(axis=0)         # [N_cells]
    x_min_obs = float(x_means_per_cell.min())
    x_max_obs = float(x_means_per_cell.max())

    A     = A_samps
    alpha = alpha_samps
    Vmax  = Vmax_samps

    # ---------------- full-range FC with n sign ----------------
    # sign of n determines whether y increases or decreases with x
    n_sign = np.sign(n_samps)
    # treat near-zero n as flat (no direction)
    n_sign[np.abs(n_samps) < 1e-6] = 1.0

    y_low  = A
    y_high = A + alpha * Vmax

    # For positive n: FC = y(∞) / y(0) = y_high / y_low
    # For negative n: FC = y(0) / y(∞) = y_low / y_high
    # So directional FC (always in direction of increasing x):
    FC_full = np.where(n_sign > 0, y_high / (y_low + eps), y_low / (y_high + eps))
    log2fc_full = np.log2(FC_full + eps)

    # ---------------- observed-range FC ----------------
    K = K_samps
    n = n_samps

    # Evaluate at observed x_min and x_max
    def hill_eval(x_val):
        xn = np.power(np.abs(x_val) + eps, np.abs(n))
        Kn = np.power(K + eps, np.abs(n))
        return A + alpha * Vmax * xn / (Kn + xn)

    y_at_xmin = hill_eval(x_min_obs)
    y_at_xmax = hill_eval(x_max_obs)

    # Directional FC in direction of increasing x
    FC_obs = np.where(n_sign > 0,
                     y_at_xmax / (y_at_xmin + eps),
                     y_at_xmin / (y_at_xmax + eps))
    log2fc_obs = np.log2(FC_obs + eps)

    return log2fc_full, log2fc_obs, x_min_obs, x_max_obs


def compute_log2fc_obs_for_cells(
    A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
    x_true_samps, cell_mask, guide_labels=None, eps=1e-6
):
    """
    Compute observed-range log2FC for each gene, given a subset of cells.

    Parameters
    ----------
    A_samps, alpha_samps, Vmax_samps, K_samps, n_samps : array-like, shape (n_samples, n_features)
        Posterior samples
    x_true_samps : array-like, shape (n_samples, n_cells)
        Cis x_true samples
    cell_mask : array-like, shape (n_cells,)
        Boolean mask selecting cells to use for x_min/x_max
    guide_labels : array-like, shape (n_cells,), optional
        Guide IDs. If provided, first average x_true per guide, then take min/max
        across guide means
    eps : float
        Numerical stability constant

    Returns
    -------
    log2fc_obs : np.ndarray, shape (n_samples, n_features)
        Per-sample, per-gene, directional log2FC
    x_min_obs, x_max_obs : float
        Observed range in this subset
    """
    A_samps     = np.asarray(A_samps)
    alpha_samps = np.asarray(alpha_samps)
    Vmax_samps  = np.asarray(Vmax_samps)
    K_samps     = np.asarray(K_samps)
    n_samps     = np.asarray(n_samps)
    X           = np.asarray(x_true_samps)

    # subset cells
    X_sub = X[:, cell_mask]  # [S, N_sub]

    # per-cell means over posterior samples
    x_means_per_cell = X_sub.mean(axis=0)  # [N_sub]

    if guide_labels is not None:
        # aggregate by guide first
        guide_labels_sub = np.asarray(guide_labels)[cell_mask]
        unique_guides = np.unique(guide_labels_sub)
        guide_means = []
        for g in unique_guides:
            mask_g = guide_labels_sub == g
            guide_means.append(x_means_per_cell[mask_g].mean())
        x_vals_for_minmax = np.array(guide_means)
    else:
        x_vals_for_minmax = x_means_per_cell

    x_min_obs = float(x_vals_for_minmax.min())
    x_max_obs = float(x_vals_for_minmax.max())

    # Evaluate Hill at observed min/max
    A     = A_samps
    alpha = alpha_samps
    Vmax  = Vmax_samps
    K     = K_samps
    n     = n_samps

    n_sign = np.sign(n)
    n_sign[np.abs(n) < 1e-6] = 1.0

    def hill_eval(x_val):
        xn = np.power(np.abs(x_val) + eps, np.abs(n))
        Kn = np.power(K + eps, np.abs(n))
        return A + alpha * Vmax * xn / (Kn + xn)

    y_at_xmin = hill_eval(x_min_obs)
    y_at_xmax = hill_eval(x_max_obs)

    FC_obs = np.where(n_sign > 0,
                     y_at_xmax / (y_at_xmin + eps),
                     y_at_xmin / (y_at_xmax + eps))
    log2fc_obs = np.log2(FC_obs + eps)

    return log2fc_obs, x_min_obs, x_max_obs


def prepare_de_for_cg(model, de_df, cis_gene=None, color_scheme=None):
    """
    Prepare model + edgeR data for a given cis gene.

    Parameters
    ----------
    model : bayesDREAM
        Fitted model
    de_df : pd.DataFrame
        External DE results (e.g., from edgeR) with columns:
        'gene', 'logFC', 'FDR', 'guide' (optional)
    cis_gene : str, optional
        Cis gene name (defaults to model.cis_gene)
    color_scheme : ColorScheme, optional
        Color scheme

    Returns
    -------
    tuple or None
        (A_samps, alpha_samps, Vmax_samps, K_samps, n_samps, x_true_samps,
         meta, de_cg (with idx, logFC, FDR, gene, n_mean, dependent, ext_sig),
         base_target_color, base_ntc_color)
        Returns None if posterior_samples_trans not found in model.
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', None)

    # Check if model has been fitted with trans
    if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
        print(f"[{cis_gene}] posterior_samples_trans not found in model. Run fit_trans() first.")
        return None

    # Check if cis_gene matches
    if cis_gene is not None and hasattr(model, 'cis_gene') and model.cis_gene != cis_gene:
        print(f"[{cis_gene}] does not match model.cis_gene='{model.cis_gene}'.")
        return None

    A_samps     = to_np(model.posterior_samples_trans['A'][:, 0, :])
    alpha_samps = to_np(model.posterior_samples_trans['alpha'][:, 0, :])
    Vmax_samps  = to_np(model.posterior_samples_trans['Vmax_a'][:, 0, :])
    K_samps     = to_np(model.posterior_samples_trans['K_a'][:, 0, :])
    n_samps     = to_np(model.posterior_samples_trans['n_a'][:, 0, :])
    x_true_samps = to_np(model.x_true)
    meta = model.meta

    n_mean   = n_samps.mean(axis=0)             # [T]
    dep_mask = dependency_mask_from_n(n_samps)  # [T] bool
    T        = n_mean.shape[0]

    # gene names aligned to posterior arrays — modality.feature_names is the
    # single source of truth (resolved + deduped in Modality.__init__).
    gene_names = np.array(model.get_modality('gene').feature_names)
    if len(gene_names) != T:
        print(f"[{cis_gene}] WARNING: len(gene_names)={len(gene_names)} != T={T}. "
              "Trimming gene_names to first T entries.")
        gene_names = gene_names[:T]

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # external DE results (edgeR)
    de_cg = de_df[de_df['gene'].isin(gene_names)].copy()
    de_cg['idx'] = de_cg['gene'].map(gene_to_idx)
    de_cg = de_cg.dropna(subset=['idx'])
    de_cg['idx'] = de_cg['idx'].astype(int)

    de_cg['n_mean']     = n_mean[de_cg['idx'].values]
    de_cg['dependent']  = dep_mask[de_cg['idx'].values]
    de_cg['ext_sig']    = de_cg['FDR'] < 0.05

    # colors
    target = meta['target'].astype(str).iloc[0] if 'target' in meta.columns else cis_gene
    base_target_color = color_scheme.get_target_color(target, 'steelblue')
    base_ntc_color = color_scheme.get_target_color('NTC', 'gray')

    return (A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
            x_true_samps, meta, de_cg, base_target_color, base_ntc_color)


def scatter_and_heatmap_edger_vs_bayes(
    df_g,
    y_col,
    base_target_color,
    base_ntc_color,
    cis_gene,
    guide,
    ylabel,
    title_suffix,
    fc_thresh=0.5,
    flip_edger_x=True,
):
    """
    For a single guide df_g (already subset to that guide), make:
      - scatter plot: edgeR logFC vs bayesDREAM y_col
      - 3x3 heatmap of category overlap

    Parameters
    ----------
    df_g : pd.DataFrame
        Data for one guide, must contain: 'logFC', 'ext_sig', 'dependent', y_col
    y_col : str
        Column name for bayesDREAM log2FC
    base_target_color : color
        Base color for target
    base_ntc_color : color
        Base color for NTC
    cis_gene : str
        Cis gene name
    guide : str
        Guide name
    ylabel : str
        Y-axis label
    title_suffix : str
        Title suffix
    fc_thresh : float
        Fold-change threshold for classification
    flip_edger_x : bool
        If True, flip sign of edgeR logFC

    Returns
    -------
    None (displays plots)
    """
    g_str = str(guide)

    # 4 color classes for scatter (computed before finite mask)
    colors = []
    for dep, sig in zip(df_g['dependent'], df_g['ext_sig']):
        if (not dep) and (not sig):
            # neither method calls it
            c = base_ntc_color
        elif sig and (not dep):
            # edgeR only (FDR<0.05, not dependent in bayesDREAM)
            c = lighten(base_target_color, 0.4)
        elif dep and (not sig):
            # bayesDREAM only (dependent, FDR>=0.05)
            c = base_target_color
        else:
            # both: dependent & FDR<0.05
            c = darken(base_target_color, 0.4)
        colors.append(c)

    # restrict to finite values
    finite = np.isfinite(df_g['logFC']) & np.isfinite(df_g[y_col])
    df_plot = df_g[finite]
    if df_plot.empty:
        return
    colors = np.array(colors)[finite.values]

    # x vs y values
    x_raw = df_plot['logFC'].values
    x_vals = -x_raw if flip_edger_x else x_raw
    y_vals = df_plot[y_col].values

    # same scale on x & y
    v_min = min(x_vals.min(), y_vals.min())
    v_max = max(x_vals.max(), y_vals.max())
    pad = 0.05 * (v_max - v_min + 1e-6)
    x_lim = (v_min - pad, v_max + pad)
    y_lim = (v_min - pad, v_max + pad)

    # ---------- SCATTER ----------
    plt.figure(figsize=(5.5, 5))
    plt.scatter(
        x_vals,
        y_vals,
        s=10,
        c=colors,
        alpha=0.8,
        edgecolor='none',
    )

    # reference lines
    plt.axhline(0,    color='black', linestyle=':', linewidth=1)
    plt.axvline(0,    color='black', linestyle=':', linewidth=1)
    plt.axhline( fc_thresh,  color='black', linestyle='--', linewidth=0.8)
    plt.axhline(-fc_thresh,  color='black', linestyle='--', linewidth=0.8)
    plt.axvline( fc_thresh,  color='black', linestyle='--', linewidth=0.8)
    plt.axvline(-fc_thresh,  color='black', linestyle='--', linewidth=0.8)

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    x_label_prefix = '-' if flip_edger_x else ''
    plt.xlabel(fr'{x_label_prefix}log$_2$FC (edgeR; per-guide)')
    plt.ylabel(ylabel)
    plt.title(f'{cis_gene} {g_str} {title_suffix}')

    # legend
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='', color=base_ntc_color,
              markersize=6, label='neither'),
        Line2D([0], [0], marker='o', linestyle='', color=lighten(base_target_color, 0.4),
              markersize=6, label='edgeR only'),
        Line2D([0], [0], marker='o', linestyle='', color=base_target_color,
              markersize=6, label='bayesDREAM only'),
        Line2D([0], [0], marker='o', linestyle='', color=darken(base_target_color, 0.4),
              markersize=6, label='both'),
    ]
    plt.legend(handles=legend_handles, fontsize=8, loc='upper left', frameon=True)
    plt.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 3x3 HEATMAP ----------
    # Classify genes by direction for both methods
    def classify_direction(vals, thresh):
        """Return 'up', 'down', or 'flat'"""
        result = np.full(len(vals), 'flat', dtype=object)
        result[vals > thresh] = 'up'
        result[vals < -thresh] = 'down'
        return result

    edger_dir = classify_direction(x_vals, fc_thresh)
    bayes_dir = classify_direction(y_vals, fc_thresh)

    # Build confusion matrix
    categories = ['down', 'flat', 'up']
    confusion = np.zeros((3, 3), dtype=int)
    for i, cat_edger in enumerate(categories):
        for j, cat_bayes in enumerate(categories):
            confusion[i, j] = ((edger_dir == cat_edger) & (bayes_dir == cat_bayes)).sum()

    plt.figure(figsize=(5, 4.5))
    plt.imshow(confusion, cmap='Blues', aspect='auto')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(confusion[i, j]), ha='center', va='center',
                    fontsize=12, color='black' if confusion[i, j] < confusion.max()/2 else 'white')

    plt.xticks(range(3), categories)
    plt.yticks(range(3), categories)
    plt.xlabel('bayesDREAM direction')
    plt.ylabel('edgeR direction')
    plt.title(f'{cis_gene} {g_str}: direction overlap')
    plt.colorbar(label='count')
    plt.tight_layout()
    plt.show()


def plot_edger_vs_bayes_full_range(cis_genes, model, de_df, fc_thresh=0.5,
                                   flip_edger_x=True, color_scheme=None):
    """
    Full-range comparison: edgeR vs bayesDREAM log2FC_full (directional).

    Parameters
    ----------
    cis_genes : list of str
        Cis gene names
    model : bayesDREAM
        Fitted model
    de_df : pd.DataFrame
        External DE results
    fc_thresh : float
        Fold-change threshold
    flip_edger_x : bool
        If True, flip sign of edgeR logFC
    color_scheme : ColorScheme, optional
        Color scheme

    Returns
    -------
    None (displays plots)
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    for cg in cis_genes:
        print(f"\n=== {cg} (full-range) ===")

        prep = prepare_de_for_cg(model, de_df, cg, color_scheme)
        if prep is None:
            continue
        (A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
         x_true_samps, meta, de_cg, base_target_color, base_ntc_color) = prep

        # full-range log2FC from bayesDREAM
        log2fc_full, _, _, _ = compute_log2fc_metrics(
            A_samps, alpha_samps, Vmax_samps, K_samps, n_samps, x_true_samps
        )
        log2fc_full_mean = log2fc_full.mean(axis=0)  # [T]
        de_cg['log2fc_full'] = log2fc_full_mean[de_cg['idx'].values]

        guides_in_model = set(meta['guide'].astype(str).unique())

        for guide in sorted(de_cg['guide'].unique()):
            g_str = str(guide)
            if g_str not in guides_in_model:
                continue

            df_g = de_cg[de_cg['guide'] == guide].copy()
            if df_g.empty:
                continue

            scatter_and_heatmap_edger_vs_bayes(
                df_g=df_g,
                y_col='log2fc_full',
                base_target_color=base_target_color,
                base_ntc_color=base_ntc_color,
                cis_gene=cg,
                guide=g_str,
                ylabel=r'log$_2$FC$_{\mathrm{full}}$ '
                       r'(bayesDREAM, $x:0\to\infty$)',
                title_suffix='(full-range log$_2$FC)',
                fc_thresh=fc_thresh,
                flip_edger_x=flip_edger_x,
            )


def plot_edger_vs_bayes_observed_range(cis_genes, model, de_df, fc_thresh=0.5,
                                      flip_edger_x=True, aggregate_by_guide=True,
                                      color_scheme=None):
    """
    Observed-range comparison: edgeR vs bayesDREAM log2FC_obs (guide+NTC).

    Parameters
    ----------
    cis_genes : list of str
        Cis gene names
    model : bayesDREAM
        Fitted model
    de_df : pd.DataFrame
        External DE results
    fc_thresh : float
        Fold-change threshold
    flip_edger_x : bool
        If True, flip sign of edgeR logFC
    aggregate_by_guide : bool
        If True, aggregate x_true by guide before computing min/max
    color_scheme : ColorScheme, optional
        Color scheme

    Returns
    -------
    None (displays plots)
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    for cg in cis_genes:
        print(f"\n=== {cg} (observed-range) ===")

        prep = prepare_de_for_cg(model, de_df, cg, color_scheme)
        if prep is None:
            continue
        (A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
         x_true_samps, meta, de_cg, base_target_color, base_ntc_color) = prep

        guides_in_model = set(meta['guide'].astype(str).unique())
        guide_labels_all = meta['guide'].astype(str).values

        for guide in sorted(de_cg['guide'].unique()):
            g_str = str(guide)
            if g_str not in guides_in_model:
                continue

            df_g = de_cg[de_cg['guide'] == guide].copy()
            if df_g.empty:
                continue

            # cells: this guide + NTC-target cells
            guide_mask = guide_labels_all == g_str
            ntc_mask   = meta['target'].astype(str).str.upper().str.contains('NTC').to_numpy()
            cell_mask  = guide_mask | ntc_mask

            log2fc_obs_guide, x_min_obs, x_max_obs = compute_log2fc_obs_for_cells(
                A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
                x_true_samps,
                cell_mask=cell_mask,
                guide_labels=guide_labels_all if aggregate_by_guide else None,
            )
            log2fc_obs_mean_guide = log2fc_obs_guide.mean(axis=0)
            de_cg.loc[df_g.index, 'log2fc_obs_guide'] = \
                log2fc_obs_mean_guide[df_g['idx'].values]

            df_g = de_cg.loc[df_g.index].copy()

            scatter_and_heatmap_edger_vs_bayes(
                df_g=df_g,
                y_col='log2fc_obs_guide',
                base_target_color=base_target_color,
                base_ntc_color=base_ntc_color,
                cis_gene=cg,
                guide=g_str,
                ylabel=r'log$_2$FC$_{\mathrm{obs}}$ '
                       r'(bayesDREAM; guide+NTC x-range)',
                title_suffix='(observed log$_2$FC)',
                fc_thresh=fc_thresh,
                flip_edger_x=flip_edger_x,
            )
