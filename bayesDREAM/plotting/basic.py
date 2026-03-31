"""
Basic plotting functions for x_true distributions.

Provides scatter, violin, and density plots colored by guide.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .helpers import to_np
from .colors import ColorScheme


def scatter_by_guide(model, cis_gene=None, log2=False, color_scheme=None, ax=None, show=True):
    """
    Scatter plot of per-cell mean vs std of x_true, colored by guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str, optional
        Cis gene name (used for title, defaults to model.cis_gene)
    log2 : bool, default False
        Whether to use log2 scale
    color_scheme : ColorScheme, optional
        Custom color scheme. If None, uses default.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    df = model.meta.copy()
    X = to_np(model.x_true)

    if log2:
        # filter strictly positive before log
        mask_pos = (X > 0).all(axis=0)
        X = X[:, mask_pos]
        df = df.loc[mask_pos].reset_index(drop=True)
        X = np.log2(X)

    x_mean, x_std = X.mean(axis=0), X.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for guide, subidx in df.groupby('guide').groups.items():
        color = color_scheme.get_guide_color(guide, 'black')
        ax.scatter(x_mean[subidx], x_std[subidx], s=14, alpha=0.8,
                  color=color, label=guide)

    ax.set_xlabel('mean x_true' + (' (log2)' if log2 else ''))
    ax.set_ylabel('std x_true' + (' (log2)' if log2 else ''))
    ax.set_title(f'{cis_gene}: mean vs std of x_true' + (' (log2)' if log2 else ''))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def scatter_ci95_by_guide(model, cis_gene=None, log2=False, full_width=False,
                          color_scheme=None, ax=None, show=True):
    """
    Scatter of per-cell mean vs 95% CI width (or half-width) of x_true samples.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str, optional
        Cis gene name (used for title, defaults to model.cis_gene)
    log2 : bool, default False
        Whether to use log2 scale
    full_width : bool, default False
        If True, plot full width. If False, plot half-width.
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    df = model.meta.copy()
    X = to_np(model.x_true)  # shape [S, N] (samples x cells)

    if log2:
        mask_pos = (X > 0).all(axis=0)
        X = X[:, mask_pos]
        df = df.loc[mask_pos].reset_index(drop=True)
        X = np.log2(X)

    x_mean = X.mean(axis=0)
    q_lo  = np.percentile(X, 2.5, axis=0)
    q_hi  = np.percentile(X, 97.5, axis=0)
    y_val = (q_hi - q_lo) if full_width else 0.5 * (q_hi - q_lo)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for guide, idx in df.groupby('guide').groups.items():
        color = color_scheme.get_guide_color(guide, 'black')
        ax.scatter(x_mean[idx], y_val[idx], s=14, alpha=0.85,
                  color=color, label=guide)

    ax.set_xlabel('mean x_true' + (' (log2)' if log2 else ''))
    ylabel = '95% CI ' + ('width' if full_width else 'half-width')
    ylabel += ' of x_true' + (' (log2)' if log2 else '')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{cis_gene}: mean vs 95% CI of x_true' + (' (log2)' if log2 else ''))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def violin_by_guide_log2(model, cis_gene=None, color_scheme=None, ax=None, show=True):
    """
    Violin plot of x_true (log2) grouped by guide, colored by target.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str, optional
        Cis gene name (used for title, defaults to model.cis_gene)
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    df = model.meta.copy()
    X = to_np(model.x_true)

    pos_mask = (X > 0).all(axis=0)
    X = X[:, pos_mask]
    df = df.loc[pos_mask].reset_index(drop=True)

    Xlog = np.log2(X)
    x_cell_mean = Xlog.mean(axis=0)
    df = df.assign(x_true_mean_log2=x_cell_mean)

    guide_order = sorted(df['guide'].astype(str).unique(),
                        key=lambda g: (g.split('_')[0], int(g.split('_')[1])
                                     if '_' in g and g.split('_')[1].isdigit() else 0))
    data = [df.loc[df['guide'] == g, 'x_true_mean_log2'].values for g in guide_order]

    colors = []
    for g in guide_order:
        target = g.split('_')[0] if '_' in g else g
        colors.append(color_scheme.get_target_color(target, 'gray'))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(data, positions=np.arange(1, len(guide_order)+1),
                         showmeans=True, showextrema=True, widths=0.7)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(0.85)

    for pc_key in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
        if pc_key in parts:
            parts[pc_key].set_edgecolor('black')
            parts[pc_key].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(guide_order)+1))
    ax.set_xticklabels(guide_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('x_true mean (log₂)', fontsize=11)
    ax.set_title(f'{cis_gene}: x_true distribution by guide', fontsize=12)
    ax.grid(axis='y', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def filled_density_by_guide_log2(model, cis_gene=None, bw=None, color_scheme=None,
                                 ax=None, show=True):
    """
    Filled KDE density plot of x_true (log2), colored by guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str, optional
        Cis gene name (used for title, defaults to model.cis_gene)
    bw : float, optional
        KDE bandwidth. If None, uses scott's rule.
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    if cis_gene is None:
        cis_gene = getattr(model, 'cis_gene', 'cis')

    df = model.meta.copy()
    X = to_np(model.x_true)

    pos_mask = (X > 0).all(axis=0)
    X = X[:, pos_mask]
    df = df.loc[pos_mask].reset_index(drop=True)

    Xlog = np.log2(X)
    x_cell_mean = Xlog.mean(axis=0)
    df = df.assign(x_true_mean_log2=x_cell_mean)

    xmin, xmax = np.percentile(x_cell_mean, [0.5, 99.5])
    x_grid = np.linspace(xmin, xmax, 500)

    guide_order = sorted(df['guide'].astype(str).unique(),
                        key=lambda g: (g.split('_')[0], int(g.split('_')[1])
                                     if '_' in g and g.split('_')[1].isdigit() else 0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for g in guide_order:
        sub = df.loc[df['guide'] == g, 'x_true_mean_log2'].values
        if len(sub) < 2:
            continue

        kde = gaussian_kde(sub, bw_method=bw)
        density = kde(x_grid)

        color = color_scheme.get_guide_color(g, 'black')
        ax.fill_between(x_grid, density, alpha=0.4, color=color, label=g)
        ax.plot(x_grid, density, color=color, linewidth=1.5)

    ax.set_xlabel('x_true mean (log₂)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{cis_gene}: x_true density by guide', fontsize=12)
    ax.legend(title='guide', fontsize=8, frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def scatter_param_mean_vs_ci(
    param_samps,
    param_name='parameter',
    subset_mask=None,
    color_by=None,
    color_label='color metric',
    cmap='Blues_r',
    vmin=None,
    vmax=None,
    log2=False,
    ax=None,
    show=True,
    title=None,
    figsize=(7, 5),
):
    """
    Scatter plot of parameter mean vs 95% CI width, with optional color coding.

    This is useful for visualizing parameter uncertainty vs magnitude, with
    optional coloring by dependency masks, NaN fractions, or other metrics.

    Parameters
    ----------
    param_samps : np.ndarray
        Parameter samples, shape (n_samples, n_features)
    param_name : str
        Parameter name for axis labels (default: 'parameter')
    subset_mask : np.ndarray, optional
        Boolean mask to subset features. If provided, plots two groups:
        masked (colored) and unmasked (grey).
    color_by : np.ndarray, optional
        Values to color points by (length n_features). Requires subset_mask.
        Common uses:
        - NaN fraction: color by how many samples are NaN
        - Dependency metric: color by strength of effect
    color_label : str
        Label for colorbar (default: 'color metric')
    cmap : str
        Colormap name (default: 'Blues_r' for darker = fewer NaNs)
    vmin, vmax : float, optional
        Color scale limits. If None, uses data range.
    log2 : bool
        Whether param_samps are on log2 scale (affects axis label only)
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool
        Whether to display the plot (default: True)
    title : str, optional
        Plot title. If None, auto-generates from param_name.
    figsize : tuple
        Figure size (default: (7, 5))

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    >>> # Example 1: Inflection point with NaN fraction coloring
    >>> xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> frac_nan = np.mean(np.isnan(xinf_samps), axis=0)
    >>> fig = scatter_param_mean_vs_ci(
    ...     xinf_samps,
    ...     param_name='x_infl',
    ...     subset_mask=dep_mask,
    ...     color_by=frac_nan,
    ...     color_label='fraction NaN (lighter = more NaN)',
    ...     cmap='Blues_r',
    ...     log2=True
    ... )

    >>> # Example 2: Hill coefficient n with dependency coloring
    >>> n_samps = model.posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> fig = scatter_param_mean_vs_ci(
    ...     n_samps,
    ...     param_name='n (Hill coefficient)',
    ...     subset_mask=dep_mask,
    ...     log2=False
    ... )
    """
    param_samps = np.asarray(param_samps)

    # Compute mean and CI width
    param_mean = np.nanmean(param_samps, axis=0)
    param_lo = np.nanpercentile(param_samps, 2.5, axis=0)
    param_hi = np.nanpercentile(param_samps, 97.5, axis=0)
    param_ci_width = param_hi - param_lo

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Case 1: No subsetting - plot all points in one color
    if subset_mask is None:
        ax.scatter(param_mean, param_ci_width, s=8, alpha=0.6, color='blue')

    # Case 2: Subsetting without color coding
    elif color_by is None:
        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey', label='not selected'
            )

        # Plot masked points in blue
        ax.scatter(
            param_mean[subset_mask],
            param_ci_width[subset_mask],
            s=5, alpha=0.2, color='blue', label='selected'
        )
        ax.legend(frameon=False, loc='best')

    # Case 3: Subsetting with color coding
    else:
        color_by = np.asarray(color_by)
        if len(color_by) != len(subset_mask):
            raise ValueError(f"color_by length ({len(color_by)}) must match subset_mask length ({len(subset_mask)})")

        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey'
            )

        # Plot masked points with color coding
        valid_mask = subset_mask & np.isfinite(param_mean) & np.isfinite(param_ci_width)

        if vmin is None:
            vmin = np.nanmin(color_by[valid_mask])
        if vmax is None:
            vmax = np.nanmax(color_by[valid_mask])

        sc = ax.scatter(
            param_mean[valid_mask],
            param_ci_width[valid_mask],
            c=color_by[valid_mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=8,
            alpha=0.9
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_label)

    # Labels and formatting
    xlabel = f'Mean {param_name}' + (' (log₂)' if log2 else '')
    ylabel = f'95% CI width of {param_name}' + (' (log₂)' if log2 else '')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f'{param_name}: mean vs uncertainty'
    ax.set_title(title)

    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def plot_parameter_ci_panel(
    model,
    params: list,
    modality_name: str = None,
    genes: list = None,
    ci_level: float = 95.0,
    sort_by: str = 'none',
    filter_dependent: bool = False,
    dependency_params: list = None,
    max_genes: int = 100,
    ymin: float = None,
    ymax: float = None,
    title: str = None,
    ylabel: str = 'value',
    figsize: tuple = None,
    color_palette: dict = None,
    marker_size: int = 18,
    capsize: int = 3,
    show_zero_line: bool = True,
    show_gene_separators: bool = True,
    ax=None,
    show: bool = True,
    fdr_df=None,
    fdr_threshold: float = 0.05,
):
    """
    Forest plot (dot + whisker CI) for posterior parameters across trans genes.

    Creates a plot with genes on the x-axis and parameter values (median + CI) on
    the y-axis. Multiple parameters are dodged side-by-side for comparison.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model with posterior_samples_trans
    params : list of str
        Parameter names to plot (e.g., ['n_a', 'n_b'] or ['alpha', 'beta']).
        These must exist in posterior_samples_trans.
    modality_name : str, optional
        Modality name. If None, uses primary modality.
    genes : list of str, optional
        Specific genes to plot. If None, plots all genes (subject to max_genes).
        Gene names must match feature names in the modality.
    ci_level : float
        Credible interval level (default: 95.0 for 95% CI)
    sort_by : str
        How to sort genes on x-axis:
        - 'none': Keep original order
        - 'alphabetical': Sort alphabetically by gene name
        - 'median': Sort by median of first parameter (ascending)
        - 'abs_median': Sort by absolute median of first parameter (descending)
        - 'effect': Sort by max absolute effect across all params (descending)
    filter_dependent : bool
        If True, only show genes where CI excludes 0 for any param in
        dependency_params (default: False)
    dependency_params : list, optional
        Parameters to use for dependency filtering. If None, uses all params.
        Common: ['n_a', 'n_b'] for Hill coefficients.
    max_genes : int
        Maximum number of genes to plot (default: 100). If more genes would be
        plotted, raises ValueError with suggestions. Set to None to disable limit.
    ymin, ymax : float, optional
        Y-axis limits. If None, auto-scaled.
    title : str, optional
        Plot title. If None, auto-generated.
    ylabel : str
        Y-axis label (default: 'value')
    figsize : tuple, optional
        Figure size. If None, auto-scaled based on number of genes.
    color_palette : dict, optional
        Custom colors for parameters. Keys are param names, values are colors.
        If None, uses seaborn color palette.
    marker_size : int
        Size of median markers (default: 18)
    capsize : int
        Size of error bar caps (default: 3)
    show_zero_line : bool
        Whether to draw horizontal line at y=0 (default: True)
    show_gene_separators : bool
        Whether to draw vertical lines between genes (default: True).
        Helps visually distinguish which parameters belong to which gene.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool
        Whether to display the plot (default: True)
    fdr_df : pd.DataFrame, optional
        trans_summary DataFrame (output of save_trans_summary). When provided,
        parameters belonging to FDR-inactive components (fdr_alpha or fdr_beta
        >= fdr_threshold) are rendered in light grey instead of their normal
        color. Component mapping: alpha/n_a/K_a/Vmax_a → fdr_alpha;
        beta/n_b/K_b/Vmax_b → fdr_beta.
    fdr_threshold : float
        FDR threshold for inactivity (default: 0.05). Used with fdr_df.

    Returns
    -------
    fig : matplotlib Figure (if ax was None)
    ax : matplotlib Axes

    Examples
    --------
    >>> # Plot n_a and n_b for all genes
    >>> fig, ax = model.plot_parameter_ci_panel(['n_a', 'n_b'])

    >>> # Plot only dependent genes, sorted by effect size
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['n_a', 'n_b'],
    ...     filter_dependent=True,
    ...     sort_by='effect'
    ... )

    >>> # Plot alpha and beta with custom colors
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['alpha', 'beta'],
    ...     color_palette={'alpha': 'crimson', 'beta': 'dodgerblue'}
    ... )

    >>> # Plot for a specific modality
    >>> fig, ax = model.plot_parameter_ci_panel(
    ...     ['n_a', 'n_b'],
    ...     modality_name='splicing_sj'
    ... )
    """
    import seaborn as sns

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Get posterior samples
    if modality_name == model.primary_modality:
        posterior = model.posterior_samples_trans
    else:
        posterior = modality.posterior_samples_trans

    if posterior is None:
        raise ValueError(
            f"No posterior_samples_trans found for modality '{modality_name}'. "
            "Must run fit_trans() first."
        )

    # Validate params exist
    missing = [p for p in params if p not in posterior]
    if missing:
        available = list(posterior.keys())
        raise ValueError(
            f"Parameters {missing} not found in posterior. "
            f"Available: {available}"
        )

    # Get gene names from modality
    gene_names = modality.feature_names
    if gene_names is None:
        # Fallback to feature_meta
        for col in ['gene_name', 'gene', 'feature_id', 'feature']:
            if col in modality.feature_meta.columns:
                gene_names = modality.feature_meta[col].tolist()
                break
    if gene_names is None:
        gene_names = [str(i) for i in range(modality.dims['n_features'])]

    # Extract samples for each parameter
    # Shape: posterior[param] is typically (S, n_cis, T) where n_cis=1
    samples_dict = {}
    for param in params:
        samps = to_np(posterior[param])
        # Handle different shapes
        if samps.ndim == 3:
            samps = samps[:, 0, :]  # (S, 1, T) -> (S, T)
        elif samps.ndim == 1:
            samps = samps.reshape(-1, 1)  # (S,) -> (S, 1)
        samples_dict[param] = samps

    T = samples_dict[params[0]].shape[1]

    # Ensure gene_names matches T
    if len(gene_names) != T:
        gene_names = gene_names[:T]

    # Build per-param FDR inactive masks from fdr_df (if provided)
    _PARAM_TO_FDR_COL = {
        'alpha': 'fdr_alpha', 'n_a': 'fdr_alpha', 'K_a': 'fdr_alpha', 'Vmax_a': 'fdr_alpha',
        'beta':  'fdr_beta',  'n_b': 'fdr_beta',  'K_b': 'fdr_beta',  'Vmax_b': 'fdr_beta',
    }
    inactive_masks = {}  # {param: bool array of length T}
    if fdr_df is not None:
        _gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        _name_col = next((c for c in ['gene_name', 'gene', 'feature'] if c in fdr_df.columns), None)
        for param in params:
            fdr_col = _PARAM_TO_FDR_COL.get(param)
            if fdr_col and fdr_col in fdr_df.columns and _name_col:
                mask = np.zeros(T, dtype=bool)
                for _, row in fdr_df.iterrows():
                    gname = row[_name_col]
                    if gname in _gene_to_idx and np.isfinite(row[fdr_col]):
                        mask[_gene_to_idx[gname]] = row[fdr_col] >= fdr_threshold
                inactive_masks[param] = mask

    # Compute CI bounds
    lo_q = (100 - ci_level) / 2.0
    hi_q = 100 - lo_q

    stats = {}  # {param: {'median': array, 'lo': array, 'hi': array}}
    for param, samps in samples_dict.items():
        stats[param] = {
            'median': np.nanmedian(samps, axis=0),
            'lo': np.nanpercentile(samps, lo_q, axis=0),
            'hi': np.nanpercentile(samps, hi_q, axis=0),
        }

    # Filter to dependent genes if requested
    gene_mask = np.ones(T, dtype=bool)
    if filter_dependent:
        dep_params = dependency_params if dependency_params else params
        # Start with all False, then OR with each param's dependency
        gene_mask = np.zeros(T, dtype=bool)
        for param in dep_params:
            if param in samples_dict:
                samps = samples_dict[param]
                lo = np.nanpercentile(samps, lo_q, axis=0)
                hi = np.nanpercentile(samps, hi_q, axis=0)
                param_dep = (lo > 0) | (hi < 0)
                gene_mask = gene_mask | param_dep

        n_dep = gene_mask.sum()
        print(f"[FILTER] {n_dep}/{T} genes pass dependency filter (CI excludes 0)")

    # Get indices of genes to plot
    gene_indices = np.where(gene_mask)[0]

    # Filter to user-specified genes if provided
    if genes is not None:
        # Map gene names to indices
        name_to_idx = {name: i for i, name in enumerate(gene_names)}
        user_indices = []
        missing_genes = []
        for g in genes:
            if g in name_to_idx:
                idx = name_to_idx[g]
                if idx in gene_indices:  # Respect dependency filter
                    user_indices.append(idx)
            else:
                missing_genes.append(g)
        if missing_genes:
            import warnings
            warnings.warn(f"Genes not found in modality: {missing_genes[:5]}{'...' if len(missing_genes) > 5 else ''}")
        gene_indices = np.array(user_indices)

    n_genes = len(gene_indices)

    if n_genes == 0:
        print("No genes to plot after filtering.")
        return None, None

    # Check max_genes limit
    if max_genes is not None and n_genes > max_genes:
        raise ValueError(
            f"Too many genes to plot ({n_genes} > {max_genes}). Options:\n"
            f"  1. Use filter_dependent=True to show only dependent genes\n"
            f"  2. Use genes=['gene1', 'gene2', ...] to specify specific genes\n"
            f"  3. Use sort_by='effect' with filter_dependent=True for top effects\n"
            f"  4. Set max_genes=None to disable this limit (not recommended)"
        )
    elif n_genes > 100:
        import warnings
        warnings.warn(
            f"Plotting {n_genes} genes. Consider using filter_dependent=True "
            f"or genes=[...] to reduce the number of genes."
        )

    # Sort genes
    if sort_by == 'alphabetical':
        # Get gene names for current indices, then sort alphabetically
        names_for_sort = [gene_names[i] for i in gene_indices]
        order = np.argsort(names_for_sort)
        gene_indices = gene_indices[order]
    elif sort_by == 'median':
        sort_vals = stats[params[0]]['median'][gene_indices]
        order = np.argsort(sort_vals)
        gene_indices = gene_indices[order]
    elif sort_by == 'abs_median':
        sort_vals = np.abs(stats[params[0]]['median'][gene_indices])
        order = np.argsort(sort_vals)[::-1]  # Descending
        gene_indices = gene_indices[order]
    elif sort_by == 'effect':
        # Max absolute median across all params
        max_effect = np.zeros(n_genes)
        for param in params:
            max_effect = np.maximum(max_effect, np.abs(stats[param]['median'][gene_indices]))
        order = np.argsort(max_effect)[::-1]  # Descending
        gene_indices = gene_indices[order]

    # Get sorted gene names
    sorted_gene_names = [gene_names[i] for i in gene_indices]

    # Create figure
    if ax is None:
        if figsize is None:
            fig_w = min(max(0.5 * n_genes, 12), 28)
            fig_h = 5.5
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set up colors
    if color_palette is None:
        colors = sns.color_palette(n_colors=len(params))
        color_palette = dict(zip(params, colors))

    # Set up x positions with dodging
    # Use smaller width to keep params for same gene close together
    x_base = np.arange(n_genes)
    n_params = len(params)
    width = 0.3  # Reduced from 0.7 to keep params closer within gene
    if n_params > 1:
        offsets = np.linspace(-width/2, width/2, n_params)
    else:
        offsets = np.array([0.0])

    # Plot each parameter
    for j, param in enumerate(params):
        medians = stats[param]['median'][gene_indices]
        los = stats[param]['lo'][gene_indices]
        his = stats[param]['hi'][gene_indices]

        x = x_base + offsets[j]
        color = color_palette.get(param, 'blue')

        # Determine FDR inactive subset (greyed out)
        raw_inactive = inactive_masks.get(param)
        inactive_plot = raw_inactive[gene_indices] if raw_inactive is not None else np.zeros(len(gene_indices), dtype=bool)
        active_plot = ~inactive_plot

        # Plot active genes with full color
        if active_plot.any():
            ax.scatter(x[active_plot], medians[active_plot], label=param,
                       s=marker_size, zorder=3, color=color)
            yerr_act = np.vstack([medians[active_plot] - los[active_plot],
                                   his[active_plot] - medians[active_plot]])
            ax.errorbar(x[active_plot], medians[active_plot], yerr=yerr_act,
                        fmt='none', elinewidth=1.5, capsize=capsize, color=color, zorder=2)
        else:
            # No active genes — add phantom entry for legend
            ax.scatter([], [], label=param, s=marker_size, color=color)

        # Plot inactive genes greyed out
        if inactive_plot.any():
            ax.scatter(x[inactive_plot], medians[inactive_plot],
                       s=marker_size, zorder=3, color='lightgray', alpha=0.5)
            yerr_inact = np.vstack([medians[inactive_plot] - los[inactive_plot],
                                     his[inactive_plot] - medians[inactive_plot]])
            ax.errorbar(x[inactive_plot], medians[inactive_plot], yerr=yerr_inact,
                        fmt='none', elinewidth=1.5, capsize=capsize,
                        color='lightgray', alpha=0.5, zorder=2)

    # Styling
    if title is None:
        param_str = ', '.join(params)
        title = f"{model.cis_gene} → trans genes: {param_str}"
        if filter_dependent:
            title += f" (n={n_genes} dependent)"

    ax.set_title(title)
    ax.set_xlabel("Trans gene")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(sorted_gene_names, rotation=90, ha="center")

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(False)

    if show_zero_line:
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Draw vertical separators between genes
    if show_gene_separators and n_genes > 1:
        for i in range(1, n_genes):
            ax.axvline(i - 0.5, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)

    # Tighten x-axis margins
    ax.set_xlim(-0.5, n_genes - 0.5)

    if ymin is not None or ymax is not None:
        cur = ax.get_ylim()
        ax.set_ylim(
            ymin if ymin is not None else cur[0],
            ymax if ymax is not None else cur[1]
        )

    ax.legend(title='parameter', bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def extract_posterior_dataframe(
    model,
    params: list,
    modality_name: str = None,
    include_samples: bool = False,
):
    """
    Extract posterior parameters into a long-format DataFrame.

    This is useful for custom analysis or plotting with seaborn/plotnine.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    params : list of str
        Parameter names to extract (e.g., ['n_a', 'n_b', 'K_a', 'K_b'])
    modality_name : str, optional
        Modality name. If None, uses primary modality.
    include_samples : bool
        If True, includes all posterior samples (can be large).
        If False (default), only includes summary statistics.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - gene: Gene name
        - gene_idx: Gene index
        - param: Parameter name
        - median: Median value
        - lo: Lower CI bound (2.5%)
        - hi: Upper CI bound (97.5%)
        - mean: Mean value
        - std: Standard deviation
        - ci_excludes_zero: Boolean, True if CI excludes 0
        If include_samples=True, also includes:
        - sample_idx: Sample index
        - value: Sample value

    Examples
    --------
    >>> # Get summary statistics
    >>> df = extract_posterior_dataframe(model, ['n_a', 'n_b', 'K_a', 'K_b'])
    >>> df_dependent = df[df['ci_excludes_zero']]

    >>> # Get all samples for custom analysis
    >>> df_samples = extract_posterior_dataframe(model, ['n_a'], include_samples=True)
    >>> sns.violinplot(data=df_samples, x='gene', y='value')
    """
    import pandas as pd

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Get posterior samples
    if modality_name == model.primary_modality:
        posterior = model.posterior_samples_trans
    else:
        posterior = modality.posterior_samples_trans

    if posterior is None:
        raise ValueError(
            f"No posterior_samples_trans found for modality '{modality_name}'. "
            "Must run fit_trans() first."
        )

    # Get gene names
    gene_names = modality.feature_names
    if gene_names is None:
        for col in ['gene_name', 'gene', 'feature_id', 'feature']:
            if col in modality.feature_meta.columns:
                gene_names = modality.feature_meta[col].tolist()
                break
    if gene_names is None:
        gene_names = [str(i) for i in range(modality.dims['n_features'])]

    rows = []

    for param in params:
        if param not in posterior:
            print(f"[WARNING] Parameter '{param}' not found in posterior, skipping.")
            continue

        samps = to_np(posterior[param])

        # Handle different shapes
        if samps.ndim == 3:
            samps = samps[:, 0, :]  # (S, 1, T) -> (S, T)
        elif samps.ndim == 1:
            samps = samps.reshape(-1, 1)

        S, T = samps.shape

        # Ensure gene_names matches T
        gene_names_use = gene_names[:T] if len(gene_names) >= T else gene_names + [f'gene_{i}' for i in range(len(gene_names), T)]

        for i in range(T):
            gene_samps = samps[:, i]

            # Compute statistics
            median_val = np.nanmedian(gene_samps)
            lo_val = np.nanpercentile(gene_samps, 2.5)
            hi_val = np.nanpercentile(gene_samps, 97.5)
            mean_val = np.nanmean(gene_samps)
            std_val = np.nanstd(gene_samps)
            ci_excludes_zero = (lo_val > 0) or (hi_val < 0)

            if include_samples:
                # Add one row per sample
                for s_idx, val in enumerate(gene_samps):
                    rows.append({
                        'gene': gene_names_use[i],
                        'gene_idx': i,
                        'param': param,
                        'sample_idx': s_idx,
                        'value': float(val),
                        'median': median_val,
                        'lo': lo_val,
                        'hi': hi_val,
                        'mean': mean_val,
                        'std': std_val,
                        'ci_excludes_zero': ci_excludes_zero,
                    })
            else:
                # Add one summary row per gene
                rows.append({
                    'gene': gene_names_use[i],
                    'gene_idx': i,
                    'param': param,
                    'median': median_val,
                    'lo': lo_val,
                    'hi': hi_val,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_excludes_zero': ci_excludes_zero,
                })

    return pd.DataFrame(rows)
