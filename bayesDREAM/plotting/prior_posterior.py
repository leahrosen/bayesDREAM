"""
Prior/posterior goodness-of-fit plots for bayesDREAM parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
import warnings

from .utils import (
    compute_distribution_overlap,
    compute_kl_divergence,
    compute_posterior_coverage,
    compute_all_metrics,
    prepare_violin_data,
    compute_log2fc_vs_overlap,
    subset_features_by_mismatch
)


def plot_scalar_parameter(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    param_name: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 5),
    metric: str = 'posterior_coverage',
    **kwargs
) -> plt.Figure:
    """
    Plot prior vs posterior density for a scalar parameter (e.g., beta_o, alpha_x).

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples,)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples,)
    param_name : str
        Parameter name for plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    figsize : tuple
        Figure size if creating new figure
    metric : str
        Comparison metric: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
    **kwargs
        Additional arguments passed to sns.kdeplot

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> # Plot beta_o from technical fit
    >>> beta_o_prior = model.posterior_samples_ntc['beta_o']  # Prior samples
    >>> beta_o_post = model.posterior_samples_ntc['beta_o']   # Posterior samples
    >>> fig = plot_scalar_parameter(beta_o_prior, beta_o_post, 'beta_o')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Check for extreme prior values and clip symmetrically if needed
    prior_max = np.max(np.abs(prior_samples))
    post_max = np.max(np.abs(posterior_samples))
    if prior_max > 1e10 or prior_max > 1000 * post_max:
        # Clip prior to reasonable range for visualization (symmetric bounds)
        clip_bound = np.percentile(np.abs(posterior_samples), 99.9) * 10
        prior_samples = np.clip(prior_samples, -clip_bound, clip_bound)
        warnings.warn(f"Prior samples contain extreme values (max abs={prior_max:.2e}). "
                     f"Clipping to ±{clip_bound:.2e} for visualization. "
                     f"This does not affect the actual model fitting.")

    # Default KDE kwargs
    kde_kwargs = {'fill': True, 'alpha': 0.5, 'linewidth': 2}
    kde_kwargs.update(kwargs)

    # Plot densities with error handling for low-variance data
    try:
        sns.kdeplot(prior_samples, ax=ax, label='Prior', color='#1f77b4', **kde_kwargs)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fall back to histogram if KDE fails (e.g., singular covariance)
        warnings.warn(f"KDE failed for prior samples ({str(e)}), using histogram instead")
        ax.hist(prior_samples, bins=30, alpha=0.5, color='#1f77b4', label='Prior', density=True)

    try:
        sns.kdeplot(posterior_samples, ax=ax, label='Posterior', color='#ff7f0e', **kde_kwargs)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fall back to histogram if KDE fails
        warnings.warn(f"KDE failed for posterior samples ({str(e)}), using histogram instead")
        ax.hist(posterior_samples, bins=30, alpha=0.5, color='#ff7f0e', label='Posterior', density=True)

    # Compute chosen metric
    if metric == 'overlap':
        metric_value = compute_distribution_overlap(prior_samples, posterior_samples)
        metric_label = f'{metric_value:.1f}% overlap'
    elif metric == 'kl_divergence':
        metric_value = compute_kl_divergence(posterior_samples, prior_samples)
        metric_label = f'KL divergence: {metric_value:.3f} nats'
    elif metric == 'posterior_coverage':
        metric_value = compute_posterior_coverage(posterior_samples, prior_samples)
        metric_label = f'{metric_value:.1f}% posterior coverage'
    else:
        raise ValueError(f"Unknown metric: {metric}. Must be 'overlap', 'kl_divergence', or 'posterior_coverage'")

    # Add vertical lines for means
    ax.axvline(prior_samples.mean(), color='#1f77b4', linestyle='--', alpha=0.7,
               label=f'Prior mean: {prior_samples.mean():.3f}')
    ax.axvline(posterior_samples.mean(), color='#ff7f0e', linestyle='--', alpha=0.7,
               label=f'Posterior mean: {posterior_samples.mean():.3f}')

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{param_name}\n({metric_label})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_1d_parameter(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    param_name: str,
    order_by: str = 'mean',
    custom_order: Optional[List[str]] = None,
    subset_features: Optional[List[str]] = None,
    plot_type: str = 'auto',
    max_violin_features: int = 100,
    metric: str = 'posterior_coverage',
    is_additive: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot prior vs posterior for 1D parameter (e.g., alpha_y).

    Automatically chooses between violin plot (<100 features) or scatter plot (>=100 features).

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples, n_features)
    feature_names : List[str]
        Feature names (genes, etc.)
    param_name : str
        Parameter name for plot title
    order_by : str
        How to order features: 'mean', 'difference', 'alphabetical', 'custom', 'input'
    custom_order : List[str], optional
        Custom feature ordering (only if order_by='custom')
    subset_features : List[str], optional
        Subset to these specific features
    plot_type : str
        'auto', 'violin', or 'scatter'
    max_violin_features : int
        Maximum features for violin plot before switching to scatter
    metric : str
        Comparison metric for scatter plots: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
    is_additive : bool
        If True, use difference (posterior - prior) for scatter plots.
        If False, use log2 fold change. (default: False)
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    figsize : tuple, optional
        Figure size. If None, auto-computed based on plot type
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> # Plot alpha_y from technical fit
    >>> alpha_y = model.posterior_samples_ntc['alpha_y']  # shape: (samples, C, genes)
    >>> # Extract for one cell line (e.g., first)
    >>> alpha_y_cl0 = alpha_y[:, 0, :]
    >>> prior_alpha_y = ...  # Prior samples
    >>> gene_names = model.modalities['gene'].feature_meta['gene'].tolist()
    >>>
    >>> # Violin plot (auto if <100 genes)
    >>> fig = plot_1d_parameter(prior_alpha_y, alpha_y_cl0, gene_names, 'alpha_y')
    >>>
    >>> # Scatter plot for many genes
    >>> fig = plot_1d_parameter(prior_alpha_y, alpha_y_cl0, gene_names, 'alpha_y',
    >>>                         plot_type='scatter')
    >>>
    >>> # Subset to specific genes
    >>> fig = plot_1d_parameter(prior_alpha_y, alpha_y_cl0, gene_names, 'alpha_y',
    >>>                         subset_features=['GFI1B', 'TET2', 'MYB'])
    """
    n_features = prior_samples.shape[1]

    # Apply feature subset if provided
    if subset_features is not None:
        if len(subset_features) > 100:
            warnings.warn(f"Plotting {len(subset_features)} features - plot may be crowded")

        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        indices = [name_to_idx[name] for name in subset_features if name in name_to_idx]

        prior_samples = prior_samples[:, indices]
        posterior_samples = posterior_samples[:, indices]
        feature_names = [feature_names[i] for i in indices]
        n_features = len(indices)

    # Decide plot type
    if plot_type == 'auto':
        if n_features <= max_violin_features:
            plot_type = 'violin'
        else:
            # Subset to top mismatches for violin, or use scatter
            plot_type = 'scatter'

    if plot_type == 'violin':
        return _plot_1d_violin(
            prior_samples, posterior_samples, feature_names, param_name,
            order_by, custom_order, ax, figsize, **kwargs
        )
    elif plot_type == 'scatter':
        return _plot_1d_scatter(
            prior_samples, posterior_samples, feature_names, param_name,
            max_violin_features, metric, is_additive, ax, figsize, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'auto', 'violin', or 'scatter'")


def _plot_1d_violin(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    param_name: str,
    order_by: str = 'mean',
    custom_order: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> plt.Figure:
    """Internal: Create split violin plot."""
    n_features = len(feature_names)

    # Auto-compute figure size
    if figsize is None:
        width = max(12, n_features * 0.3)
        figsize = (width, 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Check for extreme prior values and clip if needed
    prior_max = np.max(prior_samples)
    post_max = np.max(posterior_samples)
    if prior_max > 1e10 or prior_max > 1000 * post_max:
        # Clip prior to reasonable range for visualization
        clip_upper = np.percentile(posterior_samples, 99.9) * 10
        prior_samples_clipped = np.clip(prior_samples, None, clip_upper)
        warnings.warn(f"Prior samples contain extreme values (max={prior_max:.2e}). "
                     f"Clipping to {clip_upper:.2e} for visualization. "
                     f"This does not affect the actual model fitting.")
        prior_samples = prior_samples_clipped

    # Prepare data
    df = prepare_violin_data(
        prior_samples, posterior_samples, feature_names,
        order_by, custom_order, subset_features=None
    )

    # Create split violin plot with error handling
    try:
        sns.violinplot(
            data=df,
            x='feature',
            y='value',
            hue='distribution',
            split=True,
            ax=ax,
            palette={'Prior': '#1f77b4', 'Posterior': '#ff7f0e'},
            **kwargs
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fall back to boxplot if violin fails (e.g., singular covariance)
        warnings.warn(f"Violin plot failed ({str(e)}), using boxplot instead")
        sns.boxplot(
            data=df,
            x='feature',
            y='value',
            hue='distribution',
            ax=ax,
            palette={'Prior': '#1f77b4', 'Posterior': '#ff7f0e'},
            **kwargs
        )

    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.set_title(f'{param_name} (Prior vs Posterior)\nOrdered by: {order_by}')
    ax.legend(title='Distribution')

    # Rotate x-axis labels if many features
    if n_features > 20:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return fig


def _plot_1d_scatter(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    param_name: str,
    max_features_for_subset: int = 100,
    metric: str = 'posterior_coverage',
    is_additive: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_top_n: int = 10,
    **kwargs
) -> plt.Figure:
    """Internal: Create scatter plot of mean shift vs metric."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute mean shift and chosen metric for all features
    df = compute_log2fc_vs_overlap(prior_samples, posterior_samples, feature_names,
                                    metric=metric, is_additive=is_additive)

    # Determine shift column name and x-axis label
    if is_additive:
        shift_col = 'difference'
        xlabel = 'Posterior mean - Prior mean'
    else:
        shift_col = 'log2fc'
        xlabel = 'log2(Posterior mean / Prior mean)'

    # Determine metric column name and y-axis label
    if metric == 'overlap':
        metric_col = 'overlap'
        ylabel = 'Distribution Overlap (%)'
        ascending = True  # Lower is more different
    elif metric == 'kl_divergence':
        metric_col = 'kl_divergence'
        ylabel = 'KL Divergence (nats)'
        ascending = False  # Higher is more different
    elif metric == 'posterior_coverage':
        metric_col = 'posterior_coverage'
        ylabel = 'Posterior Coverage (%)'
        ascending = True  # Lower indicates prior-posterior mismatch

    # Plot scatter
    scatter_kwargs = {'alpha': 0.6, 's': 30}
    scatter_kwargs.update(kwargs)

    ax.scatter(df[shift_col], df[metric_col], **scatter_kwargs)

    # Annotate top mismatches
    if show_top_n > 0:
        if ascending:
            # For overlap/coverage: lower is more different
            top_mismatch = df.nsmallest(show_top_n, metric_col)
        else:
            # For KL divergence: higher is more different
            top_mismatch = df.nlargest(show_top_n, metric_col)

        for _, row in top_mismatch.iterrows():
            ax.annotate(
                row['feature'],
                xy=(row[shift_col], row[metric_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{param_name} (Prior vs Posterior)\n'
                 f'{len(feature_names)} features')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_2d_parameter(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    dimension_names: List[str],
    param_name: str,
    plot_type: str = 'auto',
    color_by: str = 'dimension',
    separate_plots: bool = False,
    order_by: str = 'mean',
    custom_order: Optional[List[str]] = None,
    subset_features: Optional[List[str]] = None,
    max_violin_features: int = 100,
    metric: str = 'posterior_coverage',
    is_additive: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Union[plt.Figure, List[plt.Figure]]:
    """
    Plot prior vs posterior for 2D parameter (e.g., alpha_y with multiple cell lines).

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features, n_dims) or (n_samples, n_dims, n_features)
    posterior_samples : np.ndarray
        Posterior samples, same shape as prior_samples
    feature_names : List[str]
        Feature names (genes, etc.)
    dimension_names : List[str]
        Dimension names (e.g., cell lines, SpliZVD dimensions)
    param_name : str
        Parameter name for plot title
    plot_type : str
        'auto', 'violin', or 'scatter'
    color_by : str
        'dimension' or 'feature' (for scatter plots)
    separate_plots : bool
        If True, create separate scatter plots per dimension
    order_by : str
        How to order features in violin plots
    custom_order : List[str], optional
        Custom feature ordering
    subset_features : List[str], optional
        Subset to specific features
    max_violin_features : int
        Max features for violin before switching to scatter
    metric : str
        Comparison metric for scatter plots: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
    is_additive : bool
        If True, use difference (posterior - prior) for scatter plots.
        If False, use log2 fold change. (default: False)
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional plotting arguments

    Returns
    -------
    plt.Figure or List[plt.Figure]
        Matplotlib figure(s)

    Examples
    --------
    >>> # Plot alpha_y across multiple cell lines
    >>> alpha_y = model.posterior_samples_ntc['alpha_y']  # (samples, C, genes)
    >>> prior_alpha_y = ...  # Prior samples
    >>> gene_names = model.modalities['gene'].feature_meta['gene'].tolist()
    >>> cell_lines = ['K562', 'Jurkat', 'THP1']
    >>>
    >>> # Violin plot with rows per cell line
    >>> fig = plot_2d_parameter(prior_alpha_y, alpha_y, gene_names, cell_lines, 'alpha_y')
    >>>
    >>> # Scatter plot colored by dimension
    >>> fig = plot_2d_parameter(prior_alpha_y, alpha_y, gene_names, cell_lines, 'alpha_y',
    >>>                         plot_type='scatter', color_by='dimension')
    >>>
    >>> # Separate scatter plots per cell line
    >>> figs = plot_2d_parameter(prior_alpha_y, alpha_y, gene_names, cell_lines, 'alpha_y',
    >>>                          plot_type='scatter', separate_plots=True)
    """
    # Determine shape: (samples, features, dims) or (samples, dims, features)
    if prior_samples.shape[1] == len(feature_names):
        # (samples, features, dims)
        n_features = prior_samples.shape[1]
        n_dims = prior_samples.shape[2]
    elif prior_samples.shape[2] == len(feature_names):
        # (samples, dims, features) - transpose to (samples, features, dims)
        prior_samples = prior_samples.transpose(0, 2, 1)
        posterior_samples = posterior_samples.transpose(0, 2, 1)
        n_features = prior_samples.shape[1]
        n_dims = prior_samples.shape[2]
    else:
        raise ValueError(f"Cannot determine parameter shape. Expected {len(feature_names)} features.")

    # Apply feature subset
    if subset_features is not None:
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        indices = [name_to_idx[name] for name in subset_features if name in name_to_idx]

        prior_samples = prior_samples[:, indices, :]
        posterior_samples = posterior_samples[:, indices, :]
        feature_names = [feature_names[i] for i in indices]
        n_features = len(indices)

    # Decide plot type
    if plot_type == 'auto':
        if n_features <= max_violin_features and n_dims < 10:
            plot_type = 'violin'
        else:
            plot_type = 'scatter'

    if plot_type == 'violin':
        return _plot_2d_violin(
            prior_samples, posterior_samples, feature_names, dimension_names,
            param_name, order_by, custom_order, figsize, **kwargs
        )
    elif plot_type == 'scatter':
        return _plot_2d_scatter(
            prior_samples, posterior_samples, feature_names, dimension_names,
            param_name, color_by, separate_plots, metric, is_additive, figsize, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def _plot_2d_violin(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    dimension_names: List[str],
    param_name: str,
    order_by: str,
    custom_order: Optional[List[str]],
    figsize: Optional[Tuple[int, int]],
    **kwargs
) -> plt.Figure:
    """Internal: Create multi-row violin plot for 2D parameters."""
    n_dims = len(dimension_names)
    n_features = len(feature_names)

    # Check for extreme prior values and clip symmetrically if needed
    prior_max = np.max(np.abs(prior_samples))
    post_max = np.max(np.abs(posterior_samples))
    if prior_max > 1e10 or prior_max > 1000 * post_max:
        # Clip prior to reasonable range for visualization (symmetric bounds)
        clip_bound = np.percentile(np.abs(posterior_samples), 99.9) * 10
        prior_samples = np.clip(prior_samples, -clip_bound, clip_bound)
        warnings.warn(f"Prior samples contain extreme values (max abs={prior_max:.2e}). "
                     f"Clipping to ±{clip_bound:.2e} for visualization. "
                     f"This does not affect the actual model fitting.")

    # Auto-compute figure size
    if figsize is None:
        width = max(12, n_features * 0.3)
        height = max(4, n_dims * 3)
        figsize = (width, height)

    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]

    # Plot each dimension
    for dim_idx, (ax, dim_name) in enumerate(zip(axes, dimension_names)):
        # Extract samples for this dimension
        prior_dim = prior_samples[:, :, dim_idx]
        post_dim = posterior_samples[:, :, dim_idx]

        # Prepare data
        df = prepare_violin_data(
            prior_dim, post_dim, feature_names,
            order_by, custom_order, subset_features=None
        )

        # Plot with error handling
        try:
            sns.violinplot(
                data=df,
                x='feature',
                y='value',
                hue='distribution',
                split=True,
                ax=ax,
                palette={'Prior': '#1f77b4', 'Posterior': '#ff7f0e'},
                **kwargs
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fall back to boxplot if violin fails
            warnings.warn(f"Violin plot failed for {dim_name} ({str(e)}), using boxplot instead")
            sns.boxplot(
                data=df,
                x='feature',
                y='value',
                hue='distribution',
                ax=ax,
                palette={'Prior': '#1f77b4', 'Posterior': '#ff7f0e'},
                **kwargs
            )

        ax.set_ylabel(f'{dim_name}\nValue')
        ax.set_xlabel('')
        if dim_idx == 0:
            ax.set_title(f'{param_name} (Prior vs Posterior)\nOrdered by: {order_by}')
            ax.legend(title='Distribution', loc='upper right')
        else:
            ax.legend().remove()

        ax.grid(True, alpha=0.3, axis='y')

    # Set x-label on bottom plot
    axes[-1].set_xlabel('Feature')

    # Rotate x-axis labels if many features
    if n_features > 20:
        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=90, ha='right')

    plt.tight_layout()
    return fig


def _plot_2d_scatter(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    dimension_names: List[str],
    param_name: str,
    color_by: str,
    separate_plots: bool,
    metric: str,
    is_additive: bool,
    figsize: Optional[Tuple[int, int]],
    **kwargs
) -> Union[plt.Figure, List[plt.Figure]]:
    """Internal: Create scatter plot(s) for 2D parameters."""
    n_dims = len(dimension_names)

    if separate_plots:
        # Create separate scatter plot for each dimension
        figs = []
        for dim_idx, dim_name in enumerate(dimension_names):
            fig = _plot_1d_scatter(
                prior_samples[:, :, dim_idx],
                posterior_samples[:, :, dim_idx],
                feature_names,
                f'{param_name} ({dim_name})',
                max_features_for_subset=100,
                metric=metric,
                is_additive=is_additive,
                figsize=figsize or (10, 6),
                **kwargs
            )
            figs.append(fig)
        return figs

    else:
        # Single scatter plot with all dimensions
        if figsize is None:
            figsize = (10, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # Determine shift column name and x-axis label
        if is_additive:
            shift_col = 'difference'
            xlabel = 'Posterior mean - Prior mean'
        else:
            shift_col = 'log2fc'
            xlabel = 'log2(Posterior mean / Prior mean)'

        # Determine metric column name and y-axis label
        if metric == 'overlap':
            metric_col = 'overlap'
            ylabel = 'Distribution Overlap (%)'
        elif metric == 'kl_divergence':
            metric_col = 'kl_divergence'
            ylabel = 'KL Divergence (nats)'
        elif metric == 'posterior_coverage':
            metric_col = 'posterior_coverage'
            ylabel = 'Posterior Coverage (%)'

        # Compute mean shift and metric for all features and dimensions
        all_data = []
        for dim_idx, dim_name in enumerate(dimension_names):
            df = compute_log2fc_vs_overlap(
                prior_samples[:, :, dim_idx],
                posterior_samples[:, :, dim_idx],
                feature_names,
                metric=metric,
                is_additive=is_additive
            )
            df['dimension'] = dim_name
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)

        # Plot with coloring
        if color_by == 'dimension':
            for dim_name in dimension_names:
                df_dim = df_all[df_all['dimension'] == dim_name]
                ax.scatter(df_dim[shift_col], df_dim[metric_col],
                          label=dim_name, alpha=0.6, s=30, **kwargs)

            # Use "Technical Group" for alpha parameters, "Dimension" for others
            legend_title = 'Technical Group' if 'alpha' in param_name.lower() else 'Dimension'
            ax.legend(title=legend_title)

        elif color_by == 'feature':
            # Color by feature (only practical if few features)
            if len(feature_names) > 20:
                warnings.warn("Many features - coloring by feature may be hard to interpret")

            for feature in feature_names:
                df_feat = df_all[df_all['feature'] == feature]
                ax.scatter(df_feat[shift_col], df_feat[metric_col],
                          label=feature, alpha=0.6, s=30, **kwargs)

            if len(feature_names) <= 20:
                ax.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{param_name} (Prior vs Posterior)\n'
                     f'{len(feature_names)} features × {n_dims} dimensions')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
