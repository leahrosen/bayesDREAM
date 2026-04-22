"""
Plotting module for bayesDREAM.

This module provides comprehensive visualization functions for CRISPR screen analysis,
including:
- Prior/posterior goodness-of-fit plots
- X_true distributions (scatter, violin, density)
- Posterior density line plots
- DE comparisons with external methods (e.g., edgeR)
- Diagnostic plots (sum factors, etc.)
- X-Y relationship plots
"""

# Prior/posterior goodness-of-fit plots
from .prior_posterior import (
    plot_scalar_parameter,
    plot_1d_parameter,
    plot_2d_parameter
)

# Color scheme management
from .colors import ColorScheme, build_guide_colors, lighten, darken

# Helper utilities
from .helpers import to_np, per_cell_mean_std, resolve_guide_labels
from .utils import (
    hill_xinf_samples,
    dependency_mask_from_n,
    abs_n_gt_tol_mask,
    log2_pos,
    hill_y
)

# Basic x_true plots
from .basic import (
    scatter_by_guide,
    scatter_ci95_by_guide,
    violin_by_guide_log2,
    filled_density_by_guide_log2,
    scatter_param_mean_vs_ci,
    plot_parameter_ci_panel,
    extract_posterior_dataframe
)

# Posterior density plots
from .posterior import (
    plot_posterior_density_lines,
    plot_xtrue_density_by_guide,
    plot_parameter_density_with_xtrue
)

# DE comparison plots
from .de_comparison import (
    compute_log2fc_metrics,
    compute_log2fc_obs_for_cells,
    prepare_de_for_cg,
    scatter_and_heatmap_edger_vs_bayes,
    plot_edger_vs_bayes_full_range,
    plot_edger_vs_bayes_observed_range,
    dependency_mask_from_n
)

# Diagnostic plots
from .diagnostics import plot_sum_factor_comparison

__all__ = [
    # Prior/posterior
    'plot_scalar_parameter',
    'plot_1d_parameter',
    'plot_2d_parameter',
    # Colors
    'ColorScheme',
    'build_guide_colors',
    'lighten',
    'darken',
    # Helpers
    'to_np',
    'per_cell_mean_std',
    'resolve_guide_labels',
    'hill_xinf_samples',
    'abs_n_gt_tol_mask',
    'log2_pos',
    'hill_y',
    # Basic plots
    'scatter_by_guide',
    'scatter_ci95_by_guide',
    'violin_by_guide_log2',
    'filled_density_by_guide_log2',
    'scatter_param_mean_vs_ci',
    'plot_parameter_ci_panel',
    'extract_posterior_dataframe',
    # Posterior plots
    'plot_posterior_density_lines',
    'plot_xtrue_density_by_guide',
    'plot_parameter_density_with_xtrue',
    # DE comparison
    'compute_log2fc_metrics',
    'compute_log2fc_obs_for_cells',
    'prepare_de_for_cg',
    'scatter_and_heatmap_edger_vs_bayes',
    'plot_edger_vs_bayes_full_range',
    'plot_edger_vs_bayes_observed_range',
    'dependency_mask_from_n',
    # Diagnostics
    'plot_sum_factor_comparison',
]
