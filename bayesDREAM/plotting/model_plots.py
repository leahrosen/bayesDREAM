"""
Convenience plotting methods for bayesDREAM model parameters.

These are added as methods to the bayesDREAM model class for easy access.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import warnings

from .prior_posterior import (
    plot_scalar_parameter,
    plot_1d_parameter,
    plot_2d_parameter
)
from .prior_sampling import get_prior_samples


class PlottingMixin:
    """Mixin class providing plotting methods for bayesDREAM models."""

    def _get_technical_group_names(self) -> List[str]:
        """
        Get informative names for technical groups from model metadata.

        Returns
        -------
        List[str]
            Names for each technical group (e.g., ['K562', 'Jurkat'] instead of ['TG_0', 'TG_1'])
        """
        if not hasattr(self, 'meta') or 'technical_group_code' not in self.meta.columns:
            # No technical groups, return generic name
            return ['TG_0']

        # Get unique technical group codes and their corresponding covariate values
        if hasattr(self, 'technical_group_col') and self.technical_group_col:
            # Use the stored column name if available
            group_col = self.technical_group_col
            unique_groups = self.meta.groupby('technical_group_code')[group_col].first().sort_index()
            return unique_groups.tolist()
        else:
            # Try to infer from commonly used columns
            possible_cols = ['cell_line', 'condition', 'batch', 'sample']
            for col in possible_cols:
                if col in self.meta.columns:
                    unique_groups = self.meta.groupby('technical_group_code')[col].first().sort_index()
                    # Check if it actually varies by technical group
                    if len(unique_groups.unique()) == len(unique_groups):
                        return unique_groups.tolist()

            # Fall back to generic names if can't find informative column
            n_groups = self.meta['technical_group_code'].nunique()
            return [f'TG_{i}' for i in range(n_groups)]

    def plot_technical_fit(
        self,
        param: str = 'alpha_y',
        modality_name: Optional[str] = None,
        technical_group_index: Optional[int] = None,
        order_by: str = 'mean',
        subset_features: Optional[List[str]] = None,
        plot_type: str = 'auto',
        metric: str = 'posterior_coverage',
        cell_line_index: Optional[int] = None,  # Deprecated, backward compatibility
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for technical fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'beta_o', 'alpha_x', 'alpha_y', 'mu_ntc', 'o_y'
        modality_name : str, optional
            Modality name (default: primary modality)
        technical_group_index : int, optional
            Technical group index for alpha_y (e.g., 0 for first group, 1 for second).
            If None, plots all technical groups as a 2D plot.
        order_by : str
            Feature ordering: 'mean', 'difference', 'alphabetical', 'input'
        subset_features : List[str], optional
            Subset to specific features
        plot_type : str
            'auto', 'violin', or 'scatter'
        metric : str
            Prior/posterior comparison metric: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot beta_o (scalar parameter)
        >>> fig = model.plot_technical_fit('beta_o')
        >>>
        >>> # Plot alpha_y for first technical group
        >>> fig = model.plot_technical_fit('alpha_y', technical_group_index=0)
        >>>
        >>> # Plot alpha_y for all technical groups (2D parameter)
        >>> fig = model.plot_technical_fit('alpha_y')
        >>>
        >>> # Plot alpha_y for specific genes
        >>> fig = model.plot_technical_fit('alpha_y', subset_features=['GFI1B', 'TET2'])
        """
        # Handle backward compatibility for cell_line_index
        if cell_line_index is not None:
            if technical_group_index is not None:
                raise ValueError("Cannot specify both technical_group_index and cell_line_index. "
                               "Use technical_group_index (cell_line_index is deprecated).")
            warnings.warn("cell_line_index is deprecated, use technical_group_index instead", DeprecationWarning)
            technical_group_index = cell_line_index

        # Check for invalid technical_group_index
        if technical_group_index is not None and technical_group_index == 0:
            raise ValueError(
                "technical_group_index=0 is the baseline/reference group and has no variation "
                "(all values are set to baseline: 1.0 for multiplicative, 0.0 for additive). "
                "Please specify a different technical group index (1, 2, ...) or omit to plot all groups."
            )

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)

        # Get modality-specific posterior samples
        if not hasattr(modality, 'posterior_samples_technical') or modality.posterior_samples_technical is None:
            raise ValueError(f"No technical fit found for modality '{modality_name}'. Run fit_technical(modality_name='{modality_name}') first.")

        posterior = modality.posterior_samples_technical

        # Get feature names - will be adjusted based on alpha_y source later
        if 'gene' in modality.feature_meta.columns:
            modality_feature_names = modality.feature_meta['gene'].tolist()
        elif 'gene_name' in modality.feature_meta.columns:
            modality_feature_names = modality.feature_meta['gene_name'].tolist()
        else:
            modality_feature_names = modality.feature_meta.index.tolist()

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='technical',
            modality_name=modality_name,
            nsamples=posterior[list(posterior.keys())[0]].shape[0],  # Match posterior sample count
            distribution=modality.distribution
        )

        # Extract parameter and prior
        if param == 'beta_o':
            # Scalar parameter
            if 'beta_o' not in posterior:
                raise ValueError("beta_o not found in posterior_samples_technical")

            post_samples = posterior['beta_o'].numpy() if hasattr(posterior['beta_o'], 'numpy') else posterior['beta_o']
            prior_samples = prior_dict['beta_o'].numpy() if hasattr(prior_dict['beta_o'], 'numpy') else prior_dict['beta_o']

            # Squeeze to 1D if needed (handle shape (n_samples, 1, 1) or (n_samples, 1))
            post_samples = np.squeeze(post_samples)
            prior_samples = np.squeeze(prior_samples)

            return plot_scalar_parameter(prior_samples, post_samples, 'beta_o', metric=metric, **kwargs)

        elif param == 'alpha_x':
            # 1D or 2D parameter (per technical group)
            if not hasattr(self, 'alpha_x_prefit'):
                raise ValueError("alpha_x not found. Check if cis gene was included in technical fit.")

            post_samples = self.alpha_x_prefit
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # For alpha_x, we need cis priors (will implement below)
            # For now use technical priors as approximation
            if 'alpha_x' in prior_dict:
                prior_samples = prior_dict['alpha_x'].numpy() if hasattr(prior_dict['alpha_x'], 'numpy') else prior_dict['alpha_x']
            else:
                # Fall back to alpha_y structure if alpha_x not in prior_dict
                warnings.warn("alpha_x not in prior_dict - using alpha_y structure as approximation")
                prior_samples = prior_dict['alpha_y'].numpy() if hasattr(prior_dict['alpha_y'], 'numpy') else prior_dict['alpha_y']
                if prior_samples.ndim == 3:
                    prior_samples = prior_samples[:, :, 0]  # Take first gene dimension

            # Get informative technical group names
            group_names = self._get_technical_group_names()

            # Handle different dimensionalities
            if post_samples.ndim == 1:
                # (samples,) - single value across all groups
                return plot_scalar_parameter(prior_samples, post_samples, 'alpha_x', metric=metric, **kwargs)
            elif post_samples.ndim == 2:
                # (samples, technical_groups) - one value per group
                if technical_group_index is not None:
                    # Check if trying to plot baseline group
                    if technical_group_index == 0:
                        # Already caught by earlier check, but this makes it explicit
                        raise ValueError("Cannot plot technical_group_index=0 (baseline group with no variation)")

                    # Plot single technical group
                    prior_tg = prior_samples[:, technical_group_index]
                    post_tg = post_samples[:, technical_group_index]
                    group_name = group_names[technical_group_index]

                    return plot_scalar_parameter(
                        prior_tg, post_tg, f'alpha_x ({group_name})', metric=metric, **kwargs
                    )
                else:
                    # Plot all technical groups - treat as separate features
                    # Exclude baseline group (TG_0) which is always constant
                    if post_samples.shape[1] > 1:
                        prior_samples = prior_samples[:, 1:]  # Skip first group
                        post_samples = post_samples[:, 1:]
                        group_names = group_names[1:]  # Skip first name
                        print(f"Note: Excluding baseline technical group (index 0) which has no variation")

                    return plot_1d_parameter(
                        prior_samples, post_samples, group_names, 'alpha_x',
                        order_by='input', plot_type='violin', metric=metric, **kwargs
                    )
            else:
                raise ValueError(f"Unexpected alpha_x shape: {post_samples.shape}")

        elif param == 'alpha_y':
            # 1D or 2D parameter: (samples, genes) or (samples, cell_lines, genes)

            # Determine if we should plot in log2 space based on distribution
            # Multiplicative (negbinom): plot log2(alpha_y_mult) - baseline is 0 in log2 space
            # Additive (normal, binomial, etc.): plot alpha_y_add (already in log2 space) - baseline is 0
            is_multiplicative = modality.distribution == 'negbinom'
            is_additive = not is_multiplicative  # For scatter plot x-axis

            if is_multiplicative:
                # For negbinom: use multiplicative and convert to log2
                if 'alpha_y_mult' in posterior:
                    post_samples = posterior['alpha_y_mult']
                elif 'alpha_y' in posterior:
                    post_samples = posterior['alpha_y']
                else:
                    raise ValueError(f"alpha_y not found for modality '{modality.name}'. "
                                   "Run fit_technical(modality_name='{modality.name}') first.")

                if hasattr(post_samples, 'numpy'):
                    post_samples = post_samples.numpy()

                # Convert to log2 space (baseline group will be 0 since log2(1)=0)
                post_samples = np.log2(post_samples + 1e-10)  # Small epsilon to avoid log(0)

                # Use log2 priors
                prior_samples = prior_dict['log2_alpha_y'].numpy() if hasattr(prior_dict['log2_alpha_y'], 'numpy') else prior_dict['log2_alpha_y']
                # Add baseline row of zeros
                if prior_samples.ndim == 2:
                    prior_samples = np.concatenate([np.zeros((1, prior_samples.shape[1])), prior_samples], axis=0)
                elif prior_samples.ndim == 3:
                    prior_samples = np.concatenate([np.zeros((prior_samples.shape[0], 1, prior_samples.shape[2])), prior_samples], axis=1)

                param_label = 'log2(alpha_y)'
            else:
                # For additive distributions: use additive (already in log2 space)
                if 'alpha_y_add' in posterior:
                    post_samples = posterior['alpha_y_add']
                elif 'alpha_y' in posterior:
                    post_samples = posterior['alpha_y']
                else:
                    raise ValueError(f"alpha_y not found for modality '{modality.name}'. "
                                   "Run fit_technical(modality_name='{modality.name}') first.")

                if hasattr(post_samples, 'numpy'):
                    post_samples = post_samples.numpy()

                # Already in additive (log2) space
                # Use log2 priors (same as multiplicative, baseline is 0)
                prior_samples = prior_dict['log2_alpha_y'].numpy() if hasattr(prior_dict['log2_alpha_y'], 'numpy') else prior_dict['log2_alpha_y']
                # Add baseline row of zeros
                if prior_samples.ndim == 2:
                    prior_samples = np.concatenate([np.zeros((1, prior_samples.shape[1])), prior_samples], axis=0)
                elif prior_samples.ndim == 3:
                    prior_samples = np.concatenate([np.zeros((prior_samples.shape[0], 1, prior_samples.shape[2])), prior_samples], axis=1)

                param_label = 'alpha_y (additive)'

            # Use modality feature names (posterior is modality-specific)
            feature_names = modality_feature_names

            # If primary modality with cis gene extracted, prior includes cis gene but posterior doesn't
            # Need to exclude cis gene from prior to match posterior shape
            if modality_name == self.primary_modality and hasattr(self, 'counts') and \
               self.cis_gene is not None and 'cis' in self.modalities:
                # Check if cis gene is in original counts
                if isinstance(self.counts, pd.DataFrame) and self.cis_gene in self.counts.index:
                    all_genes_orig = self.counts.index.tolist()
                    cis_idx_orig = all_genes_orig.index(self.cis_gene)

                    # Prior was sampled with all features including cis gene
                    # Exclude cis gene to match modality alpha_y (which excludes cis)
                    if prior_samples.ndim == 2:
                        # (samples, features)
                        if cis_idx_orig < prior_samples.shape[1]:
                            all_idx = list(range(prior_samples.shape[1]))
                            trans_idx = [i for i in all_idx if i != cis_idx_orig]
                            prior_samples = prior_samples[:, trans_idx]
                    elif prior_samples.ndim == 3:
                        # (samples, technical_groups, features)
                        if cis_idx_orig < prior_samples.shape[2]:
                            all_idx = list(range(prior_samples.shape[2]))
                            trans_idx = [i for i in all_idx if i != cis_idx_orig]
                            prior_samples = prior_samples[:, :, trans_idx]

            # Handle dimensionality and shape checking
            if post_samples.ndim == 2:
                # (samples, genes) - check shape compatibility
                expected_n_features = len(feature_names)
                actual_n_features = post_samples.shape[-1]
                if actual_n_features != expected_n_features:
                    raise ValueError(
                        f"Shape mismatch: alpha_y has {actual_n_features} features, "
                        f"but modality '{modality.name}' has {expected_n_features} features. "
                        f"Try specifying modality_name explicitly or check that fit_technical "
                        f"was run for this modality."
                    )
                return plot_1d_parameter(
                    prior_samples, post_samples, feature_names, param_label,
                    order_by, subset_features=subset_features, plot_type=plot_type,
                    metric=metric, is_additive=is_additive, **kwargs
                )
            elif post_samples.ndim == 3:
                # (samples, technical_groups, genes)
                # Get informative technical group names
                group_names = self._get_technical_group_names()

                if technical_group_index is not None:
                    # Plot single technical group - select FIRST, then check shape
                    prior_tg = prior_samples[:, technical_group_index, :]
                    post_tg = post_samples[:, technical_group_index, :]

                    # Now check shape compatibility
                    expected_n_features = len(feature_names)
                    actual_n_features = post_tg.shape[-1]
                    if actual_n_features != expected_n_features:
                        raise ValueError(
                            f"Shape mismatch: alpha_y has {actual_n_features} features, "
                            f"but modality '{modality.name}' has {expected_n_features} features. "
                            f"Try specifying modality_name explicitly or check that fit_technical "
                            f"was run for this modality."
                        )

                    # Filter out features where posterior is constant (no variance in NTC data)
                    # These were set to baseline because NTC had no variation to fit
                    # For multiplicative: baseline=1, for additive: baseline=0
                    post_std = np.std(post_tg, axis=0)
                    non_constant_mask = post_std > 1e-10  # Not all the same value

                    if not non_constant_mask.all():
                        n_constant = (~non_constant_mask).sum()
                        warnings.warn(
                            f"Excluding {n_constant} features with no variance in NTC data "
                            f"(set to baseline: 1.0 for mult, 0.0 for add)"
                        )

                        prior_tg = prior_tg[:, non_constant_mask]
                        post_tg = post_tg[:, non_constant_mask]
                        feature_names = [name for name, keep in zip(feature_names, non_constant_mask) if keep]

                    group_name = group_names[technical_group_index]
                    return plot_1d_parameter(
                        prior_tg, post_tg, feature_names, f'{param_label} ({group_name})',
                        order_by, subset_features=subset_features, plot_type=plot_type,
                        metric=metric, is_additive=is_additive, **kwargs
                    )
                else:
                    # Plot all technical groups (2D plot) - check shape compatibility
                    expected_n_features = len(feature_names)
                    actual_n_features = post_samples.shape[-1]
                    if actual_n_features != expected_n_features:
                        raise ValueError(
                            f"Shape mismatch: alpha_y has {actual_n_features} features, "
                            f"but modality '{modality.name}' has {expected_n_features} features. "
                            f"Try specifying modality_name explicitly or check that fit_technical "
                            f"was run for this modality."
                        )

                    # Exclude baseline group (TG_0) which is always constant
                    if post_samples.shape[1] > 1:
                        prior_samples = prior_samples[:, 1:, :]  # Skip first group
                        post_samples = post_samples[:, 1:, :]
                        group_names = group_names[1:]  # Skip first name
                        print(f"Note: Excluding baseline technical group (index 0) which has no variation")

                    return plot_2d_parameter(
                        prior_samples, post_samples, feature_names, group_names, param_label,
                        order_by=order_by, subset_features=subset_features,
                        plot_type=plot_type, metric=metric, is_additive=is_additive, **kwargs
                    )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be one of: "
                           "'beta_o', 'alpha_x', 'alpha_y', 'mu_ntc', 'o_y'")

    def plot_cis_fit(
        self,
        param: str = 'x_true',
        order_by: str = 'mean',
        metric: str = 'posterior_coverage',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for cis fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'x_true', 'mu_x', 'log2_x_eff'
        order_by : str
            Guide ordering: 'mean', 'difference', 'alphabetical', 'input'
        metric : str
            Prior/posterior comparison metric: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot x_true (cis gene expression per guide)
        >>> fig = model.plot_cis_fit('x_true')
        """
        if not hasattr(self, 'posterior_samples_cis'):
            raise ValueError("No cis fit found. Run fit_cis() first.")

        posterior = self.posterior_samples_cis

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='cis',
            nsamples=posterior[list(posterior.keys())[0]].shape[0]
        )

        # Get guide names
        guide_names = self.meta_guides['guide'].tolist() if hasattr(self, 'meta_guides') else None
        if guide_names is None:
            n_guides = posterior['x_true'].shape[1] if 'x_true' in posterior else posterior['mu_x'].shape[1]
            guide_names = [f'Guide_{i}' for i in range(n_guides)]

        if param == 'x_true':
            if 'x_true' not in posterior:
                raise ValueError("x_true not found in posterior_samples_cis")

            post_samples = posterior['x_true']
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # Use sampled priors
            prior_samples = prior_dict['x_true'].numpy() if hasattr(prior_dict['x_true'], 'numpy') else prior_dict['x_true']

            return plot_1d_parameter(
                prior_samples, post_samples, guide_names, 'x_true',
                order_by, plot_type='violin', metric=metric, **kwargs
            )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be 'x_true'")

    def plot_trans_fit(
        self,
        param: str = 'theta',
        modality_name: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        order_by: str = 'mean',
        plot_type: str = 'auto',
        function_type: str = 'additive_hill',
        metric: str = 'posterior_coverage',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for trans fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'theta', 'gamma', 'mu_y'
        modality_name : str, optional
            Modality name (default: primary modality)
        subset_features : List[str], optional
            Subset to specific features
        order_by : str
            Feature ordering
        plot_type : str
            'auto', 'violin', or 'scatter'
        function_type : str
            Function type used in fit: 'additive_hill', 'single_hill', 'polynomial'
        metric : str
            Prior/posterior comparison metric: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot theta (Hill function parameters)
        >>> fig = model.plot_trans_fit('theta', function_type='additive_hill')
        >>>
        >>> # Plot for specific genes
        >>> fig = model.plot_trans_fit('theta', subset_features=['GFI1B', 'TET2'])
        """
        if not hasattr(self, 'posterior_samples_trans'):
            raise ValueError("No trans fit found. Run fit_trans() first.")

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)
        posterior = self.posterior_samples_trans

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='trans',
            modality_name=modality_name,
            nsamples=posterior[list(posterior.keys())[0]].shape[0],
            function_type=function_type,
            distribution=modality.distribution
        )

        # Get feature names
        if 'gene' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene'].tolist()
        elif 'gene_name' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene_name'].tolist()
        else:
            feature_names = modality.feature_meta.index.tolist()

        if param == 'theta':
            if 'theta' not in posterior:
                raise ValueError("theta not found in posterior_samples_trans")

            post_samples = posterior['theta']
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # Use sampled priors
            prior_samples = prior_dict['theta'].numpy() if hasattr(prior_dict['theta'], 'numpy') else prior_dict['theta']

            # theta is typically (samples, genes, n_params)
            if post_samples.ndim == 3:
                # Plot as 2D parameter with dimensions = parameter indices
                param_names = [f'param_{i}' for i in range(post_samples.shape[2])]
                return plot_2d_parameter(
                    prior_samples, post_samples, feature_names, param_names, 'theta',
                    order_by=order_by, subset_features=subset_features,
                    plot_type=plot_type, metric=metric, **kwargs
                )
            elif post_samples.ndim == 2:
                # Single parameter per gene
                return plot_1d_parameter(
                    prior_samples, post_samples, feature_names, 'theta',
                    order_by, subset_features=subset_features, plot_type=plot_type,
                    metric=metric, **kwargs
                )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be 'theta', 'gamma', or 'mu_y'")

    def plot_xy_data(self, *args, **kwargs):
        """
        Plot raw x-y data showing relationship between cis gene expression and modality values.

        This method is delegated to xy_plots.plot_xy_data(). See that function for full documentation.

        Parameters
        ----------
        feature : str
            Feature name (gene, junction, donor, etc.)
        modality_name : str, optional
            Modality name (default: primary modality)
        window : int
            k-NN window size for smoothing (default: 100 cells)
        show_correction : str
            'uncorrected', 'corrected', or 'both' (default: 'corrected')
        min_counts : int
            Minimum denominator for binomial/minimum total for multinomial (default: 3)
        color_palette : dict, optional
            Custom colors for technical groups
        show_hill_function : bool
            Overlay Hill function if trans model fitted (default: True)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure or plt.Axes
            Matplotlib figure or axes object

        Examples
        --------
        >>> # Plot gene counts with Hill function overlay
        >>> fig = model.plot_xy_data('TET2', window=100, show_hill_function=True)
        >>>
        >>> # Plot splice junction with filtering
        >>> fig = model.plot_xy_data('chr1:12345:67890:+',
        ...                          modality_name='splicing_sj',
        ...                          min_counts=5)
        >>>
        >>> # Plot with both corrected and uncorrected
        >>> fig = model.plot_xy_data('GFI1B', show_correction='both')
        """
        from .xy_plots import plot_xy_data
        return plot_xy_data(self, *args, **kwargs)

    def plot_trans_functions(self, features, **kwargs):
        """
        Plot fitted trans functions and/or their derivatives.

        Simple plot showing just the fitted Hill functions (no smoothed data).
        Useful for comparing multiple genes or viewing function shape with derivatives.

        Parameters
        ----------
        features : str or list of str
            Single feature name or list of feature names to plot
        modality_name : str, optional
            Modality name (default: primary modality)
        show_function : bool
            Show the fitted function y(x) (default: True)
        show_first_derivative : bool
            Show first derivative dy/dx (default: False)
        show_second_derivative : bool
            Show second derivative d²y/dx² (default: False)
        x_range : np.ndarray, optional
            X values to plot at. If None, uses range from model's x_true.
        n_points : int
            Number of points for x_range if auto-generated (default: 200)
        use_log2_x : bool
            Use log2(x) for x-axis (default: True)
        colors : str, list, or dict, optional
            Colors for each feature
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title
        legend : bool
            Show legend (default: True)
        overlay_roots : pd.DataFrame, pd.Series, dict, or None
            Summary DataFrame (or single row) from ``save_trans_summary``.
            When provided, draws dashed vertical lines at derivative roots on
            each subplot.  See ``plot_trans_functions`` in ``xy_plots.py`` for
            full documentation of the root overlay behaviour.
        overlay_roots_lw : float
            Line width for root vlines (default 1.0).
        overlay_roots_alpha : float
            Transparency for root vlines (default 0.8).
        overlay_roots_also_on_function : bool
            If True (default), also draw roots on the function subplot.

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot function and derivatives for one gene
        >>> model.plot_trans_functions('TET2', show_first_derivative=True,
        ...                            show_second_derivative=True)

        >>> # Overlay derivative roots from summary df
        >>> df = model.save_trans_summary(compute_derivative_roots=True)
        >>> model.plot_trans_functions('TET2', use_log2fc=True,
        ...                            show_function=True,
        ...                            show_first_derivative=True,
        ...                            show_second_derivative=True,
        ...                            show_third_derivative=True,
        ...                            overlay_roots=df)

        >>> # Plot first derivative of multiple genes
        >>> model.plot_trans_functions(['TET2', 'MYB', 'GFI1B'],
        ...                            show_function=False,
        ...                            show_first_derivative=True)
        """
        from .xy_plots import plot_trans_functions
        return plot_trans_functions(self, features, **kwargs)

    def plot_parameter_ci_panel(self, params: list, **kwargs):
        """
        Forest plot (dot + whisker CI) for posterior parameters across trans genes.

        Creates a plot with genes on the x-axis and parameter values (median + CI) on
        the y-axis. Multiple parameters are dodged side-by-side for comparison.

        This method is delegated to basic.plot_parameter_ci_panel().
        See that function for full documentation.

        Parameters
        ----------
        params : list of str
            Parameter names to plot (e.g., ['n_a', 'n_b'] or ['alpha', 'beta']).
            These must exist in posterior_samples_trans.
        modality_name : str, optional
            Modality name. If None, uses primary modality.
        genes : list of str, optional
            Specific genes to plot. If None, plots all genes (subject to max_genes).
        ci_level : float
            Credible interval level (default: 95.0 for 95% CI)
        sort_by : str
            How to sort genes: 'none', 'alphabetical', 'median', 'abs_median', 'effect'
        filter_dependent : bool
            If True, only show genes where CI excludes 0 (default: False)
        max_genes : int
            Maximum genes to plot (default: 100). Raises error if exceeded.
            Set to None to disable limit.
        ymin, ymax : float, optional
            Y-axis limits. If None, auto-scaled.
        title : str, optional
            Plot title. If None, auto-generated.
        color_palette : dict, optional
            Custom colors for parameters.
        show_gene_separators : bool
            Draw vertical lines between genes (default: True)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes

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

        >>> # Plot specific genes, sorted alphabetically
        >>> fig, ax = model.plot_parameter_ci_panel(
        ...     ['n_a', 'n_b'],
        ...     genes=['TET2', 'NFE2', 'MYB'],
        ...     sort_by='alphabetical'
        ... )
        """
        from .basic import plot_parameter_ci_panel
        return plot_parameter_ci_panel(self, params, **kwargs)

    def extract_posterior_dataframe(self, params: list, **kwargs):
        """
        Extract posterior parameters into a long-format DataFrame.

        This is useful for custom analysis or plotting with seaborn/plotnine.

        This method is delegated to basic.extract_posterior_dataframe().
        See that function for full documentation.

        Parameters
        ----------
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
            - gene, gene_idx, param, median, lo, hi, mean, std, ci_excludes_zero
            If include_samples=True, also: sample_idx, value

        Examples
        --------
        >>> # Get summary statistics
        >>> df = model.extract_posterior_dataframe(['n_a', 'n_b', 'K_a', 'K_b'])
        >>> df_dependent = df[df['ci_excludes_zero']]

        >>> # Get all samples for custom analysis
        >>> df_samples = model.extract_posterior_dataframe(['n_a'], include_samples=True)
        """
        from .basic import extract_posterior_dataframe
        return extract_posterior_dataframe(self, params, **kwargs)
