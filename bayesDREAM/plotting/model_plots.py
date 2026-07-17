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

__pdoc__ = {"ModelPlottingMixin": False}

class ModelPlottingMixin:
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
        if not hasattr(modality, 'posterior_samples_ntc') or modality.posterior_samples_ntc is None:
            raise ValueError(f"No technical fit found for modality '{modality_name}'. Run fit_ntc(modality_name='{modality_name}') first.")

        posterior = modality.posterior_samples_ntc

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
                raise ValueError("beta_o not found in posterior_samples_ntc")

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
                                   "Run fit_ntc(modality_name='{modality.name}') first.")

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
                                   "Run fit_ntc(modality_name='{modality.name}') first.")

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
        order_by: str = 'mean',
        metric: str = 'posterior_coverage',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for x_true from the cis fit.

        Parameters
        ----------
        order_by : str, default 'mean'
            Guide ordering: ``'mean'``, ``'difference'``, ``'alphabetical'``, ``'input'``.
        metric : str, default 'posterior_coverage'
            Prior/posterior comparison metric: ``'overlap'``, ``'kl_divergence'``,
            or ``'posterior_coverage'``.
        **kwargs
            Additional arguments forwarded to ``plot_1d_parameter``.

        Returns
        -------
        plt.Figure
        """
        if not hasattr(self, 'posterior_samples_cis'):
            raise ValueError("No cis fit found. Run fit_cis() first.")

        posterior = self.posterior_samples_cis

        if 'x_true' not in posterior:
            raise ValueError("x_true not found in posterior_samples_cis")

        prior_dict = get_prior_samples(
            self,
            fit_type='cis',
            nsamples=posterior[list(posterior.keys())[0]].shape[0]
        )

        guide_names = self.meta_guides['guide'].tolist() if hasattr(self, 'meta_guides') else None
        if guide_names is None:
            n_guides = posterior['x_true'].shape[1]
            guide_names = [f'Guide_{i}' for i in range(n_guides)]

        post_samples = posterior['x_true']
        if hasattr(post_samples, 'numpy'):
            post_samples = post_samples.numpy()

        prior_samples = prior_dict['x_true']
        if hasattr(prior_samples, 'numpy'):
            prior_samples = prior_samples.numpy()

        return plot_1d_parameter(
            prior_samples, post_samples, guide_names, 'x_true',
            order_by, plot_type='violin', metric=metric, **kwargs
        )

    def plot_trans_fit(
        self,
        modality_name: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        order_by: str = 'mean',
        plot_type: str = 'auto',
        function_type: str = 'additive_hill',
        metric: str = 'posterior_coverage',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for theta (trans function parameters).

        Parameters
        ----------
        modality_name : str, optional
            Modality name. Defaults to ``model.primary_modality``.
        subset_features : List[str], optional
            Subset to specific features by name.
        order_by : str, default 'mean'
            Feature ordering: ``'mean'``, ``'difference'``, ``'alphabetical'``, ``'input'``.
        plot_type : str, default 'auto'
            ``'auto'``, ``'violin'``, or ``'scatter'``.
        function_type : str, default 'additive_hill'
            Function type used in the trans fit: ``'additive_hill'``,
            ``'single_hill'``, or ``'polynomial'``.
        metric : str, default 'posterior_coverage'
            Prior/posterior comparison metric: ``'overlap'``, ``'kl_divergence'``,
            or ``'posterior_coverage'``.
        **kwargs
            Additional arguments forwarded to ``plot_1d_parameter`` or
            ``plot_2d_parameter``.

        Returns
        -------
        plt.Figure
        """
        if not hasattr(self, 'posterior_samples_trans'):
            raise ValueError("No trans fit found. Run fit_trans() first.")

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)
        posterior = self.posterior_samples_trans

        if 'theta' not in posterior:
            raise ValueError("theta not found in posterior_samples_trans")

        prior_dict = get_prior_samples(
            self,
            fit_type='trans',
            modality_name=modality_name,
            nsamples=posterior[list(posterior.keys())[0]].shape[0],
            function_type=function_type,
            distribution=modality.distribution
        )

        if 'gene' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene'].tolist()
        elif 'gene_name' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene_name'].tolist()
        else:
            feature_names = modality.feature_meta.index.tolist()

        post_samples = posterior['theta']
        if hasattr(post_samples, 'numpy'):
            post_samples = post_samples.numpy()

        prior_samples = prior_dict['theta']
        if hasattr(prior_samples, 'numpy'):
            prior_samples = prior_samples.numpy()

        if post_samples.ndim == 3:
            param_names = [f'param_{i}' for i in range(post_samples.shape[2])]
            return plot_2d_parameter(
                prior_samples, post_samples, feature_names, param_names, 'theta',
                order_by=order_by, subset_features=subset_features,
                plot_type=plot_type, metric=metric, **kwargs
            )
        else:
            return plot_1d_parameter(
                prior_samples, post_samples, feature_names, 'theta',
                order_by, subset_features=subset_features, plot_type=plot_type,
                metric=metric, **kwargs
            )

    def plot_xy_data(self, *args, **kwargs):
        """
        Plot raw x-y data (cis expression vs modality values) with k-NN smoothing.

        Delegates to ``xy_plots.plot_xy_data``; see that function for full
        parameter documentation.
        """
        from .xy_plots import plot_xy_data
        return plot_xy_data(self, *args, **kwargs)

    def plot_trans_functions(self, features, **kwargs):
        """
        Plot fitted trans functions and/or their derivatives.

        Parameters
        ----------
        features : str or list of str
            Feature name(s) to plot.

        Delegates to ``xy_plots.plot_trans_functions``; see that function for
        full parameter documentation.
        """
        from .xy_plots import plot_trans_functions
        return plot_trans_functions(self, features, **kwargs)

    def plot_parameter_ci_panel(self, params: list, **kwargs):
        """
        Forest plot (dot + whisker CI) for posterior parameters across trans genes.

        Parameters
        ----------
        params : list of str
            Parameter names to plot (e.g., ``['n_a', 'n_b']`` or
            ``['alpha', 'beta']``). Must exist in ``posterior_samples_trans``.

        Delegates to ``basic.plot_parameter_ci_panel``; see that function for
        full parameter documentation.

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes
        """
        from .basic import plot_parameter_ci_panel
        return plot_parameter_ci_panel(self, params, **kwargs)

    def extract_posterior_dataframe(self, params: list, **kwargs):
        """
        Extract posterior parameters into a long-format DataFrame.

        Parameters
        ----------
        params : list of str
            Parameter names to extract (e.g., ``['n_a', 'n_b', 'K_a', 'K_b']``).

        Delegates to ``basic.extract_posterior_dataframe``; see that function for
        full parameter documentation.
        """
        from .basic import extract_posterior_dataframe
        return extract_posterior_dataframe(self, params, **kwargs)
