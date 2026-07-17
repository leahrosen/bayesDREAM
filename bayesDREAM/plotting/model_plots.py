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
        Plot raw x-y data showing relationship between cis gene expression and modality values.

        Requires x_true to be set (must run fit_cis() first).

        Parameters
        ----------
        feature : str
            Feature name (junction, donor, etc.) OR gene name.
            - If a specific feature name (e.g., 'chr1:999788:999865'), plots that feature
            - If a gene name (e.g., 'HES4'), plots all features for that gene in subplots
            - Requires modality to have gene information ('gene', 'gene_name', or 'gene_id' columns)
        modality_name : str, optional
            Modality name (default: primary modality)
        window : int
            k-NN window size for smoothing (default: 100 cells)
        show_correction : str
            'uncorrected': no technical correction
            'corrected': apply alpha_y technical correction
            'both': show both side-by-side (default)
        min_counts : int, optional
            Minimum raw counts to include cells.
            Default depends on distribution: 0 for negbinom, 3 for binomial/multinomial.
            For negbinom: excludes cells with y_obs < min_counts
            For binomial: minimum denominator
            For multinomial: minimum total counts
        color_palette : dict, optional
            Custom colors for technical groups, keyed by the group label string.
            Example: {'K562': 'crimson', 'TF1': 'dodgerblue'}
        show_hill_function : bool
            Overlay fitted trans function if trans model fitted (all distributions, default: True).
            Works with all function types: additive_hill, single_hill, polynomial.
            Automatically detects function type from posterior_samples_trans.
        show_ntc_gradient : bool
            Color lines by NTC proportion in k-NN window (default: False).
            Lighter colors = more NTC cells, Darker colors = fewer NTC cells.
            Only applies to uncorrected plots.
            Fully implemented for: negbinom, binomial, normal, studentt.
            Not yet implemented for: multinomial (will issue warning).
        sum_factor_col : str
            Column name in model.meta for sum factors (default: 'sum_factor').
            Can be 'sum_factor', 'sum_factor_adj', or any other sum factor column.
            Only used for negbinom distribution (gene expression).
        xlabel : str
            X-axis label (default: "log2(x_true)")
        figsize : tuple, optional
            Figure size as ``(width, height)`` in inches. When None (default) the
            size is chosen automatically: 6 inches per column × 3 inches per row
            for multi-panel plots, or (8, 5) / (14, 5) for single-panel plots.
        src_barcodes : np.ndarray, optional
            Source barcode order if x_true not in model.meta order
        subset_meta : dict, optional
            Subset cells by metadata columns. Dictionary of {column: value} pairs.
            Example: {'target': 'ntc'} — plot only NTC cells.
            Example: {'cell_line': 'K562'} — plot only K562 cells.
            Multiple conditions are combined with AND logic.
        only_dependent : bool
            If True and plotting multiple features (gene name), filter to only "dependent" features
            where the Hill coefficient (n_a or n_b) credible interval excludes 0 (default: False).
            Requires fit_trans() to have been run with function_type='additive_hill'.
            Ignored for single-feature plots.
        ci_level : float
            Credible interval level for dependency filtering and parameter marker
            classification (default: 95.0).
            Used by only_dependent=True and mark_params=True.
        mark_params : bool or str
            Overlay meaningful parameter markers on the Hill function (default: False).
            Requires show_hill_function=True and fit_trans() completed with
            function_type='single_hill' or 'additive_hill'.

            - True or 'both': markers for both the fitted curve and the reference curve
            - 'fit': markers for the fitted curve only
            - 'reference': markers for the reference curve only (requires reference_df)
            - False or None: no markers

            Markers drawn depend on the fitted regime:

            - **Single Hill / effectively single**: log2(A), log2(A+α·Vmax), log2(EC50)
            - **Same-sign additive** (both Hills in same direction):
              log2(A), log2(A+α·Vmax_a+β·Vmax_b), log2(EC50_lower), log2(EC50_upper)
            - **Non-monotonic** (opposite-sign Hills):
              log2(y: x→0), log2(y: x→∞), log2(EC50_lower), log2(EC50_upper);
              additionally log2(A+α·Vmax_a+β·Vmax_b) [peak] if activation EC50 < inhibition EC50

            For non-negbinom distributions, all y-axis markers are in linear space.
            Note: asymptotes include the α/β weight factors (e.g. A+α·Vmax, not A+Vmax).
        log2fc : bool
            If True, plot log2FC relative to NTC instead of raw log2 counts
            (default: False). Only applies to negbinom modalities.
            x-axis: log2(x_true) - log2(mean NTC x_true)
            y-axis: log2(counts) - log2(mean NTC counts, reference group)
            A grey dotted crosshair is drawn at (0, 0) to mark the NTC reference.
        color_by : str or list of two str
            What to color smoothed lines by (default: ``'technical_group'``).

            - ``'technical_group'``: one line per cell-line / technical group (default)
            - ``'targeting'``: two lines — NTC vs Targeting
            - **Any column name in model.meta**: one line per unique value.
              A warning is issued if the column looks continuous (float dtype with many
              unique values) or has more than 30 unique values.
            - **List of two** (e.g. ``['technical_group', 'targeting']``): cross-product of
              two groupings — the first sets hue, the second sets shade. When the secondary
              is ``'targeting'``, NTC lines are drawn as a light tint of the group colour
              and targeting lines as the full colour (matching the classic "shades" style).
              Override any auto-generated colour via ``color_palette`` using the combined
              label ``"K562 / NTC"``, ``"K562 / Targeting"``, etc.

            Alpha-y technical correction is always applied per technical group regardless
            of this setting.
        facet_by : str or list of str, optional
            Column name(s) to facet by (default: None). Accepts either a single string or
            a list of **at most two** strings. Each value may be a real column name in
            model.meta **or** one of the same special keywords supported by ``color_by``:

            - ``'targeting'`` — two panels: NTC / Targeting
            - ``'technical_group'`` — one panel per technical group code

            **1 column** (``facet_by='cell_line'``) — all unique values become
            side-by-side column groups::

                rows = n_features,  cols = n_facets × n_corrections

            **2 columns** (``facet_by=['cell_line', 'perturbation']``) — first
            column maps to grid *rows*, second column maps to grid *columns*::

                rows = n_facet1_values × n_features
                cols = n_facet2_values × n_corrections

            ``subset_meta`` applies first; ``color_by`` works independently within
            each panel. Use ``legend_outside=True`` for many panels.
            Not yet supported for multinomial distributions (ignored with a warning).
        expand_x_to_params : bool
            When True, extend the x-axis (and the Hill curve) beyond the data range
            so that all K/EC50 parameters are visible on the plot (default: False).
            Requires ``show_hill_function=True`` and a fitted trans model.
            Ignored for polynomial function types (which have no K parameters).
        expand_y_to_params : bool
            When True, extend the y-axis to show the full range of the fitted Hill
            curve over the entire displayed x range (default: False). Normally the
            Hill curve's y values are only used for y-axis scaling within the data
            x range; this flag removes that restriction.
        legend_outside : bool
            Place the legend outside the panel to the right, shared across all panels
            (default: False). Useful when many lines clutter the plot area.
        filename : str, optional
            If provided, save the figure to ``model.output_dir / filename``.
            A ``.png`` extension is added automatically if none is given.
            Use a full path to save elsewhere.
        reference_df : pd.DataFrame, optional
            A trans_summary DataFrame with fitted Hill parameters (columns ending in
            ``_median``, e.g. ``A_median``, ``Vmax_a_median``). When provided, a
            reference Hill curve is overlaid in red dashed on each negbinom panel.
            Feature names are matched via a ``gene_name`` or ``gene`` column.
            Only applied to negbinom modalities.
        fdr_df : pd.DataFrame, optional
            A trans_summary DataFrame with ``fdr_alpha`` and ``fdr_beta`` columns
            (e.g. the output of ``save_trans_summary``). Used for greying out
            FDR-inactive parameter markers (``mark_params`` mode) and for classifying
            the Hill regime. Feature names are matched via ``gene_name`` or ``gene``.
        fdr_threshold : float
            FDR threshold for classifying components as active (default: 0.05).
            Components with ``fdr_alpha`` or ``fdr_beta`` >= this threshold are
            treated as inactive. Only used when ``fdr_df`` is provided.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        plt.Figure or plt.Axes
            Matplotlib figure or axes object.

        Raises
        ------
        ValueError
            If x_true not set (must run fit_cis first),
            if feature not found in modality,
            or if show_correction='corrected' but fit_technical not run.

        Warnings
        --------
        If fit_technical not run for modality and show_correction='corrected',
        warns and plots uncorrected only.

        Examples
        --------
        >>> # Plot single gene with Hill function
        >>> model.plot_xy_data('TET2', window=100, show_hill_function=True)
        >>>
        >>> # Plot specific splice junction with min_counts filter
        >>> model.plot_xy_data('chr1:12345:67890:+', modality_name='splicing_sj',
        ...                     min_counts=5)
        >>>
        >>> # Plot all splice junctions for a gene (creates multi-panel figure)
        >>> model.plot_xy_data('HES4', modality_name='splicing_sj')
        >>>
        >>> # Plot with custom colors
        >>> model.plot_xy_data('GFI1B', color_palette={'K562': 'red', 'TF1': 'blue'})
        >>>
        >>> # Show both corrected and uncorrected (default)
        >>> model.plot_xy_data('TET2', show_correction='both')
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
