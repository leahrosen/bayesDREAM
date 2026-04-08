"""
Prior sampling utilities for bayesDREAM plotting.

This module implements prior sampling from the Pyro model distributions
to enable true prior/posterior comparison plots.
"""

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist


def _compute_data_driven_priors(model, modality, distribution, epsilon=1e-6):
    """
    Compute data-driven prior parameters (mu_x_mean_tensor, mu_x_sd_tensor) from NTC data.

    This replicates the logic from bayesDREAM/fitting/technical.py:469-494.

    Parameters
    ----------
    model : bayesDREAM
        The model instance
    modality : Modality
        The modality to compute priors for
    distribution : str
        Distribution type
    epsilon : float
        Small value to ensure positivity

    Returns
    -------
    mu_x_mean_tensor : torch.Tensor
        Feature-specific prior means [T] or [T, D]
    mu_x_sd_tensor : torch.Tensor
        Feature-specific prior standard deviations [T] or [T, D]
    """
    # Always use modality.counts for consistency
    # (model.counts might be subset to cis gene only, but modality has all features)
    counts = modality.counts
    cell_names = modality.cell_names if modality.cell_names is not None else \
                 model.meta['cell'].values[:counts.shape[modality.cells_axis]]

    # Subset to NTC cells
    cell_set = set(cell_names)
    meta_subset = model.meta[model.meta['cell'].isin(cell_set)].copy()
    meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()

    ntc_cell_list = meta_ntc["cell"].tolist()
    ntc_indices = [i for i, c in enumerate(cell_names) if c in ntc_cell_list]

    if len(ntc_indices) == 0:
        raise ValueError("No NTC cells found for computing data-driven priors")

    # Extract NTC counts
    if counts.ndim == 2:
        counts_ntc = counts[:, ntc_indices] if modality.cells_axis == 1 else counts[ntc_indices, :]
    elif counts.ndim == 3:
        counts_ntc = counts[:, ntc_indices, :]
    else:
        raise ValueError(f"Unexpected dimensions: {counts.ndim}")

    # Get technical groups and guides for NTC cells
    groups_ntc = meta_ntc['technical_group_code'].values
    guides_ntc = meta_ntc['guide_code'].values

    # Prepare data for prior computation (handle 3D by summing/averaging last axis)
    if counts_ntc.ndim == 3:
        if distribution == 'multinomial':
            y_obs_ntc_for_priors = counts_ntc.sum(axis=2)  # [T, N] or [N, T]
        else:
            y_obs_ntc_for_priors = counts_ntc.sum(axis=2)
    else:
        y_obs_ntc_for_priors = counts_ntc

    # Transpose if needed to get [N, T]
    if modality.cells_axis == 1:
        y_obs_ntc_for_priors = y_obs_ntc_for_priors.T

    # Apply size factors for negbinom
    if distribution == 'negbinom' and 'sum_factor' in meta_ntc.columns:
        y_obs_ntc_factored = y_obs_ntc_for_priors / meta_ntc['sum_factor'].values.reshape(-1, 1)
    else:
        y_obs_ntc_factored = y_obs_ntc_for_priors

    # Compute mu_x_mean and mu_x_sd based on distribution
    if distribution == 'negbinom':
        baseline_mask = (groups_ntc == 0)
        mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :], axis=0)
        guide_means = np.array([np.mean(y_obs_ntc_factored[guides_ntc == g], axis=0)
                               for g in np.unique(guides_ntc)])
        mu_x_sd = np.std(guide_means, axis=0) + epsilon
        mu_x_mean = mu_x_mean + epsilon  # strictly positive for Gamma

    elif distribution == 'normal':
        baseline_mask = (groups_ntc == 0)
        mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :], axis=0)
        guide_means = np.array([np.mean(y_obs_ntc_factored[guides_ntc == g], axis=0)
                               for g in np.unique(guides_ntc)])
        mu_x_sd = np.std(guide_means, axis=0) + epsilon

    else:
        # binomial & multinomial: not used
        T = y_obs_ntc_for_priors.shape[1]
        mu_x_mean = np.zeros(T, dtype=float)
        mu_x_sd = np.ones(T, dtype=float)

    # Convert to tensors
    mu_x_mean_tensor = torch.tensor(mu_x_mean, dtype=torch.float32)
    mu_x_sd_tensor = torch.tensor(mu_x_sd, dtype=torch.float32)

    return mu_x_mean_tensor, mu_x_sd_tensor


def sample_technical_priors(
    model,
    modality_name=None,
    nsamples=1000,
    distribution=None
):
    """
    Sample from the prior distributions defined in _model_technical.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    modality_name : str, optional
        Modality name (default: primary modality)
    nsamples : int
        Number of prior samples to draw
    distribution : str, optional
        Distribution type (default: from modality)

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_technical structure
    """
    if modality_name is None:
        modality_name = model.primary_modality

    modality = model.get_modality(modality_name)

    if distribution is None:
        distribution = modality.distribution

    # Get model dimensions - always use modality.counts for consistency
    # (model.counts might be subset to cis gene only, but modality has all features)
    if modality.counts.ndim == 2:
        T = modality.counts.shape[0 if modality.cells_axis == 1 else 1]
    else:
        T = modality.counts.shape[0]

    C = model.meta['technical_group_code'].nunique()

    # Sample priors based on distribution
    prior_samples = {}

    # Cell-line effects: log2_alpha_y ~ StudentT(df=3, loc=0, scale=20.0)
    # This matches the actual prior used in fitting (bayesDREAM/fitting/technical.py:77)
    # Wide prior allows data to constrain the posterior appropriately
    log2_alpha_y = dist.StudentT(df=3, loc=0.0, scale=20.0).sample((nsamples, C - 1, T))
    prior_samples['log2_alpha_y'] = log2_alpha_y
    prior_samples['alpha_y_mul'] = 2.0 ** log2_alpha_y  # multiplicative
    prior_samples['delta_y_add'] = log2_alpha_y  # additive/logit

    # Add baseline row (first technical group is always 1.0 or 0.0)
    alpha_full_mul = torch.cat([torch.ones(nsamples, 1, T), prior_samples['alpha_y_mul']], dim=1)
    alpha_full_add = torch.cat([torch.zeros(nsamples, 1, T), prior_samples['delta_y_add']], dim=1)
    prior_samples['alpha_y'] = alpha_full_mul  # back-compat
    prior_samples['alpha_y_mult'] = alpha_full_mul
    prior_samples['alpha_y_add'] = alpha_full_add

    # Compute data-driven prior parameters from NTC data
    mu_x_mean_tensor, mu_x_sd_tensor = _compute_data_driven_priors(model, modality, distribution)

    # Ensure T matches the actual computed prior parameters
    # (handles edge cases where modality might have been modified after fitting)
    if mu_x_mean_tensor.numel() > 0 and mu_x_mean_tensor.numel() != T:
        print(f"[WARNING] Modality has {T} features but computed priors have {mu_x_mean_tensor.numel()} features. "
              f"Using actual prior size ({mu_x_mean_tensor.numel()}).")
        T = mu_x_mean_tensor.numel()

    # Distribution-specific parameters
    if distribution == 'negbinom':
        # Default prior parameters from fit_technical
        beta_o_alpha = 9.0
        beta_o_beta = 3.0

        # beta_o ~ Gamma(alpha, beta)
        beta_o = dist.Gamma(beta_o_alpha, beta_o_beta).sample((nsamples,))
        prior_samples['beta_o'] = beta_o

        # o_y ~ Exponential(beta_o) for each feature
        o_y = dist.Exponential(beta_o.unsqueeze(-1)).sample((T,)).squeeze()  # (nsamples, T)
        prior_samples['o_y'] = o_y

        # mu_ntc ~ Gamma(alpha, beta) with data-driven parameters
        # Shape-rate parameterization: alpha = mu^2/sigma^2, beta = mu/sigma^2
        alpha_param = (mu_x_mean_tensor ** 2) / (mu_x_sd_tensor ** 2)
        beta_param = mu_x_mean_tensor / (mu_x_sd_tensor ** 2)

        # Sample for each feature
        mu_ntc_list = []
        for i in range(T):
            mu_ntc_feat = dist.Gamma(alpha_param[i], beta_param[i]).sample((nsamples,))
            mu_ntc_list.append(mu_ntc_feat)
        mu_ntc = torch.stack(mu_ntc_list, dim=1)  # (nsamples, T)
        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'normal':
        # sigma_y ~ HalfCauchy(10.0)
        sigma_y = dist.HalfCauchy(10.0).sample((nsamples, T))
        prior_samples['sigma_y'] = sigma_y

        # mu_ntc ~ Normal(mu_x_mean, mu_x_sd) with data-driven parameters
        mu_ntc = dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor).sample((nsamples,))
        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'binomial':
        # mu_ntc ~ Beta(a, b) with empirical Bayes parameters from NTC data
        # This matches the logic in technical.py:156-116

        # Get denominator if available
        if modality.denominator is not None:
            # Get cell names from modality for consistency
            cell_names = modality.cell_names if modality.cell_names is not None else \
                         model.meta['cell'].values[:modality.counts.shape[modality.cells_axis]]

            meta_subset = model.meta[model.meta['cell'].isin(set(cell_names))].copy()
            meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()
            ntc_indices = [i for i, c in enumerate(cell_names) if c in meta_ntc["cell"].tolist()]

            # Get NTC counts and denominators
            counts = modality.counts
            denominator = modality.denominator

            if counts.ndim == 2:
                counts_ntc = counts[:, ntc_indices] if modality.cells_axis == 1 else counts[ntc_indices, :]
                denom_ntc = denominator[:, ntc_indices] if modality.cells_axis == 1 else denominator[ntc_indices, :]
            else:
                counts_ntc = counts[:, ntc_indices, :]
                denom_ntc = denominator[:, ntc_indices, :]

            # Sum across cells to get empirical estimates
            if modality.cells_axis == 1:
                y_sum = counts_ntc.sum(axis=1)  # [T]
                den_sum = denom_ntc.sum(axis=1)  # [T]
            else:
                y_sum = counts_ntc.sum(axis=0)  # [T]
                den_sum = denom_ntc.sum(axis=0)  # [T]

            # Compute Beta parameters (matching technical.py)
            p_hat = (y_sum + 0.5) / (den_sum + 1.0)
            p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
            kappa = np.clip(den_sum, 20.0, 200.0)
            a = p_hat * kappa + 1e-3
            b = (1.0 - p_hat) * kappa + 1e-3

            # Sample for each feature
            mu_ntc_list = []
            for i in range(T):
                mu_ntc_feat = dist.Beta(torch.tensor(a[i]), torch.tensor(b[i])).sample((nsamples,))
                mu_ntc_list.append(mu_ntc_feat)
            mu_ntc = torch.stack(mu_ntc_list, dim=1)  # (nsamples, T)
        else:
            # Fallback to uniform if no denominator
            mu_ntc = dist.Beta(1.0, 1.0).sample((nsamples, T))

        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'multinomial':
        # Get K from modality
        K = modality.counts.shape[2] if modality.counts.ndim == 3 else None
        if K is None:
            raise ValueError("Cannot determine K for multinomial distribution")

        # probs_baseline ~ Dirichlet(concentration) with empirical concentration from NTC data
        # This matches technical.py:118-123

        # Get cell names from modality for consistency
        cell_names = modality.cell_names if modality.cell_names is not None else \
                     model.meta['cell'].values[:modality.counts.shape[modality.cells_axis]]

        meta_subset = model.meta[model.meta['cell'].isin(set(cell_names))].copy()
        meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()
        ntc_indices = [i for i, c in enumerate(cell_names) if c in meta_ntc["cell"].tolist()]

        # Get NTC counts (3D: T, N, K or N, T, K)
        counts_ntc = modality.counts[:, ntc_indices, :]  # Assuming cells_axis=1 for 3D

        # Sum across cells: [T, K]
        total_counts_per_feature = counts_ntc.sum(axis=1)  # [T, K]
        concentration = torch.tensor(total_counts_per_feature + 1.0, dtype=torch.float32)

        # Sample for each feature
        probs_list = []
        for i in range(T):
            probs_feat = dist.Dirichlet(concentration[i]).sample((nsamples,))
            probs_list.append(probs_feat)
        probs_baseline = torch.stack(probs_list, dim=1)  # (nsamples, T, K)
        prior_samples['probs_baseline'] = probs_baseline

    return prior_samples


def sample_cis_priors(
    model,
    nsamples=1000
):
    """
    Sample from the prior distributions defined in _model_x (cis model).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    nsamples : int
        Number of prior samples to draw

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_cis structure
    """
    # Get dimensions
    N = model.meta.shape[0]  # number of cells
    n_guides = model.meta['guide_code'].nunique()

    prior_samples = {}

    # x_true ~ LogNormal(mu, sigma) per guide
    # mu ~ Normal(data-driven mean, sd)
    # sigma ~ HalfNormal(1.0) or similar

    # For now, sample from broad priors
    # In practice, these would be data-driven like in fit_cis
    mu = dist.Normal(0.0, 2.0).sample((nsamples, n_guides))
    sigma = dist.HalfNormal(1.0).sample((nsamples, n_guides))

    # x_true ~ LogNormal(mu[guide], sigma[guide])
    # Expand to all cells based on guide assignment
    guide_codes = torch.tensor(model.meta['guide_code'].values, dtype=torch.long)

    x_true_samples = []
    for s in range(nsamples):
        x_true_per_cell = []
        for i, g in enumerate(guide_codes):
            x_true_per_cell.append(
                dist.LogNormal(mu[s, g], sigma[s, g]).sample()
            )
        x_true_samples.append(torch.stack(x_true_per_cell))

    prior_samples['x_true'] = torch.stack(x_true_samples)  # (nsamples, N)

    # alpha_x_prefit is always [C] (1D point estimate, reference group at index 0)
    if hasattr(model, 'alpha_x_prefit') and model.alpha_x_prefit is not None:
        C = model.alpha_x_prefit.shape[0]
        log2_alpha_x = dist.StudentT(df=3, loc=0.0, scale=20.0).sample((nsamples, C - 1, 1))
        alpha_x_mul = 2.0 ** log2_alpha_x
        alpha_full_mul = torch.cat([torch.ones(nsamples, 1, 1), alpha_x_mul], dim=1)
        prior_samples['alpha_x'] = alpha_full_mul.squeeze(-1)  # (nsamples, C)

    # beta_o, o_x for negbinom (same as technical model)
    beta_o_alpha = 9.0
    beta_o_beta = 3.0
    beta_o = dist.Gamma(beta_o_alpha, beta_o_beta).sample((nsamples,))
    prior_samples['beta_o'] = beta_o

    o_x = dist.Exponential(beta_o).sample()
    prior_samples['o_x'] = o_x

    # mu, sigma for per-guide variation
    prior_samples['mu'] = mu
    prior_samples['sigma'] = sigma

    return prior_samples


def sample_trans_priors(
    model,
    modality_name=None,
    function_type='additive_hill',
    nsamples=1000,
    distribution=None
):
    """
    Sample from the prior distributions defined in _model_y (trans model).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    modality_name : str, optional
        Modality name (default: primary modality)
    function_type : str
        Function type used in trans fit ('additive_hill', 'single_hill', 'polynomial')
    nsamples : int
        Number of prior samples to draw
    distribution : str, optional
        Distribution type (default: from modality)

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_trans structure
    """
    if modality_name is None:
        modality_name = model.primary_modality

    modality = model.get_modality(modality_name)

    if distribution is None:
        distribution = modality.distribution

    # Get dimensions
    T = len(model.trans_genes)

    prior_samples = {}

    # Baseline A ~ Normal or LogNormal depending on distribution
    if distribution == 'negbinom':
        A = dist.Gamma(2.0, 0.1).sample((nsamples, T))
    elif distribution == 'normal':
        A = dist.Normal(0.0, 10.0).sample((nsamples, T))
    elif distribution == 'binomial':
        A = dist.Beta(1.0, 1.0).sample((nsamples, T))
    else:
        A = dist.Normal(0.0, 10.0).sample((nsamples, T))

    prior_samples['A'] = A

    # Function-specific parameters
    if function_type == 'additive_hill':
        # Gamma priors for sparsity on effect sizes
        lambda_a = dist.Gamma(1.0, 0.1).sample((nsamples,))
        lambda_b = dist.Gamma(1.0, 0.1).sample((nsamples,))
        prior_samples['lambda_a'] = lambda_a
        prior_samples['lambda_b'] = lambda_b

        # Effect sizes with sparsity priors
        alpha = dist.Gamma(1.0, lambda_a.unsqueeze(-1)).sample((T,)).squeeze()
        beta = dist.Gamma(1.0, lambda_b.unsqueeze(-1)).sample((T,)).squeeze()
        prior_samples['alpha'] = alpha
        prior_samples['beta'] = beta

        # Hill parameters
        Vmax_a = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        Vmax_b = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K_a = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K_b = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        n_a = dist.LogNormal(0.0, 1.0).sample((nsamples, T))
        n_b = dist.LogNormal(0.0, 1.0).sample((nsamples, T))

        prior_samples['Vmax_a'] = Vmax_a
        prior_samples['Vmax_b'] = Vmax_b
        prior_samples['K_a'] = K_a
        prior_samples['K_b'] = K_b
        prior_samples['n_a'] = n_a
        prior_samples['n_b'] = n_b

        # Theta is composite
        theta = torch.stack([A, alpha, beta, Vmax_a, Vmax_b, K_a, K_b, n_a, n_b], dim=-1)
        prior_samples['theta'] = theta

    elif function_type == 'single_hill':
        # Single Hill function priors
        Vmax = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        n = dist.LogNormal(0.0, 1.0).sample((nsamples, T))

        prior_samples['Vmax'] = Vmax
        prior_samples['K'] = K
        prior_samples['n'] = n

        theta = torch.stack([A, Vmax, K, n], dim=-1)
        prior_samples['theta'] = theta

    elif function_type == 'polynomial':
        # Polynomial coefficients
        degree = 6  # default
        coef_names = [f'coef_{i}' for i in range(degree + 1)]

        coefs = []
        for i in range(degree + 1):
            # Decreasing variance for higher order terms
            scale = 10.0 / (i + 1)
            coef = dist.Normal(0.0, scale).sample((nsamples, T))
            prior_samples[f'coef_{i}'] = coef
            coefs.append(coef)

        theta = torch.stack(coefs, dim=-1)
        prior_samples['theta'] = theta

    return prior_samples


def get_prior_samples(
    model,
    fit_type='technical',
    modality_name=None,
    nsamples=1000,
    **kwargs
):
    """
    Unified interface for sampling priors from any fit type.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    fit_type : str
        Type of fit: 'technical', 'cis', or 'trans'
    modality_name : str, optional
        Modality name (default: primary modality)
    nsamples : int
        Number of prior samples to draw
    **kwargs
        Additional parameters passed to specific sampling functions

    Returns
    -------
    dict
        Dictionary of prior samples
    """
    if fit_type == 'technical':
        return sample_technical_priors(model, modality_name, nsamples, **kwargs)
    elif fit_type == 'cis':
        return sample_cis_priors(model, nsamples, **kwargs)
    elif fit_type == 'trans':
        return sample_trans_priors(model, modality_name, nsamples=nsamples, **kwargs)
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")
