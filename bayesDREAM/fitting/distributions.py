"""
Distribution-specific observation samplers for multi-modal bayesDREAM.

This module provides observation likelihoods for different distributions.
These are called by the main Pyro models (_model_y, _model_ntc) after
computing the dose-response function parameters.

Supported distributions:
- negbinom: Negative binomial (gene counts, transcript counts)
- multinomial: Categorical/proportional data (isoform usage, donor/acceptor usage)
- binomial: Binary outcomes with denominator (exon skipping PSI, raw SJ counts)
- normal: Continuous measurements
- studentt: Heavy-tailed, continuous measurements (SpliZ scores)

Technical group covariate handling:
- negbinom: Multiplicative effects on mu via alpha_y
- normal/studentt: Additive effects on mu
- binomial: Effects on logit scale
- multinomial: Not supported yet (complex - need to maintain probability simplex)
"""

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist


#######################################
# NEGATIVE BINOMIAL
#######################################

def sample_negbinom_trans(
    y_obs_tensor,
    mu_y,
    phi_y_used,
    alpha_y_full,
    groups_tensor,
    sum_factor_tensor,
    N,
    T,
    C=None,
    use_epsilon=True
):
    """
    Sample observations from negative binomial likelihood for trans model.

    Technical group effects: Multiplicative on mu (via alpha_y parameter).

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed counts, shape [N, T]
    mu_y : torch.Tensor
        Expected expression from f(x), shape [N, T] or [T]
    phi_y_used : torch.Tensor
        Overdispersion parameter, shape [1, T] or [T]
    alpha_y_full : torch.Tensor or None
        Technical group effects, shape [C, T] or [S, C, T] if provided
    groups_tensor : torch.Tensor or None
        Technical group codes, shape [N] if provided
    sum_factor_tensor : torch.Tensor
        Size factors, shape [N]
    N : int
        Number of guides/observations
    T : int
        Number of features (trans genes)
    C : int or None
        Number of technical groups
    use_epsilon : bool
        If True (default), add 1e-8 epsilon for numerical stability in logits.
        If False, use log(mu) - log(phi) directly (archive behavior).

    Notes
    -----
    mu_final = mu_y * alpha_y[group] * sum_factor
    Handles both 2D ([C, T]) and 3D ([S, C, T]) alpha tensors for Predictive sampling.
    """
    eps = 1e-8 if use_epsilon else 0.0
    # Apply technical group effects if present (multiplicative)
    if alpha_y_full is not None and groups_tensor is not None:
        if alpha_y_full.dim() == 2:
            # [C, T] -> index by group per observation -> [N, T]
            alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
            mu_adjusted = mu_y * alpha_y_used

            # Apply sum factors
            mu_final = mu_adjusted * sum_factor_tensor.unsqueeze(-1)  # [N, T]

            # Sample observations
            # NOTE: use_epsilon=True (default) adds 1e-8 for numerical stability
            # use_epsilon=False matches archive behavior (log(mu) - log(phi) directly)
            with pyro.plate("data_plate", N, dim=-2):
                pyro.sample(
                    "y_obs",
                    dist.NegativeBinomial(
                        total_count=phi_y_used,
                        logits=torch.log(mu_final + eps) - torch.log(phi_y_used + eps)
                    ),
                    obs=y_obs_tensor
                )

        elif alpha_y_full.dim() == 3:
            # [S, C, T] -> index groups along dim=1 -> [S, N, T]
            # CRITICAL: Use gather() to avoid creating huge intermediate tensors
            S = alpha_y_full.size(0)

            # Create index tensor for gather: [S, N, T] filled with group indices
            # groups_tensor: [N] with values in [0, C-1]
            groups_expanded = groups_tensor.view(1, N, 1).expand(S, N, T)  # [S, N, T]

            # Use gather on dim=1 (the C dimension) to select correct group per cell
            alpha_y_used = torch.gather(alpha_y_full, dim=1, index=groups_expanded)  # [S, N, T]

            # Broadcast mu_y to [S, N, T] if needed
            if mu_y.dim() == 1:  # [T]
                mu_y_expanded = mu_y.unsqueeze(0).unsqueeze(0).expand(S, N, -1)  # [S, N, T]
            elif mu_y.dim() == 2:  # [N, T]
                mu_y_expanded = mu_y.unsqueeze(0).expand(S, -1, -1)  # [S, N, T]
            else:
                mu_y_expanded = mu_y  # Assume already [S, N, T]

            mu_adjusted = mu_y_expanded * alpha_y_used  # [S, N, T]

            # Broadcast sum_factor_tensor to [S, N, 1]
            sum_factor_expanded = sum_factor_tensor.unsqueeze(0).unsqueeze(-1).expand(S, -1, -1)  # [S, N, 1]
            mu_final = mu_adjusted * sum_factor_expanded  # [S, N, T]

            # Broadcast phi_y_used to [S, 1, T]
            if phi_y_used.dim() == 1:  # [T]
                phi_expanded = phi_y_used.unsqueeze(0).unsqueeze(0).expand(S, 1, -1)
            elif phi_y_used.dim() == 2:  # [1, T]
                phi_expanded = phi_y_used.unsqueeze(0).expand(S, -1, -1)
            else:
                phi_expanded = phi_y_used

            # Expand y_obs_tensor to [S, N, T]
            y_obs_expanded = y_obs_tensor.unsqueeze(0).expand(S, -1, -1)

            # Sample observations with sample dimension
            # NOTE: use_epsilon=True (default) adds 1e-8 for numerical stability
            with pyro.plate("data_plate", S * N * T):
                pyro.sample(
                    "y_obs",
                    dist.NegativeBinomial(
                        total_count=phi_expanded,
                        logits=torch.log(mu_final + eps) - torch.log(phi_expanded + eps)
                    ),
                    obs=y_obs_expanded.reshape(-1)
                )
        else:
            raise ValueError(f"Unexpected alpha_y_full shape: {tuple(alpha_y_full.shape)}")
    else:
        # No technical group effects
        mu_adjusted = mu_y

        # Apply sum factors
        mu_final = mu_adjusted * sum_factor_tensor.unsqueeze(-1)  # [N, T]

        # Sample observations
        # NOTE: use_epsilon=True (default) adds 1e-8 for numerical stability
        with pyro.plate("data_plate", N, dim=-2):
            pyro.sample(
                "y_obs",
                dist.NegativeBinomial(
                    total_count=phi_y_used,
                    logits=torch.log(mu_final + eps) - torch.log(phi_y_used + eps)
                ),
                obs=y_obs_tensor
            )


#######################################
# MULTINOMIAL
#######################################

#######################################
# MULTINOMIAL (robust masked softmax)
#######################################

def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    logits: [..., K] real (unmasked positions can be any finite value)
    mask:   [..., K] bool, True = masked/disallowed category
    Returns probs with:
      - probs[..., k] = 0 where mask is True
      - rows with at least one unmasked get a proper softmax over unmasked
      - rows with all entries masked fall back to uniform (no NaNs)
    """
    # Set masked to -inf, unmasked unchanged
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask, neg_inf, logits)

    # Numerically stable shift by max over (potentially -inf) entries
    max_logits = masked_logits.max(dim=dim, keepdim=True).values
    # If row is all -inf, max is -inf; replace by 0 to avoid NaNs in exp
    max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))

    exps = torch.exp(masked_logits - max_logits)
    # Zero out masked entries explicitly (exp(-inf)=0 already; this is just defensive)
    exps = torch.where(mask, torch.zeros_like(exps), exps)

    sums = exps.sum(dim=dim, keepdim=True)
    # Normal case: divide by sum
    probs = torch.where(sums > 0, exps / sums, exps)  # if sums==0 we’ll fix below

    # Handle fully-masked rows (sums==0): return uniform over K (can’t infer anything)
    all_masked = (sums == 0)
    if all_masked.any():
        K = logits.size(dim)
        uniform = torch.full_like(probs, 1.0 / K)
        probs = torch.where(all_masked, uniform, probs)

    return probs


def sample_multinomial_trans(
    y_obs_tensor,         # [N, T, K] integer counts
    mu_y,                 # [T, K] (baseline probs) or [N, T, K]; must be 0 at masked K
    alpha_y_full,         # [C, T, K] or [S, C, T, K] additive logits
    groups_tensor,        # [N] int in [0..C-1]
    N, T, K, C=None
):
    # Basic sanity checks (fail fast instead of NaNs downstream)
    if not torch.isfinite(y_obs_tensor).all():
        raise ValueError("y_obs_tensor contains non-finite values")
    if (y_obs_tensor < 0).any():
        raise ValueError("y_obs_tensor contains negative counts")

    # Totals per (obs, feature)
    total_counts = y_obs_tensor.sum(dim=-1)  # [N, T]

    # Ensure mu_y has shape [N, T, K]
    if mu_y.dim() == 2:  # [T, K] -> [N, T, K]
        mu_y = mu_y.unsqueeze(0).expand(N, -1, -1)

    # Zero-category mask from mu_y (assumed exact zeros at masked K)
    zero_mask = (mu_y == 0)  # [N, T, K] (rows identical if mu_y was [T,K])

    # Base logits: log(mu_y) where unmasked; will never be taken at masked due to masked softmax
    safe_log_mu = torch.where(zero_mask, torch.zeros_like(mu_y), mu_y).log()  # avoid log(0)

    def _probs_from_logits(base_log_mu, alpha):
        logits = base_log_mu + alpha
        # Use masked softmax so rows with all masked don't produce NaN
        return _masked_softmax(logits, mask=zero_mask, dim=-1)

    # Add α and compute probabilities
    if alpha_y_full is not None and groups_tensor is not None:
        if alpha_y_full.dim() == 3:
            # [C, T, K] -> index by group per observation -> [N, T, K]
            alpha_used = alpha_y_full[groups_tensor, :, :]
            probs = _probs_from_logits(safe_log_mu, alpha_used)        # [N, T, K]
        elif alpha_y_full.dim() == 4:
            # [S, C, T, K] -> index groups along dim=1 -> [S, N, T, K]
            # CRITICAL: Use gather() to avoid creating huge intermediate tensors
            S = alpha_y_full.size(0)

            # Create index tensor for gather: [S, N, T, K] filled with group indices
            # groups_tensor: [N] with values in [0, C-1]
            groups_expanded = groups_tensor.view(1, N, 1, 1).expand(S, N, T, K)  # [S, N, T, K]

            # Use gather on dim=1 (the C dimension) to select correct group per cell
            alpha_used = torch.gather(alpha_y_full, dim=1, index=groups_expanded)  # [S, N, T, K]

            base_logits_S = safe_log_mu.unsqueeze(0).expand(S, -1, -1, -1)
            probs = _probs_from_logits(base_logits_S, alpha_used)      # [S, N, T, K]
        else:
            raise ValueError(f"Unexpected alpha_y_full shape: {tuple(alpha_y_full.shape)}")
    else:
        # No α: just take masked softmax of baseline logits (respects zeros)
        probs = _masked_softmax(safe_log_mu, mask=zero_mask, dim=-1)   # [N, T, K] or [S, N, T, K]

    # Log-likelihood (stable)
    if probs.dim() == 3:
        log_probs = torch.log(probs.clamp_min(1e-12))
        ll = (
            torch.lgamma(total_counts + 1.0)
            - torch.lgamma(y_obs_tensor + 1.0).sum(dim=-1)
            + (y_obs_tensor * log_probs).sum(dim=-1)
        )  # [N, T]
        with pyro.plate("obs_plate", N * T):
            pyro.factor("y_obs", ll.reshape(-1))
    elif probs.dim() == 4:
        S = probs.size(0)
        y = y_obs_tensor.unsqueeze(0).expand(S, -1, -1, -1)            # [S, N, T, K]
        total_counts_S = total_counts.unsqueeze(0).expand(S, -1, -1)   # [S, N, T]
        log_probs = torch.log(probs.clamp_min(1e-12))
        ll = (
            torch.lgamma(total_counts_S + 1.0)
            - torch.lgamma(y + 1.0).sum(dim=-1)
            + (y * log_probs).sum(dim=-1)
        )  # [S, N, T]
        with pyro.plate("obs_plate", S * N * T):
            pyro.factor("y_obs", ll.reshape(-1))
    else:
        raise ValueError(f"Unexpected probs shape: {tuple(probs.shape)}")


#######################################
# BINOMIAL
#######################################

def sample_binomial_trans(
    y_obs_tensor,
    denominator_tensor,
    mu_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    C=None
):
    """
    Sample observations from binomial likelihood for trans model.

    Technical group effects: Applied on LOGIT scale to maintain p ∈ [0, 1].

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed success counts, shape [N, T]
    denominator_tensor : torch.Tensor
        Total counts (denominator), shape [N, T]
        Observations with denominator=0 are masked (excluded from likelihood)
    mu_y : torch.Tensor
        Expected probability from f(x), shape [N, T]
        Should be in [0, 1] but we'll apply sigmoid anyway
    alpha_y_full : torch.Tensor or None
        Technical group effects (additive on logit scale), shape [C, T] if provided
    groups_tensor : torch.Tensor or None
        Technical group codes, shape [N] if provided
    N : int
        Number of guides/observations
    T : int
        Number of features
    C : int or None
        Number of technical groups

    Notes
    -----
    logit(p) = logit(mu_y) + alpha_y[group]
    p = sigmoid(logit(p))

    Observations where denominator_tensor=0 are masked and contribute zero to the log-likelihood.
    """
    # Create mask for valid observations (denominator > 0)
    valid_mask = denominator_tensor > 0

    # If no valid observations, return early
    if not valid_mask.any():
        print("[WARNING] No valid observations (all denominators are 0)")
        return

    # Convert mu_y to logit scale
    mu_y_clamped = torch.clamp(mu_y, min=1e-6, max=1-1e-6)
    logit_mu = torch.log(mu_y_clamped) - torch.log(1 - mu_y_clamped)

    # Apply technical group effects on logit scale (additive)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        logit_final = logit_mu + alpha_y_used
    else:
        logit_final = logit_mu

    # Numerically stable binomial log-probability via binary_cross_entropy_with_logits.
    #
    # The naive sigmoid → clamp(p, 1e-12, 1-1e-12) → log(1-p) approach fails in float32
    # because 1 - 1e-12 = 1.0 exactly (float32 epsilon ≈ 1.2e-7), so the upper clamp
    # provides no protection. sigmoid(logit) = 1.0 for logit > ~16, making log(0) = -inf
    # and producing NaN gradients via the chain rule (0 * inf = nan from d(clamp)/d(p)=0).
    #
    # F.binary_cross_entropy_with_logits uses the numerically stable identity:
    #   BCE(logit, t) = max(logit, 0) - logit*t + log(1 + exp(-|logit|))
    # which is safe for any finite logit.
    #
    # Binomial log P(y|n,p) = y*log(p) + (n-y)*log(1-p) = n * [−BCE(logit, y/n)]
    y_safe = torch.where(valid_mask, y_obs_tensor.float(), torch.zeros_like(y_obs_tensor.float()))
    n_safe = torch.where(valid_mask, denominator_tensor.float(), torch.ones_like(denominator_tensor.float()))
    target = y_safe / n_safe  # fraction of successes in [0, 1]; n_safe >= 1 always

    log_prob = n_safe * (-F.binary_cross_entropy_with_logits(logit_final, target, reduction='none'))

    # Zero out masked entries
    log_prob = torch.where(valid_mask, log_prob, torch.zeros_like(log_prob))

    # Attach as factor (summed over all observations)
    with pyro.plate("obs_plate", N * T):
        pyro.factor("y_obs", log_prob.reshape(-1))


#######################################
# NORMAL
#######################################

def sample_normal_trans(
    y_obs_tensor,
    mu_y,
    sigma_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    C=None
):
    """
    Normal likelihood with *masked* observations.

    - NaNs / inf in y_obs_tensor are treated as missing
      and contribute 0 to the log-likelihood (factor of 1 in prob).
    - Technical group effects are additive on the mean only; sigma_y is unchanged.
    """

    # Apply technical group effects if present (additive on the mean)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_adjusted = mu_y + alpha_y_used              # [N, T]
    else:
        mu_adjusted = mu_y                             # [N, T]

    # Broadcast sigma_y to [N, T]
    if sigma_y.dim() == 1:
        # [T] -> [N, T]
        sigma_b = sigma_y.unsqueeze(0).expand(N, -1)
    else:
        # Assume already broadcastable to [N, T]
        sigma_b = sigma_y

    # Mask of *valid* entries: finite = observed, non-finite = missing
    mask = torch.isfinite(y_obs_tensor)  # [N, T]
    if not mask.any():
        # Everything missing -> no likelihood contribution
        return

    # Replace missing y's by mu_adjusted so log_prob is defined;
    # they will get zeroed out later via the mask.
    y_clean = torch.where(mask, y_obs_tensor, mu_adjusted)

    # Log-prob under Normal
    dist_normal = dist.Normal(mu_adjusted, sigma_b)
    log_prob = dist_normal.log_prob(y_clean)          # [N, T]

    # Zero-out missing entries so they don't affect the joint density
    log_prob = torch.where(mask, log_prob,
                           torch.zeros_like(log_prob))

    # Flatten and attach as a factor
    with pyro.plate("obs_plate", N * T):
        pyro.factor("y_obs", log_prob.reshape(-1))

#######################################
# STUDENT-T
#######################################

def sample_studentt_trans(
    y_obs_tensor,
    mu_y,
    sigma_y,
    nu_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    C=None,
):
    """
    Student-T likelihood with *masked* observations.

    - NaNs / inf in y_obs_tensor are treated as missing and contribute 0
      to the log-likelihood.
    - Technical group effects are additive on the mean only (same as Normal).
    - df (nu_y) can be a scalar or tensor; we just pass it through.
    """

    # Apply technical group effects if present (additive on the mean)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_adjusted = mu_y + alpha_y_used              # [N, T]
    else:
        mu_adjusted = mu_y                             # [N, T]

    # Broadcast sigma_y to [N, T]
    if sigma_y.dim() == 1:
        sigma_b = sigma_y.unsqueeze(0).expand(N, -1)   # [N, T]
    else:
        sigma_b = sigma_y

    # Mask of *valid* entries
    mask = torch.isfinite(y_obs_tensor)  # [N, T]
    if not mask.any():
        return

    # Replace missing y's so log_prob is defined; then mask them out
    y_clean = torch.where(mask, y_obs_tensor, mu_adjusted)

    # Student-T log-prob
    dist_t = dist.StudentT(df=nu_y, loc=mu_adjusted, scale=sigma_b)
    log_prob = dist_t.log_prob(y_clean)               # [N, T]

    # Zero-out missing entries
    log_prob = torch.where(mask, log_prob,
                           torch.zeros_like(log_prob))

    with pyro.plate("obs_plate", N * T):
        pyro.factor("y_obs", log_prob.reshape(-1))


#######################################
# DISTRIBUTION REGISTRY
#######################################

DISTRIBUTION_REGISTRY = {
    'negbinom': {
        'trans': sample_negbinom_trans,
        'requires_denominator': False,
        'requires_sum_factor': True,
        'is_3d': False,
        'supports_technical_group': True,
        'technical_group_type': 'multiplicative'
    },
    'multinomial': {
        'trans': sample_multinomial_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': True,
        'supports_technical_group': True,
        'technical_group_type': 'logit'
    },
    'binomial': {
        'trans': sample_binomial_trans,
        'requires_denominator': True,
        'requires_sum_factor': False,
        'is_3d': False,
        'supports_technical_group': True,
        'technical_group_type': 'logit'
    },
    'normal': {
        'trans': sample_normal_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': False,
        'supports_technical_group': True,
        'technical_group_type': 'additive'
    },
    'studentt': {
        'trans': sample_studentt_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': False,
        'supports_technical_group': True,
        'technical_group_type': 'additive',
    },
}


def get_observation_sampler(distribution, model_type='trans'):
    """
    Get the appropriate observation sampler for a distribution.

    Parameters
    ----------
    distribution : str
        Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'studentt'
    model_type : str
        Model type: currently only 'trans' is fully supported

    Returns
    -------
    callable
        Observation sampler function

    Raises
    ------
    ValueError
        If distribution is not recognized
    """
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Must be one of: {list(DISTRIBUTION_REGISTRY.keys())}"
        )

    if model_type != 'trans':
        raise NotImplementedError(
            f"Model type '{model_type}' not yet implemented. "
            f"Currently only 'trans' model is supported."
        )

    return DISTRIBUTION_REGISTRY[distribution][model_type]


def requires_denominator(distribution):
    """Check if a distribution requires a denominator array (e.g., binomial)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['requires_denominator']


def requires_sum_factor(distribution):
    """Check if a distribution requires sum factors (only negbinom)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['requires_sum_factor']


def is_3d_distribution(distribution):
    """Check if a distribution uses 3D data structure (multinomial)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['is_3d']


def supports_technical_group_effects(distribution):
    """Check if technical group covariate effects are supported for this distribution."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['supports_technical_group']


def get_technical_group_effect_type(distribution):
    """
    Get the type of technical group effects for this distribution.

    Returns
    -------
    str or None
        'multiplicative', 'additive', 'logit', or None if not supported
    """
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['technical_group_type']


# Backward compatibility aliases
supports_cell_line_effects = supports_technical_group_effects
get_cell_line_effect_type = get_technical_group_effect_type
