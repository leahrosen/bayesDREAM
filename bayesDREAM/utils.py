"""
Utility functions for bayesDREAM.

This module contains helper functions used throughout the bayesDREAM package:
- Thread configuration
- Numerical solvers
- Dose-response functions (Hill, polynomial)
- Tensor utilities
"""

import os
import numpy as np
import torch
from scipy.special import betainc
from scipy.optimize import brentq


########################################
# Thread Configuration
########################################

def set_max_threads(cores: int):
    """
    Set the number of threads for Pyro and backend computations.

    Parameters
    ----------
    cores : int
        Number of CPU cores to use
    """
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)


########################################
# Numerical Solvers
########################################

def find_beta(alpha: float, gamma_threshold: float, p0: float, epsilon: float = 1e-6) -> float:
    """
    Numerically solve for beta given Betainc(alpha, beta, gamma_threshold) = p0.

    Parameters
    ----------
    alpha : float
        Alpha parameter for incomplete beta function
    gamma_threshold : float
        Threshold value
    p0 : float
        Target probability
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    float
        Solved beta value
    """
    def equation(bet):
        return betainc(alpha, bet, gamma_threshold) - p0
    beta_lower = epsilon
    beta_upper = 1
    return brentq(equation, beta_lower, beta_upper)


def calculate_mu_x_guide(guide, x_obs_ntc_factored, guides_ntc):
    """
    Helper function to calculate mean per guide for mu_x_sd.

    Parameters
    ----------
    guide : str or int
        Guide identifier
    x_obs_ntc_factored : torch.Tensor
        Factored observations for NTC cells
    guides_ntc : torch.Tensor
        Guide assignments for NTC cells

    Returns
    -------
    torch.Tensor
        Mean value for the specified guide
    """
    mask = guides_ntc == guide
    return torch.mean(x_obs_ntc_factored[mask])


def update_convergence_state(loss, smoothed_loss, alpha_ewma, tolerance):
    """
    Update EWMA-smoothed loss and decide whether SVI should stop early.

    Returns
    -------
    tuple[bool, float]
        A tuple of (has_converged, updated_smoothed_loss).
    """
    if smoothed_loss is None:
        return False, loss

    has_converged = abs(alpha_ewma * (loss - smoothed_loss)) < tolerance
    updated_smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss
    return has_converged, updated_smoothed_loss


########################################
# Dose-Response Functions
########################################

def default_sigmoid_clamp(dtype):
    # log(max_float) is ~88.7 for float32, ~709.8 for float64
    log_max = torch.log(torch.tensor(torch.finfo(dtype).max))
    return float(0.75 * log_max)  # 0.75 is conservative

def Hill_based_positive_logK(x, Vmax, A, logK, n):
    tiny = torch.finfo(x.dtype).tiny
    x_safe = x.clamp_min(tiny)
    logit = n * (torch.log(x_safe) - logK)  # logK broadcasts across cells
    clamp_logit = default_sigmoid_clamp(logit.dtype)
    # Soft (tanh) squeeze instead of hard clamp: gradient is sech²(logit/c) ≥ sech²(1) ≈ 0.42
    # at the "boundary" |logit| = clamp_logit, never zero — fixes dead gradient for extreme n.
    # Behaviour is identical to hard clamp for |logit| << clamp_logit (typical operation).
    logit = clamp_logit * torch.tanh(logit / clamp_logit)
    return Vmax * torch.sigmoid(logit) + A


def Hill_based_positive(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Positive Hill equation: Vmax * (x^n / (K^n + x^n)) + A

    Used for modeling activation dose-response curves.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum response
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal response (EC50)
    n : torch.Tensor or float
        Hill coefficient (cooperativity)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon  # Ensure K is positive
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(n * x_log)
    K_n = torch.exp(n * K_log)
    denominator = K_n + x_n
    return Vmax * (x_n / denominator) + A


def Hill_based_negative(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Negative Hill equation: Vmax * (K^n / (K^n + x^n)) + A

    Used for modeling repression dose-response curves.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum repression
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal inhibition (IC50)
    n : torch.Tensor or float
        Hill coefficient (cooperativity)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(n * x_log)
    K_n = torch.exp(n * K_log)
    fraction = K_n / (K_n + x_n)
    return Vmax * fraction + A


def Hill_based_piecewise(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Piecewise Hill equation: switches between positive and negative based on sign of n.

    Parameters
    ----------
    x : torch.Tensor
        Input values (e.g., cis gene expression)
    Vmax : torch.Tensor or float
        Maximum response
    A : torch.Tensor or float
        Baseline response
    K : torch.Tensor or float
        Half-maximal response
    n : torch.Tensor or float
        Hill coefficient (sign determines direction)
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    torch.Tensor
        Predicted response values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon  # Ensure K is positive
    x_log = torch.log(x_safe)
    K_log = torch.log(K_safe)
    x_n = torch.exp(torch.abs(n) * x_log)
    K_n = torch.exp(torch.abs(n) * K_log)
    denominator = K_n + x_n
    fraction = torch.where(n < 0, K_n / denominator, x_n / denominator)
    return Vmax * fraction + A


def Polynomial_function(x, coeffs):
    """
    Polynomial dose-response function.

    Computes: sum_{i=1}^{degree} coeffs[i] * x^i

    Parameters
    ----------
    x : torch.Tensor
        Input values, shape [N] or [N, 1]
    coeffs : torch.Tensor
        Polynomial coefficients, shape [degree, T] or [S, degree, T]
        where T is number of features and S is number of samples

    Returns
    -------
    torch.Tensor
        Predicted response values
        - Shape [N, T] if coeffs is [degree, T]
        - Shape [S, N, T] if coeffs is [S, degree, T]
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    powers = torch.arange(1, coeffs.shape[-2] + 1, device=x.device).view(1, -1)  # [1, degree]
    x_powers = x ** powers  # [N, degree]

    if coeffs.dim() == 2:
        return x_powers @ coeffs  # [N, T]
    elif coeffs.dim() == 3:
        x_powers = x_powers.unsqueeze(0)  # [1, N, degree]
        return torch.matmul(x_powers, coeffs).squeeze(-2)  # [S, N, T]
    else:
        raise ValueError(f"Expected coeffs to have 2 or 3 dims, got shape {coeffs.shape}")


########################################
# Sigmoid and Cutoff Functions
########################################

def cutoff_sigmoid(x, threshold=0.1, slope=50.0):
    """
    Smooth cutoff function using sigmoid.

    Returns x * sigmoid(slope * (threshold - x))
    - Near threshold, the factor is ~0.5
    - For x << threshold, the factor approaches 1
    - For x >> threshold, the factor approaches 0

    Parameters
    ----------
    x : torch.Tensor
        Input values
    threshold : float
        Cutoff threshold
    slope : float
        Steepness of sigmoid transition

    Returns
    -------
    torch.Tensor
        Smoothly cutoff values
    """
    return x * torch.sigmoid(slope * (threshold - x))


########################################
# Tensor Utilities
########################################

def sample_or_use_point(name, value, device):
    """
    Handles whether `value` is a point estimate (float/tensor) or needs conversion.

    Parameters
    ----------
    name : str
        Name of the variable (for error messages)
    value : torch.Tensor, float, int, or np.ndarray
        Value to process
    device : torch.device
        Computation device (CPU/GPU)

    Returns
    -------
    torch.Tensor
        Processed tensor on specified device

    Raises
    ------
    TypeError
        If value is not a supported type
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)  # Treat as a fixed tensor (point estimate)
    elif isinstance(value, (int, float, np.ndarray)):  # Scalars or numpy arrays
        return torch.tensor(value, dtype=torch.float32, device=device)
    else:
        raise TypeError(f"Expected a tensor, float, or numpy array for {name}, but got {type(value)}.")


def check_tensor(name, tensor):
    """
    Debug utility to check tensor properties.

    Prints shape, min/max values, and checks for NaN/Inf.

    Parameters
    ----------
    name : str
        Name of the tensor (for display)
    tensor : torch.Tensor
        Tensor to check
    """
    print(f"--- {name} ---")
    print(f"  shape: {tensor.shape}")
    print(f"  min: {tensor.min().item()}, max: {tensor.max().item()}")
    print(f"  has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  has Inf: {torch.isinf(tensor).any().item()}")
