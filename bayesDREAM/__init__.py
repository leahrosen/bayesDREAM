"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling perturbation effects across multiple
molecular modalities using PyTorch and Pyro.
"""

from .model import bayesDREAM
from .modality import Modality
from .plotting.colors import ColorScheme
from .fitting.distributions import (
    get_observation_sampler,
    requires_denominator,
    requires_sum_factor,
    is_3d_distribution,
    supports_cell_line_effects,
    get_cell_line_effect_type,
    DISTRIBUTION_REGISTRY
)
from .plotting import (
    plot_scalar_parameter,
    plot_1d_parameter,
    plot_2d_parameter
)
from .simulation import simulate_from_trans_summary, permute_from_ntc
from .diagnostics import DiagnosticsMixin

__version__ = "1.0.0"
__all__ = [
    "bayesDREAM",
    "Modality",
    "ColorScheme",
    "get_observation_sampler",
    "requires_denominator",
    "requires_sum_factor",
    "is_3d_distribution",
    "supports_cell_line_effects",
    "get_cell_line_effect_type",
    "DISTRIBUTION_REGISTRY",
    "plot_scalar_parameter",
    "plot_1d_parameter",
    "plot_2d_parameter",
    "simulate_from_trans_summary",
    "permute_from_ntc",
    "DiagnosticsMixin",
]
