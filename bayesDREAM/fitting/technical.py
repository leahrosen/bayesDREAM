"""
Backward-compatibility shim. The NTC fitter now lives in ntc.py.
"""
import warnings
from .ntc import NTCFitter

warnings.warn(
    "bayesDREAM.fitting.technical is deprecated — import from bayesDREAM.fitting.ntc instead.",
    DeprecationWarning,
    stacklevel=2,
)

TechnicalFitter = NTCFitter

__all__ = ['TechnicalFitter', 'NTCFitter']
