"""
Fitting methods for bayesDREAM.

This module contains the model fitting logic separated by stage:
- ntc: NTC fitting (overdispersion estimation and batch effect correction)
- cis: Cis gene expression fitting
- trans: Trans effects fitting
"""

from .ntc import NTCFitter
from .cis import CisFitter
from .trans import TransFitter

# Backward-compat alias
TechnicalFitter = NTCFitter

__all__ = ['NTCFitter', 'CisFitter', 'TransFitter', 'TechnicalFitter']
