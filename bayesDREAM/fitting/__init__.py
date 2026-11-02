"""
Fitting methods for bayesDREAM.

This module contains the model fitting logic separated by stage:
- ntc: NTC fitting (overdispersion estimation and batch effect correction)
- cis: Cis gene expression fitting
- trans: Trans effects fitting
"""

import torch


class FitterBase:
	def _t(self, x, dtype=torch.float32):
		return torch.as_tensor(x, dtype=dtype, device=self.model.device)

	def _to_cpu(self, x):
		return x.detach().cpu() if isinstance(x, torch.Tensor) else x


from .ntc import NTCFitter
from .cis import CisFitter
from .trans import TransFitter

__all__ = ['NTCFitter', 'CisFitter', 'TransFitter']
