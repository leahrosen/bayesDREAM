"""
Custom modality methods for bayesDREAM.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Union

from ..modality import Modality


def _subset_to_model_cells(counts_array, file_cells, model_cells, name):
    """
    Subset the cell axis (axis 1) to the intersection of file_cells and model_cells,
    preserving model cell order.

    Unlike zero-filling, model cells absent from file_cells are excluded from the
    output rather than filled with zeros.  Extra cells in file_cells that are not
    in the model are silently dropped.

    Returns
    -------
    (out_array, kept_cells)
        out_array : ndarray with shape[1] == len(kept_cells)
        kept_cells : list of cell identifiers (subset of model_cells in model order)
    """
    file_idx = {c: i for i, c in enumerate(file_cells)}
    model_cells_set = set(model_cells)

    n_extra = sum(1 for c in file_cells if c not in model_cells_set)
    if n_extra:
        print(f"[INFO] [{name}] Dropping {n_extra} cells present in data but not in model.")

    kept_cells = [c for c in model_cells if c in file_idx]
    n_excluded = len(model_cells) - len(kept_cells)
    if n_excluded:
        print(f"[INFO] [{name}] {n_excluded}/{len(model_cells)} model cells absent from "
              "modality data; modality will cover a subset of model cells.")

    out_shape = list(counts_array.shape)
    out_shape[1] = len(kept_cells)
    out = np.zeros(out_shape, dtype=counts_array.dtype)

    for j, cell in enumerate(kept_cells):
        if counts_array.ndim == 2:
            out[:, j] = counts_array[:, file_idx[cell]]
        elif counts_array.ndim == 3:
            out[:, j, :] = counts_array[:, file_idx[cell], :]

    return out, kept_cells


class CustomModalityMixin:
    """Mixin for custom modality support."""

    def add_custom_modality(
        self,
        name: str,
        counts: Union[np.ndarray, pd.DataFrame],
        feature_meta: pd.DataFrame,
        distribution: str,
        denominator: Optional[np.ndarray] = None,
        cell_names: Optional[List[str]] = None,
        overwrite: bool = False
    ):
        """
        Add a custom user-defined modality with distribution-specific filtering.

        Cells present in the input data but absent from the model are dropped.
        Model cells absent from the input data are excluded from the modality
        (the modality may therefore cover fewer cells than the model).
        fit_trans automatically subsets x_true to the modality's cell set.

        Parameters
        ----------
        name : str
            Modality name
        counts : array or DataFrame
            Measurement data. If DataFrame, cell names come from columns.
            If ndarray, use cell_names to specify cell identifiers.
        feature_meta : pd.DataFrame
            Feature metadata
        distribution : str
            'negbinom', 'multinomial', 'binomial', 'normal', or 'studentt'
        denominator : array, optional
            For binomial: denominator counts (same shape as counts)
        cell_names : list of str, optional
            Cell names for the input data (only used when counts is ndarray).
        overwrite : bool, default=False
            Whether to overwrite existing modality with the same name
        """
        # Extract counts array and file cell names from input
        if isinstance(counts, pd.DataFrame):
            counts_array = counts.values
            file_cells = counts.columns.tolist()
        else:
            counts_array = np.asarray(counts)
            file_cells = cell_names  # may be None

        # Align to model cell order — subset to intersection (no zero-fill)
        model_cells = self.meta['cell'].tolist()
        if file_cells is not None:
            counts_array, effective_cells = _subset_to_model_cells(
                counts_array, file_cells, model_cells, name
            )
            if denominator is not None:
                denom_raw = (denominator.values
                             if isinstance(denominator, pd.DataFrame)
                             else np.asarray(denominator))
                denominator, _ = _subset_to_model_cells(
                    denom_raw, file_cells, model_cells, name
                )
        else:
            warnings.warn(
                f"[{name}] cell_names not provided; assuming counts columns "
                "are already in model cell order."
            )
            effective_cells = model_cells

        # Denominator validation for binomial
        if distribution == 'binomial':
            if denominator is None:
                raise ValueError(
                    f"denominator required for binomial distribution in '{name}' modality"
                )
            if not isinstance(denominator, np.ndarray):
                denominator = np.asarray(denominator)

        # Apply distribution-specific feature filtering
        valid_features = None

        if distribution in ['negbinom', 'normal', 'studentt']:
            if counts_array.ndim == 2:
                feature_stds = counts_array.std(axis=1)
                valid_features = feature_stds != 0
                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero std "
                          f"in '{name}' modality ({distribution})")

        elif distribution == 'binomial':
            denom_array = denominator

            if counts_array.shape != denom_array.shape:
                raise ValueError(
                    f"counts and denominator must have same shape for binomial "
                    f"in '{name}' modality"
                )

            if counts_array.ndim == 2:
                n_features = counts_array.shape[0]
                valid_features = np.ones(n_features, dtype=bool)

                for i in range(n_features):
                    numer = counts_array[i, :]
                    denom = denom_array[i, :]
                    valid_mask = denom > 0
                    if valid_mask.sum() == 0:
                        valid_features[i] = False
                        continue
                    ratios = numer[valid_mask] / denom[valid_mask]
                    if ratios.std() == 0:
                        valid_features[i] = False

                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero ratio "
                          f"variance in '{name}' modality (binomial)")

        elif distribution == 'multinomial':
            if counts_array.ndim != 3:
                raise ValueError(
                    f"multinomial requires 3D counts (features, cells, categories) "
                    f"in '{name}' modality"
                )

            n_features = counts_array.shape[0]
            valid_features = np.ones(n_features, dtype=bool)

            for i in range(n_features):
                feature_counts = counts_array[i, :, :]  # (cells, categories)
                totals = feature_counts.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)
                ratio_stds = ratios.std(axis=0)
                if np.all(ratio_stds == 0):
                    valid_features[i] = False

            num_filtered = (~valid_features).sum()
            if num_filtered > 0:
                print(f"[INFO] Filtering {num_filtered} feature(s) with zero variance "
                      f"in ALL category ratios in '{name}' modality (multinomial)")

        # Apply feature mask
        if valid_features is not None:
            if not np.any(valid_features):
                raise ValueError(
                    f"No features left after filtering zero-variance features "
                    f"in '{name}' modality!"
                )
            if not np.all(valid_features):
                counts_array = counts_array[valid_features]
                feature_meta = feature_meta.iloc[valid_features].copy()
                if denominator is not None:
                    denominator = denominator[valid_features]

        modality = Modality(
            name=name,
            counts=counts_array,
            feature_meta=feature_meta,
            distribution=distribution,
            denominator=denominator,
            cells_axis=1,
            cell_names=effective_cells,
        )
        self.add_modality(name, modality, overwrite=overwrite)
