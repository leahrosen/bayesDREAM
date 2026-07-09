"""
Permute NTC expression into perturbed cells to generate null distributions.

``permute_from_ntc`` is the main entry point.  It operates on a single
``Modality`` object and modifies its counts in-place.

Distribution-specific strategies
---------------------------------
negbinom
    NTC counts are normalised by sum_factor, resampled with replacement within
    each covariate group, then rescaled by the perturbed cell's own sum_factor.
    Counts are rounded to integers.  Requires ``modality.sum_factors[sum_factor_col]``.

binomial
    NTC fractions (count / denominator) are resampled, then scaled by the
    perturbed cell's real denominator.  This preserves variation in total counts
    while permuting the proportion signal.
    Requires ``modality.denominator``.

multinomial
    NTC category fractions (counts[cell, :] / total[cell]) are resampled, then
    scaled by the perturbed cell's real total counts.  Same principle as binomial.
    ``modality.counts`` must be a 3-D array (n_features, n_cells, n_categories).

normal / studentt
    NTC values are resampled directly with no normalisation — the additive scale
    makes sum-factor correction meaningless here.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union


def permute_from_ntc(
    modality,
    meta: pd.DataFrame,
    features2permute: Optional[Union[list, str]] = 'All',
    covariates: Optional[list] = None,
    sum_factor_col: str = 'sum_factor_adj',
    seed: Optional[int] = None,
) -> None:
    """
    Permute NTC expression into perturbed cells in-place.

    For each feature in ``features2permute``, within each covariate group,
    samples the NTC expression distribution (with replacement) and assigns
    those values to perturbed (non-NTC) cells.  The sampling strategy is
    distribution-aware — see module docstring for details.

    Parameters
    ----------
    modality : Modality
        The modality whose counts will be permuted.  Modified in-place.
    meta : pd.DataFrame
        Cell metadata.  Required columns: ``'cell'``, ``'target'``
        (NTC cells have ``target == 'ntc'``).
    features2permute : list of str/int, ``'All'``, or ``None``
        Feature names (or integer indices) to permute.  ``'All'`` / ``None``
        permutes every feature in the modality.
    covariates : list of str, optional
        Columns in ``meta`` used to stratify permutation.  Permutation is
        performed independently within each unique combination of covariate
        values so that each group's NTC distribution is preserved separately.
        If ``None``, all cells are treated as one group.
    sum_factor_col : str, default ``'sum_factor_adj'``
        Column in ``modality.sum_factors`` used to normalise counts for
        ``negbinom`` permutation.  Ignored for other distributions.
    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dist = modality.distribution

    # ------------------------------------------------------------------ #
    # Resolve feature list                                                  #
    # ------------------------------------------------------------------ #
    feature_names = modality.feature_names
    n_features = modality.dims['n_features']

    if features2permute is None or features2permute == 'All' or features2permute == ['All']:
        feat_indices = list(range(n_features))
    else:
        if isinstance(features2permute, str):
            features2permute = [features2permute]
        if feature_names is not None:
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            feat_indices = []
            for f in features2permute:
                if isinstance(f, str) and f in name_to_idx:
                    feat_indices.append(name_to_idx[f])
                elif isinstance(f, int) and f < n_features:
                    feat_indices.append(f)
        else:
            feat_indices = [f for f in features2permute if isinstance(f, int) and f < n_features]

    if not feat_indices:
        return

    # ------------------------------------------------------------------ #
    # Validate distribution-specific prerequisites                          #
    # ------------------------------------------------------------------ #
    if dist == 'negbinom':
        if modality.sum_factors is None or sum_factor_col not in modality.sum_factors.columns:
            raise ValueError(
                f"No column '{sum_factor_col}' in modality.sum_factors. "
                "Run adjust_ntc_sum_factor() first."
            )

    if dist == 'binomial' and modality.denominator is None:
        raise ValueError(
            "modality.denominator is required for binomial permutation."
        )

    if dist == 'multinomial' and modality.counts.ndim != 3:
        raise ValueError(
            "multinomial permutation requires 3-D counts (n_features, n_cells, n_categories)."
        )

    # ------------------------------------------------------------------ #
    # Build cell → column index mapping                                    #
    # ------------------------------------------------------------------ #
    cell_names = modality.cell_names
    if cell_names is None:
        raise ValueError("modality.cell_names must be set for permutation.")
    cell_to_idx = {c: i for i, c in enumerate(cell_names)}

    # ------------------------------------------------------------------ #
    # Work on a copy of counts (write back at the end)                     #
    # ------------------------------------------------------------------ #
    is_sparse_2d = sparse.issparse(modality.counts)

    if dist == 'multinomial':
        counts_work = np.array(modality.counts, dtype=float)   # (T, C, K)
    elif is_sparse_2d:
        counts_work = modality.counts.copy()
    else:
        counts_work = np.array(modality.counts, dtype=float)   # (T, C)

    # Denominator copy for binomial
    if dist == 'binomial':
        if sparse.issparse(modality.denominator):
            denom_arr = modality.denominator.toarray().astype(float)
        else:
            denom_arr = np.array(modality.denominator, dtype=float)   # (T, C)

    # ------------------------------------------------------------------ #
    # Covariate-stratified permutation                                     #
    # ------------------------------------------------------------------ #
    groups = meta.groupby(covariates) if covariates else [(None, meta)]

    for _key, group in groups:
        pert_cells = group.loc[group['target'] != 'ntc', 'cell'].values
        ntc_cells  = group.loc[group['target'] == 'ntc',  'cell'].values

        if len(pert_cells) == 0 or len(ntc_cells) == 0:
            continue

        pert_idx = [cell_to_idx[c] for c in pert_cells if c in cell_to_idx]
        ntc_idx  = [cell_to_idx[c] for c in ntc_cells  if c in cell_to_idx]

        if not pert_idx or not ntc_idx:
            continue

        # ---- negbinom ------------------------------------------------- #
        if dist == 'negbinom':
            sf_ntc  = modality.sum_factors.loc[ntc_cells,  sum_factor_col].values
            sf_pert = modality.sum_factors.loc[pert_cells, sum_factor_col].values

            for feat_row in feat_indices:
                if is_sparse_2d:
                    ntc_counts = np.asarray(
                        counts_work[feat_row, ntc_idx].todense()
                    ).flatten().astype(float)
                else:
                    ntc_counts = counts_work[feat_row, ntc_idx]

                rates = ntc_counts / np.maximum(sf_ntc, 1e-12)
                sampled_rates = rng.choice(rates, size=len(pert_idx), replace=True)
                new_counts = np.round(sampled_rates * sf_pert)

                if is_sparse_2d:
                    counts_work = counts_work.tolil()
                    for i, col in enumerate(pert_idx):
                        counts_work[feat_row, col] = new_counts[i]
                    counts_work = counts_work.tocsr()
                else:
                    counts_work[feat_row, pert_idx] = new_counts

        # ---- binomial ------------------------------------------------- #
        elif dist == 'binomial':
            for feat_row in feat_indices:
                ntc_counts = counts_work[feat_row, ntc_idx]
                ntc_denom  = denom_arr[feat_row, ntc_idx]

                fracs = ntc_counts / np.maximum(ntc_denom, 1e-12)
                sampled_fracs = rng.choice(fracs, size=len(pert_idx), replace=True)
                pert_denom = denom_arr[feat_row, pert_idx]
                new_counts = np.round(sampled_fracs * pert_denom)

                counts_work[feat_row, pert_idx] = new_counts

        # ---- multinomial ---------------------------------------------- #
        elif dist == 'multinomial':
            for feat_row in feat_indices:
                ntc_cat_counts = counts_work[feat_row, ntc_idx, :]    # (N_ntc, K)
                ntc_totals = ntc_cat_counts.sum(axis=1)                # (N_ntc,)

                with np.errstate(invalid='ignore', divide='ignore'):
                    ntc_fracs = ntc_cat_counts / np.maximum(ntc_totals[:, np.newaxis], 1e-12)

                sampled_row_idx = rng.integers(0, len(ntc_idx), size=len(pert_idx))
                sampled_fracs = ntc_fracs[sampled_row_idx, :]          # (N_pert, K)

                pert_totals = counts_work[feat_row, pert_idx, :].sum(axis=1)  # (N_pert,)
                new_counts = np.round(
                    sampled_fracs * np.maximum(pert_totals[:, np.newaxis], 0)
                )

                counts_work[feat_row, pert_idx, :] = new_counts

        # ---- normal / studentt ---------------------------------------- #
        elif dist in ('normal', 'studentt'):
            for feat_row in feat_indices:
                ntc_vals = counts_work[feat_row, ntc_idx]
                sampled = rng.choice(ntc_vals, size=len(pert_idx), replace=True)
                counts_work[feat_row, pert_idx] = sampled

        else:
            raise ValueError(
                f"permute_from_ntc does not support distribution '{dist}'. "
                "Supported: negbinom, binomial, multinomial, normal, studentt."
            )

    # ------------------------------------------------------------------ #
    # Write back                                                           #
    # ------------------------------------------------------------------ #
    modality.counts = counts_work
