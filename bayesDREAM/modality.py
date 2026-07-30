"""
Modality data structure for multi-modal bayesDREAM.

This module defines the Modality class for storing and validating
different types of molecular measurements (genes, transcripts, splicing, etc.)
"""

import warnings
import numpy as np
import pandas as pd
import torch
from typing import Literal, Optional, Union, Dict, Any
from scipy import sparse

from .utils import make_names_unique, resolve_feature_names


class Modality:
    """
    Stores data and metadata for a single molecular modality.

    Attributes
    ----------
    name : str
        Name of the modality (e.g., 'gene', 'transcript', 'splicing')
    counts : np.ndarray or pd.DataFrame
        The count/measurement data. Shape depends on distribution:
        - negbinom/normal/studentt: (features, cells) or (cells, features)
        - multinomial: (features, cells, categories)
        - binomial: (features, cells) with separate denominator
    feature_meta : pd.DataFrame
        Feature-level metadata (genes, junctions, etc.)
    distribution : str
        Distribution family: 'negbinom', 'multinomial', 'binomial', 'normal', 'studentt'
    denominator : np.ndarray, optional
        For binomial distribution: (features, cells) array of trial counts
    """

    VALID_DISTRIBUTIONS = {'negbinom', 'multinomial', 'binomial', 'normal', 'studentt'}

    def __init__(
        self,
        name: str,
        counts: Union[np.ndarray, pd.DataFrame],
        feature_meta: pd.DataFrame,
        distribution: Literal['negbinom', 'multinomial', 'binomial', 'normal', 'studentt'],
        feature_names: Optional[list] = None,
        denominator: Optional[np.ndarray] = None,
        cells_axis: int = 1,  # 0 if cells are rows, 1 if cells are columns
        cell_names: Optional[list] = None,  # Explicit cell names (when counts is ndarray)
        min_count: int = 1,
        # Exon skipping specific parameters
        inc1: Optional[np.ndarray] = None,
        inc2: Optional[np.ndarray] = None,
        skip: Optional[np.ndarray] = None,
        exon_aggregate_method: Optional[str] = None,
    ):
        """
        Initialize a Modality.

        Parameters
        ----------
        name : str
            Modality name
        counts : np.ndarray or pd.DataFrame
            Count/measurement data
        feature_meta : pd.DataFrame
            Feature metadata with index matching features in counts
        distribution : str
            Distribution type
        denominator : np.ndarray, optional
            For binomial: denominator counts (e.g., total gene expression for SJ usage)
        cells_axis : int
            Which axis represents cells (0 or 1 for 2D data)
        cell_names : list, optional
            Explicit cell names/identifiers (used when counts is ndarray, not DataFrame)
        inc1 : np.ndarray, optional
            For exon skipping: inclusion counts from first junction (d1->a2)
        inc2 : np.ndarray, optional
            For exon skipping: inclusion counts from second junction (d2->a3)
        skip : np.ndarray, optional
            For exon skipping: skipping counts (d1->a3)
        exon_aggregate_method : str, optional
            For exon skipping: how inc1 and inc2 were aggregated ('min' or 'mean')
        """
        if distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {self.VALID_DISTRIBUTIONS}, got {distribution}")

        self.name = name
        self.distribution = distribution
        self.cells_axis = cells_axis

        # Handle different count input formats
        # Check if counts is sparse matrix
        self.is_sparse = sparse.issparse(counts)

        if isinstance(counts, pd.DataFrame):
            # DataFrame: store both values and DataFrame
            self.counts = counts.values
            self.count_df = counts
            self.is_sparse = False  # DataFrames are dense
            # Feature names: use explicit parameter if provided, else from index/columns
            if cells_axis == 1:
                self.feature_names = feature_names if feature_names is not None else counts.index.tolist()
                self.cell_names = counts.columns.tolist()
            else:
                self.feature_names = feature_names if feature_names is not None else counts.columns.tolist()
                self.cell_names = counts.index.tolist()
        elif self.is_sparse:
            # Sparse matrix: keep sparse, don't densify!
            self.counts = counts
            self.count_df = None
            self.feature_names = feature_names if feature_names is not None else None
            self.cell_names = cell_names if cell_names is not None else None
            pass  # sparse matrix stored as-is
        else:
            # Dense array: convert to numpy array
            self.counts = np.asarray(counts)
            self.count_df = None
            self.feature_names = feature_names if feature_names is not None else None
            self.cell_names = cell_names if cell_names is not None else None

        # If no explicit feature_names and no DataFrame index/columns to derive
        # from (raw ndarray/sparse counts, e.g. add_custom_modality), fall back
        # to feature_meta — same priority order used everywhere else in the
        # codebase, so downstream code can always rely on modality.feature_names
        # instead of re-deriving its own guess from feature_meta.
        if self.feature_names is None:
            self.feature_names = resolve_feature_names(feature_meta, context=f"Modality '{name}'")

        # Disambiguate duplicate feature names (e.g. Ensembl gene_name collisions
        # from pseudogenes/readthrough transcripts, or a many-to-one fallback
        # column like 'gene' repeating once per SJ belonging to that gene).
        # Name-based lookups throughout the codebase (feature_names.index(),
        # pd.Series indexed by feature_names, get_feature_subset(), plotting by
        # gene name, etc.) all assume uniqueness.
        if self.feature_names is not None:
            deduped = make_names_unique(self.feature_names)
            if deduped != self.feature_names:
                n_dup = sum(1 for a, b in zip(deduped, self.feature_names) if a != b)
                warnings.warn(
                    f"Modality '{name}' had {n_dup} duplicate feature name(s); "
                    "disambiguated by appending -1, -2, ... to repeats (scanpy-style). "
                    "Original identifiers are still available in feature_meta.",
                    UserWarning
                )
            self.feature_names = deduped

        self.feature_meta = feature_meta.copy()

        # For multinomial: ensure the implicit Kth residual slot (position K_max-1) is
        # always a REAL category, not a padded zero.
        #
        # Background: the K-1 residual parameterization computes
        #   p_{K_max-1} = 1 - sum(p_0 .. p_{K_max-2})
        # For features with K_actual < K_max the last slot is padded (zero observations).
        # If the real categories together already account for all probability mass (~1),
        # then p_{K_max-1} ≈ 0 or negative, and the Dirichlet prior on A inappropriately
        # assigns mass to the phantom categories that compete with real ones.
        #
        # Fix: swap the last real category (column K_actual-1) into the residual slot
        # (column K_max-1) for any feature where K_actual < K_max.  After reordering:
        #   columns 0 .. K_actual-2  : real explicit Hills
        #   columns K_actual-1 .. K_max-2 : phantom (padded, zero data)
        #   column  K_max-1              : real residual category  ✓
        #
        # n_categories (= K_actual per feature) is computed before reordering and is
        # unaffected by it (still counts non-zero columns, just at different positions).
        if distribution == 'multinomial':
            counts_arr = np.asarray(self.counts)  # safe: multinomial must be dense 3D
            if counts_arr.ndim == 3:
                # Compute per-feature actual category count from non-zero data (before reordering).
                # This always overwrites any user-provided n_categories to ensure it is
                # consistent with the reordering performed below.  Zero-count categories
                # (present in the tensor but with no reads in any cell) are treated as phantom.
                n_cats = (counts_arr.sum(axis=1) > 0).sum(axis=1).astype(int)
                self.feature_meta['n_categories'] = n_cats

                # Reorder: swap last real category to residual position for any feature
                # with K_actual < K_max.  Work on a copy so original input is untouched.
                K_max = counts_arr.shape[2]
                if K_max > 1 and np.any(n_cats < K_max):
                    counts_arr = counts_arr.copy()
                    for i in range(counts_arr.shape[0]):
                        K_actual = int(n_cats[i])
                        if 0 < K_actual < K_max:
                            # Swap col K_actual-1 (last real) with col K_max-1 (padded residual slot)
                            tmp = counts_arr[i, :, K_actual - 1].copy()
                            counts_arr[i, :, K_actual - 1] = 0.0  # was real, now becomes padding
                            counts_arr[i, :, K_max - 1] = tmp     # residual slot now real
                    self.counts = counts_arr

                # ---- Filter: drop multinomial features with ≤1 real category or
                #              zero ratio-variance across cells ----
                #
                # Filter 1: Features with ≤1 real category cannot have multinomial
                #           proportions (no structure to model).
                # Filter 2: Features where all real-category ratios (count/cell_total)
                #           have zero std across cells carry no fitting signal.
                keep = n_cats > 1  # [T] bool, Filter 1

                for i in range(len(n_cats)):
                    if not keep[i]:
                        continue
                    feat = counts_arr[i]  # [n_cells, K_max]
                    cell_totals = feat.sum(axis=1, keepdims=True)  # [n_cells, 1]
                    real_cat_mask = feat.sum(axis=0) > 0  # [K_max]
                    real_counts = feat[:, real_cat_mask]  # [n_cells, K_actual]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratios = np.where(
                            cell_totals > 0,
                            real_counts / cell_totals,
                            0.0,
                        )  # [n_cells, K_actual]
                    if np.all(ratios.std(axis=0) == 0):
                        keep[i] = False  # Filter 2: zero ratio variance

                n_dropped = int((~keep).sum())
                if n_dropped > 0:
                    n_le1_cat = int((n_cats <= 1).sum())
                    n_zero_var = n_dropped - n_le1_cat
                    warnings.warn(
                        f"[Modality '{name}'] Dropped {n_dropped} multinomial feature(s): "
                        f"{n_le1_cat} with ≤1 real category, "
                        f"{n_zero_var} with zero category-ratio variance across cells.",
                        UserWarning,
                        stacklevel=2,
                    )
                    counts_arr = counts_arr[keep]
                    if counts_arr.shape[0] == 0:
                        raise ValueError(
                            f"[Modality '{name}'] All multinomial features were filtered out "
                            f"(≤1 real category or zero ratio variance). Check input data."
                        )
                    self.feature_meta = self.feature_meta[keep].reset_index(drop=True)
                    self.feature_meta['n_categories'] = n_cats[keep]
                    self.counts = counts_arr
                    if self.feature_names is not None:
                        self.feature_names = [
                            fn for fn, k in zip(self.feature_names, keep) if k
                        ]

        # Handle denominator (can also be sparse)
        if denominator is not None:
            if sparse.issparse(denominator):
                self.denominator = denominator  # Keep sparse
            else:
                self.denominator = np.asarray(denominator)
        else:
            self.denominator = None

        # Exon skipping specific storage
        self.inc1 = np.asarray(inc1) if inc1 is not None else None
        self.inc2 = np.asarray(inc2) if inc2 is not None else None
        self.skip = np.asarray(skip) if skip is not None else None
        self.exon_aggregate_method = exon_aggregate_method
        self._technical_fit_aggregate_method = None  # Track method used during technical fit

        # Store unfiltered versions for recovery when switching methods
        # (Only used for exon skipping modalities)
        self._unfiltered_inc1 = None
        self._unfiltered_inc2 = None
        self._unfiltered_skip = None
        self._unfiltered_feature_meta = None

        # Per-modality fitting results storage
        # Distribution-specific alpha_y parameters (use these directly, not alpha_y_prefit)
        self.alpha_y_prefit_mult = None     # For negbinom: multiplicative correction [C, T]
        self.alpha_y_prefit_add = None      # For normal/studentt/binomial/multinomial: additive correction [C, T]
        self.posterior_samples_ntc = None  # NTC fit: full posterior samples
        self.posterior_samples_trans = None      # Trans fit: full posterior samples

        # Sum factors DataFrame (negbinom modalities only).
        # Index: cell barcode. Columns: named sum factor variants
        # (e.g. 'sum_factor', 'sum_factor_adj', 'sum_factor_new').
        # Populated by bayesDREAM._init_sum_factors() after model init.
        # The 'cis' modality holds a reference to the same object as the primary modality.
        self.sum_factors: Optional[pd.DataFrame] = None

        self.min_count = min_count

        # Validate shapes
        self._validate()

        # Store dimensionality info
        self.dims = self._compute_dims()

        # Drop features with zero total counts or zero variance across cells.
        # This runs at creation and automatically re-runs after cell subsetting
        # (get_cell_subset returns a new Modality, triggering __init__).
        self._filter_zero_features()

    def _filter_zero_features(self) -> None:
        """
        Remove features with total counts below min_count or zero std across cells.

        Called at the end of __init__, so it also fires automatically after any cell
        subsetting (get_cell_subset returns a new Modality which calls __init__).
        Modifies self in place; updates counts, feature_meta, feature_names, denominator,
        inc1/inc2/skip, and dims.

        Per-distribution logic
        ----------------------
        negbinom   : drop feature if total counts < min_count OR std across cells == 0
        normal/
        studentt   : drop feature if std across cells == 0
                     (sum of values < min_count is also checked for consistency)
        binomial   : drop feature if denominator sum < min_count (too few observations)
                     OR ratio std (numer/denom) across cells == 0
        multinomial: drop feature if total counts (summed over cells+categories) < min_count
                     (zero-ratio-variance is already handled earlier in __init__)
        """
        min_count = self.min_count

        # ------------------------------------------------------------------ #
        # 1. Compute keep mask per distribution                               #
        # ------------------------------------------------------------------ #
        if self.distribution == 'multinomial':
            counts_arr = np.asarray(self.counts)          # (T, C, K)
            totals = counts_arr.sum(axis=(1, 2))          # (T,)
            keep = totals >= min_count

        elif self.distribution == 'binomial':
            cell_ax = self.cells_axis

            counts_num = (self.counts.toarray() if sparse.issparse(self.counts)
                          else np.asarray(self.counts))
            if self.denominator is not None:
                denom = (self.denominator.toarray() if sparse.issparse(self.denominator)
                         else np.asarray(self.denominator))
            else:
                denom = counts_num

            denom_sums = denom.sum(axis=cell_ax)
            has_data = denom_sums >= min_count

            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(denom > 0, counts_num / denom, np.nan)
            ratio_stds = np.nanstd(ratios, axis=cell_ax)
            has_variance = ratio_stds > 0

            keep = has_data & has_variance

        else:  # negbinom, normal, studentt
            cell_ax = self.cells_axis

            if self.is_sparse:
                _sp = self.counts.astype(float)
                totals = np.array(_sp.sum(axis=cell_ax)).flatten()
                _mean = np.array(_sp.mean(axis=cell_ax)).flatten()
                _sp2 = _sp.copy()
                _sp2.data **= 2
                _sq_mean = np.array(_sp2.mean(axis=cell_ax)).flatten()
                stds = np.sqrt(np.maximum(_sq_mean - _mean ** 2, 0))
            else:
                counts_arr = np.asarray(self.counts)
                totals = counts_arr.sum(axis=cell_ax)
                stds = counts_arr.std(axis=cell_ax)

            keep = (totals >= min_count) & (stds > 0)

        n_removed = int((~keep).sum())
        if n_removed == 0:
            return

        keep_idx = np.where(keep)[0]
        n_before = self.dims['n_features']

        print(
            f"[Modality '{self.name}'] Removed {n_removed}/{n_before} feature(s) "
            f"with total counts < {min_count} or zero variance across cells."
        )

        # ------------------------------------------------------------------ #
        # 2. Apply mask to counts and auxiliary arrays                        #
        # ------------------------------------------------------------------ #
        if self.distribution == 'multinomial':
            self.counts = np.asarray(self.counts)[keep_idx, :, :]

        elif self.cells_axis == 1:
            if self.is_sparse:
                self.counts = self.counts[keep_idx, :]
            else:
                self.counts = np.asarray(self.counts)[keep_idx, :]

            if self.denominator is not None:
                if sparse.issparse(self.denominator):
                    self.denominator = self.denominator[keep_idx, :]
                else:
                    self.denominator = np.asarray(self.denominator)[keep_idx, :]

            for _attr in ('inc1', 'inc2', 'skip'):
                _val = getattr(self, _attr)
                if _val is not None:
                    setattr(self, _attr, _val[keep_idx, :])

        else:  # cells_axis == 0
            if self.is_sparse:
                self.counts = self.counts[:, keep_idx]
            else:
                self.counts = np.asarray(self.counts)[:, keep_idx]

            if self.denominator is not None:
                if sparse.issparse(self.denominator):
                    self.denominator = self.denominator[:, keep_idx]
                else:
                    self.denominator = np.asarray(self.denominator)[:, keep_idx]

        # count_df is now stale — drop it to avoid inconsistency
        self.count_df = None

        # ------------------------------------------------------------------ #
        # 3. Update metadata and dims                                         #
        # ------------------------------------------------------------------ #
        self.feature_meta = self.feature_meta.iloc[keep_idx].reset_index(drop=True)
        if self.feature_names is not None:
            self.feature_names = [self.feature_names[i] for i in keep_idx]

        self.dims = self._compute_dims()

    def _validate(self):
        """Validate data shapes and requirements."""
        # Get ndim and shape (sparse matrices have these attributes too)
        counts_ndim = self.counts.ndim if hasattr(self.counts, 'ndim') else len(self.counts.shape)
        counts_shape = self.counts.shape

        if self.distribution in ['negbinom', 'normal', 'binomial', 'studentt']:
            if counts_ndim != 2:
                raise ValueError(f"{self.distribution} modality requires 2D counts, got shape {counts_shape}")
            n_features = counts_shape[1 - self.cells_axis]

        elif self.distribution == 'multinomial':
            if counts_ndim != 3:
                raise ValueError(f"multinomial modality requires 3D counts (features, cells, categories), got shape {counts_shape}")
            n_features = counts_shape[0]
        
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError(
                    f"feature_names has length {len(self.feature_names)} but counts has {n_features} features"
                )
        
        # Validate feature_meta matches
        if len(self.feature_meta) != n_features:
            raise ValueError(f"feature_meta has {len(self.feature_meta)} rows but counts has {n_features} features")

        # Validate denominator for binomial
        if self.distribution == 'binomial':
            if self.denominator is None:
                raise ValueError("binomial distribution requires denominator")
            if self.denominator.shape != self.counts.shape:
                raise ValueError(f"denominator shape {self.denominator.shape} must match counts shape {self.counts.shape}")
            counts_dense = self.counts.toarray() if sparse.issparse(self.counts) else self.counts
            denom_dense = self.denominator.toarray() if sparse.issparse(self.denominator) else self.denominator
            if np.any(counts_dense > denom_dense):
                raise ValueError(
                    f"[Modality '{self.name}'] counts exceed denominator for binomial distribution "
                    f"(found {int(np.sum(counts_dense > denom_dense))} violating entries)"
                )

        # Validate exon skipping data if provided
        if self.inc1 is not None or self.inc2 is not None or self.skip is not None:
            if self.distribution != 'binomial':
                raise ValueError("inc1/inc2/skip are only valid for binomial distribution (exon skipping)")
            if self.inc1 is None or self.inc2 is None or self.skip is None:
                raise ValueError("For exon skipping, must provide all three: inc1, inc2, skip")
            if self.inc1.shape != self.counts.shape or self.inc2.shape != self.counts.shape or self.skip.shape != self.counts.shape:
                raise ValueError(f"inc1, inc2, skip must all have same shape as counts: {self.counts.shape}")
            if self.exon_aggregate_method not in ['min', 'mean']:
                raise ValueError(f"exon_aggregate_method must be 'min' or 'mean', got: {self.exon_aggregate_method}")

    def _compute_dims(self) -> Dict[str, int]:
        """Compute dimensionality information."""
        dims = {}

        if self.distribution in ['negbinom', 'normal', 'binomial', 'studentt']:
            dims['n_features'] = self.counts.shape[1 - self.cells_axis]
            dims['n_cells'] = self.counts.shape[self.cells_axis]

        elif self.distribution == 'multinomial':
            dims['n_features'] = self.counts.shape[0]
            dims['n_cells'] = self.counts.shape[1]
            dims['n_categories'] = self.counts.shape[2]

        return dims

    def to_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Convert counts to torch tensor. Densifies if sparse."""
        if self.is_sparse:
            # Convert sparse to dense, then to tensor
            counts_dense = self.counts.toarray()
            return torch.tensor(counts_dense, dtype=torch.float32, device=device)
        else:
            return torch.tensor(self.counts, dtype=torch.float32, device=device)

    def get_feature_subset(self, feature_indices: Union[list, np.ndarray]) -> 'Modality':
        """
        Return a new Modality with subset of features.

        Parameters
        ----------
        feature_indices : list or array
            Indices or names of features to keep

        Returns
        -------
        Modality
            New modality with subset of features
        """
        if isinstance(feature_indices, (list, np.ndarray)) and len(feature_indices) > 0:
            if isinstance(feature_indices[0], str):
                # Convert names to indices
                if self.feature_names is None:
                    raise ValueError("Cannot subset by name: feature_names not available")
                feature_indices = [self.feature_names.index(n) for n in feature_indices]

        # Subset counts (maintain sparsity if counts is sparse)
        if self.distribution in ['negbinom', 'normal', 'binomial', 'studentt']:
            if self.cells_axis == 1:
                # Sparse-aware indexing
                new_counts = self.counts[feature_indices, :]
                new_denom = self.denominator[feature_indices, :] if self.denominator is not None else None
                new_inc1 = self.inc1[feature_indices, :] if self.inc1 is not None else None
                new_inc2 = self.inc2[feature_indices, :] if self.inc2 is not None else None
                new_skip = self.skip[feature_indices, :] if self.skip is not None else None
            else:
                new_counts = self.counts[:, feature_indices]
                new_denom = self.denominator[:, feature_indices] if self.denominator is not None else None
                new_inc1 = self.inc1[:, feature_indices] if self.inc1 is not None else None
                new_inc2 = self.inc2[:, feature_indices] if self.inc2 is not None else None
                new_skip = self.skip[:, feature_indices] if self.skip is not None else None
        else:
            # multinomial: features on axis 0 (not supported as sparse currently)
            new_counts = self.counts[feature_indices, :, :]
            new_denom = None
            new_inc1 = None
            new_inc2 = None
            new_skip = None

        # Subset metadata
        new_feature_meta = self.feature_meta.iloc[feature_indices].copy()
        # NEW: subset feature_names if present
        new_feature_names = None
        if self.feature_names is not None:
            new_feature_names = [self.feature_names[i] for i in feature_indices]
        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=new_feature_meta,
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis,
            feature_names=new_feature_names,
            cell_names=self.cell_names,
            min_count=self.min_count,
            inc1=new_inc1,
            inc2=new_inc2,
            skip=new_skip,
            exon_aggregate_method=self.exon_aggregate_method
        )

    def get_cell_subset(self, cell_indices: Union[list, np.ndarray]) -> 'Modality':
        """
        Return a new Modality with subset of cells.

        Parameters
        ----------
        cell_indices : list or array
            Indices or names of cells to keep

        Returns
        -------
        Modality
            New modality with subset of cells
        """
        if isinstance(cell_indices, (list, np.ndarray)) and len(cell_indices) > 0:
            if isinstance(cell_indices[0], str):
                # Convert names to indices
                if self.cell_names is None:
                    raise ValueError("Cannot subset by name: cell_names not available")
                cell_indices = [self.cell_names.index(n) for n in cell_indices]

        # Subset counts (maintain sparsity if counts is sparse)
        if self.distribution in ['negbinom', 'normal', 'binomial', 'studentt']:
            if self.cells_axis == 1:
                # Sparse-aware indexing
                new_counts = self.counts[:, cell_indices]
                new_denom = self.denominator[:, cell_indices] if self.denominator is not None else None
                new_inc1 = self.inc1[:, cell_indices] if self.inc1 is not None else None
                new_inc2 = self.inc2[:, cell_indices] if self.inc2 is not None else None
                new_skip = self.skip[:, cell_indices] if self.skip is not None else None
            else:
                new_counts = self.counts[cell_indices, :]
                new_denom = self.denominator[cell_indices, :] if self.denominator is not None else None
                new_inc1 = self.inc1[cell_indices, :] if self.inc1 is not None else None
                new_inc2 = self.inc2[cell_indices, :] if self.inc2 is not None else None
                new_skip = self.skip[cell_indices, :] if self.skip is not None else None
        else:
            # multinomial: cells on axis 1 (not supported as sparse currently)
            new_counts = self.counts[:, cell_indices, :]
            new_denom = None
            new_inc1 = None
            new_inc2 = None
            new_skip = None

        # Subset cell_names if available
        new_cell_names = [self.cell_names[i] for i in cell_indices] if self.cell_names is not None else None

        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=self.feature_meta.copy(),
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis,
            feature_names=self.feature_names,  # Preserve feature names during cell subsetting
            cell_names=new_cell_names,
            min_count=self.min_count,
            inc1=new_inc1,
            inc2=new_inc2,
            skip=new_skip,
            exon_aggregate_method=self.exon_aggregate_method
        )

    def set_exon_aggregate_method(self, method: str, allow_after_technical_fit: bool = False, recover_filtered: bool = True):
        """
        Change the exon skipping aggregation method and recompute inclusion counts.

        This recomputes self.counts (inclusion) and self.denominator (total) based on
        the new aggregation method. If unfiltered data was stored during modality creation,
        this will reprocess from that data and potentially recover features that were
        filtered with the old method but pass filtering with the new method.

        Parameters
        ----------
        method : str
            Aggregation method: 'min' or 'mean'
        allow_after_technical_fit : bool
            If True, allow changing method even after technical fit. Default: False.
            WARNING: Changing aggregation after technical fit invalidates the prefit
            overdispersion parameters.
        recover_filtered : bool
            If True (default), reprocess from unfiltered data to potentially recover
            features that were filtered with old method. If False or if unfiltered
            data is not available, just recompute from current data.

        Raises
        ------
        ValueError
            If method is invalid, modality is not exon skipping, or changing after
            technical fit without explicit permission.
        """
        if self.inc1 is None:
            raise ValueError("This modality does not have exon skipping data (inc1/inc2/skip)")

        if method not in ['min', 'mean']:
            raise ValueError(f"method must be 'min' or 'mean', got: {method}")

        # Check if technical fit was done with different method
        if self._technical_fit_aggregate_method is not None:
            if self._technical_fit_aggregate_method != method and not allow_after_technical_fit:
                raise ValueError(
                    f"Technical fit was performed with '{self._technical_fit_aggregate_method}' aggregation. "
                    f"Cannot change to '{method}' without invalidating prefit parameters. "
                    f"Set allow_after_technical_fit=True to override (not recommended)."
                )

        # Check if we can recover from unfiltered data
        can_recover = (recover_filtered and
                      self._unfiltered_inc1 is not None and
                      self._unfiltered_inc2 is not None and
                      self._unfiltered_skip is not None and
                      self._unfiltered_feature_meta is not None)

        if can_recover:
            # Reprocess from unfiltered data with new method
            print(f"[INFO] Reprocessing from unfiltered data with method='{method}'...")

            # Apply binomial ratio filtering with new method
            inc1_unfilt = self._unfiltered_inc1
            inc2_unfilt = self._unfiltered_inc2
            skip_unfilt = self._unfiltered_skip
            meta_unfilt = self._unfiltered_feature_meta

            n_events = inc1_unfilt.shape[0]

            # Compute inclusion using new method
            if method == 'min':
                inc_for_filter = np.minimum(inc1_unfilt, inc2_unfilt)
            elif method == 'mean':
                inc_for_filter = (inc1_unfilt + inc2_unfilt) / 2.0

            tot_for_filter = inc_for_filter + skip_unfilt

            # Check ratio variance for each event
            valid_events = []
            n_zero_var = 0

            for i in range(n_events):
                numer = inc_for_filter[i, :]  # inclusion counts across cells
                denom = tot_for_filter[i, :]   # total counts across cells

                # Compute ratios, excluding cells where denominator is 0
                valid_mask = denom > 0
                if valid_mask.sum() == 0:
                    n_zero_var += 1
                    continue

                ratios = numer[valid_mask] / denom[valid_mask]
                if ratios.std() == 0:
                    n_zero_var += 1
                    continue

                valid_events.append(i)

            if n_zero_var > 0:
                print(f"[INFO] Filtered {n_zero_var} exon skipping event(s) with zero variance in inclusion ratio (method='{method}')")

            if len(valid_events) == 0:
                raise ValueError(f"No exon skipping events with variable inclusion ratios after refiltering with method='{method}'!")

            # Update to valid events
            self.inc1 = inc1_unfilt[valid_events, :]
            self.inc2 = inc2_unfilt[valid_events, :]
            self.skip = skip_unfilt[valid_events, :]
            self.feature_meta = meta_unfilt.iloc[valid_events].reset_index(drop=True)

            # Recompute inclusion and total
            if method == 'min':
                self.counts = np.minimum(self.inc1, self.inc2)
            elif method == 'mean':
                self.counts = (self.inc1 + self.inc2) / 2.0

            self.denominator = self.counts + self.skip
            self.exon_aggregate_method = method

            # Update dims to reflect new number of features
            self.dims = self._compute_dims()

            # Report recovery stats
            n_before = inc1_unfilt.shape[0]
            n_after = len(valid_events)
            n_recovered = n_after - (n_before - n_zero_var)  # Difference from current filtered set
            print(f"[INFO] Recomputed with method='{method}': {n_after}/{n_before} events retained")

        else:
            # Just recompute from current filtered data (no recovery)
            if recover_filtered and self._unfiltered_inc1 is None:
                print(f"[INFO] Unfiltered data not available - recomputing from current filtered data")

            # Recompute inclusion
            if method == 'min':
                inclusion = np.minimum(self.inc1, self.inc2)
            elif method == 'mean':
                inclusion = (self.inc1 + self.inc2) / 2.0

            # Recompute total
            total = inclusion + self.skip

            # Update
            self.counts = inclusion
            self.denominator = total
            self.exon_aggregate_method = method

            # Update dims (though feature count shouldn't change in this branch)
            self.dims = self._compute_dims()

            print(f"[INFO] Recomputed exon skipping inclusion with method='{method}' (no feature recovery)")

    def is_exon_skipping(self) -> bool:
        """Check if this is an exon skipping modality with inc1/inc2/skip data."""
        return self.inc1 is not None

    def mark_technical_fit_complete(self):
        """
        Mark that technical fit has been performed with current aggregation method.

        This locks the aggregation method to prevent accidental changes that would
        invalidate the prefit overdispersion parameters.
        """
        if self.is_exon_skipping():
            self._technical_fit_aggregate_method = self.exon_aggregate_method

    def __repr__(self) -> str:
        exon_info = f", exon_agg='{self.exon_aggregate_method}'" if self.is_exon_skipping() else ""
        return f"Modality(name='{self.name}', distribution='{self.distribution}', dims={self.dims}{exon_info})"

    @property
    def alpha_y_prefit(self):
        """
        Get the distribution-appropriate alpha_y_prefit.

        Returns alpha_y_prefit_mult for negbinom, alpha_y_prefit_add for others.
        This property provides a unified interface while ensuring the correct
        parameter type is used for each distribution.
        """
        if self.distribution == 'negbinom':
            return self.alpha_y_prefit_mult
        else:
            return self.alpha_y_prefit_add

    @alpha_y_prefit.setter
    def alpha_y_prefit(self, value):
        """
        Set the distribution-appropriate alpha_y_prefit.

        Stores to alpha_y_prefit_mult for negbinom, alpha_y_prefit_add for others.
        This property provides a unified interface while ensuring the correct
        parameter type is stored for each distribution.
        """
        if self.distribution == 'negbinom':
            self.alpha_y_prefit_mult = value
        else:
            self.alpha_y_prefit_add = value
