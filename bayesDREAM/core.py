"""
Core bayesDREAM implementation.

This module contains the _BayesDREAMCore base class with delegation to
specialized fitters for technical, cis, and trans modeling.
"""

import os
import functools
import warnings
from typing import Union
import numpy as np
import pandas as pd
import torch
import pyro
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from scipy import sparse

# Import utility functions and modules
from .utils import (
    set_max_threads,
    find_beta,
    calculate_mu_x_guide,
    Hill_based_positive,
    Hill_based_negative,
    Hill_based_piecewise,
    Polynomial_function,
    cutoff_sigmoid,
    sample_or_use_point,
    check_tensor
)
from .modality import Modality
from .fitting.distributions import get_observation_sampler, requires_denominator, is_3d_distribution

# Import fitters
from .fitting import NTCFitter, CisFitter, TransFitter
from .io import ModelSaver, ModelLoader, ModelSummarizer
from .plotting.model_plots import ModelPlottingMixin
from .diagnostics import DiagnosticsMixin

warnings.simplefilter(action="ignore", category=FutureWarning)


class _BayesDREAMCore(ModelPlottingMixin, DiagnosticsMixin):
    """
    Internal core class for the three-step Bayesian Dosage Response Effects Across Modalities framework:

    1) Optional technical group prefit (modeling alpha_y for NTC),
    2) Fitting cis effects (model_x),
    3) Fitting trans effects (model_y).
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame,
        gene_meta: pd.DataFrame = None,
        cis_gene: str = None,
        guide_covariates: list[str] = None,
        guide_covariates_ntc: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        output_dir: str = "./model_out",
        label: str = None,
        device: str = None,
        random_seed: int = 2402,
        cores: int = 1,
        guide_assignment: np.ndarray = None,
        guide_meta: pd.DataFrame = None,
        guide_target: pd.DataFrame = None,
        exclude_targets: list[str] = None,
        exclude_guides: list[str] = None,
        require_ntc: bool = True
    ):
        """
        Initialize the model with the metadata and count matrices.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell metadata DataFrame. For single-guide mode: includes columns cell, guide, target, sum_factor, etc.
            For high MOI mode: includes columns cell, sum_factor, etc. (NO guide or target columns)
            May optionally include technical group identifiers like 'cell_line', 'batch', 'lane', etc.
        counts : pd.DataFrame
            Counts DataFrame (genes as rows, cell barcodes as columns)
        gene_meta : pd.DataFrame, optional
            Gene metadata DataFrame with genes as rows. Required to have at least one identifier column.
            Recommended columns: 'gene' (or use index), 'gene_name', 'gene_id'
            If not provided, will create minimal metadata from counts.index
        cis_gene : str, optional
            The 'X' gene for cis modeling. May be omitted (in both single-guide and
            high-MOI mode) to defer commitment — call add_cis_gene() later, after
            fit_ntc(), to specify it. label must be provided explicitly when cis_gene
            is omitted.
        guide_covariates : list of str
            List of columns used to construct guide_used for non-NTC guides (single-guide mode only).
        guide_covariates_ntc : list of str or None
            List of columns used to construct guide_used for NTC guides (single-guide mode only).
        output_dir : str
            Where to save results
        label : str
            A label to prefix output files
        device : str or None
            "cuda" or "cpu" or None. If None, auto-detect.
        random_seed : int
            Random seed for reproducibility
        cores : int
            Number of CPU cores for Pyro to use
        guide_assignment : np.ndarray, optional
            Binary matrix [N, G] for high MOI mode. Each row represents a cell, each column a guide.
            guide_assignment[i, j] = 1 if cell i has guide j, else 0.
            If provided, must also provide guide_meta. Activates high MOI mode.
            Note: If dimensions are [G, N], will auto-transpose with a warning.
        guide_meta : pd.DataFrame, optional
            Guide metadata DataFrame for high MOI mode. Must have column 'guide'.
            Can optionally have 'target' column for simple one-to-one guide-target mapping.
            Index must match the column order of guide_assignment matrix.
            If provided, must also provide guide_assignment.
        guide_target : pd.DataFrame, optional
            High MOI only: Many-to-many guide-target relationship DataFrame.
            Must have columns 'guide' and 'target'. Multiple rows can have the same guide
            (one guide can target multiple genes). If provided, overrides guide_meta['target'].
            This allows flexible specification of guides with multiple possible targets.
            NTC guides can be specified with 'ntc', 'NTC', 'non-targeting', or 'non-targeting-control'.
        exclude_targets : list[str], optional
            High MOI only: List of target gene names to exclude. Cells with ANY guide targeting
            a gene in this list will be removed from analysis, regardless of other guides present.
            Example: exclude_targets=['MYB'] will remove cells with guides targeting MYB,
            even if they also have NTC or cis-targeting guides.
        exclude_guides : list[str], optional
            List of guide names to exclude. Cells carrying any guide in this list will be removed
            from analysis before fitting. Works in both single-guide and high MOI modes.
            In high MOI mode, a cell is excluded if it has ANY of the listed guides, regardless
            of other guides it carries.
        require_ntc : bool, optional
            If True (default), requires NTC cells in meta for single-guide mode.
            Set to False when subsetting a model that has already had technical fitting done,
            or when NTC cells are not needed (e.g., stress testing without NTC).
        """
        
        if label is None and cis_gene is not None:
            label = cis_gene
        elif label is None:
            raise ValueError(
                "label must be provided when cis_gene is not specified at initialization. "
                "When cis_gene is provided, label defaults to the gene name."
            )

        # Basic assignments
        self.meta = meta.copy()

        # Handle counts - can be DataFrame or sparse matrix
        self.is_sparse_counts = sparse.issparse(counts)
        if self.is_sparse_counts:
            self.counts = counts.copy() if hasattr(counts, 'copy') else counts.tocsr()
            self._cell_names = counts.columns.tolist() if isinstance(counts, pd.DataFrame) else None
        else:
            if isinstance(counts, pd.DataFrame):
                self.counts = counts.copy()
                self._cell_names = counts.columns.tolist()
            else:
                self.counts = counts.copy() if hasattr(counts, 'copy') else counts
                self._cell_names = None

        self.cis_gene = cis_gene
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.label = label
        self.require_ntc = require_ntc

        # ==============================================================================
        # Detect high MOI mode and validate
        # ==============================================================================
        if guide_assignment is not None or guide_meta is not None:
            if guide_assignment is None or guide_meta is None:
                raise ValueError(
                    "Both guide_assignment and guide_meta must be provided for high MOI mode. "
                    "Got guide_assignment={}, guide_meta={}".format(
                        type(guide_assignment).__name__ if guide_assignment is not None else None,
                        type(guide_meta).__name__ if guide_meta is not None else None
                    )
                )
            self.is_high_moi = True

            # Validate guide_assignment shape
            if guide_assignment.ndim != 2:
                raise ValueError(
                    f"guide_assignment must be a 2D matrix (cells × guides), "
                    f"but got shape {guide_assignment.shape} with {guide_assignment.ndim} dimensions"
                )

            # Auto-detect and transpose if dimensions are swapped
            # Expected: (n_cells, n_guides)
            # If user provides (n_guides, n_cells), transpose it
            dim0, dim1 = guide_assignment.shape
            n_guides_meta = len(guide_meta)
            n_cells_meta = len(meta)

            # Check if dimensions match expected orientation
            if dim1 == n_guides_meta and dim0 == n_cells_meta:
                # Correct orientation: (cells, guides)
                N_cells_assignment, G_guides = dim0, dim1
            elif dim0 == n_guides_meta and dim1 == n_cells_meta:
                # Transposed: (guides, cells) - auto-fix
                warnings.warn(
                    f"[HIGH MOI] guide_assignment appears to be transposed (shape {guide_assignment.shape} = guides × cells). "
                    f"Expected (cells × guides). Auto-transposing to ({dim1}, {dim0}).",
                    UserWarning
                )
                guide_assignment = guide_assignment.T
                N_cells_assignment, G_guides = guide_assignment.shape
            else:
                # Cannot determine orientation - provide helpful error
                raise ValueError(
                    f"guide_assignment shape {guide_assignment.shape} does not match expected dimensions:\n"
                    f"  - guide_meta has {n_guides_meta} guides\n"
                    f"  - meta has {n_cells_meta} cells\n"
                    f"Expected guide_assignment shape: ({n_cells_meta}, {n_guides_meta}) [cells × guides]\n"
                    f"Got: {guide_assignment.shape}\n"
                    f"Please check your guide_assignment matrix orientation."
                )

            # Validate guide_meta matches resolved dimensions
            if len(guide_meta) != G_guides:
                raise ValueError(
                    f"guide_meta has {len(guide_meta)} rows but guide_assignment has {G_guides} guides (columns). "
                    f"These dimensions must match."
                )

            # Validate guide_meta has 'guide' column
            if 'guide' not in guide_meta.columns:
                raise ValueError(
                    f"guide_meta missing required column 'guide'. "
                    f"Available columns: {list(guide_meta.columns)}"
                )

            # Store guide assignment and metadata
            self.guide_assignment = guide_assignment.copy()
            self.guide_meta = guide_meta.copy()

            # Create guide_code mapping for guide_meta
            self.guide_meta['guide_code'] = range(G_guides)

            # Process guide-target relationships
            # Priority: guide_target > guide_meta['target']
            if guide_target is not None:
                # Validate guide_target DataFrame
                required_gt_cols = {'guide', 'target'}
                missing_gt_cols = required_gt_cols - set(guide_target.columns)
                if missing_gt_cols:
                    raise ValueError(
                        f"guide_target missing required columns: {missing_gt_cols}. "
                        f"Available columns: {list(guide_target.columns)}"
                    )

                # Create guide -> list of targets mapping
                guide_targets_dict = {}
                for _, row in guide_target.iterrows():
                    guide_name = row['guide']
                    target = row['target']
                    if guide_name not in guide_targets_dict:
                        guide_targets_dict[guide_name] = []
                    guide_targets_dict[guide_name].append(target)

                # Store for later use
                self.guide_targets_dict = guide_targets_dict

            elif 'target' in guide_meta.columns:
                # Use simple one-to-one mapping from guide_meta
                guide_targets_dict = {
                    row['guide']: [row['target']]
                    for _, row in guide_meta.iterrows()
                }
                self.guide_targets_dict = guide_targets_dict
            else:
                raise ValueError(
                    "Either guide_target DataFrame or guide_meta['target'] column must be provided "
                    "to specify guide-target relationships in high MOI mode."
                )

            print(f"[INFO] High MOI: {G_guides} guides, avg {guide_assignment.sum(axis=1).mean():.2f} per cell")

        else:
            self.is_high_moi = False

        # Ensure guide_covariates and guide_covariates_ntc are always lists
        if guide_covariates is None:
            guide_covariates = []

        if guide_covariates_ntc is None:
            guide_covariates_ntc = []

        # Store guide covariates for later use (e.g., subset_cells)
        self.guide_covariates = guide_covariates
        self.guide_covariates_ntc = guide_covariates_ntc

        # Input checks - different requirements for single-guide vs high MOI mode
        if self.is_high_moi:
            # High MOI mode: do NOT require 'guide' or 'target' in meta
            required_cols = {"cell", sum_factor_col} | set(guide_covariates) | set(guide_covariates_ntc)
            missing_cols = required_cols - set(self.meta.columns)
            if missing_cols:
                raise ValueError(f"[High MOI] Missing required columns in meta: {missing_cols}")

            # Validate guide_assignment matches meta length
            if len(self.meta) != N_cells_assignment:
                raise ValueError(
                    f"[High MOI] guide_assignment has {N_cells_assignment} rows but meta has {len(self.meta)} rows. "
                    f"These dimensions must match."
                )

        else:
            # Single-guide mode: require 'guide' and 'target' in meta
            required_cols = {"target", "cell", sum_factor_col, "guide"} | set(guide_covariates) | set(guide_covariates_ntc)
            missing_cols = required_cols - set(self.meta.columns)
            if missing_cols:
                raise ValueError(f"[Single-guide] Missing required columns in meta: {missing_cols}")

            if require_ntc and "ntc" not in self.meta["target"].values:
                raise ValueError(
                    "No NTC detected in the 'target' column. "
                    "If this is correct (e.g., you have already run fit_ntc() or don't need NTC cells), "
                    "use require_ntc=False."
                )

        # Populate cell names if not already set
        if self._cell_names is None:
            if isinstance(counts, pd.DataFrame):
                self._cell_names = counts.columns.tolist()
            else:
                # Use meta['cell'] as cell names
                self._cell_names = self.meta['cell'].tolist()

        if not set(self.meta["cell"]).issubset(set(self._cell_names)):
            raise ValueError("The 'cell' column in meta must correspond 1:1 with the cell names in counts.")

        if (self.meta[sum_factor_col] <= 0).any():
            raise ValueError(f"All values in sum_factor_col={sum_factor_col} column must be strictly greater than 0.")

        # For high MOI mode, create 'target' column based on guide assignment
        if self.is_high_moi:
            # Helper function to normalize NTC target names (case-insensitive)
            def is_ntc_target(target_name):
                """Check if target name is NTC (flexible matching)."""
                ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
                return target_name in ntc_variants

            # Classify each guide based on its targets (from guide_targets_dict)
            # A guide can have multiple targets, so we check if any match NTC, cis, or excluded
            ntc_guide_indices = []
            cis_guide_indices = []
            exclude_guide_indices = []

            for pos_idx, (guide_idx, guide_row) in enumerate(self.guide_meta.iterrows()):
                guide_name = guide_row['guide']
                targets = self.guide_targets_dict.get(guide_name, [])

                # Check if this guide has ANY NTC target
                if any(is_ntc_target(t) for t in targets):
                    ntc_guide_indices.append(pos_idx)

                # Check if this guide has ANY cis_gene target
                if self.cis_gene in targets:
                    cis_guide_indices.append(pos_idx)

                # Check if this guide has ANY excluded target
                if exclude_targets is not None and any(t in exclude_targets for t in targets):
                    exclude_guide_indices.append(pos_idx)

                # Check if this guide itself is excluded by name
                if exclude_guides is not None and guide_name in exclude_guides:
                    exclude_guide_indices.append(pos_idx)

            ntc_guide_indices = np.array(ntc_guide_indices)
            cis_guide_indices = np.array(cis_guide_indices)
            exclude_guide_indices = np.array(list(dict.fromkeys(exclude_guide_indices)))  # deduplicate, preserve order

            # Determine which cells have these guide types
            if len(exclude_guide_indices) > 0:
                has_excluded_guide = self.guide_assignment[:, exclude_guide_indices].sum(axis=1) > 0
            else:
                has_excluded_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            if len(ntc_guide_indices) > 0:
                has_ntc_guide = self.guide_assignment[:, ntc_guide_indices].sum(axis=1) > 0
            else:
                has_ntc_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            if len(cis_guide_indices) > 0:
                has_cis_guide = self.guide_assignment[:, cis_guide_indices].sum(axis=1) > 0
            else:
                has_cis_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            # Cell classification:
            # - If cell has ANY excluded guides -> target = 'excluded' (will be removed)
            # - Else if cell has ANY cis guides -> target = cis_gene
            # - Else if cell has ANY NTC guides (but no cis) -> target = 'ntc'
            # - Else -> target = 'other' (will be removed)
            targets = []
            for i in range(len(self.guide_assignment)):
                if has_excluded_guide[i]:
                    # Cell has guide(s) targeting excluded gene(s) - remove
                    targets.append('excluded')
                elif has_cis_guide[i]:
                    # Cell has cis guide(s) - regardless of other guides
                    targets.append(self.cis_gene)
                elif has_ntc_guide[i]:
                    # Cell has NTC guide(s) but no cis guides
                    # (may also have "other" guides - these are ignored)
                    targets.append('ntc')
                else:
                    # Cell has ONLY "other" guides (no NTC, no cis, no excluded)
                    targets.append('other')

            self.meta['target'] = targets

            # Add guide_code column (not meaningful in high MOI, marked as -1)
            self.meta['guide_code'] = -1

            ntc_count = (np.array(targets) == 'ntc').sum()
            other_count = (np.array(targets) == 'other').sum()
            excluded_count = (np.array(targets) == 'excluded').sum()
            print(f"[INFO] Cell classification before subsetting:")
            print(f"  NTC cells (NTC guides, no cis): {ntc_count}")
            if self.cis_gene is not None:
                cis_count = (np.array(targets) == self.cis_gene).sum()
                print(f"  {self.cis_gene}-targeting cells (any cis guides): {cis_count}")
                print(f"  Other-only cells (will be removed): {other_count}")
            else:
                print(f"  Other/unclassified cells (cis_gene deferred — target unknown until add_cis_gene()): {other_count}")
            if exclude_targets is not None or exclude_guides is not None:
                print(f"  Excluded cells (exclude_targets/exclude_guides): {excluded_count}")

        # Save original cell order before subsetting (for guide_assignment row alignment in high MOI)
        if self.is_high_moi:
            if isinstance(self.counts, pd.DataFrame):
                _ga_original_cell_names = list(self.counts.columns)
            else:
                _ga_original_cell_names = list(self._cell_names) if hasattr(self, '_cell_names') and self._cell_names else []

        # Drop cells carrying explicitly excluded guides (single-guide mode only;
        # high MOI handles this above via exclude_guide_indices → target='excluded')
        if exclude_guides is not None and not self.is_high_moi:
            excluded_guide_set = set(exclude_guides)
            n_before = len(self.meta)
            self.meta = self.meta[~self.meta['guide'].isin(excluded_guide_set)].copy()
            n_excluded = n_before - len(self.meta)
            if n_excluded > 0:
                print(f"[INFO] Excluded {n_excluded} cells carrying guides in exclude_guides={exclude_guides}")

        # For high MOI mode, drop 'excluded' cells unconditionally (independent of cis_gene,
        # since exclusion is determined purely by exclude_targets/exclude_guides). Doing this
        # here — rather than folding it into the cis_gene-dependent filter below — ensures
        # excluded cells don't linger into fit_ntc() when cis_gene is deferred.
        if self.is_high_moi:
            n_before_excl = len(self.meta)
            self.meta = self.meta[self.meta["target"] != "excluded"].copy()
            n_after_excl = len(self.meta)
            if n_after_excl < n_before_excl:
                print(f"[INFO] Excluded {n_before_excl - n_after_excl} cells (target='excluded')")

        # Subset meta and counts to relevant cells
        if self.cis_gene is not None:
            valid_cells = self.meta[self.meta["target"].isin(["ntc", self.cis_gene])]["cell"].unique()
            n_cells_before = len(self.meta["cell"].unique())
            if len(valid_cells) < n_cells_before:
                print(f"[INFO] Cells: {n_cells_before} → {len(valid_cells)} (kept NTC + {self.cis_gene} only)")
            self.meta = self.meta[self.meta["cell"].isin(valid_cells)].copy()
        else:
            # cis_gene not yet specified — keep all cells; add_cis_gene() will subset later
            valid_cells = self.meta["cell"].unique()
            print(f"[INFO] No cis_gene at init — keeping all {len(valid_cells)} cells. "
                  "Call add_cis_gene() before fit_cis().")

        # Subset counts by cells - works for both DataFrame and sparse
        if isinstance(self.counts, pd.DataFrame):
            self.counts = self.counts[valid_cells].copy()
        else:
            # Sparse or dense array - subset by column indices
            valid_cells_set = set(valid_cells)
            cell_indices = [i for i, cell in enumerate(self._cell_names) if cell in valid_cells_set]
            if self.is_sparse_counts:
                self.counts = self.counts[:, cell_indices]
            else:
                self.counts = self.counts[:, cell_indices]
            # Update cell names
            self._cell_names = [self._cell_names[i] for i in cell_indices]

        # For high MOI: subset guide_assignment to remove "other"-targeting guide columns.
        # Only possible once cis_gene is known — if deferred, keep the full guide panel;
        # add_cis_gene() will prune it once the cis gene is committed.
        if self.is_high_moi and self.cis_gene is not None:
            # Keep only NTC and cis-gene targeting guides
            # A guide is kept if it has ANY NTC or cis target among its possible targets
            keep_guide_indices = []

            def is_ntc_target(target_name):
                """Check if target name is NTC (flexible matching)."""
                ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
                return target_name in ntc_variants

            for pos_idx, (guide_idx, guide_row) in enumerate(self.guide_meta.iterrows()):
                guide_name = guide_row['guide']
                targets = self.guide_targets_dict.get(guide_name, [])

                # Keep if ANY target is NTC or cis_gene
                if any(is_ntc_target(t) for t in targets) or self.cis_gene in targets:
                    keep_guide_indices.append(pos_idx)

            keep_guide_indices = np.array(keep_guide_indices)

            n_guides_before = self.guide_assignment.shape[1]
            n_guides_after = len(keep_guide_indices)

            # Subset guide_assignment columns
            self.guide_assignment = self.guide_assignment[:, keep_guide_indices]

            # Subset guide_meta rows
            self.guide_meta = self.guide_meta.iloc[keep_guide_indices].copy()

            # Update guide_code to match new indices
            self.guide_meta['guide_code'] = range(len(self.guide_meta))

            # Also update guide_targets_dict to only include kept guides
            kept_guide_names = set(self.guide_meta['guide'].values)
            self.guide_targets_dict = {
                guide: targets
                for guide, targets in self.guide_targets_dict.items()
                if guide in kept_guide_names
            }

            print(f"[INFO] Subsetted guides from {n_guides_before} to {n_guides_after} (keeping NTC + {self.cis_gene} guides only)")
        elif self.is_high_moi:
            print(f"[INFO] No cis_gene at init — keeping all {self.guide_assignment.shape[1]} guide columns. "
                  "Call add_cis_gene() to prune to NTC + cis-gene guides.")

        # Remove genes with zero total counts - works for DataFrame, dense, and sparse
        if isinstance(self.counts, pd.DataFrame):
            gene_sums = self.counts.sum(axis=1).values
        elif self.is_sparse_counts:
            # Sparse matrix - sum along axis 1 (cells)
            gene_sums = np.array(self.counts.sum(axis=1)).flatten()
        else:
            # Dense array
            gene_sums = self.counts.sum(axis=1)

        detected_mask = gene_sums > 0
        num_removed = (~detected_mask).sum()

        # Cis gene existence and zero-variance are already validated by _extract_cis_from_gene
        # (called in bayesDREAM.__init__ before super().__init__).
        # For DataFrame counts only: double-check the gene is still in the index after cell subsetting.
        if self.cis_gene is not None and isinstance(self.counts, pd.DataFrame):
            if self.cis_gene not in self.counts.index:
                raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' not found in counts index!")
            cis_idx = self.counts.index.get_loc(self.cis_gene)
            if not detected_mask[cis_idx]:
                raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' has zero counts after subsetting!")

        # Subset counts to detected genes only (counts freed at end of init; used for cell alignment below)
        if isinstance(self.counts, pd.DataFrame):
            self.counts = self.counts.loc[detected_mask]
        else:
            gene_indices = np.where(detected_mask)[0]
            self.counts = self.counts[gene_indices, :]

        # zero-count genes removed here from self.counts; Modality.__init__ reports its own filtering
        # Feature names for trans modelling are read directly from the primary modality
        # (via model.get_modality(primary_modality).feature_names) so no separate trans_genes attribute.
        
        # Ensure same order of meta and counts
        # Use _cell_names for sparse/dense array compatibility
        if isinstance(self.counts, pd.DataFrame):
            self.meta = self.meta.set_index("cell", drop=False).loc[self.counts.columns]
        else:
            # For sparse/dense arrays, reorder meta to match _cell_names
            self.meta = self.meta.set_index("cell", drop=False).loc[self._cell_names]

        # Construct guide_used column (single-guide mode only)
        if not self.is_high_moi:
            self.meta["guide_used"] = self.meta.apply(
                lambda row: f"{row['guide']}_{'_'.join(str(row[cov]) for cov in (guide_covariates_ntc if row['target'] == 'ntc' else set(guide_covariates_ntc + guide_covariates)))}",
                axis=1
            )
            # one-hot encode guides
            self.meta['guide_code'] = pd.Categorical(self.meta['guide_used']).codes
        else:
            # High MOI: guide_used not needed, guide_code already set to -1
            self.meta["guide_used"] = "highmoi"  # Placeholder for compatibility
            # Convert guide_assignment to tensor and store after subsetting
            # Need to subset guide_assignment rows to match the subsetted cells.
            # Use the ORIGINAL cell order (saved before counts subsetting) so indices
            # correctly reference the 100-row guide_assignment, not the 50-row subset.
            cell_indices = [_ga_original_cell_names.index(cell) for cell in self.meta['cell']]
            self.guide_assignment = self.guide_assignment[cell_indices, :]
            self.guide_assignment_tensor = torch.tensor(
                self.guide_assignment,
                dtype=torch.float32,
                device='cpu'  # Will move to device below
            )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Move guide_assignment_tensor to device if in high MOI mode
        if self.is_high_moi:
            self.guide_assignment_tensor = self.guide_assignment_tensor.to(self.device)

        # Set random seeds & threads
        pyro.set_rng_seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        set_max_threads(cores)

        # Bookkeeping for results
        self.alpha_x_prefit = None    # from step1: shape [C], always a mean point estimate
        # NOTE: alpha_y_prefit is stored per-modality as a mean point estimate [C, T]
        self.trace_cellline = None    # from step1
        self.trace_x = None          # from step2
        self.trace_y = None          # from step3

        # Initialize fitter objects and helpers
        self._ntc_fitter = NTCFitter(self)
        self._cis_fitter = CisFitter(self)
        self._trans_fitter = TransFitter(self)
        self._saver = ModelSaver(self)
        self._loader = ModelLoader(self)

        self._summarizer = ModelSummarizer(self)

        # Free the raw counts matrix – data now lives in modalities.
        # After all init logic above has consumed self.counts, drop it to save RAM.
        self.counts = None
        self.is_sparse_counts = None  # no longer meaningful without the matrix

    def set_alpha_x(
        self,
        alpha_x,
        covariates: list[str] = None,
    ):
        """
        Set cis-gene overdispersion scaling factors from a pre-fitted or external estimate.

        Stores ``alpha_x`` as a point-estimate tensor of shape ``[C]``, where index 0
        is the reference group (value fixed to 1.0).  Must be called before
        ``fit_cis()`` when overdispersion is supplied externally rather than estimated
        by ``fit_ntc()``.

        Parameters
        ----------
        alpha_x : array-like or torch.Tensor
            Overdispersion scale factors, shape ``[C]`` (including reference group at
            index 0).  Accepts numpy arrays, lists, or tensors.
        covariates : list of str, optional
            Column names in ``meta`` used to define technical groups (e.g.,
            ``['cell_line']``).  If provided, creates/overwrites
            ``meta['technical_group_code']``.  If ``None``, assumes
            ``technical_group_code`` was already set (raises ``ValueError`` if not).
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_x corresponds.")
        self.alpha_x_prefit = sample_or_use_point("alpha_x_posterior", alpha_x, self.device).flatten()

    def set_alpha_y(
        self,
        alpha_y,
        covariates: list[str] = None,
    ):
        """
        Set trans-gene overdispersion scaling factors from a pre-fitted or external estimate.

        Stores ``alpha_y`` as a point-estimate tensor of shape ``[C, T]`` on the
        primary modality, where ``C`` is the number of technical groups (index 0 =
        reference group) and ``T`` is the number of trans features.  Must be called
        before ``fit_trans()`` when overdispersion is supplied externally rather than
        estimated by ``fit_ntc()``.

        Parameters
        ----------
        alpha_y : array-like or torch.Tensor
            Overdispersion scale factors, shape ``[C, T]`` (reference group at index 0).
            Accepts numpy arrays, lists, or tensors.
        covariates : list of str, optional
            Column names in ``meta`` used to define technical groups (e.g.,
            ``['cell_line']``).  If provided, creates/overwrites
            ``meta['technical_group_code']``.  If ``None``, assumes
            ``technical_group_code`` was already set (raises ``ValueError`` if not).
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_xy corresponds.")

        # Convert alpha_y to tensor
        alpha_y = sample_or_use_point("alpha_y_posterior", alpha_y, self.device)

        # Store in primary modality (not model-level)
        primary_mod = self.get_modality(self.primary_modality)
        primary_mod.alpha_y_prefit = alpha_y  # Uses property to store in distribution-specific attribute

    def set_x_true(
        self,
        x_true,
    ):
        """
        Set the posterior cis-gene expression estimate from an external source.

        Stores ``x_true`` as a 1-D point-estimate tensor of shape ``[N]`` (one value
        per cell, matching ``len(self.meta)``).  Must be called before ``fit_trans()``
        when ``x_true`` comes from a previously saved fit rather than from running
        ``fit_cis()`` directly.

        Parameters
        ----------
        x_true : array-like or torch.Tensor
            Posterior mean cis-gene expression, shape ``[N]``.
            Accepts numpy arrays, lists, or tensors.

        Raises
        ------
        ValueError
            If the provided array does not have exactly ``N`` elements.
        """
        N = len(self.meta)
        x_true = sample_or_use_point("x_true_posterior", x_true, self.device)
        if not (x_true.ndim == 1 and x_true.shape[0] == N):
            raise ValueError(
                f"x_true must have shape N ({N},), but got {x_true.shape}."
            )
        self.x_true = x_true


    def adjust_ntc_sum_factor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_adj: str = "sum_factor_adj",
        covariates: list[str] = None # Technical group covariates (e.g., ["lane", "cell_line"]) or could be empty
    ):
        """
        Step 1 of sum factor adjustment: Normalize guides to NTC controls.

        Use BEFORE fit_cis() to account for guide-level technical variation.
        Computes adjustment factor = mean_ntc_sum_factor / mean_guide_sum_factor
        within covariate groups (e.g., technical groups like cell_line, lane).

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit' (default output name)
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column in meta (default: 'sum_factor')
        sum_factor_col_adj : str
            Name for adjusted sum factor column to create (default: 'sum_factor_adj')
        covariates : list of str, optional
            Technical group covariates to group by for adjustment (e.g., ['cell_line', 'lane']).
            If None or empty, uses global mean across all cells.

        Returns
        -------
        None
            Creates new column sum_factor_col_adj in self.meta

        Notes
        -----
        **Single-guide mode** (low MOI): for each guide within each covariate group:
        - Compute mean NTC sum factor: mean_ntc
        - Compute mean guide sum factor: mean_guide
        - Adjustment factor = mean_ntc / mean_guide
        - Adjusted sum factor = sum_factor * adjustment_factor

        **High MOI mode**: guide effects are assumed additive in log-space (same as fit_cis).
        For each guide g within each covariate group:
        - mean_log_sf_g = mean(log(sum_factor)) over cells containing guide g
        - weighted_NTC = weighted mean of mean_log_sf_g for NTC guides (weights = cells_per_guide)
        - delta_g = mean_log_sf_g - weighted_NTC
        For each cell c: log(sf_adj_c) = log(sf_c) - sum(delta_g for guides in cell c)
        """
        primary_mod = self.get_modality(self.primary_modality)

        meta_out = self.meta.copy()
        meta_out["original_index"] = np.arange(len(meta_out))

        # Prefer sum_factors on the modality; fall back to meta for the initial
        # 'sum_factor' column (present in meta from initialisation).
        if sum_factor_col_old not in meta_out.columns:
            if (primary_mod.sum_factors is not None
                    and sum_factor_col_old in primary_mod.sum_factors.columns):
                meta_out[sum_factor_col_old] = primary_mod.sum_factors.loc[
                    meta_out['cell'].values, sum_factor_col_old
                ].values
            else:
                raise ValueError(
                    f"No column '{sum_factor_col_old}' found in meta or modality sum_factors."
                )

        # Drop existing adjustment_factor column if it exists (prevents merge conflicts)
        if "adjustment_factor" in meta_out.columns:
            meta_out = meta_out.drop(columns=["adjustment_factor"])

        # Make sure all covariates are actually in meta_out
        if covariates:
            missing_cols = [c for c in covariates if c not in meta_out.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing covariate columns: {missing_cols}. "
                    f"Available columns are: {list(meta_out.columns)}"
            )
        
        if self.is_high_moi:
            # High MOI: guide effects are additive in log-space (same assumption as fit_cis).
            # For a cell with guides g1, g2: log(sf_adj) = log(sf) - delta_g1 - delta_g2
            # where delta_g = mean_log_sf_g - weighted_NTC_log_sf (per covariate group).
            # Weights for NTC mean = cells_per_guide, matching fit_cis.

            guide_assignment = self.guide_assignment  # (n_cells, n_guides)

            # Determine which guides are NTC — works for both guide_meta['target'] and guide_targets_dict
            _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
            if 'target' in self.guide_meta.columns:
                is_ntc_guide = self.guide_meta['target'].isin(_ntc_variants).values
            elif hasattr(self, 'guide_targets_dict') and self.guide_targets_dict:
                is_ntc_guide = np.array([
                    any(t in _ntc_variants for t in self.guide_targets_dict.get(row['guide'], []))
                    for _, row in self.guide_meta.iterrows()
                ])
            else:
                raise ValueError(
                    "adjust_ntc_sum_factor (high MOI): cannot identify NTC guides — "
                    "guide_meta has no 'target' column and guide_targets_dict is unavailable."
                )

            log_sf = np.log(meta_out[sum_factor_col_old].values.astype(float))
            log_sf_adj = log_sf.copy()

            def _apply_highmoi_correction(pos_indices):
                ga = guide_assignment[pos_indices]   # (n_sub, n_guides)
                lsf = log_sf[pos_indices]            # (n_sub,)

                cells_per_guide = ga.sum(axis=0).astype(float)  # (n_guides,)

                # (1) Mean log(sf) per guide via matrix multiply
                mean_log_sf_g = np.where(
                    cells_per_guide > 0,
                    (ga.T @ lsf) / np.maximum(cells_per_guide, 1.0),
                    np.nan
                )  # (n_guides,)

                # (2) Weighted NTC mean (weights = cells_per_guide, matching fit_cis)
                ntc_means = mean_log_sf_g[is_ntc_guide]
                ntc_counts = cells_per_guide[is_ntc_guide]
                valid = ~np.isnan(ntc_means) & (ntc_counts > 0)
                if valid.sum() == 0:
                    return  # No NTC guides in group; skip correction
                weighted_NTC_log_sf = np.average(ntc_means[valid], weights=ntc_counts[valid])

                # (3) Per-guide delta; guides with no cells in this group get delta=0
                delta_g = np.where(~np.isnan(mean_log_sf_g), mean_log_sf_g - weighted_NTC_log_sf, 0.0)

                # (4) Additive correction per cell in log-space
                log_sf_adj[pos_indices] -= ga @ delta_g

            if not covariates:
                _apply_highmoi_correction(np.arange(len(meta_out)))
            else:
                meta_out["_pos"] = np.arange(len(meta_out))
                for _, group_df in meta_out.groupby(covariates):
                    _apply_highmoi_correction(group_df["_pos"].values)
                meta_out.drop(columns=["_pos"], inplace=True)

            meta_out[sum_factor_col_adj] = np.exp(log_sf_adj)

        elif not covariates:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            mean_ntc_value = meta_out.loc[meta_out['target'] == 'ntc', sum_factor_col_old].mean()

            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )

            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            df_guide["adjustment_factor"] = (
                mean_ntc_value / (df_guide["mean_SumFacs_guide"])
            )

            # (4) Merge that ratio back onto meta_out
            meta_out = pd.merge(
                meta_out,
                df_guide[["guide_used", "adjustment_factor"]],
                on="guide_used",
                how="left"
            )

            # (5) Multiply original sum_factor by ratio
            meta_out[sum_factor_col_adj] = meta_out[sum_factor_col_old] * meta_out["adjustment_factor"]

        else:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            df_ntc = (
                meta_out.loc[meta_out["target"] == "ntc"]
                .groupby(covariates)[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_ntc")
            )

            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )

            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
            merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]

            # (4) Merge that ratio back onto meta_out
            merge_cols = covariates + ["guide_used", "adjustment_factor"]
            meta_out = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")

            # (5) Multiply original sum_factor by ratio
            meta_out[sum_factor_col_adj] = meta_out[sum_factor_col_old] * meta_out["adjustment_factor"]

        meta_out.sort_values("original_index", inplace=True)
        meta_out.drop(columns="original_index", inplace=True)

        # Write adjusted sum factor to modality, not back to meta
        if primary_mod.sum_factors is None:
            primary_mod.sum_factors = pd.DataFrame(index=meta_out['cell'].values)
        primary_mod.sum_factors[sum_factor_col_adj] = meta_out.set_index('cell')[sum_factor_col_adj]

        print(f"[INFO] Created '{sum_factor_col_adj}' in modality sum_factors with NTC-based guide-level adjustment.")

    def refit_sumfactor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_refit: str = "sum_factor_refit",
        covariates: list[str] = None,
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 0.1,
        use_per_group_splines: bool = None,
        use_log2: bool = True
    ):
        """
        Step 2 of sum factor adjustment: Remove cis gene contribution.

        Use AFTER fit_cis() and BEFORE fit_trans().
        Fits a spline regression: (sum_factor - baseline_ntc) ~ f(log2_x_true)
        Then removes the predicted x_true contribution from sum factors.

        This ensures trans modeling isn't confounded by cis expression levels.

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit' (default)  <-- This step
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column (typically from adjust_ntc_sum_factor)
        sum_factor_col_refit : str
            Name for refitted sum factor column to create (default: 'sum_factor_refit')
        covariates : list of str, optional
            Technical group covariates to group by for baseline NTC calculation (e.g., ['cell_line', 'lane'])
        n_knots : int
            Number of spline knots for regression (default: 5)
        degree : int
            Polynomial degree of spline pieces (default: 3)
        alpha : float
            Ridge regression regularization parameter (default: 0.1)
        use_per_group_splines : bool, optional
            If True, fit separate splines per group and merge them in the overlap region.
            If False, use the original algorithm with NTC baseline.
            If None (default), auto-detect: use per-group splines if no NTC cells exist.
        use_log2 : bool, default True
            If True (recommended), use log2(x_true) for the spline regression.
            This typically gives better fits since the sum_factor vs x_true relationship
            is more linear in log space.

        Returns
        -------
        None
            Creates new column sum_factor_col_refit in self.meta

        Notes
        -----
        Algorithm (with NTC):
        1. Compute baseline NTC sum factor for each covariate group
        2. Compute leftover = sum_factor - baseline_ntc
        3. Fit spline model: leftover ~ f(log2_x_true)
        4. Predict x_true contribution: y_pred = model(log2_x_true)
        5. Adjusted sum factor = max(0, leftover - y_pred + baseline_ntc)

        Algorithm (without NTC, per-group splines):
        1. Fit separate spline: sum_factor ~ f(log2_x_true) for each group
        2. Choose reference group (group with widest x_true range)
        3. For other groups, compute offset to align with reference in overlap region
        4. Alignment is weighted by density (KDE) in overlap region
        5. Apply offsets and compute residuals from aligned splines

        Requires:
        - self.x_true (or self.log2_x_true) must be set (from fit_cis())
        - self.x_true must be set (shape [N], always a point estimate)
        """
        from scipy.stats import gaussian_kde

        primary_mod = self.get_modality(self.primary_modality)

        # Read sum factor from modality sum_factors, falling back to meta for the
        # initial 'sum_factor' column that was present at model initialisation.
        if (primary_mod.sum_factors is not None
                and sum_factor_col_old in primary_mod.sum_factors.columns):
            sum_factor_data = primary_mod.sum_factors.loc[
                self.meta['cell'].values, sum_factor_col_old
            ].values.astype(float)  # shape (N,)
        elif sum_factor_col_old in self.meta.columns:
            sum_factor_data = self.meta[sum_factor_col_old].values.astype(float)
        else:
            raise ValueError(
                f"No column '{sum_factor_col_old}' found in modality sum_factors or meta."
            )

        if covariates is None:
            covariates = []

        if covariates:
            # Create a single group identifier by concatenating covariate values
            tech_group = self.meta[covariates].astype(str).agg('_'.join, axis=1)
            groups, group_id = np.unique(tech_group, return_inverse=True)
            n_groups = len(groups)
        else:
            # No grouping, treat all samples as a single group
            groups, group_id = np.array(["all"]), np.zeros(len(self.meta), dtype=int)
            n_groups = 1

        # Get x_true values (optionally in log2 space); x_true is always [N] point estimate
        if use_log2:
            if hasattr(self, 'log2_x_true') and self.log2_x_true is not None:
                X_true = self.log2_x_true.cpu().numpy() if hasattr(self.log2_x_true, 'cpu') else np.array(self.log2_x_true)
            else:
                x_raw = self.x_true.cpu().numpy() if hasattr(self.x_true, 'cpu') else np.array(self.x_true)
                X_true = np.log2(np.maximum(x_raw, 1e-6))
            print(f"[INFO] Using log2(x_true) for spline fitting (range: [{X_true.min():.2f}, {X_true.max():.2f}])")
        else:
            X_true = self.x_true.cpu().numpy() if hasattr(self.x_true, 'cpu') else np.array(self.x_true)
            print(f"[INFO] Using x_true (linear scale) for spline fitting (range: [{X_true.min():.2f}, {X_true.max():.2f}])")

        # Check if we have NTC cells
        has_ntc = 'target' in self.meta.columns and (self.meta['target'] == 'ntc').any()

        # Auto-detect whether to use per-group splines
        if use_per_group_splines is None:
            use_per_group_splines = not has_ntc and n_groups > 1
            if use_per_group_splines:
                print(f"[INFO] No NTC cells detected with {n_groups} groups. Using per-group spline alignment.")

        if use_per_group_splines and n_groups > 1:
            # =========================================================================
            # Per-group spline fitting with density-weighted alignment
            # =========================================================================

            # Step 1: Fit separate splines for each group
            group_models = {}
            group_x_ranges = {}

            for grp_idx, grp_name in enumerate(groups):
                mask = (group_id == grp_idx)
                x_grp = X_true[mask]
                y_grp = sum_factor_data[mask]

                if len(x_grp) < 10:
                    print(f"[WARN] Group '{grp_name}' has only {len(x_grp)} cells, skipping spline fit")
                    continue

                model = make_pipeline(
                    SplineTransformer(n_knots=n_knots, degree=degree),
                    Ridge(alpha=alpha)
                )
                model.fit(x_grp.reshape(-1, 1), y_grp)

                group_models[grp_idx] = model
                group_x_ranges[grp_idx] = (x_grp.min(), x_grp.max())

                print(f"[INFO] Fitted spline for group '{grp_name}': x_true range [{x_grp.min():.2f}, {x_grp.max():.2f}]")

            if len(group_models) < 2:
                print(f"[WARN] Only {len(group_models)} group(s) with enough cells. Falling back to global spline.")
                use_per_group_splines = False
            else:
                # Step 2: Choose reference group (widest x_true range)
                ref_grp = max(group_x_ranges.keys(), key=lambda g: group_x_ranges[g][1] - group_x_ranges[g][0])
                ref_name = groups[ref_grp]
                print(f"[INFO] Reference group: '{ref_name}' (widest x_true range)")

                # Step 3: Compute offsets for other groups
                group_offsets = {ref_grp: 0.0}

                for grp_idx in group_models:
                    if grp_idx == ref_grp:
                        continue

                    grp_name = groups[grp_idx]

                    # Find overlap region
                    ref_min, ref_max = group_x_ranges[ref_grp]
                    grp_min, grp_max = group_x_ranges[grp_idx]

                    overlap_min = max(ref_min, grp_min)
                    overlap_max = min(ref_max, grp_max)

                    if overlap_min >= overlap_max:
                        print(f"[WARN] No overlap between '{ref_name}' and '{grp_name}'. Using median offset.")
                        # Fallback: use median difference at closest points
                        ref_mask = (group_id == ref_grp)
                        grp_mask = (group_id == grp_idx)
                        offset = np.median(sum_factor_data[ref_mask]) - np.median(sum_factor_data[grp_mask])
                        group_offsets[grp_idx] = offset
                        continue

                    # Create evaluation grid in overlap region
                    x_eval = np.linspace(overlap_min, overlap_max, 100)

                    # Predict from both splines
                    y_ref = group_models[ref_grp].predict(x_eval.reshape(-1, 1))
                    y_grp = group_models[grp_idx].predict(x_eval.reshape(-1, 1))

                    # Compute density weights using KDE
                    ref_mask = (group_id == ref_grp)
                    grp_mask = (group_id == grp_idx)

                    x_ref_data = X_true[ref_mask]
                    x_grp_data = X_true[grp_mask]

                    # Filter to overlap region for KDE
                    x_ref_overlap = x_ref_data[(x_ref_data >= overlap_min) & (x_ref_data <= overlap_max)]
                    x_grp_overlap = x_grp_data[(x_grp_data >= overlap_min) & (x_grp_data <= overlap_max)]

                    if len(x_ref_overlap) < 5 or len(x_grp_overlap) < 5:
                        print(f"[WARN] Sparse overlap between '{ref_name}' and '{grp_name}'. Using uniform weights.")
                        weights = np.ones(len(x_eval))
                    else:
                        # KDE for both groups
                        try:
                            kde_ref = gaussian_kde(x_ref_overlap)
                            kde_grp = gaussian_kde(x_grp_overlap)

                            # Combined density (geometric mean)
                            density_ref = kde_ref(x_eval)
                            density_grp = kde_grp(x_eval)
                            weights = np.sqrt(density_ref * density_grp)
                            weights = weights / weights.sum()  # Normalize
                        except Exception as e:
                            print(f"[WARN] KDE failed for '{grp_name}': {e}. Using uniform weights.")
                            weights = np.ones(len(x_eval)) / len(x_eval)

                    # Compute weighted offset: minimize sum(w * (y_ref - (y_grp + offset))^2)
                    # Optimal offset = sum(w * (y_ref - y_grp)) / sum(w)
                    offset = np.sum(weights * (y_ref - y_grp)) / np.sum(weights)
                    group_offsets[grp_idx] = offset

                    print(f"[INFO] Offset for group '{grp_name}': {offset:.4f} (overlap: [{overlap_min:.2f}, {overlap_max:.2f}])")

                # Step 4: Compute predictions with offsets
                y_pred = np.zeros(len(X_true))

                for grp_idx in group_models:
                    mask = (group_id == grp_idx)
                    x_grp = X_true[mask]

                    # Predict and add offset
                    pred = group_models[grp_idx].predict(x_grp.reshape(-1, 1))
                    pred_aligned = pred + group_offsets[grp_idx]
                    y_pred[mask] = pred_aligned

                # Handle any groups without models (too few cells)
                missing_mask = np.zeros(len(X_true), dtype=bool)
                for grp_idx in range(n_groups):
                    if grp_idx not in group_models:
                        missing_mask |= (group_id == grp_idx)

                if missing_mask.any():
                    # Use reference group's model for missing groups
                    y_pred[missing_mask] = group_models[ref_grp].predict(X_true[missing_mask].reshape(-1, 1))

                # Compute residuals (sum_factor - predicted trend)
                residuals = sum_factor_data - y_pred

                # The adjusted sum factor removes the x_true-dependent trend
                # We want to keep the baseline level, so add back the global mean predicted value
                global_baseline = np.mean(y_pred)
                adjusted = residuals + global_baseline

                # Clamp to a small positive floor rather than 0 to prevent mu_final=0 in
                # trans fitting (log(0) = -inf logits → NaN gradients).
                # Floor = 1% of the minimum positive sum factor in the data.
                sf_min_positive = sum_factor_data[sum_factor_data > 0].min() if (sum_factor_data > 0).any() else 1e-6
                sf_floor = 0.01 * sf_min_positive
                n_clamped = int((adjusted < sf_floor).sum())
                if n_clamped > 0:
                    print(f"[WARNING] refit_sumfactor: {n_clamped} cell(s) had adjusted sum factor "
                          f"below floor ({sf_floor:.4g}); clamped to floor. "
                          f"This can happen when the spline overpredicts the cis-gene contribution. "
                          f"These cells will have near-zero library size in trans fitting.")
                if primary_mod.sum_factors is None:
                    primary_mod.sum_factors = pd.DataFrame(index=self.meta['cell'].values)
                primary_mod.sum_factors[sum_factor_col_refit] = np.maximum(sf_floor, adjusted)

                print(f"[INFO] Created '{sum_factor_col_refit}' in modality sum_factors using per-group spline alignment.")
                return

        # =========================================================================
        # Original algorithm: NTC-based baseline subtraction
        # =========================================================================
        baseline_ntc_of_group = np.zeros(n_groups)
        if covariates:
            for grp, grp_name in enumerate(groups):
                mask_grp = (tech_group == grp_name)
                # Among that group, pick rows with target == 'ntc'
                if 'target' in self.meta.columns:
                    mask_ntc = (self.meta['target'] == 'ntc') & mask_grp
                else:
                    mask_ntc = np.zeros(len(self.meta), dtype=bool)

                # If a group has no NTC, use fallback
                if not np.any(mask_ntc):
                    baseline_ntc_of_group[grp] = 1.0  # fallback
                else:
                    baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])
        else:
            grp = 0
            if 'target' in self.meta.columns:
                mask_ntc = (self.meta['target'] == 'ntc')
            else:
                mask_ntc = np.zeros(len(self.meta), dtype=bool)

            if not np.any(mask_ntc):
                baseline_ntc_of_group[grp] = 1.0  # fallback
            else:
                baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])

        # leftover_data[i] = sum_factor[i] - baseline_ntc_of_group[group_id[i]]
        leftover_data = sum_factor_data - baseline_ntc_of_group[group_id]

        # Build spline + ridge pipeline
        model_spline_ridge = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree),
            Ridge(alpha=alpha)
        )

        # Fit the model
        model_spline_ridge.fit(X_true.reshape(-1, 1), leftover_data)

        # Predict and adjust
        y_pred = model_spline_ridge.predict(X_true.reshape(-1, 1))
        adjusted = leftover_data - y_pred + baseline_ntc_of_group[group_id]

        # Clamp to a small positive floor rather than 0 to prevent mu_final=0 in
        # trans fitting (log(0) = -inf logits → NaN gradients).
        # Floor = 1% of the minimum positive sum factor in the data.
        sf_min_positive = sum_factor_data[sum_factor_data > 0].min() if (sum_factor_data > 0).any() else 1e-6
        sf_floor = 0.01 * sf_min_positive
        n_clamped = int((adjusted < sf_floor).sum())
        if n_clamped > 0:
            print(f"[WARNING] refit_sumfactor: {n_clamped} cell(s) had adjusted sum factor "
                  f"below floor ({sf_floor:.4g}); clamped to floor. "
                  f"This can happen when the spline overpredicts the cis-gene contribution. "
                  f"These cells will have near-zero library size in trans fitting.")
        if primary_mod.sum_factors is None:
            primary_mod.sum_factors = pd.DataFrame(index=self.meta['cell'].values)
        primary_mod.sum_factors[sum_factor_col_refit] = np.maximum(sf_floor, adjusted)
        print(f"[INFO] Created '{sum_factor_col_refit}' in modality sum_factors with x_true-based adjustment.")

    def permute_x_true(
        self,
        covariates: list[str] = None,
        sum_factor_col: str = 'sum_factor_adj',
    ):
        """
        Resample ``x_true`` and cis counts among NTC cells to break residual
        cis correlation before trans fitting.

        Operates only on NTC cells: within each covariate group, draws a
        bootstrap resample of the NTC indices (with replacement) and applies
        it simultaneously to ``self.x_true`` and the ``'cis'`` modality counts,
        keeping them in sync.

        Call *after* ``fit_cis()`` and ``adjust_ntc_sum_factor()``, and
        *before* ``fit_trans()``.  Typically paired with ``permute_from_ntc``
        on the trans modality::

            from bayesDREAM.simulation import permute_from_ntc
            permute_from_ntc(model.get_modality('gene'), model.meta,
                             covariates=['cell_line'])
            model.permute_x_true(covariates=['cell_line'])

        Parameters
        ----------
        covariates : list of str, optional
            Columns in ``meta`` used to stratify permutation.  If ``None``,
            all cells are treated as one group.
        sum_factor_col : str, default ``'sum_factor_adj'``
            Column in the primary modality's ``sum_factors`` used to
            normalise cis counts before resampling.
        """
        cis_mod = self.get_modality('cis') if 'cis' in self.modalities else None
        if cis_mod is None or self.x_true is None:
            raise ValueError(
                "permute_x_true requires a fitted 'cis' modality and x_true. "
                "Run fit_cis() first."
            )

        primary_mod = self.get_modality(self.primary_modality)
        if primary_mod.sum_factors is None or sum_factor_col not in primary_mod.sum_factors.columns:
            raise ValueError(
                f"No column '{sum_factor_col}' in primary modality sum_factors. "
                "Run adjust_ntc_sum_factor() first."
            )

        cis_cell_to_col = {cell: i for i, cell in enumerate(cis_mod.cell_names)}
        cis_counts_work = cis_mod.counts.copy()
        cis_is_sparse = sparse.issparse(cis_counts_work)
        cis_row = 0  # cis modality has exactly 1 feature row

        if isinstance(self.x_true, torch.Tensor):
            x_true_np = self.x_true.detach().cpu().numpy().copy()
        else:
            x_true_np = np.array(self.x_true)

        groups = self.meta.groupby(covariates) if covariates else [(None, self.meta)]
        for _key, group in groups:
            ntc_cells = group.loc[group['target'] == 'ntc', 'cell'].values
            if len(ntc_cells) == 0:
                continue

            ntc_col_idx = [cis_cell_to_col[c] for c in ntc_cells if c in cis_cell_to_col]
            if not ntc_col_idx:
                continue

            sf_ntc = primary_mod.sum_factors.loc[ntc_cells, sum_factor_col].values

            if cis_is_sparse:
                ntc_expr = np.asarray(
                    cis_counts_work[cis_row, ntc_col_idx].todense()
                ).flatten().astype(float)
            else:
                ntc_expr = np.asarray(cis_counts_work)[cis_row, ntc_col_idx].astype(float)

            perm_idx = np.random.choice(len(ntc_cells), size=len(ntc_cells), replace=True)
            new_cis = np.round((ntc_expr / np.maximum(sf_ntc, 1e-12))[perm_idx] * sf_ntc)

            if cis_is_sparse:
                cis_counts_work = cis_counts_work.tolil()
                for i, col in enumerate(ntc_col_idx):
                    cis_counts_work[cis_row, col] = new_cis[i]
                cis_counts_work = cis_counts_work.tocsr()
            else:
                cis_counts_work = np.asarray(cis_counts_work)
                cis_counts_work[cis_row, ntc_col_idx] = new_cis

            meta_ntc_idx = self.meta.index[self.meta['cell'].isin(ntc_cells)].tolist()
            x_true_np[meta_ntc_idx] = x_true_np[meta_ntc_idx][perm_idx]

        cis_mod.counts = cis_counts_work
        self.x_true = torch.tensor(x_true_np, dtype=self.x_true.dtype, device=self.x_true.device)

    def subset_cells(
        self,
        cell_mask: Union[np.ndarray, pd.Series, list] = None,
        query: str = None,
        preserve_fits: bool = True
    ):
        """
        Create a new model instance with a subset of cells.

        Useful for testing without technical correction by subsetting to a single
        technical group (e.g., one cell line only).

        Parameters
        ----------
        cell_mask : np.ndarray, pd.Series, or list, optional
            Boolean mask or list of cell names to keep. If None, must provide query.
        query : str, optional
            Pandas query string to filter cells (e.g., "cell_line == 'K562'").
            Applied to self.meta. If None, must provide cell_mask.
        preserve_fits : bool
            If True (default), copy fitted parameters (alpha_x_prefit, alpha_y_prefit,
            x_true, etc.) to the new model. Set to False to start fresh with the subset.

        Returns
        -------
        bayesDREAM
            New model instance with subsetted cells

        Examples
        --------
        # Subset by query
        model_k562 = model.subset_cells(query="cell_line == 'K562'")

        # Subset by mask
        mask = model.meta['cell_line'].str.contains('K562')
        model_k562 = model.subset_cells(cell_mask=mask)

        # Subset by cell list
        cells = model.meta[model.meta['cell_line'] == 'K562']['cell'].tolist()
        model_k562 = model.subset_cells(cell_mask=cells)
        """
        if cell_mask is None and query is None:
            raise ValueError("Must provide either cell_mask or query")

        if query is not None:
            # Filter meta using query
            meta_subset = self.meta.query(query).copy()
            cells_to_keep = meta_subset['cell'].values
        elif isinstance(cell_mask, (list, np.ndarray, pd.Series)):
            # Handle list of cell names
            if isinstance(cell_mask, list) and len(cell_mask) > 0 and isinstance(cell_mask[0], str):
                cells_to_keep = cell_mask
                meta_subset = self.meta[self.meta['cell'].isin(cells_to_keep)].copy()
            # Handle boolean mask
            elif isinstance(cell_mask, (np.ndarray, pd.Series)):
                if cell_mask.dtype == bool:
                    meta_subset = self.meta[cell_mask].copy()
                    cells_to_keep = meta_subset['cell'].values
                else:
                    # Assume it's a list of cell names
                    cells_to_keep = cell_mask
                    meta_subset = self.meta[self.meta['cell'].isin(cells_to_keep)].copy()
            else:
                raise ValueError("cell_mask must be boolean array/Series or list of cell names")
        else:
            raise ValueError("Invalid cell_mask type")

        if len(meta_subset) == 0:
            raise ValueError("Subset resulted in zero cells")

        print(f"[INFO] Subsetting from {len(self.meta)} to {len(meta_subset)} cells")

        # Subset counts — reconstruct from modalities (self.counts is None after init)
        cells_to_keep_set = set(cells_to_keep)
        gene_mod = self.get_modality(self.primary_modality)
        cis_mod  = self.get_modality('cis') if 'cis' in self.modalities else None

        # Get column indices in the gene modality matching cells_to_keep
        if gene_mod.cell_names:
            gene_cell_indices = [i for i, c in enumerate(gene_mod.cell_names) if c in cells_to_keep_set]
        else:
            gene_cell_indices = [i for i, cell in enumerate(self.meta['cell']) if cell in cells_to_keep_set]

        gene_counts_sub  = gene_mod.counts[:, gene_cell_indices]
        gene_feat_meta   = gene_mod.feature_meta.copy() if gene_mod.feature_meta is not None else None

        if cis_mod is not None:
            if cis_mod.cell_names:
                cis_cell_indices = [i for i, c in enumerate(cis_mod.cell_names) if c in cells_to_keep_set]
            else:
                cis_cell_indices = gene_cell_indices

            cis_counts_row = cis_mod.counts[:, cis_cell_indices]   # shape [1, N_sub]
            cis_feat_meta  = cis_mod.feature_meta.copy()

            # Stack: cis row first (row 0), then gene rows
            if sparse.issparse(gene_counts_sub):
                cis_sp       = (cis_counts_row.tocsr() if sparse.issparse(cis_counts_row)
                                else sparse.csr_matrix(np.asarray(cis_counts_row).reshape(1, -1)))
                counts_subset = sparse.vstack([cis_sp, gene_counts_sub], format='csr')
            else:
                cis_dense     = np.asarray(cis_counts_row).reshape(1, -1)
                counts_subset = np.vstack([cis_dense, np.asarray(gene_counts_sub)])

            # Build combined feature_meta with sequential 0-based index
            # (constructor's _extract_cis_from_gene searches this by value, not position)
            cis_fm  = cis_feat_meta.copy();  cis_fm.index  = [0]
            if gene_feat_meta is not None:
                gene_fm = gene_feat_meta.copy(); gene_fm.index = range(1, 1 + len(gene_feat_meta))
                feature_meta_subset = pd.concat([cis_fm, gene_fm], ignore_index=True)
            else:
                feature_meta_subset = cis_fm
        else:
            counts_subset       = gene_counts_sub
            feature_meta_subset = gene_feat_meta

        # Subset high MOI guide_assignment if applicable
        guide_assignment_subset = None
        if self.is_high_moi:
            # Get indices of subsetted cells in original meta
            cell_indices = [i for i, cell in enumerate(self.meta['cell']) if cell in cells_to_keep]
            guide_assignment_subset = self.guide_assignment[cell_indices, :]

        # Create new model instance
        from .model import bayesDREAM

        # For high MOI mode, reconstruct guide_target DataFrame from guide_targets_dict
        # This preserves many-to-many guide-target relationships
        guide_target_df = None
        if self.is_high_moi and hasattr(self, 'guide_targets_dict') and self.guide_targets_dict:
            guide_target_rows = []
            for guide, targets in self.guide_targets_dict.items():
                for target in targets:
                    guide_target_rows.append({'guide': guide, 'target': target})
            if guide_target_rows:
                guide_target_df = pd.DataFrame(guide_target_rows)

        model_new = bayesDREAM(
            meta=meta_subset,
            counts=counts_subset,
            modality_name=self.primary_modality,  # Use same primary modality name
            feature_meta=feature_meta_subset,
            cis_gene=self.cis_gene,
            guide_covariates=self.guide_covariates,  # Preserve guide covariates
            guide_covariates_ntc=self.guide_covariates_ntc,  # Preserve NTC guide covariates
            output_dir=self.output_dir,
            label=f"{self.label}_subset",
            device=str(self.device),
            random_seed=2402,
            cores=1,
            guide_assignment=guide_assignment_subset,
            guide_meta=self.guide_meta.copy() if self.is_high_moi else None,
            guide_target=guide_target_df,  # Preserve many-to-many guide-target relationships
            require_ntc=False  # Allow subsetting without NTC cells
        )


        # Copy additional modalities (beyond the primary 'gene' modality)
        if hasattr(self, 'modalities'):
            for mod_name, modality in self.modalities.items():
                if mod_name not in ['gene', 'cis']:  # Skip primary and cis modalities (already handled)
                    # Subset the modality to the selected cells using cell names
                    subset_modality = modality.get_cell_subset(cells_to_keep)
                    model_new.modalities[mod_name] = subset_modality
            print(f"[INFO] Copied {len(self.modalities) - 2} additional modalities to subset model")

        # Optionally preserve fitted parameters
        if preserve_fits:
            # Copy technical fit parameters if they exist
            if self.alpha_x_prefit is not None:
                model_new.alpha_x_prefit = self.alpha_x_prefit.clone() if isinstance(self.alpha_x_prefit, torch.Tensor) else self.alpha_x_prefit

            # Copy all fit attributes from original modalities to new modalities
            # (including 'gene' and 'cis' which were recreated during initialization)
            # NOTE: alpha_y_prefit is stored per-modality as a [C, T] mean point estimate

            # Helper to clone tensors
            def _clone_attr(val):
                if hasattr(val, 'clone'):
                    return val.clone()
                return val

            # Modality-level attributes to copy from fit_technical and fit_trans
            # NOTE: alpha_y_prefit is a property, not an attribute - it derives from _mult/_add
            modality_attrs = [
                'alpha_y_prefit_mult', 'alpha_y_prefit_add',
                'posterior_samples_ntc', 'posterior_samples_trans', 'losses_trans'
            ]
            for mod_name in self.modalities:
                if mod_name in model_new.modalities:
                    orig_mod = self.modalities[mod_name]
                    new_mod = model_new.modalities[mod_name]
                    copied_attrs = []
                    for attr in modality_attrs:
                        orig_val = getattr(orig_mod, attr, None)
                        if orig_val is not None:
                            setattr(new_mod, attr, _clone_attr(orig_val))
                            copied_attrs.append(attr)
                    if copied_attrs:
                        print(f"[INFO] Copied {copied_attrs} from '{mod_name}' modality")
                    else:
                        print(f"[DEBUG] No attributes to copy from '{mod_name}' modality")
                        # Debug: show which attrs exist
                        for attr in modality_attrs:
                            val = getattr(orig_mod, attr, "MISSING")
                            print(f"  {attr}: {type(val).__name__ if val != 'MISSING' else 'MISSING'}")
                else:
                    print(f"[WARN] Modality '{mod_name}' not found in new model")

            # Copy cis fit parameters if they exist
            # Get cell indices for subsetting cell-indexed tensors
            cell_indices_torch = torch.tensor(
                [i for i, cell in enumerate(self.meta['cell']) if cell in cells_to_keep],
                dtype=torch.long
            )

            if hasattr(self, 'x_true') and self.x_true is not None:
                # x_true is always shape [N] - subset to new cells
                if isinstance(self.x_true, torch.Tensor):
                    model_new.x_true = self.x_true[cell_indices_torch].clone()

            # Copy log2_x_true (also needs subsetting)
            if hasattr(self, 'log2_x_true') and self.log2_x_true is not None:
                if isinstance(self.log2_x_true, torch.Tensor):
                    model_new.log2_x_true = self.log2_x_true[cell_indices_torch].clone()

            # Copy posterior_samples_cis with cell-indexed tensors subsetted
            if hasattr(self, 'posterior_samples_cis') and self.posterior_samples_cis is not None:
                n_cells_orig = len(self.meta)
                subsetted_cis = {}
                for key, val in self.posterior_samples_cis.items():
                    if isinstance(val, torch.Tensor) and val.shape[-1] == n_cells_orig:
                        # This tensor has cell dimension - subset it
                        subsetted_cis[key] = val[..., cell_indices_torch].clone()
                    elif isinstance(val, torch.Tensor):
                        subsetted_cis[key] = val.clone()
                    else:
                        subsetted_cis[key] = val
                model_new.posterior_samples_cis = subsetted_cis
            if hasattr(self, 'loss_x') and self.loss_x is not None:
                model_new.loss_x = self.loss_x
            if hasattr(self, 'posterior_samples_trans') and self.posterior_samples_trans is not None:
                model_new.posterior_samples_trans = self.posterior_samples_trans
            if hasattr(self, 'losses_trans') and self.losses_trans is not None:
                model_new.losses_trans = self.losses_trans

            # Copy traces if they exist
            if self.trace_cellline is not None:
                model_new.trace_cellline = self.trace_cellline
            if self.trace_x is not None:
                model_new.trace_x = self.trace_x
            if self.trace_y is not None:
                model_new.trace_y = self.trace_y

            print("[INFO] Preserved fitted parameters in subset model")

        # Unconditionally copy sum_factors to all modalities that have them.
        # _init_sum_factors already seeded primary/cis from meta columns, but
        # dynamically-added columns (sum_factor_adj, sum_factor_refit, …) live only
        # on the modality object and must be transferred here.
        # get_cell_subset() for non-primary modalities leaves sum_factors=None,
        # so this also fixes those.
        cells_to_keep_index = set(cells_to_keep)
        for mod_name, orig_mod in self.modalities.items():
            if orig_mod.sum_factors is None:
                continue
            if mod_name not in model_new.modalities:
                continue
            keep_idx = orig_mod.sum_factors.index.intersection(list(cells_to_keep_index))
            if len(keep_idx) == 0:
                continue
            model_new.modalities[mod_name].sum_factors = orig_mod.sum_factors.loc[keep_idx].copy()
        # cis always shares the same DataFrame object as primary
        if ('cis' in model_new.modalities
                and model_new.primary_modality in model_new.modalities):
            model_new.modalities['cis'].sum_factors = (
                model_new.modalities[model_new.primary_modality].sum_factors
            )

        return model_new


    # ========================================================================
    # Delegation methods to fitters
    # ========================================================================

    def _model_ntc(self, *args, **kwargs):
        """Delegate to NTCFitter."""
        return self._ntc_fitter._model_ntc(*args, **kwargs)

    @functools.wraps(NTCFitter.set_technical_groups)
    def set_technical_groups(self, *args, **kwargs):
        return self._ntc_fitter.set_technical_groups(*args, **kwargs)

    @functools.wraps(NTCFitter.fit_ntc)
    def fit_ntc(self, *args, **kwargs):
        return self._ntc_fitter.fit_ntc(*args, **kwargs)

    def _model_x(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter._model_x(*args, **kwargs)

    def fit_cis(self, *args, force: bool = False, **kwargs):
        # Docstring built programmatically below the class to stay DRY.
        if not force and hasattr(self, '_compute_ntc_log2_exprs_from_fit'):
            data = self._compute_ntc_log2_exprs_from_fit()
            if data is not None:
                cis_log2 = data['cis_log2_expr']
                if cis_log2 < -1:
                    raise ValueError(
                        f"Cis gene '{self.cis_gene}' has low NTC expression "
                        f"(log2 = {cis_log2:.2f} < -1). "
                        f"Overdispersion estimated from near-zero counts may be unreliable. "
                        f"Call plot_ntc_expression() to inspect, or pass force=True to proceed anyway."
                    )
        return self._cis_fitter.fit_cis(*args, **kwargs)

    def _model_y(self, *args, **kwargs):
        """Delegate to TransFitter."""
        return self._trans_fitter._model_y(*args, **kwargs)

    @functools.wraps(TransFitter.fit_trans)
    def fit_trans(self, *args, **kwargs):
        return self._trans_fitter.fit_trans(*args, **kwargs)

    @functools.wraps(ModelSaver.save_ntc_fit)
    def save_ntc_fit(self, *args, **kwargs):
        return self._saver.save_ntc_fit(*args, **kwargs)

    @functools.wraps(ModelSaver.save_cis_fit)
    def save_cis_fit(self, *args, **kwargs):
        return self._saver.save_cis_fit(*args, **kwargs)

    @functools.wraps(ModelSaver.save_trans_fit)
    def save_trans_fit(self, *args, **kwargs):
        return self._saver.save_trans_fit(*args, **kwargs)

    @functools.wraps(ModelLoader.load_ntc_fit)
    def load_ntc_fit(self, *args, **kwargs):
        return self._loader.load_ntc_fit(*args, **kwargs)

    @functools.wraps(ModelLoader.load_cis_fit)
    def load_cis_fit(self, *args, **kwargs):
        return self._loader.load_cis_fit(*args, **kwargs)

    @functools.wraps(ModelLoader.load_trans_fit)
    def load_trans_fit(self, *args, **kwargs):
        return self._loader.load_trans_fit(*args, **kwargs)

    @functools.wraps(ModelSummarizer.save_ntc_summary)
    def save_ntc_summary(self, *args, **kwargs):
        return self._summarizer.save_ntc_summary(*args, **kwargs)

    @functools.wraps(ModelSummarizer.save_cis_summary)
    def save_cis_summary(self, *args, **kwargs):
        return self._summarizer.save_cis_summary(*args, **kwargs)

    @functools.wraps(ModelSummarizer.save_trans_summary)
    def save_trans_summary(self, *args, **kwargs):
        return self._summarizer.save_trans_summary(*args, **kwargs)

    @functools.wraps(ModelSummarizer.classify_second_deriv_roots)
    def classify_second_deriv_roots(self, *args, **kwargs):
        return self._summarizer.classify_second_deriv_roots(*args, **kwargs)


# ---------------------------------------------------------------------------
# Build fit_cis docstring from CisFitter, injecting the wrapper-level `force`
# parameter so the single source of truth stays in CisFitter.fit_cis.
# ---------------------------------------------------------------------------
_force_doc = """\
        force : bool, default False
            If True, skip the low-expression check and proceed regardless.
            Use plot_ntc_expression() to inspect the expression distribution
            before overriding.
"""
_base = CisFitter.fit_cis.__doc__ or ""
_BayesDREAMCore.fit_cis.__doc__ = _base.replace(
    "Parameters\n        ----------",
    "Parameters\n        ----------\n" + _force_doc,
    1,
)
