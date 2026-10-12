"""Input normalization and cell metadata preparation for the core model."""

import os

import numpy as np
import pandas as pd
from scipy import sparse

from ..guides import classify_target_from_guides


def validate_and_classify_metadata(
    self,
    counts,
    sum_factor_col,
    guide_covariates,
    guide_covariates_ntc,
    n_cells_assignment,
    exclude_targets,
    exclude_guides,
    require_ntc,
):
    """Validate cell metadata and assign target classes before filtering."""
    if self.is_high_moi:
        required_cols = {"cell", sum_factor_col} | set(guide_covariates) | set(guide_covariates_ntc)
        missing_cols = required_cols - set(self.meta.columns)
        if missing_cols:
            raise ValueError(f"[High MOI] Missing required columns in meta: {missing_cols}")
        if len(self.meta) != n_cells_assignment:
            raise ValueError(
                f"[High MOI] guide_assignment has {n_cells_assignment} rows but meta has {len(self.meta)} rows. "
                "These dimensions must match."
            )
    else:
        required_cols = {"cell", sum_factor_col, "guide"} | set(guide_covariates) | set(guide_covariates_ntc)
        if self.guide_targets_dict is None:
            required_cols.add("target")
        missing_cols = required_cols - set(self.meta.columns)
        if missing_cols:
            raise ValueError(f"[Single-guide] Missing required columns in meta: {missing_cols}")

        if self.guide_targets_dict is not None:
            self.meta['target'] = self.meta['guide'].map(
                lambda guide_name: classify_target_from_guides(
                    [guide_name], self.guide_targets_dict, self.cis_gene,
                    exclude_targets=exclude_targets, exclude_guides=exclude_guides,
                )
            )
            n_before = len(self.meta)
            self.meta = self.meta[self.meta['target'] != 'excluded'].copy()
            n_excluded = n_before - len(self.meta)
            if n_excluded > 0:
                print(f"[INFO] Excluded {n_excluded} cells (guide targets a gene in exclude_targets={exclude_targets})")

            target_counts = self.meta['target'].value_counts()
            print("[INFO] Single-guide target classification from guide_target mapping:")
            print(f"  NTC cells: {target_counts.get('ntc', 0)}")
            if self.cis_gene is not None:
                print(f"  {self.cis_gene}-targeting cells: {target_counts.get(self.cis_gene, 0)}")
                print(f"  Other-only cells (will be removed): {target_counts.get('other', 0)}")
            else:
                print(
                    "  Other/unclassified cells (cis_gene deferred — target unknown until add_cis_gene()): "
                    f"{target_counts.get('other', 0)}"
                )

        if require_ntc and "ntc" not in self.meta["target"].values:
            raise ValueError(
                "No NTC detected in the 'target' column. "
                "If this is correct (e.g., you have already run fit_ntc() or don't need NTC cells), "
                "use require_ntc=False."
            )

    if self._cell_names is None:
        self._cell_names = counts.columns.tolist() if isinstance(counts, pd.DataFrame) else self.meta['cell'].tolist()
    if not set(self.meta["cell"]).issubset(set(self._cell_names)):
        raise ValueError("The 'cell' column in meta must correspond 1:1 with the cell names in counts.")
    if (self.meta[sum_factor_col] <= 0).any():
        raise ValueError(f"All values in sum_factor_col={sum_factor_col} column must be strictly greater than 0.")

    if self.is_high_moi:
        guide_names = self.guide_meta['guide'].tolist()
        targets = [
            classify_target_from_guides(
                [guide_names[index] for index, assigned in enumerate(assignment) if assigned],
                self.guide_targets_dict, self.cis_gene,
                exclude_targets=exclude_targets, exclude_guides=exclude_guides,
            )
            for assignment in self.guide_assignment
        ]
        self.meta['target'] = targets
        self.meta['guide_code'] = -1
        ntc_count = (np.array(targets) == 'ntc').sum()
        other_count = (np.array(targets) == 'other').sum()
        excluded_count = (np.array(targets) == 'excluded').sum()
        print("[INFO] Cell classification before subsetting:")
        print(f"  NTC cells (NTC guides, no cis): {ntc_count}")
        if self.cis_gene is not None:
            cis_count = (np.array(targets) == self.cis_gene).sum()
            print(f"  {self.cis_gene}-targeting cells (any cis guides): {cis_count}")
            print(f"  Other-only cells (will be removed): {other_count}")
        else:
            print(f"  Other/unclassified cells (cis_gene deferred — target unknown until add_cis_gene()): {other_count}")
        if exclude_targets is not None or exclude_guides is not None:
            print(f"  Excluded cells (exclude_targets/exclude_guides): {excluded_count}")


def prepare_cell_subset(self, exclude_targets, exclude_guides):
    """Apply exclusions and subset metadata and counts to the active cells."""
    original_cell_names = list(self._cell_names) if self.is_high_moi else []
    if exclude_guides is not None and not self.is_high_moi:
        n_before = len(self.meta)
        self.meta = self.meta[~self.meta['guide'].isin(set(exclude_guides))].copy()
        if len(self.meta) < n_before:
            print(f"[INFO] Excluded {n_before - len(self.meta)} cells carrying guides in exclude_guides={exclude_guides}")

    if self.is_high_moi:
        n_before = len(self.meta)
        self.meta = self.meta[self.meta["target"] != "excluded"].copy()
        if len(self.meta) < n_before:
            print(f"[INFO] Excluded {n_before - len(self.meta)} cells (target='excluded')")

    if self.cis_gene is not None:
        valid_cells = self.meta[self.meta["target"].isin(["ntc", self.cis_gene])]["cell"].unique()
        n_cells_before = len(self.meta["cell"].unique())
        if len(valid_cells) < n_cells_before:
            print(f"[INFO] Cells: {n_cells_before} → {len(valid_cells)} (kept NTC + {self.cis_gene} only)")
        self.meta = self.meta[self.meta["cell"].isin(valid_cells)].copy()
    else:
        valid_cells = self.meta["cell"].unique()
        print(f"[INFO] No cis_gene at init — keeping all {len(valid_cells)} cells. "
              "Call add_cis_gene() before fit_cis().")

    if isinstance(self.counts, pd.DataFrame):
        self.counts = self.counts[valid_cells].copy()
    else:
        valid_cells_set = set(valid_cells)
        cell_indices = [i for i, cell in enumerate(self._cell_names) if cell in valid_cells_set]
        self.counts = self.counts[:, cell_indices]
        self._cell_names = [self._cell_names[i] for i in cell_indices]
    return original_cell_names, valid_cells


def initialize_input_state(self, meta, counts, cis_gene, output_dir, label, require_ntc):
    if label is None and cis_gene is not None:
        label = cis_gene
    elif label is None:
        raise ValueError(
            "label must be provided when cis_gene is not specified at initialization. "
            "When cis_gene is provided, label defaults to the gene name."
        )

    self.meta = meta.copy()
    self.is_sparse_counts = sparse.issparse(counts)
    if self.is_sparse_counts:
        self.counts = counts.copy() if hasattr(counts, 'copy') else counts.tocsr()
        self._cell_names = counts.columns.tolist() if isinstance(counts, pd.DataFrame) else None
    elif isinstance(counts, pd.DataFrame):
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