"""
ATAC-seq modality methods for bayesDREAM.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union

from ..modality import Modality



class ATACModalityMixin:
    """Mixin for ATAC-seq modality support."""
    
    def add_atac_modality(
        self,
        atac_counts: Union[np.ndarray, pd.DataFrame],
        region_meta: pd.DataFrame,
        name: str = 'atac',
        cis_region: Optional[str] = None,
        cell_names: Optional[List[str]] = None,
        overwrite: bool = False
    ):
        """
        Add ATAC-seq modality with genomic region annotations.

        ATAC fragment counts are treated as negative binomial data (like gene expression).
        Regions can be promoters, gene bodies, or distal elements (enhancers).

        **IMPORTANT**: This method does NOT extract a 'cis' modality by default.
        The 'cis' modality is only created during bayesDREAM() initialization.
        However, if cis_region is specified AND no 'cis' modality exists yet,
        this method will create one from the specified ATAC region.

        Parameters
        ----------
        atac_counts : array or DataFrame
            Fragment counts per region per cell. Shape: (n_regions, n_cells)
            If DataFrame, cell names come from columns.
            If ndarray, use cell_names parameter to specify cell identifiers.
        region_meta : pd.DataFrame
            Region metadata with required columns:
            - region_id : str, unique region identifier
            - region_type : str, one of ['promoter', 'gene_body', 'distal']
            - chrom : str, chromosome
            - start : int, start coordinate (0-based)
            - end : int, end coordinate
            - gene : str, associated gene (NA for distal regions)
            Optional columns: gene_name, gene_id, strand, tss_distance
        name : str, optional
            Name for this ATAC modality (default: 'atac')
        cis_region : str, optional
            Region ID to use as cis proxy (e.g., 'chr9:132283881-132284881').
            If provided AND no 'cis' modality exists, will create 'cis' modality from this region.
            If 'cis' modality already exists, this parameter is ignored with a warning.
        cell_names : list of str, optional
            Cell names/identifiers (only used when atac_counts is ndarray, not DataFrame).
            Should match number of cells (axis 1 for 2D data).
        overwrite : bool, default=False
            Whether to overwrite existing modality with the same name

        Examples
        --------
        >>> # Method 1: Gene expression primary, ATAC secondary (no cis from ATAC)
        >>> model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
        >>> model.add_atac_modality(atac_counts, region_meta)  # Just adds ATAC, no cis extraction
        >>>
        >>> # Method 2: ATAC as cis proxy (creates 'cis' from ATAC region)
        >>> model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene=None)  # No cis yet
        >>> model.add_atac_modality(atac_counts, region_meta, cis_region='chr9:132283881-132284881')
        >>> # Now 'cis' modality exists from ATAC region
        >>>
        >>> # Fit using either approach
        >>> model.fit_ntc()
        >>> model.fit_cis()
        """
        # Validate required columns
        required_cols = ['region_id', 'region_type', 'chrom', 'start', 'end', 'gene']
        missing_cols = set(required_cols) - set(region_meta.columns)
        if missing_cols:
            raise ValueError(
                f"region_meta missing required columns: {missing_cols}\n"
                f"Required: {required_cols}"
            )

        # Validate region_type values
        valid_types = {'promoter', 'gene_body', 'distal'}
        invalid_types = set(region_meta['region_type'].unique()) - valid_types
        if invalid_types:
            raise ValueError(
                f"Invalid region_type values: {invalid_types}\n"
                f"Must be one of: {valid_types}"
            )

        # Validate gene column: should be non-null for promoter/gene_body
        for region_type in ['promoter', 'gene_body']:
            type_mask = region_meta['region_type'] == region_type
            if type_mask.any():
                null_genes = region_meta[type_mask]['gene'].isnull().sum()
                if null_genes > 0:
                    raise ValueError(
                        f"Found {null_genes} {region_type} regions with null gene annotations.\n"
                        f"All {region_type} regions must have an associated gene."
                    )

        # Set region_id as index if not already
        if 'region_id' in region_meta.columns and region_meta.index.name != 'region_id':
            region_meta = region_meta.set_index('region_id')

        # Check if we should create 'cis' modality from cis_region
        create_cis = False
        if cis_region is not None:
            if cis_region not in region_meta.index:
                raise ValueError(
                    f"cis_region '{cis_region}' not found in region_meta.index.\n"
                    f"Available regions: {region_meta.index[:5].tolist()}..."
                )
            if 'cis' in self.modalities:
                warnings.warn(
                    f"cis_region '{cis_region}' specified but 'cis' modality already exists. "
                    f"Ignoring cis_region parameter. The 'cis' modality is set during bayesDREAM initialization "
                    f"and cannot be changed by add_*_modality() methods.",
                    UserWarning
                )
            else:
                print(f"[INFO] Creating 'cis' modality from ATAC region: {cis_region}")
                create_cis = True

        # Convert to array if DataFrame
        if isinstance(atac_counts, pd.DataFrame):
            counts_array = atac_counts.values
            counts_index = atac_counts.index
            is_dataframe = True
            # Extract cell names from DataFrame columns
            extracted_cell_names = atac_counts.columns.tolist()
        else:
            counts_array = np.asarray(atac_counts)
            counts_index = None
            is_dataframe = False
            # Use provided cell_names (can be None)
            extracted_cell_names = cell_names

        # Apply negbinom filtering (zero std)
        feature_stds = counts_array.std(axis=1)
        valid_features = feature_stds != 0
        num_filtered = (~valid_features).sum()

        if num_filtered > 0:
            print(f"[INFO] Filtering {num_filtered} ATAC region(s) with zero std in '{name}' modality")
            counts_array = counts_array[valid_features, :]
            region_meta = region_meta.iloc[valid_features].copy()

        if counts_array.shape[0] == 0:
            raise ValueError(f"No ATAC regions left after filtering in '{name}' modality!")

        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            counts_final = pd.DataFrame(
                counts_array,
                index=region_meta.index,
                columns=atac_counts.columns
            )
        else:
            counts_final = counts_array

        # Create modality
        modality = Modality(
            name=name,
            counts=counts_final,
            feature_meta=region_meta,
            distribution='negbinom',  # ATAC uses negbinom like gene expression
            cells_axis=1,
            cell_names=extracted_cell_names
        )

        self.add_modality(name, modality, overwrite=overwrite)
        print(f"[INFO] Added ATAC modality '{name}' with {modality.dims['n_features']} regions")

        # Print region type summary
        region_summary = region_meta['region_type'].value_counts()
        print(f"[INFO] Region types: {region_summary.to_dict()}")

        # Create 'cis' modality from cis_region if needed
        if create_cis:
            # Extract just the cis region
            if is_dataframe:
                cis_counts_df = counts_final.loc[[cis_region]]
            else:
                # Get index of cis_region in filtered data
                cis_idx = region_meta.index.get_loc(cis_region)
                cis_counts_array = counts_array[[cis_idx], :]
                cis_counts_df = pd.DataFrame(
                    cis_counts_array,
                    index=[cis_region],
                    columns=range(cis_counts_array.shape[1])
                )

            cis_region_meta = region_meta.loc[[cis_region]].copy()

            cis_modality = Modality(
                name='cis',
                counts=cis_counts_df,
                feature_meta=cis_region_meta,
                distribution='negbinom',
                cells_axis=1,
                cell_names=extracted_cell_names
            )
            self.add_modality('cis', cis_modality, overwrite=False)
            print(f"[INFO] Created 'cis' modality with ATAC region '{cis_region}'")

