"""
Splicing data processing for bayesDREAM.

Pure Python implementation of donor/acceptor usage and exon skipping metrics.
Previously relied on R functions from CodeDump.R, now fully in Python.
"""

import warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd

from .modality import Modality


def _normalize_strand(strand_values):
    """
    Normalize strand notation to '+'/'-'.

    Accepts: 1/2 (integers), '+'/'-', 'plus'/'minus'
    Returns: '+' or '-' or None for invalid
    """
    def norm_single(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return '+' if x == 1 else ('-' if x == 2 else None)
        s = str(x).lower()
        if s in ['+', '1', 'plus']:
            return '+'
        if s in ['-', '2', 'minus']:
            return '-'
        return None

    if isinstance(strand_values, (list, pd.Series, np.ndarray)):
        return [norm_single(x) for x in strand_values]
    else:
        return norm_single(strand_values)

def _most_common(series):
    """Return most common non-null value or None."""
    s = pd.Series(series).dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()

def _build_sj_index(sj_counts: pd.DataFrame,
                    sj_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build splice junction index with donor/acceptor annotations.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
        Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)

    Returns
    -------
    pd.DataFrame
        Indexed SJ metadata with donor/acceptor positions
    """
    # Validate required columns
    required = ['coord.intron', 'chrom', 'intron_start', 'intron_end', 'strand',
                'gene_name_start', 'gene_name_end']
    missing = [c for c in required if c not in sj_meta.columns]
    if missing:
        raise ValueError(f"sj_meta missing required columns: {missing}")

    # Optional columns for Ensembl ID support
    optional = ['gene_id_start', 'gene_id_end']
    has_optional = all(c in sj_meta.columns for c in optional)
    if has_optional:
        print(f"[INFO] Found gene_id columns - will support both gene names and Ensembl IDs")

    idx = sj_meta.copy()
    idx['strand'] = _normalize_strand(idx['strand'].values)
    idx['start'] = idx['intron_start'].astype(int)
    idx['end'] = idx['intron_end'].astype(int)

    # donor/acceptor genomic positions by strand
    idx['donor']    = np.where(idx['strand'] == '+', idx['start'], idx['end'])
    idx['acceptor'] = np.where(idx['strand'] == '+', idx['end'],   idx['start'])

    # ---- NEW: per-site gene assignment by strand ----
    # donor gene is start-gene on '+' and end-gene on '-'
    idx['donor_gene_name'] = np.where(idx['strand'] == '+', idx['gene_name_start'], idx['gene_name_end'])
    idx['acceptor_gene_name'] = np.where(idx['strand'] == '+', idx['gene_name_end'], idx['gene_name_start'])

    if has_optional:
        idx['donor_gene_id'] = np.where(idx['strand'] == '+', idx['gene_id_start'], idx['gene_id_end'])
        idx['acceptor_gene_id'] = np.where(idx['strand'] == '+', idx['gene_id_end'], idx['gene_id_start'])
    else:
        idx['donor_gene_id'] = np.nan
        idx['acceptor_gene_id'] = np.nan
    # -----------------------------------------------

    idx['left'] = np.minimum(idx['start'], idx['end'])
    idx['right'] = np.maximum(idx['start'], idx['end'])

    present = sj_counts.index.tolist()
    idx = idx[idx['coord.intron'].isin(present)].copy()

    if len(idx) == 0:
        raise ValueError("No splice junctions overlap between sj_counts and sj_meta")

    return idx


def process_donor_usage(sj_counts: pd.DataFrame,
                       sj_meta: pd.DataFrame,
                       min_cell_total: int = 1) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Compute donor usage (which acceptor is used for each donor site).

    For each donor site (5' splice site), count how many reads go to each
    possible acceptor (3' splice site). Returns multinomial counts.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
        Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)
    min_cell_total : int
        Minimum total reads per donor per cell to include

    Returns
    -------
    counts_3d : np.ndarray
        Shape: (n_donors, n_cells, max_acceptors_per_donor)
    feature_meta : pd.DataFrame
        Donor site metadata with columns: chrom, strand, donor, acceptors (list), n_acceptors,
        gene (primary gene), genes (list of all genes)
    cell_names : list
        Cell identifiers
    """
    print("[INFO] Processing donor usage (5' splice sites)...")
    idx = _build_sj_index(sj_counts, sj_meta)

    # Check for gene_id columns
    has_gene_id = 'gene_id_start' in idx.columns and 'gene_id_end' in idx.columns

    # Group by (chrom, strand, donor)
    donor_groups = idx.groupby(['chrom', 'strand', 'donor'], sort=True)

    # Get cell names
    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)

    # Build feature metadata and 3D array
    feature_rows = []
    all_counts = []

    for (chrom, strand, donor), group in donor_groups:
        # Get all acceptors for this donor
        acceptors = sorted(group['acceptor'].unique())
        n_acceptors = len(acceptors)

        # Build mapping from acceptor to index
        acceptor_to_idx = {acc: i for i, acc in enumerate(acceptors)}

        # Initialize counts for this donor: (n_cells, n_acceptors)
        donor_counts = np.zeros((n_cells, n_acceptors), dtype=float)

        # Fill in counts for each junction
        for _, row in group.iterrows():
            coord = row['coord.intron']
            if coord not in sj_counts.index:
                continue
            acceptor = row['acceptor']
            acc_idx = acceptor_to_idx[acceptor]

            # Add counts for all cells
            donor_counts[:, acc_idx] += sj_counts.loc[coord].values

        # Apply min_cell_total filter
        if min_cell_total > 0:
            cell_totals = donor_counts.sum(axis=1)
            donor_counts[cell_totals < min_cell_total, :] = 0

        all_counts.append(donor_counts)

        # ---- REPLACE gene collection with site-centric assignment ----
        gene_name = _most_common(group['donor_gene_name'])
        gene_id   = _most_common(group['donor_gene_id'])
        # --------------------------------------------------------------

        feature_rows.append({
            'chrom': chrom,
            'strand': strand,
            'donor': donor,
            'acceptors': acceptors,
            'n_acceptors': n_acceptors,
            'gene_name': gene_name,
            'gene_id': gene_id
        })

    # Filter donors with only one acceptor OR with zero variance in ALL ratios
    n_before_filter = len(feature_rows)
    filtered_rows = []
    filtered_counts = []
    n_single_acceptor = 0
    n_zero_var = 0

    for i, row in enumerate(feature_rows):
        donor_counts = all_counts[i]  # Shape: (n_cells, n_acceptors)

        # Filter 1: Only one acceptor
        if row['n_acceptors'] <= 1:
            n_single_acceptor += 1
            continue

        # Filter 2: Check if ALL ratios have zero variance across cells
        # Compute total counts per cell
        totals = donor_counts.sum(axis=1, keepdims=True)  # (n_cells, 1)

        # Compute ratios (proportion of each acceptor per cell)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(totals > 0, donor_counts / totals, 0)  # (n_cells, n_acceptors)

        # Check if ALL ratios have zero std across cells
        ratio_stds = ratios.std(axis=0)  # std for each acceptor across cells
        if np.all(ratio_stds == 0):
            n_zero_var += 1
            continue

        filtered_rows.append(row)
        filtered_counts.append(all_counts[i])

    if n_single_acceptor > 0:
        print(f"[INFO] Filtered {n_single_acceptor} donor site(s) with only one acceptor (no alternative splicing to model)")

    if n_zero_var > 0:
        print(f"[INFO] Filtered {n_zero_var} donor site(s) with zero variance in all acceptor usage ratios")

    if len(filtered_rows) == 0:
        warnings.warn("No donors with multiple acceptors found after filtering!")
        # Return empty arrays
        return (np.zeros((0, n_cells, 0), dtype=float),
                pd.DataFrame(columns=['chrom', 'strand', 'donor', 'acceptors', 'n_acceptors', 'gene_name', 'gene_id']),
                cell_names)

    # Stack into 3D array: (n_donors, n_cells, max_acceptors)
    max_acceptors = max(row['n_acceptors'] for row in filtered_rows)
    n_donors = len(filtered_rows)

    counts_3d = np.zeros((n_donors, n_cells, max_acceptors), dtype=float)
    for i, donor_counts in enumerate(filtered_counts):
        n_acc = donor_counts.shape[1]
        counts_3d[i, :, :n_acc] = donor_counts

    feature_meta = pd.DataFrame(filtered_rows)

    # Add category labels showing acceptor coordinates for each donor
    # Pad to max_acceptors to match the counts_3d dimensions
    def make_labels(acceptors):
        labels = [f"A:{a}" for a in acceptors]
        # Pad with empty strings to match max_acceptors
        while len(labels) < max_acceptors:
            labels.append("")
        return labels

    feature_meta['category_labels'] = feature_meta['acceptors'].apply(make_labels)

    return counts_3d, feature_meta, cell_names


def process_acceptor_usage(sj_counts: pd.DataFrame,
                           sj_meta: pd.DataFrame,
                           min_cell_total: int = 1) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Compute acceptor usage (which donor is used for each acceptor site).

    For each acceptor site (3' splice site), count how many reads come from each
    possible donor (5' splice site). Returns multinomial counts.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
        Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)
    min_cell_total : int
        Minimum total reads per acceptor per cell to include

    Returns
    -------
    counts_3d : np.ndarray
        Shape: (n_acceptors, n_cells, max_donors_per_acceptor)
    feature_meta : pd.DataFrame
        Acceptor site metadata with columns: chrom, strand, acceptor, donors (list), n_donors,
        gene (primary gene), genes (list of all genes)
    cell_names : list
        Cell identifiers
    """
    print("[INFO] Processing acceptor usage (3' splice sites)...")
    idx = _build_sj_index(sj_counts, sj_meta)

    # Check for gene_id columns
    has_gene_id = 'gene_id_start' in idx.columns and 'gene_id_end' in idx.columns

    # Group by (chrom, strand, acceptor)
    acceptor_groups = idx.groupby(['chrom', 'strand', 'acceptor'], sort=True)

    # Get cell names
    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)

    # Build feature metadata and 3D array
    feature_rows = []
    all_counts = []

    for (chrom, strand, acceptor), group in acceptor_groups:
        # Get all donors for this acceptor
        donors = sorted(group['donor'].unique())
        n_donors = len(donors)

        # Build mapping from donor to index
        donor_to_idx = {don: i for i, don in enumerate(donors)}

        # Initialize counts for this acceptor: (n_cells, n_donors)
        acceptor_counts = np.zeros((n_cells, n_donors), dtype=float)

        # Fill in counts for each junction
        for _, row in group.iterrows():
            coord = row['coord.intron']
            if coord not in sj_counts.index:
                continue
            donor = row['donor']
            don_idx = donor_to_idx[donor]

            # Add counts for all cells
            acceptor_counts[:, don_idx] += sj_counts.loc[coord].values

        # Apply min_cell_total filter
        if min_cell_total > 0:
            cell_totals = acceptor_counts.sum(axis=1)
            acceptor_counts[cell_totals < min_cell_total, :] = 0

        all_counts.append(acceptor_counts)

        # ---- REPLACE gene collection with site-centric assignment ----
        gene_name = _most_common(group['acceptor_gene_name'])
        gene_id   = _most_common(group['acceptor_gene_id'])
        # --------------------------------------------------------------

        feature_rows.append({
            'chrom': chrom,
            'strand': strand,
            'acceptor': acceptor,
            'donors': donors,
            'n_donors': n_donors,
            'gene_name': gene_name,
            'gene_id': gene_id
        })

    # Filter acceptors with only one donor OR with zero variance in ALL ratios
    n_before_filter = len(feature_rows)
    filtered_rows = []
    filtered_counts = []
    n_single_donor = 0
    n_zero_var = 0

    for i, row in enumerate(feature_rows):
        acceptor_counts = all_counts[i]  # Shape: (n_cells, n_donors)

        # Filter 1: Only one donor
        if row['n_donors'] <= 1:
            n_single_donor += 1
            continue

        # Filter 2: Check if ALL ratios have zero variance across cells
        # Compute total counts per cell
        totals = acceptor_counts.sum(axis=1, keepdims=True)  # (n_cells, 1)

        # Compute ratios (proportion of each donor per cell)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(totals > 0, acceptor_counts / totals, 0)  # (n_cells, n_donors)

        # Check if ALL ratios have zero std across cells
        ratio_stds = ratios.std(axis=0)  # std for each donor across cells
        if np.all(ratio_stds == 0):
            n_zero_var += 1
            continue

        filtered_rows.append(row)
        filtered_counts.append(all_counts[i])

    if n_single_donor > 0:
        print(f"[INFO] Filtered {n_single_donor} acceptor site(s) with only one donor (no alternative splicing to model)")

    if n_zero_var > 0:
        print(f"[INFO] Filtered {n_zero_var} acceptor site(s) with zero variance in all donor usage ratios")

    if len(filtered_rows) == 0:
        warnings.warn("No acceptors with multiple donors found after filtering!")
        # Return empty arrays
        return (np.zeros((0, n_cells, 0), dtype=float),
                pd.DataFrame(columns=['chrom', 'strand', 'acceptor', 'donors', 'n_donors', 'gene_name', 'gene_id']),
                cell_names)

    # Stack into 3D array: (n_acceptors, n_cells, max_donors)
    max_donors = max(row['n_donors'] for row in filtered_rows)
    n_acceptors = len(filtered_rows)

    counts_3d = np.zeros((n_acceptors, n_cells, max_donors), dtype=float)
    for i, acceptor_counts in enumerate(filtered_counts):
        n_don = acceptor_counts.shape[1]
        counts_3d[i, :, :n_don] = acceptor_counts

    feature_meta = pd.DataFrame(filtered_rows)

    # Add category labels showing donor coordinates for each acceptor
    # Pad to max_donors to match the counts_3d dimensions
    def make_labels(donors):
        labels = [f"D:{d}" for d in donors]
        # Pad with empty strings to match max_donors
        while len(labels) < max_donors:
            labels.append("")
        return labels

    feature_meta['category_labels'] = feature_meta['donors'].apply(make_labels)

    return counts_3d, feature_meta, cell_names


def _find_cassette_triplets_strand(sj_counts: pd.DataFrame,
                                   sj_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Find cassette exon triplets using strand-aware coordinates.

    A cassette exon event consists of:
    - sj_skip: Junction that skips the exon (donor d1 -> acceptor a3)
    - sj_inc1: Junction including the exon's 5' end (donor d1 -> acceptor a2)
    - sj_inc2: Junction including the exon's 3' end (donor d2 -> acceptor a3)

    Strand-specific ordering:
    - Plus strand: d1 < a2 < d2 < a3
    - Minus strand: d1 > a2 > d2 > a3

    Returns a DataFrame with per-site gene annotations:
      gene_name_d1/gene_id_d1, gene_name_d2/gene_id_d2,
      gene_name_a1/gene_id_a1, gene_name_a2/gene_id_a2,
    and if all four agree, unified gene_name/gene_id; otherwise NA.
    """
    # Build SJ index with donor/acceptor + site-centric genes
    idx = _build_sj_index(sj_counts, sj_meta)

    # Keep only clean junctions
    idx = idx[idx['strand'].notna() & idx['donor'].notna() & idx['acceptor'].notna()].copy()
    idx = idx.drop_duplicates(subset=['chrom', 'strand', 'donor', 'acceptor', 'coord.intron'])

    cols_empty = [
        'trip_id','chrom','strand','d1','a2','d2','a3',
        'sj_inc1','sj_inc2','sj_skip',
        'gene_name_d1','gene_id_d1','gene_name_d2','gene_id_d2',
        'gene_name_a1','gene_id_a1','gene_name_a2','gene_id_a2',
        'gene_name','gene_id'
    ]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols_empty)

    # Fast lookups
    idx_by_donor   = idx.groupby(['chrom', 'strand', 'donor'])
    idx_by_acceptor= idx.groupby(['chrom', 'strand', 'acceptor'])
    idx_by_pair = idx.set_index(['chrom', 'strand', 'donor', 'acceptor'])['coord.intron'].to_dict()

    # Site -> (most common) site-centric gene
    donor_name_map = idx.groupby(['chrom','strand','donor'])['donor_gene_name'].apply(_most_common).to_dict()
    donor_id_map   = idx.groupby(['chrom','strand','donor'])['donor_gene_id'].apply(_most_common).to_dict()
    acc_name_map   = idx.groupby(['chrom','strand','acceptor'])['acceptor_gene_name'].apply(_most_common).to_dict()
    acc_id_map     = idx.groupby(['chrom','strand','acceptor'])['acceptor_gene_id'].apply(_most_common).to_dict()

    triplets = []

    # For each potential skip junction (d1 -> a3)
    for _, row in idx.iterrows():
        chrom  = row['chrom']
        strand = row['strand']
        d1     = row['donor']
        a3     = row['acceptor']
        sj_skip= row['coord.intron']

        # All acceptors from d1
        try:
            group_d1 = idx_by_donor.get_group((chrom, strand, d1))
            a2_candidates = group_d1['acceptor'].unique()
        except KeyError:
            continue

        # All donors to a3
        try:
            group_a3 = idx_by_acceptor.get_group((chrom, strand, a3))
            d2_candidates = group_a3['donor'].unique()
        except KeyError:
            continue

        for a2 in a2_candidates:
            if pd.isna(a2):
                continue
            for d2 in d2_candidates:
                if pd.isna(d2):
                    continue

                # Check strand-specific exon order
                if strand == '+':
                    if not (d1 < a2 < d2 < a3):
                        continue
                else:  # '-'
                    if not (d1 > a2 > d2 > a3):
                        continue

                # Inclusion junctions must exist
                sj_inc1 = idx_by_pair.get((chrom, strand, d1, a2))  # d1->a2
                sj_inc2 = idx_by_pair.get((chrom, strand, d2, a3))  # d2->a3
                if sj_inc1 is None or sj_inc2 is None:
                    continue

                # Per-site genes
                gene_name_d1 = donor_name_map.get((chrom, strand, d1))
                gene_id_d1   = donor_id_map.get((chrom, strand, d1))
                gene_name_d2 = donor_name_map.get((chrom, strand, d2))
                gene_id_d2   = donor_id_map.get((chrom, strand, d2))

                gene_name_a1 = acc_name_map.get((chrom, strand, a2))
                gene_id_a1   = acc_id_map.get((chrom, strand, a2))
                gene_name_a2 = acc_name_map.get((chrom, strand, a3))
                gene_id_a2   = acc_id_map.get((chrom, strand, a3))

                # Unified gene if all 4 sites agree (both name and id)
                names = [gene_name_d1, gene_name_d2, gene_name_a1, gene_name_a2]
                ids   = [gene_id_d1,   gene_id_d2,   gene_id_a1,   gene_id_a2]
                if all(x is not None for x in names) and len(set(names)) == 1 and \
                   all(x is not None for x in ids)   and len(set(ids))   == 1:
                    gene_name_u = names[0]
                    gene_id_u   = ids[0]
                else:
                    gene_name_u = None
                    gene_id_u   = None

                triplets.append({
                    'chrom': chrom,
                    'strand': strand,
                    'd1': d1,
                    'a2': a2,
                    'd2': d2,
                    'a3': a3,
                    'sj_inc1': sj_inc1,
                    'sj_inc2': sj_inc2,
                    'sj_skip': sj_skip,
                    'gene_name_d1': gene_name_d1,
                    'gene_id_d1':   gene_id_d1,
                    'gene_name_d2': gene_name_d2,
                    'gene_id_d2':   gene_id_d2,
                    'gene_name_a1': gene_name_a1,
                    'gene_id_a1':   gene_id_a1,
                    'gene_name_a2': gene_name_a2,
                    'gene_id_a2':   gene_id_a2,
                    'gene_name':    gene_name_u,
                    'gene_id':      gene_id_u,
                })

    if not triplets:
        return pd.DataFrame(columns=cols_empty)

    trips_df = pd.DataFrame(triplets)

    # Deduplicate using event-defining columns only
    dedup_cols = ['chrom','strand','d1','a2','d2','a3','sj_inc1','sj_inc2','sj_skip']
    trips_df = trips_df.drop_duplicates(subset=dedup_cols, ignore_index=True)
    trips_df['trip_id'] = range(len(trips_df))

    # Ensure column order
    trips_df = trips_df[cols_empty]
    return trips_df


def _find_cassette_triplets_genomic(sj_counts: pd.DataFrame,
                                    sj_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Find cassette exon triplets using genomic coordinates (fallback when strand is unreliable).

    Uses left/right genomic positions instead of strand-aware donor/acceptor.
    Pattern: L1 < R2 < L2 < R3

    Returns a DataFrame with per-site gene annotations where possible (requires strand);
    if strand is missing for a row, per-site genes are left as NA for that event.
    """
    idx = _build_sj_index(sj_counts, sj_meta)

    # Keep only junctions with computed left/right
    idx = idx[idx['left'].notna() & idx['right'].notna()].copy()

    cols_empty = [
        'trip_id','chrom','strand','d1','a2','d2','a3',
        'sj_inc1','sj_inc2','sj_skip',
        'gene_name_d1','gene_id_d1','gene_name_d2','gene_id_d2',
        'gene_name_a1','gene_id_a1','gene_name_a2','gene_id_a2',
        'gene_name','gene_id'
    ]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols_empty)

    # Lookups by coordinates
    idx_by_left  = idx.groupby(['chrom', 'left'])
    idx_by_right = idx.groupby(['chrom', 'right'])
    idx_by_coords = idx.set_index(['chrom', 'left', 'right'])['coord.intron'].to_dict()

    # Site -> (most common) site-centric gene (requires strand)
    donor_name_map = idx.groupby(['chrom','strand','donor'])['donor_gene_name'].apply(_most_common).to_dict()
    donor_id_map   = idx.groupby(['chrom','strand','donor'])['donor_gene_id'].apply(_most_common).to_dict()
    acc_name_map   = idx.groupby(['chrom','strand','acceptor'])['acceptor_gene_name'].apply(_most_common).to_dict()
    acc_id_map     = idx.groupby(['chrom','strand','acceptor'])['acceptor_gene_id'].apply(_most_common).to_dict()

    triplets = []

    for _, row in idx.iterrows():
        chrom = row['chrom']
        L1    = row['left']
        R3    = row['right']
        sj_skip = row['coord.intron']
        strand = row.get('strand', None)
        strand_val = strand if pd.notna(strand) else None

        # Candidates with left=L1 and right between L1 and R3  -> (L1, R2)
        try:
            group_L1 = idx_by_left.get_group((chrom, L1))
            R2_candidates = group_L1[(group_L1['right'] > L1) & (group_L1['right'] < R3)]['right'].unique()
        except KeyError:
            continue

        # Candidates with right=R3 and left between L1 and R3  -> (L2, R3)
        try:
            group_R3 = idx_by_right.get_group((chrom, R3))
            L2_candidates = group_R3[(group_R3['left'] > L1) & (group_R3['left'] < R3)]['left'].unique()
        except KeyError:
            continue

        for R2 in R2_candidates:
            sj_inc1 = idx_by_coords.get((chrom, L1, R2))
            if sj_inc1 is None:
                continue

            for L2 in L2_candidates:
                sj_inc2 = idx_by_coords.get((chrom, L2, R3))
                if sj_inc2 is None:
                    continue

                # Map genomic to site notation:
                d1 = L1
                a2 = R2
                d2 = L2
                a3 = R3

                # Per-site genes (only if strand known)
                if strand_val is not None:
                    gene_name_d1 = donor_name_map.get((chrom, strand_val, d1))
                    gene_id_d1   = donor_id_map.get((chrom, strand_val, d1))
                    gene_name_d2 = donor_name_map.get((chrom, strand_val, d2))
                    gene_id_d2   = donor_id_map.get((chrom, strand_val, d2))

                    gene_name_a1 = acc_name_map.get((chrom, strand_val, a2))
                    gene_id_a1   = acc_id_map.get((chrom, strand_val, a2))
                    gene_name_a2 = acc_name_map.get((chrom, strand_val, a3))
                    gene_id_a2   = acc_id_map.get((chrom, strand_val, a3))
                else:
                    gene_name_d1 = gene_id_d1 = None
                    gene_name_d2 = gene_id_d2 = None
                    gene_name_a1 = gene_id_a1 = None
                    gene_name_a2 = gene_id_a2 = None

                # Unified gene if all 4 sites agree
                names = [gene_name_d1, gene_name_d2, gene_name_a1, gene_name_a2]
                ids   = [gene_id_d1,   gene_id_d2,   gene_id_a1,   gene_id_a2]
                if all(x is not None for x in names) and len(set(names)) == 1 and \
                   all(x is not None for x in ids)   and len(set(ids))   == 1:
                    gene_name_u = names[0]
                    gene_id_u   = ids[0]
                else:
                    gene_name_u = None
                    gene_id_u   = None

                triplets.append({
                    'chrom': chrom,
                    'strand': strand_val,
                    'd1': d1,
                    'a2': a2,
                    'd2': d2,
                    'a3': a3,
                    'sj_inc1': sj_inc1,
                    'sj_inc2': sj_inc2,
                    'sj_skip': sj_skip,
                    'gene_name_d1': gene_name_d1,
                    'gene_id_d1':   gene_id_d1,
                    'gene_name_d2': gene_name_d2,
                    'gene_id_d2':   gene_id_d2,
                    'gene_name_a1': gene_name_a1,
                    'gene_id_a1':   gene_id_a1,
                    'gene_name_a2': gene_name_a2,
                    'gene_id_a2':   gene_id_a2,
                    'gene_name':    gene_name_u,
                    'gene_id':      gene_id_u,
                })

    if not triplets:
        return pd.DataFrame(columns=cols_empty)

    trips_df = pd.DataFrame(triplets)

    # Deduplicate using event-defining columns only
    dedup_cols = ['chrom','strand','d1','a2','d2','a3','sj_inc1','sj_inc2','sj_skip']
    trips_df = trips_df.drop_duplicates(subset=dedup_cols, ignore_index=True)
    trips_df['trip_id'] = range(len(trips_df))

    # Ensure column order
    trips_df = trips_df[cols_empty]
    return trips_df


def process_exon_skipping(sj_counts: pd.DataFrame,
                         sj_meta: pd.DataFrame,
                         min_total_exon: int = 2,
                         method: str = 'min',
                         fallback_genomic: bool = True,
                         return_unfiltered: bool = True) -> Tuple:
    """
    Compute exon skipping (cassette exon) inclusion counts.

    For each cassette exon event, compute:
    - inc: Inclusion reads (min or mean of inc1 and inc2)
    - tot: Total reads (inc + skip)

    Returns binomial data (inclusion count, total count).

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
    min_total_exon : int
        Minimum total reads per event per cell
    method : str
        'min' (default) or 'mean' for computing inclusion from inc1 and inc2
    fallback_genomic : bool
        If True, use genomic coordinates when strand-aware search finds no events
    return_unfiltered : bool
        If True (default), also return unfiltered data before variance filtering.
        This allows recovery of filtered events when changing aggregation method.

    Returns
    -------
    inc1_counts : np.ndarray
        Shape: (n_events_filtered, n_cells) - Filtered counts for first inclusion junction (d1->a2)
    inc2_counts : np.ndarray
        Shape: (n_events_filtered, n_cells) - Filtered counts for second inclusion junction (d2->a3)
    skip_counts : np.ndarray
        Shape: (n_events_filtered, n_cells) - Filtered counts for skipping junction (d1->a3)
    feature_meta : pd.DataFrame
        Filtered event metadata with columns: trip_id, chrom, strand, d1, a2, d2, a3, sj_inc1, sj_inc2, sj_skip
    cell_names : list
        Cell identifiers
    inc1_unfiltered : np.ndarray or None
        If return_unfiltered=True: Shape (n_events_total, n_cells) - ALL events before variance filtering
    inc2_unfiltered : np.ndarray or None
        If return_unfiltered=True: Shape (n_events_total, n_cells) - ALL events before variance filtering
    skip_unfiltered : np.ndarray or None
        If return_unfiltered=True: Shape (n_events_total, n_cells) - ALL events before variance filtering
    feature_meta_unfiltered : pd.DataFrame or None
        If return_unfiltered=True: ALL event metadata before variance filtering
    """
    print(f"[INFO] Processing exon skipping (cassette exons) with method='{method}'...")
    # Find cassette triplets
    trips = _find_cassette_triplets_strand(sj_counts, sj_meta)

    if len(trips) == 0 and fallback_genomic:
        print("[INFO] No strand-aware cassette exons found, trying genomic coordinates...")
        trips = _find_cassette_triplets_genomic(sj_counts, sj_meta)

    if len(trips) == 0:
        # Return empty arrays
        cell_names = sj_counts.columns.tolist()
        empty_meta = pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                           'sj_inc1', 'sj_inc2', 'sj_skip', 'gene', 'genes'])
        if return_unfiltered:
            return (np.zeros((0, len(cell_names))), np.zeros((0, len(cell_names))),
                    np.zeros((0, len(cell_names))), empty_meta, cell_names,
                    None, None, None, None)
        else:
            return (np.zeros((0, len(cell_names))), np.zeros((0, len(cell_names))),
                    np.zeros((0, len(cell_names))), empty_meta, cell_names)

    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)
    n_events = len(trips)

    # Initialize arrays for raw counts
    inc1_counts = np.zeros((n_events, n_cells), dtype=float)
    inc2_counts = np.zeros((n_events, n_cells), dtype=float)
    skip_counts = np.zeros((n_events, n_cells), dtype=float)

    # Get counts for all needed junctions
    needed_sjs = pd.concat([trips['sj_inc1'], trips['sj_inc2'], trips['sj_skip']]).unique()
    needed_sjs = [sj for sj in needed_sjs if sj in sj_counts.index]
    sj_data = sj_counts.loc[needed_sjs]

    for i, row in trips.iterrows():
        sj_inc1 = row['sj_inc1']
        sj_inc2 = row['sj_inc2']
        sj_skip = row['sj_skip']

        # Get raw counts
        inc1 = sj_data.loc[sj_inc1].values if sj_inc1 in sj_data.index else np.zeros(n_cells)
        inc2 = sj_data.loc[sj_inc2].values if sj_inc2 in sj_data.index else np.zeros(n_cells)
        skip = sj_data.loc[sj_skip].values if sj_skip in sj_data.index else np.zeros(n_cells)

        # Compute aggregated inclusion for filtering only
        if method == 'min':
            inc_agg = np.minimum(inc1, inc2)
        elif method == 'mean':
            inc_agg = (inc1 + inc2) / 2.0
        else:
            raise ValueError(f"method must be 'min' or 'mean', got: {method}")

        # Total = aggregated inclusion + skipping
        tot = inc_agg + skip

        # Apply minimum total filter to ALL raw counts
        if min_total_exon > 0:
            mask = tot < min_total_exon
            inc1[mask] = 0
            inc2[mask] = 0
            skip[mask] = 0

        # Store raw counts
        inc1_counts[i, :] = inc1
        inc2_counts[i, :] = inc2
        skip_counts[i, :] = skip

    # Store unfiltered data before variance filtering
    inc1_unfiltered = inc1_counts.copy() if return_unfiltered else None
    inc2_unfiltered = inc2_counts.copy() if return_unfiltered else None
    skip_unfiltered = skip_counts.copy() if return_unfiltered else None
    trips_unfiltered = trips.copy() if return_unfiltered else None

    # Filter events with zero variance in inclusion/total ratio across cells
    # Compute inclusion using specified method
    if method == 'min':
        inc_for_filter = np.minimum(inc1_counts, inc2_counts)
    elif method == 'mean':
        inc_for_filter = (inc1_counts + inc2_counts) / 2.0

    tot_for_filter = inc_for_filter + skip_counts

    # Check ratio variance for each event
    valid_events = []
    n_zero_var = 0

    for i in range(n_events):
        numer = inc_for_filter[i, :]  # inclusion counts across cells
        denom = tot_for_filter[i, :]   # total counts across cells

        # Compute ratios, excluding cells where denominator is 0
        valid_mask = denom > 0
        if valid_mask.sum() == 0:
            # All denominators are zero - can't compute ratio
            n_zero_var += 1
            continue

        ratios = numer[valid_mask] / denom[valid_mask]
        if ratios.std() == 0:
            n_zero_var += 1
            continue

        valid_events.append(i)

    if n_zero_var > 0:
        print(f"[INFO] Filtered {n_zero_var} exon skipping event(s) with zero variance in inclusion ratio")

    if len(valid_events) == 0:
        warnings.warn("No exon skipping events with variable inclusion ratios found after filtering!")
        empty_meta = pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                           'sj_inc1', 'sj_inc2', 'sj_skip', 'gene', 'genes'])
        if return_unfiltered:
            return (np.zeros((0, n_cells)), np.zeros((0, n_cells)), np.zeros((0, n_cells)),
                    empty_meta, cell_names,
                    inc1_unfiltered, inc2_unfiltered, skip_unfiltered, trips_unfiltered)
        else:
            return (np.zeros((0, n_cells)), np.zeros((0, n_cells)), np.zeros((0, n_cells)),
                    empty_meta, cell_names)

    # Subset to valid events
    inc1_counts_filt = inc1_counts[valid_events, :]
    inc2_counts_filt = inc2_counts[valid_events, :]
    skip_counts_filt = skip_counts[valid_events, :]
    trips_filt = trips.iloc[valid_events].reset_index(drop=True)
    trips_filt['trip_id'] = range(len(trips_filt))

    if return_unfiltered:
        return (inc1_counts_filt, inc2_counts_filt, skip_counts_filt, trips_filt, cell_names,
                inc1_unfiltered, inc2_unfiltered, skip_unfiltered, trips_unfiltered)
    else:
        return inc1_counts_filt, inc2_counts_filt, skip_counts_filt, trips_filt, cell_names


def process_sj_counts(sj_counts: pd.DataFrame,
                      sj_meta: pd.DataFrame,
                      gene_counts: pd.DataFrame,
                      cis_gene_counts: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw SJ counts for binomial modality.

    Creates binomial modality where:
    - Numerator: SJ counts
    - Denominator: Gene-level counts (from primary gene counts)

    Only includes SJs that map to genes.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
        Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)
    gene_counts : pd.DataFrame
        Gene-level counts to use as denominator (genes × cells)
    cis_gene_counts : pd.DataFrame, optional
        Additional gene counts for cis gene(s) not in gene_counts.
        This allows including SJs for the cis gene even when cis gene
        has been removed from gene_counts for trans modeling.
        Should have genes as rows and cells as columns.

    Returns
    -------
    sj_counts_filtered : pd.DataFrame
        Filtered SJ counts (junctions × cells)
    gene_counts_denom : pd.DataFrame
        Gene counts for denominator (junctions × cells), same shape as sj_counts_filtered
    sj_meta_filtered : pd.DataFrame
        Filtered SJ metadata
    """
    print("[INFO] Processing raw splice junction counts...")
    # Find overlapping cells between sj_counts and gene_counts
    common_cells = [c for c in sj_counts.columns if c in gene_counts.columns]

    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between sj_counts and gene_counts")

    # Subset sj_counts to common cells
    sj_counts_subset = sj_counts[common_cells].copy()

    # Check for cells in sj_counts but not in gene_counts (informational)
    extra_cells = set(sj_counts.columns) - set(gene_counts.columns)
    if extra_cells:
        print(f"[INFO] {len(extra_cells)} cell(s) in sj_counts are not in gene_counts and will be excluded")

    idx = _build_sj_index(sj_counts_subset, sj_meta)

    # Track reasons for dropping SJs
    n_total = len(idx)
    n_dropped_no_gene = 0
    n_dropped_diff_gene = 0
    n_dropped_gene_not_in_counts = 0

    # Assign gene identifiers for start and end
    has_gene_id = 'gene_id_start' in idx.columns and 'gene_id_end' in idx.columns

    if has_gene_id:
        # Try gene names first, fall back to IDs for start
        idx['gene_start'] = idx['gene_name_start'].fillna(idx['gene_id_start'])
        # Try gene names first, fall back to IDs for end
        idx['gene_end'] = idx['gene_name_end'].fillna(idx['gene_id_end'])
    else:
        # Only gene names available
        idx['gene_start'] = idx['gene_name_start']
        idx['gene_end'] = idx['gene_name_end']

    # Remove SJs without gene assignment for both start and end
    no_gene_mask = idx['gene_start'].isna() | idx['gene_end'].isna()
    n_dropped_no_gene = no_gene_mask.sum()
    idx = idx[~no_gene_mask].copy()

    if len(idx) == 0:
        raise ValueError("No splice junctions with gene annotations found")

    # Keep only SJs where start and end are in the same gene
    same_gene_mask = idx['gene_start'] == idx['gene_end']
    n_dropped_diff_gene = (~same_gene_mask).sum()
    idx = idx[same_gene_mask].copy()

    if len(idx) == 0:
        raise ValueError("No splice junctions with matching start/end genes found")

    # Assign the gene (since start == end)
    idx['gene'] = idx['gene_start']

    # Filter SJ counts to these junctions
    sj_filtered = sj_counts_subset.loc[idx['coord.intron']].copy()

    # Subset gene_counts to common cells
    gene_counts_subset = gene_counts[common_cells].copy()

    # Build denominator: for each SJ, get the corresponding gene count
    # Track which SJs to keep (only those where gene is found in gene_counts or cis_gene_counts)
    valid_sjs = []
    gene_denom_data = []

    # Prepare cis_gene_counts subset if provided
    cis_gene_counts_subset = None
    if cis_gene_counts is not None:
        cis_common_cells = [c for c in common_cells if c in cis_gene_counts.columns]
        if len(cis_common_cells) > 0:
            cis_gene_counts_subset = cis_gene_counts[cis_common_cells].copy()
            # Reindex to match gene_counts_subset columns
            cis_gene_counts_subset = cis_gene_counts_subset.reindex(columns=common_cells, fill_value=0)

    for sj_id, row in idx.iterrows():
        gene = row['gene']
        coord = row['coord.intron']

        # Try to find gene in gene_counts by name or ID
        found = False
        gene_expr = None

        if gene in gene_counts_subset.index:
            gene_expr = gene_counts_subset.loc[gene].values
            found = True
        else:
            # If not found, try alternate identifier
            # If we have a name, try finding by ID (and vice versa)
            if has_gene_id:
                # Try all possible gene identifiers for this SJ
                for gene_col in ['gene_name_start', 'gene_name_end', 'gene_id_start', 'gene_id_end']:
                    if gene_col in row.index and pd.notna(row[gene_col]):
                        alt_gene = row[gene_col]
                        if alt_gene in gene_counts_subset.index:
                            gene_expr = gene_counts_subset.loc[alt_gene].values
                            found = True
                            break

        # If not found in gene_counts, check cis_gene_counts
        if not found and cis_gene_counts_subset is not None:
            if gene in cis_gene_counts_subset.index:
                gene_expr = cis_gene_counts_subset.loc[gene].values
                found = True
            elif has_gene_id:
                # Try all possible gene identifiers for this SJ in cis_gene_counts
                for gene_col in ['gene_name_start', 'gene_name_end', 'gene_id_start', 'gene_id_end']:
                    if gene_col in row.index and pd.notna(row[gene_col]):
                        alt_gene = row[gene_col]
                        if alt_gene in cis_gene_counts_subset.index:
                            gene_expr = cis_gene_counts_subset.loc[alt_gene].values
                            found = True
                            break

        if found:
            valid_sjs.append(coord)
            gene_denom_data.append(gene_expr)
        else:
            n_dropped_gene_not_in_counts += 1

    # Filter to valid SJs only
    if len(valid_sjs) == 0:
        raise ValueError("No splice junctions with genes found in gene_counts")

    sj_filtered = sj_filtered.loc[valid_sjs].copy()
    idx = idx[idx['coord.intron'].isin(valid_sjs)].copy()

    # Build denominator DataFrame
    gene_denom = pd.DataFrame(
        data=np.array(gene_denom_data),
        index=valid_sjs,
        columns=sj_filtered.columns
    )

    # Check that numerator <= denominator for all SJs
    # If not, clip numerator to min(numerator, denominator) and warn
    violations = (sj_filtered.values > gene_denom.values)
    n_violations_total = violations.sum()

    if n_violations_total > 0:
        # Count how many SJs have at least one violation
        n_sjs_with_violations = (violations.sum(axis=1) > 0).sum()
        warnings.warn(
            f"Found {n_violations_total} cell(s) where SJ counts > gene counts across {n_sjs_with_violations} junction(s). "
            f"This is biologically implausible (splice junction reads cannot exceed total gene expression). "
            f"Clipping SJ counts to min(SJ, gene) for affected cells.",
            UserWarning
        )
        # Apply clipping
        sj_filtered = pd.DataFrame(
            np.minimum(sj_filtered.values, gene_denom.values),
            index=sj_filtered.index,
            columns=sj_filtered.columns
        )

    # Filter SJs with zero variance in SJ/gene ratio across cells
    valid_sjs_final = []
    sj_denom_final = []
    idx_final = []
    n_zero_var = 0

    for i, sj in enumerate(valid_sjs):
        numer = sj_filtered.loc[sj].values  # SJ counts across cells
        denom = gene_denom.loc[sj].values   # Gene counts across cells

        # Compute ratios, excluding cells where denominator is 0
        valid_mask = denom > 0
        if valid_mask.sum() == 0:
            # All denominators are zero - can't compute ratio
            n_zero_var += 1
            continue

        ratios = numer[valid_mask] / denom[valid_mask]
        if ratios.std() == 0:
            n_zero_var += 1
            continue

        valid_sjs_final.append(sj)
        sj_denom_final.append(gene_denom.loc[sj].values)
        idx_final.append(sj)

    if n_zero_var > 0:
        print(f"[INFO] Filtered {n_zero_var} splice junction(s) with zero variance in SJ/gene ratio")

    if len(valid_sjs_final) == 0:
        raise ValueError("No splice junctions left after filtering zero-variance SJ/gene ratios!")

    # Update to final filtered SJs
    sj_filtered = sj_filtered.loc[valid_sjs_final].copy()
    gene_denom = pd.DataFrame(
        data=np.array(sj_denom_final),
        index=valid_sjs_final,
        columns=sj_filtered.columns
    )
    idx = idx[idx['coord.intron'].isin(valid_sjs_final)].copy()

    # Print summary of dropped SJs
    n_kept = len(valid_sjs_final)
    n_total_dropped = n_total - n_kept
    if n_total_dropped > 0:
        reasons = []
        if n_dropped_no_gene > 0:
            reasons.append(f"{n_dropped_no_gene} missing gene annotation")
        if n_dropped_diff_gene > 0:
            reasons.append(f"{n_dropped_diff_gene} spanning different genes")
        if n_dropped_gene_not_in_counts > 0:
            reasons.append(f"{n_dropped_gene_not_in_counts} gene not in gene_counts")
        if n_zero_var > 0:
            reasons.append(f"{n_zero_var} zero variance in SJ/gene ratio")

        print(f"[INFO] Kept {n_kept}/{n_total} splice junctions. Dropped {n_total_dropped} SJs: {', '.join(reasons)}")

    return sj_filtered, gene_denom, idx


def create_splicing_modality(sj_counts: pd.DataFrame,
                             sj_meta: pd.DataFrame,
                             splicing_type: str,
                             gene_counts: pd.DataFrame,
                             min_cell_total: int = 1,
                             min_total_exon: int = 2,
                             cell_names: Optional[List[str]] = None,
                             cis_gene_counts: Optional[pd.DataFrame] = None,
                             **kwargs) -> Modality:
    """
    Create a Modality object for splicing data.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
    splicing_type : str
        Type of splicing metric: 'sj', 'donor', 'acceptor', or 'exon_skip'
    gene_counts : pd.DataFrame
        Gene counts for 'sj' type denominator (genes × cells). Required for all types.
    min_cell_total : int
        Minimum reads for donor/acceptor usage
    min_total_exon : int
        Minimum reads for exon skipping
    cell_names : list of str, optional
        Cell identifiers (extracted from DataFrame or explicitly provided)
    cis_gene_counts : pd.DataFrame, optional
        Additional gene counts for cis gene(s) not in gene_counts.
        Only used for splicing_type='sj' to include SJs mapping to the cis gene.
    **kwargs
        Additional arguments (e.g., method, fallback_genomic for exon_skip)

    Returns
    -------
    Modality
        Modality object with appropriate distribution and metadata
    """
    def _donor_feature_names(df: pd.DataFrame) -> list[str]:
        # D:<chrom>:<strand>:<donor>
        return [f"D:{c}:{s}:{d}" for c, s, d in zip(df['chrom'], df['strand'], df['donor'])]
    
    def _acceptor_feature_names(df: pd.DataFrame) -> list[str]:
        # A:<chrom>:<strand>:<acceptor>
        return [f"A:{c}:{s}:{a}" for c, s, a in zip(df['chrom'], df['strand'], df['acceptor'])]
    
    def _exon_skip_feature_names(df: pd.DataFrame) -> list[str]:
        # X:<chrom>:<strand>:<d1>-<a2>|<d2>-<a3>  (stable & human-readable)
        # fall back safely if strand is NA
        s_list = df['strand'].astype(object).where(df['strand'].notna(), '?')
        return [f"X:{c}:{s}:{d1}-{a2}|{d2}-{a3}"
                for c, s, d1, a2, d2, a3 in zip(df['chrom'], s_list, df['d1'], df['a2'], df['d2'], df['a3'])]
    
    if splicing_type == 'sj':
        # Raw SJ counts as binomial with gene counts as denominator
        sj_filtered, gene_denom, sj_meta_filtered = process_sj_counts(
            sj_counts, sj_meta, gene_counts, cis_gene_counts=cis_gene_counts
        )

        # Extract cell_names if not provided
        if cell_names is None:
            cell_names = sj_filtered.columns.tolist()

        return Modality(
            name='splicing_sj',
            counts=sj_filtered,
            feature_meta=sj_meta_filtered,
            distribution='binomial',
            denominator=gene_denom.values,
            cells_axis=1,
            cell_names=cell_names
        )

    elif splicing_type == 'donor':
        counts_3d, feature_meta, cell_names = process_donor_usage(
            sj_counts, sj_meta, min_cell_total
        )
        feature_names = _donor_feature_names(feature_meta)
        # Convert to DataFrame to preserve cell names
        # For multinomial, we can't use DataFrame directly, so we'll store metadata separately
        return Modality(
            name='splicing_donor',
            counts=counts_3d,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1,
            cell_names=cell_names,
            feature_names=feature_names,  # <-- pass names
        )

    elif splicing_type == 'acceptor':
        counts_3d, feature_meta, cell_names = process_acceptor_usage(
            sj_counts, sj_meta, min_cell_total
        )
        feature_names = _acceptor_feature_names(feature_meta)
        
        return Modality(
            name='splicing_acceptor',
            counts=counts_3d,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1,
            cell_names=cell_names,
            feature_names=feature_names,  # <-- pass names
        )

    elif splicing_type == 'exon_skip':
        method = kwargs.get('method', 'min')
        fallback_genomic = kwargs.get('fallback_genomic', True)

        result = process_exon_skipping(
            sj_counts, sj_meta, min_total_exon, method, fallback_genomic, return_unfiltered=True
        )

        # Unpack result (with or without unfiltered data)
        if len(result) == 9:
            inc1_counts, inc2_counts, skip_counts, feature_meta, cell_names, \
                inc1_unfilt, inc2_unfilt, skip_unfilt, meta_unfilt = result
        else:
            inc1_counts, inc2_counts, skip_counts, feature_meta, cell_names = result
            inc1_unfilt, inc2_unfilt, skip_unfilt, meta_unfilt = None, None, None, None
        
        feature_names = _exon_skip_feature_names(feature_meta)

        # Compute inclusion and total using specified method
        if method == 'min':
            inc_counts = np.minimum(inc1_counts, inc2_counts)
        elif method == 'mean':
            inc_counts = (inc1_counts + inc2_counts) / 2.0
        tot_counts = inc_counts + skip_counts

        # Create modality
        modality = Modality(
            name='splicing_exon_skip',
            counts=inc_counts,
            feature_meta=feature_meta,
            distribution='binomial',
            denominator=tot_counts,
            cells_axis=1,
            cell_names=cell_names,
            feature_names=feature_names,  # <-- pass names
            inc1=inc1_counts,
            inc2=inc2_counts,
            skip=skip_counts,
            exon_aggregate_method=method
        )

        # Store unfiltered data for potential recovery when switching methods
        if inc1_unfilt is not None:
            modality._unfiltered_inc1 = inc1_unfilt
            modality._unfiltered_inc2 = inc2_unfilt
            modality._unfiltered_skip = skip_unfilt
            modality._unfiltered_feature_meta = meta_unfilt

        return modality

    else:
        raise ValueError(f"splicing_type must be 'sj', 'donor', 'acceptor', or 'exon_skip', got: {splicing_type}")
