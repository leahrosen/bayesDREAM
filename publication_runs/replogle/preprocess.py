"""
One-time Replogle preprocessing: reassemble a [full NTC population] union
[chosen cis genes' cells] subset from the pre-built (read-only)
`for_bayesDREAM/K562_combined/` parquet tree, and recompute `sum_factor`
ONCE on that combined population -- mirroring Domingo/Morris's "scran
computed once upstream, shared everywhere" rule (see
publication_runs/README.md's "sum_factor recomputation" section and
publication_runs/replogle/STRATEGY.md §5/§7).

Why this exists (see STRATEGY.md §5): the ad hoc notebooks
(tmp/06_bayesDREAM_fit_ntc_combined.ipynb, tmp/10_bayesDREAM_fit_trans_MYB.ipynb)
read a `sum_factor` column that ships pre-computed inside
`for_bayesDREAM/K562_combined/` -- its own provenance/scope (e.g. whether it
was computed genome-wide across every screened gene, not just the 7 chosen
ones) is not visible anywhere in this repo. This script instead computes
`sum_factor` fresh, restricted to exactly [the full NTC population] union
[cells targeting one of `--cis-genes`], via `quickCluster`+`computeSumFactors`
blocked by `--batch-col` (default 'batch') -- Morris-style (unsupervised
quickCluster, NOT Domingo's guide-identity-clustered/ref.clust='ntc' style),
matching `adjust_ntc_sum_factor(covariates=["batch"])`'s own choice of
covariate in the existing notebooks. Confirmed with the user (2026-09-08):
'batch' only, not 'experiment', even though `set_technical_groups` uses both.

**NTC source -- corrected 2026-09-09, see STRATEGY.md §2/§10.** The trans
notebook's own `NTC_subset/counts_subset.parquet` (used by an earlier
version of this script) turns out to be ALREADY subsetted to a chosen set
of NTC guides -- confirmed by the user, not the full NTC population. Per
the user's explicit instruction, this script instead reads `NTC/` (the SAME
directory `tmp/06_bayesDREAM_fit_ntc_combined.ipynb` -- literally named
"fit_ntc_combined" -- itself reads, 85711 cells) as the true full-NTC
source, used for BOTH `meta_ntc.csv`/`gene_counts_ntc.npz` (feeds
`ntc_shared`) and the combined `meta.csv`/`gene_counts.npz` (feeds the
`all_indmu` cis_variant, since its own `subset_per_gene.py` step never
narrows the NTC pool -- see replogle/config.yaml's `cis_variants`).
`NTC_subset/` is no longer read by this script at all. A future, explicit
guide-level NTC restriction (a SPECIFIC list of NTC guides -- distinct from
the batch-based `bm` restriction `common/subset_per_gene.py`'s
`batch_match_ntc` flag already implements) is intentionally left
unimplemented here pending that list from the user.

Input directory layout it reads (READ-ONLY -- confirmed no write access,
2026-09-08 -- never written to by this script):
    <indir>/NTC/cell_meta.csv, gene_meta.csv, gene_counts.npz,
            gene_counts_genes.npy, gene_counts_cells.npy
                                      the FULL NTC population (every NTC
                                      guide) -- same file
                                      tmp/06_bayesDREAM_fit_ntc_combined.ipynb
                                      reads. gene_counts_genes.npy/
                                      gene_counts_cells.npy give this
                                      block's row/column order; gene_meta.csv/
                                      cell_meta.csv are asserted aligned to
                                      them (mirrors the reference notebook's
                                      own two assert statements).
    <indir>/cell_meta_full.parquet   cell-level metadata, EVERY screened
                                      gene's target cells + all NTC cells
                                      (not just the 7 chosen genes) --
                                      used here ONLY to authoritatively
                                      label which cells in each
                                      <gene_id>/counts.parquet truly target
                                      that gene (see below) -- its own NTC
                                      rows are NOT used (NTC/ above is used
                                      instead, per the correction above).
    <indir>/gene_meta_full.parquet   gene_id (Ensembl)-indexed, gene_name
                                      column -- the fixed 8202-gene trans
                                      panel used everywhere in this pipeline
    <indir>/<gene_id>/counts.parquet
                                      counts for cells in that gene's own
                                      screen -- NOT assumed pure (see next
                                      paragraph); columns are filtered
                                      against cell_meta_full's 'target'
                                      column before use.

**Per-gene files are NOT assumed to contain only that gene's target
cells** (per the user's explicit instruction, point 2: "remove the NTC
cells from the cis gene data load") -- each `<gene_id>/counts.parquet`'s
columns are intersected against `cell_meta_full`'s cells where
`target == gene_id` before being used, dropping any column (e.g. an NTC
cell) that isn't authoritatively labelled as targeting THIS gene. This
also means NTC cells are never double-counted between the `NTC/` block and
a per-gene block: any NTC cell an over-inclusive per-gene file might have
carried is dropped here, so every NTC cell in the final output comes from
`NTC/`, and only `NTC/`.

Writes to <outdir> (NOT <indir> -- read-only, see above):
    meta.csv          cell metadata for [full NTC population] union
                       [chosen cis genes' true target cells], sum_factor
                       RECOMPUTED (not any input column)
    gene_counts.npz   sparse (genes x cells), columns matching meta.csv's
                       cell order exactly
    gene_meta.csv     gene_id (Ensembl), gene_name (real symbol) -- row order
                      matches gene_counts.npz. Model identity is pinned to
                      gene_id via config.yaml/generate_slurm.py's explicit
                      `model.feature_name_col: gene_id`, not by this file's
                      column priority -- see STRATEGY.md §11.
    meta_ntc.csv / gene_counts_ntc.npz
                      NTC-only split of the same two files above (the FULL
                      NTC population, unchanged from NTC/ except for the
                      recomputed sum_factor) -- for the ntc_shared stage
                      (mirrors Domingo's meta_ntc/counts_ntc convention).

Cis gene -> Ensembl gene_id mapping is hardcoded below (REPLOGLE_GENE_TO_ID),
duplicated (deliberately, not imported) from comparative/datasets.py's dict
of the same name -- publication_runs/ (the production pipeline) shouldn't
depend on comparative/ (a cross-dataset plotting/comparison tool that
itself reads THIS pipeline's output); keep both in sync by hand if this
list ever changes.

Usage
-----
    python preprocess.py \\
        --indir /cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/pr_data/for_bayesDREAM/K562_combined \\
        --outdir /cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Replogle2022/processed_Leah
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from compute_scran_sum_factor import _compute_scran_sizefactors  # noqa: E402

# Duplicated from comparative/datasets.py's REPLOGLE_GENE_TO_ID -- see
# module docstring for why this isn't imported instead.
REPLOGLE_GENE_TO_ID = {
    'MYB':   'ENSG00000118513',
    'NFE2':  'ENSG00000123405',
    'HHEX':  'ENSG00000152804',
    'RUNX1': 'ENSG00000159216',
    'GFI1B': 'ENSG00000165702',
    'TET2':  'ENSG00000168769',
    'IKZF1': 'ENSG00000185811',
}

DEFAULT_CIS_GENES = list(REPLOGLE_GENE_TO_ID.keys())


def _read_parquet(path: str) -> pd.DataFrame:
    """pd.read_parquet(path, engine='fastparquet'), not the pyarrow default.

    Ported from comparative/reconstruct_export_replogle.py's `_read_parquet`:
    PyArrow 19.0.0 (confirmed on this cluster's pyroenv, 2026-08-26) fails to
    read Replogle's parquet inputs with 'OSError: Repetition level histogram
    size mismatch' -- a PyArrow-internal bug in its Parquet page-index
    validation. fastparquet is a separate, pure-Python/numba implementation
    with no shared code path, so it isn't affected.
    """
    return pd.read_parquet(path, engine='fastparquet')


def _bin_small_batches(meta: pd.DataFrame, batch_col: str, min_size: int) -> pd.Series:
    """Return a NEW Series (not written into `meta`) merging every value of
    `meta[batch_col]` with fewer than `min_size` rows into one pooled
    `'_pooled_small_batches'` bucket, for use ONLY as scran's `quickCluster`
    blocking variable.

    `quickCluster(sce, block=batch)` clusters cells SEPARATELY within each
    block, and raises ("fewer cells than the minimum cluster size") if any
    single block has fewer cells than its own `min.size` (scran's default:
    100) -- confirmed on Replogle's real combined [full NTC + 7 genes]
    population, 2026-09-10 (`BiocParallel errors ... 1 remote errors,
    element index: 27` -- exactly ONE batch value too small; the "499
    unevaluated" alongside it is BiocParallel's cascade from that one
    worker dying, not 499 separate failures).

    Deliberately does NOT touch the real `batch` column anywhere else in
    this pipeline (`set_technical_groups`, `adjust_ntc_sum_factor`, `bm`'s
    `batch_match_ntc`) -- this pooling exists ONLY to make scran's
    clustering step succeed; every other stage still sees genuine,
    unmodified batch identity. If the pooled bucket ITSELF is still below
    `min_size` (i.e. there's no way to satisfy scran's own minimum by
    pooling every rare batch together), raises rather than silently
    proceeding -- that would need actual investigation, not automatic
    handling.
    """
    counts = meta[batch_col].value_counts()
    small = counts[counts < min_size].index
    if len(small) == 0:
        return meta[batch_col]

    pooled_label = "_pooled_small_batches"
    binned = meta[batch_col].astype(str).where(~meta[batch_col].isin(small), pooled_label)

    pooled_size = int((binned == pooled_label).sum())
    if pooled_size < min_size:
        raise ValueError(
            f"_bin_small_batches: {len(small)} batch value(s) in {batch_col!r} have fewer than "
            f"{min_size} cells each ({dict(counts[counts < min_size])}), and pooling ALL of them "
            f"together still only totals {pooled_size} cells (< {min_size}) -- scran's quickCluster "
            f"would still fail on this bucket. Needs manual investigation, not automatic binning."
        )
    print(f"[preprocess] _bin_small_batches: pooled {len(small)} batch value(s) with < {min_size} "
          f"cells each ({dict(counts[counts < min_size])}) into '{pooled_label}' ({pooled_size} cells) "
          f"for scran's blocking variable only -- the real {batch_col!r} column is unchanged.")
    return binned


def _read_full_ntc(indir: str):
    """Read the FULL NTC population from <indir>/NTC/ -- same files
    tmp/06_bayesDREAM_fit_ntc_combined.ipynb reads, including its own two
    alignment assertions. Returns (cell_meta, gene_meta, counts_sp) with
    counts_sp's rows in gene_meta's order and columns in cell_meta's order.
    """
    ntc_dir = os.path.join(indir, "NTC")
    cell_meta = pd.read_csv(os.path.join(ntc_dir, "cell_meta.csv"))
    gene_meta = pd.read_csv(os.path.join(ntc_dir, "gene_meta.csv"))
    counts_sp = sparse.load_npz(os.path.join(ntc_dir, "gene_counts.npz")).tocsr()
    counts_genes = np.load(os.path.join(ntc_dir, "gene_counts_genes.npy"), allow_pickle=True)
    counts_cells = np.load(os.path.join(ntc_dir, "gene_counts_cells.npy"), allow_pickle=True)

    if not (gene_meta["gene_id"].to_numpy() == counts_genes).all():
        raise ValueError("NTC/gene_meta.csv and NTC/gene_counts_genes.npy are not row-aligned.")
    if not (cell_meta["cell"].to_numpy() == counts_cells).all():
        raise ValueError("NTC/cell_meta.csv and NTC/gene_counts_cells.npy are not column-aligned.")

    return cell_meta, gene_meta, counts_sp


def preprocess(indir: str, outdir: str, cis_genes, batch_col: str = "batch", seed: int = 42,
               min_block_size: int = 100) -> None:
    os.makedirs(outdir, exist_ok=True)

    missing_genes = [g for g in cis_genes if g not in REPLOGLE_GENE_TO_ID]
    if missing_genes:
        raise ValueError(f"No Ensembl gene_id mapping for: {missing_genes} -- add to REPLOGLE_GENE_TO_ID.")
    gene_ids = [REPLOGLE_GENE_TO_ID[g] for g in cis_genes]

    # ---- full NTC population -- the canonical [gene panel, cell] frame everything else aligns to ----
    print(f"[preprocess] reading NTC/ (full NTC population, every NTC guide) from {indir} (read-only)...")
    ntc_cell_meta, ntc_gene_meta, ntc_counts_sp = _read_full_ntc(indir)
    canonical_gene_order = ntc_gene_meta["gene_id"].to_numpy()
    print(f"[preprocess] full NTC population: {ntc_counts_sp.shape[1]} cells x {ntc_counts_sp.shape[0]} genes")

    # ---- cell_meta_full/gene_meta_full: used ONLY to authoritatively label per-gene target cells ----
    print(f"[preprocess] reading cell_meta_full/gene_meta_full (target labels for the per-gene files)...")
    cell_meta_full = _read_parquet(os.path.join(indir, "cell_meta_full.parquet"))
    gene_meta_full = _read_parquet(os.path.join(indir, "gene_meta_full.parquet"))
    if "gene_id" not in gene_meta_full.columns:
        gene_meta_full = gene_meta_full.reset_index().rename(columns={"index": "gene_id"})

    # ---- per-gene target-only counts + meta, NTC (and any other stray) cells stripped ----
    per_gene_meta = []
    per_gene_sparse = []
    for gene, gene_id in zip(cis_genes, gene_ids):
        print(f"[preprocess] reading {gene} ({gene_id}) counts...")
        tgt_counts = _read_parquet(os.path.join(indir, gene_id, "counts.parquet")).set_index("gene")
        if set(tgt_counts.index) != set(canonical_gene_order):
            raise ValueError(f"{gene} ({gene_id})'s counts.parquet gene panel differs from NTC/'s "
                              f"-- can't align by position.")
        tgt_counts = tgt_counts.reindex(canonical_gene_order)  # row-align to the canonical gene order

        true_target_cells = set(cell_meta_full.loc[cell_meta_full["target"] == gene_id, "cell"])
        n_cols_before = tgt_counts.shape[1]
        keep_cols = [c for c in tgt_counts.columns if c in true_target_cells]
        n_dropped = n_cols_before - len(keep_cols)
        if n_dropped:
            print(f"[preprocess] {gene}: dropped {n_dropped}/{n_cols_before} cell(s) from "
                  f"{gene_id}/counts.parquet not labelled target={gene_id!r} in cell_meta_full "
                  f"(e.g. NTC cells mixed into the per-gene file -- see module docstring)")
        if not keep_cols:
            raise ValueError(f"{gene} ({gene_id}): no cells survived the true-target filter.")
        tgt_counts = tgt_counts[keep_cols]

        # meta rows, order-aligned to keep_cols (tgt_counts' own column order)
        meta_piece = cell_meta_full[cell_meta_full["cell"].isin(keep_cols)].set_index("cell").loc[keep_cols]
        meta_piece = meta_piece.reset_index()
        per_gene_meta.append(meta_piece)
        per_gene_sparse.append(sparse.csr_matrix(tgt_counts.values.astype(np.float32)))

    # ---- assemble combined counts/meta: [full NTC] union [chosen cis genes' true target cells] ----
    combined_counts = sparse.hstack([ntc_counts_sp] + per_gene_sparse).tocsr()
    combined_meta = pd.concat([ntc_cell_meta] + per_gene_meta, ignore_index=True, sort=False)
    if combined_meta["cell"].duplicated().any():
        dupes = combined_meta.loc[combined_meta["cell"].duplicated(), "cell"].tolist()
        raise ValueError(f"Duplicate cell(s) across NTC/ and/or per-gene counts files: {dupes[:5]}")
    if combined_counts.shape[1] != len(combined_meta):
        raise ValueError(f"combined_counts has {combined_counts.shape[1]} columns but combined_meta "
                          f"has {len(combined_meta)} rows -- alignment bug.")

    # ---- gene_meta, aligned to the canonical gene order (label-based, not positional) ----
    # 'gene_name' holds the REAL gene symbol here (not overwritten to equal
    # gene_id, unlike an earlier version of this script -- see STRATEGY.md
    # §11). Replogle's model is addressed by Ensembl ID everywhere downstream
    # (config.yaml's cis_gene_ids, add_cis_gene(<ensembl_id>) calls) via an
    # EXPLICIT `feature_name_col: gene_id` override (config.yaml/
    # generate_slurm.py's model.feature_name_col) rather than by shaping this
    # file to win bayesDREAM's identifier-column-priority cascade -- so there's
    # no need to sacrifice the real symbol to pin identity anymore.
    gm_indexed = gene_meta_full.set_index("gene_id")
    missing_genes_in_meta = pd.Index(canonical_gene_order).difference(gm_indexed.index)
    if len(missing_genes_in_meta):
        raise ValueError(f"{len(missing_genes_in_meta)} gene(s) in the NTC panel not found in gene_meta_full, "
                          f"e.g. {missing_genes_in_meta[:5].tolist()}")
    gene_meta_out = gm_indexed.loc[canonical_gene_order, ["gene_name"]].reset_index()

    # ---- sum_factor: quickCluster blocked by batch_col, computed ONCE on this combined population ----
    # Rare batch values (fewer cells than scran's own quickCluster min.size,
    # default 100) are pooled into one bucket for THIS blocking variable
    # only -- see _bin_small_batches' docstring. A separate column/temp
    # frame is used so the real `batch_col` in combined_meta (used
    # everywhere else downstream) is never touched.
    scran_block_col = f"_{batch_col}_scran_block"
    scran_meta = combined_meta.copy()
    scran_meta[scran_block_col] = _bin_small_batches(combined_meta, batch_col, min_size=min_block_size)

    print(f"[preprocess] computing scran sum factors on the combined [full NTC + {len(cis_genes)} cis genes] "
          f"population ({len(combined_meta)} cells, batch_col={batch_col!r}, "
          f"min_block_size={min_block_size})...")
    combined_meta["sum_factor"] = _compute_scran_sizefactors(
        combined_counts, scran_meta, batch_col=scran_block_col, seed=seed)

    # ---- write combined output ----
    combined_meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    sparse.save_npz(os.path.join(outdir, "gene_counts.npz"), combined_counts)
    gene_meta_out.to_csv(os.path.join(outdir, "gene_meta.csv"), index=False)
    print(f"[preprocess] wrote combined meta.csv/gene_counts.npz/gene_meta.csv: "
          f"{len(combined_meta)} cells x {combined_counts.shape[0]} genes -> {outdir}")

    # ---- NTC-only split, for the ntc_shared stage (mirrors Domingo's meta_ntc/counts_ntc) ----
    # The FULL NTC population (every NTC cell from NTC/, none dropped) --
    # NOT a further subset -- with the SAME recomputed sum_factor as
    # meta.csv (derived from combined_meta, not separately recomputed).
    ntc_mask = (combined_meta["target"] == "ntc").to_numpy()
    if int(ntc_mask.sum()) != ntc_counts_sp.shape[1]:
        raise ValueError(f"Expected all {ntc_counts_sp.shape[1]} NTC/ cells to survive into combined_meta, "
                          f"found {int(ntc_mask.sum())} -- alignment bug.")
    combined_meta.loc[ntc_mask].to_csv(os.path.join(outdir, "meta_ntc.csv"), index=False)
    sparse.save_npz(os.path.join(outdir, "gene_counts_ntc.npz"), combined_counts[:, ntc_mask])
    print(f"[preprocess] wrote NTC-only meta_ntc.csv/gene_counts_ntc.npz (full NTC population): "
          f"{int(ntc_mask.sum())} cells x {combined_counts.shape[0]} genes -> {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", required=True,
                         help="e.g. /cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/"
                              "Replogle_data/pr_data/for_bayesDREAM/K562_combined (read-only)")
    parser.add_argument("--outdir", required=True,
                         help="e.g. /cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/"
                              "Replogle2022/processed_Leah")
    parser.add_argument("--cis-genes", default=",".join(DEFAULT_CIS_GENES),
                         help=f"Comma-separated gene symbols (default: {','.join(DEFAULT_CIS_GENES)})")
    parser.add_argument("--batch-col", default="batch",
                         help="Blocking column for scran's quickCluster (default: 'batch', matching "
                              "adjust_ntc_sum_factor(covariates=['batch']) in the existing notebooks).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-block-size", type=int, default=100,
                         help="Any batch_col value with fewer cells than this is pooled into one "
                              "bucket for scran's quickCluster blocking variable ONLY (default: 100, "
                              "matching quickCluster's own min.size default) -- the real batch_col "
                              "column used everywhere else downstream is never touched. See "
                              "_bin_small_batches' docstring.")
    args = parser.parse_args()
    cis_genes = [g.strip() for g in args.cis_genes.split(",") if g.strip()]
    preprocess(args.indir, args.outdir, cis_genes, batch_col=args.batch_col, seed=args.seed,
               min_block_size=args.min_block_size)


if __name__ == "__main__":
    main()
