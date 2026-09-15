"""
One-time Domingo raw-data preprocessing: filters CRISPRi/CRISPRa pseudo-gene
rows, renames CCDC173 -> CFAP210, computes scran sum factors (guide-identity
clusters, ref.clust='NTC' -- see below), builds gene_meta from the GTF, and
writes the cell_meta.csv/gene_counts.csv/gene_meta.csv schema
domingo/config.yaml expects.

Ports (and replaces the need to separately run) the R preprocessing script
this pipeline was originally validated against to Python + rpy2 -- so the
whole pipeline runs from one language/environment, matching Morris's own
preprocess.py. Faithfully reproduces that R script's logic EXCEPT the
adjustment_factor/clustered.sum.factor.adj guide-normalization step, which
is deliberately dropped -- see "What's different from the original R
script" below.

IMPORTANT: uses a DIFFERENT scran clustering scheme than
common/compute_scran_sum_factor.py (Morris's quickCluster-based, computed
per-model-cell-subset). Domingo clusters by GUIDE IDENTITY (`guide_crispr`,
with any guide name containing "NTC" collapsed to a single 'NTC' cluster)
with ref.clust='NTC', computed ONCE on the full raw dataset before any
model exists. Do not reuse compute_scran_sum_factor.py for this -- its own
docstring already says so explicitly.

Run this manually, once, before generate_slurm.py -- analogous to
morris/preprocess.py.

Inputs (in --indir), confirmed via header/content inspection on Dardel
(not guessed):
    domingo_cellmeta.txt.gz   raw colData(sce) export -- COMMA-separated
                              despite the .txt extension. Has 'lane' already
                              derived (sub("_.*", "", L_cell_barcode)) but
                              NOT cell/target/guide/sum_factor -- those are
                              derived here.
    domingo_GEXcounts.csv     raw counts(sce) export (genes x cells),
                              UNFILTERED -- confirmed to still include
                              CRISPRi/CRISPRa pseudo-gene rows and
                              un-renamed CCDC173 (exactly what
                              calculateSumFactors(counts(sce), ...) needs
                              in the original R script -- computed BEFORE
                              CRISPRi/CRISPRa removal, not after). No
                              header label for the gene-name column (R
                              write-style row-names convention) --
                              index_col=0 handles this.

Writes to --outdir:
    cell_meta.csv       all raw columns + clustered.sum.factor/cell/target/
                        sum_factor/guide
    gene_counts.csv     genes x cells, CRISPRi/CRISPRa dropped, CCDC173
                        renamed CFAP210, 'gene' column header (matches
                        toydata/gene_counts.csv's existing, working format)
    gene_meta.csv       attr, chr, start, end, strand, gene_id,
                        gene_id.version, gene_name, gene_type -- filtered to
                        genes present in gene_counts.csv (matches
                        toydata/gene_meta.csv's existing schema exactly)
    cell_meta_ntc.csv       NTC-only subset of cell_meta.csv (target=='ntc')
    gene_counts_ntc.npz     NTC-only subset of gene_counts.csv, sparse --
                            for ntc_shared, which only ever fits on NTC
                            cells (use_all_cells=False, the low-MOI
                            default) but previously still had to load+
                            classify the full 20001-cell dataset to get
                            there. No per-gene reduction here (ntc_shared
                            estimates alpha_y for every gene) -- cells only.

What's different from the original R script
---------------------------------------------
Does NOT compute adjustment_factor/clustered.sum.factor.adj (the guide-level
mean-sum-factor-vs-NTC-baseline normalization step). That column is
ALWAYS renamed to adjustment_factor_old and never read
(config_utils.apply_sum_factor_adjustments's docstring / domingo/README.md's
"Why apply_sum_factor_adjustments renames adjustment_factor" -- it's
provably redundant with model.adjust_ntc_sum_factor(), which recomputes an
equivalent column itself), so it's dropped here rather than reimplemented
for no downstream use.

Usage
-----
    python preprocess.py --indir <raw data dir> --outdir <clean output dir> \\
        [--gtf <path to gencode annotation gtf.gz>]

Requires rpy2 with the R package `scran` available in whichever conda env
this runs under (confirm on Dardel before relying on it -- this is a fresh
dependency, install/verify separately from Morris's own rpy2 needs even
though both use bayesdream_cpu).
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse

_DEFAULT_GTF = (
    "/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/refdata/"
    "refdata-gex-GRCh38-2024-A/genes/gencode.v44.annotation.gtf.gz"
)

_DROP_GENES = ["CRISPRi", "CRISPRa"]
_RENAME_GENES = {"CCDC173": "CFAP210"}


def _compute_domingo_sum_factors(counts_raw: pd.DataFrame, myclusts: np.ndarray) -> np.ndarray:
    """scran::calculateSumFactors(counts(sce), clusters=myclusts, ref.clust='NTC')
    -- on the RAW (unfiltered) counts, matching the original R script's
    exact call order (BEFORE CRISPRi/CRISPRa removal). counts_raw is
    (genes x cells); small enough here (~94 genes) to convert as a plain
    dense R matrix, no sparse dgCMatrix construction needed (contrast
    common/compute_scran_sum_factor.py, sized for genome-wide matrices)."""
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    conv = default_converter + numpy2ri.converter + pandas2ri.converter
    ro.r("suppressMessages(library(scran))")

    with localconverter(conv):
        ro.globalenv["counts_mat"] = counts_raw.values.astype(float)
        ro.globalenv["clusters_vec"] = np.asarray(myclusts, dtype=str)

    ro.r("sum_factors_out <- calculateSumFactors(counts_mat, clusters=clusters_vec, ref.clust='NTC')")

    with localconverter(conv):
        sum_factors = np.asarray(ro.globalenv["sum_factors_out"])
    return sum_factors


def _build_gene_meta(gtf_path: str, keep_gene_names) -> pd.DataFrame:
    """Parses gene_id/gene_id.version/gene_name/gene_type from GTF rows
    where feature=='gene', filtered to `keep_gene_names` -- mirrors the R
    script's regex-based attribute extraction and toydata/gene_meta.csv's
    exact column set/order."""
    gtf = pd.read_csv(
        gtf_path, sep="\t", header=None,
        names=["chr", "source", "feature", "start", "end", "score", "strand", "frame", "attr"],
        comment="#", compression="gzip", low_memory=False,
    )
    genes = gtf.loc[gtf["feature"] == "gene", ["attr", "chr", "start", "end", "strand"]].copy()

    def extract(key: str) -> pd.Series:
        return genes["attr"].str.extract(rf'{key} "([^"]+)"', expand=False)

    genes["gene_id.version"] = extract("gene_id")
    genes["gene_id"] = genes["gene_id.version"].str.replace(r"\..*$", "", regex=True)
    genes["gene_name"] = extract("gene_name")
    genes["gene_type"] = extract("gene_type")

    genes = genes[genes["gene_name"].isin(set(keep_gene_names))]
    return genes[["attr", "chr", "start", "end", "strand", "gene_id", "gene_id.version", "gene_name", "gene_type"]]


def preprocess(indir: str, outdir: str, gtf_path: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    cell_meta = pd.read_csv(os.path.join(indir, "domingo_cellmeta.txt.gz"), compression="gzip")
    gene_counts_raw = pd.read_csv(os.path.join(indir, "domingo_GEXcounts.csv"), index_col=0)

    # ---- align gene_counts columns to cell_meta's row order, BY NAME ----
    # (real lookup, not positional trust -- domingo_GEXcounts.csv's column
    # headers are literal L_cell_barcode-format strings, confirmed reliable,
    # unlike Morris's guide_assignment_cells.npy labels.)
    cell_order = cell_meta["L_cell_barcode"].values
    missing = [c for c in cell_order if c not in gene_counts_raw.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} cell(s) in domingo_cellmeta.txt.gz missing from "
            f"domingo_GEXcounts.csv, e.g. {missing[:5]}"
        )
    gene_counts_raw = gene_counts_raw[cell_order]

    # ---- scran sum factors: guide-identity clusters, on RAW (unfiltered) counts ----
    myclusts = cell_meta["guide_crispr"].astype(str).to_numpy()
    myclusts = np.where(pd.Series(myclusts).str.contains("NTC").to_numpy(), "NTC", myclusts)
    sum_factors = _compute_domingo_sum_factors(gene_counts_raw, myclusts)

    cell_meta = cell_meta.copy()
    cell_meta["clustered.sum.factor"] = sum_factors
    cell_meta["cell"] = cell_meta["L_cell_barcode"]
    cell_meta["target"] = cell_meta["gene"]
    cell_meta["sum_factor"] = cell_meta["clustered.sum.factor"]
    cell_meta["guide"] = cell_meta["short_ID"]

    # ---- gene counts: drop CRISPRi/CRISPRa, rename CCDC173 -> CFAP210 ----
    gene_counts = gene_counts_raw.drop(index=_DROP_GENES, errors="ignore").rename(index=_RENAME_GENES)

    gene_meta = _build_gene_meta(gtf_path, gene_counts.index)

    # ---- NTC-only subset, for ntc_shared (which only ever fits on NTC cells) ----
    ntc_mask = (cell_meta["target"] == "ntc").to_numpy()
    cell_meta_ntc = cell_meta.loc[ntc_mask].reset_index(drop=True)
    gene_counts_ntc = sparse.csr_matrix(gene_counts.loc[:, ntc_mask].values)

    # ---- write ----
    cell_meta.to_csv(os.path.join(outdir, "cell_meta.csv"), index=False)
    gene_counts.to_csv(os.path.join(outdir, "gene_counts.csv"), index_label="gene")
    gene_meta.to_csv(os.path.join(outdir, "gene_meta.csv"), index=False)
    cell_meta_ntc.to_csv(os.path.join(outdir, "cell_meta_ntc.csv"), index=False)
    sparse.save_npz(os.path.join(outdir, "gene_counts_ntc.npz"), gene_counts_ntc)

    print(f"[preprocess] {len(cell_meta)} cells x {len(gene_counts)} genes "
          f"({len(gene_counts_raw) - len(gene_counts)} CRISPRi/CRISPRa row(s) dropped) -> {outdir}")
    print(f"[preprocess] NTC-only subset: {len(cell_meta_ntc)} cells x {gene_counts_ntc.shape[0]} genes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gtf", default=_DEFAULT_GTF)
    args = parser.parse_args()
    preprocess(args.indir, args.outdir, args.gtf)


if __name__ == "__main__":
    main()
