"""
One-time Morris raw-data preprocessing: gene name mapping, guide_assignment
alignment+transpose, guide_meta/guide_target construction, cell meta
construction. Factors out steps 1-5 of the original per-gene Morris
preprocessing script (previously re-derived inline by every single-gene
run) into its own script that runs ONCE.

`sum_factor` is scran (quickCluster + computeSumFactors, block='lane'),
computed ONCE here on the FULL dataset via
common/compute_scran_sum_factor.py's `_compute_scran_sizefactors` -- the
SAME function every per-gene stage used to call separately (previously
recomputed per-gene, as 'sum_factor_new'; see that module's docstring for
why a single shared computation, used everywhere downstream, matters:
fit_ntc's alpha_y_mult/alpha_x_prefit and fit_cis/fit_trans's sum_factor_adj
are composed multiplicatively, so they must be calibrated against the SAME
sum_factor column). Requires rpy2 + R (scran/Matrix/SingleCellExperiment/
S4Vectors) in whichever env runs this script -- NOT necessarily the same
env generate_slurm.py's jobs run in, since this is a one-time, run-once-
locally-or-on-a-login-node step, unlike per-gene compute_scran calls used
to be. Previously this column came from the raw `cellmeta_wSF.csv`'s own
`sizeFactor` (an externally-computed factor from the original Morris 2023
dataset's own processing, not scran/lane-blocked the way this pipeline's
own per-gene sum_factor_new was) -- replaced now that sum_factor_new no
longer exists as a separate per-gene recomputation.

Run this manually, once, before generate_slurm.py. Not part of the SLURM
pipeline itself -- analogous to Domingo's R preprocessing, which also runs
once, upstream, outside this pipeline (see domingo/README.md).

Cell order in the written meta.csv is authoritative: every other output
file is explicitly reordered to match it (rather than trusting the raw
inputs' own row order, which the original script only spot-checked with a
commented-out assertion -- this script checks for real and raises if
anything doesn't line up), with ONE exception: guide_assignment.npz's
columns are reordered to meta.csv's cell order POSITIONALLY, not by looking
up guide_assignment_cells.npy's own cell-name labels -- those labels are
confirmed unreliable/garbled (see the validated GFI1B reference script,
which discovers this same thing and discards them for cell_meta's own names
instead, calling it "hacky because the cell names are messed up"). The only
check possible here is that guide_assignment.npz's column count matches
meta.csv's row count exactly (raises if not) -- there's no independent join
key to verify the ORDER beyond that, so this is a trusted assumption, not a
proven one; gene_counts_cells.npy's labels, by contrast, ARE reliable and
gene_counts.npz is aligned by real name-based lookup.

Usage
-----
    python preprocess.py --indir <raw data dir> --outdir <clean output dir>

Writes to <outdir>:
    meta.csv               cell metadata incl. sum_factor (scran, computed
                            once here on the full dataset -- see above)
                            -- fixed row order, everything else aligns to it
    gene_counts.npz         sparse (genes x cells), columns reordered to
                            match meta.csv's cell order exactly
    gene_meta.csv           gene_id, gene_name -- row order matches
                            gene_counts.npz's rows
    guide_assignment.npy    DENSE (cells x guides), rows reordered to match
                            meta.csv's cell order (bayesDREAM's high-MOI mode
                            requires dense, see docs/HIGH_MOI_GUIDE.md)
    guide_meta.csv          guide -- column order matches guide_assignment.npy
    guide_target.csv        guide, target (gene NAMES, not Ensembl IDs)

bayesDREAM aligns a bare (unlabelled) sparse counts matrix to `meta` PURELY
BY POSITION (confirmed: bayesDREAM/core.py:331-336) -- gene_counts.npz's
column order matching meta.csv's row order exactly is a hard correctness
requirement, not a convenience.
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


def _infer_target(guide_name: str) -> str:
    """For guides missing from guide_target_meta: NTC-* -> 'ntc', GENE-* -> GENE."""
    prefix = guide_name.split("-")[0]
    return "ntc" if prefix == "NTC" else prefix


def preprocess(indir: str, outdir: str, batch_col: str = "lane", seed: int = 42) -> None:
    os.makedirs(outdir, exist_ok=True)

    cell_meta = pd.read_csv(os.path.join(indir, "cellmeta_wSF.csv"))
    gene_meta = pd.read_csv(os.path.join(indir, "gene_meta.csv"))
    guide_target_meta = pd.read_csv(os.path.join(indir, "guide_target_meta.csv"))

    gene_counts_sp = sparse.load_npz(os.path.join(indir, "gene_counts.npz"))
    gene_counts_genes = np.load(os.path.join(indir, "gene_counts_genes.npy"), allow_pickle=True)
    gene_counts_cells = np.load(os.path.join(indir, "gene_counts_cells.npy"), allow_pickle=True)

    guide_assignment_sp = sparse.load_npz(os.path.join(indir, "guide_assignment.npz"))
    guide_assignment_guides = np.load(os.path.join(indir, "guide_assignment_guides.npy"), allow_pickle=True)
    # guide_assignment_cells.npy is loaded ONLY to fail loudly if its length
    # doesn't match -- its actual string labels are known-unreliable ("cell
    # names are messed up", confirmed against the validated GFI1B reference
    # script), so we deliberately do NOT use them for a by-name lookup like
    # gene_counts_cells.npy below. guide_assignment_sp's raw column order is
    # trusted to already match cell_meta's row order positionally (confirmed:
    # shape is (n_guides x 94075), exactly cell_meta's row count, and the
    # reference script relies on this same positional assumption).
    guide_assignment_cells = np.load(os.path.join(indir, "guide_assignment_cells.npy"), allow_pickle=True)

    if not np.array_equal(gene_counts_genes, gene_meta["V1"].values):
        raise ValueError(
            "gene_counts_genes.npy and gene_meta.csv['V1'] are not row-aligned -- "
            "the original preprocessing scripts only spot-checked this with a "
            "commented-out assertion; it's a hard requirement here."
        )

    ensg_to_name = dict(zip(gene_meta["V1"], gene_meta["V2"]))

    # ---- fixed cell order: cell_meta's own row order, everything else aligns to it ----
    cell_order = cell_meta["cell"].values

    gc_cell_to_idx = {c: i for i, c in enumerate(gene_counts_cells)}
    missing_gc = [c for c in cell_order if c not in gc_cell_to_idx]
    if missing_gc:
        raise ValueError(
            f"{len(missing_gc)} cell(s) in cellmeta_wSF.csv missing from "
            f"gene_counts_cells.npy, e.g. {missing_gc[:5]}"
        )
    gc_col_order = np.array([gc_cell_to_idx[c] for c in cell_order])
    gene_counts_aligned = gene_counts_sp[:, gc_col_order].tocsr()

    # Positional identity, NOT a by-name lookup -- see the load comment above.
    if guide_assignment_sp.shape[1] != len(cell_order):
        raise ValueError(
            f"guide_assignment.npz has {guide_assignment_sp.shape[1]} columns, "
            f"but cellmeta_wSF.csv has {len(cell_order)} cells -- positional "
            f"alignment assumption doesn't hold, can't proceed without a real "
            f"cell-name join key (guide_assignment_cells.npy's labels are known-unreliable)."
        )
    ga_col_order = np.arange(len(cell_order))
    guide_assign = (
        guide_assignment_sp[:, ga_col_order]
        .T
        .toarray()
        .astype(np.float32)
    )

    # ---- guide_meta / guide_target ----
    gt = guide_target_meta.rename(columns={"grna_id": "guide", "response_id": "target"}).copy()
    gt["target"] = gt["target"].map(ensg_to_name).fillna(gt["target"])

    missing_guides = guide_assignment_guides[~np.isin(guide_assignment_guides, gt["guide"])]
    extra_rows = pd.DataFrame({
        "guide": missing_guides,
        "target": [_infer_target(g) for g in missing_guides],
    })
    gt = pd.concat([gt, extra_rows], ignore_index=True)

    guide_meta_bd = pd.DataFrame({"guide": guide_assignment_guides})

    # ---- cell metadata ----
    meta = cell_meta.copy()
    lib_sizes = np.asarray(gene_counts_aligned.sum(axis=0)).flatten()
    meta["lib_factor"] = lib_sizes / np.median(lib_sizes)

    # scran, once, on the FULL dataset -- see module docstring. gene_counts_aligned
    # is already (genes x cells) with columns in `meta`'s row order (both derive
    # from cell_order = cell_meta['cell'].values), so this is a straight positional
    # pairing, same as every other counts/meta alignment in this script.
    print(f"[preprocess] computing scran sum factors on the full dataset (batch_col={batch_col!r})...")
    meta["sum_factor"] = _compute_scran_sizefactors(
        gene_counts_aligned, meta, batch_col=batch_col, seed=seed,
    )

    # ---- feature metadata ----
    bd_gene_meta = pd.DataFrame({
        "gene_id": gene_meta["V1"].values,
        "gene_name": gene_meta["V2"].values,
    })

    # ---- write ----
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    sparse.save_npz(os.path.join(outdir, "gene_counts.npz"), gene_counts_aligned)
    bd_gene_meta.to_csv(os.path.join(outdir, "gene_meta.csv"), index=False)
    np.save(os.path.join(outdir, "guide_assignment.npy"), guide_assign)
    guide_meta_bd.to_csv(os.path.join(outdir, "guide_meta.csv"), index=False)
    gt.to_csv(os.path.join(outdir, "guide_target.csv"), index=False)

    print(f"[preprocess] {len(meta)} cells x {gene_counts_aligned.shape[0]} genes x "
          f"{guide_assign.shape[1]} guides -> {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-col", default="lane",
                         help="Blocking column for scran's quickCluster (default: 'lane', matching "
                              "config.yaml's sum_factor.batch_col -- keep these in sync).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(args.indir, args.outdir, batch_col=args.batch_col, seed=args.seed)


if __name__ == "__main__":
    main()
