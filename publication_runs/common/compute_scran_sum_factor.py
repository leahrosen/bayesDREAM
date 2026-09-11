"""
Compute scran-based per-cell sum factors via rpy2 (quickCluster +
computeSumFactors, batch-aware). Matches the Morris preprocessing script
verbatim.

As of 2026-08, computed ONCE on Morris's FULL dataset in
morris/preprocess.py (writes the 'sum_factor' column every downstream stage
reads by default) -- NOT per-gene-subset anymore. Previously this was
recomputed separately for every gene's own NTC+target cell subset (via
config_utils.apply_sum_factor_adjustments's 'compute_scran' step, called
from subset_per_gene.py), which meant fit_ntc's alpha_y_mult/alpha_x_prefit
(estimated once, from either the shared ntc_shared fit or each primary
gene's own fit_ntc, against whatever 'sum_factor' existed at THAT time) and
fit_cis/fit_trans's sum_factor_adj (derived from a freshly, separately
recomputed 'sum_factor_new' per gene) were calibrated against two DIFFERENT
normalizations -- since bayesDREAM composes them multiplicatively
(`mu_final = mu_y * alpha_y * sum_factor`, see bayesDREAM/fitting/trans.py;
`mu_obs = alpha_x * x_true * sum_factor`, bayesDREAM/fitting/cis.py),
alpha_y_mult/alpha_x_prefit's per-lane correction (fit against sizeFactor)
would not necessarily be valid against sum_factor_adj (derived from a
separately-computed sum_factor_new). Computing scran once, up front, and
using that SAME 'sum_factor' column everywhere (mirroring Domingo's design,
where adjust_ntc_sum_factor()/refit_sumfactor() are provably anchored to
each covariate group's own NTC baseline -- see their docstrings) closes
that gap. `_compute_scran_sizefactors` below is the shared core so
`compute_scran_sum_factor` (still available for a per-model/per-subset
recomputation, if some future use genuinely needs one) and
morris/preprocess.py's one-time full-dataset call go through the exact same
R logic, not two copies that could drift apart.

Requires rpy2 with R packages: scran, Matrix, SingleCellExperiment,
S4Vectors, available in whichever conda env this runs under (confirm scran
etc. are installed in bayesdream_cpu / bayesdream_rocm on Dardel before
relying on this).

Domingo does NOT need this at all: its 'sum_factor' column is already
scran-derived once, upstream, outside this pipeline entirely (see the R
preprocessing script that produces cell_meta.csv, in domingo/README.md) --
a different clustering strategy besides (guide-identity clusters with
ref.clust='NTC', not quickCluster).
"""

import numpy as np
import pandas as pd
from scipy import sparse


def _compute_scran_sizefactors(
    counts_sp,
    meta_df: pd.DataFrame,
    batch_col: str = "lane",
    seed: int = 42,
) -> np.ndarray:
    """Core rpy2 call: counts_sp is (features x cells) CSC, meta_df is a
    cells-aligned DataFrame (row order must match counts_sp's columns
    exactly -- purely positional, like every other counts/meta pairing in
    this pipeline). Returns a 1D numpy array of per-cell size factors, same
    order as counts_sp's columns.

    Mirrors this rpy2 block from the Morris preprocessing script exactly:

        clusters <- quickCluster(sce, block = batch)
        sce <- computeSumFactors(sce, clusters = clusters)
        sum_factors <- sizeFactors(sce)

    Falls back to `computeSumFactors(sce, clusters=NULL)` if `batch_col` is
    not a column in meta_df (same fallback as the reference script). NOT
    NTC-referenced (no `ref.clust` argument) -- unlike Domingo's own
    `calculateSumFactors(..., clusters=myclusts, ref.clust='NTC')`, this is
    plain unsupervised `quickCluster`, blocked by `batch_col` only.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    counts_sp = counts_sp.tocsc() if sparse.issparse(counts_sp) else sparse.csc_matrix(counts_sp)

    conv = default_converter + numpy2ri.converter + pandas2ri.converter

    ro.r("""
    suppressMessages({
        library(scran)
        library(Matrix)
        library(SingleCellExperiment)
        library(S4Vectors)
    })
    """)

    with localconverter(conv):
        ro.globalenv["i"] = counts_sp.indices.astype(np.int32)
        ro.globalenv["p"] = counts_sp.indptr.astype(np.int32)
        ro.globalenv["x"] = counts_sp.data.astype(float)
        ro.globalenv["dims"] = np.array(counts_sp.shape, dtype=np.int32)
        ro.globalenv["cell_meta"] = meta_df
        ro.globalenv["batch_var"] = batch_col
        ro.globalenv["seed_val"] = seed

    ro.r("""
    counts_mat <- methods::new(
      "dgCMatrix",
      i = as.integer(i),
      p = as.integer(p),
      x = as.numeric(x),
      Dim = as.integer(dims)
    )

    sce <- SingleCellExperiment(
      assays = list(counts = counts_mat),
      colData = S4Vectors::DataFrame(cell_meta)
    )

    set.seed(seed_val)

    if (batch_var %in% colnames(colData(sce))) {
        batch <- factor(colData(sce)[[batch_var]])
        clusters <- quickCluster(sce, block = batch)
        sce <- computeSumFactors(sce, clusters = clusters)
    } else {
        sce <- computeSumFactors(sce, clusters = NULL)
    }

    sum_factors_out <- sizeFactors(sce)
    """)

    with localconverter(conv):
        sum_factors = np.asarray(ro.globalenv["sum_factors_out"])

    return sum_factors


def compute_scran_sum_factor(
    model,
    modality_name: str = None,
    batch_col: str = "lane",
    sum_factor_col_out: str = "sum_factor_new",
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Compute scran sum factors on model's CURRENT cells, write into
    model.get_modality(modality_name).sum_factors[sum_factor_col_out] (and
    model.meta, so it survives a subset_per_gene.py-style CSV write -- see
    the note below).

    NOTE: no longer called anywhere in Morris's actual SLURM pipeline as of
    2026-08 (morris/preprocess.py now computes 'sum_factor' once, up front,
    on the full dataset -- see this module's docstring) -- kept as a public
    function for any future genuinely-per-subset use, going through the
    same `_compute_scran_sizefactors` core.
    """
    modality_name = modality_name or model.primary_modality
    mod = model.get_modality(modality_name)

    counts = mod.counts
    # Modality stores (features, cells) when cells_axis==1, else (cells, features) --
    # scran wants (features, cells).
    if mod.cells_axis != 1:
        counts = counts.T

    meta_sub = model.meta.copy()

    sum_factors = _compute_scran_sizefactors(counts, meta_sub, batch_col=batch_col, seed=seed)

    mod.sum_factors[sum_factor_col_out] = sum_factors
    # ALSO write into model.meta directly, not just mod.sum_factors -- this
    # is what makes the value actually PERSIST to disk if a caller then does
    # model.meta.to_csv(...) (e.g. subset_per_gene.py-style usage). Every
    # downstream per-gene stage rebuilds a fresh model.meta from that file
    # and re-derives modality.sum_factors from its *sum_factor* columns at
    # __init__/add_cis_gene() time (bayesDREAM/core.py's
    # _init_sum_factors()) -- writing here only would be silently lost the
    # moment this process exits.
    model.meta[sum_factor_col_out] = mod.sum_factors.loc[
        model.meta["cell"].values, sum_factor_col_out
    ].values
    if verbose:
        print(f"[compute_scran_sum_factor] {modality_name}: wrote '{sum_factor_col_out}' "
              f"for {len(sum_factors)} cells (batch_col={batch_col!r})")
