"""
Compute scran-based per-cell sum factors on a model's CURRENT cell set,
via rpy2. Matches the Morris preprocessing script verbatim
(quickCluster + computeSumFactors, batch-aware), and must be called AFTER
the model's cell subsetting is final (i.e. after high-MOI construction with
cis_gene set, or after add_cis_gene() for low-MOI) -- it is a genuinely
per-cell-subset computation, not something that can be precomputed once and
shared like fit_ntc's alpha (see morris/README.md for why the shared-ntc
design still applies only to alpha, not to sum factors).

Domingo does NOT need this: its 'sum_factor' column is already scran-derived
once, upstream, outside this pipeline entirely (see the R preprocessing
script that produces cell_meta.csv, in domingo/README.md) -- calling this
module for Domingo would be redundant and wrong (different clustering
strategy: guide-identity clusters with ref.clust='NTC', not quickCluster).

Requires rpy2 with R packages: scran, Matrix, SingleCellExperiment,
S4Vectors, available in whichever conda env this runs under (confirm scran
etc. are installed in bayesdream_cpu / bayesdream_rocm on Dardel before
relying on this).
"""

import numpy as np
from scipy import sparse


def compute_scran_sum_factor(
    model,
    modality_name: str = None,
    batch_col: str = "lane",
    sum_factor_col_out: str = "sum_factor_new",
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Compute scran sum factors on model's current cells, write into
    model.get_modality(modality_name).sum_factors[sum_factor_col_out].

    Mirrors this rpy2 block from the Morris preprocessing script exactly:

        clusters <- quickCluster(sce, block = batch)
        sce <- computeSumFactors(sce, clusters = clusters)
        sum_factors <- sizeFactors(sce)

    Falls back to `computeSumFactors(sce, clusters=NULL)` if `batch_col` is
    not a column in model.meta (same fallback as the reference script).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    modality_name = modality_name or model.primary_modality
    mod = model.get_modality(modality_name)

    counts = mod.counts
    # Modality stores (features, cells) when cells_axis==1, else (cells, features) --
    # scran wants (features, cells).
    if mod.cells_axis != 1:
        counts = counts.T
    counts_sp = counts.tocsc() if sparse.issparse(counts) else sparse.csc_matrix(counts)

    meta_sub = model.meta.copy()

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
        ro.globalenv["cell_meta"] = meta_sub
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

    mod.sum_factors[sum_factor_col_out] = sum_factors
    # ALSO write into model.meta directly, not just mod.sum_factors --
    # adjust_ntc_sum_factor()/refit_sumfactor() (bayesDREAM/core.py) always
    # look up sum_factor_col_old via self.get_modality(self.primary_modality),
    # with no way to point them at a different modality. When modality_name
    # here is 'cis' (the deferred-cis_gene, cis_only-subset pipeline's own
    # data, needed because the primary 'gene' modality has 0 features at
    # this point -- see morris/generate_slurm.py's render_cis_stage_config),
    # that lookup would otherwise miss the column entirely. meta lookup is
    # checked FIRST in adjust_ntc_sum_factor's own fallback logic, so this
    # is a no-op risk-wise for the normal (primary-modality) case too.
    model.meta[sum_factor_col_out] = mod.sum_factors.loc[
        model.meta["cell"].values, sum_factor_col_out
    ].values
    if verbose:
        print(f"[compute_scran_sum_factor] {modality_name}: wrote '{sum_factor_col_out}' "
              f"for {len(sum_factors)} cells (batch_col={batch_col!r})")
