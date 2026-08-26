"""
Automates save_model_for_plotting() + hill_eval backfilling for Replogle.

Unlike comparative/reconstruct_export.py (Domingo/Morris), there is no
rendered YAML config to replay -- Replogle's pipeline is a hand-written
papermill notebook (10_bayesDREAM_fit_trans_<GENE>.ipynb), not the
publication_runs/ config system. This module instead re-derives the model
directly from the raw parquet inputs, PORTED (not imported) from that
notebook's own load_gene_model_inputs()/build_gene_to_id()/build_trans_model()
functions, with two deliberate deviations: the final model.fit_trans(...)
call is swapped for model.load_trans_fit(...) (reload the already-completed
posterior, don't re-fit), and load_ntc_fit() drops the notebook's own
lean=True (save_model_for_plotting() needs the full NTC posterior to
re-save it -- see reconstruct_model()'s comment on this).

Also unlike Domingo/Morris, the backfilled trans_feature_summary_gene.csv is
NOT written in place into OUTDIR/<label>/ -- that directory belongs to
whoever ran the original Replogle fits, and you may only have read access
there (confirmed via a real PermissionError, 2026-08-26). Everything this
module writes (the summary CSV and the save_model_for_plotting() export)
goes into REPLOGLE.save_for_plotting_dir_fn()'s directory instead (your own
Comparative/input/ tree) -- see comparative/datasets.py's REPLOGLE.run_dir_fn
for why that's also where trans_param_compare.py reads from for this dataset.

ASSUMPTION -- please confirm before running reconstruct_and_export_all():
10_bayesDREAM_fit_trans_MYB.ipynb is a papermill-parameterized template, and
its 6 siblings (10_bayesDREAM_fit_trans_{GFI1B,NFE2,HHEX,RUNX1,TET2,IKZF1}.ipynb)
are the SAME template with only the `gene`/`NITERS_TRANS` parameter cell
changed. This module hardcodes every other constant from the MYB copy
(INDIR/WD/NTC_FIT/CIS_FIT/OUTDIR paths, MIN_LOG2_MU_NTC_TRANS=-4.0,
function_type="single_hill", HILL_LOG2FC_TARGETS=(-1.0,)) across ALL 7
genes on that assumption. If any of the other 6 notebooks was hand-edited
differently (a different MIN_LOG2_MU_NTC_TRANS, a different function_type,
etc.), this will silently reconstruct that gene's model slightly wrong.

Usage
-----
    from comparative.reconstruct_export_replogle import reconstruct_and_export_all
    reconstruct_and_export_all()          # all 7 Replogle cis genes
    reconstruct_and_export_all(genes=['GFI1B'])   # just one
"""

import gc
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import bayesDREAM as _bayesdream_pkg
from bayesDREAM import bayesDREAM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_bayesdream_pkg.__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from save_for_plotting import save_model_for_plotting  # noqa: E402

from .datasets import REPLOGLE, REPLOGLE_GENE_TO_ID  # noqa: E402
from .hill_eval import add_log2fc_at_columns, HILL_LOG2FC_TARGETS, is_already_backfilled  # noqa: E402

# ── Paths / constants, from 10_bayesDREAM_fit_trans_MYB.ipynb (see the
# module-level ASSUMPTION note above) ────────────────────────────────────────
INDIR = "/cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/pr_data/for_bayesDREAM/K562_combined"
WD = "/cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/bayesDREAM"
NTC_FIT = "/cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/bayesDREAM/output/combined_ntc"
CIS_FIT = os.path.join(WD, "output/fit_cis")
OUTDIR = os.path.join(WD, "output/fit_trans")
MIN_LOG2_MU_NTC_TRANS = -4.0

_shared_cache: Dict = {}


def _read_parquet(path: str) -> pd.DataFrame:
    """pd.read_parquet(path, engine='fastparquet'), not the pyarrow default.

    PyArrow 19.0.0 (confirmed on this cluster's pyroenv, 2026-08-26) fails to
    read Replogle's parquet inputs with 'OSError: Repetition level histogram
    size mismatch' -- a PyArrow-internal bug in its Parquet page-index
    validation, reproduced identically via both pd.read_parquet's default
    (Dataset/Scanner API) and the older pyarrow.parquet.ParquetFile.read()
    path, so it isn't specific to one PyArrow entry point. fastparquet is a
    separate, pure-Python/numba implementation with no shared code path, so
    it isn't affected; confirmed working against the same file this session.
    If PyArrow ever fixes this (or the file is rewritten with an older
    PyArrow), this indirection can just be reverted to a plain
    pd.read_parquet call.
    """
    return pd.read_parquet(path, engine='fastparquet')


def _load_shared_inputs() -> Dict:
    """Load + cache the genome-wide inputs shared by every cis gene:
    cell_meta_full, gene_meta_full, technical_group_code mapping,
    kept_trans_genes -- ported verbatim from
    10_bayesDREAM_fit_trans_MYB.ipynb's cells 3/4/6.
    """
    if _shared_cache:
        return _shared_cache

    cell_meta_full = _read_parquet(os.path.join(INDIR, "cell_meta_full.parquet"))
    gene_meta_full = _read_parquet(os.path.join(INDIR, "gene_meta_full.parquet")).set_index("gene_id")
    gene_meta_full["feature_id"] = gene_meta_full.index
    gene_meta_full["gene_symbol"] = gene_meta_full["gene_name"]
    gene_meta_full["gene_name"] = gene_meta_full.index

    code_mapping = pd.read_csv(os.path.join(NTC_FIT, "technical_group_code_mapping.csv"))
    # guard against '1' becoming '1.0' if pandas read batch as float
    batch_str = code_mapping['batch'].astype('Int64').astype(str)
    exp_str = code_mapping['experiment'].astype(str).str.strip()
    code_mapping['key'] = exp_str + '-' + batch_str
    dupes = code_mapping.loc[code_mapping['key'].duplicated(keep=False)]
    assert dupes.empty, f"non-unique keys:\n{dupes}"
    group_dict = dict(zip(code_mapping['key'], code_mapping['technical_group_code']))
    keys = cell_meta_full['batch'].astype(str).str.strip()
    cell_meta_full['technical_group_code'] = keys.map(group_dict).astype('Int64')
    missing = sorted(set(keys) - set(group_dict))
    if missing:
        print(f"[reconstruct_export_replogle] {len(missing)} unmapped batch values: {missing[:10]}")

    mu_stats = pd.read_csv(os.path.join(NTC_FIT, "mu_ntc_per_gene.csv"))
    mu_stats = mu_stats.drop_duplicates(subset="gene_id").set_index("gene_id")
    log2_mu_by_gene = np.log2(mu_stats["mu_ntc"]).to_dict()
    trans_candidates = gene_meta_full.index
    kept_trans_genes = {g for g in trans_candidates
                        if log2_mu_by_gene.get(g, -np.inf) >= MIN_LOG2_MU_NTC_TRANS}
    n_missing = sum(g not in log2_mu_by_gene for g in trans_candidates)
    print(f"[reconstruct_export_replogle] {len(kept_trans_genes)}/{len(trans_candidates)} trans genes "
          f"kept at log2(mu_ntc) >= {MIN_LOG2_MU_NTC_TRANS} ({n_missing} absent from mu table)")

    _shared_cache.update(dict(
        cell_meta_full=cell_meta_full,
        gene_meta_full=gene_meta_full,
        kept_trans_genes=kept_trans_genes,
    ))
    return _shared_cache


def _load_gene_model_inputs(gene_id: str, cell_meta_full: pd.DataFrame,
                              gene_meta_full: pd.DataFrame, keep_genes):
    """Ported verbatim from 10_bayesDREAM_fit_trans_MYB.ipynb's
    load_gene_model_inputs()."""
    tgt = _read_parquet(f"{INDIR}/{gene_id}/counts.parquet").set_index("gene")
    ntc = _read_parquet(f"{INDIR}/NTC_subset/counts_subset.parquet").set_index("gene")
    assert tgt.index.equals(ntc.index), "gene order differs between files"

    mask = tgt.index.isin(keep_genes)
    n_kept = int(mask.sum())
    assert n_kept > 0, f"no rows survived the mu filter for {gene_id}"
    tgt = tgt.loc[mask]
    ntc = ntc.loc[mask]

    meta = cell_meta_full[cell_meta_full["target"].isin([gene_id, "ntc"])].copy()
    is_tgt = meta["target"] != "ntc"
    keys = set(meta.loc[is_tgt, "batch"])
    meta = meta[is_tgt | meta["batch"].isin(keys)]

    counts = pd.concat([ntc, tgt], axis=1)
    meta = meta[meta["cell"].isin(counts.columns)]
    assert not meta["cell"].duplicated().any()
    counts = counts.loc[:, meta["cell"]]

    feature_meta = gene_meta_full.loc[counts.index.rename("gene_id")].reset_index()
    return meta, counts, feature_meta


def reconstruct_model(gene_symbol: str, *, device: Optional[str] = None):
    """Rebuild a fully-loaded bayesDREAM model (ntc+cis+trans posteriors)
    for Replogle's `gene_symbol` cis gene, mirroring
    10_bayesDREAM_fit_trans_MYB.ipynb's build_trans_model() exactly, but
    reloading the already-completed trans fit instead of re-fitting.

    Returns (model, label, output_dir).
    """
    if gene_symbol not in REPLOGLE_GENE_TO_ID:
        raise KeyError(f"{gene_symbol!r} not in REPLOGLE_GENE_TO_ID -- add its Ensembl ID to "
                        "comparative/datasets.py first.")
    gene_id = REPLOGLE_GENE_TO_ID[gene_symbol]
    shared = _load_shared_inputs()

    meta, counts, feature_meta = _load_gene_model_inputs(
        gene_id, shared['cell_meta_full'], shared['gene_meta_full'], shared['kept_trans_genes'],
    )
    label = f"papermill_{gene_id}"
    output_dir = os.path.join(OUTDIR, label)

    model_kwargs = dict(meta=meta, counts=counts, feature_meta=feature_meta,
                        output_dir=OUTDIR, label=label)
    if device:
        model_kwargs['device'] = str(device)
    model = bayesDREAM(**model_kwargs)

    # NOT lean=True (unlike the source notebook's own build_trans_model(),
    # which only ever needs point estimates for its own summary/plotting use)
    # -- save_model_for_plotting() below re-saves the NTC fit via
    # save_ntc_fit(), which hard-requires the full posterior (raises
    # ValueError on a lean-loaded modality; confirmed 2026-08-26). Costs more
    # memory/time to load than the notebook's own lean=True call does -- see
    # reconstruct_export.py's docstring on memory at this gene-panel scale.
    model.load_ntc_fit(input_dir=NTC_FIT, mask_features=True)
    model.add_cis_gene(gene_id)
    model.load_cis_fit(input_dir=os.path.join(CIS_FIT, f"cis_{gene_id}"))
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.exclude_trans_genes(min_log2_mu_ntc=MIN_LOG2_MU_NTC_TRANS)

    # Loads the EXISTING saved posterior -- does not fit anything.
    model.load_trans_fit(modalities=["gene"], subset_features=True)

    return model, label, output_dir


def reconstruct_and_export(
    gene_symbol: str, *, hill_log2fc_targets=HILL_LOG2FC_TARGETS, device: Optional[str] = None,
    force: bool = False,
) -> str:
    """Full pipeline for one Replogle cis gene: reconstruct, write
    trans_feature_summary_gene.csv (with hill_eval's y_at_x_log2fc{...}
    columns) and the save_model_for_plotting() export into
    REPLOGLE.save_for_plotting_dir_fn()'s directory. Returns that directory.

    NOT backfilled in place into OUTDIR/<label>/ (unlike Domingo/Morris) --
    that's your student's directory; you have read access (needed for
    reconstruct_model()'s loads below) but confirmed NOT write access
    (PermissionError, 2026-08-26) writing a summary CSV there. Everything
    this function writes goes to your own Comparative/input/ tree instead --
    see comparative/datasets.py's REPLOGLE.run_dir_fn for why that's also
    where trans_param_compare.py reads from for this dataset.

    Checkpointed like comparative/reconstruct_export.py's version: if this
    gene's summary CSV already has every requested column and its export
    already exists, skips the reconstruction entirely (pass force=True to
    redo it anyway). Also explicitly frees the model (del + gc.collect())
    before returning -- see that module's docstring for why this matters at
    Replogle's transcriptome-wide gene-panel scale.
    """
    if gene_symbol not in REPLOGLE_GENE_TO_ID:
        raise KeyError(f"{gene_symbol!r} not in REPLOGLE_GENE_TO_ID -- add its Ensembl ID to "
                        "comparative/datasets.py first.")
    save_dir = REPLOGLE.save_for_plotting_dir_fn(gene_symbol)

    if not force and is_already_backfilled(save_dir, "gene", save_dir, hill_log2fc_targets):
        print(f"[Replogle/{gene_symbol}] already backfilled + exported -- skipping (force=True to redo)")
        return save_dir

    print(f"[Replogle/{gene_symbol}] reconstructing model...")
    model, label, _read_only_output_dir = reconstruct_model(gene_symbol, device=device)

    print(f"[Replogle/{gene_symbol}] computing summary + y_at_x_log2fc{{...}} columns...")
    os.makedirs(save_dir, exist_ok=True)
    df = model.save_trans_summary(output_dir=save_dir, modality_name="gene")
    df = add_log2fc_at_columns(model, df, modality_name="gene", targets=hill_log2fc_targets)
    csv_path = os.path.join(save_dir, "trans_feature_summary_gene.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Replogle/{gene_symbol}] wrote {csv_path}")

    print(f"[Replogle/{gene_symbol}] exporting for plotting -> {save_dir}")
    save_model_for_plotting(model, save_dir=save_dir)

    del model, df
    gc.collect()

    return save_dir


def reconstruct_and_export_all(genes: Optional[List[str]] = None, **kwargs) -> Dict[str, str]:
    """Loop reconstruct_and_export() over every Replogle cis gene (default:
    REPLOGLE.cis_genes, i.e. all 7).

    Raises immediately (does not skip) if any gene's raw inputs are missing
    or fail an assertion -- REPLOGLE.cis_genes is defined as "has a
    completed fit_trans run", so a gene listed there with missing/broken
    data is a real inconsistency to fix, not an expected gap.
    """
    genes = genes or REPLOGLE.cis_genes
    results: Dict[str, str] = {}
    for g in genes:
        results[g] = reconstruct_and_export(g, **kwargs)
    return results
