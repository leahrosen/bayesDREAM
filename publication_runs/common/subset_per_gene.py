"""
Precompute a per-cis-gene, NTC+cis-cells-only subset of a dataset's raw
meta/counts/(guide_assignment for high-MOI). Eliminates the dominant cost
observed in real profiling: every downstream per-gene SLURM job (fit_cis/
compensation/fit_trans/permutation/recapitulation) currently re-loads and
re-classifies the FULL dataset from scratch (e.g. Morris: 31468 genes x
52852 cells, full high-MOI classification against 1871 guides) just to
throw away most of it (down to ~12684 cells for one gene). This script
does that classification ONCE per gene, writing the result to disk, so
every downstream job just loads an already-small file.

Builds the SAME deferred-cis_gene model each dataset's real pipeline
already builds for fit_cis (build_model_from_config + add_cis_gene() --
see run_cis_deferred.py, which this mirrors), then writes the resulting
subsetted meta/counts/(guide_assignment/guide_meta for high-MOI) to
--outdir INSTEAD of proceeding to fit_cis(). Not part of the SLURM
pipeline itself -- run once, per gene, as part of preprocessing (see
morris/preprocess.py, domingo/preprocess.py), analogous to how those
scripts already turn raw exports into the base meta.csv/gene_counts.npz/
gene_meta.csv used here as input.

Optional ``batch_match_ntc`` step (top-level config key, disabled by
default): restricts the NTC cells kept after ``add_cis_gene()`` to only
those from a technical batch where THIS gene was actually targeted --

    is_tgt = model.meta['target'] != 'ntc'
    keys = set(model.meta.loc[is_tgt, batch_col])
    keep = is_tgt | model.meta[batch_col].isin(keys)

i.e. every cis-gene-targeting cell is kept regardless of batch; only the
NTC pool is restricted. Ported from Replogle's own ad hoc
`load_gene_model_inputs()` (see publication_runs/replogle/STRATEGY.md §2) --
useful whenever a dataset pools cells from many technical batches/
experiments but any single gene's guides only appear in a handful of them,
so comparing that gene's cells against NTCs from batches that never
contained the gene's guides would mix in batch-to-batch variation beyond
what `sum_factor`/`alpha_y` correct for::

    batch_match_ntc:
      enabled: true
      batch_col: batch   # column in model.meta; default 'batch'

Optional ``select_ntc_guides`` step (top-level config key, disabled by
default): restricts the NTC cells kept after ``add_cis_gene()`` to only
those carrying one of an explicit, curated list of NTC guides --

    is_tgt = model.meta['target'] != 'ntc'
    keep = is_tgt | model.meta['guide'].isin(guides)

Composes with ``batch_match_ntc`` above (both, if enabled, are ANDed
together for the NTC portion -- cis-gene-targeting cells are always kept
regardless of either)::

    select_ntc_guides:
      enabled: true
      guides: ["non-targeting_01407|non-targeting_03530", ...]   # exact model.meta['guide'] values

Ported from Replogle's own pipeline: its pre-built `NTC_subset/` input
(read by an earlier version of `replogle/preprocess.py`) turned out to
already be restricted to a curated NTC guide list this way -- see
`publication_runs/replogle/STRATEGY.md` §10. `replogle/preprocess.py` now
reads the FULL (unrestricted) NTC population instead, so this flag is what
reproduces that curation explicitly, at the per-gene subsetting stage,
rather than baking it silently into an upstream input file.

Both flags are applied AFTER `add_cis_gene()` (so `model.meta['target']`
is already reduced to {cis_gene, 'ntc'}) but BEFORE the per-mode writing
loop below -- computed once, reused for both `full`/`cis_only` (identical
cell set either way; only the feature panel differs between modes).
Deliberately computed before `compute_scran` runs (kept default off for
both Domingo/Morris anyway, see below) rather than after, so a
hypothetical future per-subset scran call still sees the SAME cell
population `add_cis_gene()` produced, matching this script's pre-existing
behavior when neither flag is set.

--modes (comma-separated, e.g. "full,cis_only") -- one subdirectory of
--outdir written per requested mode, from a SINGLE model construction
(add_cis_gene() already separates 'cis' from the trans panel; both pieces
are in memory regardless of which mode(s) you asked for, so requesting
both is one classification pass, not two):

  full/       entire trans-gene panel (min_count-filtered on the REDUCED
              cell subset -- this is bayesDREAM's own
              Modality._filter_zero_features(), which add_cis_gene()
              already re-runs internally after cell subsetting; nothing
              extra needed here to replicate it), with the cis gene's own
              row put back in (so a downstream job can still construct
              EAGERLY with cis_gene=<gene> at construction and have
              _extract_cis_from_gene find it, exactly as today). For
              stages that need the full trans panel: fit_ntc, compensation,
              fit_trans, permutation, recapitulation.
  cis_only/   ONLY the cis gene's own row -- nothing else. A downstream job
              constructing from this file needs bayesDREAM's cis_only=True
              (see bayesDREAM/model.py's cis_only docstring entry) for
              EAGER cis_gene-at-construction (fit_cis for primary genes);
              the DEFERRED cis_gene=None + add_cis_gene() pattern tolerates
              this file with no flag needed at all (confirmed by direct
              test -- no equivalent to the eager path's "No genes left
              after filtering!" raise), which is what Morris's cis-only
              sweep over non-primary genes uses.

Does NOT call load_ntc_fit() -- earlier versions of this script did (to
mirror run_cis_deferred.py's exact call order), but add_cis_gene()'s cell/
gene filtering doesn't depend on it, and requiring a pre-existing ntc fit
here would be actively wrong once callers stop assuming a single shared
ntc_shared serves every gene (see morris/README.md's per-primary-gene
fit_ntc section).

CAN run compute_scran (via apply_sum_factor_adjustments, steps=
("compute_scran",)) if the config's sum_factor: block enables it -- right
here, right after add_cis_gene(), while the primary modality still holds
the FULL trans panel (a pooling-based normalisation across many genes, so
running it on the cis_only mode's single-gene panel would be meaningless,
not merely awkward -- quickCluster/computeSumFactors on 1 feature errors
inside R: "need at least 2 points to select a bandwidth automatically").
NEITHER dataset's config actually enables this anymore as of 2026-08:
Domingo never did (its sum_factor column is computed upstream, once, in
domingo/preprocess.py); Morris used to (recomputed scran separately per
gene here, writing 'sum_factor_new') but now also computes scran ONCE,
upstream, in morris/preprocess.py, on the full dataset -- see
morris/README.md's "sum_factor: scran, computed once, shared everywhere"
for why per-gene recomputation was actually a correctness problem (fit_ntc's
alpha_y_mult/alpha_x_prefit calibrated against a different sum_factor than
fit_cis/fit_trans composed it with). This code path is kept only as
generic, dataset-agnostic infrastructure for a hypothetical future dataset
that genuinely needs per-gene-subset scran; it is currently unused by both
datasets' own generated configs.

Output files always written as sparse .npz for gene_counts (regardless of
whether the dataset's OWN base files are dense CSV, e.g. Domingo) --
per-gene subsets are small either way, and config_utils._read_counts()
already dispatches on the .npz extension for any dataset.

Usage
-----
    python subset_per_gene.py --config <gene_cis_stage_config.yaml> \\
        --outdir <per-gene output dir> --modes full,cis_only

Config: same schema as run_cis_deferred.py (model.cis_gene omitted,
top-level cis_gene: key) -- reuse the SAME rendered config generate_slurm.py
already writes for that gene's cis stage, EXCEPT its sum_factor: block
(omitted entirely for Morris now -- see "DOES run compute_scran" above; a
dataset that DID want per-gene scran here would need a subset_input-specific
block with compute_scran enabled and no adjust_ntc_sum_factor, distinct from
the real cis stage's own compute_scran-disabled block). A top-level
ntc_shared_dir: key, if present, is simply ignored (see "Does NOT call
load_ntc_fit()" above).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import (  # noqa: E402
    build_model_from_config, load_bayesdream_yaml, apply_sum_factor_adjustments,
    is_enabled, normalize_stage_args,
)

VALID_MODES = ("full", "cis_only")


def subset_per_gene(cfg: dict, outdir: str, modes) -> None:
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("cis_gene"):
        raise ValueError(
            "subset_per_gene: config's model.cis_gene must be omitted (deferred) -- "
            "found model.cis_gene={!r}.".format(model_cfg["cis_gene"])
        )

    cis_gene = cfg.get("cis_gene")
    if not cis_gene:
        raise ValueError("subset_per_gene: config needs a top-level 'cis_gene' key.")
    bad_modes = set(modes) - set(VALID_MODES)
    if bad_modes or not modes:
        raise ValueError(f"subset_per_gene: --modes must be a non-empty subset of {VALID_MODES}, got {modes!r}")

    model = build_model_from_config(cfg)

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    model.add_cis_gene(cis_gene)

    # Both restrictions apply only to the NTC portion of model.meta (every
    # cis-gene-targeting cell is always kept); when both are enabled they
    # combine via AND, matching the ACTUAL original Replogle pipeline's
    # composition (NTC_subset's curated guide list, then batch-matched on
    # top of that -- see STRATEGY.md §10).
    cell_mask = None

    def _and_into(new_mask, label):
        nonlocal cell_mask
        cell_mask = new_mask if cell_mask is None else (cell_mask & new_mask)
        n_dropped = int((~new_mask).sum())
        print(f"[subset_per_gene] {cis_gene}: {label} drops {n_dropped} NTC cell(s) "
              f"({int(new_mask.sum())}/{len(new_mask)} cells kept by this step alone)")

    bm_cfg = cfg.get("batch_match_ntc") or {}
    if is_enabled(bm_cfg, default=False):
        batch_args = normalize_stage_args(bm_cfg)
        batch_col = batch_args.get("batch_col", "batch")
        if batch_col not in model.meta.columns:
            raise ValueError(f"batch_match_ntc: batch_col {batch_col!r} not found in model.meta columns.")
        is_tgt = model.meta["target"] != "ntc"
        keys = set(model.meta.loc[is_tgt, batch_col])
        batch_mask = (is_tgt | model.meta[batch_col].isin(keys)).to_numpy()
        _and_into(batch_mask, f"batch_match_ntc (batch_col={batch_col!r})")

    sg_cfg = cfg.get("select_ntc_guides") or {}
    if is_enabled(sg_cfg, default=False):
        sg_args = normalize_stage_args(sg_cfg)
        guides = sg_args.get("guides")
        if not guides:
            raise ValueError("select_ntc_guides: 'guides' (non-empty list) is required.")
        if "guide" not in model.meta.columns:
            raise ValueError("select_ntc_guides: model.meta has no 'guide' column.")
        is_tgt = model.meta["target"] != "ntc"
        guide_mask = (is_tgt | model.meta["guide"].isin(guides)).to_numpy()
        _and_into(guide_mask, f"select_ntc_guides ({len(guides)} guide(s))")

    if cell_mask is not None:
        print(f"[subset_per_gene] {cis_gene}: combined NTC restriction keeps "
              f"{int(cell_mask.sum())}/{len(cell_mask)} cells total")

    # scran (compute_scran) needs the FULL gene panel to mean anything --
    # pooling-based normalisation across many genes -- so it must run HERE,
    # right after add_cis_gene() and before the mode-specific subsetting
    # below strips the primary modality down to the cis-only single-gene
    # panel. compute_scran_sum_factor() writes its output into model.meta
    # (not just modality.sum_factors), so it survives into every mode's
    # meta.csv written below -- downstream per-gene stages (cis/trans/
    # permutation/recapitulation) then just read that precomputed column
    # instead of each redundantly (and, for the cis-only-subset stage,
    # meaninglessly) recomputing it themselves. See morris/generate_slurm.py's
    # sum_factor_precompute_block/sum_factor_cis_block split. No-op for
    # datasets whose config doesn't enable compute_scran (e.g. Domingo,
    # which computes its sum_factor column upstream, once, in preprocess.py).
    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {}, steps=("compute_scran",))

    primary_mod = model.get_modality(model.primary_modality)
    cis_mod = model.get_modality("cis")
    assert cis_mod.cells_axis == 1 and primary_mod.cells_axis == 1, (
        "subset_per_gene assumes cells_axis=1 (genes x cells) throughout this codebase"
    )

    id_cols = [c for c in ("gene_id", "gene_name") if c in cis_mod.feature_meta.columns]
    cis_counts_sp = sparse.csr_matrix(cis_mod.counts)

    for mode in modes:
        mode_outdir = os.path.join(outdir, mode)
        os.makedirs(mode_outdir, exist_ok=True)

        if mode == "full":
            counts = sparse.vstack([cis_counts_sp, sparse.csr_matrix(primary_mod.counts)]).tocsr()
            feature_meta = pd.concat(
                [cis_mod.feature_meta[id_cols], primary_mod.feature_meta[id_cols]],
                ignore_index=True,
            )
        else:  # cis_only
            counts = cis_counts_sp
            feature_meta = cis_mod.feature_meta[id_cols].reset_index(drop=True)

        meta_out = model.meta if cell_mask is None else model.meta.loc[cell_mask].reset_index(drop=True)
        counts_out = counts if cell_mask is None else counts[:, cell_mask]

        meta_out.to_csv(os.path.join(mode_outdir, "meta.csv"), index=False)
        sparse.save_npz(os.path.join(mode_outdir, "gene_counts.npz"), counts_out)
        feature_meta.to_csv(os.path.join(mode_outdir, "gene_meta.csv"), index=False)

        if model.is_high_moi:
            guide_assignment_out = model.guide_assignment if cell_mask is None else model.guide_assignment[cell_mask]
            np.save(os.path.join(mode_outdir, "guide_assignment.npy"), guide_assignment_out)
            model.guide_meta.to_csv(os.path.join(mode_outdir, "guide_meta.csv"), index=False)

        print(f"[subset_per_gene] {cis_gene} (mode={mode}): {len(meta_out)} cells x "
              f"{counts_out.shape[0]} genes -> {mode_outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--modes", required=True, help="Comma-separated subset of full,cis_only")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    subset_per_gene(load_bayesdream_yaml(Path(args.config)), args.outdir, modes)


if __name__ == "__main__":
    main()
