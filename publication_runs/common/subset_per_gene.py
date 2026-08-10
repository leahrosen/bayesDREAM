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
already writes for that gene's cis stage. A top-level ntc_shared_dir: key,
if present, is simply ignored (see "Does NOT call load_ntc_fit()" above).
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

from config_utils import build_model_from_config, load_bayesdream_yaml  # noqa: E402

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

        model.meta.to_csv(os.path.join(mode_outdir, "meta.csv"), index=False)
        sparse.save_npz(os.path.join(mode_outdir, "gene_counts.npz"), counts)
        feature_meta.to_csv(os.path.join(mode_outdir, "gene_meta.csv"), index=False)

        if model.is_high_moi:
            np.save(os.path.join(mode_outdir, "guide_assignment.npy"), model.guide_assignment)
            model.guide_meta.to_csv(os.path.join(mode_outdir, "guide_meta.csv"), index=False)

        print(f"[subset_per_gene] {cis_gene} (mode={mode}): {len(model.meta)} cells x "
              f"{counts.shape[0]} genes -> {mode_outdir}")


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
