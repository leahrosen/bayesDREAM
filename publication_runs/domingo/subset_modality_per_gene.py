"""
Precompute a per-(cis-gene, modality) subset of Domingo's splicing/velocity
data: cell-aligned to that gene's own NTC+cis-cells (matching its `full`
common/subset_per_gene.py subset) AND min_count-feature-filtered
(bayesDREAM's own Modality._filter_zero_features(), which
add_custom_modality() already runs internally via attach_modality() --
nothing extra needed here to replicate it). Eliminates the cost every
07_modality_<gene>_<mod>.sh (and its permutation/recapitulation children)
previously repaid: reading and cell-aligning the FULL, un-subsetted raw
splicing directory (modalities.data_dir) from scratch, every single time.

Builds a model from that gene's ALREADY-subsetted `full` primary-modality
data (eager cis_gene-at-construction -- the SAME shape the real modality
job already builds from), calls load_modalities.attach_modality() (the SAME
function the real job used to call directly, doing the real cell-alignment
+ -- for SJ -- gene-expression-denominator computation), then writes the
resulting Modality's counts/feature_meta/(denominator for binomial)/
cell_names to --outdir instead of proceeding to fit_ntc()/fit_trans().
07_modality_<gene>_<mod>.sh (and modality permutation/recapitulation, via
their attach_modality: config block) then read this file through
load_modalities.attach_modality_precomputed() instead.

Usage
-----
    python subset_modality_per_gene.py --config <gene_full_subset_config.yaml> \\
        --modality-name sj --modality-spec config_modalities.yaml \\
        --data-dir <modalities.data_dir> --outdir <per-(gene,modality) output dir>

Config: the SAME gene-scoped bayesdream-CLI-schema config
generate_slurm.py already renders for that gene's real 07_modality_<gene>_<mod>.sh
job (cis_gene set directly, data: pointing at that gene's `full` primary-
modality subset -- see common/subset_per_gene.py).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import build_model_from_config, load_bayesdream_yaml  # noqa: E402
from load_modalities import attach_modality  # noqa: E402


def subset_modality_per_gene(cfg: dict, modality_name: str, modality_spec_path: str, data_dir: str, outdir: str) -> None:
    with open(modality_spec_path) as f:
        specs = {m["stype"]: m for m in yaml.safe_load(f)["modalities"]}
    spec = specs[modality_name]

    model = build_model_from_config(cfg)
    resolved_name = attach_modality(model, spec, data_dir)
    mod = model.get_modality(resolved_name)

    os.makedirs(outdir, exist_ok=True)

    if spec["distribution"] == "multinomial":
        np.save(os.path.join(outdir, "counts.npy"), np.asarray(mod.counts))
    else:
        sparse.save_npz(os.path.join(outdir, "counts.npz"), sparse.csr_matrix(mod.counts))
        if mod.denominator is not None:
            sparse.save_npz(os.path.join(outdir, "denominator.npz"), sparse.csr_matrix(mod.denominator))

    mod.feature_meta.to_csv(os.path.join(outdir, "feature_meta.csv"), index=False)
    with open(os.path.join(outdir, "cell_names.txt"), "w") as f:
        f.write("\n".join(mod.cell_names) + "\n")

    print(f"[subset_modality_per_gene] {resolved_name}: {mod.dims} -> {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--modality-name", required=True)
    parser.add_argument("--modality-spec", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    subset_modality_per_gene(
        load_bayesdream_yaml(Path(args.config)), args.modality_name,
        args.modality_spec, args.data_dir, args.outdir,
    )


if __name__ == "__main__":
    main()
