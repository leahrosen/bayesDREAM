"""
Build design_matrix.csv for the single-Hill recovery study (docs/SIMULATION_STUDY_PLAN.md).

Enumerates the 108 cell-design scenarios (§3.1) x 5 replicates = 540 rows, each with a
deterministic integer seed (§6): seed = MASTER_SEED + 1000*scenario_id + replicate_id.
No use of Python's builtin hash() — that's salted per-process and not reproducible
across separate SLURM job launches (see plan §2).

Usage:
    python build_design_matrix.py --outdir ./sim_study_out [--master_seed 20260727] [--n_replicates 5]
"""

import argparse
import os

import pandas as pd

from bayesDREAM.simulation.cis_panel_simulation import GUIDE_PATTERNS

CELLS_PER_GENE_VALUES = (100, 500, 1000)
LOG2_X_NTC_VALUES = (0, 1, 2)
LOG2_O_X_VALUES = (-1.5, 0.0)
SIGMA_EFF = 0.7
MASTER_SEED_DEFAULT = 20260727
N_REPLICATES_DEFAULT = 5


def build_design_matrix(
    master_seed: int = MASTER_SEED_DEFAULT,
    n_replicates: int = N_REPLICATES_DEFAULT,
) -> pd.DataFrame:
    """Enumerate all cell-design scenarios x replicates in a fixed nested order
    (cells_per_gene -> guide design -> X_NTC -> o_x -> replicate), per plan §6."""
    rows = []
    scenario_id = 0
    for cells_per_gene in CELLS_PER_GENE_VALUES:
        for (n_guides, guide_shape) in sorted(GUIDE_PATTERNS.keys()):
            for log2_X_NTC in LOG2_X_NTC_VALUES:
                for log2_o_x in LOG2_O_X_VALUES:
                    for replicate_id in range(n_replicates):
                        seed = master_seed + 1000 * scenario_id + replicate_id
                        rows.append(dict(
                            row_index=len(rows),
                            scenario_id=scenario_id,
                            replicate_id=replicate_id,
                            seed=seed,
                            cells_per_gene=cells_per_gene,
                            n_guides=n_guides,
                            guide_shape=guide_shape,
                            sigma_eff=SIGMA_EFF,
                            log2_X_NTC=log2_X_NTC,
                            log2_o_x=log2_o_x,
                        ))
                    scenario_id += 1
    df = pd.DataFrame(rows)
    assert df['seed'].is_unique, "seed collision in design matrix"
    assert df['row_index'].is_unique
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--master_seed', type=int, default=MASTER_SEED_DEFAULT)
    parser.add_argument('--n_replicates', type=int, default=N_REPLICATES_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    design_matrix = build_design_matrix(
        master_seed=args.master_seed, n_replicates=args.n_replicates,
    )
    out_path = os.path.join(args.outdir, 'design_matrix.csv')
    design_matrix.to_csv(out_path, index=False)

    n_scenarios = design_matrix['scenario_id'].nunique()
    print(f"Wrote {len(design_matrix)} rows ({n_scenarios} scenarios x "
          f"{args.n_replicates} replicates) to {out_path}")
