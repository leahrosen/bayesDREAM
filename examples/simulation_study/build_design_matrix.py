"""
Build design_matrix.csv for the single-Hill recovery study (docs/SIMULATION_STUDY_PLAN.md).

Enumerates the 240 cell-design scenarios (§3.1) x 5 replicates = 1200 rows, each with a
deterministic integer seed (§6): seed = MASTER_SEED + 1000*scenario_id + replicate_id.
(Grid widened 2026-08-04: added cells_per_gene=250 and log2_X_NTC=-2, up from 144
scenarios/720 rows -- see docs/SIMULATION_STUDY_PLAN.md §3.1 update.)
No use of Python's builtin hash() — that's salted per-process and not reproducible
across separate SLURM job launches (see plan §2).

Also creates (and by default pushes) a stable git tag at the current commit, so the
exact code used stays reachable/reproducible even if the source branch is later
deleted or force-pushed past this point (a commit hash alone doesn't guarantee this --
see _git_provenance.py). The tag name and per-run git provenance become columns in
design_matrix.csv, and each scenario's own config.json re-checks provenance again at
simulate time (see simulate_scenario.py) so drift between build-time and run-time can
be detected.

Each run writes into its own dated subdirectory of --outdir (added 2026-08-04, at the
user's request after --outdir accumulated multiple unrelated projects' output
alongside this study's -- keeping each run of this pipeline self-contained under its
own timestamped directory means re-running this script never collides with or
overwrites a previous run's data, and old runs stay trivially easy to find/archive/
delete independently). The subdirectory shares the exact same timestamp as the git
tag (when tagging isn't skipped), so the two are trivially cross-referenceable:
    <outdir>/<tag_prefix>-<UTC timestamp>/design_matrix.csv
e.g. .../sim_study_out/sim-study-20260804T153012Z/design_matrix.csv

Usage:
    python build_design_matrix.py --outdir ./sim_study_out [--master_seed 20260728] [--n_replicates 5]
    python build_design_matrix.py --outdir ./sim_study_out --no_tag   # skip git tagging

    # Downstream steps then use the printed dated directory, e.g.:
    #   export OUT=./sim_study_out/sim-study-20260804T153012Z
    #   export DATA=$OUT/data
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from bayesDREAM.simulation.cis_panel_simulation import GUIDE_PATTERNS
from _git_provenance import create_stable_snapshot_tag, git_provenance

CELLS_PER_GENE_VALUES = (100, 250, 500, 1000)
LOG2_X_NTC_VALUES = (-2, -1, 0, 1, 2)
LOG2_O_X_VALUES = (-1.5, 0.0)
SIGMA_EFF = 0.7
MASTER_SEED_DEFAULT = 20260728
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
    parser.add_argument('--tag_prefix', default='sim-study',
                         help="Prefix for the stable snapshot git tag (default: 'sim-study').")
    parser.add_argument('--no_tag', action='store_true',
                         help="Skip creating a stable snapshot git tag.")
    parser.add_argument('--no_push', action='store_true',
                         help="Create the tag locally but don't push it to origin.")
    args = parser.parse_args()

    # Generated once, up front, so the run directory and the git tag (if created)
    # share the exact same timestamp rather than two independently-generated ones a
    # few milliseconds apart.
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = os.path.join(args.outdir, f"{args.tag_prefix}-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    design_matrix = build_design_matrix(
        master_seed=args.master_seed, n_replicates=args.n_replicates,
    )

    prov = git_provenance()
    if args.no_tag:
        tag_info = {'bayesdream_tag': None, 'bayesdream_tag_pushed': False}
    else:
        tag_info = create_stable_snapshot_tag(prefix=args.tag_prefix, push=not args.no_push,
                                               timestamp=timestamp)
    for col, val in {**prov, **tag_info}.items():
        design_matrix[col] = val

    out_path = os.path.join(run_dir, 'design_matrix.csv')
    design_matrix.to_csv(out_path, index=False)

    n_scenarios = design_matrix['scenario_id'].nunique()
    print(f"Wrote {len(design_matrix)} rows ({n_scenarios} scenarios x "
          f"{args.n_replicates} replicates) to {out_path}")
    if tag_info['bayesdream_tag']:
        print(f"Stable snapshot tag: {tag_info['bayesdream_tag']} "
              f"(pushed: {tag_info['bayesdream_tag_pushed']})")
    else:
        print("No stable snapshot tag created (see warnings above) — "
              "design_matrix.csv only carries the commit hash/branch, which is not "
              "guaranteed to remain reachable if the branch is later deleted.")
    print(f"\nRun directory: {run_dir}")
    print(f"Update your shell for subsequent steps:")
    print(f"  export OUT={run_dir}")
    print(f"  export DATA=$OUT/data")
