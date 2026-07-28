"""
Simulate one cell-design scenario (array-job entry point). See docs/SIMULATION_STUDY_PLAN.md.

This step is pure NumPy/negbinom sampling (no SVI) — no torch/pyro dependency and no
seeding needed beyond the design matrix's per-row seed, which is threaded entirely
through numpy.random.Generator instances (never global numpy/torch/pyro state).
torch/pyro seeding happens in run_recovery_fit.py, which is the step that actually
uses them.

Usage (single task):
    python simulate_scenario.py --design_matrix design_matrix.csv --row_index 0 \
        --outdir ./sim_study_out

Usage (SLURM array; reads $SLURM_ARRAY_TASK_ID if --row_index is omitted):
    python simulate_scenario.py --design_matrix design_matrix.csv --outdir ./sim_study_out

Writes to <outdir>/scenario_<scenario_id>/rep_<replicate_id>/:
    config.json, meta.csv, counts.csv, cis_ground_truth.csv,
    guide_ground_truth.csv, trans_ground_truth.csv
"""

import argparse
import json
import os
import subprocess

import pandas as pd

from bayesDREAM.simulation import simulate_scenario


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_one(design_matrix: pd.DataFrame, row_index: int, outdir: str) -> str:
    row = design_matrix.loc[design_matrix['row_index'] == row_index].iloc[0]

    result = simulate_scenario(
        cells_per_gene=int(row['cells_per_gene']),
        n_guides=int(row['n_guides']),
        guide_shape=row['guide_shape'],
        sigma_eff=float(row['sigma_eff']),
        log2_X_NTC=float(row['log2_X_NTC']),
        log2_o_x=float(row['log2_o_x']),
        seed=int(row['seed']),
    )

    scen_dir = os.path.join(
        outdir, f"scenario_{int(row['scenario_id'])}", f"rep_{int(row['replicate_id'])}",
    )
    os.makedirs(scen_dir, exist_ok=True)

    config = dict(result['config'])
    config.update(dict(
        row_index=int(row_index),
        scenario_id=int(row['scenario_id']),
        replicate_id=int(row['replicate_id']),
        bayesdream_commit=_git_commit_hash(),
    ))
    with open(os.path.join(scen_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    result['meta'].to_csv(os.path.join(scen_dir, 'meta.csv'), index=False)
    result['counts'].to_csv(os.path.join(scen_dir, 'counts.csv'))
    result['cis_ground_truth'].to_csv(os.path.join(scen_dir, 'cis_ground_truth.csv'), index=False)
    result['guide_ground_truth'].to_csv(os.path.join(scen_dir, 'guide_ground_truth.csv'), index=False)
    result['trans_ground_truth'].to_csv(os.path.join(scen_dir, 'trans_ground_truth.csv'), index=False)

    return scen_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_matrix', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--row_index', type=int, default=None,
                         help="Row to simulate. Defaults to $SLURM_ARRAY_TASK_ID.")
    args = parser.parse_args()

    row_index = args.row_index
    if row_index is None:
        row_index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    design_matrix = pd.read_csv(args.design_matrix)
    scen_dir = run_one(design_matrix, row_index, args.outdir)
    print(f"[row {row_index}] wrote {scen_dir}")
