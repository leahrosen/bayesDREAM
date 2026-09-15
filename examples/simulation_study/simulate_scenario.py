"""
Simulate one cell-design scenario (array-job entry point). See docs/SIMULATION_STUDY_PLAN.md.

This step is NumPy/negbinom sampling (no SVI, no torch/pyro) plus one Rscript
subprocess call (scran::calculateSumFactors, to recompute a realistic sum_factor from
the simulated counts -- see recompute_sum_factor_scran). No seeding needed beyond the
design matrix's per-row seed, which is threaded entirely through numpy.random.Generator
instances (never global numpy/torch/pyro state). torch/pyro seeding happens in
run_recovery_fit.py, which is the step that actually uses them.

Requires Rscript on PATH with bioconductor-scran and r-data.table installed (declared
in environment_{cpu,cuda,rocm}.yml).

Usage (single task):
    python simulate_scenario.py --design_matrix design_matrix.csv --row_index 0 \
        --outdir ./sim_study_out

Usage (SLURM array; reads $SLURM_ARRAY_TASK_ID if --row_index is omitted):
    python simulate_scenario.py --design_matrix design_matrix.csv --outdir ./sim_study_out

Writes to <outdir>/scenario_<scenario_id>/rep_<replicate_id>/:
    config.json, meta.csv (sum_factor = scran-recomputed), counts.csv,
    cis_ground_truth.csv (includes sum_factor_true, the value actually used to
    generate the data), guide_ground_truth.csv, trans_ground_truth.csv,
    sum_factor_scran/ (R script + intermediate CSVs, kept for debuggability)
"""

import argparse
import json
import os

import pandas as pd

from bayesDREAM.simulation import simulate_scenario, recompute_sum_factor_scran
from _git_provenance import git_provenance


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
    ))
    # Fresh provenance check at simulate-time, independent of whatever was recorded
    # in design_matrix.csv at build-time -- if the two disagree (e.g. new commits were
    # pulled between building the design matrix and running this scenario), that drift
    # is visible by comparing bayesdream_commit here against build_commit below.
    config.update(git_provenance())
    build_tag = row.get('bayesdream_tag', None)
    config['build_tag'] = None if pd.isna(build_tag) else str(build_tag)
    build_commit = row.get('bayesdream_commit', None)
    config['build_commit'] = None if pd.isna(build_commit) else str(build_commit)
    with open(os.path.join(scen_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    counts_path = os.path.join(scen_dir, 'counts.csv')
    result['counts'].to_csv(counts_path)

    # simulate_from_trans_summary's own docstring: the sum factor used to *generate*
    # the data is not a valid stand-in for what a real analysis would estimate from
    # the resulting counts. meta.csv gets the scran-recomputed value (what a real
    # analyst would have); the true simulated value is preserved separately in
    # cis_ground_truth.csv as 'sum_factor_true'.
    meta = result['meta']
    scran_workdir = os.path.join(scen_dir, 'sum_factor_scran')
    meta['sum_factor'] = recompute_sum_factor_scran(counts_path, meta, workdir=scran_workdir)

    meta.to_csv(os.path.join(scen_dir, 'meta.csv'), index=False)
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
