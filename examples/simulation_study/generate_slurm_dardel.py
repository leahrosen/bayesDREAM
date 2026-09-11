"""
Generate SLURM array scripts for the additive-Hill recovery study on Dardel (PDC),
CPU-only. See docs/SIMULATION_STUDY_PLAN.md §7/§7b and the Berzelius counterpart,
generate_slurm.py.

Unlike Berzelius's generator, this does NOT use bayesDREAM.slurm_jobgen.SlurmJobGenerator
for resource sizing -- that machinery estimates GPU memory tiers, which don't apply here.
Dardel's `shared` CPU partition hands out `cores * mem_per_core` automatically from
--cpus-per-task alone (no separate --mem needed), and the core-count-scaling benchmark run
on this study's own dataset sizes (plan §7, 2026-07-29) showed fit_ntc/fit_cis/fit_trans
don't meaningfully benefit from more than a handful of cores -- so resource sizing here is
just a fixed --cpus-per-task passed in by the caller, not a per-scenario estimate.

Writes a SINGLE combined array script (not separate simulate+fit arrays with a
dependency, see §7b 2026-07-31 update):
    01_run.sh        array 0-N, one task = simulate_scenario.py then run_recovery_fit.py
    submit_all.sh    submits 01_run.sh

Originally this was two scripts (01_simulate.sh + 02_fit.sh) chained with
--dependency=aftercorr, matching generate_slurm.py's Berzelius shape. That breaks on
Dardel: sbatch registers every array task as a distinct submitted job record
immediately, regardless of dependency state (a dependency only delays when a task is
*allowed to run*, not when it's *counted* against the account's submit quota) -- so two
720-task arrays outstanding at once is 1440 submitted records. Dardel's per-account
MaxSubmitJobs was found to be 1024 (`sacctmgr show assoc user=$USER format=...
MaxSubmitJobs`), so the second array submission failed outright with
AssocMaxSubmitJobLimit, and even the first array's *running* count was throttled well
below the requested %N. Merging both steps into one script per array task keeps total
submitted records at 720 (one array), safely under the limit, with identical per-task
behavior (simulate always runs immediately before fit for that same scenario/replicate,
same effective ordering `aftercorr` was providing).

Usage:
    python generate_slurm_dardel.py --design_matrix $OUT/design_matrix.csv \
        --outdir $OUT/slurm --data_path $DATA \
        --python_env $PYTHON_ENV --examples_path $EXAMPLES \
        --account <dardel-account> --cores 2
"""

import argparse
import os

import pandas as pd


SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err
#SBATCH --array=0-{max_index}%{max_concurrent}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}

set -euo pipefail
"""


def _hours_to_slurm_time(hours: float) -> str:
    total_minutes = max(1, int(round(hours * 60)))
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _resource_block(cpus: int, max_index: int, max_concurrent: int, job_name: str,
                     log_dir: str, time_str: str, account: str, partition: str) -> str:
    return SBATCH_HEADER.format(
        job_name=job_name, account=account, log_dir=log_dir, max_index=max_index,
        max_concurrent=max_concurrent, time=time_str, partition=partition, cpus=cpus,
    )


def generate_slurm_scripts(
    design_matrix_path: str,
    outdir: str,
    data_path: str,
    python_env: str,
    examples_path: str,
    account: str,
    cores: int = 2,
    partition: str = 'shared',
    max_concurrent_jobs: int = 128,
    sim_time_hours: float = 0.25,
    fit_time_hours: float = 18.0,
):
    design_matrix = pd.read_csv(design_matrix_path)
    max_index = int(design_matrix['row_index'].max())

    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # --- 01_run.sh: simulate_scenario.py then run_recovery_fit.py, one array task each ---
    # Calibrated 2026-07-30 from checkpoint timestamps of a real (timed-out) additive_hill
    # run at --cores 2 on the largest scenario (cells_per_gene=1000, scenario_96/rep_0):
    # fit_ntc=4590s + fit_cis=1305s + fit_trans (extrapolated from 3.97 steps/s over
    # 155,556 total steps [55,556 warmup + 100,000 main] = 39,201s) = 12.53h raw estimate.
    # 18h fit budget adds margin for (a) the rep_0 rate coming from a single 42-minute
    # interval, the noisiest of the 5 core counts tested, and (b) Phase 2 (additive_hill
    # proper, 2 Hill components) plausibly costing somewhat more per step than Phase 1
    # (single_hill warmup), which a flat-rate extrapolation from mostly-Phase-1 data
    # doesn't fully capture. Subsequently confirmed by real completed additive_hill runs
    # at --cores 2 (2026-07-31): 12.4-13.6h total, in line with the estimate. See
    # docs/SIMULATION_STUDY_PLAN.md §7b. If you change --cores away from 2, override
    # --fit_time_hours -- see the §7b table for other core counts (4/8/16/32 cores
    # measured at 7.8h/6.2h/5.2h/5.9h total respectively).
    total_time_hours = sim_time_hours + fit_time_hours
    total_time_str = _hours_to_slurm_time(total_time_hours)
    run_script = _resource_block(
        cores, max_index, max_concurrent_jobs, 'sim_study_dardel_run', log_dir,
        time_str=total_time_str, account=account, partition=partition,
    )
    run_script += (
        # Thread pinning for simulate_scenario.py's own process/subprocesses (e.g. its
        # Rscript/scran call) -- run_recovery_fit.py pins its own threads internally
        # (see its module docstring), but simulate_scenario.py doesn't, so pin at the
        # shell level here to match --cpus-per-task and avoid oversubscribing this
        # task's cgroup-allocated cores on Dardel's shared (multi-tenant) partition.
        f"\nexport OMP_NUM_THREADS={cores} OPENBLAS_NUM_THREADS={cores} "
        f"MKL_NUM_THREADS={cores} VECLIB_MAXIMUM_THREADS={cores} "
        f"NUMEXPR_NUM_THREADS={cores}\n"
        # ROW_OFFSET (unset -> 0) lets a design matrix bigger than SLURM's MaxArraySize
        # (a per-job cap on the max *array index* value, separate from MaxSubmitJobs)
        # be submitted in multiple waves, each using array indices 0..(wave_size-1) --
        # safely under any reasonable MaxArraySize -- while still resolving the correct
        # row_index via e.g. `sbatch --array=0-599 --export=ALL,ROW_OFFSET=600 01_run.sh`
        # for a second wave covering row_index 600-1199. Normal single-wave submission
        # (ROW_OFFSET unset) is unaffected: ROW_INDEX == SLURM_ARRAY_TASK_ID as before.
        f"\nROW_INDEX=$((SLURM_ARRAY_TASK_ID + ${{ROW_OFFSET:-0}}))\n"
        f"\n{python_env} {examples_path}/simulate_scenario.py "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $ROW_INDEX "
        f"--outdir {data_path}\n"
        f"\n{python_env} {examples_path}/run_recovery_fit.py "
        f"--data_root {data_path} "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $ROW_INDEX "
        f"--device cpu --cores {cores}\n"
    )
    with open(os.path.join(outdir, '01_run.sh'), 'w') as f:
        f.write(run_script)

    # --- submit_all.sh ---
    submit_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "RUN_JOB=$(sbatch --parsable 01_run.sh)\n"
        "echo \"Submitted 01_run.sh: $RUN_JOB\"\n"
    )
    submit_path = os.path.join(outdir, 'submit_all.sh')
    with open(submit_path, 'w') as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)

    n_tasks = max_index + 1
    print(f"Wrote SLURM scripts for {n_tasks} array tasks to {outdir}")
    print(f"  01_run.sh: --cpus-per-task={cores}, no explicit --mem (Dardel's `{partition}` "
          f"partition derives memory from cores automatically), time budget {total_time_str} "
          f"(sim {_hours_to_slurm_time(sim_time_hours)} + fit {_hours_to_slurm_time(fit_time_hours)})")
    print(f"  {fit_time_hours}h fit budget is calibrated for --cores 2 from real completed "
          f"additive_hill runs (2026-07-30/31) -- see docs/SIMULATION_STUDY_PLAN.md §7b. "
          f"If --cores={cores} != 2, override --fit_time_hours using the §7b table "
          f"(measured totals: 2c=12.5h, 4c=7.8h, 8c=6.2h, 16c=5.2h, 32c=5.9h -- add your "
          f"own margin). Also confirm this fits within Dardel's `shared` partition's max "
          f"walltime.")
    print(f"  Single array of {n_tasks} tasks -- total submitted job records "
          f"({n_tasks}) is what counts against your account's MaxSubmitJobs, "
          f"independent of --max_concurrent_jobs={max_concurrent_jobs} (which only "
          f"throttles how many run *simultaneously*, via --array=0-{max_index}%"
          f"{max_concurrent_jobs}). Check `sacctmgr show assoc user=$USER "
          f"format=MaxSubmitJobs` before submitting if unsure this fits.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_matrix', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--data_path', required=True,
                         help="Cluster path where simulate_scenario.py writes/reads "
                              "per-scenario data (passed through as --outdir/--data_root).")
    parser.add_argument('--python_env', required=True)
    parser.add_argument('--examples_path', required=True,
                         help="Cluster path to examples/simulation_study/ "
                              "(containing simulate_scenario.py, run_recovery_fit.py).")
    parser.add_argument('--account', required=True,
                         help="Dardel project/account for #SBATCH --account.")
    parser.add_argument('--cores', type=int, default=2,
                         help="CPU cores per array task (shared sequentially by "
                              "simulate_scenario.py then run_recovery_fit.py's --cores, "
                              "which pins OMP/MKL/OpenBLAS/NumExpr/torch thread counts "
                              "before any heavy import -- required for correctness on "
                              "Dardel's shared partition). Default 2, per the "
                              "core-count-scaling benchmark showing near-flat/sublinear "
                              "returns beyond a few cores for this workload -- see "
                              "plan §7/§7b.")
    parser.add_argument('--partition', default='shared')
    parser.add_argument('--max_concurrent_jobs', type=int, default=128,
                         help="Throttles how many array tasks run simultaneously "
                              "(--array=0-N%%<this>). Does NOT reduce how many job "
                              "records are submitted -- that's controlled by the "
                              "design matrix size (720) and must stay under your "
                              "account's MaxSubmitJobs regardless of this value.")
    parser.add_argument('--sim_time_hours', type=float, default=0.25)
    parser.add_argument('--fit_time_hours', type=float, default=18.0,
                         help="Calibrated for --cores 2 (real completed additive_hill "
                              "runs, 2026-07-30/31). Override if using a different "
                              "--cores -- see plan §7b table (2c=12.5h, 4c=7.8h, "
                              "8c=6.2h, 16c=5.2h, 32c=5.9h raw measured totals, add "
                              "your own margin).")
    args = parser.parse_args()

    generate_slurm_scripts(
        design_matrix_path=args.design_matrix,
        outdir=args.outdir,
        data_path=args.data_path,
        python_env=args.python_env,
        examples_path=args.examples_path,
        account=args.account,
        cores=args.cores,
        partition=args.partition,
        max_concurrent_jobs=args.max_concurrent_jobs,
        sim_time_hours=args.sim_time_hours,
        fit_time_hours=args.fit_time_hours,
    )
