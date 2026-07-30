"""
Generate SLURM array scripts for the additive-Hill recovery study on Dardel (PDC),
CPU-only. See docs/SIMULATION_STUDY_PLAN.md §7 and the Berzelius counterpart,
generate_slurm.py.

Unlike Berzelius's generator, this does NOT use bayesDREAM.slurm_jobgen.SlurmJobGenerator
for resource sizing -- that machinery estimates GPU memory tiers, which don't apply here.
Dardel's `shared` CPU partition hands out `cores * mem_per_core` automatically from
--cpus-per-task alone (no separate --mem needed), and the core-count-scaling benchmark run
on this study's own dataset sizes (plan §7, 2026-07-29) showed fit_ntc/fit_cis/fit_trans
don't meaningfully benefit from more than a handful of cores -- so resource sizing here is
just a fixed --cpus-per-task passed in by the caller, not a per-scenario estimate.

Writes the same three-script shape as generate_slurm.py:
    01_simulate.sh   array 0-N, 1 core, one task = one simulate_scenario.py call
    02_fit.sh        array 0-N, depends on 01, one task = one run_recovery_fit.py call
    submit_all.sh    submits 01 then 02 with --dependency=aftercorr

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
    max_concurrent_jobs: int = 50,
    sim_cores: int = 1,
    sim_time_hours: float = 0.25,
    fit_time_hours: float = 18.0,
):
    design_matrix = pd.read_csv(design_matrix_path)
    max_index = int(design_matrix['row_index'].max())

    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # --- 01_simulate.sh: cheap, no SVI, no bayesDREAM fitting ---
    sim_script = _resource_block(
        sim_cores, max_index, max_concurrent_jobs, 'sim_study_dardel_simulate', log_dir,
        time_str=_hours_to_slurm_time(sim_time_hours), account=account, partition=partition,
    )
    sim_script += (
        f"\n{python_env} {examples_path}/simulate_scenario.py "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $SLURM_ARRAY_TASK_ID "
        f"--outdir {data_path}\n"
    )
    with open(os.path.join(outdir, '01_simulate.sh'), 'w') as f:
        f.write(sim_script)

    # --- 02_fit.sh: fit_ntc + fit_cis + fit_trans(additive_hill), fixed core count ---
    # Calibrated 2026-07-30 from checkpoint timestamps of a real (timed-out) additive_hill
    # run at --cores 2 on the largest scenario (cells_per_gene=1000, scenario_96/rep_0):
    # fit_ntc=4590s + fit_cis=1305s + fit_trans (extrapolated from 3.97 steps/s over
    # 155,556 total steps [55,556 warmup + 100,000 main] = 39,201s) = 12.53h raw estimate.
    # 18h default here adds margin for (a) the rep_0 rate coming from a single 42-minute
    # interval, the noisiest of the 5 core counts tested, and (b) Phase 2 (additive_hill
    # proper, 2 Hill components) plausibly costing somewhat more per step than Phase 1
    # (single_hill warmup), which a flat-rate extrapolation from mostly-Phase-1 data
    # doesn't fully capture. See docs/SIMULATION_STUDY_PLAN.md §7. If you change --cores
    # away from 2, this default no longer applies -- see the §7 table for other core
    # counts (4/8/16/32 cores measured at 7.8h/6.2h/5.2h/5.9h total respectively).
    fit_time_str = _hours_to_slurm_time(fit_time_hours)
    fit_script = _resource_block(
        cores, max_index, max_concurrent_jobs, 'sim_study_dardel_fit', log_dir,
        time_str=fit_time_str, account=account, partition=partition,
    )
    fit_script += (
        f"\n{python_env} {examples_path}/run_recovery_fit.py "
        f"--data_root {data_path} "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $SLURM_ARRAY_TASK_ID "
        f"--device cpu --cores {cores}\n"
    )
    with open(os.path.join(outdir, '02_fit.sh'), 'w') as f:
        f.write(fit_script)

    # --- submit_all.sh ---
    submit_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "SIM_JOB=$(sbatch --parsable 01_simulate.sh)\n"
        "echo \"Submitted 01_simulate.sh: $SIM_JOB\"\n"
        "FIT_JOB=$(sbatch --parsable --dependency=aftercorr:$SIM_JOB 02_fit.sh)\n"
        "echo \"Submitted 02_fit.sh: $FIT_JOB (dependency: aftercorr:$SIM_JOB)\"\n"
    )
    submit_path = os.path.join(outdir, 'submit_all.sh')
    with open(submit_path, 'w') as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)

    print(f"Wrote SLURM scripts for {max_index + 1} array tasks to {outdir}")
    print(f"  fit step: --cpus-per-task={cores}, no explicit --mem (Dardel's `{partition}` "
          f"partition derives memory from cores automatically), time budget {fit_time_str}")
    print(f"  {fit_time_hours}h default is calibrated (2026-07-30) for --cores 2 from real "
          f"checkpoint-timestamp throughput on the largest scenario, with margin for "
          f"noise/Phase-2-vs-Phase-1 rate uncertainty -- see docs/SIMULATION_STUDY_PLAN.md "
          f"§7. If --cores={cores} != 2, override --fit_time_hours using the §7 table "
          f"(measured totals: 2c=12.5h, 4c=7.8h, 8c=6.2h, 16c=5.2h, 32c=5.9h -- add your "
          f"own margin). Also confirm this fits within Dardel's `shared` partition's max "
          f"walltime.")
    print(f"  max_concurrent_jobs={max_concurrent_jobs} (--array=0-{max_index}%{max_concurrent_jobs}) "
          f"-- since each fit task only requests {cores} cores, this is very likely far "
          f"below what your Dardel allocation can support concurrently; check your "
          f"project's core limit and raise --max_concurrent_jobs accordingly for better "
          f"throughput (see docs/SIMULATION_STUDY_PLAN.md §7 discussion on sublinear "
          f"core scaling -- more concurrent small jobs beats fewer big ones here).")
    print("NOTE: --dependency=aftercorr ties each fit array task to the matching "
          "simulate array task (same index), not to the whole simulate array "
          "finishing first -- this lets fitting start on scenario k as soon as "
          "scenario k's data exists.")


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
                         help="CPU cores per fit task (passed through to "
                              "run_recovery_fit.py's --cores, which pins OMP/MKL/"
                              "OpenBLAS/NumExpr/torch thread counts before any heavy "
                              "import -- required for correctness on Dardel's shared "
                              "partition). Default 2, per the core-count-scaling "
                              "benchmark showing near-flat/sublinear returns beyond a "
                              "few cores for this workload -- see plan §7.")
    parser.add_argument('--partition', default='shared')
    parser.add_argument('--max_concurrent_jobs', type=int, default=50)
    parser.add_argument('--sim_cores', type=int, default=1)
    parser.add_argument('--sim_time_hours', type=float, default=0.25)
    parser.add_argument('--fit_time_hours', type=float, default=18.0,
                         help="Calibrated for --cores 2 (2026-07-30, real checkpoint-"
                              "timestamp throughput, largest scenario, plus margin). "
                              "Override if using a different --cores -- see plan §7 "
                              "table (2c=12.5h, 4c=7.8h, 8c=6.2h, 16c=5.2h, 32c=5.9h "
                              "raw measured totals, add your own margin).")
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
        sim_cores=args.sim_cores,
        sim_time_hours=args.sim_time_hours,
        fit_time_hours=args.fit_time_hours,
    )
