"""
Generate SLURM array scripts for the single-Hill recovery study on Berzelius.
See docs/SIMULATION_STUDY_PLAN.md §7.

bayesDREAM.slurm_jobgen.SlurmJobGenerator is built around "N cis genes within one
dataset" (1 fit_ntc + N fit_cis + N fit_trans jobs). This study is the opposite shape:
540 independent tiny datasets (scenario x replicate), each with its own
fit_ntc+fit_cis+fit_trans. So this script does NOT call
SlurmJobGenerator.generate_all_scripts() — it reuses SlurmJobGenerator only for its
memory/time *estimation* math (estimate_memory_requirements /
estimate_time_requirements), run once on the largest scenario (cells_per_gene=1000)
as a representative sizing reference, then writes two custom array scripts:

    01_simulate.sh   array 0-539, CPU, one task = one simulate_scenario.py call
    02_fit.sh        array 0-539, depends on 01, one task = one run_recovery_fit.py call
    submit_all.sh    submits 01 then 02 with --dependency=afterok

Usage:
    python generate_slurm.py --design_matrix ./sim_study_out/design_matrix.csv \
        --outdir ./sim_study_out/slurm \
        --data_path /proj/.../sim_study_out/data \
        --python_env /proj/.../pyroenv/bin/python \
        --bayesdream_path /proj/.../bayesDREAM \
        --examples_path /proj/.../bayesDREAM/examples/simulation_study
"""

import argparse
import os

import numpy as np
import pandas as pd

from bayesDREAM.simulation import simulate_scenario
from bayesDREAM.slurm_jobgen import SlurmJobGenerator


def _representative_dataset(design_matrix: pd.DataFrame):
    """Simulate the largest scenario (cells_per_gene=1000) once, to size fit_trans
    (the expensive step: T~1737 features x up to 1000 cells) for the whole study.
    All 540 scenarios share the same 1736-feature trans panel, so T is constant;
    only N (cells_per_gene) varies, and 1000 is the max."""
    row = design_matrix.loc[design_matrix['cells_per_gene'] == design_matrix['cells_per_gene'].max()].iloc[0]
    result = simulate_scenario(
        cells_per_gene=int(row['cells_per_gene']),
        n_guides=int(row['n_guides']),
        guide_shape=row['guide_shape'],
        sigma_eff=float(row['sigma_eff']),
        log2_X_NTC=float(row['log2_X_NTC']),
        log2_o_x=float(row['log2_o_x']),
        seed=int(row['seed']),
    )
    return result['meta'], result['counts']


SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err
#SBATCH --array=0-{max_index}%{max_concurrent}
#SBATCH --time={time}
#SBATCH --partition={partition}
{constraint_line}{gpu_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G

set -euo pipefail
module load Anaconda/2021.05-nsc1 2>/dev/null || true
"""


def _slurm_time_to_hours(time_str: str) -> float:
    h, m, s = (int(x) for x in time_str.split(':'))
    return h + m / 60.0 + s / 3600.0


def _hours_to_slurm_time(hours: float) -> str:
    total_minutes = max(1, int(round(hours * 60)))
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _resource_block(resources: dict, max_index: int, max_concurrent: int, job_name: str,
                     log_dir: str, time_str: str) -> str:
    constraint_line = f"#SBATCH -C {resources['constraint']}\n" if resources.get('constraint') else ""
    gpu_line = f"#SBATCH --gpus={resources['gpus']}\n" if resources.get('gpus') else ""
    return SBATCH_HEADER.format(
        job_name=job_name, log_dir=log_dir, max_index=max_index, max_concurrent=max_concurrent,
        time=time_str, partition=resources['partition'], constraint_line=constraint_line,
        gpu_line=gpu_line, cpus=resources.get('cpus', 1), mem_gb=int(np.ceil(resources['mem_gb'])),
    )


def generate_slurm_scripts(
    design_matrix_path: str,
    outdir: str,
    data_path: str,
    python_env: str,
    bayesdream_path: str,
    examples_path: str,
    max_concurrent_jobs: int = 50,
    time_multiplier: float = 1.0,
    partition_preference: str = 'auto',
    min_fit_hours: float = 2.0,
):
    design_matrix = pd.read_csv(design_matrix_path)
    max_index = int(design_matrix['row_index'].max())

    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    meta, counts = _representative_dataset(design_matrix)
    gen = SlurmJobGenerator(
        meta=meta, counts=counts, cis_genes=['CisGene'],
        output_dir=os.path.join(outdir, '_sizing_scratch'),
        label='sim_study_sizing',
        max_concurrent_jobs=max_concurrent_jobs,
        time_multiplier=time_multiplier,
        partition_preference=partition_preference,
        python_env=python_env, bayesdream_path=bayesdream_path, data_path=data_path,
    )
    memory = gen.estimate_memory_requirements()
    times = gen.estimate_time_requirements(memory)

    # --- 01_simulate.sh: cheap, CPU-only, no bayesDREAM SVI ---
    sim_resources = dict(partition='berzelius-cpu', cpus=1, mem_gb=8.0)
    sim_script = _resource_block(
        sim_resources, max_index, max_concurrent_jobs, f"{gen.label}_simulate", log_dir,
        time_str='00:15:00',
    )
    sim_script += (
        f"\n{python_env} {examples_path}/simulate_scenario.py "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $SLURM_ARRAY_TASK_ID "
        f"--outdir {data_path}\n"
    )
    with open(os.path.join(outdir, '01_simulate.sh'), 'w') as f:
        f.write(sim_script)

    # --- 02_fit.sh: fit_ntc + fit_cis + fit_trans, sized off the largest scenario ---
    fit_resources = memory['resources'].get('fit_trans', {
        'partition': 'berzelius-cpu', 'cpus': 1, 'mem_gb': memory['fit_trans_ram_gb'],
    })
    # SlurmJobGenerator's time estimator scales purely off T*N relative to a
    # "20K genes x 30K cells" baseline. At this study's scale (T~1737, N<=1000) that
    # scaling factor is ~0.003, so the raw estimate rounds to 00:00:00 -- the fixed
    # per-SVI-iteration overhead (independent of T*N) dominates wall time here, not
    # dataset size, which the estimator doesn't model. Enforce a floor instead of
    # trusting the size-based estimate at this scale.
    raw_trans_time = times.get('fit_trans', '00:00:00')
    fit_time_hours = max(_slurm_time_to_hours(raw_trans_time), min_fit_hours)
    fit_time_str = _hours_to_slurm_time(fit_time_hours)
    fit_script = _resource_block(
        fit_resources, max_index, max_concurrent_jobs, f"{gen.label}_fit", log_dir,
        time_str=fit_time_str,
    )
    device = 'cuda' if fit_resources.get('gpus') else 'cpu'
    fit_script += (
        f"\n{python_env} {examples_path}/run_recovery_fit.py "
        f"--data_root {data_path} "
        f"--design_matrix {design_matrix_path} "
        f"--row_index $SLURM_ARRAY_TASK_ID "
        f"--device {device}\n"
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
    print(f"  fit_trans sizing (from cells_per_gene=1000 scenario): "
          f"{fit_resources.get('mem_gb', memory['fit_trans_ram_gb']):.1f} GB RAM, "
          f"partition={fit_resources.get('partition')}, "
          f"constraint={fit_resources.get('constraint')}, "
          f"gpus={fit_resources.get('gpus', 0)}")
    print(f"  fit step time budget: {fit_time_str} (raw size-based estimate was "
          f"{raw_trans_time}; floored to --min_fit_hours={min_fit_hours}). "
          f"Recalibrate --min_fit_hours from a real timed run before submitting the "
          f"full 540-task array — see docs/SIMULATION_STUDY_PLAN.md §7.")
    print("NOTE: --dependency=aftercorr ties each fit array task to the matching "
          "simulate array task (same index), not to the whole simulate array "
          "finishing first — this lets fitting start on scenario k as soon as "
          "scenario k's data exists.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_matrix', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--data_path', required=True,
                         help="Cluster path where simulate_scenario.py writes/reads "
                              "per-scenario data (passed through as --outdir/--data_root).")
    parser.add_argument('--python_env', required=True)
    parser.add_argument('--bayesdream_path', required=True)
    parser.add_argument('--examples_path', required=True,
                         help="Cluster path to examples/simulation_study/ "
                              "(containing simulate_scenario.py, run_recovery_fit.py).")
    parser.add_argument('--max_concurrent_jobs', type=int, default=50)
    parser.add_argument('--time_multiplier', type=float, default=1.0)
    parser.add_argument('--partition_preference', default='auto')
    parser.add_argument('--min_fit_hours', type=float, default=2.0,
                         help="Floor for the fit step's SLURM time limit, since "
                              "SlurmJobGenerator's size-based estimator underestimates "
                              "at this study's scale (see printed NOTE). Recalibrate "
                              "from a real timed run before the full submission.")
    args = parser.parse_args()

    generate_slurm_scripts(
        design_matrix_path=args.design_matrix,
        outdir=args.outdir,
        data_path=args.data_path,
        python_env=args.python_env,
        bayesdream_path=args.bayesdream_path,
        examples_path=args.examples_path,
        min_fit_hours=args.min_fit_hours,
        max_concurrent_jobs=args.max_concurrent_jobs,
        time_multiplier=args.time_multiplier,
        partition_preference=args.partition_preference,
    )
