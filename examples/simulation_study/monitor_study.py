"""
Monitor progress of the full simulation study across all design_matrix.csv tasks --
cross-references SLURM job state (via sacct) with per-scenario fit progress (via
fit/recovery/fit_stats.json) into one summary, and flags tasks whose SLURM job ended
without the fit actually completing (crash, timeout, OOM, etc.).

Usage (single submission wave):
    python monitor_study.py --design_matrix $OUT/design_matrix.csv --data_root $DATA \
        --job_id <id from submit_all.sh's "Submitted 01_run.sh: <id>">

Usage (multiple waves, e.g. because the design matrix exceeds SLURM's MaxArraySize --
see generate_slurm_dardel.py's ROW_OFFSET): pass --job_id once per wave, as
JOBID or JOBID:OFFSET (offset defaults to 0, matching a wave submitted without
--export=ROW_OFFSET=...):
    python monitor_study.py --design_matrix $OUT/design_matrix.csv --data_root $DATA \
        --job_id 22498334 --job_id 22512368:600

Each wave's own sacct array task IDs always start at 0 regardless of ROW_OFFSET (SLURM
itself has no notion of the offset -- it's purely a shell variable 01_run.sh applies),
so the offset given here must match whatever --export=ROW_OFFSET=<N> that wave was
actually submitted with, or the SLURM-state column will be silently misaligned against
the wrong row_index.

--job_id is optional -- without it, only data-level progress is shown (no SLURM state
column, no failure-detection section), which still works for spot-checking progress
without needing to know the current job ID(s).
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

# SLURM states that mean a task is still active (not yet finished one way or another).
_ACTIVE_STATES = {'RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'RESIZING', 'SUSPENDED'}


def _load_fit_stats(scenario_dir: str):
    path = os.path.join(scenario_dir, 'fit', 'recovery', 'fit_stats.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Being written concurrently by a still-running task -- transient, not a real error.
        return 'unreadable'


def _progress_stage(scenario_dir: str, stats) -> str:
    if stats == 'unreadable':
        return 'unreadable'
    if stats is None:
        return 'simulated' if os.path.exists(os.path.join(scenario_dir, 'meta.csv')) else 'not_simulated'
    if 'total_elapsed_sec' in stats:
        return 'fit_trans_complete'
    steps = stats.get('steps', {})
    if 'fit_cis' in steps:
        return 'fit_cis_done'
    if 'fit_ntc' in steps:
        return 'fit_ntc_done'
    return 'fit_started'


def _sacct_states(job_id: str) -> dict:
    """{array_task_id: SLURM state} for one array job, via sacct. Empty dict (with a
    warning) if sacct isn't available or the query fails -- callers should degrade
    gracefully rather than crash, since this is a monitoring convenience, not critical
    infrastructure."""
    try:
        out = subprocess.run(
            ['sacct', '-j', str(job_id), '--format=JobID,State', '-X', '-n', '-P'],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"[WARN] sacct query failed for job {job_id} ({e}); its tasks' SLURM "
              f"state will show as 'unknown'.", file=sys.stderr)
        return {}

    states = {}
    for line in out.strip().splitlines():
        parts = line.split('|')
        if len(parts) != 2:
            continue
        jobid_field, state = parts
        if '_' not in jobid_field:
            continue  # the array job's own summary line (no _<task>), not a per-task line
        _, task_id_str = jobid_field.split('_', 1)
        try:
            task_id = int(task_id_str)
        except ValueError:
            continue
        # e.g. "CANCELLED by 0" -> "CANCELLED"; keep the first token only.
        states[task_id] = state.split()[0]
    return states


def _parse_job_spec(spec: str):
    """'22498334' -> ('22498334', 0); '22512368:600' -> ('22512368', 600)."""
    if ':' in spec:
        job_id, offset_str = spec.split(':', 1)
        return job_id, int(offset_str)
    return spec, 0


def _merged_slurm_states(job_specs) -> dict:
    """row_index -> (SLURM state, job_id, offset) across every given (job_id, offset)
    pair. If the same row_index is covered by more than one job (e.g. a task was
    resubmitted as its own standalone job), the LAST job_spec in the list wins --
    matches the intuitive "later submission is the current one" reading."""
    merged = {}
    for job_id, offset in job_specs:
        for task_id, state in _sacct_states(job_id).items():
            merged[task_id + offset] = (state, job_id, offset)
    return merged


def main(design_matrix_path: str, data_root: str, job_specs=None):
    job_specs = job_specs or []
    design_matrix = pd.read_csv(design_matrix_path)
    slurm_states = _merged_slurm_states(job_specs) if job_specs else {}

    rows = []
    for _, row in design_matrix.iterrows():
        row_index = int(row['row_index'])
        scenario_id = int(row['scenario_id'])
        replicate_id = int(row['replicate_id'])
        scenario_dir = os.path.join(data_root, f"scenario_{scenario_id}", f"rep_{replicate_id}")
        stats = _load_fit_stats(scenario_dir)
        stage = _progress_stage(scenario_dir, stats)
        total_elapsed_sec = stats.get('total_elapsed_sec') if isinstance(stats, dict) else None
        state, src_job, src_offset = slurm_states.get(
            row_index, ('unknown' if job_specs else '', None, None))
        rows.append({
            'row_index': row_index, 'scenario_id': scenario_id, 'replicate_id': replicate_id,
            'stage': stage, 'slurm_state': state, 'src_job': src_job, 'src_offset': src_offset,
            'total_elapsed_sec': total_elapsed_sec,
        })
    df = pd.DataFrame(rows)

    print(f"Total tasks: {len(df)}\n")
    print("Progress stage counts:")
    print(df['stage'].value_counts().to_string())
    print()

    if job_specs:
        print("SLURM state counts:")
        print(df['slurm_state'].value_counts().to_string())
        print()

        terminal = ~df['slurm_state'].isin(_ACTIVE_STATES | {'unknown'})
        not_done = df['stage'] != 'fit_trans_complete'
        problem = df[terminal & not_done]
        if len(problem):
            print(f"POSSIBLE FAILURES ({len(problem)} tasks -- SLURM job ended but "
                  f"fit_trans never completed):")
            print(problem[['row_index', 'scenario_id', 'replicate_id', 'stage',
                            'slurm_state', 'src_job']].to_string(index=False))
            print()
            print(f"--array spec(s) for resubmitting just these tasks (fit_trans will "
                  f"auto-resume from each one's last checkpoint, if any -- see "
                  f"docs/SIMULATION_STUDY_PLAN.md). Grouped by the offset each task's "
                  f"row_index needs, since a resubmission must use the same "
                  f"--export=ROW_OFFSET as whichever wave originally covered it:")
            for offset, group in problem.groupby('src_offset', dropna=False):
                offset = 0 if pd.isna(offset) else int(offset)
                local_indices = sorted(int(r) - offset for r in group['row_index'])
                array_spec = ','.join(str(i) for i in local_indices)
                print(f"  sbatch --array={array_spec} --export=ALL,ROW_OFFSET={offset} "
                      f"<path to 01_run.sh>")
        else:
            print("No terminal-but-incomplete tasks detected.")
        print()

    completed = df[df['total_elapsed_sec'].notna()]
    if len(completed):
        hrs = completed['total_elapsed_sec'] / 3600.0
        print(f"Completed task timing (n={len(completed)}): "
              f"mean={hrs.mean():.2f}h, min={hrs.min():.2f}h, max={hrs.max():.2f}h")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_matrix', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--job_id', action='append', default=None,
                         help="SLURM array job ID from submit_all.sh's/sbatch's "
                              "'Submitted batch job <id>' output. Pass multiple times "
                              "for multiple submission waves, as JOBID or "
                              "JOBID:OFFSET (offset matches that wave's "
                              "--export=ROW_OFFSET=<N>, default 0). Enables the SLURM "
                              "state column and failure-detection section.")
    args = parser.parse_args()
    job_specs = [_parse_job_spec(s) for s in args.job_id] if args.job_id else []
    main(args.design_matrix, args.data_root, job_specs)
