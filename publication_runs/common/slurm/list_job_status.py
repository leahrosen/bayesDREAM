"""
List SLURM status for every job submitted by a dataset's submit_all.sh, for
manual review. Per project policy: only fit_trans-derived stages
(trans/permutation/recapitulation) auto-resubmit on timeout (see
sbatch_blocks.py's auto_requeue_on_timeout) -- everything else (ntc/cis/
compensation/cis_sweep) that fails or times out is left for YOU to look at
and decide on, not auto-retried. This script is that "look at" step: no
guessing, no auto-anything, just sacct state joined back to which
stage/gene each job was.

Usage
-----
    python list_job_status.py <submitted_jobs.tsv> [<submitted_jobs.tsv> ...]

Each <submitted_jobs.tsv> is written by a dataset's submit_all.sh
(columns: stage, label, jobid, script), one row per `sbatch` call.

Prints one row per job: stage, label, jobid, SLURM state, elapsed. Jobs
whose state is not COMPLETED are grouped into a "NEEDS ATTENTION" section at
the end, with the sbatch script path to resubmit if you decide to.
"""

import argparse
import subprocess
import sys

import pandas as pd

_ACTIVE_STATES = {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "RESIZING", "SUSPENDED"}


def _sacct_states(job_ids: list) -> dict:
    """{jobid: (State, Elapsed)} via one batched sacct call. Degrades to
    'unknown' per job (with a warning) rather than crashing -- this is a
    monitoring convenience, not something that should block you from seeing
    the rest of the table if sacct hiccups."""
    if not job_ids:
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-j", ",".join(str(j) for j in job_ids),
             "--format=JobID,State,Elapsed", "-X", "-n", "-P"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"[WARN] sacct query failed ({e}); states will show as 'unknown'.", file=sys.stderr)
        return {}

    states = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobid, state, elapsed = parts[0], parts[1], parts[2]
        states[jobid] = (state, elapsed)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tsv_paths", nargs="+", help="One or more submitted_jobs.tsv files.")
    args = parser.parse_args()

    rows = []
    for path in args.tsv_paths:
        df = pd.read_csv(path, sep="\t")
        df["source_tsv"] = path
        rows.append(df)
    jobs = pd.concat(rows, ignore_index=True)

    states = _sacct_states(jobs["jobid"].tolist())
    jobs["state"] = jobs["jobid"].astype(str).map(lambda j: states.get(j, ("unknown", ""))[0])
    jobs["elapsed"] = jobs["jobid"].astype(str).map(lambda j: states.get(j, ("unknown", ""))[1])

    print(jobs[["stage", "label", "jobid", "state", "elapsed"]].to_string(index=False))

    needs_attention = jobs[~jobs["state"].isin({"COMPLETED", *_ACTIVE_STATES})]
    if len(needs_attention):
        print("\n=== NEEDS ATTENTION (not COMPLETED, not currently running/pending) ===")
        for _, row in needs_attention.iterrows():
            print(f"  [{row['state']}] {row['stage']} / {row['label']} (job {row['jobid']}) -> {row['script']}")
    else:
        print("\nAll jobs COMPLETED or still active.")


if __name__ == "__main__":
    main()
