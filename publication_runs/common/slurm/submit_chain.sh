#!/bin/bash
# Submit a list of sbatch scripts, chaining each to the previous one via
# --dependency=afterok:<jobid>. Each dataset's generate_slurm.py writes a
# submit_all.sh that calls this (or does the equivalent inline) rather than
# hand-rolling dependency chains per dataset.
#
# IMPORTANT: --dependency only delays when a job is allowed to *run* -- it
# does NOT delay when the job is *counted* against your account's
# MaxSubmitJobs quota (every `sbatch` call, including array jobs, registers
# immediately). Chaining N scripts here still submits N job records right
# away. If you're chaining an array job (many tasks, one submission) with
# downstream single-job steps, that's fine (bounded record count); if you're
# chaining multiple LARGE arrays end-to-end, check
# `sacctmgr show assoc user=$USER format=MaxSubmitJobs` first -- see
# publication_runs/README.md and
# examples/simulation_study/generate_slurm_dardel.py's module docstring for
# the incident that motivated this warning.
#
# Usage:
#   submit_chain.sh script1.sh script2.sh script3.sh ...
# Each script2..N is submitted with --dependency=afterok:<previous jobid>.
# Prints "name: jobid" per script, and a final space-separated list of all
# jobids to stdout's last line (for capture by a caller).

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: submit_chain.sh script1.sh [script2.sh ...]" >&2
    exit 1
fi

prev_jobid=""
all_jobids=()

for script in "$@"; do
    if [ -n "$prev_jobid" ]; then
        jobid=$(sbatch --parsable --dependency=afterok:"$prev_jobid" "$script")
    else
        jobid=$(sbatch --parsable "$script")
    fi
    echo "$(basename "$script"): $jobid"
    all_jobids+=("$jobid")
    prev_jobid="$jobid"
done

echo "${all_jobids[@]}"
