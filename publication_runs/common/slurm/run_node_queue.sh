#!/bin/bash
# Run a plain-text list of shell commands, up to <concurrency> at a time,
# inside a single SLURM node allocation. Used to pack several sub-node-sized
# GPU steps onto one whole Dardel GPU node instead of one node per step (see
# publication_runs/README.md "GPU node packing").
#
# No retry logic, no dynamic rebalancing -- deliberately simple, matching the
# fixed-cost-per-task pattern already used for the CPU array jobs in
# examples/simulation_study/generate_slurm_dardel.py. If a task fails, its
# line's exit status is captured in its own log and printed at the end;
# other tasks are not aborted.
#
# Usage:
#   run_node_queue.sh <tasklist.txt> <concurrency>
#
# <tasklist.txt>: one full shell command per line (e.g. a `python -m
# bayesDREAM fit-cis --config .../gene_X.yaml` invocation). Blank lines and
# lines starting with '#' are skipped. Each line's stdout/stderr goes to
# <tasklist_dir>/logs/task_<line_number>.log.

set -uo pipefail  # deliberately not -e: one failed task must not kill the pool

# `wait -n` (used below to cap concurrency) needs bash >=4.3. Dardel's login/
# compute nodes run a modern Linux bash, but fail loudly rather than
# busy-looping if this is ever run somewhere older (e.g. macOS's stock bash).
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]; }; then
    echo "run_node_queue.sh requires bash >= 4.3 (found ${BASH_VERSION}) for 'wait -n'." >&2
    exit 1
fi

TASKLIST="$1"
CONCURRENCY="${2:?concurrency required}"

if [ ! -f "$TASKLIST" ]; then
    echo "run_node_queue.sh: tasklist not found: $TASKLIST" >&2
    exit 1
fi

LOG_DIR="$(dirname "$TASKLIST")/logs"
mkdir -p "$LOG_DIR"

run_task() {
    local idx="$1"
    local cmd="$2"
    local logfile="$LOG_DIR/task_${idx}.log"
    {
        echo "[run_node_queue] task $idx starting on $(hostname): $cmd"
        eval "$cmd"
        status=$?
        echo "[run_node_queue] task $idx exited $status"
        exit $status
    } >"$logfile" 2>&1
}

n_total=0
n_failed=0
idx=0
while IFS= read -r cmd || [ -n "$cmd" ]; do
    [[ -z "${cmd// }" ]] && continue
    [[ "$cmd" =~ ^[[:space:]]*# ]] && continue
    idx=$((idx + 1))
    n_total=$((n_total + 1))

    run_task "$idx" "$cmd" &

    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do
        wait -n
    done
done <"$TASKLIST"

wait

for f in "$LOG_DIR"/task_*.log; do
    [ -e "$f" ] || continue
    if ! tail -n1 "$f" | grep -q "exited 0$"; then
        n_failed=$((n_failed + 1))
        echo "[run_node_queue] FAILED: $f"
    fi
done

echo "[run_node_queue] $((n_total - n_failed))/$n_total tasks succeeded"
if [ "$n_failed" -gt 0 ]; then
    exit 1
fi
