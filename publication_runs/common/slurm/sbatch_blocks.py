"""
SBATCH header builders shared by every dataset's generate_slurm.py.

Conventions follow examples/simulation_study/generate_slurm_dardel.py
(the one script in this repo already tuned against real Dardel behavior):

- CPU steps target the 'shared' partition (`-p shared`), sized by
  --cpus-per-task alone -- confirmed via `scontrol show partition shared`:
  DefMemPerCPU=MaxMemPerCPU=888 (MB), i.e. RAM is a FIXED 888MB per core,
  not a flexible pool -- don't pass --mem, and treat --cpus-per-task as your
  actual memory dial (cores * 888MB = your job's RAM budget). MaxTime on
  this partition is 7-00:00:00.
- GPU steps target `-p gpu`.
- Thread-pinning env vars (OMP/MKL/OPENBLAS/NUMEXPR) are exported in every
  CPU block so a task's own BLAS threadpool can't oversubscribe its
  cgroup's cores on a shared, multi-tenant node.
- GPU whole-node blocks are for steps that need a GPU at all; see
  run_node_queue.sh for packing several such steps onto one node allocation
  instead of requesting one node per step.

Restart policy (see publication_runs/README.md): only fit_trans-derived
stages (trans/permutation/recapitulation) auto-resubmit on timeout, via
`auto_requeue_on_timeout=True` below -- `fit_trans()` has its own built-in
checkpoint/resume (bayesDREAM/fitting/trans.py), so a requeued task picks
back up close to where it left off. fit_ntc/fit_cis/compensation have NO
checkpoint support in bayesDREAM at all -- requeuing those would just
restart from scratch, so they are deliberately left to fail/time out and be
reviewed manually (`common/slurm/list_job_status.py`) rather than
auto-resubmitted.

Dardel-specific values you MUST still confirm for your account (sacctmgr,
not guessed here):
- --account value (`sacctmgr show assoc user=$USER format=account`).
- MaxSubmitJobs (`sacctmgr show assoc user=$USER format=MaxSubmitJobs`) --
  see publication_runs/README.md's note on array-job submit-quota limits.
"""

from dataclasses import dataclass, field
from typing import List, Optional


def hours_to_slurm_time(hours: float) -> str:
    total_minutes = max(1, int(round(hours * 60)))
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _thread_pin_exports(cores: int) -> str:
    return (
        f"export OMP_NUM_THREADS={cores} OPENBLAS_NUM_THREADS={cores} "
        f"MKL_NUM_THREADS={cores} VECLIB_MAXIMUM_THREADS={cores} "
        f"NUMEXPR_NUM_THREADS={cores}\n"
    )


def _provenance_echo(repo_dir: str) -> str:
    return (
        f'python "{repo_dir}/publication_runs/common/git_provenance.py" '
        f"|| true  # best-effort; never fails the job\n"
    )


# Seconds before the SLURM time limit that --signal fires. Must leave enough
# time for the trap to run `scontrol requeue` and exit before SLURM SIGKILLs
# the job outright.
_REQUEUE_SIGNAL_LEAD_SECONDS = 120


def _requeue_trap(is_array: bool) -> str:
    requeue_target = '"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' if is_array else '"$SLURM_JOB_ID"'
    return (
        f"on_timeout() {{\n"
        f'    echo "[requeue] caught imminent timeout, requeuing {requeue_target}"\n'
        f"    scontrol requeue {requeue_target}\n"
        f"    exit 0\n"
        f"}}\n"
        f"trap on_timeout USR1\n"
    )


def _run_in_background_and_wait(commands: List[str]) -> List[str]:
    """Wraps commands so the USR1 trap can fire while they're running --
    a foreground `command` blocks signal delivery to the trap until it
    returns, so it must run backgrounded with an explicit `wait`."""
    body = "\n".join(commands)
    return [f"(\n{body}\n) &\nwait $!\n"]


@dataclass
class SbatchStep:
    """One SBATCH script for a single CPU step (not an array).

    auto_requeue_on_timeout: only set True for trans/permutation/
    recapitulation (fit_trans has its own checkpoint/resume) -- see this
    module's docstring for why ntc/cis/compensation must NOT set this.
    """

    job_name: str
    account: str
    log_dir: str
    time_hours: float
    cpus: int
    commands: List[str]
    partition: str = "shared"
    repo_dir: Optional[str] = None
    extra_sbatch_lines: List[str] = field(default_factory=list)
    auto_requeue_on_timeout: bool = False

    def render(self) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --output={self.log_dir}/%x_%j.out",
            f"#SBATCH --error={self.log_dir}/%x_%j.err",
            f"#SBATCH --time={hours_to_slurm_time(self.time_hours)}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --cpus-per-task={self.cpus}",
        ]
        if self.auto_requeue_on_timeout:
            lines.append(f"#SBATCH --signal=B:USR1@{_REQUEUE_SIGNAL_LEAD_SECONDS}")
            lines.append("#SBATCH --requeue")
        lines += [
            *self.extra_sbatch_lines,
            "",
            "set -euo pipefail",
            "",
            _thread_pin_exports(self.cpus),
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
        if self.auto_requeue_on_timeout:
            lines.append(_requeue_trap(is_array=False))
            lines.extend(_run_in_background_and_wait(self.commands))
        else:
            lines.extend(self.commands)
        return "\n".join(lines) + "\n"


@dataclass
class SbatchArray:
    """One SBATCH script for a CPU array job (%N throttles concurrency, does
    NOT reduce submitted-job-record count -- see publication_runs/README.md).

    auto_requeue_on_timeout: see SbatchStep's docstring -- same rule.
    """

    job_name: str
    account: str
    log_dir: str
    time_hours: float
    cpus: int
    max_index: int
    max_concurrent: int
    commands: List[str]
    partition: str = "shared"
    repo_dir: Optional[str] = None
    extra_sbatch_lines: List[str] = field(default_factory=list)
    auto_requeue_on_timeout: bool = False

    def render(self) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --output={self.log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={self.log_dir}/%x_%A_%a.err",
            f"#SBATCH --array=0-{self.max_index}%{self.max_concurrent}",
            f"#SBATCH --time={hours_to_slurm_time(self.time_hours)}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --cpus-per-task={self.cpus}",
        ]
        if self.auto_requeue_on_timeout:
            lines.append(f"#SBATCH --signal=B:USR1@{_REQUEUE_SIGNAL_LEAD_SECONDS}")
            lines.append("#SBATCH --requeue")
        lines += [
            *self.extra_sbatch_lines,
            "",
            "set -euo pipefail",
            "",
            _thread_pin_exports(self.cpus),
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
        if self.auto_requeue_on_timeout:
            lines.append(_requeue_trap(is_array=True))
            lines.extend(_run_in_background_and_wait(self.commands))
        else:
            lines.extend(self.commands)
        return "\n".join(lines) + "\n"


@dataclass
class SbatchGpuNodeQueue:
    """One whole-GPU-node allocation running run_node_queue.sh against a
    plain-text task list, for packing several sub-node-sized GPU steps onto
    one node instead of requesting a node each. See run_node_queue.sh.

    `gpu_partition` and `gpu_sbatch_lines` are deliberately left for the
    caller to fill in -- see this module's docstring for why they can't be
    guessed here.

    auto_requeue_on_timeout: same `--signal=B:USR1@120` + trap + `scontrol
    requeue` idiom as SbatchStep/SbatchArray, but requeues the WHOLE packed
    job, not an individual task -- run_node_queue.sh has no "skip tasks that
    already finished" logic, so a requeue re-runs every task in the tasklist,
    including ones that already succeeded. This is safe but wasteful for
    trans (each individual fit_trans call still resumes close to where it
    left off via its own internal checkpoint -- see this module's docstring
    -- so a redundant re-run is a few cheap iterations, not a cold restart),
    and unconditionally correct either way since every run_*.py stage here
    is idempotent (overwrites the same saved output). Set True for packed
    trans/permutation/recapitulation jobs, same rule as SbatchStep/SbatchArray.
    """

    job_name: str
    account: str
    log_dir: str
    time_hours: float
    tasklist_path: str
    concurrency: int
    gpu_partition: str
    node_queue_script: str  # path to common/slurm/run_node_queue.sh
    repo_dir: Optional[str] = None
    gpu_sbatch_lines: List[str] = field(default_factory=list)  # e.g. ["#SBATCH --gpus=8", "#SBATCH -N 1"]
    auto_requeue_on_timeout: bool = False

    def render(self) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --output={self.log_dir}/%x_%j.out",
            f"#SBATCH --error={self.log_dir}/%x_%j.err",
            f"#SBATCH --time={hours_to_slurm_time(self.time_hours)}",
            f"#SBATCH --partition={self.gpu_partition}",
            *self.gpu_sbatch_lines,
        ]
        if self.auto_requeue_on_timeout:
            lines.append(f"#SBATCH --signal=B:USR1@{_REQUEUE_SIGNAL_LEAD_SECONDS}")
            lines.append("#SBATCH --requeue")
        lines += [
            "",
            "set -euo pipefail",
            "",
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
        queue_cmd = f'bash "{self.node_queue_script}" "{self.tasklist_path}" {self.concurrency}'
        if self.auto_requeue_on_timeout:
            lines.append(_requeue_trap(is_array=False))
            lines.extend(_run_in_background_and_wait([queue_cmd]))
        else:
            lines.append(queue_cmd)
        return "\n".join(lines) + "\n"
