"""
SBATCH header builders shared by every dataset's generate_slurm.py.

Conventions follow examples/simulation_study/generate_slurm_dardel.py
(the one script in this repo already tuned against real Dardel behavior):

- CPU steps target the 'shared' partition, sized by --cpus-per-task alone
  (Dardel derives --mem from cores automatically there; don't pass --mem).
- Thread-pinning env vars (OMP/MKL/OPENBLAS/NUMEXPR) are exported in every
  block so a task's own BLAS threadpool can't oversubscribe its cgroup's
  cores on a shared, multi-tenant node.
- GPU whole-node blocks are for steps that need a GPU at all; see
  run_node_queue.sh for packing several such steps onto one node allocation
  instead of requesting one node per step.

Dardel-specific values you MUST confirm for your account before using this
(sinfo / sacctmgr, not guessed here):
- GPU partition name and any node --constraint (Dardel's GPU nodes are not
  the same partition as 'shared'; check `sinfo -o "%P %G %N"`).
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


@dataclass
class SbatchStep:
    """One SBATCH script for a single CPU step (not an array)."""

    job_name: str
    account: str
    log_dir: str
    time_hours: float
    cpus: int
    commands: List[str]
    partition: str = "shared"
    repo_dir: Optional[str] = None
    extra_sbatch_lines: List[str] = field(default_factory=list)

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
            *self.extra_sbatch_lines,
            "",
            "set -euo pipefail",
            "",
            _thread_pin_exports(self.cpus),
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
        lines.extend(self.commands)
        return "\n".join(lines) + "\n"


@dataclass
class SbatchArray:
    """One SBATCH script for a CPU array job (%N throttles concurrency, does
    NOT reduce submitted-job-record count -- see publication_runs/README.md).
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
            *self.extra_sbatch_lines,
            "",
            "set -euo pipefail",
            "",
            _thread_pin_exports(self.cpus),
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
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
    gpu_sbatch_lines: List[str] = field(default_factory=list)  # e.g. ["#SBATCH --gpus=1", "#SBATCH -N 1"]

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
            "",
            "set -euo pipefail",
            "",
        ]
        if self.repo_dir:
            lines.append(_provenance_echo(self.repo_dir))
        lines.append("")
        lines.append(f'bash "{self.node_queue_script}" "{self.tasklist_path}" {self.concurrency}')
        return "\n".join(lines) + "\n"
