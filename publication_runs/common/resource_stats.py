"""
Per-job wall-clock + peak-memory tracking for publication_runs/ stage
scripts, factored out of examples/simulation_study/run_recovery_fit.py's
fit_stats.json pattern (see that module's docstring for the full rationale)
so every common/run_*.py script (and domingo/load_modalities.py) can reuse
one implementation instead of re-deriving it per script.

Two distinct resume behaviors, matching which stages have real checkpointing
(see publication_runs/README.md's "Restart policy"):

- ntc/cis (NO internal checkpoint in bayesDREAM -- a killed run restarts
  completely from scratch, there's no partial progress to add onto):
  `step_completed()` lets a caller skip re-running an already-fully-completed
  step entirely (load instead of re-fit) and carry its prior stats forward
  verbatim via `carry_forward_step()`. "Resuming" here means "don't redo
  finished work", not "continue mid-fit" -- an incomplete prior attempt's
  time is simply not counted (its work was discarded, not built upon).
- trans/permutation/recapitulation (fit_trans() has its own internal
  checkpoint, and these are the only stages SLURM auto-resubmits on timeout
  -- see sbatch_blocks.py's auto_requeue_on_timeout): `record_trans_step()`
  reads the TRUE cumulative wall-clock across every resume attempt straight
  from fit_trans()'s own '_complete.pt' checkpoint (a single process's own
  timer can't see time spent in earlier, separately-killed attempts) --
  that's the actual "add on when we restart" behavior. Peak memory across
  resumes is max(prior, this attempt), never summed -- separate process
  attempts were never running simultaneously, so summing would overstate
  true peak memory pressure at any single moment.

Every stats file is written incrementally (after each step) so a later step
crashing/getting killed still leaves earlier steps' stats on disk -- same
rationale as run_recovery_fit.py.

IMPORTANT for callers of `record_trans_step()`: it reads
'<checkpoint_dir>/trans_checkpoint_<modality_name>_complete.pt'. This MUST be
a checkpoint_dir unique to this exact fit_trans() call (e.g. a permutation
replicate's own per-rep output_dir) -- see run_permutation_null.py/
run_recapitulation_sim.py's explicit `checkpoint_dir=` wiring. Relying on
fit_trans()'s own default (`<model.output_dir>/<model.label>`) for anything
other than the ONE canonical trans fit per (gene, modality) would collide
with that canonical fit's checkpoint (or with sibling replicates', since they
'd share the same default too) -- fit_trans()'s checkpoint validation only
checks structural shape (cell/feature/group counts), which a permutation
replicate matches exactly, so it would silently resume from -- and report as
already 'complete' -- the real fit's own converged, non-permuted parameters.
"""

import json
import os
import platform
import resource
import socket
import time
from contextlib import contextmanager

import torch


def peak_rss_mb() -> float:
    """Peak resident set size (high-water mark since process start) in MB.

    NOT resettable per step -- ru_maxrss only ever increases, so each step's
    reported value is the peak *up to and including* that step, not that step
    in isolation. Units differ by platform: Linux reports KB, macOS (BSD)
    reports bytes.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 if platform.system() == "Linux" else 1024.0 ** 2)


def write_stats(stats: dict, stats_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)) or ".", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_prior_stats(stats_path: str):
    """Returns the previous run's stats dict, or None if this is a fresh
    (never-attempted) job -- or the prior write was interrupted mid-write
    (plain open(...,'w'), not atomic; treated as no prior attempt rather than
    crashing)."""
    if not os.path.exists(stats_path):
        return None
    try:
        with open(stats_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def step_completed(prior_stats, step_name: str) -> bool:
    if prior_stats is None:
        return False
    return prior_stats.get("steps", {}).get(step_name) is not None


def max_metric(prior_value, new_value):
    """max(prior, new), None-safe (e.g. peak_gpu_mb is None on CPU runs, or
    there's no prior attempt at all)."""
    if prior_value is None:
        return new_value
    if new_value is None:
        return prior_value
    return max(prior_value, new_value)


def new_stats_dict(model, extra: dict = None) -> dict:
    """Base envelope every stats file starts from: device/host/SLURM
    identifiers for cross-referencing against sacct or cluster GPU-
    utilization dashboards, same fields run_recovery_fit.py records."""
    stats = {
        "device_resolved": str(model.device),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "steps": {},
    }
    if extra:
        stats.update(extra)
    return stats


def _update_total(stats: dict) -> None:
    stats["total_elapsed_sec"] = sum(
        s["elapsed_sec"] for s in stats["steps"].values() if s.get("elapsed_sec") is not None
    )


def is_cuda(model) -> bool:
    return getattr(model, "device", None) is not None and model.device.type == "cuda"


@contextmanager
def timed_step(name: str, stats: dict, device_is_cuda: bool, stats_path: str):
    """Time one non-checkpointed step (fit_ntc/fit_cis/check_systematic_shift),
    record elapsed wall-clock + peak memory, and write the stats file
    immediately -- if a later step crashes or gets killed, completed earlier
    steps' stats are still on disk. GPU memory is a true per-step peak
    (torch.cuda.reset_peak_memory_stats() before each step); CPU memory is
    not (see peak_rss_mb)."""
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    entry = {
        "elapsed_sec": elapsed,
        "peak_rss_mb": peak_rss_mb(),
        "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if device_is_cuda else None,
    }
    stats["steps"][name] = entry
    _update_total(stats)
    write_stats(stats, stats_path)
    gpu_note = f", peak_gpu={entry['peak_gpu_mb']:.0f}MB" if device_is_cuda else ""
    print(f"[TIMING] {name}: {elapsed:.1f}s, peak_rss={entry['peak_rss_mb']:.0f}MB{gpu_note}")


def carry_forward_step(stats: dict, stats_path: str, step_name: str, prior_stats: dict) -> None:
    """Copy an already-completed step's stats forward unchanged (no new work
    happened this attempt, so there's nothing new to time) and re-persist --
    used when step_completed() says a prior attempt already finished this
    (non-checkpointed) step."""
    stats["steps"][step_name] = prior_stats["steps"][step_name]
    _update_total(stats)
    write_stats(stats, stats_path)
    entry = stats["steps"][step_name]
    print(f"[RESUME] {step_name} already completed in a prior attempt "
          f"({entry['elapsed_sec']:.1f}s, peak_rss={entry['peak_rss_mb']:.0f}MB) -- reusing, not refitting.")


@contextmanager
def timed_attempt(device_is_cuda: bool):
    """Like timed_step, but yields (elapsed_sec, peak_rss_mb, peak_gpu_mb)
    for the caller to combine with checkpoint-derived cumulative time itself
    -- used by record_trans_step's callers, which need the raw per-attempt
    numbers before folding in cumulative_elapsed_sec."""
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = {}
    yield result
    result["elapsed_sec"] = time.perf_counter() - t0
    result["peak_rss_mb"] = peak_rss_mb()
    result["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if device_is_cuda else None


def record_trans_step(
    stats: dict, stats_path: str, step_name: str, modality_name: str,
    checkpoint_dir: str, this_attempt: dict, prior_stats: dict = None,
) -> dict:
    """Record a fit_trans()-backed step's stats using its OWN checkpoint's
    cumulative_elapsed_sec (true wall-clock across every resume, including
    earlier killed attempts this process's own timer never saw) rather than
    this attempt's own elapsed timer -- the actual "add on when we restart"
    behavior. Peak memory is max(prior, this attempt): separate process
    attempts were never running simultaneously, so summing would overstate
    true peak memory pressure.

    `checkpoint_dir` MUST be the exact same directory passed to
    `model.fit_trans(checkpoint_dir=...)` for this call -- see module
    docstring's "IMPORTANT for callers" note.

    `this_attempt`: dict with 'elapsed_sec'/'peak_rss_mb'/'peak_gpu_mb' for
    just this process's own attempt (e.g. from `_timed_attempt`).
    """
    complete_ckpt_path = os.path.join(checkpoint_dir, f"trans_checkpoint_{modality_name}_complete.pt")
    elapsed = this_attempt["elapsed_sec"]
    if os.path.exists(complete_ckpt_path):
        try:
            ckpt = torch.load(complete_ckpt_path, map_location="cpu", weights_only=False)
            elapsed = ckpt.get("cumulative_elapsed_sec", this_attempt["elapsed_sec"])
        except Exception as e:
            print(f"[WARNING] Could not read cumulative_elapsed_sec from {complete_ckpt_path}: {e}. "
                  f"Recording this attempt's own elapsed time only ({this_attempt['elapsed_sec']:.1f}s), "
                  f"which undercounts any prior killed attempts.")

    prior_entry = ((prior_stats or {}).get("steps", {}) or {}).get(step_name) or {}
    entry = {
        "elapsed_sec": elapsed,
        "peak_rss_mb": max_metric(prior_entry.get("peak_rss_mb"), this_attempt["peak_rss_mb"]),
        "peak_gpu_mb": max_metric(prior_entry.get("peak_gpu_mb"), this_attempt["peak_gpu_mb"]),
    }
    stats["steps"][step_name] = entry
    _update_total(stats)
    write_stats(stats, stats_path)
    gpu_note = f", peak_gpu={entry['peak_gpu_mb']:.0f}MB" if entry["peak_gpu_mb"] is not None else ""
    print(f"[TIMING] {step_name}: {elapsed:.1f}s cumulative "
          f"({this_attempt['elapsed_sec']:.1f}s this attempt), "
          f"peak_rss={entry['peak_rss_mb']:.0f}MB{gpu_note}")
    return entry
