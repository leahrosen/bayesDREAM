"""
Fit the recovery model for one simulated cell-design scenario/replicate, following
bayesDREAM's documented full workflow (docs/FIT_TRANS_GUIDE.md "Complete Workflow
Example"): fit_ntc -> adjust_ntc_sum_factor -> fit_cis -> refit_sumfactor ->
fit_trans(additive_hill). See docs/SIMULATION_STUDY_PLAN.md §7.

niters/nsamples are intentionally NOT exposed here -- every fit_ntc/fit_cis/fit_trans
call uses bayesDREAM's own library defaults, deliberately: the point of this study is
to test whether the default settings recover known parameters, not some other,
study-specific number of iterations.

Reuses the scenario's own seed (recorded in config.json by simulate_scenario.py) for
numpy/torch/pyro, so a rerun of a given scenario+replicate is deterministic end to end
(plan §6).

Writes fit/recovery/fit_stats.json: per-step (fit_ntc/fit_cis/fit_trans) wall-clock
time, peak CPU RSS, and peak GPU memory (torch.cuda.max_memory_allocated(), true
per-step since it's reset before each step -- CPU RSS is a running high-water mark,
not resettable, see _peak_rss_mb), plus hostname/SLURM job+array-task ID/resolved
device for cross-referencing against sacct or cluster GPU-utilization dashboards.
Written incrementally after each step, not just at the end, so a later step crashing
or getting killed (e.g. by an external SIGTERM, which skips Python cleanup code)
still leaves completed earlier steps' stats on disk.

Resuming a killed/timed-out run (re-running this script on the same --scenario_dir):
  - fit_ntc/fit_cis: auto-detected via the *previous* fit_stats.json -- if it recorded
    that step as completed, this run calls model.load_ntc_fit()/load_cis_fit() instead
    of re-fitting, and copies that step's prior elapsed_sec/peak_rss_mb/peak_gpu_mb
    forward unchanged (no new SVI work happened, so there's nothing new to time).
    Deliberately keyed off fit_stats.json rather than just checking whether the
    output .pt files exist: save_ntc_fit()/save_cis_fit() use plain torch.save(), not
    the atomic write-then-rename pattern fit_trans's checkpointing uses, so a file
    that exists isn't proof it's complete/uncorrupted if the process died mid-write --
    fit_stats.json's step entry is only written after that step's `with _timed_step`
    block exits normally, so its presence is a much stronger completion signal.
  - fit_trans: already resumes from its own last checkpoint automatically (bayesDREAM
    default, restart_from_checkpoint=True) -- no change needed to trigger that. What
    IS new: fit_trans's checkpoints now also carry 'cumulative_elapsed_sec' (wall-clock
    summed across every resume, not just this process's own timer -- a single
    process's clock can't see time spent in earlier, separately-killed attempts). This
    script reads that back from the checkpoint after fit_trans() returns and records
    THAT as fit_trans's elapsed_sec, rather than this attempt's own (undercounting)
    timer. Peak memory for fit_trans is max(this attempt, prior attempt) -- unlike
    time, memory across separate processes was never simultaneous, so summing would
    overstate true peak usage.

Usage (single task):
    python run_recovery_fit.py --scenario_dir ./sim_study_out/data/scenario_0/rep_0

Usage (SLURM array over design_matrix rows; resolves scenario_dir from
--data_root/--design_matrix/$SLURM_ARRAY_TASK_ID):
    python run_recovery_fit.py --data_root ./sim_study_out/data \
        --design_matrix ./sim_study_out/design_matrix.csv

Thread pinning (--cores / $SLURM_CPUS_PER_TASK): resolved and applied *before*
numpy/pandas/pyro/torch/bayesDREAM are imported anywhere in this process, and
deliberately not via bayesDREAM.utils.set_max_threads() -- that helper sets
OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS, which are
read once when each library's C-extension threadpool first initializes, not on every
call, and bayesDREAM.utils itself imports numpy/torch at module load, so calling it
the "normal" way is already too late. On a partition that hands out a whole node
(or a fixed-size GPU slice) per job this doesn't matter -- there's nothing else on
the node to oversubscribe. On a *shared* CPU partition (e.g. Dardel) where SLURM
routinely co-locates multiple array tasks on the same physical node, it matters a
lot: without this, each task's BLAS threadpool may try to use every core visible on
the node rather than just the cores actually allocated to it, oversubscribing every
co-located job at once.
"""

import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario_dir', default=None,
                         help="Directory written by simulate_scenario.py for one "
                              "scenario/replicate. Mutually exclusive with "
                              "--data_root/--design_matrix/--row_index.")
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--design_matrix', default=None)
    parser.add_argument('--row_index', type=int, default=None,
                         help="Defaults to $SLURM_ARRAY_TASK_ID when using "
                              "--data_root/--design_matrix.")
    parser.add_argument('--device', default=None,
                         help="'cpu' or 'cuda'. Defaults to bayesDREAM's own "
                              "auto-detection (cuda if available, else cpu) — pass "
                              "explicitly to force one or the other.")
    parser.add_argument('--cores', type=int, default=None,
                         help="CPU threads to pin OMP/OpenBLAS/MKL/NumExpr and "
                              "PyTorch's intra-op threadpool to. Defaults to "
                              "$SLURM_CPUS_PER_TASK when set (normal under SLURM "
                              "with --cpus-per-task); pass explicitly for local runs "
                              "or schedulers that don't set that variable. See the "
                              "module docstring for why this must be resolved before "
                              "any heavy import in this file.")
    return parser.parse_args()


args = _parse_args()

# Must happen before numpy/torch/bayesDREAM are imported anywhere in this process --
# see module docstring ("Thread pinning").
_cores = args.cores or os.environ.get('SLURM_CPUS_PER_TASK')
if _cores is not None:
    _cores = str(int(_cores))
    for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[_var] = _cores

import json
import platform
import resource
import socket
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyro
import torch

from bayesDREAM import bayesDREAM

if _cores is not None:
    # The env vars above only bind the C-extension (BLAS/OpenMP) threadpools;
    # PyTorch's own intra-op threadpool is a separate runtime-settable knob that
    # works regardless of import order, so it's set here explicitly too.
    torch.set_num_threads(int(_cores))


def resolve_scenario_dir(data_root: str, design_matrix_path: str, row_index: int) -> str:
    design_matrix = pd.read_csv(design_matrix_path)
    row = design_matrix.loc[design_matrix['row_index'] == row_index].iloc[0]
    return os.path.join(
        data_root, f"scenario_{int(row['scenario_id'])}", f"rep_{int(row['replicate_id'])}",
    )


def _peak_rss_mb() -> float:
    """Peak resident set size (high-water mark since process start) in MB.

    NOT resettable per step (unlike GPU memory tracking below) -- ru_maxrss only
    ever increases, so each step's reported value is the peak *up to and including*
    that step, not that step in isolation. Still informative: shows the running
    high-water mark growing through the pipeline. Units differ by platform: Linux
    reports KB, macOS (BSD) reports bytes.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 if platform.system() == 'Linux' else 1024.0 ** 2)


def _write_stats(stats: dict, stats_path: str) -> None:
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


def _load_prior_stats(stats_path: str):
    """Returns the previous run's fit_stats.json dict, or None if this is a fresh
    (never-attempted) scenario/replicate."""
    if not os.path.exists(stats_path):
        return None
    try:
        with open(stats_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Prior attempt was killed mid-write to this exact file (plain open(...,'w'),
        # not atomic) -- treat as if there were no prior attempt rather than crashing.
        return None


def _step_completed(prior_stats, step_name: str) -> bool:
    if prior_stats is None:
        return False
    return prior_stats.get('steps', {}).get(step_name) is not None


def _max_metric(prior_value, new_value):
    """max(prior, new), None-safe (e.g. peak_gpu_mb is None on CPU runs, or there's
    no prior attempt at all)."""
    if prior_value is None:
        return new_value
    if new_value is None:
        return prior_value
    return max(prior_value, new_value)


@contextmanager
def _timed_step(name: str, stats: dict, device_is_cuda: bool, stats_path: str):
    """Time one fit step, record elapsed wall-clock + peak memory, and write the
    stats file immediately -- if a later step crashes or gets killed (e.g. by an
    external SIGTERM, which doesn't give Python a chance to run cleanup code), the
    completed earlier steps' stats are still on disk rather than lost. GPU memory
    is a true per-step peak (torch.cuda.reset_peak_memory_stats() before each step);
    CPU memory is not (see _peak_rss_mb).
    """
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    entry = {
        'elapsed_sec': elapsed,
        'peak_rss_mb': _peak_rss_mb(),
        'peak_gpu_mb': (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if device_is_cuda else None,
    }
    stats['steps'][name] = entry
    _write_stats(stats, stats_path)
    gpu_note = f", peak_gpu={entry['peak_gpu_mb']:.0f}MB" if device_is_cuda else ""
    print(f"[TIMING] {name}: {elapsed:.1f}s, peak_rss={entry['peak_rss_mb']:.0f}MB{gpu_note}")


def run_recovery_fit(
    scenario_dir: str,
    device: str = None,
) -> bayesDREAM:
    with open(os.path.join(scenario_dir, 'config.json')) as f:
        config = json.load(f)
    seed = config['seed']

    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    meta = pd.read_csv(os.path.join(scenario_dir, 'meta.csv'))
    counts = pd.read_csv(os.path.join(scenario_dir, 'counts.csv'), index_col=0)

    fit_dir = os.path.join(scenario_dir, 'fit')
    os.makedirs(fit_dir, exist_ok=True)

    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene=config['cis_gene_name'],
        output_dir=fit_dir,
        label='recovery',
        device=device,
        random_seed=seed,
    )
    device_is_cuda = model.device.type == 'cuda'

    recovery_dir = os.path.join(fit_dir, 'recovery')
    os.makedirs(recovery_dir, exist_ok=True)
    stats_path = os.path.join(recovery_dir, 'fit_stats.json')

    prior_stats = _load_prior_stats(stats_path)
    resumed_ntc = _step_completed(prior_stats, 'fit_ntc')
    resumed_cis = _step_completed(prior_stats, 'fit_cis')

    stats = {
        'device_requested': device,
        'device_resolved': str(model.device),
        'cores_requested': _cores,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'resumed_fit_ntc': resumed_ntc,
        'resumed_fit_cis': resumed_cis,
        # fit_ntc/fit_cis carried forward verbatim below if resumed; fit_trans always
        # gets a fresh entry once it returns (see the checkpoint-time-read below).
        'steps': {
            'fit_ntc': prior_stats['steps']['fit_ntc'] if resumed_ntc else None,
            'fit_cis': prior_stats['steps']['fit_cis'] if resumed_cis else None,
        },
    }
    stats['steps'] = {k: v for k, v in stats['steps'].items() if v is not None}
    _write_stats(stats, stats_path)

    # single technical group (plan §2 / §4.1): C=1 is an explicit no-group-effect
    # code path in fit_ntc/fit_cis/fit_trans, not a degenerate/unsupported case.
    model.set_technical_groups(['cell_line'])
    if resumed_ntc:
        print(f"[RESUME] fit_ntc already completed in a prior attempt "
              f"({prior_stats['steps']['fit_ntc']['elapsed_sec']:.1f}s) — loading instead of refitting.")
        model.load_ntc_fit()
    else:
        with _timed_step('fit_ntc', stats, device_is_cuda, stats_path):
            model.fit_ntc(sum_factor_col='sum_factor')
        # Save immediately after each step, not just at the end: fit_cis/fit_trans are
        # the steps most likely to fail (longer runtime, more iterations, e.g. the
        # AutoIAFNormal NaN failure mode this study surfaced) -- if a later step
        # crashes or times out, the earlier steps' results (and this step's
        # timing/memory stats) are still on disk to inspect rather than lost entirely.
        model.save_ntc_fit()

    # meta's 'sum_factor' is scran-recomputed from realized counts (see
    # simulate_scenario.py), so guide identity is genuinely correlated with it (strong
    # cis perturbations shift library composition) -- follow the documented full
    # workflow (docs/FIT_TRANS_GUIDE.md) rather than feeding raw sum_factor straight
    # into fit_cis/fit_trans. Not persisted to disk by save_ntc_fit/save_cis_fit, so
    # this runs every time regardless of whether fit_ntc/fit_cis were loaded or fit.
    model.adjust_ntc_sum_factor(
        sum_factor_col_old='sum_factor',
        sum_factor_col_adj='sum_factor_adj',
        covariates=['cell_line'],
    )

    if resumed_cis:
        print(f"[RESUME] fit_cis already completed in a prior attempt "
              f"({prior_stats['steps']['fit_cis']['elapsed_sec']:.1f}s) — loading instead of refitting.")
        model.load_cis_fit()
    else:
        # force=True: fit_cis() refuses by default when the cis gene's fitted NTC log2
        # expression is < -1 (overdispersion from near-zero counts can be unreliable on
        # real data). This study's log2_X_NTC grid deliberately sweeps down to -1 (plan
        # §3.1), and sampling noise (sigma_eff=0.7) means the *fitted* value can land
        # below -1 by chance even when the true value sits right at the boundary --
        # that's exactly the low-expression stress-test scenario this study is designed
        # to include, not a real data-quality problem, so the safety check doesn't
        # apply here.
        with _timed_step('fit_cis', stats, device_is_cuda, stats_path):
            model.fit_cis(sum_factor_col='sum_factor_adj', force=True)
        model.save_cis_fit()

    model.refit_sumfactor(
        sum_factor_col_old='sum_factor_adj',
        sum_factor_col_refit='sum_factor_refit',
        covariates=['cell_line'],
    )

    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
    _fit_trans_t0 = time.perf_counter()
    model.fit_trans(
        sum_factor_col='sum_factor_refit',
        function_type='additive_hill',
    )
    _fit_trans_this_attempt_sec = time.perf_counter() - _fit_trans_t0

    # fit_trans's own checkpoint carries the TRUE cumulative wall-clock across every
    # resume attempt (bayesDREAM/fitting/trans.py writes 'cumulative_elapsed_sec' into
    # every checkpoint, seeded from whichever checkpoint it resumed from) -- read that
    # back rather than trusting this process's own timer, which can't see time spent
    # in earlier, separately-killed attempts. Falls back to this attempt's own timer
    # (undercounts prior attempts, but degrades gracefully) if the checkpoint is
    # missing the field for any reason (e.g. an older bayesDREAM version).
    _complete_ckpt_path = os.path.join(recovery_dir, 'trans_checkpoint_gene_complete.pt')
    _fit_trans_elapsed = _fit_trans_this_attempt_sec
    if os.path.exists(_complete_ckpt_path):
        try:
            _ckpt = torch.load(_complete_ckpt_path, map_location='cpu', weights_only=False)
            _fit_trans_elapsed = _ckpt.get('cumulative_elapsed_sec', _fit_trans_this_attempt_sec)
        except Exception as e:
            print(f"[WARNING] Could not read cumulative_elapsed_sec from {_complete_ckpt_path}: {e}. "
                  f"Recording this attempt's own elapsed time only ({_fit_trans_this_attempt_sec:.1f}s), "
                  f"which undercounts any prior killed attempts.")

    _this_attempt_peak_rss = _peak_rss_mb()
    _this_attempt_peak_gpu = (torch.cuda.max_memory_allocated() / 1024.0 ** 2) if device_is_cuda else None
    _prior_fit_trans_entry = (prior_stats or {}).get('steps', {}).get('fit_trans') or {}
    stats['steps']['fit_trans'] = {
        'elapsed_sec': _fit_trans_elapsed,
        # max, not sum: separate SLURM job attempts are different processes that were
        # never running simultaneously, so summing their peaks would overstate true
        # peak memory pressure at any single moment.
        'peak_rss_mb': _max_metric(_prior_fit_trans_entry.get('peak_rss_mb'), _this_attempt_peak_rss),
        'peak_gpu_mb': _max_metric(_prior_fit_trans_entry.get('peak_gpu_mb'), _this_attempt_peak_gpu),
    }
    _write_stats(stats, stats_path)
    gpu_note = (f", peak_gpu={stats['steps']['fit_trans']['peak_gpu_mb']:.0f}MB"
                if device_is_cuda else "")
    print(f"[TIMING] fit_trans: {_fit_trans_elapsed:.1f}s cumulative "
          f"({_fit_trans_this_attempt_sec:.1f}s this attempt), "
          f"peak_rss={stats['steps']['fit_trans']['peak_rss_mb']:.0f}MB{gpu_note}")

    model.save_trans_fit()
    model.save_trans_summary(fdr_threshold=0.05)

    stats['total_elapsed_sec'] = sum(s['elapsed_sec'] for s in stats['steps'].values())
    _write_stats(stats, stats_path)

    return model


if __name__ == '__main__':
    if args.scenario_dir is not None:
        scenario_dir = args.scenario_dir
    else:
        row_index = args.row_index
        if row_index is None:
            row_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        scenario_dir = resolve_scenario_dir(args.data_root, args.design_matrix, row_index)

    run_recovery_fit(scenario_dir=scenario_dir, device=args.device)
    print(f"Done: {scenario_dir}")
