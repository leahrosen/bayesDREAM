"""Memory planning helpers for NTC posterior prediction."""

import multiprocessing
import os

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


def estimate_predictive_memory_and_set_minibatch(
    N, T, C, nsamples, minibatch_size=None,
    distribution='negbinom',
    safety_factor=None,
):
    """
    Estimate memory requirements for Predictive and auto-set minibatch_size.

    Parameters
    ----------
    N : int
        Number of observations (cells)
    T : int
        Number of features
    C : int
        Number of technical groups
    nsamples : int
        Number of posterior samples requested
    minibatch_size : int or None
        If provided, use this value. If None, auto-compute.
    distribution : str
        Distribution type
    safety_factor : float or None
        Fraction of available RAM to use. If None (default), auto-determined:
        - 0.7 (70%) when SLURM allocation detected (dedicated resources)
        - 0.35 (35%) on shared nodes without SLURM (conservative)

    Returns
    -------
    minibatch_size : int or None
        Recommended minibatch size, or None if full batch fits
    use_parallel : bool
        Whether to use parallel=True in Predictive
    """
    if minibatch_size is not None:
        print(f"[MEMORY] Using user-specified minibatch_size={minibatch_size}")
        return minibatch_size, True

    if psutil is None:
        print("[INFO] psutil not available - using conservative defaults (minibatch_size=50)")
        return 50, False

    available_gb = psutil.virtual_memory().available / (1024**3)
    slurm_mem_gb = None
    if 'SLURM_MEM_PER_NODE' in os.environ:
        try:
            slurm_mem_gb = int(os.environ['SLURM_MEM_PER_NODE']) / 1024
        except (ValueError, KeyError):
            pass

    if slurm_mem_gb is None and 'SLURM_MEM_PER_CPU' in os.environ:
        try:
            mem_per_cpu_mb = int(os.environ['SLURM_MEM_PER_CPU'])
            n_cpus = next((int(os.environ[key]) for key in (
                'SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE'
            ) if key in os.environ), None)
            if n_cpus is not None:
                slurm_mem_gb = mem_per_cpu_mb * n_cpus / 1024
        except (ValueError, KeyError):
            pass

    if slurm_mem_gb is not None:
        usable_gb = slurm_mem_gb
        safety_factor = 0.7 if safety_factor is None else safety_factor
    else:
        usable_gb = available_gb
        safety_factor = 0.35 if safety_factor is None else safety_factor

    params_per_sample_gb = (2 * C * T + 3 * T) * 4 / (1024**3)
    input_data_gb = (N * T + N) * 4 / (1024**3)
    # Four full-size temporary tensors can coexist during the model forward pass.
    memory_per_sample_gb = params_per_sample_gb + 4 * (N * T) * 4 / (1024**3)

    n_workers = min(multiprocessing.cpu_count(), 32)
    total_parallel_gb = n_workers * (input_data_gb + params_per_sample_gb * 10)
    use_parallel = total_parallel_gb <= usable_gb * safety_factor
    available_for_sampling_gb = usable_gb * safety_factor - input_data_gb
    max_samples_at_once = int(available_for_sampling_gb / memory_per_sample_gb)

    if max_samples_at_once >= nsamples:
        return None, use_parallel

    recommended_batch = (
        max(10, min(100, max_samples_at_once))
        if max_samples_at_once >= 10 else max(1, max_samples_at_once)
    )
    n_batches = int(np.ceil(nsamples / recommended_batch))
    print(f"[INFO] Minibatching: {recommended_batch} samples/batch × {n_batches} batches (parallel=False)")
    return recommended_batch, False
