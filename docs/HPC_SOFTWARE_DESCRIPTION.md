# Software Description for HPC Allocation

## Software

The analyses will use **bayesDREAM** (custom open-source Python package, not centre-provided), a Bayesian framework for modelling perturbation effects in single-cell CRISPR screens. bayesDREAM is built on **PyTorch** (≥2.2) for tensor computation and automatic differentiation, and **Pyro** (≥1.9, probabilistic programming library) for specifying and fitting hierarchical probabilistic models via stochastic variational inference (SVI). Supporting libraries include NumPy, SciPy, pandas, and statsmodels. Data preprocessing uses the Bioconductor package **scran** (R) for sum-factor normalisation.

## Parallelisation Strategy

The pipeline is embarrassingly parallel at the level of perturbation targets (cis genes). It consists of three sequential steps per cis gene — technical/NTC fitting (`fit_ntc`), cis-effect fitting (`fit_cis`), and trans-effect fitting (`fit_trans`) — where each step depends on the output of the previous one, but all cis genes are independent of one another. Jobs are submitted as **SLURM job arrays**, with one array task per cis gene for the cis and trans steps, and with configurable throttling (typically 50 concurrent tasks) to avoid saturating the queue. Job dependencies are encoded via `--dependency=afterok` so steps chain automatically. An automated SLURM script generator (`bayesDREAM.slurm_jobgen`) analyses dataset characteristics to select appropriate resources (GPU fat/thin node or CPU partition), estimate wall-clock time, and write all scripts from a single function call.

Within each job, computation runs on a single GPU via PyTorch CUDA. Gradient computation, tensor operations, and SVI optimisation (using PyTorch's Adam or Pyro's ClippedAdam) all execute on the GPU. Posterior sampling uses PyTorch's parallelised tensor operations. For CPU fallback runs, PyTorch uses multi-threaded BLAS (OpenBLAS or MKL), with thread count set explicitly via `torch.set_num_threads()` to match the SLURM `--cpus-per-task` allocation.

## Scaling Characteristics

Memory and runtime scale primarily with the number of genes *T* and cells *N*. The `fit_ntc` and `fit_trans` steps are the most demanding:

| Dataset size | Step | RAM | VRAM |
|---|---|---|---|
| 30K genes × 50K cells | `fit_ntc` | 8–14 GB | 8–14 GB |
| 30K genes × 50K cells | `fit_cis` | 5–8 GB | 5–8 GB |
| 30K genes × 50K cells | `fit_trans` | 16–28 GB | 14–24 GB |
| 50K genes × 100K cells | `fit_ntc` | 12–20 GB | 12–20 GB |
| 50K genes × 100K cells | `fit_trans` | 32–50 GB | 28–40 GB |

`fit_cis` is much lighter (4–8 GB RAM/VRAM) and typically runs on CPU.

The framework incorporates automatic memory management: it estimates whether the more accurate AutoIAFNormal variational guide fits in GPU memory and falls back to the memory-efficient mean-field AutoNormal guide for large datasets. Sparse matrix input (CSR format) reduces data memory 5–7× at typical scRNA-seq sparsity (85–90% zeros). Runtime scales approximately linearly with *T* × *N* and with the number of SVI iterations (default 50,000–200,000 depending on model complexity).

### Representative Wall-Clock Estimates

For a full analysis of 4 cis genes at medium scale (20,000 genes, 30,000 cells):

| Step | Time |
|---|---|
| `fit_ntc` (once) | ~1.5 hours |
| `fit_cis` (per gene) | ~0.5 hours |
| `fit_trans` (per gene) | ~3.0 hours |

All cis and trans jobs run concurrently as array tasks, so total wall time is dominated by the slowest single job rather than the sum.
