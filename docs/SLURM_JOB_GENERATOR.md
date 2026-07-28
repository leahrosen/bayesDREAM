# SLURM Job Generator for HPC Clusters

Automated generation of optimized SLURM submission scripts for running bayesDREAM on HPC clusters, with specific support for Berzelius (NSC, Sweden).

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Low MOI vs High MOI](#low-moi-vs-high-moi)
  - [Resource Allocation](#resource-allocation)
  - [Time Estimation](#time-estimation)
- [Generated Scripts](#generated-scripts)
- [Submitting Jobs](#submitting-jobs)
- [Monitoring Jobs](#monitoring-jobs)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

The SLURM Job Generator (`bayesDREAM.slurm_jobgen`) automates the creation of SLURM submission scripts for large-scale bayesDREAM analyses. It:

1. **Analyzes your dataset** (features, cells, sparsity, technical groups)
2. **Estimates memory requirements** (RAM and VRAM) using `memory_calculator.py`
3. **Selects optimal resources** (GPU fat/thin nodes or CPU partition)
4. **Estimates wall-clock time** based on dataset size and complexity
5. **Generates SLURM scripts** with job dependencies and array parallelization
6. **Creates documentation** (README.md with monitoring commands)

### Why Use This?

**Without the generator:**
- Manually calculate memory needs for each step
- Guess appropriate time limits
- Write SLURM scripts from scratch
- Set up job dependencies manually
- Risk over-allocating (waste) or under-allocating (failure)

**With the generator:**
- One function call generates all scripts
- Automatic resource optimization
- Built-in job dependencies
- Job arrays for parallelization
- Throttling to prevent cluster overload

---

## Quick Start

### 1. Prepare Your Data

```python
import pandas as pd
from bayesDREAM.slurm_jobgen import SlurmJobGenerator

# Load data
meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)  # genes × cells

# Save to cluster-accessible location
data_path = "/proj/berzelius-aiics-real/users/x_learo/data/run1"
meta.to_csv(f"{data_path}/meta.csv", index=False)
counts.to_csv(f"{data_path}/counts.csv")
```

### 2. Generate SLURM Scripts

```python
# Create generator
gen = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2', 'MYB', 'NFE2'],
    output_dir='./slurm_jobs',
    label='perturb_seq_batch1',

    # Paths (adjust for your cluster)
    python_env='/proj/.../mambaforge/envs/pyroenv/bin/python',
    bayesdream_path='/proj/.../bayesDREAM',
    data_path=data_path,
)

# Generate all scripts
gen.generate_all_scripts()
```

### 3. Submit on Cluster

```bash
# Transfer to Berzelius
scp -r slurm_jobs/ berzelius:/proj/.../

# On Berzelius
cd slurm_jobs
bash submit_all.sh

# Monitor
squeue -u $USER
tail -f logs/tech_*.out
```

---

## Installation

The SLURM job generator is included in bayesDREAM. No additional installation required.

**Requirements:**
- bayesDREAM installed with dependencies
- Access to HPC cluster (Berzelius or similar SLURM-based system)
- Python environment on cluster with bayesDREAM

---

## Usage Guide

### Basic Usage

```python
from bayesDREAM.slurm_jobgen import SlurmJobGenerator

generator = SlurmJobGenerator(
    # Required
    meta=meta_dataframe,           # Cell metadata (cell, guide, target, cell_line)
    counts=counts_matrix,          # Gene expression (genes × cells)
    cis_genes=['GFI1B', 'TET2'],   # List of cis genes to fit
    output_dir='./slurm_jobs',     # Where to write scripts
    label='my_experiment',         # Unique identifier

    # Cluster paths
    python_env='/path/to/pyroenv/bin/python',
    bayesdream_path='/path/to/bayesDREAM',
    data_path='/path/to/data',     # Where meta.csv and counts.csv are saved

    # Optional (see sections below)
    low_moi=True,
    use_all_cells_technical=False,
    partition_preference='auto',
    max_concurrent_jobs=50,
    time_multiplier=1.0,
)

# Generate scripts
generator.generate_all_scripts()
```

### Low MOI vs High MOI

The generator supports different experimental designs:

#### Low MOI (default)

**One guide per cell, clear NTC population**

```python
gen = SlurmJobGenerator(
    ...,
    low_moi=True,  # Default
    use_all_cells_technical=False,
)
```

**Job structure:**
- `fit_ntc`: 1 job (all NTC cells)
- `fit_cis`: N jobs (1 per cis gene)
- `fit_trans`: N jobs (1 per cis gene)

**Total jobs:** 1 + 2N

---

#### High MOI - All Cells Mode

**Multiple guides per cell, technical effects independent of perturbations**

```python
gen = SlurmJobGenerator(
    ...,
    low_moi=False,
    use_all_cells_technical=True,  # Fit technical on ALL cells
)
```

**Job structure:**
- `fit_ntc`: 1 job (all cells, once for all cis genes)
- `fit_cis`: N jobs (1 per cis gene)
- `fit_trans`: N jobs (1 per cis gene)

**Total jobs:** 1 + 2N

**Benefit:** Same total jobs as low MOI, but `fit_ntc` uses all data.

**When to use:**
- Technical variation is batch/lane specific
- Technical effects don't correlate with perturbation effects
- Want to maximize statistical power for technical correction

**When NOT to use:**
- Technical groups (e.g., CRISPRi vs CRISPRa) correlate with cis gene expression
- See warning in `fit_ntc` documentation

---

#### High MOI - NTC Per Gene Mode

**Multiple guides per cell, but want NTC-based technical correction per gene**

```python
gen = SlurmJobGenerator(
    ...,
    low_moi=False,
    use_all_cells_technical=False,  # Fit technical on NTC per gene
)
```

**Job structure:**
- `fit_ntc`: N jobs (NTC cells, 1 per cis gene)
- `fit_cis`: N jobs (1 per cis gene)
- `fit_trans`: N jobs (1 per cis gene)

**Total jobs:** 3N

**When to use:**
- High MOI but want conservative NTC-only technical correction
- Technical effects may vary by cis gene

---

### Resource Allocation

The generator automatically selects optimal resources for Berzelius:

#### Berzelius Node Types (Ampere/A100 — Hopper/H200 not covered here)

| Node Type | Constraint | GPUs per Node | VRAM per GPU | RAM per GPU | RAM per Node | Best For |
|-----------|------------|---------------|--------------|-------------|--------------|----------|
| **Thin** | `-C thin` | 8 | 40 GB (A100) | 128 GB | 1 TB | Standard fitting, small-medium datasets |
| **Fat** | `-C fat` | 8 | 80 GB (A100) | 256 GB | 2 TB | Large datasets |
| **CPU** | `--partition=berzelius-cpu` | 0 | 0 | 7.76 GB/core | — | Extremely large datasets, fit_cis |

**Note:** Full node = 8 GPUs. RAM per GPU is the node's total RAM divided by 8 (SLURM allocates RAM proportional to GPUs requested, per the node's 1:8 GPU:CPU-core ratio). VRAM is *not* shared across GPUs — each GPU carries its own dedicated 40GB (thin) or 80GB (fat), regardless of how many GPUs are requested. The generator will use up to 8 GPUs before switching to CPU.

#### Automatic Selection Logic

The generator analyzes memory requirements and selects:

**fit_ntc and fit_trans** (same logic):
```
If VRAM ≤ 40 GB and RAM ≤ 128 GB:
    → 1 thin GPU ✓ (most efficient for small datasets)
Elif VRAM ≤ 80 GB and RAM ≤ 256 GB:
    → 1 fat GPU ✓ (most common case)
Elif VRAM ≤ 160 GB and RAM ≤ 512 GB:
    → 2 fat GPUs
Elif VRAM ≤ 320 GB and RAM ≤ 1024 GB:
    → 4 fat GPUs
Elif VRAM ≤ 640 GB and RAM ≤ 2048 GB:
    → 8 fat GPUs (full node)
Else:
    → CPU partition (dataset too large for 8 GPUs)
```

**fit_cis:**
```
Default: CPU partition (doesn't need GPU for most datasets)
```

**Goal:** Use thin nodes for small datasets, scale up to full 8-GPU node before falling back to CPU.

#### Manual Override

Force specific resources:

```python
# Force CPU for all steps
gen = SlurmJobGenerator(
    ...,
    partition_preference='cpu',
)

# Force fat nodes
gen = SlurmJobGenerator(
    ...,
    partition_preference='fat',
)

# Auto-select (default, recommended)
gen = SlurmJobGenerator(
    ...,
    partition_preference='auto',
)
```

---

### Time Estimation

The generator automatically estimates wall-clock time based on:

1. **Dataset size** (T × N)
2. **Guide type** (AutoIAFNormal vs AutoNormal)
3. **Step complexity**

#### Base Estimates (20K genes, 30K cells)

| Step | Base Time | Scales With |
|------|-----------|-------------|
| `fit_ntc` | 1.5 hours | T × N × (2× if AutoNormal) |
| `fit_cis` | 0.5 hours/gene | N |
| `fit_trans` | 3.0 hours/gene | T × N |

#### Safety Margin

All estimates include 1.5× safety margin to reduce timeout risk.

#### Scaling Examples

**Small dataset (5K genes, 10K cells):**
- fit_ntc: ~0.5 hours
- fit_cis: ~0.3 hours/gene
- fit_trans: ~0.8 hours/gene

**Large dataset (50K genes, 100K cells):**
- fit_ntc: ~9 hours (if AutoNormal)
- fit_cis: ~1.5 hours/gene
- fit_trans: ~20 hours/gene

#### User Override

Scale time estimates:

```python
# Conservative (2× longer)
gen = SlurmJobGenerator(
    ...,
    time_multiplier=2.0,
)

# Aggressive (if you know your data converges fast)
gen = SlurmJobGenerator(
    ...,
    time_multiplier=0.7,
)

# Default (recommended)
gen = SlurmJobGenerator(
    ...,
    time_multiplier=1.0,
)
```

#### Why Automatic?

Dataset size matters more than user intuition. A 50K gene dataset takes 10× longer than 5K genes. The generator's estimates are based on empirical scaling laws.

---

## Generated Scripts

### Script Overview

```
slurm_jobs/
├── 01_fit_ntc.sh            # NTC fitting
├── 02_fit_cis.sh             # Cis effect fitting (job array)
├── 03_fit_trans.sh           # Trans effect fitting (job array)
├── submit_all.sh             # Master submission script
├── README.md                 # Complete documentation
└── logs/                     # Created automatically
    ├── tech_*.out
    ├── cis_*_*.out
    └── trans_*_*.out
```

### 01_fit_ntc.sh

**Single job** (low MOI or high MOI with `use_all_cells=True`):
```bash
#!/bin/bash
#SBATCH --job-name=perturb_seq_batch1_tech
#SBATCH --time=02:15:00
#SBATCH --partition=berzelius
#SBATCH -C fat
#SBATCH --gpus=1
#SBATCH --mem=11G

# Run fit_ntc on NTC cells (or all cells if high MOI)
```

**Job array** (high MOI with NTC per gene):
```bash
#SBATCH --array=0-3%50  # 4 cis genes, max 50 concurrent
```

### 02_fit_cis.sh

**Job array** (1 job per cis gene):
```bash
#!/bin/bash
#SBATCH --job-name=perturb_seq_batch1_cis
#SBATCH --array=0-3%50  # 4 cis genes
#SBATCH --time=00:30:00
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=9G

# Array: CIS_GENES=(GFI1B TET2 MYB NFE2)
# CIS_GENE=${CIS_GENES[$SLURM_ARRAY_TASK_ID]}
```

### 03_fit_trans.sh

**Job array** (1 job per cis gene):
```bash
#!/bin/bash
#SBATCH --job-name=perturb_seq_batch1_trans
#SBATCH --array=0-3%50  # 4 cis genes
#SBATCH --time=03:00:00
#SBATCH --partition=berzelius
#SBATCH -C fat
#SBATCH --gpus=1
#SBATCH --mem=18G
```

### submit_all.sh

**Master script with dependencies:**
```bash
#!/bin/bash
# Submit fit_ntc
TECH_JOB=$(sbatch --parsable 01_fit_ntc.sh)

# Submit fit_cis (depends on technical)
CIS_JOB=$(sbatch --parsable --dependency=afterok:$TECH_JOB 02_fit_cis.sh)

# Submit fit_trans (depends on cis)
TRANS_JOB=$(sbatch --parsable --dependency=afterok:$CIS_JOB 03_fit_trans.sh)

echo "Jobs submitted: $TECH_JOB, $CIS_JOB, $TRANS_JOB"
```

### README.md

Auto-generated documentation includes:
- Dataset characteristics
- Memory and time estimates
- Resource allocation rationale
- **Complete Python code** that will be executed for each step
- **Expected log output** with examples
- Usage instructions
- Monitoring commands
- Troubleshooting guide

**Key sections:**
- **"What Will Be Run"**: Shows exact Python code for fit_ntc, fit_cis, and fit_trans with all parameters
- **"Log Output"**: Shows what you'll see in logs/tech_*.out, logs/cis_*.out, and logs/trans_*.out
- Includes device selection (cuda/cpu), niters, nsamples, and all other parameters

---

## Submitting Jobs

### Method 1: Automatic (Recommended)

```bash
cd slurm_jobs
bash submit_all.sh
```

**What happens:**
1. Submits `fit_ntc`
2. Queues `fit_cis` with dependency on technical
3. Queues `fit_trans` with dependency on cis
4. Prints job IDs

**Output:**
```
Submitting bayesDREAM pipeline jobs...
Label: perturb_seq_batch1
Output directory: ./slurm_jobs

Submitting fit_ntc...
  Job ID: 12345
Submitting fit_cis (depends on 12345)...
  Job ID: 12346
Submitting fit_trans (depends on 12346)...
  Job ID: 12347

All jobs submitted successfully!
```

### Method 2: Manual

Submit jobs one at a time:

```bash
# 1. Technical
sbatch 01_fit_ntc.sh
# Note job ID (e.g., 12345)

# 2. Cis (after technical completes)
sbatch --dependency=afterok:12345 02_fit_cis.sh
# Note job ID (e.g., 12346)

# 3. Trans (after cis completes)
sbatch --dependency=afterok:12346 03_fit_trans.sh
```

**When to use:**
- Testing individual steps
- Re-running specific steps
- Custom dependency logic

---

## Monitoring Jobs

### Check Job Status

```bash
# Your jobs
squeue -u $USER

# Specific job
squeue -j 12345

# All jobs for this experiment
squeue --name=perturb_seq_batch1_tech
squeue --name=perturb_seq_batch1_cis
squeue --name=perturb_seq_batch1_trans
```

**Output:**
```
JOBID   PARTITION  NAME                    USER   ST  TIME  NODES
12345   berzelius  perturb_seq_batch1_tech learo  R   0:45  1
12346   berzelius  perturb_seq_batch1_cis  learo  PD  0:00  -  (Dependency)
12347   berzelius  perturb_seq_batch1_trans learo PD 0:00  -  (Dependency)
```

**Job states:**
- `PD`: Pending (waiting for resources or dependency)
- `R`: Running
- `CG`: Completing
- `CD`: Completed
- `F`: Failed

### View Job Details

```bash
# Detailed info
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize

# Efficiency report
seff 12345
```

### View Logs in Real-Time

```bash
# Technical fit
tail -f logs/tech_12345.out

# Cis fit (array job)
tail -f logs/cis_12346_0.out  # First array task
tail -f logs/cis_12346_*.out  # All tasks (messy)

# Trans fit
tail -f logs/trans_12347_0.out
```

### Check for Errors

```bash
# Find failed jobs
sacct -S 2025-01-17 -u $USER | grep FAILED

# Check error logs
grep -i error logs/*.err
grep -i "out of memory" logs/*.err
```

### Cancel Jobs

```bash
# Cancel specific job
scancel 12345

# Cancel all jobs for experiment
scancel --name=perturb_seq_batch1_tech
scancel --name=perturb_seq_batch1_cis
scancel --name=perturb_seq_batch1_trans

# Cancel all your jobs
scancel -u $USER
```

---

## Advanced Usage

### Job Array Throttling

Control maximum concurrent jobs:

```python
gen = SlurmJobGenerator(
    ...,
    max_concurrent_jobs=20,  # Limit to 20 at a time
)
```

**Generated SLURM directive:**
```bash
#SBATCH --array=0-99%20  # 100 jobs, max 20 concurrent
```

**When to use:**
- Cluster has queue limits
- Being considerate to other users
- Testing on subset before full run

### Multiple Experiments

Generate scripts for different configurations:

```python
# Experiment 1: Full dataset
gen1 = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2', 'MYB', 'NFE2'],
    output_dir='./slurm_jobs_full',
    label='full_dataset',
)
gen1.generate_all_scripts()

# Experiment 2: Test on 2 genes
gen2 = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2'],  # Subset
    output_dir='./slurm_jobs_test',
    label='test_run',
    time_multiplier=0.5,  # Faster for testing
)
gen2.generate_all_scripts()
```

### Custom Python Environment

Specify custom environment:

```python
gen = SlurmJobGenerator(
    ...,
    python_env='/proj/.../custom_env/bin/python',
    bayesdream_path='/proj/.../custom_bayesdream',
)
```

### Sparse Data Optimization

For very sparse data (>90% zeros), provide hint:

```python
gen = SlurmJobGenerator(
    ...,
    sparsity=0.95,  # 95% zeros
)
```

Generator will use this for accurate memory estimation.

### Different Distributions

For non-negbinom distributions:

```python
# Multinomial (splicing)
gen = SlurmJobGenerator(
    ...,
    distribution='multinomial',
)

# Binomial (exon skipping)
gen = SlurmJobGenerator(
    ...,
    distribution='binomial',
)
```

---

## Troubleshooting

### Job Fails: "Out of Memory"

**Symptoms:**
```
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=12345
Job killed due to memory usage
```

**Solutions:**

1. **Check actual memory usage:**
   ```bash
   seff 12345  # Shows peak memory
   ```

2. **Increase memory request:**
   - Edit SLURM script: `#SBATCH --mem=20G` (was 11G)
   - Or regenerate with `partition_preference='fat'` for more RAM

3. **Use more GPUs:**
   ```python
   # Each GPU on fat nodes provides 256GB RAM
   # 2 GPUs = 512GB RAM
   ```

4. **Switch to CPU:**
   ```python
   gen = SlurmJobGenerator(
       ...,
       partition_preference='cpu',
   )
   ```

5. **Reduce dataset size:**
   - Filter low-count features
   - Subset cells for testing

### Job Times Out

**Symptoms:**
```
Job reached time limit and was terminated
State: TIMEOUT
```

**Solutions:**

1. **Check actual runtime:**
   ```bash
   sacct -j 12345 --format=JobID,Elapsed,State
   ```

2. **Increase time limit:**
   - Edit SLURM script: `#SBATCH --time=06:00:00`
   - Or regenerate with `time_multiplier=2.0`

3. **Check convergence:**
   - View logs to see if fitting was still making progress
   - May need more iterations (edit Python code in script)

### Job Stuck in Queue (PD)

**Check reason:**
```bash
squeue -j 12345 -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.20R"
```

**Common reasons:**
- `(Dependency)`: Waiting for parent job → Normal
- `(Resources)`: Waiting for nodes → Be patient
- `(Priority)`: Other jobs have higher priority → Be patient
- `(QOSMaxGpuLimit)`: Too many GPUs requested → Reduce

**Solutions:**
- Wait (usually just need patience)
- Reduce resource requests
- Check cluster status: `sinfo`

### Job Fails: "File Not Found"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/proj/.../data/meta.csv'
```

**Solutions:**

1. **Verify data path:**
   ```bash
   ls /proj/.../data/meta.csv
   ls /proj/.../data/counts.csv
   ```

2. **Check permissions:**
   ```bash
   ls -lh /proj/.../data/
   ```

3. **Regenerate scripts with correct path:**
   ```python
   gen = SlurmJobGenerator(
       ...,
       data_path='/correct/path/to/data',
   )
   ```

### Dependencies Not Working

**Symptoms:**
- Cis jobs start before technical completes
- Trans jobs start before cis completes

**Solutions:**

1. **Check SLURM version supports `--parsable`:**
   ```bash
   sbatch --help | grep parsable
   ```

2. **Manual submission with explicit IDs:**
   ```bash
   TECH_JOB=$(sbatch --parsable 01_fit_ntc.sh)
   echo $TECH_JOB  # Should print job ID
   sbatch --dependency=afterok:$TECH_JOB 02_fit_cis.sh
   ```

3. **Check dependency status:**
   ```bash
   squeue -j 12346 -o "%.18i %.30E"  # Shows dependency
   ```

### AutoNormal vs AutoIAFNormal

**Question:** How do I know which guide is being used?

**Answer:** Check the generator output:

```
[INFO] AutoIAFNormal estimated at 2.5 GB VRAM
[INFO] Will use AutoIAFNormal with niters=50,000
```

or

```
[INFO] AutoIAFNormal would require 146.0 GB VRAM
[INFO] Will use AutoNormal (mean-field) with niters=100,000
```

**Threshold:** 20 GB VRAM
- If IAF < 20 GB → AutoIAFNormal (faster convergence)
- If IAF ≥ 20 GB → AutoNormal (memory-efficient, needs more iterations)

### Job Array Task Failures

**Symptoms:**
```
Array job 12346 has 1/4 tasks failed
```

**Find which task failed:**
```bash
sacct -j 12346 --format=JobID,State | grep FAILED
```

**Output:**
```
12346_2    FAILED
```

**Check logs for that task:**
```bash
cat logs/cis_12346_2.err
cat logs/cis_12346_2.out
```

**Rerun specific task:**
```bash
# Get cis gene for task 2
CIS_GENES=(GFI1B TET2 MYB NFE2)
echo ${CIS_GENES[2]}  # MYB

# Edit script to run just that gene, or resubmit:
sbatch --array=2 02_fit_cis.sh  # Just task 2
```

---

## API Reference

### SlurmJobGenerator Class

```python
class SlurmJobGenerator:
    """
    Generate SLURM job submission scripts for bayesDREAM pipeline on Berzelius.
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts,  # pd.DataFrame or sparse matrix
        gene_meta: Optional[pd.DataFrame] = None,
        cis_genes: Optional[List[str]] = None,
        output_dir: str = './slurm_jobs',
        label: str = 'bayesdream_run',
        low_moi: bool = True,
        use_all_cells_technical: bool = False,
        distribution: str = 'negbinom',
        sparsity: Optional[float] = None,
        n_groups: Optional[int] = None,
        max_concurrent_jobs: int = 50,
        time_multiplier: float = 1.0,
        partition_preference: str = 'auto',
        python_env: str = '/proj/.../pyroenv/bin/python',
        bayesdream_path: str = '/proj/.../bayesDREAM',
        data_path: Optional[str] = None,
        nsamples: int = 1000,
    ):
```

#### Parameters

**Required:**
- `meta` (pd.DataFrame): Cell metadata with columns: `cell`, `guide`, `target`, `cell_line`
- `counts` (pd.DataFrame or sparse): Gene expression counts (features × cells)
- `cis_genes` (list of str): List of cis genes to fit

**Output:**
- `output_dir` (str): Directory to write SLURM scripts (default: `'./slurm_jobs'`)
- `label` (str): Unique identifier for this run (default: `'bayesdream_run'`)

**Experiment Design:**
- `low_moi` (bool): Low MOI mode vs high MOI (default: `True`)
- `use_all_cells_technical` (bool): Use all cells for technical fit (default: `False`)

**Data Characteristics:**
- `distribution` (str): `'negbinom'`, `'multinomial'`, `'binomial'`, `'normal'`, `'studentt'` (default: `'negbinom'`)
- `sparsity` (float): Fraction of zeros (default: auto-detect)
- `n_groups` (int): Number of technical groups (default: auto-detect)
- `gene_meta` (pd.DataFrame): Gene metadata (optional)

**Resource Allocation:**
- `partition_preference` (str): `'auto'`, `'fat'`, `'thin'`, or `'cpu'` (default: `'auto'`)
- `max_concurrent_jobs` (int): Max concurrent array jobs (default: `50`)
- `time_multiplier` (float): Scale time estimates (default: `1.0`)

**Cluster Configuration:**
- `python_env` (str): Path to Python executable with bayesDREAM
- `bayesdream_path` (str): Path to bayesDREAM repository
- `data_path` (str): Path to saved data (meta.csv, counts.csv)

**Fitting Parameters:**
- `nsamples` (int): Number of posterior samples (default: `1000`)

#### Methods

**`generate_all_scripts(cis_genes: Optional[List[str]] = None)`**

Generate all SLURM scripts.

```python
gen.generate_all_scripts()
```

**`estimate_memory_requirements() -> Dict[str, float]`**

Estimate RAM and VRAM for each step.

```python
memory = gen.estimate_memory_requirements()
print(f"Technical: {memory['fit_ntc_ram_gb']:.1f} GB RAM")
```

**Returns:**
```python
{
    'fit_ntc_ram_gb': 5.7,
    'fit_ntc_vram_gb': 7.1,
    'fit_cis_ram_gb': 4.3,
    'fit_cis_vram_gb': 4.2,
    'fit_trans_ram_gb': 12.6,
    'fit_trans_vram_gb': 9.1,
    'min_ram_gb': 12.6,
    'min_vram_gb': 9.1,
    'recommended_ram_gb': 19.0,
    'recommended_vram_gb': 16.0,
    'resources': {...}
}
```

**`estimate_time_requirements(memory: Dict) -> Dict[str, str]`**

Estimate wall-clock time.

```python
times = gen.estimate_time_requirements(memory)
print(f"Technical: {times['fit_ntc']}")  # "02:15:00"
```

---

## Examples

### Example 1: Standard Low MOI

```python
from bayesDREAM.slurm_jobgen import SlurmJobGenerator
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
counts = pd.read_csv('counts.csv', index_col=0)

# Save to cluster
data_path = "/proj/.../data/run1"
meta.to_csv(f"{data_path}/meta.csv", index=False)
counts.to_csv(f"{data_path}/counts.csv")

# Generate scripts
gen = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2', 'MYB', 'NFE2'],
    output_dir='./slurm_jobs',
    label='perturb_seq_lowmoi',
    low_moi=True,
    python_env='/proj/.../pyroenv/bin/python',
    bayesdream_path='/proj/.../bayesDREAM',
    data_path=data_path,
)

gen.generate_all_scripts()
```

### Example 2: High MOI with All Cells

```python
gen = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2'],
    output_dir='./slurm_jobs_highmoi',
    label='perturb_seq_highmoi',
    low_moi=False,
    use_all_cells_technical=True,  # Key difference
    python_env='/proj/.../pyroenv/bin/python',
    bayesdream_path='/proj/.../bayesDREAM',
    data_path=data_path,
)

gen.generate_all_scripts()
```

### Example 3: CPU-Only (Very Large Dataset)

```python
gen = SlurmJobGenerator(
    meta=meta,
    counts=counts_large,  # 100K genes × 200K cells
    cis_genes=['GFI1B'],
    output_dir='./slurm_jobs_cpu',
    label='perturb_seq_large',
    partition_preference='cpu',  # Force CPU
    time_multiplier=3.0,  # CPU is slower
    python_env='/proj/.../pyroenv/bin/python',
    bayesdream_path='/proj/.../bayesDREAM',
    data_path=data_path,
)

gen.generate_all_scripts()
```

### Example 4: Testing with Subset

```python
# Test on 2 genes with conservative settings
gen = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=['GFI1B', 'TET2'],  # Just 2 genes
    output_dir='./slurm_jobs_test',
    label='test_run',
    time_multiplier=2.0,  # Extra time for safety
    max_concurrent_jobs=2,  # Don't overwhelm cluster
    python_env='/proj/.../pyroenv/bin/python',
    bayesdream_path='/proj/.../bayesDREAM',
    data_path=data_path,
)

gen.generate_all_scripts()
```

---

## Notes for Other Clusters

While this generator is optimized for Berzelius, it can be adapted for other SLURM clusters:

### Required Changes

1. **Update resource specs** in `_recommend_resources()`:
   ```python
   # Example for different cluster
   # - GPU nodes: 40GB VRAM, 256GB RAM
   # - CPU nodes: 8GB RAM per core
   ```

2. **Update module loading** in generated scripts:
   ```bash
   # Replace:
   module load Anaconda/2021.05-nsc1

   # With your cluster's modules:
   module load python/3.9
   module load cuda/11.8
   ```

3. **Update partition names**:
   ```python
   # Change partition names to match your cluster
   'partition': 'gpu',  # Instead of 'berzelius'
   'constraint': 'v100',  # Instead of 'fat'
   ```

### Contact

For questions about:
- **bayesDREAM code**: See repository documentation
- **Berzelius cluster**: Contact NSC support
- **SLURM issues**: Consult your cluster's documentation

---

## See Also

- [Memory Requirements Guide](MEMORY_REQUIREMENTS.md)
- [Memory Calculator](memory_calculator.py)
- [bayesDREAM Documentation](../README.md)
- [Example Script](../examples/generate_slurm_jobs.py)
