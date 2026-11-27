# bayesDREAM Installation Guide

This guide provides instructions for installing bayesDREAM on different computing environments, including HPC clusters.

## Quick Start (Recommended)

### Choose Your Environment

bayesDREAM provides architecture-specific conda environments:

| Environment File | Hardware | Use Case |
|-----------------|----------|----------|
| `environment_cpu.yml` | CPU-only | Local dev, login nodes, M1/M2 Mac |
| `environment_cuda.yml` | NVIDIA GPU | CUDA 12.1 (most GPU clusters) |
| `environment_cuda_aarch64.yml` | NVIDIA GPU, ARM CPU | Grace Hopper (GH200) / other aarch64 GPU nodes |
| `environment_rocm.yml` | AMD GPU | ROCm 6.0 (AMD GPU clusters) |

**Quick reference:**
- **NVIDIA GPU cluster** → `environment_cuda.yml`
- **NVIDIA GPU cluster with ARM nodes** (`uname -m` prints `aarch64`, e.g. GH200) → `environment_cuda_aarch64.yml` (see [ARM (aarch64) GPU nodes](#arm-aarch64-gpu-nodes-eg-grace-hopper))
- **AMD GPU cluster** → `environment_rocm.yml`
- **No GPU / Local Mac** → `environment_cpu.yml`

### Using Conda (Recommended for most users)

**⚠️ Important:** If you have an **AMD GPU cluster**, skip this Quick Start and go directly to the [AMD GPUs (ROCm)](#amd-gpus-rocm) section below, as it requires additional pip installation steps.

#### For CPU-only or NVIDIA GPU:

1. **Create the conda environment** (choose appropriate file):
   ```bash
   # For NVIDIA GPUs
   conda env create -f environment_cuda.yml

   # For CPU-only
   conda env create -f environment_cpu.yml
   ```

2. **Activate the environment:**
   ```bash
   # Adjust name based on which file you used
   conda activate bayesdream_cuda    # or bayesdream_cpu
   ```

3. **Install bayesDREAM in development mode:**
   ```bash
   pip install -e .
   ```

4. **Register R kernel with Jupyter (for using R in notebooks):**
   ```bash
   R -e "IRkernel::installspec(user = FALSE)"
   ```

5. **Verify GPU support (if using GPU cluster):**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Using pip only

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install bayesDREAM:**
   ```bash
   pip install -e .
   ```

## GPU Support

### NVIDIA GPUs (CUDA)

**Using `environment_cuda.yml`** (recommended):
- Default: CUDA 12.1 (works with CUDA 12.x drivers)
- For CUDA 11.8: Edit file and change `pytorch-cuda=12.1` to `pytorch-cuda=11.8`

**Check your CUDA version:**
```bash
nvidia-smi  # Shows CUDA driver version (e.g., 12.2)
nvcc --version  # Shows CUDA toolkit version (if installed)
```

**Verify GPU after installation:**
```bash
conda activate bayesdream_cuda
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

You should see:
```
CUDA available: True
CUDA version: 12.1
```

### ARM (aarch64) GPU nodes (e.g. Grace Hopper)

Some clusters (e.g. NAISS Arrhenius) have x86_64 login nodes but ARM (`aarch64`) GPU
nodes. Check with `uname -m` on both. If they differ:

- **A conda installed on the x86 login node cannot run on the aarch64 GPU node.** A
  typical symptom is `conda` failing with `import: unable to open X server` followed by a
  bash syntax error on `sys.argv`: the x86 Python can't execute, so bash falls back to
  interpreting the `conda` Python script as a shell script.
- `environment_cuda.yml` won't solve on aarch64 either: the `pytorch`/`nvidia` channels
  have no aarch64 CUDA builds.

Install a second, aarch64 conda alongside the x86 one, **from the GPU node**:

```bash
# On an aarch64 GPU node (e.g. inside salloc)
cd /path/to/software
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $PWD/miniforge3_aarch64
source miniforge3_aarch64/etc/profile.d/conda.sh

cd /path/to/bayesDREAM_forClaude
conda env create -f environment_cuda_aarch64.yml
conda activate bayesdream_cuda
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Tested on NAISS Arrhenius GH200 nodes (2026-11): PyTorch 2.5.1 (CUDA 12.6, from
conda-forge), Pyro 1.9.2 and scran 1.40.0 (bioconda linux-aarch64) all install and run.

To have `~/.bashrc` pick the right conda per node, wrap the `conda init` block in a
check on `uname -m`:

```bash
case "$(uname -m)" in
    aarch64) __conda_root=/path/to/software/miniforge3_aarch64 ;;
    *)       __conda_root=/path/to/software/miniconda3 ;;
esac
__conda_setup="$("$__conda_root/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
elif [ -f "$__conda_root/etc/profile.d/conda.sh" ]; then
    . "$__conda_root/etc/profile.d/conda.sh"
else
    export PATH="$__conda_root/bin:$PATH"
fi
unset __conda_setup __conda_root
```

This replaces the `# >>> conda initialize >>>` ... `# <<< conda initialize <<<` block.
Leave those markers out: `conda init` rewrites whatever sits between them.

Optionally, `export CONDA_PKGS_DIRS=...` inside the `aarch64)` branch to keep a
separate package cache for the ARM install. Without it, conda caches packages in
`<install>/pkgs`.

**Git on compute nodes:** if you authenticate to GitHub through a forwarded SSH agent
(`ForwardAgent yes` on your laptop), the agent reaches the login node but not a compute
node entered via `salloc` + `jobsh`/`srun`. There `git fetch` fails with
`Permission denied (publickey)`. Run git on the login node (same filesystem), and the
install/fitting on the compute node.

### AMD GPUs (ROCm)

**Using `environment_rocm.yml`** (recommended):

ROCm requires a two-step installation (PyTorch must be installed via pip):

```bash
# 1. Check your ROCm version
rocm-smi --version

# 2. Create environment (without PyTorch)
conda env create -f environment_rocm.yml

# 3. Activate environment
conda activate bayesdream_rocm

# 4. Install PyTorch with ROCm via pip
# For ROCm 6.2 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 6.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# For ROCm 6.0:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 5. Install Pyro
pip install pyro-ppl

# 6. Install bayesDREAM
cd /path/to/bayesDREAM_forClaude
pip install -e .
```

**Verify GPU after installation:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"  # Uses 'cuda' API for AMD too
python -c "import torch; print('ROCm version:', torch.version.hip)"
```

**Important notes for ROCm:**
- PyTorch ROCm packages must be installed via pip (conda packages are incomplete)
- This matches the working pyroenv setup with torch 2.5.1+rocm6.2
- PyTorch uses 'cuda' API for both NVIDIA and AMD GPUs
- ROCm support is generally good but less mature than CUDA
- If you encounter issues, fall back to `environment_cpu.yml`

### CPU-Only Installation

**Using `environment_cpu.yml`** (recommended):
- No GPU support, optimized for CPU-only workloads
- Use for: local dev, login nodes, M1/M2 Mac (MPS not fully supported by Pyro)

```bash
conda env create -f environment_cpu.yml
conda activate bayesdream_cpu
pip install -e .
```

## HPC Cluster Installation

### SLURM Clusters (with GPU)

1. **Load required modules (if applicable):**
   ```bash
   module load anaconda3
   # or
   module load python/3.11
   ```

2. **Create environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate and install:**
   ```bash
   conda activate bayesdream
   pip install -e .
   ```

4. **Verify GPU support:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

5. **In your SLURM job script (GPU job):**
   ```bash
   #!/bin/bash
   #SBATCH --job-name=bayesdream
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --gres=gpu:1          # Request 1 GPU
   #SBATCH --time=24:00:00

   # Load modules if needed
   module load anaconda3

   # Activate environment
   conda activate bayesdream

   # Verify GPU is visible
   nvidia-smi
   echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

   # Run your script
   python your_script.py
   ```

### PBS/Torque Clusters

Similar to SLURM, but use PBS directives:
```bash
#!/bin/bash
#PBS -N bayesdream
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
conda activate bayesdream
python your_script.py
```

## Verification

Test your installation:
```bash
python -c "from bayesDREAM import bayesDREAM; print('✓ bayesDREAM imported successfully')"
```

Run the test suite:
```bash
cd tests
python test_trans_quick.py
```

Start JupyterLab for interactive analysis:
```bash
jupyter lab
# Or for classic notebook interface:
jupyter notebook
```

Verify R kernel is available in Jupyter:
```bash
jupyter kernelspec list
# Should show both 'python3' and 'ir' kernels
```

## Dependencies

### Required
- Python 3.11 (recommended for best package compatibility)
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- pandas ≥ 2.0.0
- scikit-learn ≥ 1.3.0
- PyTorch ≥ 2.2.0
- Pyro ≥ 1.9.0
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- h5py ≥ 3.8.0

**Note**: Version constraints have been relaxed for better cross-platform compatibility. The conda environment uses Python 3.11 (better package support than 3.12 on Linux clusters).

### For Preprocessing (included in conda environment)
- R ≥ 4.0
- Bioconductor scran ≥ 1.20 (for calculating sum factors)
- r-data.table (for fast data manipulation in R)

### For Interactive Analysis (included in conda environment)
- JupyterLab ≥ 4.0
- ipykernel (Python kernel for Jupyter)
- IRkernel (R kernel for Jupyter)
- notebook (classic Jupyter notebook)

**Note**: If using the conda environment (recommended), R, scran, and Jupyter components are installed automatically. After installation, register the R kernel with: `R -e "IRkernel::installspec(user = FALSE)"`

### Optional
- pytest ≥ 7.0.0 (for running tests)

## Troubleshooting

### Conda environment creation fails (LibMambaUnsatisfiableError)

If you get dependency conflicts when creating the environment:

**Option 1: Use mamba (faster and better at solving)**
```bash
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

**Option 2: Try with fewer channels**
Edit `environment.yml` and comment out the pytorch channel:
```yaml
channels:
  # - pytorch  # Comment this out
  - conda-forge
  - bioconda
  - defaults
```
Then create the environment again.

**Option 3: Create minimal environment and add packages**
```bash
# Create minimal environment
conda create -n bayesdream python=3.11 numpy scipy pandas scikit-learn pytorch -c conda-forge

# Activate it
conda activate bayesdream

# Install Pyro via pip
pip install pyro-ppl

# Add R and bioconda packages
conda install r-base bioconductor-scran r-irkernel r-data.table -c bioconda -c conda-forge

# Add Jupyter and visualization
conda install jupyterlab ipykernel notebook ipywidgets matplotlib seaborn h5py -c conda-forge

# Register R kernel
R -e "IRkernel::installspec(user = FALSE)"

# Install bayesDREAM
cd /path/to/bayesDREAM_forClaude
pip install -e .
```

### Import errors
If you see `ModuleNotFoundError: No module named 'bayesDREAM'`:
- Make sure you ran `pip install -e .` from the repository root
- Check that the environment is activated: `conda activate bayesdream`

### PyTorch GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs
```

If False:
- Verify CUDA is installed: `nvcc --version`
- Reinstall PyTorch with the correct CUDA version (see GPU Support above)

### Out of memory errors
- Reduce batch sizes in your scripts
- Use CPU instead of GPU for smaller datasets
- Request more memory in your SLURM/PBS script

### Slow performance
- Enable OpenBLAS or MKL for faster linear algebra:
  ```bash
  conda install openblas
  # or
  conda install mkl
  ```
- Use GPU if available
- Increase `--cpus-per-task` in SLURM scripts

## Support

For issues:
1. Check the documentation in `docs/`
2. Review test files in `tests/` for usage examples
3. Check `docs/OUTSTANDING_TASKS.md` for known issues

## Development Installation

For contributing to bayesDREAM:
```bash
conda env create -f environment.yml
conda activate bayesdream
pip install -e ".[dev]"  # Installs development dependencies
```

This installs additional tools:
- pytest and pytest-cov for testing
- black for code formatting
- flake8 for linting
