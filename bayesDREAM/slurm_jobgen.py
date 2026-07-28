#!/usr/bin/env python3
"""
SLURM Job Generator for bayesDREAM on Berzelius HPC

Analyzes dataset characteristics and generates optimized SLURM job scripts
for fit_technical, fit_cis, and fit_trans with automatic resource allocation.

Usage:
    from bayesDREAM.slurm_jobgen import SlurmJobGenerator

    gen = SlurmJobGenerator(
        meta=meta,
        counts=counts,
        cis_genes=['GFI1B', 'TET2', 'MYB'],
        output_dir='./slurm_jobs',
        label='perturb_seq_batch1'
    )

    gen.generate_all_scripts()

Then on Berzelius:
    cd slurm_jobs
    bash submit_all.sh
"""

import pandas as pd
import numpy as np
from scipy import sparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings


class SlurmJobGenerator:
    """
    Generate SLURM job submission scripts for bayesDREAM pipeline on Berzelius.

    Berzelius resource specs (Ampere/A100 nodes only):
    - Fat nodes (-C fat): A100 80GB VRAM per GPU, 256GB RAM per GPU (2TB per 8-GPU node)
    - Thin nodes (-C thin): A100 40GB VRAM per GPU, 128GB RAM per GPU (1TB per 8-GPU node)
    - CPU partition (berzelius-cpu): 7.76GB RAM per core
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
        python_env: str = '/proj/berzelius-aiics-real/users/x_learo/mambaforge/envs/pyroenv/bin/python',
        bayesdream_path: str = '/proj/berzelius-aiics-real/users/x_learo/bayesDREAM',
        data_path: Optional[str] = None,
        nsamples: int = 1000,
        force_iaf: bool = False,
    ):
        """
        Initialize SLURM job generator.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell metadata with columns: cell, guide, target, cell_line, etc.
        counts : pd.DataFrame or sparse matrix
            Gene expression counts (features × cells)
        gene_meta : pd.DataFrame, optional
            Gene metadata with gene identifiers
        cis_genes : list of str, optional
            List of cis genes to fit. If None, must be provided to generate_all_scripts()
        output_dir : str
            Directory to write SLURM scripts
        label : str
            Unique label for this run
        low_moi : bool
            If True, low MOI mode (separate cis gene fits with NTC per gene)
            If False, high MOI mode (see use_all_cells_technical)
        use_all_cells_technical : bool
            High MOI mode: if True, fit_technical uses all cells (1 job total)
            If False, fit_technical uses NTC per cis gene (1 job per cis gene)
        distribution : str
            Distribution for modeling: 'negbinom', 'multinomial', 'binomial', etc.
        sparsity : float, optional
            Fraction of zeros in counts. Auto-detected if None.
        n_groups : int, optional
            Number of technical groups. Auto-detected from meta if None.
        max_concurrent_jobs : int
            Maximum concurrent jobs in arrays (--array=0-N%M)
        time_multiplier : float
            Scale all time estimates by this factor (default: 1.0)
        partition_preference : str
            'auto' (default), 'fat', 'thin', or 'cpu'
        python_env : str
            Path to python executable with bayesDREAM environment
        bayesdream_path : str
            Path to bayesDREAM repository root
        data_path : str, optional
            Path to directory containing saved data. If None, assumes data will be
            loaded from memory (not implemented - user must save data first)
        nsamples : int
            Number of posterior samples
        force_iaf : bool
            fit_ntc's guide defaults to AutoNormal (mean-field) — robust across
            dataset sizes and sparsity levels. Set True to request AutoIAFNormal
            instead (matches fit_ntc's own force_iaf parameter, and is threaded into
            the generated fit_technical script). Only recommended for small,
            deeply-sequenced datasets with no near-zero-count NTC features — IAF's
            single shared network across all features means one sparse feature's
            exploding gradient can corrupt every other feature's fit in one step.
        """
        self.meta = meta
        self.counts = counts
        self.gene_meta = gene_meta
        self.cis_genes = cis_genes
        self.output_dir = Path(output_dir)
        self.label = label
        self.low_moi = low_moi
        self.use_all_cells_technical = use_all_cells_technical
        self.distribution = distribution
        self.max_concurrent_jobs = max_concurrent_jobs
        self.time_multiplier = time_multiplier
        self.partition_preference = partition_preference
        self.python_env = python_env
        self.bayesdream_path = bayesdream_path
        self.data_path = data_path
        self.nsamples = nsamples
        self.force_iaf = force_iaf

        # Auto-detect dataset characteristics
        self.n_features = counts.shape[0]
        self.n_cells = counts.shape[1]

        if sparsity is None:
            if sparse.issparse(counts):
                self.sparsity = 1.0 - (counts.nnz / (counts.shape[0] * counts.shape[1]))
            else:
                self.sparsity = float(np.asarray(counts == 0).sum()) / counts.size
        else:
            self.sparsity = sparsity

        if n_groups is None:
            if 'technical_group_code' in meta.columns:
                self.n_groups = meta['technical_group_code'].nunique()
            elif 'cell_line' in meta.columns:
                self.n_groups = meta['cell_line'].nunique()
            else:
                self.n_groups = 2  # Conservative default
                warnings.warn("Could not auto-detect n_groups, using default=2")
        else:
            self.n_groups = n_groups

        # Count NTC fraction
        if 'target' in meta.columns:
            self.ntc_fraction = (meta['target'] == 'ntc').sum() / len(meta)
        else:
            self.ntc_fraction = 0.4  # Conservative default
            warnings.warn("Could not detect NTC fraction, using default=0.4")

        # Estimate n_guides (for fit_cis memory)
        if 'guide' in meta.columns:
            self.n_guides = meta['guide'].nunique()
        else:
            self.n_guides = max(100, self.n_features // 20)
            warnings.warn(f"Could not detect n_guides, estimating {self.n_guides}")

        print(f"Dataset characteristics:")
        print(f"  Features (T): {self.n_features:,}")
        print(f"  Cells (N): {self.n_cells:,}")
        print(f"  Technical groups (C): {self.n_groups}")
        print(f"  Guides (G): {self.n_guides:,}")
        print(f"  Sparsity: {self.sparsity*100:.1f}%")
        print(f"  NTC fraction: {self.ntc_fraction*100:.1f}%")
        print(f"  Distribution: {self.distribution}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def estimate_memory_requirements(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate RAM and VRAM requirements for each step using memory_calculator.

        Returns
        -------
        dict
            Memory estimates for each step with resource recommendations.
        """
        # docs/ is a sibling of bayesDREAM/ at the repo root, not an installed package --
        # only importable as `docs.memory_calculator` when the repo root is on sys.path.
        # That's true when a script is run with the repo root as cwd (or PYTHONPATH set
        # to it), but NOT when a script is invoked by its full path from elsewhere (e.g.
        # `$PYTHON_ENV $EXAMPLES/generate_slurm.py ...`), which puts the script's own
        # directory on sys.path[0] instead. Resolve relative to this file so the import
        # works regardless of how/from-where the caller was invoked.
        import sys
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from docs.memory_calculator import estimate_memory

        # Estimate for fit_technical
        results = estimate_memory(
            n_features=self.n_features,
            n_cells=self.n_cells,
            n_groups=self.n_groups,
            n_guides=self.n_guides,
            sparsity=self.sparsity,
            distribution=self.distribution,
            ntc_fraction=self.ntc_fraction,
            use_all_cells=self.use_all_cells_technical,
            nsamples=self.nsamples,
            verbose=False
        )

        # fit_ntc defaults to AutoNormal (mean-field) regardless of dataset size —
        # see bayesDREAM/fitting/ntc.py's fit_ntc docstring. AutoIAFNormal is opt-in
        # via force_iaf=True (matches fit_ntc's own parameter, threaded into the
        # generated fit_technical script below). This estimator used to auto-select
        # IAF whenever it fit an (essentially always crossed at realistic gene
        # counts) 20GB VRAM budget, which silently assumed IAF's 50k-iteration
        # profile almost universally — that no longer reflects fit_ntc's real
        # default, so the choice here now mirrors self.force_iaf directly.
        n_latent = (self.n_groups - 1 + 2) * self.n_features
        iaf_vram_gb = (n_latent ** 2 * 4 * 3 * 1.5) / 1e9
        self.use_autonormal = not self.force_iaf

        if self.use_autonormal:
            print(f"\n[INFO] Using AutoNormal (mean-field, default) with niters=100,000")
            print(f"[INFO] (AutoIAFNormal would need ~{iaf_vram_gb:.1f} GB VRAM — pass force_iaf=True to use it)")
        else:
            print(f"\n[INFO] force_iaf=True: using AutoIAFNormal (estimated {iaf_vram_gb:.1f} GB VRAM) with niters=50,000")

        # Add resource recommendations
        results['resources'] = self._recommend_resources(results)

        return results

    def _recommend_resources(self, memory: Dict[str, float]) -> Dict[str, Dict]:
        """
        Recommend Berzelius resources (fat/thin/cpu) based on memory requirements.

        Parameters
        ----------
        memory : dict
            Memory estimates from estimate_memory()

        Returns
        -------
        dict
            Resource recommendations for each step
        """
        resources = {}

        # fit_technical (same conditions as fit_trans)
        tech_ram = memory['fit_ntc_ram_gb']
        tech_vram = memory['fit_ntc_vram_gb']

        if self.partition_preference != 'auto':
            # User override
            if self.partition_preference == 'cpu':
                resources['fit_technical'] = {
                    'partition': 'berzelius-cpu',
                    'constraint': None,
                    'gpus': 0,
                    'cpus': max(1, int(np.ceil(tech_ram / 7.76))),
                    'mem_gb': tech_ram,
                    'vram_gb': 0,
                    'rationale': 'User requested CPU'
                }
            elif self.partition_preference == 'fat':
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'fat',
                    'gpus': max(1, int(np.ceil(tech_vram / 80))),
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'User requested fat nodes'
                }
            else:  # thin
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'thin',
                    'gpus': max(1, int(np.ceil(tech_vram / 40))),
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'User requested thin nodes'
                }
        else:
            # Auto-select (same logic as fit_trans)
            if tech_vram <= 40 and tech_ram <= 128:
                # Single thin GPU
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'thin',
                    'gpus': 1,
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'Fits on 1 thin GPU (40GB VRAM, 128GB RAM)'
                }
            elif tech_vram <= 80 and tech_ram <= 256:
                # Single fat GPU
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'fat',
                    'gpus': 1,
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'Fits on 1 fat GPU (80GB VRAM, 256GB RAM)'
                }
            elif tech_vram <= 160 and tech_ram <= 512:
                # 2 fat GPUs
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'fat',
                    'gpus': 2,
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'Requires 2 fat GPUs (160GB VRAM, 512GB RAM)'
                }
            elif tech_vram <= 320 and tech_ram <= 1024:
                # 4 fat GPUs
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'fat',
                    'gpus': 4,
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'Requires 4 fat GPUs (320GB VRAM, 1024GB RAM)'
                }
            elif tech_vram <= 640 and tech_ram <= 2048:
                # 8 fat GPUs (full node)
                resources['fit_technical'] = {
                    'partition': 'berzelius',
                    'constraint': 'fat',
                    'gpus': 8,
                    'cpus': 1,
                    'mem_gb': tech_ram,
                    'vram_gb': tech_vram,
                    'rationale': 'Requires 8 fat GPUs (640GB VRAM, 2048GB RAM) - full node'
                }
            else:
                # Too large for GPU, use CPU
                resources['fit_technical'] = {
                    'partition': 'berzelius-cpu',
                    'constraint': None,
                    'gpus': 0,
                    'cpus': max(1, int(np.ceil(tech_ram / 7.76))),
                    'mem_gb': tech_ram,
                    'vram_gb': 0,
                    'rationale': 'Dataset too large for 8 GPUs, using CPU'
                }

        # fit_cis (usually CPU is fine)
        cis_ram = memory['fit_cis_ram_gb']
        cis_vram = memory['fit_cis_vram_gb']

        resources['fit_cis'] = {
            'partition': 'berzelius-cpu',
            'constraint': None,
            'gpus': 0,
            'cpus': max(1, int(np.ceil(cis_ram / 7.76))),
            'mem_gb': cis_ram,
            'vram_gb': 0,
            'rationale': 'CPU sufficient for cis fitting'
        }

        # fit_trans (needs GPU)
        trans_ram = memory['fit_trans_ram_gb']
        trans_vram = memory['fit_trans_vram_gb']

        if trans_vram <= 40 and trans_ram <= 128:
            # Single thin GPU
            resources['fit_trans'] = {
                'partition': 'berzelius',
                'constraint': 'thin',
                'gpus': 1,
                'cpus': 1,
                'mem_gb': trans_ram,
                'vram_gb': trans_vram,
                'rationale': 'Fits on 1 thin GPU (40GB VRAM, 128GB RAM)'
            }
        elif trans_vram <= 80 and trans_ram <= 256:
            # Single fat GPU
            resources['fit_trans'] = {
                'partition': 'berzelius',
                'constraint': 'fat',
                'gpus': 1,
                'cpus': 1,
                'mem_gb': trans_ram,
                'vram_gb': trans_vram,
                'rationale': 'Fits on 1 fat GPU (80GB VRAM, 256GB RAM)'
            }
        elif trans_vram <= 160 and trans_ram <= 512:
            # 2 fat GPUs
            resources['fit_trans'] = {
                'partition': 'berzelius',
                'constraint': 'fat',
                'gpus': 2,
                'cpus': 1,
                'mem_gb': trans_ram,
                'vram_gb': trans_vram,
                'rationale': 'Requires 2 fat GPUs (160GB VRAM, 512GB RAM)'
            }
        elif trans_vram <= 320 and trans_ram <= 1024:
            # 4 fat GPUs
            resources['fit_trans'] = {
                'partition': 'berzelius',
                'constraint': 'fat',
                'gpus': 4,
                'cpus': 1,
                'mem_gb': trans_ram,
                'vram_gb': trans_vram,
                'rationale': 'Requires 4 fat GPUs (320GB VRAM, 1024GB RAM)'
            }
        elif trans_vram <= 640 and trans_ram <= 2048:
            # 8 fat GPUs (full node)
            resources['fit_trans'] = {
                'partition': 'berzelius',
                'constraint': 'fat',
                'gpus': 8,
                'cpus': 1,
                'mem_gb': trans_ram,
                'vram_gb': trans_vram,
                'rationale': 'Requires 8 fat GPUs (640GB VRAM, 2048GB RAM) - full node'
            }
        else:
            # Too large for GPU, use CPU
            resources['fit_trans'] = {
                'partition': 'berzelius-cpu',
                'constraint': None,
                'gpus': 0,
                'cpus': max(1, int(np.ceil(trans_ram / 7.76))),
                'mem_gb': trans_ram,
                'vram_gb': 0,
                'rationale': 'Dataset too large for 8 GPUs, using CPU'
            }

        return resources

    def estimate_time_requirements(self, memory: Dict[str, float]) -> Dict[str, str]:
        """
        Estimate wall-clock time for each step based on dataset size.

        Parameters
        ----------
        memory : dict
            Memory estimates (used to infer complexity)

        Returns
        -------
        dict
            Time estimates in HH:MM:SS format for each step
        """
        # Base estimates for medium dataset (20k genes, 30k cells)
        base_tech_hours = 1.5  # fit_technical
        base_cis_hours = 0.5   # fit_cis per gene
        base_trans_hours = 3.0 # fit_trans per gene

        # Scale by dataset size (rough heuristic)
        size_factor = (self.n_features / 20000) * (self.n_cells / 30000)

        # Adjust for AutoNormal (takes longer than AutoIAFNormal)
        if self.use_autonormal:
            tech_hours = base_tech_hours * size_factor * 2.0  # 2× for AutoNormal
        else:
            tech_hours = base_tech_hours * size_factor

        cis_hours = base_cis_hours * (self.n_cells / 30000)
        trans_hours = base_trans_hours * size_factor

        # Apply user multiplier
        tech_hours *= self.time_multiplier
        cis_hours *= self.time_multiplier
        trans_hours *= self.time_multiplier

        # Add safety margin (1.5×)
        tech_hours *= 1.5
        cis_hours *= 1.5
        trans_hours *= 1.5

        # Convert to HH:MM:SS
        def hours_to_slurm(hours):
            h = int(hours)
            m = int((hours - h) * 60)
            return f"{h:02d}:{m:02d}:00"

        return {
            'fit_technical': hours_to_slurm(tech_hours),
            'fit_cis': hours_to_slurm(cis_hours),
            'fit_trans': hours_to_slurm(trans_hours),
        }

    def generate_all_scripts(self, cis_genes: Optional[List[str]] = None):
        """
        Generate all SLURM job scripts with dependencies.

        Parameters
        ----------
        cis_genes : list of str, optional
            List of cis genes. Uses self.cis_genes if None.
        """
        if cis_genes is not None:
            self.cis_genes = cis_genes

        if self.cis_genes is None:
            raise ValueError("cis_genes must be provided either at init or to generate_all_scripts()")

        print(f"\n{'='*70}")
        print(f"Generating SLURM scripts for {len(self.cis_genes)} cis genes")
        print(f"{'='*70}\n")

        # Estimate memory and time
        memory = self.estimate_memory_requirements()
        times = self.estimate_time_requirements(memory)
        resources = memory['resources']

        # Print resource summary
        self._print_resource_summary(memory, times, resources)

        # Generate scripts
        self._write_fit_technical_script(resources, times)
        self._write_fit_cis_script(resources, times)
        self._write_fit_trans_script(resources, times)
        self._write_submit_all_script()
        self._write_readme(memory, times, resources)

        print(f"\n{'='*70}")
        print(f"Scripts written to: {self.output_dir}")
        print(f"{'='*70}")
        print(f"\nTo submit jobs on Berzelius:")
        print(f"  cd {self.output_dir}")
        print(f"  bash submit_all.sh")
        print(f"\nOr submit individually:")
        print(f"  sbatch 01_fit_technical.sh")
        print(f"  sbatch 02_fit_cis.sh  # After technical completes")
        print(f"  sbatch 03_fit_trans.sh  # After cis completes")

    def _print_resource_summary(self, memory: Dict, times: Dict, resources: Dict):
        """Print summary of resource allocation."""
        print(f"Memory Requirements:")
        print(f"  fit_technical: {memory['fit_ntc_ram_gb']:.1f} GB RAM, "
              f"{memory['fit_ntc_vram_gb']:.1f} GB VRAM")
        print(f"  fit_cis:       {memory['fit_cis_ram_gb']:.1f} GB RAM, "
              f"{memory['fit_cis_vram_gb']:.1f} GB VRAM")
        print(f"  fit_trans:     {memory['fit_trans_ram_gb']:.1f} GB RAM, "
              f"{memory['fit_trans_vram_gb']:.1f} GB VRAM")

        print(f"\nEstimated Time:")
        print(f"  fit_technical: {times['fit_technical']} (wall time)")
        print(f"  fit_cis:       {times['fit_cis']} per gene")
        print(f"  fit_trans:     {times['fit_trans']} per gene")

        print(f"\nResource Allocation:")
        for step, res in resources.items():
            if res['partition'] == 'berzelius-cpu':
                print(f"  {step}: {res['cpus']} CPUs ({res['mem_gb']:.1f} GB RAM)")
            else:
                print(f"  {step}: {res['gpus']} GPU(s) on {res['constraint']} nodes "
                      f"({res['vram_gb']:.1f} GB VRAM, {res['mem_gb']:.1f} GB RAM)")
            print(f"    → {res['rationale']}")

    def _write_fit_technical_script(self, resources: Dict, times: Dict):
        """Generate fit_technical SLURM script."""
        res = resources['fit_technical']
        time = times['fit_technical']

        # Determine number of jobs
        if self.low_moi or self.use_all_cells_technical:
            # Single job for all cis genes
            n_jobs = 1
            job_type = "single"
        else:
            # High MOI with NTC per cis gene
            n_jobs = len(self.cis_genes)
            job_type = "array"

        script = f"""#!/bin/bash
#SBATCH --job-name={self.label}_tech
#SBATCH --output={self.output_dir}/logs/tech_%j.out
#SBATCH --error={self.output_dir}/logs/tech_%j.err
#SBATCH --time={time}
"""

        if job_type == "array":
            script += f"#SBATCH --array=0-{n_jobs-1}%{min(self.max_concurrent_jobs, n_jobs)}\n"

        if res['partition'] == 'berzelius-cpu':
            script += f"#SBATCH --partition={res['partition']}\n"
            script += f"#SBATCH --cpus-per-task={res['cpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"
        else:
            script += f"#SBATCH --partition={res['partition']}\n"
            script += f"#SBATCH -C {res['constraint']}\n"
            script += f"#SBATCH --gpus={res['gpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"

        script += f"""
# Environment setup
export PYTHONPATH="{self.bayesdream_path}:$PYTHONPATH"
module load Anaconda/2021.05-nsc1

# Create logs directory
mkdir -p {self.output_dir}/logs

# Determine cis gene if array job
"""

        if job_type == "array":
            script += f"""CIS_GENES=({' '.join(self.cis_genes)})
CIS_GENE=${{CIS_GENES[$SLURM_ARRAY_TASK_ID]}}
echo "Processing cis gene: $CIS_GENE"
"""
        else:
            script += f"""CIS_GENE="ALL"
echo "Processing all cis genes (shared technical fit)"
"""

        # Python command
        niters = 100_000 if self.use_autonormal else 50_000
        use_all_cells_flag = "True" if self.use_all_cells_technical else "False"

        script += f"""
# Run fit_technical
{self.python_env} << 'PYEOF'
import sys
sys.path.insert(0, "{self.bayesdream_path}")

from bayesDREAM import bayesDREAM
import pandas as pd
import pickle
import os

# Load data
print("Loading data...")
if "{self.data_path}" != "None":
    meta = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/meta.csv")
    counts = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/counts.csv", index_col=0)
else:
    raise ValueError("data_path must be provided")

# Initialize model
cis_gene = os.environ.get('CIS_GENE', 'ALL')
print(f"Initializing bayesDREAM for cis_gene: {{cis_gene}}")

if cis_gene == "ALL":
    cis_gene = None  # Shared technical fit

model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if res['gpus'] > 0 else 'cpu'}"
)

# Set technical groups
model.set_technical_groups(['cell_line'])  # Adjust as needed

# Run fit_ntc
print("Running fit_ntc...")
model.fit_ntc(
    niters={niters},
    use_all_cells={use_all_cells_flag},
    nsamples={self.nsamples},
    force_iaf={self.force_iaf}
)

print("fit_ntc completed successfully")
PYEOF

echo "Job completed: $SLURM_JOB_ID"
"""

        script_path = self.output_dir / "01_fit_technical.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)

        print(f"✓ Generated: {script_path}")

    def _write_fit_cis_script(self, resources: Dict, times: Dict):
        """Generate fit_cis SLURM script (job array over cis genes)."""
        res = resources['fit_cis']
        time = times['fit_cis']
        n_genes = len(self.cis_genes)

        script = f"""#!/bin/bash
#SBATCH --job-name={self.label}_cis
#SBATCH --output={self.output_dir}/logs/cis_%A_%a.out
#SBATCH --error={self.output_dir}/logs/cis_%A_%a.err
#SBATCH --array=0-{n_genes-1}%{min(self.max_concurrent_jobs, n_genes)}
#SBATCH --time={time}
#SBATCH --partition={res['partition']}
"""

        if res['partition'] == 'berzelius-cpu':
            script += f"#SBATCH --cpus-per-task={res['cpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"
        else:
            script += f"#SBATCH -C {res['constraint']}\n"
            script += f"#SBATCH --gpus={res['gpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"

        script += f"""
# Environment setup
export PYTHONPATH="{self.bayesdream_path}:$PYTHONPATH"
module load Anaconda/2021.05-nsc1

# Get cis gene from array
CIS_GENES=({' '.join(self.cis_genes)})
CIS_GENE=${{CIS_GENES[$SLURM_ARRAY_TASK_ID]}}
echo "Processing cis gene: $CIS_GENE (array task $SLURM_ARRAY_TASK_ID)"

# Run fit_cis
{self.python_env} << 'PYEOF'
import sys
sys.path.insert(0, "{self.bayesdream_path}")

from bayesDREAM import bayesDREAM
import pandas as pd
import os

# Load data
print("Loading data...")
if "{self.data_path}" != "None":
    meta = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/meta.csv")
    counts = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/counts.csv", index_col=0)
else:
    raise ValueError("data_path must be provided")

cis_gene = os.environ.get('CIS_GENE')
print(f"Initializing bayesDREAM for cis_gene: {{cis_gene}}")

model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if res['gpus'] > 0 else 'cpu'}"
)

# Load technical fit results
print("Loading technical fit results...")
model.load_technical_results()  # Assumes standard save location

# Run fit_cis
print("Running fit_cis...")
model.fit_cis(sum_factor_col='sum_factor', nsamples={self.nsamples})

print(f"fit_cis completed successfully for {{cis_gene}}")
PYEOF

echo "Job completed: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
"""

        script_path = self.output_dir / "02_fit_cis.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)

        print(f"✓ Generated: {script_path}")

    def _write_fit_trans_script(self, resources: Dict, times: Dict):
        """Generate fit_trans SLURM script (job array over cis genes)."""
        res = resources['fit_trans']
        time = times['fit_trans']
        n_genes = len(self.cis_genes)

        script = f"""#!/bin/bash
#SBATCH --job-name={self.label}_trans
#SBATCH --output={self.output_dir}/logs/trans_%A_%a.out
#SBATCH --error={self.output_dir}/logs/trans_%A_%a.err
#SBATCH --array=0-{n_genes-1}%{min(self.max_concurrent_jobs, n_genes)}
#SBATCH --time={time}
"""

        if res['partition'] == 'berzelius-cpu':
            script += f"#SBATCH --partition={res['partition']}\n"
            script += f"#SBATCH --cpus-per-task={res['cpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"
        else:
            script += f"#SBATCH --partition={res['partition']}\n"
            script += f"#SBATCH -C {res['constraint']}\n"
            script += f"#SBATCH --gpus={res['gpus']}\n"
            script += f"#SBATCH --mem={int(res['mem_gb']+5)}G\n"

        script += f"""
# Environment setup
export PYTHONPATH="{self.bayesdream_path}:$PYTHONPATH"
module load Anaconda/2021.05-nsc1

# Get cis gene from array
CIS_GENES=({' '.join(self.cis_genes)})
CIS_GENE=${{CIS_GENES[$SLURM_ARRAY_TASK_ID]}}
echo "Processing cis gene: $CIS_GENE (array task $SLURM_ARRAY_TASK_ID)"

# Run fit_trans
{self.python_env} << 'PYEOF'
import sys
sys.path.insert(0, "{self.bayesdream_path}")

from bayesDREAM import bayesDREAM
import pandas as pd
import os

# Load data
print("Loading data...")
if "{self.data_path}" != "None":
    meta = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/meta.csv")
    counts = pd.read_csv(f"{{os.environ.get('DATA_PATH', '{self.data_path}')}}/counts.csv", index_col=0)
else:
    raise ValueError("data_path must be provided")

cis_gene = os.environ.get('CIS_GENE')
print(f"Initializing bayesDREAM for cis_gene: {{cis_gene}}")

model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if res['gpus'] > 0 else 'cpu'}"
)

# Load technical and cis fit results
print("Loading previous results...")
model.load_technical_results()
model.load_cis_results()

# Run fit_trans
print("Running fit_trans...")
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    nsamples={self.nsamples}
)

print(f"fit_trans completed successfully for {{cis_gene}}")
PYEOF

echo "Job completed: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
"""

        script_path = self.output_dir / "03_fit_trans.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)

        print(f"✓ Generated: {script_path}")

    def _write_submit_all_script(self):
        """Generate master submission script with dependencies."""
        script = f"""#!/bin/bash
# Master submission script with job dependencies
# Generated for: {self.label}

set -e  # Exit on error

echo "Submitting bayesDREAM pipeline jobs..."
echo "Label: {self.label}"
echo "Output directory: {self.output_dir}"
echo ""

# Create logs directory
mkdir -p {self.output_dir}/logs

# Submit fit_technical
echo "Submitting fit_technical..."
TECH_JOB=$(sbatch --parsable 01_fit_technical.sh)
echo "  Job ID: $TECH_JOB"

# Submit fit_cis (depends on technical)
echo "Submitting fit_cis (depends on $TECH_JOB)..."
CIS_JOB=$(sbatch --parsable --dependency=afterok:$TECH_JOB 02_fit_cis.sh)
echo "  Job ID: $CIS_JOB"

# Submit fit_trans (depends on cis)
echo "Submitting fit_trans (depends on $CIS_JOB)..."
TRANS_JOB=$(sbatch --parsable --dependency=afterok:$CIS_JOB 03_fit_trans.sh)
echo "  Job ID: $TRANS_JOB"

echo ""
echo "All jobs submitted successfully!"
echo ""
echo "Monitor jobs:"
echo "  squeue -u $USER"
echo "  sacct -j $TECH_JOB,$CIS_JOB,$TRANS_JOB"
echo ""
echo "Cancel jobs:"
echo "  scancel $TECH_JOB $CIS_JOB $TRANS_JOB"
"""

        script_path = self.output_dir / "submit_all.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)

        print(f"✓ Generated: {script_path}")

    def _write_readme(self, memory: Dict, times: Dict, resources: Dict):
        """Generate README with instructions and resource summary."""
        readme = f"""# bayesDREAM SLURM Jobs - {self.label}

Generated: {pd.Timestamp.now()}

## Dataset Characteristics

- Features: {self.n_features:,}
- Cells: {self.n_cells:,}
- Technical groups: {self.n_groups}
- Guides: {self.n_guides:,}
- Sparsity: {self.sparsity*100:.1f}%
- Distribution: {self.distribution}
- Cis genes: {len(self.cis_genes)} ({', '.join(self.cis_genes[:5])}{'...' if len(self.cis_genes) > 5 else ''})

## Memory Requirements

| Step | RAM | VRAM |
|------|-----|------|
| fit_technical | {memory['fit_ntc_ram_gb']:.1f} GB | {memory['fit_ntc_vram_gb']:.1f} GB |
| fit_cis | {memory['fit_cis_ram_gb']:.1f} GB | {memory['fit_cis_vram_gb']:.1f} GB |
| fit_trans | {memory['fit_trans_ram_gb']:.1f} GB | {memory['fit_trans_vram_gb']:.1f} GB |

## Estimated Time

| Step | Time per job |
|------|--------------|
| fit_technical | {times['fit_technical']} |
| fit_cis | {times['fit_cis']} |
| fit_trans | {times['fit_trans']} |

## Resource Allocation

"""
        for step, res in resources.items():
            readme += f"### {step}\n\n"
            if res['partition'] == 'berzelius-cpu':
                readme += f"- Partition: CPU\n"
                readme += f"- CPUs: {res['cpus']}\n"
                readme += f"- RAM: {res['mem_gb']:.1f} GB\n"
            else:
                readme += f"- Partition: {res['partition']}\n"
                readme += f"- Constraint: {res['constraint']}\n"
                readme += f"- GPUs: {res['gpus']}\n"
                readme += f"- VRAM: {res['vram_gb']:.1f} GB\n"
                readme += f"- RAM: {res['mem_gb']:.1f} GB\n"
            readme += f"- Rationale: {res['rationale']}\n\n"

        # Determine niters based on AutoNormal vs AutoIAFNormal
        niters = 100_000 if self.use_autonormal else 50_000

        readme += f"""
## What Will Be Run

### fit_technical

**Script**: `01_fit_technical.sh`

**Python Code**:
```python
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene={'None' if not self.low_moi else 'cis_gene_per_job'},
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if resources['fit_technical']['gpus'] > 0 else 'cpu'}"
)

model.set_technical_groups(['cell_line'])
model.fit_ntc(
    niters={niters},
    use_all_cells={str(self.use_all_cells_technical)},
    nsamples={self.nsamples},
    force_iaf={self.force_iaf}
)
```

**Output**: `alpha_y_prefit.pt`, `posterior_samples_ntc.pt`

### fit_cis

**Script**: `02_fit_cis.sh` (job array over {len(self.cis_genes)} cis genes)

**Python Code** (per cis gene):
```python
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,  # e.g., 'GFI1B'
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if resources['fit_cis']['gpus'] > 0 else 'cpu'}"
)

model.load_technical_results()
model.fit_cis(sum_factor_col='sum_factor', nsamples={self.nsamples})
```

**Output** (per gene): `{{gene}}_run/x_true.pt`, `{{gene}}_run/posterior_samples_cis.pt`

### fit_trans

**Script**: `03_fit_trans.sh` (job array over {len(self.cis_genes)} cis genes)

**Python Code** (per cis gene):
```python
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,  # e.g., 'GFI1B'
    label="{self.label}",
    output_dir="{self.output_dir}/output",
    device="{'cuda' if resources['fit_trans']['gpus'] > 0 else 'cpu'}"
)

model.load_technical_results()
model.load_cis_results()
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    nsamples={self.nsamples}
)
```

**Output** (per gene): `{{gene}}_run/posterior_samples_trans_additive_hill_none.pt`

## Log Output

Each script produces logs in `logs/` with stdout and stderr:

**fit_ntc** (`logs/tech_*.out`):
```
Processing cis gene: GFI1B
Loading data...
Initializing bayesDREAM for cis_gene: GFI1B
Running fit_ntc...
[Pyro iteration output with ELBO values]
fit_ntc completed successfully
Job completed: 123456
```

**fit_cis** (`logs/cis_*_*.out`):
```
Processing cis gene: GFI1B (array task 0)
Loading data...
Initializing bayesDREAM for cis_gene: GFI1B
Loading technical fit results...
Running fit_cis...
[Pyro iteration output with ELBO values]
fit_cis completed successfully for GFI1B
Job completed: 123457 (array task 0)
```

**fit_trans** (`logs/trans_*_*.out`):
```
Processing cis gene: GFI1B (array task 0)
Loading data...
Initializing bayesDREAM for cis_gene: GFI1B
Loading previous results...
Running fit_trans...
[Pyro iteration output with ELBO values]
fit_trans completed successfully for GFI1B
Job completed: 123458 (array task 0)
```

**Monitor logs in real-time**:
```bash
tail -f logs/tech_*.out
tail -f logs/cis_*_*.out
tail -f logs/trans_*_*.out
```

## Usage

### Quick Start (recommended)

Submit all jobs with dependencies:
```bash
bash submit_all.sh
```

### Manual Submission

Submit jobs one at a time:
```bash
# 1. Technical fitting
sbatch 01_fit_technical.sh
# Wait for completion, note JOB_ID

# 2. Cis fitting (after technical completes)
sbatch --dependency=afterok:TECH_JOB_ID 02_fit_cis.sh
# Wait for completion, note JOB_ID

# 3. Trans fitting (after cis completes)
sbatch --dependency=afterok:CIS_JOB_ID 03_fit_trans.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job details
sacct -j JOB_ID

# View logs
tail -f logs/tech_*.out
tail -f logs/cis_*_*.out
tail -f logs/trans_*_*.out
```

### Cancel Jobs

```bash
# Cancel all jobs for this label
scancel --name={self.label}_tech
scancel --name={self.label}_cis
scancel --name={self.label}_trans

# Or cancel specific job
scancel JOB_ID
```

## Job Array Throttling

The fit_cis and fit_trans scripts use job arrays with throttling:
- Maximum concurrent jobs: {self.max_concurrent_jobs}
- Total cis genes: {len(self.cis_genes)}

This prevents overwhelming the cluster while parallelizing across cis genes.

## Output Structure

```
{self.output_dir}/
├── output/
│   └── {self.label}/
│       ├── alpha_y_prefit.pt
│       ├── GFI1B_run/
│       │   ├── x_true.pt
│       │   ├── posterior_samples_cis.pt
│       │   └── posterior_samples_trans_additive_hill_none.pt
│       ├── TET2_run/
│       └── ...
└── logs/
    ├── tech_*.out
    ├── cis_*_*.out
    └── trans_*_*.out
```

## Troubleshooting

### Job fails with "Out of memory"
- Increase memory request in SLURM script
- Use more GPUs or fat nodes
- Consider CPU partition for very large datasets

### Job times out
- Increase time limit in SLURM script
- Set time_multiplier > 1.0 when generating scripts

### Job fails to find data
- Ensure data_path is correct
- Check that meta.csv and counts.csv exist
- Verify file permissions

### Dependencies not working
- Check job IDs with `squeue -u $USER`
- Ensure --parsable flag is working
- Submit jobs manually with correct dependencies

## Contact

For issues with:
- bayesDREAM code: See repository documentation
- Berzelius cluster: Contact NSC support
- SLURM errors: Check Berzelius documentation
"""

        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)

        print(f"✓ Generated: {readme_path}")


if __name__ == "__main__":
    print("bayesDREAM SLURM Job Generator")
    print("Import this module and use SlurmJobGenerator class")
    print("See docstring for usage examples")
