"""
Example: Load technical fit, run cis fit, and save results.

This script demonstrates how to:
1. Initialize bayesDREAM
2. Load previously fitted technical parameters
3. Run fit_cis()
4. Save cis fit results

Usage:
    python run_cis_example.py --outdir ./results --label my_analysis
"""

import os
import argparse
import pandas as pd
import torch
import pyro
import numpy as np
from bayesDREAM import bayesDREAM

def set_max_threads(cores: int):
    """Set maximum threads for various libraries."""
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    torch.set_num_threads(cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run bayesDREAM cis fit')
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--label", required=True, help="Label for this analysis")
    parser.add_argument("--cis_gene", default="GFI1B", help="Cis gene name")
    parser.add_argument("--meta", required=True, help="Path to metadata CSV")
    parser.add_argument("--counts", required=True, help="Path to counts CSV")
    parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores")
    args = parser.parse_args()

    set_max_threads(args.cores)

    # Set random seed
    seed = abs(hash(args.label + "_cis_" + args.cis_gene)) % (2**32)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print(f"Loading data...")
    meta = pd.read_csv(args.meta)
    counts = pd.read_csv(args.counts, index_col=0)

    # Initialize model
    output_dir = os.path.join(args.outdir, args.label)
    print(f"\nInitializing bayesDREAM...")
    print(f"  Output: {output_dir}")
    print(f"  Cis gene: {args.cis_gene}")

    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene=args.cis_gene,
        output_dir=output_dir,
        label=args.label,
        random_seed=seed,
        cores=args.cores,
        modality_name='gene'
    )

    # Load technical fit
    print(f"\nLoading technical fit...")
    model.load_ntc_fit()

    # Fit cis
    print(f"\nRunning fit_cis...")
    model.fit_cis(
        sum_factor_col='sum_factor',
        tolerance=0,
        niters=100000
    )

    # Save results
    print(f"\nSaving cis fit...")
    saved_files = model.save_cis_fit()

    print(f"\n✓ Cis fit complete!")
    print(f"\nSaved files:")
    for name, path in saved_files.items():
        if not name.endswith('_type'):
            print(f"  {name}: {path}")
