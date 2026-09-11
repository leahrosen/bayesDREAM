"""
Example: Run technical fit and save results.

This script demonstrates how to:
1. Load data
2. Initialize bayesDREAM with multi-modal support
3. Run fit_ntc() on the primary modality
4. Save fitted parameters for later use

Usage:
    python run_technical_example.py --outdir ./results --label my_analysis
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
    parser = argparse.ArgumentParser(description='Run bayesDREAM technical fit')
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--label", required=True, help="Label for this analysis")
    parser.add_argument("--cis_gene", default="GFI1B", help="Cis gene name")
    parser.add_argument("--meta", required=True, help="Path to metadata CSV")
    parser.add_argument("--counts", required=True, help="Path to counts CSV")
    parser.add_argument("--covariates", nargs='+', default=['cell_line'],
                       help="Technical covariates (space-separated)")
    parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores")
    args = parser.parse_args()

    set_max_threads(args.cores)

    # Set random seed
    seed = abs(hash(args.label + "_technical")) % (2**32)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print(f"Loading data...")
    meta = pd.read_csv(args.meta)
    counts = pd.read_csv(args.counts, index_col=0)
    print(f"  Meta: {meta.shape}")
    print(f"  Counts: {counts.shape}")

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

    # Set technical groups
    print(f"\nSetting technical groups: {args.covariates}")
    model.set_technical_groups(args.covariates)

    # Fit technical
    print(f"\nRunning fit_ntc...")
    model.fit_ntc(
        modality_name='gene',
        sum_factor_col='sum_factor',
        tolerance=0
    )

    # Save results
    print(f"\nSaving technical fit...")
    saved_files = model.save_ntc_fit()

    print(f"\n✓ Technical fit complete!")
    print(f"\nSaved files:")
    for name, path in saved_files.items():
        if not name.endswith('_type'):
            print(f"  {name}: {path}")
