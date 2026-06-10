#!/usr/bin/env python3
"""
bayesDREAM Memory Requirements Calculator

Run this script to estimate RAM and VRAM requirements for your dataset.

Usage:
    python memory_calculator.py

Or import and use programmatically:
    from memory_calculator import estimate_memory
    memory = estimate_memory(n_features=30000, n_cells=50000, n_groups=2)
    print(memory)
"""

def estimate_memory(
    n_features: int,
    n_cells: int,
    n_groups: int = 2,
    n_guides: int = None,
    n_categories: int = None,
    sparsity: float = 0.85,
    distribution: str = 'negbinom',
    ntc_fraction: float = 0.4,
    function_type: str = 'additive_hill',
    nsamples: int = 1000,
    use_all_cells: bool = False,
    verbose: bool = True
):
    """
    Estimate memory requirements for bayesDREAM fitting.

    Parameters
    ----------
    n_features : int
        Number of features (genes, junctions, etc.)
    n_cells : int
        Total number of cells
    n_groups : int
        Number of technical groups (e.g., cell lines, batches)
    n_guides : int, optional
        Number of guides (defaults to n_features / 20)
    n_categories : int, optional
        Number of categories for multinomial (e.g., acceptors per donor)
    sparsity : float
        Fraction of zeros in counts matrix (0.0-1.0)
    distribution : str
        Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'studentt'
    ntc_fraction : float
        Fraction of cells that are non-targeting controls (ignored if use_all_cells=True)
    function_type : str
        Trans function: 'single_hill', 'additive_hill', 'polynomial'
    nsamples : int
        Number of posterior samples
    use_all_cells : bool, default False
        If True, fit_ntc uses ALL cells instead of NTC-only (high MOI mode).
        When True, ignores ntc_fraction and uses 100% of cells for technical fitting.
        Increases fit_ntc memory by ~2.5× but saves compute (only run once per dataset).
    verbose : bool
        Print detailed breakdown

    Returns
    -------
    dict
        Memory estimates with keys:
        - 'fit_ntc_ram_gb'
        - 'fit_ntc_vram_gb'
        - 'fit_cis_ram_gb'
        - 'fit_cis_vram_gb'
        - 'fit_trans_ram_gb'
        - 'fit_trans_vram_gb'
        - 'recommended_ram_gb'
        - 'recommended_vram_gb'
        - 'min_ram_gb'
        - 'min_vram_gb'
    """

    # Default values
    if n_guides is None:
        n_guides = max(100, n_features // 20)

    # Override ntc_fraction if use_all_cells mode
    if use_all_cells:
        ntc_fraction = 1.0

    # Validate inputs
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")
    if not 0 < ntc_fraction <= 1:
        raise ValueError("ntc_fraction must be between 0 and 1")

    # Calculate derived quantities
    n_ntc = int(n_cells * ntc_fraction)

    # Function-specific parameters
    function_params = {
        'single_hill': 5,
        'additive_hill': 7,
        'polynomial': 8,
    }
    n_func_params = function_params.get(function_type, 7)

    # Distribution-specific multipliers
    if distribution == 'multinomial':
        if n_categories is None:
            raise ValueError("n_categories required for multinomial distribution")
        data_multiplier = n_categories
        has_denominator = False
    elif distribution == 'binomial':
        data_multiplier = 2  # numerator + denominator
        has_denominator = True
    else:
        data_multiplier = 1
        has_denominator = False

    # ========================================
    # fit_ntc
    # ========================================

    # Data memory (in GB)
    if distribution == 'multinomial':
        data_tech_gb = (n_features * n_ntc * n_categories * 4 * (1 - sparsity)) / 1e9
    else:
        data_tech_gb = (n_features * n_ntc * 4 * (1 - sparsity)) / 1e9

    # Denominator (binomial only)
    if has_denominator:
        data_tech_gb *= 2

    # Guide memory (AutoNormal - IAF is too large and auto-falls back)
    if distribution == 'multinomial':
        # More latent variables for multinomial
        n_latent_tech = (n_groups - 1) * n_features * n_categories
    else:
        # negbinom: log2_alpha_y (C-1 × T) + o_y (T) + mu_ntc (T)
        n_latent_tech = (n_groups - 1 + 2) * n_features

    guide_tech_gb = (n_latent_tech * 2 * 4) / 1e9  # mean + std

    # Gradients and optimizer state
    grad_tech_gb = guide_tech_gb * 3  # params + gradients + momentum

    # Posterior samples (stored on CPU, counts toward RAM)
    posterior_tech_gb = (nsamples * n_latent_tech * 4) / 1e9

    # Total
    ram_tech = 4 + data_tech_gb * 1.5 + guide_tech_gb + grad_tech_gb + posterior_tech_gb
    vram_tech = 6 + data_tech_gb * 1.2 + guide_tech_gb + grad_tech_gb

    # ========================================
    # fit_cis
    # ========================================

    # Cis gene counts (negligible)
    data_cis_gb = (1 * n_cells * 4) / 1e9

    # Guide assignment (main component for high MOI)
    guide_assign_gb = (n_cells * n_guides * 4) / 1e9

    # x_true estimates
    xtrue_gb = (n_guides * 4) / 1e9

    # Posteriors
    posterior_cis_gb = (nsamples * n_guides * 4) / 1e9

    # Total
    ram_cis = 4 + guide_assign_gb * 1.5 + posterior_cis_gb
    vram_cis = 4 + guide_assign_gb * 1.2

    # ========================================
    # fit_trans
    # ========================================

    # Data memory
    if distribution == 'multinomial':
        data_trans_gb = (n_features * n_cells * n_categories * 4 * (1 - sparsity)) / 1e9
    else:
        data_trans_gb = (n_features * n_cells * 4 * (1 - sparsity)) / 1e9

    if has_denominator:
        data_trans_gb *= 2

    # Parameters
    n_params_trans = n_func_params * n_features
    params_gb = (n_params_trans * 4) / 1e9

    # Gradients (largest component)
    grad_trans_gb = params_gb * 3

    # Posterior samples
    posterior_trans_gb = (nsamples * n_params_trans * 4) / 1e9

    # Total
    ram_trans = 10 + data_trans_gb * 2 + params_gb + grad_trans_gb + posterior_trans_gb
    vram_trans = 8 + data_trans_gb * 1.2 + params_gb + grad_trans_gb * 1.5

    # ========================================
    # Recommendations
    # ========================================

    # Minimum: peak usage across steps
    min_ram = max(ram_tech, ram_cis, ram_trans)
    min_vram = max(vram_tech, vram_cis, vram_trans)

    # Recommended: 1.5× minimum for safety + overhead
    rec_ram = min_ram * 1.5
    rec_vram = min_vram * 1.5

    # Round up to common sizes
    def round_to_power2(x):
        """Round up to nearest power of 2 (GPU sizes)."""
        import math
        return 2 ** math.ceil(math.log2(x))

    rec_vram_rounded = round_to_power2(rec_vram)

    results = {
        'fit_ntc_ram_gb': ram_tech,
        'fit_ntc_vram_gb': vram_tech,
        'fit_cis_ram_gb': ram_cis,
        'fit_cis_vram_gb': vram_cis,
        'fit_trans_ram_gb': ram_trans,
        'fit_trans_vram_gb': vram_trans,
        'min_ram_gb': min_ram,
        'min_vram_gb': min_vram,
        'recommended_ram_gb': rec_ram,
        'recommended_vram_gb': rec_vram_rounded,
    }

    if verbose:
        print("=" * 70)
        print("bayesDREAM Memory Requirements Estimate")
        print("=" * 70)
        print(f"\nDataset Characteristics:")
        print(f"  Features (T):           {n_features:,}")
        print(f"  Cells (N):              {n_cells:,}")
        if use_all_cells:
            print(f"  fit_ntc mode:     ALL CELLS (high MOI mode)")
            print(f"  Cells for technical:    {n_ntc:,} (100%)")
        else:
            print(f"  fit_ntc mode:     NTC-only (standard)")
            print(f"  NTC cells:              {n_ntc:,} ({ntc_fraction*100:.0f}%)")
        print(f"  Technical groups (C):   {n_groups}")
        print(f"  Guides (G):             {n_guides:,}")
        if distribution == 'multinomial':
            print(f"  Categories (K):         {n_categories}")
        print(f"  Distribution:           {distribution}")
        print(f"  Sparsity:               {sparsity*100:.0f}% zeros")
        print(f"  Trans function:         {function_type}")

        print(f"\n{'Step':<20} {'RAM (GB)':<12} {'VRAM (GB)':<12}")
        print("-" * 50)
        print(f"{'fit_ntc':<20} {ram_tech:>10.1f}   {vram_tech:>10.1f}")
        print(f"{'fit_cis':<20} {ram_cis:>10.1f}   {vram_cis:>10.1f}")
        print(f"{'fit_trans':<20} {ram_trans:>10.1f}   {vram_trans:>10.1f}")
        print("-" * 50)
        print(f"{'Peak (minimum)':<20} {min_ram:>10.1f}   {min_vram:>10.1f}")
        print(f"{'Recommended':<20} {rec_ram:>10.1f}   {rec_vram_rounded:>10.0f}")

        print("\n" + "=" * 70)
        print("Hardware Recommendations:")
        print("=" * 70)

        # RAM recommendation
        if rec_ram < 16:
            ram_rec = "16 GB (basic workstation)"
        elif rec_ram < 32:
            ram_rec = "32 GB (standard workstation)"
        elif rec_ram < 64:
            ram_rec = "64 GB (high-end workstation)"
        elif rec_ram < 128:
            ram_rec = "128 GB (server/cluster node)"
        else:
            ram_rec = f"{int(rec_ram):,} GB (high-memory server)"

        print(f"RAM:  {ram_rec}")

        # GPU recommendation
        gpu_options = {
            8: "RTX 3070 / RTX 4060 Ti (8 GB)",
            12: "RTX 3060 / RTX 4070 (12 GB)",
            16: "V100 / RTX 4080 (16 GB)",
            24: "RTX 3090 / RTX 4090 (24 GB)",
            32: "A100 40GB / RTX 6000 Ada (32-48 GB)",
            40: "A100 40GB (40 GB)",
            48: "RTX 6000 Ada (48 GB)",
            80: "A100 80GB (80 GB)",
        }

        for vram, gpu in gpu_options.items():
            if rec_vram_rounded <= vram:
                print(f"GPU:  {gpu}")
                break
        else:
            print(f"GPU:  {rec_vram_rounded:.0f} GB (custom/multi-GPU)")

        print("\n" + "=" * 70)

        # Warnings
        if min_vram > 80:
            print("⚠️  WARNING: Dataset may be too large for single GPU.")
            print("   Consider: feature filtering, CPU mode, or multi-GPU setup")
        elif min_vram > 40:
            print("⚠️  Note: Large dataset - ensure cluster access (A100 80GB recommended)")

        if sparsity < 0.5:
            print("⚠️  Warning: Low sparsity - consider filtering low-count features")

        print("=" * 70)

    return results


def interactive_calculator():
    """Interactive command-line calculator."""
    print("\n" + "=" * 70)
    print("bayesDREAM Memory Requirements Calculator (Interactive)")
    print("=" * 70 + "\n")

    # Get user inputs
    try:
        n_features = int(input("Number of features (genes/junctions): "))
        n_cells = int(input("Number of cells: "))
        n_groups = int(input("Number of technical groups (default 2): ") or "2")

        dist = input("Distribution (negbinom/multinomial/binomial/normal) [negbinom]: ").strip() or "negbinom"

        n_categories = None
        if dist == 'multinomial':
            n_categories = int(input("Number of categories (e.g., acceptors per donor): "))

        sparsity_input = input("Sparsity (fraction of zeros, 0-1) [0.85]: ").strip()
        sparsity = float(sparsity_input) if sparsity_input else 0.85

        n_guides_input = input(f"Number of guides [auto: ~{n_features//20}]: ").strip()
        n_guides = int(n_guides_input) if n_guides_input else None

        func_type = input("Trans function (single_hill/additive_hill/polynomial) [additive_hill]: ").strip()
        func_type = func_type if func_type else "additive_hill"

        use_all_str = input("Use all cells for fit_ntc (high MOI mode)? (y/n) [n]: ").strip().lower()
        use_all_cells = use_all_str in ('y', 'yes')

        print("\nCalculating...\n")

        estimate_memory(
            n_features=n_features,
            n_cells=n_cells,
            n_groups=n_groups,
            n_guides=n_guides,
            n_categories=n_categories,
            sparsity=sparsity,
            distribution=dist,
            function_type=func_type,
            use_all_cells=use_all_cells,
            verbose=True
        )

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError or interrupted: {e}")
        return


if __name__ == "__main__":
    import sys

    # Check if interactive mode or example mode
    if len(sys.argv) == 1:
        # Interactive mode
        interactive_calculator()
    elif sys.argv[1] == "--examples":
        # Show examples
        print("\nExample 1: Small dataset (testing)")
        print("-" * 70)
        estimate_memory(
            n_features=5000,
            n_cells=10000,
            n_groups=2,
            n_guides=100,
            sparsity=0.85,
            verbose=True
        )

        print("\n" * 2)
        print("Example 2: Medium dataset (typical)")
        print("-" * 70)
        estimate_memory(
            n_features=20000,
            n_cells=30000,
            n_groups=2,
            n_guides=500,
            sparsity=0.85,
            verbose=True
        )

        print("\n" * 2)
        print("Example 3: Large dataset (published study)")
        print("-" * 70)
        estimate_memory(
            n_features=30000,
            n_cells=50000,
            n_groups=2,
            n_guides=1000,
            sparsity=0.85,
            verbose=True
        )

        print("\n" * 2)
        print("Example 4: Multinomial (splicing donor usage)")
        print("-" * 70)
        estimate_memory(
            n_features=5000,
            n_cells=50000,
            n_groups=2,
            n_categories=10,
            sparsity=0.70,
            distribution='multinomial',
            verbose=True
        )

        print("\n" * 2)
        print("Example 5: High MOI mode (use_all_cells=True)")
        print("-" * 70)
        estimate_memory(
            n_features=30000,
            n_cells=50000,
            n_groups=2,
            n_guides=1000,
            sparsity=0.85,
            use_all_cells=True,  # Compare fit_ntc to Example 3
            verbose=True
        )
    else:
        print("Usage:")
        print("  python memory_calculator.py              # Interactive mode")
        print("  python memory_calculator.py --examples   # Show examples")
