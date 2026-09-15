# Memory Requirements Guide

This guide helps you estimate RAM and VRAM requirements for fitting bayesDREAM models. Use the formulas below to calculate memory needs based on your dataset characteristics.

## Quick Reference

| Step | Typical RAM | Typical VRAM | Bottleneck |
|------|-------------|--------------|------------|
| **fit_ntc** | 2-4× data size | 4-8× data size | Guide (AutoIAFNormal) |
| **fit_cis** | 1-2× data size | 2-4× data size | Data tensors |
| **fit_trans** | 3-5× data size | 6-10× data size | Gradients + posteriors |

**Rule of thumb**: For negbinom with **T genes**, **N cells**, plan for:
- **Minimum**: 32 GB RAM, 16 GB VRAM
- **Recommended**: 64 GB RAM, 32 GB VRAM
- **Large datasets** (>20K genes): 128 GB RAM, 80 GB VRAM

---

## Input Parameters

To estimate memory requirements, you'll need:

| Parameter | Description | Example |
|-----------|-------------|---------|
| **T** | Number of features (genes, junctions, etc.) | 30,000 |
| **N** | Number of cells | 50,000 |
| **C** | Number of technical groups | 2 |
| **K** | Number of categories (multinomial only) | 10 |
| **G** | Number of guides | 1,000 |
| **sparsity** | Fraction of zeros in counts matrix | 0.85 |
| **distribution** | negbinom, multinomial, binomial, normal, studentt | negbinom |

---

## 1. fit_ntc

### Fitting Modes

bayesDREAM offers two modes for technical fitting:

#### Standard Mode (use_all_cells=False, default)
Fits only on **NTC cells** (~20-50% of total cells). This is the recommended approach for most experiments.

```python
model.fit_ntc(use_all_cells=False)  # Default
```

**When to use**:
- Standard low MOI experiments
- Technical groups correlate with perturbation effects (e.g., CRISPRi vs CRISPRa cell lines)
- Separate technical correction per cis gene

#### High MOI Mode (use_all_cells=True)
Fits on **ALL cells** (100% of dataset). Use for high MOI experiments where technical effects are batch/lane specific.

```python
model.fit_ntc(use_all_cells=True)  # High MOI mode
```

**When to use**:
- High MOI experiments (multiple guides per cell)
- Technical variation is batch/lane specific, independent of perturbations
- Want to run fit_ntc once per dataset (not per cis gene)

**When NOT to use**:
- Technical groups correlate with cis gene expression
- CRISPRi vs CRISPRa as technical groups
- Low MOI with clear NTC vs perturbed distinction

**Memory impact**: Data component increases by ~2.5× (all cells vs 40% NTC), but total memory increase is more modest (~15-20%) due to fixed overheads.

### Data Memory

**N_ntc** = number of cells used for fitting:
- **Standard mode**: N_ntc ≈ 0.4 × N (40% NTC cells, typical)
- **High MOI mode (use_all_cells=True)**: N_ntc = N (100% of cells)

| Component | Size Formula | Notes |
|-----------|--------------|-------|
| **Counts (dense)** | `T × N_ntc × 4 bytes` | Float32 |
| **Counts (sparse)** | `T × N_ntc × 4 × (1 - sparsity)` | CSR format |
| **Sum factors** | `N_ntc × 4 bytes` | Per-cell normalization |
| **Metadata** | `N_ntc × 8 bytes × n_columns` | Negligible |
| **Denominator** (binomial) | `T × N_ntc × 4 bytes` | Same as counts |

**Total data memory (dense)**:
```
Data_GB = (T × N_ntc × 4) / 1e9
```

**Total data memory (sparse, 85% zeros)**:
```
Data_GB = (T × N_ntc × 4 × 0.15) / 1e9
```

**High MOI mode**: Data component increases by 2.5× (N_ntc = N instead of 0.4 × N), but total memory increase is ~15-20% due to fixed overheads (guide, gradients, etc.).

### Guide Memory

The variational guide stores learned parameters. **This is the bottleneck for large datasets.**

#### AutoNormal Guide (mean-field, memory-efficient)
```
n_latent = (C - 1 + 2) × T  # For negbinom: alpha, o_y, mu_ntc

Guide_GB = (n_latent × 2 × 4) / 1e9  # Mean + std per latent
```

**Example (30K genes, C=2)**:
- n_latent = (2-1 + 2) × 30,000 = 90,000
- Guide = 90,000 × 2 × 4 / 1e9 = **0.72 GB** ✓

#### AutoIAFNormal Guide (autoregressive, memory-intensive)
```
n_latent = (C - 1 + 2) × T

Guide_GB = (n_latent² × 4 × 3 × 1.5) / 1e9  # Transformation matrices
```

**Example (30K genes, C=2)**:
- n_latent = 90,000
- Guide = 90,000² × 4 × 3 × 1.5 / 1e9 = **146 GB** ✗ (too large!)

**bayesDREAM automatic selection**:
- Estimates IAF memory requirement
- Falls back to AutoNormal if > (total_VRAM - 10 GB)
- For large datasets, **always uses AutoNormal** (memory-efficient)

### fit_ntc Total Memory

| Component | RAM | VRAM (GPU) |
|-----------|-----|------------|
| Data tensors | 1-2 GB | 2-4 GB |
| Guide (AutoNormal) | 0.5-1 GB | 0.5-1 GB |
| Gradients + optimizer | 1-2 GB | 2-4 GB |
| Posterior samples (1000) | 2-4 GB | - (stored on CPU) |
| **Total** | **5-9 GB** | **5-9 GB** |

**Scaling formula (AutoNormal)**:
```
RAM_technical_GB = 4 + (T × N_ntc × 4 / 1e9) × 3
VRAM_technical_GB = 6 + (T × N_ntc × 4 / 1e9) × 2
```

---

## 2. fit_cis (Single gene fitting)

Fits one cis gene across all cells (NTC + perturbed).

### Memory Components

| Component | Size Formula | Notes |
|-----------|--------------|-------|
| **Cis gene counts** | `1 × N × 4 bytes` | Tiny (~200 KB for 50K cells) |
| **Guide assignment** (high MOI) | `N × G × 4 bytes` | Binary or continuous |
| **x_true estimates** | `G × 4 bytes` | One value per guide |
| **Sum factors** | `N × 4 bytes` | Per-cell |
| **Posteriors** | `1000 × G × 4 bytes` | ~4 KB per guide |

**Total**:
```
RAM_cis_GB = 4 + (N × G × 4) / 1e9
VRAM_cis_GB = 4 + (N × G × 4) / 1e9
```

**Example (50K cells, 1000 guides)**:
```
RAM = 4 + (50,000 × 1,000 × 4) / 1e9 = 4.2 GB
VRAM = 4.2 GB
```

**Scaling**: Linear in N × G. High MOI mode requires more memory for guide assignments.

---

## 3. fit_trans (Genome-wide dose-response)

Fits dose-response curves for **all trans genes** simultaneously. **Most memory-intensive step.**

### Data Memory

| Component | Size Formula | Notes |
|-----------|--------------|-------|
| **Trans counts (dense)** | `T × N × 4 bytes` | ~6 GB for 30K × 50K |
| **Trans counts (sparse)** | `T × N × 4 × (1 - sparsity)` | ~0.9 GB at 85% sparsity |
| **x_true** | `N × 4 bytes` | Cis expression per cell |
| **Sum factors** | `N × 4 bytes` | Per-cell normalization |

### Model Parameters

Number of parameters depends on **function type**:

| Function Type | Parameters per Gene | Total Parameters |
|---------------|---------------------|------------------|
| **single_hill** | 5 (β₀, β, K, n, σ) | 5 × T |
| **additive_hill** | 7 (β₀, β⁺, K⁺, n⁺, β⁻, K⁻, n⁻) | 7 × T |
| **polynomial** (degree 6) | 8 (β₀...β₆, σ) | 8 × T |

**Gradient memory** (largest component):
```
n_params = function_params × T
Gradient_GB = (n_params × 4 × 3) / 1e9  # Params + gradients + momentum
```

**Example (30K genes, additive_hill)**:
```
n_params = 7 × 30,000 = 210,000
Gradient = 210,000 × 4 × 3 / 1e9 = 2.5 GB
```

### Posterior Samples

```
Posterior_GB = (nsamples × n_params × 4) / 1e9
```

**Example (1000 samples, 210K params)**:
```
Posterior = 1,000 × 210,000 × 4 / 1e9 = 0.84 GB
```

### fit_trans Total Memory

| Component | RAM (GB) | VRAM (GB) |
|-----------|----------|-----------|
| Data (sparse) | 1-2 | 1-2 |
| Parameters | 1-2 | 1-2 |
| Gradients | 3-6 | 3-6 |
| Posterior samples | 1-2 | - (CPU) |
| Overhead | 2-4 | 2-4 |
| **Total** | **8-16** | **7-14** |

**Scaling formula**:
```
RAM_trans_GB = 10 + (T × N × 4 × (1 - sparsity) / 1e9) × 2
VRAM_trans_GB = 8 + (T × function_params × 4 / 1e9) × 5
```

---

## Distribution-Specific Requirements

### Negative Binomial (negbinom)

**Most common for gene expression counts.**

| Step | RAM | VRAM |
|------|-----|------|
| fit_ntc | 5-10 GB | 5-10 GB |
| fit_cis | 4-6 GB | 4-6 GB |
| fit_trans | 10-20 GB | 8-16 GB |

**Memory scales with**: T × N (data) + T² (guide, if IAF)

### Multinomial (donor/acceptor usage)

**3D data**: (features, cells, categories)

**Additional memory**:
```
Multinomial_extra_GB = (T × N × K × 4) / 1e9
```

| Step | RAM | VRAM | Notes |
|------|-----|------|-------|
| fit_ntc | 8-15 GB | 10-20 GB | K categories increase params |
| fit_cis | Same as negbinom | Same | Cis gene is still 1D |
| fit_trans | Not yet implemented | - | - |

**Scaling**: T × N × K (3D array dominates)

**Example (5K donors, 50K cells, K=10)**:
```
Data = 5,000 × 50,000 × 10 × 4 / 1e9 = 10 GB
```

### Binomial (exon skipping PSI)

**Requires denominator array** (same size as counts).

| Step | RAM | VRAM | Notes |
|------|-----|------|-------|
| fit_ntc | 6-12 GB | 6-12 GB | 2× negbinom (counts + denom) |
| fit_cis | Same as negbinom | Same | - |
| fit_trans | 12-24 GB | 10-20 GB | 2× negbinom |

**Memory**: ~2× negbinom (numerator + denominator)

### Normal / Student's t

**Continuous values** (e.g., SpliZ scores).

| Step | RAM | VRAM | Notes |
|------|-----|------|-------|
| fit_ntc | 5-10 GB | 5-10 GB | Same as negbinom |
| fit_cis | Same as negbinom | Same | - |
| fit_trans | 10-20 GB | 8-16 GB | Similar to negbinom |

**Memory**: Similar to negbinom (same data structure)

---

## Example Calculations

### Example 1: Small Dataset (Testing)
```
T = 5,000 genes
N = 10,000 cells
C = 2 technical groups
G = 100 guides
sparsity = 85%
```

**fit_ntc**:
- N_ntc = 5,000 cells (50% NTC)
- Data (sparse) = 5,000 × 5,000 × 4 × 0.15 / 1e9 = 0.15 GB
- Guide (AutoNormal) = (3 × 5,000) × 2 × 4 / 1e9 = 0.12 GB
- **Total: 4-6 GB RAM, 4-6 GB VRAM** ✓

**fit_cis**:
- Guide assignment = 10,000 × 100 × 4 / 1e9 = 0.004 GB
- **Total: 4-5 GB RAM, 4-5 GB VRAM** ✓

**fit_trans**:
- Data (sparse) = 5,000 × 10,000 × 4 × 0.15 / 1e9 = 0.3 GB
- Gradients = 7 × 5,000 × 4 × 3 / 1e9 = 0.42 GB
- **Total: 6-10 GB RAM, 6-10 GB VRAM** ✓

**Recommended**: 16 GB RAM, 8 GB VRAM (e.g., RTX 3070)

---

### Example 2: Medium Dataset (Typical)
```
T = 20,000 genes
N = 30,000 cells
C = 2 technical groups
G = 500 guides
sparsity = 85%
```

**fit_ntc**:
- N_ntc = 12,000 cells
- Data (sparse) = 20,000 × 12,000 × 4 × 0.15 / 1e9 = 1.44 GB
- Guide (AutoNormal) = (3 × 20,000) × 2 × 4 / 1e9 = 0.48 GB
- **Total: 6-10 GB RAM, 6-10 GB VRAM** ✓

**fit_cis**:
- Guide assignment = 30,000 × 500 × 4 / 1e9 = 0.06 GB
- **Total: 4-6 GB RAM, 4-6 GB VRAM** ✓

**fit_trans**:
- Data (sparse) = 20,000 × 30,000 × 4 × 0.15 / 1e9 = 3.6 GB
- Gradients = 7 × 20,000 × 4 × 3 / 1e9 = 1.68 GB
- **Total: 12-20 GB RAM, 10-16 GB VRAM** ✓

**Recommended**: 32 GB RAM, 16 GB VRAM (e.g., V100 16GB, RTX 4080)

---

### Example 3: Large Dataset (Published study scale)
```
T = 30,000 genes
N = 50,000 cells
C = 2 technical groups
G = 1,000 guides
sparsity = 85%
```

**fit_ntc**:
- N_ntc = 20,000 cells
- Data (sparse) = 30,000 × 20,000 × 4 × 0.15 / 1e9 = 3.6 GB
- Guide (AutoNormal) = (3 × 30,000) × 2 × 4 / 1e9 = 0.72 GB
- **Total: 8-14 GB RAM, 8-14 GB VRAM** ✓

**fit_cis**:
- Guide assignment = 50,000 × 1,000 × 4 / 1e9 = 0.2 GB
- **Total: 5-8 GB RAM, 5-8 GB VRAM** ✓

**fit_trans**:
- Data (sparse) = 30,000 × 50,000 × 4 × 0.15 / 1e9 = 9 GB
- Gradients = 7 × 30,000 × 4 × 3 / 1e9 = 2.52 GB
- **Total: 16-28 GB RAM, 14-24 GB VRAM** ✓

**Recommended**: 64 GB RAM, 32 GB VRAM (e.g., A100 40GB, RTX 6000 Ada)

---

### Example 4: Very Large Dataset (Perturb-seq atlas)
```
T = 50,000 genes
N = 100,000 cells
C = 4 technical groups
G = 2,000 guides
sparsity = 90%
```

**fit_ntc**:
- N_ntc = 30,000 cells
- Data (sparse) = 50,000 × 30,000 × 4 × 0.10 / 1e9 = 6 GB
- Guide (AutoNormal) = (5 × 50,000) × 2 × 4 / 1e9 = 2 GB
- **Total: 12-20 GB RAM, 12-20 GB VRAM** ✓

**fit_cis**:
- Guide assignment = 100,000 × 2,000 × 4 / 1e9 = 0.8 GB
- **Total: 6-10 GB RAM, 6-10 GB VRAM** ✓

**fit_trans**:
- Data (sparse) = 50,000 × 100,000 × 4 × 0.10 / 1e9 = 20 GB
- Gradients = 7 × 50,000 × 4 × 3 / 1e9 = 4.2 GB
- **Total: 32-50 GB RAM, 28-40 GB VRAM** ✓

**Recommended**: 128 GB RAM, 80 GB VRAM (e.g., A100 80GB)

---

### Example 5: High MOI Mode (use_all_cells=True)
```
T = 30,000 genes
N = 50,000 cells
C = 2 technical groups
G = 1,000 guides
sparsity = 85%
use_all_cells = True  # Compare to Example 3
```

**fit_ntc** (HIGH MOI MODE):
- N_ntc = 50,000 cells (100% of cells, not just NTC)
- Data (sparse) = 30,000 × 50,000 × 4 × 0.15 / 1e9 = **0.9 GB** (vs 0.36 GB in Example 3, 2.5× increase)
- Fixed overhead + guide + gradients ≈ 4.8 GB (same as Example 3)
- **Total: 5.7 GB RAM, 7.1 GB VRAM** (vs 4.9/6.4 GB in Example 3, ~15% increase)

**fit_cis**:
- Same as Example 3: **5-8 GB RAM, 5-8 GB VRAM** ✓

**fit_trans**:
- Same as Example 3: **16-28 GB RAM, 14-24 GB VRAM** ✓

**Recommended**: 32 GB RAM, 16 GB VRAM (same as Example 3)

**Key takeaway**: Using `use_all_cells=True` increases fit_ntc data memory by 2.5×, but total memory increase is modest (~15%) due to fixed overheads. The computational benefit is significant: fit_ntc only needs to run once per dataset instead of once per cis gene.

---

## Memory Optimization Strategies

### 1. Use Sparse Matrices
```python
import scipy.sparse as sp
counts_sparse = sp.csr_matrix(counts_dense)
model = bayesDREAM(meta=meta, counts=counts_sparse, ...)
```

**Savings**: 5-7× reduction at 85% sparsity

### 2. Reduce Posterior Samples
```python
model.fit_ntc(nsamples=500)  # Default: 1000
```

**Savings**: ~50% reduction in posterior storage

### 3. Use Minibatching (experimental)
```python
model.fit_ntc(minibatch_size=100)  # Sample in batches
```

**Savings**: Reduces peak memory during Predictive sampling

### 4. Feature Filtering
```python
# Keep only well-detected genes
keep_genes = (counts > 0).sum(axis=1) >= 100
model = bayesDREAM(meta=meta, counts=counts[keep_genes, :], ...)
```

**Savings**: Linear reduction in T

### 5. Run on CPU (if GPU memory insufficient)
```python
model = bayesDREAM(meta=meta, counts=counts, device='cpu')
```

**Tradeoff**: Slower (5-10×) but unlimited RAM

---

## GPU Selection Guide

| GPU | VRAM | Recommended Dataset | Notes |
|-----|------|---------------------|-------|
| **RTX 3060** | 12 GB | T ≤ 10K, N ≤ 20K | Entry-level |
| **RTX 3070/4070** | 8-12 GB | T ≤ 15K, N ≤ 30K | Consumer |
| **V100** | 16 GB | T ≤ 25K, N ≤ 50K | Standard cluster |
| **RTX 4080** | 16 GB | T ≤ 25K, N ≤ 50K | High-end consumer |
| **A100 40GB** | 40 GB | T ≤ 40K, N ≤ 80K | Research cluster |
| **A100 80GB** | 80 GB | T ≤ 60K, N ≤ 150K | Large-scale studies |
| **RTX 6000 Ada** | 48 GB | T ≤ 40K, N ≤ 100K | Workstation |

---

## Troubleshooting

### "CUDA out of memory" Error

**During fit_ntc**:
1. Check if AutoNormal fallback activated (see console output)
2. If not, reduce features: `counts = counts[top_genes, :]`
3. Reduce posterior samples: `nsamples=500`

**During fit_trans**:
1. Ensure sparse matrices: `sp.issparse(counts)`
2. Use minibatching: `minibatch_size=100`
3. Reduce function complexity: use `single_hill` instead of `additive_hill`

### "MemoryError" (RAM)

1. Use sparse matrices (most effective)
2. Reduce cell count: subset to key conditions
3. Run in batches: fit trans genes in chunks
4. Increase swap space (not recommended, very slow)

### Slow Performance

1. Ensure GPU is being used: check `model.device == 'cuda'`
2. Reduce `niters` if converging early
3. Use fewer posterior samples: `nsamples=500`
4. Profile with: `nvidia-smi` (GPU) or `htop` (CPU)

---

## Summary Formulas

### Quick Calculator

**Input your values**:
```python
T = 30000  # features
N = 50000  # cells
C = 2      # technical groups
G = 1000   # guides
sparsity = 0.85
```

**Calculate**:
```python
# fit_ntc
N_ntc = N * 0.4  # Assume 40% NTC
data_gb = (T * N_ntc * 4 * (1 - sparsity)) / 1e9
guide_gb = ((C - 1 + 2) * T * 2 * 4) / 1e9
ram_tech = 4 + data_gb * 3
vram_tech = 6 + data_gb * 2

# fit_cis
ram_cis = 4 + (N * G * 4) / 1e9
vram_cis = ram_cis

# fit_trans
data_trans_gb = (T * N * 4 * (1 - sparsity)) / 1e9
grad_gb = (7 * T * 4 * 3) / 1e9
ram_trans = 10 + data_trans_gb * 2
vram_trans = 8 + grad_gb * 2

print(f"fit_ntc: {ram_tech:.1f} GB RAM, {vram_tech:.1f} GB VRAM")
print(f"fit_cis: {ram_cis:.1f} GB RAM, {vram_cis:.1f} GB VRAM")
print(f"fit_trans: {ram_trans:.1f} GB RAM, {vram_trans:.1f} GB VRAM")
print(f"\nRecommended: {max(ram_tech, ram_cis, ram_trans) * 2:.0f} GB RAM, {max(vram_tech, vram_cis, vram_trans) * 1.5:.0f} GB VRAM")
```

---

## Additional Resources

- **AutoNormal vs AutoIAFNormal**: See `bayesDREAM/fitting/ntc.py:1237-1301`
- **Sparse matrix handling**: See `bayesDREAM/fitting/ntc.py:540-547`
- **Memory optimization**: See `bayesDREAM/fitting/ntc.py:980-1035`
- **GPU memory check**: Automatic in `fit_ntc()` (lines 1255-1271)

For questions or issues, see: https://github.com/anthropics/bayesDREAM/issues
