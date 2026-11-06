# bayesDREAM Examples

This directory contains example scripts demonstrating how to use bayesDREAM with the new save/load system.

## Pipeline Scripts

Run the pipeline in stages, saving results at each step:

### 1. Technical Fit

```bash
python run_technical_example.py \
    --outdir ./results \
    --label GFI1B_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --covariates cell_line \
    --cores 4
```

**Output**:
- `results/GFI1B_analysis/alpha_x_prefit.pt`
- `results/GFI1B_analysis/alpha_y_prefit.pt`
- `results/GFI1B_analysis/posterior_samples_technical.pt`

### 2. Cis Fit

```bash
python run_cis_example.py \
    --outdir ./results \
    --label GFI1B_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --use_posterior \
    --cores 4
```

**Output**:
- `results/GFI1B_analysis/x_true.pt`
- `results/GFI1B_analysis/posterior_samples_cis.pt`

### 3. Trans Fit

```bash
python run_trans_example.py \
    --outdir ./results \
    --label GFI1B_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --function_type additive_hill \
    --modality gene \
    --use_posterior \
    --cores 4
```

**Output**:
- `results/GFI1B_analysis/posterior_samples_trans.pt`
- `results/GFI1B_analysis/posterior_samples_trans_gene.pt` (if per-modality)

## Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--outdir` | Output directory | Required |
| `--label` | Analysis label | Required |
| `--meta` | Path to metadata CSV | Required |
| `--counts` | Path to counts CSV | Required |
| `--cis_gene` | Cis gene name | `GFI1B` |
| `--cores` | Number of CPU cores | `1` |

### Technical Fit (`run_technical_example.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--covariates` | Technical covariates (space-separated) | `cell_line` |

### Cis Fit (`run_cis_example.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_posterior` | Use full posterior samples | False (uses point estimates) |

### Trans Fit (`run_trans_example.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--function_type` | Function type: `single_hill`, `additive_hill` | `additive_hill` |
| `--modality` | Modality to fit trans on | `gene` |
| `--use_posterior` | Use full posterior samples | False |

## Complete Example

Run all three stages:

```bash
# Stage 1: Technical
python run_technical_example.py \
    --outdir ./results \
    --label my_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --cores 8

# Stage 2: Cis (loads technical fit)
python run_cis_example.py \
    --outdir ./results \
    --label my_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --use_posterior \
    --cores 8

# Stage 3: Trans (loads technical and cis fits)
python run_trans_example.py \
    --outdir ./results \
    --label my_analysis \
    --meta data/meta.csv \
    --counts data/counts.csv \
    --cis_gene GFI1B \
    --function_type additive_hill \
    --use_posterior \
    --cores 8
```

## Benefits of Staged Pipeline

1. **Modularity**: Run stages independently
2. **Efficiency**: Reuse expensive fits
3. **Experimentation**: Try different parameters in later stages without refitting early stages
4. **Scalability**: Submit different stages to different compute resources
5. **Debugging**: Easier to identify and fix issues at specific stages

## Advanced: Modality-Specific Save/Load

The save/load methods support modality-specific parameters for finer control:

```bash
# In Python script:
# Save only specific modalities
model.save_technical_fit(modalities=['gene', 'atac'])

# Load only what you need
model.load_technical_fit(modalities=['gene'])

# Skip model-level backward compatibility params
model.save_trans_fit(modalities=['atac'], save_model_level=False)
```

See `docs/SAVE_LOAD_GUIDE.md` for detailed examples and use cases.

## Notes

- Each script automatically handles random seed setting for reproducibility
- The `--use_posterior` flag controls whether to load full posterior samples or point estimates
- Point estimates (posterior means) are faster and use less memory
- Full posteriors are needed for proper uncertainty propagation

For more details, see `docs/SAVE_LOAD_GUIDE.md`.
