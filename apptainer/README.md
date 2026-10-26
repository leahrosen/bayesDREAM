# Apptainer Support

This folder provides CPU, CUDA (NVIDIA), and ROCm (AMD) Apptainer recipes for `bayesDREAM`.

## Build

```bash
apptainer build bayesdream_cpu.sif apptainer/bayesdream_cpu.def
apptainer build bayesdream_cuda.sif apptainer/bayesdream_cuda.def
apptainer build bayesdream_rocm.sif apptainer/bayesdream_rocm.def
```

If your cluster requires rootless builds, use your site's remote builder (if configured):

```bash
apptainer build --remote bayesdream_cpu.sif apptainer/bayesdream_cpu.def
```

## Run Python Workflows

```bash
# CPU
apptainer exec --bind $PWD:/work bayesdream_cpu.sif \
  python /work/your_workflow.py

# NVIDIA GPU
apptainer exec --nv --bind $PWD:/work bayesdream_cuda.sif \
  python /work/your_workflow.py

# AMD GPU
apptainer exec --rocm --bind $PWD:/work bayesdream_rocm.sif \
  python /work/your_workflow.py
```

Run mode forwards arguments to the container command:

```bash
apptainer run --bind $PWD:/work bayesdream_cpu.sif python /work/your_workflow.py
```

## Snakemake integration

Use the generated `.sif` directly in rule definitions:

```python
rule bayesdream_pipeline:
    input:
        cfg="config.yaml"
    output:
        done="results/.done"
    container:
        "bayesdream_cpu.sif"
    shell:
        "python scripts/run_analysis.py --config {input.cfg} && touch {output.done}"
```

Run with container support enabled:

```bash
# CPU
snakemake --use-singularity --singularity-args "--bind $PWD:/work"

# NVIDIA GPU
snakemake --use-singularity --singularity-args "--nv --bind $PWD:/work"

# AMD GPU
snakemake --use-singularity --singularity-args "--rocm --bind $PWD:/work"
```

## Notes

- CPU images are the most portable baseline.
- CUDA images require NVIDIA drivers and `apptainer exec --nv`.
- ROCm images require AMD ROCm-compatible host setup and `apptainer exec --rocm`.

## GitHub Actions + GHCR

A workflow is provided at `.github/workflows/apptainer-ghcr.yml` that builds and publishes on each push.

Published refs:

- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:cuda-amd64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:rocm-amd64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:cuda-amd64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:rocm-amd64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-latest` (default branch only)
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-latest` (default branch only)
- `ghcr.io/<owner>/<repo>-apptainer:cuda-amd64-latest` (default branch only)
- `ghcr.io/<owner>/<repo>-apptainer:rocm-amd64-latest` (default branch only)
