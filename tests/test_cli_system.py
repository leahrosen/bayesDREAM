"""System-level tests for bayesDREAM CLI.

Includes:
1) Local CLI execution via `python -m bayesDREAM`
2) Optional containerized CLI execution via Apptainer + GHCR/local SIF
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_quick_test_data(n_cells: int = 50, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    guides = np.repeat([f"guide_{i}" for i in range(5)], n_cells // 5)
    guide_to_target = {
        "guide_0": "ntc",
        "guide_1": "ntc",
        "guide_2": "GFI1B",
        "guide_3": "GFI1B",
        "guide_4": "GFI1B",
    }

    meta = pd.DataFrame(
        {
            "cell": [f"cell_{i}" for i in range(n_cells)],
            "guide": guides,
            "target": [guide_to_target[g] for g in guides],
            "cell_line": np.tile(["A", "B"], n_cells // 2),
            "sum_factor": rng.uniform(0.8, 1.2, n_cells),
        }
    )

    gene_names = ["GFI1B", "gene_1", "gene_2"]
    counts = pd.DataFrame(
        rng.poisson(50, (3, n_cells)).astype(np.int64),
        index=gene_names,
        columns=meta["cell"],
    )

    return meta, counts


def _write_cli_fixture(
    tmp_path: Path, label: str = "cli_quick", container_root: str | None = None
) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)

    meta, counts = _make_quick_test_data()

    meta_path = data_dir / "meta.csv"
    counts_path = data_dir / "counts.csv"
    config_path = data_dir / "config.yaml"

    meta.to_csv(meta_path, index=False)
    counts.to_csv(counts_path)

    if container_root is None:
        cfg_meta_path = str(meta_path)
        cfg_counts_path = str(counts_path)
        cfg_output_dir = str(out_dir)
    else:
        # tmp_path is bind-mounted to container_root in the containerized test.
        cfg_meta_path = f"{container_root}/data/meta.csv"
        cfg_counts_path = f"{container_root}/data/counts.csv"
        cfg_output_dir = f"{container_root}/out"

    config = {
        "data": {
            "meta": cfg_meta_path,
            "counts": cfg_counts_path,
            "counts_read_csv_kwargs": {"index_col": 0},
        },
        "model": {
            "modality_name": "gene",
            "cis_gene": "GFI1B",
            "output_dir": cfg_output_dir,
            "label": label,
            "random_seed": 42,
            "cores": 1,
            "device": "cpu",
        },
        "run": {"steps": ["technical", "cis", "trans"]},
        "technical": {
            "set_technical_groups": ["cell_line"],
            "fit": {
                "sum_factor_col": "sum_factor",
                "niters": 100,
                "nsamples": 10,
            },
            "save": True,
        },
        "cis": {
            "load_technical": {"enabled": True, "args": {"use_posterior": False}},
            "fit": {
                "sum_factor_col": "sum_factor",
                "niters": 100,
                "nsamples": 10,
            },
            "save": True,
        },
        "trans": {
            "load_technical": {"enabled": True, "args": {"use_posterior": False}},
            "load_cis": {"enabled": True, "args": {"use_posterior": False}},
            "fit": {
                "modality_name": "gene",
                "sum_factor_col": "sum_factor",
                "function_type": "additive_hill",
                "niters": 100,
                "nsamples": 10,
            },
            "save": True,
        },
    }

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config_path, out_dir / label


def test_cli_run_local(tmp_path: Path):
    """Run full pipeline through local CLI entrypoint."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    pytest.importorskip("typer")

    config_path, run_out = _write_cli_fixture(tmp_path, label="local_cli")

    cmd = [sys.executable, "-m", "bayesDREAM", "run", "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    assert result.returncode == 0, (
        f"CLI failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    # Validate core artifacts from all three stages
    assert (run_out / "alpha_x_prefit.pt").exists()
    assert (run_out / "x_true.pt").exists()
    assert (run_out / "posterior_samples_trans_gene.pt").exists()


def test_cli_run_apptainer_ghcr(tmp_path: Path):
    """Optional containerized CLI test via Apptainer.

    Enabled only when:
    - BAYESDREAM_RUN_CONTAINER_TESTS=1
    - apptainer is available
    - BAYESDREAM_GHCR_IMAGE is set OR local bayesdream_cpu.sif exists
    """
    if os.environ.get("BAYESDREAM_RUN_CONTAINER_TESTS") != "1":
        pytest.skip("Set BAYESDREAM_RUN_CONTAINER_TESTS=1 to enable container test")

    apptainer_bin = shutil.which("apptainer") or shutil.which("singularity")
    if apptainer_bin is None:
        pytest.skip("Neither apptainer nor singularity executable found")

    image_ref = os.environ.get("BAYESDREAM_GHCR_IMAGE")
    local_sif = Path("bayesdream_cpu.sif")
    if image_ref is None:
        if local_sif.exists():
            image_ref = str(local_sif.resolve())
        else:
            pytest.skip(
                "Set BAYESDREAM_GHCR_IMAGE (e.g. oras://ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-latest) "
                "or provide ./bayesdream_cpu.sif"
            )

    config_path, run_out = _write_cli_fixture(
        tmp_path, label="container_cli", container_root="/work"
    )

    container_cfg = "/work/data/config.yaml"
    cmd = [
        apptainer_bin,
        "exec",
        "--bind",
        f"{tmp_path}:/work",
        image_ref,
        "bayesdream",
        "run",
        "--config",
        container_cfg,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    assert result.returncode == 0, (
        f"Container CLI failed with code {result.returncode}\n"
        f"Command: {' '.join(cmd)}\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    assert (run_out / "alpha_x_prefit.pt").exists()
    assert (run_out / "x_true.pt").exists()
    assert (run_out / "posterior_samples_trans_gene.pt").exists()
