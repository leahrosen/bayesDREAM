"""Lightweight system test for the full bayesDREAM pipeline.

This test runs:
1) fit_technical
2) fit_cis
3) fit_trans
4) save/load roundtrip

It is intentionally small but end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Guard: bayesDREAM transitively imports torch, which may not be available in
# every environment (e.g. wrong GLIBCXX). Skip collection gracefully.
bayesDREAM = pytest.importorskip(
    "bayesDREAM",
    reason="bayesDREAM or its torch dependency not available",
    exc_type=ImportError,
).bayesDREAM


def _make_quick_test_data(n_cells: int = 50, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    if n_cells % 5 != 0:
        raise ValueError("n_cells must be divisible by 5 for this fixture")

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

    # Keep counts numeric and simple: 1 cis + 2 trans genes
    gene_names = ["GFI1B", "gene_1", "gene_2"]
    base_counts = rng.poisson(50, (3, n_cells)).astype(np.int64)

    # Add a small perturbation effect on GFI1B for targeted guides
    targeted_mask = meta["target"].values == "GFI1B"
    base_counts[0, targeted_mask] += rng.poisson(8, targeted_mask.sum())

    counts = pd.DataFrame(base_counts, index=gene_names, columns=meta["cell"])
    return meta, counts


def test_quick_system_pipeline_and_roundtrip(tmp_path):
    """Run full fit pipeline and verify save/load roundtrip."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")

    meta, counts = _make_quick_test_data()

    outdir = tmp_path / "quick_system"
    label = "quick_test"

    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene="GFI1B",
        output_dir=str(outdir),
        label=label,
        device="cpu",
    )

    model.set_technical_groups(["cell_line"])

    # Fast settings for system smoke test.
    model.fit_technical(niters=100, nsamples=10, sum_factor_col="sum_factor")
    model.fit_cis(niters=100, nsamples=10, sum_factor_col="sum_factor")
    model.fit_trans(
        sum_factor_col="sum_factor",
        function_type="additive_hill",
        niters=100,
        nsamples=10,
    )

    # Core fit assertions
    assert model.alpha_x_prefit is not None
    assert model.x_true is not None
    assert model.posterior_samples_trans is not None
    assert model.get_modality("gene").posterior_samples_trans is not None

    # Save all stages
    tech_saved = model.save_technical_fit()
    cis_saved = model.save_cis_fit()
    trans_saved = model.save_trans_fit()

    save_root = outdir / label
    assert save_root.exists()
    assert "alpha_x_prefit" in tech_saved
    assert "x_true" in cis_saved
    assert any(k.startswith("posterior_samples_trans") for k in trans_saved)

    # Load into a fresh model and verify state is reconstructed.
    model_loaded = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene="GFI1B",
        output_dir=str(outdir),
        label=label,
        device="cpu",
    )

    model_loaded.set_technical_groups(["cell_line"])
    model_loaded.load_technical_fit(use_posterior=False)
    model_loaded.load_cis_fit(use_posterior=False)
    model_loaded.load_trans_fit()

    assert model_loaded.alpha_x_prefit is not None
    assert model_loaded.x_true is not None
    assert model_loaded.get_modality("gene").alpha_y_prefit is not None
    assert model_loaded.get_modality("gene").posterior_samples_trans is not None
