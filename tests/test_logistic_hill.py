"""Tests for fit_trans(function_type='logistic_hill')."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

bayesDREAM = pytest.importorskip(
    "bayesDREAM",
    reason="bayesDREAM or its torch dependency not available",
    exc_type=ImportError,
).bayesDREAM


def _make_data(n_cells=200, seed=0):
    """Graded knockdown of a cis gene; gene_up follows it, gene_null does not."""
    rng = np.random.default_rng(seed)
    kd = np.array([1.0, 1.0, 0.5, 0.2, 0.05])
    guide_idx = np.arange(n_cells) % len(kd)
    meta = pd.DataFrame({
        "cell": [f"cell_{i}" for i in range(n_cells)],
        "guide": [f"guide_{g}" for g in guide_idx],
        "target": np.where(guide_idx < 2, "ntc", "GFI1B"),
        "cell_line": np.tile(["A", "B"], n_cells // 2),
        "sum_factor": rng.uniform(0.8, 1.2, n_cells),
    })
    x = 40.0 * kd[guide_idx]
    counts = pd.DataFrame(
        np.vstack([
            rng.poisson(x),
            rng.poisson(5.0 + 30.0 * x / (x + 8.0)),
            rng.poisson(20.0, n_cells),
        ]),
        index=["GFI1B", "gene_up", "gene_null"],
        columns=meta["cell"],
    )
    return meta, counts, x


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory):
    import torch

    meta, counts, x = _make_data()
    model = bayesDREAM(meta=meta, counts=counts, cis_gene="GFI1B",
                       output_dir=str(tmp_path_factory.mktemp("logistic")),
                       label="logistic_test", device="cpu")
    model.set_technical_groups(["cell_line"])
    model.fit_technical(niters=100, nsamples=10, sum_factor_col="sum_factor")
    x_true = torch.tensor(x, dtype=torch.float32)
    model.x_true, model.x_true_type = x_true, "point"
    model.log2_x_true, model.log2_x_true_type = torch.log2(x_true), "point"
    return model


@pytest.mark.parametrize("guide", ["block_mvn", "normal"])
def test_logistic_hill_shapes(fitted_model, guide):
    fitted_model.fit_trans(sum_factor_col="sum_factor", function_type="logistic_hill",
                           niters=200, nsamples=20, n_components=2, logistic_guide=guide)
    ps = fitted_model.get_modality("gene").posterior_samples_trans
    T = 2
    assert ps["b"].shape == (20, T)
    assert ps["o_y"].shape == (20, T)
    for name in ("V", "k", "n", "log2_K", "hill_n"):
        assert ps[name].shape == (20, 2, T)
    assert ps["tau"].shape == (20,)
    # midpoints are ordered and slopes positive
    assert (ps["k"][:, 1] >= ps["k"][:, 0]).all()
    assert (ps["n"] > 0).all()
    assert ps["link"] == "log2"


def test_logistic_hill_detects_effect(fitted_model):
    fitted_model.fit_trans(sum_factor_col="sum_factor", function_type="logistic_hill",
                           niters=3000, nsamples=100, lr=2e-2)
    df = fitted_model.save_trans_summary()
    assert (df["function_type"] == "logistic_hill").all()
    row = df.set_index("feature")
    assert "observed_log2fc_median" in row.columns
    assert row.loc["gene_up", "observed_log2fc_median"] > 1.0
    assert row.loc["gene_null", "observed_log2fc_median"] < row.loc["gene_up", "observed_log2fc_lower"]


def test_logistic_hill_rejects_multinomial(fitted_model):
    with pytest.raises(NotImplementedError):
        fitted_model.fit_trans(function_type="logistic_hill", distribution="multinomial",
                               min_denominator=0, niters=1)
