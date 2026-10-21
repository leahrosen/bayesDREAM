"""
fit_trans x-input options: x_level ('cell'|'guide') x x_uncertainty ('median'|'sample').

x_uncertainty='sample' conditions each SVI step on a fresh fit_cis posterior draw
(fixed data, never a latent of _model_y), so trans fits must handle x uncertainty
rather than fit it.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyro")
bayesDREAM = pytest.importorskip(
    "bayesDREAM", reason="bayesDREAM or its torch dependency not available",
    exc_type=ImportError,
).bayesDREAM

from test_system_quick_pipeline import _make_quick_test_data


@pytest.fixture(scope="module")
def cis_fitted(tmp_path_factory):
    meta, counts = _make_quick_test_data(n_cells=50)
    outdir = tmp_path_factory.mktemp("x_unc")
    model = bayesDREAM(meta=meta, counts=counts, cis_gene="GFI1B",
                       output_dir=str(outdir), label="x_unc", device="cpu")
    model.set_technical_groups(["cell_line"])
    model.fit_ntc(niters=100, nsamples=10, sum_factor_col="sum_factor")
    model.fit_cis(niters=100, nsamples=20, sum_factor_col="sum_factor")
    return model


def _idx(model):
    return torch.arange(len(model.meta), dtype=torch.long)


def test_build_x_input_cell_median_is_x_true(cis_fitted):
    x_point, draws = cis_fitted._trans_fitter._build_x_input(_idx(cis_fitted))
    assert draws is None
    assert torch.equal(x_point, cis_fitted.x_true)


def test_build_x_input_cell_sample(cis_fitted):
    x_point, draws = cis_fitted._trans_fitter._build_x_input(
        _idx(cis_fitted), x_level="cell", x_uncertainty="sample")
    N = len(cis_fitted.meta)
    assert draws.shape == (20, N)
    assert torch.allclose(draws.median(dim=0).values, x_point.cpu())


def test_build_x_input_guide_constant_within_guide(cis_fitted):
    x_point, draws = cis_fitted._trans_fitter._build_x_input(
        _idx(cis_fitted), x_level="guide", x_uncertainty="sample")
    codes = cis_fitted.meta["guide_code"].values
    for g in np.unique(codes):
        cols = draws[:, codes == g]
        # every cell of a guide gets the same guide-level draw
        assert torch.allclose(cols, cols[:, :1].expand_as(cols))
    x_eff_g = cis_fitted.posterior_samples_cis["x_eff_g"].reshape(20, -1)
    assert torch.allclose(draws, x_eff_g[:, torch.as_tensor(codes, dtype=torch.long)])


def test_build_x_input_rejects_stale_draws(cis_fitted):
    m = cis_fitted
    saved = m.x_true
    try:
        m.set_x_true(saved * 1.5)
        with pytest.raises(ValueError, match="out of sync"):
            m._trans_fitter._build_x_input(_idx(m), x_uncertainty="sample")
        # median mode is unaffected
        m._trans_fitter._build_x_input(_idx(m))
    finally:
        m.x_true = saved


def test_permute_x_true_keeps_draws_in_sync(cis_fitted):
    import copy
    m = copy.deepcopy(cis_fitted)
    m.adjust_ntc_sum_factor(covariates=["cell_line"])
    np.random.seed(0)
    m.permute_x_true(covariates=["cell_line"])
    # sampling still passes the staleness check after permutation
    m._trans_fitter._build_x_input(_idx(m), x_uncertainty="sample")
    with pytest.raises(ValueError, match="permute_x_true"):
        m._trans_fitter._build_x_input(_idx(m), x_level="guide")


@pytest.mark.parametrize("x_level,x_uncertainty,per_step", [
    ("cell", "sample", 1),
    ("cell", "sample", 3),
    ("guide", "sample", 2),
    ("guide", "median", 1),
])
def test_fit_trans_runs(cis_fitted, x_level, x_uncertainty, per_step):
    import copy
    m = copy.deepcopy(cis_fitted)
    m.fit_trans(sum_factor_col="sum_factor", function_type="additive_hill",
                niters=60, nsamples=5, checkpoint_interval=None,
                x_level=x_level, x_uncertainty=x_uncertainty,
                x_draws_per_step=per_step)
    assert m.posterior_samples_trans is not None
    assert np.all(np.isfinite(m.losses_trans))
    assert m.trans_x_input["x_level"] == x_level
    assert m.trans_x_input["x_uncertainty"] == x_uncertainty


def test_trans_x_input_roundtrip(cis_fitted):
    import copy
    m = copy.deepcopy(cis_fitted)
    m.fit_trans(sum_factor_col="sum_factor", function_type="single_hill",
                niters=40, nsamples=5, checkpoint_interval=None,
                x_uncertainty="sample")
    m.save_trans_fit()
    m2 = copy.deepcopy(cis_fitted)
    m2.trans_x_input = None
    m2.load_trans_fit()
    assert m2.trans_x_input == {"x_level": "cell", "x_uncertainty": "sample",
                                "x_draws_per_step": 1}
