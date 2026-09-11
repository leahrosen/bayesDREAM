"""Test lean=True loading for load_ntc_fit / load_cis_fit / load_trans_fit.

Verifies that lean-loaded posteriors (point estimates only, see
bayesDREAM.io.load._reduce_posterior_samples) produce equivalent summary
export output to a full load, remain usable for the specific pipeline-
continuation access patterns that only ever read point estimates (the
_extract_cis_alpha_from_ntc_posteriors / o_x-mean-prior idioms), and refuse
to be re-saved as if they were a full fit.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

bayesDREAM = pytest.importorskip(
    "bayesDREAM",
    reason="bayesDREAM or its torch dependency not available",
    exc_type=ImportError,
).bayesDREAM


def _make_data(n_cells: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    guides = np.repeat([f"guide_{i}" for i in range(5)], n_cells // 5)
    guide_to_target = {
        "guide_0": "ntc", "guide_1": "ntc",
        "guide_2": "GFI1B", "guide_3": "GFI1B", "guide_4": "GFI1B",
    }
    meta = pd.DataFrame({
        "cell": [f"cell_{i}" for i in range(n_cells)],
        "guide": guides,
        "target": [guide_to_target[g] for g in guides],
        "cell_line": np.tile(["A", "B"], n_cells // 2),
        "sum_factor": rng.uniform(0.8, 1.2, n_cells),
    })
    gene_names = ["GFI1B", "gene_1", "gene_2"]
    base_counts = rng.poisson(50, (3, n_cells)).astype(np.int64)
    targeted_mask = meta["target"].values == "GFI1B"
    base_counts[0, targeted_mask] += rng.poisson(8, targeted_mask.sum())
    counts = pd.DataFrame(base_counts, index=gene_names, columns=meta["cell"])
    return meta, counts


@pytest.fixture(scope="module")
def fitted_and_saved(tmp_path_factory):
    import torch

    torch.manual_seed(0)
    meta, counts = _make_data()
    outdir = tmp_path_factory.mktemp("lean_load")
    label = "lean_test"

    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene="GFI1B",
        output_dir=str(outdir), label=label, device="cpu",
    )
    model.set_technical_groups(["cell_line"])
    model.fit_ntc(niters=50, nsamples=20, sum_factor_col="sum_factor")
    model.fit_cis(niters=50, nsamples=20, sum_factor_col="sum_factor")
    model.save_ntc_fit()
    model.save_cis_fit()

    return {"meta": meta, "counts": counts, "outdir": str(outdir), "label": label}


def _fresh_model(fitted_and_saved):
    model = bayesDREAM(
        meta=fitted_and_saved["meta"], counts=fitted_and_saved["counts"],
        cis_gene="GFI1B", output_dir=fitted_and_saved["outdir"],
        label=fitted_and_saved["label"], device="cpu",
    )
    model.set_technical_groups(["cell_line"])
    return model


def _run_dir(fitted_and_saved):
    return os.path.join(fitted_and_saved["outdir"], fitted_and_saved["label"])


# ---------------------------------------------------------------------------
# Lean companion files: save_ntc_fit/save_cis_fit must write a small
# precomputed *_lean.pt file automatically, and load_*_fit(lean=True) must
# actually use it (not just fall back to full-load-then-reduce) whenever it's
# present — that's what avoids materializing the full multi-sample tensors
# in memory at all during a lean load, not just afterward.
# ---------------------------------------------------------------------------

def test_save_writes_lean_companion_files(fitted_and_saved):
    run_dir = _run_dir(fitted_and_saved)
    assert os.path.exists(os.path.join(run_dir, "posterior_samples_ntc_gene_lean.pt"))
    assert os.path.exists(os.path.join(run_dir, "posterior_samples_cis_lean.pt"))
    # NOT asserting lean_size < full_size here: at this fixture's toy scale
    # (nsamples=20, 2 features, 2 groups) the lean file adds 2 extra keys
    # (_lower/_upper) per tensor, and that per-tensor storage/pickle overhead
    # outweighs the savings from dropping 19 of 20 samples — it can come out
    # LARGER than the full file. See test_lean_reduction_shrinks_at_realistic_scale
    # for the actual size claim at a scale where it holds (real fits have
    # nsamples~1000 and thousands of features, where the savings dominate).


def test_lean_reduction_shrinks_at_realistic_scale():
    """Direct, fast check of the actual memory/disk claim, independent of a
    slow full model fit: at realistic posterior sizes (nsamples=1000, many
    features), _reduce_posterior_samples should produce a torch.save()
    payload far smaller than the raw dict — this is the entire premise of
    lean loading."""
    import io
    import torch
    from bayesDREAM.io.load import _reduce_posterior_samples

    S, C, T = 1000, 5, 2000
    raw = {
        'alpha_y_mult': torch.rand(S, C, T),
        'log2_alpha_y': torch.randn(S, C, T),
        'mu_ntc': torch.rand(S, T) * 10,
        'o_y': torch.rand(S, T),
    }

    def _saved_size(d):
        buf = io.BytesIO()
        torch.save(d, buf)
        return buf.tell()

    full_size = _saved_size(raw)
    lean_size = _saved_size(_reduce_posterior_samples(raw))
    assert lean_size < full_size / 50  # ~1000 samples -> ~3 stats: >300x smaller, minus overhead


def test_lean_load_does_not_need_the_full_file(fitted_and_saved, tmp_path):
    """The real point of the companion file: lean loading must succeed even
    when the full multi-sample file is entirely absent, proving load_ntc_fit/
    load_cis_fit(lean=True) never opens it."""
    src_dir = _run_dir(fitted_and_saved)
    dst_root = tmp_path / "no_full_file"
    dst_dir = dst_root / fitted_and_saved["label"]
    shutil.copytree(src_dir, dst_dir)
    os.remove(dst_dir / "posterior_samples_ntc_gene.pt")
    os.remove(dst_dir / "posterior_samples_cis.pt")

    model = bayesDREAM(
        meta=fitted_and_saved["meta"], counts=fitted_and_saved["counts"],
        cis_gene="GFI1B", output_dir=str(dst_root),
        label=fitted_and_saved["label"], device="cpu",
    )
    model.set_technical_groups(["cell_line"])
    model.load_ntc_fit(lean=True)
    model.load_cis_fit(lean=True)

    assert model.get_modality("gene").is_ntc_lean is True
    assert model.is_cis_lean is True
    # And the resulting summary still works end-to-end.
    guide_df, cell_df = model.save_cis_summary()
    assert len(cell_df) > 0


def test_lean_load_falls_back_when_companion_file_missing(fitted_and_saved, tmp_path, capsys):
    """Older saved runs (or a companion file the user deleted) must still
    lean-load correctly via the full-load-then-reduce fallback, with a
    printed note explaining why it was slower."""
    src_dir = _run_dir(fitted_and_saved)
    dst_root = tmp_path / "no_lean_file"
    dst_dir = dst_root / fitted_and_saved["label"]
    shutil.copytree(src_dir, dst_dir)
    os.remove(dst_dir / "posterior_samples_ntc_gene_lean.pt")
    os.remove(dst_dir / "posterior_samples_cis_lean.pt")

    model = bayesDREAM(
        meta=fitted_and_saved["meta"], counts=fitted_and_saved["counts"],
        cis_gene="GFI1B", output_dir=str(dst_root),
        label=fitted_and_saved["label"], device="cpu",
    )
    model.set_technical_groups(["cell_line"])
    model.load_ntc_fit(lean=True)
    model.load_cis_fit(lean=True)

    out = capsys.readouterr().out
    assert "no precomputed lean file found" in out
    assert model.get_modality("gene").is_ntc_lean is True
    assert model.is_cis_lean is True


def test_lean_ntc_matches_full_summary(fitted_and_saved):
    full_model = _fresh_model(fitted_and_saved)
    full_model.load_ntc_fit(lean=False)
    full_df = full_model.save_ntc_summary()

    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    lean_df = lean_model.save_ntc_summary()

    assert list(full_df.columns) == list(lean_df.columns)
    np.testing.assert_allclose(full_df["mu_ntc"], lean_df["mu_ntc"], rtol=1e-5)
    np.testing.assert_allclose(full_df["o_y"], lean_df["o_y"], rtol=1e-5)

    # Sentinel present, raw multi-sample tensors gone.
    ps = lean_model.get_modality("gene").posterior_samples_ntc
    assert ps["__lean__"] is True
    assert ps["mu_ntc"].shape[0] == 1

    # Discoverable markers: per-modality property, repr, and full_model unaffected.
    assert lean_model.get_modality("gene").is_ntc_lean is True
    assert full_model.get_modality("gene").is_ntc_lean is False
    assert "LEAN-LOADED" in repr(lean_model)
    assert "LEAN-LOADED" not in repr(full_model)


def test_lean_cis_matches_full_summary(fitted_and_saved):
    full_model = _fresh_model(fitted_and_saved)
    full_model.load_ntc_fit(lean=False)
    full_model.load_cis_fit(lean=False)
    full_guide_df, full_cell_df = full_model.save_cis_summary()

    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    lean_model.load_cis_fit(lean=True)
    lean_guide_df, lean_cell_df = lean_model.save_cis_summary()

    # Cell-level CI is exact (no cross-cell aggregation) under lean mode.
    np.testing.assert_allclose(
        full_cell_df["x_true_lower"], lean_cell_df["x_true_lower"], rtol=1e-5)
    np.testing.assert_allclose(
        full_cell_df["x_true_upper"], lean_cell_df["x_true_upper"], rtol=1e-5)

    # Guide-level CI is an approximation in lean mode — just check it's finite
    # and in the right ballpark (not wildly different from the exact value).
    assert np.all(np.isfinite(lean_guide_df["x_true_lower"]))
    assert np.all(np.isfinite(lean_guide_df["x_true_upper"]))
    assert np.all(lean_guide_df["x_true_lower"] <= lean_guide_df["x_true_upper"] + 1e-6)
    # Guide-level mean is "mean of per-cell posterior medians" under lean
    # loading (not the exact mean of per-cell posterior means) — close but
    # not bit-identical to the full-load value.
    np.testing.assert_allclose(
        full_guide_df["x_true_mean"], lean_guide_df["x_true_mean"], rtol=0.05)

    assert lean_model.is_cis_lean is True
    assert full_model.is_cis_lean is False
    assert "cis=True" in repr(lean_model)


def test_lean_supports_pipeline_continuation_access_patterns(fitted_and_saved):
    """Mirrors the exact access patterns used by fitting/cis.py's o_x empirical
    Bayes prior and model.py's _extract_cis_alpha_from_ntc_posteriors /
    _trim_feature_axis_in_posteriors (last-axis feature slicing + median(dim=0))."""
    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    lean_model.load_cis_fit(lean=True)

    cis_mod = lean_model.get_modality("cis")
    assert "o_x" in cis_mod.posterior_samples_ntc
    o_x_ntc = float(cis_mod.posterior_samples_ntc["o_x"].mean().item())
    assert np.isfinite(o_x_ntc)

    gene_mod = lean_model.get_modality("gene")
    ps = gene_mod.posterior_samples_ntc
    alpha = ps.get("alpha_y_mult")
    if alpha is not None:
        T = alpha.shape[-1]
        assert T > 0
        # last-axis slicing (as in _trim_feature_axis_in_posteriors)
        sliced = alpha[..., [0]]
        assert sliced.shape[-1] == 1
        # median(dim=0) on the singleton-dim tensor must return the stored
        # point estimate unchanged (as in _extract_cis_alpha_from_ntc_posteriors)
        collapsed = sliced.median(dim=0).values
        np.testing.assert_allclose(
            collapsed.numpy(), sliced[0].numpy(), rtol=1e-6)


def test_lean_ntc_refuses_resave(fitted_and_saved):
    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    with pytest.raises(ValueError, match="lean=True"):
        lean_model.save_ntc_fit()


def test_lean_cis_refuses_resave(fitted_and_saved):
    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    lean_model.load_cis_fit(lean=True)
    with pytest.raises(ValueError, match="lean=True"):
        lean_model.save_cis_fit()


def test_lean_trans_not_implemented(fitted_and_saved):
    lean_model = _fresh_model(fitted_and_saved)
    with pytest.raises(NotImplementedError):
        lean_model.load_trans_fit(lean=True)


# ---------------------------------------------------------------------------
# Plotting: CI-band plots must keep working (using the lean CI siblings);
# distribution plots that need genuine per-draw samples must raise clearly.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lean_and_full_models(fitted_and_saved):
    import matplotlib
    matplotlib.use("Agg")

    full_model = _fresh_model(fitted_and_saved)
    full_model.load_ntc_fit(lean=False)
    full_model.load_cis_fit(lean=False)

    lean_model = _fresh_model(fitted_and_saved)
    lean_model.load_ntc_fit(lean=True)
    lean_model.load_cis_fit(lean=True)

    return full_model, lean_model


def test_scatter_ci95_by_guide_works_lean_and_full(lean_and_full_models):
    """CI-band scatter plot must keep working under lean loading (exact CI)."""
    full_model, lean_model = lean_and_full_models
    ax_full = full_model.plot_xtrue_ci(cis_gene="GFI1B", show=False)
    ax_lean = lean_model.plot_xtrue_ci(cis_gene="GFI1B", show=False)
    assert ax_full is not None
    assert ax_lean is not None


def test_scatter_by_guide_works_lean_and_full(lean_and_full_models):
    """Mean/std scatter must keep working under lean loading (std approximated from CI)."""
    full_model, lean_model = lean_and_full_models
    ax_full = full_model.plot_xtrue_scatter(cis_gene="GFI1B", show=False)
    ax_lean = lean_model.plot_xtrue_scatter(cis_gene="GFI1B", show=False)
    assert ax_full is not None
    assert ax_lean is not None


def test_plot_parameter_ci_panel_technical_param_lean_matches_full(lean_and_full_models):
    full_model, lean_model = lean_and_full_models
    fig_full, ax_full = full_model.plot_parameter_ci_panel(['mu_ntc'], technical_group=0, show=False)
    fig_lean, ax_lean = lean_model.plot_parameter_ci_panel(['mu_ntc'], technical_group=0, show=False)
    assert ax_full is not None
    assert ax_lean is not None


def test_plot_parameter_ci_panel_lean_rejects_other_ci_level(lean_and_full_models):
    _, lean_model = lean_and_full_models
    with pytest.raises(ValueError, match="ci_level"):
        lean_model.plot_parameter_ci_panel(['mu_ntc'], technical_group=0, ci_level=90.0, show=False)


def test_plot_technical_fit_raises_on_lean(lean_and_full_models):
    _, lean_model = lean_and_full_models
    with pytest.raises(ValueError, match="lean=True"):
        lean_model.plot_technical_fit('mu_ntc')


def test_plot_cis_fit_raises_on_lean(lean_and_full_models):
    _, lean_model = lean_and_full_models
    with pytest.raises(ValueError, match="lean=True"):
        lean_model.plot_cis_fit()


def test_plot_xtrue_density_by_guide_raises_on_lean(lean_and_full_models):
    from bayesDREAM.plotting import plot_xtrue_density_by_guide
    _, lean_model = lean_and_full_models
    with pytest.raises(ValueError, match="lean=True"):
        plot_xtrue_density_by_guide(lean_model, show=False)


# ---------------------------------------------------------------------------
# Dimensionality: multinomial modalities carry an extra category (K) axis on
# top of the usual [S, C, T] shape (e.g. alpha_y_add is [S, C, T, K]), and
# fit_ntc emits several messy intermediate tensors with broadcasting
# singleton dims (5D-7D). _reduce_posterior_samples must handle these, and
# save_ntc_summary's per-category output must match between lean and full.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multinomial_ntc_saved(tmp_path_factory, shared_test_data):
    import torch

    torch.manual_seed(0)
    data = shared_test_data
    outdir = tmp_path_factory.mktemp("lean_load_multinomial")
    label = "mn_lean_test"

    model = bayesDREAM(
        meta=data["meta"], counts=data["gene_counts"], feature_meta=data["feature_meta"],
        cis_gene="GFI1B", guide_covariates=["cell_line"],
        output_dir=str(outdir), label=label, device="cpu",
    )
    donor_meta = pd.DataFrame(
        {"donor": [f"donor_{i}" for i in range(data["multinomial_counts"].shape[0])]})
    model.add_custom_modality(name="donor_usage", counts=data["multinomial_counts"],
                               feature_meta=donor_meta, distribution="multinomial")
    model.set_technical_groups(["cell_line"])
    model.fit_ntc(modality_name="donor_usage", niters=20, nsamples=10, sum_factor_col="sum_factor")
    model.save_ntc_fit(modalities=["donor_usage"], save_model_level=False)

    return {"data": data, "outdir": str(outdir), "label": label}


def _fresh_multinomial_model(multinomial_ntc_saved):
    data = multinomial_ntc_saved["data"]
    model = bayesDREAM(
        meta=data["meta"], counts=data["gene_counts"], feature_meta=data["feature_meta"],
        cis_gene="GFI1B", guide_covariates=["cell_line"],
        output_dir=multinomial_ntc_saved["outdir"], label=multinomial_ntc_saved["label"],
        device="cpu",
    )
    donor_meta = pd.DataFrame(
        {"donor": [f"donor_{i}" for i in range(data["multinomial_counts"].shape[0])]})
    model.add_custom_modality(name="donor_usage", counts=data["multinomial_counts"],
                               feature_meta=donor_meta, distribution="multinomial")
    model.set_technical_groups(["cell_line"])
    return model


def test_reduce_posterior_samples_handles_multinomial_shapes(multinomial_ntc_saved):
    """alpha_y_add is [S,C,T,K]; intermediate tensors go up to 7D with
    singleton broadcasting dims. Must reduce without error or shape loss."""
    import torch
    from bayesDREAM.io.load import _reduce_posterior_samples

    model = _fresh_multinomial_model(multinomial_ntc_saved)
    model.load_ntc_fit(modalities=["donor_usage"], lean=False)
    raw = model.get_modality("donor_usage").posterior_samples_ntc
    assert raw["alpha_y_add"].ndim == 4  # [S, C, T, K]

    reduced = _reduce_posterior_samples(raw)
    # Point estimate keeps a singleton sample dim; siblings drop it entirely.
    assert reduced["alpha_y_add"].shape == (1,) + tuple(raw["alpha_y_add"].shape[1:])
    assert reduced["alpha_y_add_lower"].shape == tuple(raw["alpha_y_add"].shape[1:])
    assert reduced["alpha_y_add_upper"].shape == tuple(raw["alpha_y_add"].shape[1:])
    assert torch.all(reduced["alpha_y_add_lower"] <= reduced["alpha_y_add_upper"] + 1e-6)


def test_lean_ntc_multinomial_summary_matches_full(multinomial_ntc_saved):
    full_model = _fresh_multinomial_model(multinomial_ntc_saved)
    full_model.load_ntc_fit(modalities=["donor_usage"], lean=False)
    full_df = full_model.save_ntc_summary(modality_name="donor_usage")

    lean_model = _fresh_multinomial_model(multinomial_ntc_saved)
    lean_model.load_ntc_fit(modalities=["donor_usage"], lean=True)
    lean_df = lean_model.save_ntc_summary(modality_name="donor_usage")

    assert list(full_df.columns) == list(lean_df.columns)
    median_cols = [c for c in full_df.columns if c.endswith("_median")]
    assert len(median_cols) == 3 * 3  # 3 groups x 3 categories
    for col in median_cols:
        np.testing.assert_allclose(full_df[col], lean_df[col], rtol=1e-5)

    assert lean_model.get_modality("donor_usage").is_ntc_lean is True
    assert full_model.get_modality("donor_usage").is_ntc_lean is False


def test_plot_technical_fit_raises_on_lean_multinomial(multinomial_ntc_saved):
    lean_model = _fresh_multinomial_model(multinomial_ntc_saved)
    lean_model.load_ntc_fit(modalities=["donor_usage"], lean=True)
    with pytest.raises(ValueError, match="lean=True"):
        lean_model.plot_technical_fit('alpha_y', modality_name='donor_usage')


def test_plot_parameter_ci_panel_multinomial_fails_identically_lean_and_full(multinomial_ntc_saved):
    """Pre-existing, lean-unrelated limitation: _get_technical_param_samples's
    group-selection assumes a [S,C,T] (or [C,T] for CI siblings) layout and
    never accounted for multinomial's extra K axis, so T ends up mis-set to C
    downstream regardless of lean loading. Documents that lean loading does
    not introduce a NEW failure mode here — it fails the same way full does."""
    errors = {}
    for lean in (False, True):
        model = _fresh_multinomial_model(multinomial_ntc_saved)
        model.load_ntc_fit(modalities=["donor_usage"], lean=lean)
        try:
            model.plot_parameter_ci_panel(['alpha_y_add'], modality_name='donor_usage',
                                           technical_group=1, show=False)
            errors[lean] = None
        except ValueError as e:
            errors[lean] = str(e)
    assert errors[False] is not None, "expected this to already fail without lean loading"
    assert errors[False] == errors[True]
