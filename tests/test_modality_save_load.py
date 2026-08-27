"""Test modality-specific save/load functionality.

Verifies:
1. Users can save specific modalities only
2. Primary modality is NOT saved by default - must be in the list
3. save_model_level flag works correctly
4. Loading specific modalities works correctly
"""

import os
import numpy as np
import pandas as pd

import torch
import pyro
import pytest
from bayesDREAM import bayesDREAM, Modality

pytestmark = pytest.mark.slow

NITERS = 20


def _build_model(meta, gene_counts, atac_counts, atac_meta, splicing_counts, splicing_denom, splicing_meta, output_dir):
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene="GFI1B",
        output_dir=str(output_dir),
        label="modality_test",
    )

    model.modalities["atac"] = Modality(
        name="atac",
        counts=atac_counts,
        feature_meta=atac_meta,
        cell_names=meta["cell"].values,
        distribution="negbinom",
    )

    model.modalities["splicing"] = Modality(
        name="splicing",
        counts=splicing_counts,
        feature_meta=splicing_meta,
        cell_names=meta["cell"].values,
        distribution="binomial",
        denominator=splicing_denom,
    )

    return model


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory, shared_test_data):
    base_dir = tmp_path_factory.mktemp("modality_save_load")
    meta = shared_test_data["meta"]
    gene_counts = shared_test_data["gene_counts"]
    atac_counts = shared_test_data["atac_counts"]
    atac_meta = shared_test_data["atac_meta"]
    splicing_counts = shared_test_data["splicing_counts"]
    splicing_denom = shared_test_data["splicing_denom"]
    splicing_meta = shared_test_data["splicing_meta"]

    model = _build_model(
        meta,
        gene_counts,
        atac_counts,
        atac_meta,
        splicing_counts,
        splicing_denom,
        splicing_meta,
        output_dir=base_dir / "seed_model",
    )

    model.set_technical_groups(["cell_line"])
    for mod_name in ["gene", "atac", "splicing"]:
        model.fit_ntc(modality_name=mod_name, sum_factor_col="sum_factor", niters=NITERS)

    return {
        "model": model,
        "base_dir": base_dir,
        "data": shared_test_data,
    }


def test_save_excludes_primary_by_default(fitted_model):
    model = fitted_model["model"]
    outdir = fitted_model["base_dir"] / "test_excl_primary"
    outdir.mkdir(parents=True, exist_ok=True)

    model.save_ntc_fit(
        output_dir=str(outdir),
        modalities=["atac", "splicing"],
        save_model_level=False,
    )

    assert not (outdir / "alpha_y_prefit_gene.pt").exists(), "Primary 'gene' should NOT be saved when not listed"
    assert (outdir / "alpha_y_prefit_atac.pt").exists()
    assert (outdir / "alpha_y_prefit_splicing.pt").exists()


def test_save_model_level_false_skips_model_params(fitted_model):
    model = fitted_model["model"]
    outdir = fitted_model["base_dir"] / "test_no_model_level"
    outdir.mkdir(parents=True, exist_ok=True)

    model.save_ntc_fit(
        output_dir=str(outdir),
        modalities=["atac"],
        save_model_level=False,
    )

    assert not (outdir / "alpha_x_prefit.pt").exists()
    assert not (outdir / "alpha_y_prefit.pt").exists()


def test_save_primary_explicitly(fitted_model):
    model = fitted_model["model"]
    outdir = fitted_model["base_dir"] / "test_explicit_primary"
    outdir.mkdir(parents=True, exist_ok=True)

    model.save_ntc_fit(
        output_dir=str(outdir),
        modalities=["gene"],
        save_model_level=True,
    )

    assert (outdir / "alpha_y_prefit_gene.pt").exists()
    assert not (outdir / "alpha_y_prefit_atac.pt").exists()
    assert (outdir / "alpha_y_prefit.pt").exists()


def test_load_specific_modalities(fitted_model):
    source_model = fitted_model["model"]
    test_data = fitted_model["data"]
    meta = test_data["meta"]
    gene_counts = test_data["gene_counts"]
    atac_counts = test_data["atac_counts"]
    atac_meta = test_data["atac_meta"]
    splicing_counts = test_data["splicing_counts"]
    splicing_denom = test_data["splicing_denom"]
    splicing_meta = test_data["splicing_meta"]

    outdir = fitted_model["base_dir"] / "test_load_specific"
    outdir.mkdir(parents=True, exist_ok=True)

    source_model.save_ntc_fit(
        output_dir=str(outdir),
        modalities=["atac", "splicing"],
        save_model_level=False,
    )

    model2 = _build_model(
        meta,
        gene_counts,
        atac_counts,
        atac_meta,
        splicing_counts,
        splicing_denom,
        splicing_meta,
        output_dir=outdir,
    )
    model2.load_ntc_fit(modalities=["atac", "splicing"], load_model_level=False)

    gene_mod = model2.modalities["gene"]
    atac_mod = model2.modalities["atac"]
    spl_mod = model2.modalities["splicing"]

    assert not (hasattr(gene_mod, "alpha_y_prefit") and gene_mod.alpha_y_prefit is not None), "gene should NOT be loaded"
    assert atac_mod.alpha_y_prefit is not None
    assert spl_mod.alpha_y_prefit is not None


def test_default_save_includes_all_modalities(fitted_model):
    model = fitted_model["model"]
    outdir = fitted_model["base_dir"] / "test_save_all"
    outdir.mkdir(parents=True, exist_ok=True)

    model.save_ntc_fit(output_dir=str(outdir))
    for name in ["gene", "atac", "splicing"]:
        assert (outdir / f"alpha_y_prefit_{name}.pt").exists(), (
            f"alpha_y_prefit_{name}.pt should exist when modalities=None"
        )
