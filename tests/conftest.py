"""Shared pytest fixture and test data for bayesDREAM tests."""

import numpy as np
import pandas as pd
import pytest


def _create_shared_test_data(
    n_features=10,
    n_cells=50,
    n_guides=10,
    n_categories=3,
    seed=42,
):

    np.random.seed(seed)

    cells_per_guide = n_cells // n_guides
    guides, targets, cell_lines = [], [], []
    for i in range(n_guides):
        for j in range(cells_per_guide):
            guides.append(f"guide_{i}")
            targets.append("GFI1B" if i < (n_guides // 2) else "ntc")
            cell_lines.append(["A", "B", "C"][(i + j) % 3])

    meta = pd.DataFrame(
        {
            "cell": [f"cell_{i}" for i in range(n_cells)],
            "guide": guides,
            "target": targets,
            "cell_line": cell_lines,
            "sum_factor": np.random.uniform(0.8, 1.2, n_cells),
            "sum_factor_adj": np.random.uniform(0.8, 1.2, n_cells),
        }
    )

    gene_names = [f"gene_{i}" for i in range(n_features)]
    gene_counts = pd.DataFrame(
        np.random.poisson(lam=50, size=(n_features, n_cells)),
        columns=[f"cell_{i}" for i in range(n_cells)],
        index=gene_names,
    )
    gene_counts.loc["GFI1B"] = np.random.poisson(lam=100, size=n_cells)
    feature_meta = pd.DataFrame({"gene": gene_counts.index}, index=gene_counts.index)

    inclusion_counts = pd.DataFrame(
        np.random.poisson(lam=30, size=(n_features, n_cells)),
        columns=[f"cell_{i}" for i in range(n_cells)],
        index=[f"exon_{i}" for i in range(n_features)],
    )
    total_counts = pd.DataFrame(
        np.random.poisson(lam=60, size=(n_features, n_cells)),
        columns=[f"cell_{i}" for i in range(n_cells)],
        index=[f"exon_{i}" for i in range(n_features)],
    )
    multinomial_counts = np.random.poisson(
        lam=20, size=(n_features, n_cells, n_categories)
    )
    normal_scores = pd.DataFrame(
        np.random.normal(loc=0, scale=1, size=(n_features, n_cells)),
        columns=[f"cell_{i}" for i in range(n_cells)],
        index=[f"score_{i}" for i in range(n_features)],
    )

    atac_counts = np.random.poisson(50, (5, n_cells))
    atac_meta = pd.DataFrame(
        {
            "region": [f"region_{i}" for i in range(5)],
            "chrom": "chr1",
            "start": np.arange(5) * 1000,
            "end": np.arange(5) * 1000 + 500,
        }
    )
    splicing_counts = np.random.poisson(20, (3, n_cells))
    splicing_denom = np.random.poisson(100, (3, n_cells))
    splicing_meta = pd.DataFrame(
        {
            "junction": [f"junction_{i}" for i in range(3)],
            "gene": ["gene_0", "gene_1", "gene_2"],
        }
    )

    return {
        "meta": meta,
        "gene_counts": gene_counts,
        "feature_meta": feature_meta,
        "inclusion_counts": inclusion_counts,
        "total_counts": total_counts,
        "multinomial_counts": multinomial_counts,
        "normal_scores": normal_scores,
        "atac_counts": atac_counts,
        "atac_meta": atac_meta,
        "splicing_counts": splicing_counts,
        "splicing_denom": splicing_denom,
        "splicing_meta": splicing_meta,
    }


@pytest.fixture(scope="module")
def shared_test_data():
    return _create_shared_test_data(
        n_features=10,
        n_cells=50,
        n_guides=10,
        n_categories=3,
        seed=42,
    )
