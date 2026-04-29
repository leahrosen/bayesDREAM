"""Test bayesDREAM with different matrix input types.

Covers:
- Dense numpy arrays
- Sparse scipy matrices
- Pandas DataFrames
"""

import numpy as np
import pandas as pd
from scipy import sparse

import pytest


def _make_test_data(n_genes=50, n_cells=100, matrix_type='dataframe', seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'Cell_{i}' for i in range(n_cells)],
        'guide': ['guide_ntc'] * (n_cells // 2) + ['guide_GFI1B'] * (n_cells // 2),
        'target': ['ntc'] * (n_cells // 2) + ['GFI1B'] * (n_cells // 2),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'cell_line': ['line1'] * (n_cells // 2) + ['line2'] * (n_cells // 2),
    })
    counts_array = np.random.randint(0, 200, (n_genes, n_cells))

    if matrix_type == 'dataframe':
        gene_names = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        counts = pd.DataFrame(counts_array, index=gene_names, columns=meta['cell'].tolist())
        gene_meta = pd.DataFrame(
            {'gene_name': gene_names, 'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)]},
            index=gene_names,
        )
    elif matrix_type == 'dense':
        counts = counts_array
        gene_names_for_meta = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        gene_meta = pd.DataFrame(
            {'gene_name': gene_names_for_meta, 'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)]},
            index=range(n_genes),
        )
    elif matrix_type == 'sparse': 
        counts = sparse.csr_matrix(counts_array.astype(float))
        gene_names_for_meta = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        gene_meta = pd.DataFrame(
            {'gene_name': gene_names_for_meta, 'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)]},
            index=range(n_genes),
        )
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return meta, counts, gene_meta


def test_dataframe_input():
    from bayesDREAM import bayesDREAM
    meta, counts, gene_meta = _make_test_data(matrix_type='dataframe')
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene='GFI1B',
        output_dir='./test_output_dataframe',
        label='test_dataframe',
        device='cpu',
    )
    assert isinstance(model.counts, pd.DataFrame)
    gene_mod = model.get_modality('gene')
    assert len(gene_mod.feature_meta) == gene_mod.counts.shape[0]


def test_dense_numpy_input():
    from bayesDREAM import bayesDREAM
    meta, counts, gene_meta = _make_test_data(matrix_type='dense')
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene='GFI1B',
        output_dir='./test_output_dense',
        label='test_dense',
        device='cpu',
    )
    assert isinstance(model.counts, np.ndarray)
    gene_mod = model.get_modality('gene')
    assert len(gene_mod.feature_meta) == gene_mod.counts.shape[0]


def test_sparse_scipy_input():
    from bayesDREAM import bayesDREAM
    meta, counts, gene_meta = _make_test_data(matrix_type='sparse')
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene='GFI1B',
        output_dir='./test_output_sparse',
        label='test_sparse',
        device='cpu',
    )
    assert sparse.issparse(model.counts)
    gene_mod = model.get_modality('gene')
    assert len(gene_mod.feature_meta) == gene_mod.counts.shape[0]
