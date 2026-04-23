"""Test cell_names parameter for add_custom_modality with numpy arrays."""

import numpy as np
import pandas as pd

import pytest


@pytest.fixture(scope='module')
def cell_names_model(shared_test_data):
    from bayesDREAM import bayesDREAM
    meta = shared_test_data['meta'].copy()
    gene_counts_df = shared_test_data['gene_counts'].copy()
    cell_names = meta['cell'].tolist()
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts_df,
        cis_gene='GFI1B',
        output_dir='./test_output/cell_names_test',
        label='cell_names_test',
        device='cpu',
    )
    return {'model': model, 'cell_names': cell_names, 'n_cells': len(cell_names)}


def test_numpy_array_with_explicit_cell_names(cell_names_model):
    model = cell_names_model['model']
    cell_names = cell_names_model['cell_names']
    n_cells = cell_names_model['n_cells']
    
    custom_counts = np.random.randn(15, n_cells)
    custom_meta = pd.DataFrame({'feature': [f'custom_feature_{i}' for i in range(15)]})
    model.add_custom_modality(
        name='custom_array',
        counts=custom_counts,
        feature_meta=custom_meta,
        distribution='normal',
        cell_names=cell_names,
        overwrite=True,
    )
    mod = model.get_modality('custom_array')
    assert mod.cell_names is not None
    assert len(mod.cell_names) == n_cells
    assert mod.cell_names == cell_names


def test_dataframe_auto_extracts_cell_names(cell_names_model):
    model = cell_names_model['model']
    cell_names = cell_names_model['cell_names']
    n_cells = cell_names_model['n_cells']
    
    custom_counts_df = pd.DataFrame(
        np.random.randn(10, n_cells),
        index=[f'df_feature_{i}' for i in range(10)],
        columns=cell_names,
    )
    custom_meta = pd.DataFrame({'feature': [f'df_feature_{i}' for i in range(10)]})
    model.add_custom_modality(
        name='custom_dataframe',
        counts=custom_counts_df,
        feature_meta=custom_meta,
        distribution='normal',
    )
    mod = model.get_modality('custom_dataframe')
    assert mod.cell_names is not None
    assert mod.cell_names == cell_names


def test_cell_subset_preserves_cell_names(cell_names_model):
    model = cell_names_model['model']
    cell_names = cell_names_model['cell_names']
    
    # Ensure custom_array modality exists
    if 'custom_array' not in model.modalities:
        test_numpy_array_with_explicit_cell_names(cell_names_model)
    
    mod = model.get_modality('custom_array')
    subset_cells = cell_names[:20]
    subset = mod.get_cell_subset(subset_cells)
    assert subset.cell_names is not None
    assert len(subset.cell_names) == 20
    assert subset.cell_names == subset_cells


def test_no_cell_names_defaults_to_model_cells(cell_names_model):
    model = cell_names_model['model']
    cell_names = cell_names_model['cell_names']
    n_cells = cell_names_model['n_cells']
    
    custom_counts = np.random.randn(8, n_cells)
    custom_meta = pd.DataFrame({'feature': [f'no_names_feature_{i}' for i in range(8)]})
    model.add_custom_modality(
        name='custom_no_names',
        counts=custom_counts,
        feature_meta=custom_meta,
        distribution='normal',
    )
    mod = model.get_modality('custom_no_names')
    assert mod.cell_names == cell_names
