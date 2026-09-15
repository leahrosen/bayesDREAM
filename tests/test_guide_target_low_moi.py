"""Test guide_target DataFrame support in single-guide (low-MOI) mode.

Verifies that low-MOI models can resolve each cell's target from a
guide -> target(s) mapping DataFrame instead of requiring a pre-computed
'target' column in meta. This supports guides with multiple plausible
targets (e.g. ambiguous or off-target effects), which resolve differently
depending on which cis_gene is currently being fit.
"""

import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_ambiguous_guide_data(seed=42):
    """
    30 cells: guide_A -> GFI1B only
    20 cells: guide_B -> MYB only
    10 cells: guide_C -> GFI1B AND MYB (ambiguous)
    40 cells: ntc_1 -> ntc
    """
    np.random.seed(seed)
    n_cells = 100
    guides = ['guide_A'] * 30 + ['guide_B'] * 20 + ['guide_C'] * 10 + ['ntc_1'] * 40
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': guides,
        'cell_line': ['K562'] * n_cells,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_names = [f'gene_{i}' for i in range(30)]
    gene_names[0] = 'GFI1B'
    gene_names[1] = 'MYB'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (len(gene_names), n_cells)),
        index=gene_names,
        columns=meta['cell'].tolist(),
    )
    guide_target = pd.DataFrame({
        'guide':  ['guide_A', 'guide_B', 'guide_C', 'guide_C', 'ntc_1'],
        'target': ['GFI1B',   'MYB',     'GFI1B',   'MYB',     'ntc'],
    })
    return meta, counts, guide_target


@pytest.fixture(scope='module')
def ambiguous_data():
    return _make_ambiguous_guide_data()


def test_target_column_not_required_when_guide_target_given(ambiguous_data):
    from bayesDREAM import bayesDREAM
    meta, counts, guide_target = ambiguous_data
    assert 'target' not in meta.columns
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B', guide_target=guide_target,
        output_dir='./test_output', label='test_gt_gfi1b', device='cpu',
    )
    assert model.cis_gene == 'GFI1B'


def test_missing_target_and_no_guide_target_raises(ambiguous_data):
    from bayesDREAM import bayesDREAM
    meta, counts, _ = ambiguous_data
    with pytest.raises(ValueError, match='Missing required columns'):
        bayesDREAM(
            meta=meta, counts=counts, cis_gene='GFI1B',
            output_dir='./test_output', label='test_gt_missing', device='cpu',
        )


def test_classification_gfi1b(ambiguous_data):
    # Ambiguous guide_C (GFI1B + MYB) should resolve to GFI1B here;
    # guide_B (MYB only) has no NTC/GFI1B target so its cells are dropped.
    from bayesDREAM import bayesDREAM
    meta, counts, guide_target = ambiguous_data
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B', guide_target=guide_target,
        output_dir='./test_output', label='test_gt_classify_gfi1b', device='cpu',
    )
    assert set(model.meta['target'].unique()) == {'ntc', 'GFI1B'}
    assert (model.meta['target'] == 'ntc').sum() == 40
    assert (model.meta['target'] == 'GFI1B').sum() == 40  # 30 (guide_A) + 10 (guide_C)
    assert len(model.meta) == 80


def test_classification_myb(ambiguous_data):
    # Same ambiguous guide_C now resolves to MYB instead.
    from bayesDREAM import bayesDREAM
    meta, counts, guide_target = ambiguous_data
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='MYB', guide_target=guide_target,
        output_dir='./test_output', label='test_gt_classify_myb', device='cpu',
    )
    assert set(model.meta['target'].unique()) == {'ntc', 'MYB'}
    assert (model.meta['target'] == 'ntc').sum() == 40
    assert (model.meta['target'] == 'MYB').sum() == 30  # 20 (guide_B) + 10 (guide_C)
    assert len(model.meta) == 70


def test_exclude_targets_drops_ambiguous_guide(ambiguous_data):
    # guide_C targets MYB among others, so exclude_targets=['MYB'] should drop
    # ALL of guide_C's cells even though it also targets GFI1B.
    from bayesDREAM import bayesDREAM
    meta, counts, guide_target = ambiguous_data
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B', guide_target=guide_target,
        exclude_targets=['MYB'],
        output_dir='./test_output', label='test_gt_exclude', device='cpu',
    )
    assert 'guide_C' not in model.meta['guide'].values
    assert 'guide_B' not in model.meta['guide'].values
    assert set(model.meta['target'].unique()) == {'ntc', 'GFI1B'}
    assert (model.meta['target'] == 'GFI1B').sum() == 30  # guide_A only


# ----------------------------------------------------------------------
# Deferred cis_gene: cis_gene omitted at init, committed via add_cis_gene().
# ----------------------------------------------------------------------
@pytest.fixture(scope='module')
def deferred_model(ambiguous_data):
    from bayesDREAM import bayesDREAM
    meta, counts, guide_target = ambiguous_data
    return bayesDREAM(
        meta=meta, counts=counts, guide_target=guide_target,
        output_dir='./test_output', label='test_gt_deferred', device='cpu',
    )


def test_deferred_keeps_all_cells(deferred_model):
    assert deferred_model.cis_gene is None
    assert len(deferred_model.meta) == 100


def test_deferred_target_has_no_gene_label(deferred_model):
    assert set(deferred_model.meta['target'].unique()) <= {'ntc', 'other'}
    assert (deferred_model.meta['target'] == 'ntc').sum() == 40


def test_add_cis_gene_reclassifies(deferred_model):
    import copy
    model = copy.deepcopy(deferred_model)
    model.add_cis_gene('GFI1B')
    assert model.cis_gene == 'GFI1B'
    assert set(model.meta['target'].unique()) == {'ntc', 'GFI1B'}
    assert (model.meta['target'] == 'GFI1B').sum() == 40  # guide_A (30) + guide_C (10)
    assert (model.meta['target'] == 'ntc').sum() == 40


def test_add_cis_gene_unmapped_gene_raises(deferred_model):
    # gene_2 exists in counts but no guide in guide_target maps to it.
    import copy
    model = copy.deepcopy(deferred_model)
    with pytest.raises(ValueError, match='No guide in guide_target maps'):
        model.add_cis_gene('gene_2')
