"""Test exon skipping aggregation functionality."""

import numpy as np
import pandas as pd

import pytest


def _make_exon_skip_data(n_events=5, n_cells=10, seed=42):
    np.random.seed(seed)
    inc1 = np.random.poisson(10, (n_events, n_cells)).astype(float)
    inc2 = np.random.poisson(12, (n_events, n_cells)).astype(float)
    skip = np.random.poisson(8, (n_events, n_cells)).astype(float)
    feature_meta = pd.DataFrame({
        'trip_id': range(n_events),
        'chrom': ['chr1'] * n_events,
        'strand': ['+'] * n_events,
    })
    return inc1, inc2, skip, feature_meta


@pytest.fixture(scope='module')
def exon_skip_ctx():
    from bayesDREAM import Modality
    inc1, inc2, skip, feature_meta = _make_exon_skip_data()
    inclusion_min = np.minimum(inc1, inc2)
    total_min = inclusion_min + skip
    return {
        'Modality': Modality,
        'inc1': inc1,
        'inc2': inc2,
        'skip': skip,
        'feature_meta': feature_meta,
        'inclusion_min': inclusion_min,
        'total_min': total_min,
    }


def _make_modality(ctx, method='min'):
    return ctx['Modality'](
            name='exon_skip_test',
            counts=ctx['inclusion_min'].copy(),
            feature_meta=ctx['feature_meta'],
            distribution='binomial',
            denominator=ctx['total_min'].copy(),
            inc1=ctx['inc1'],
            inc2=ctx['inc2'],
            skip=ctx['skip'],
            exon_aggregate_method=method,
        )


def test_create_with_min_aggregation(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'min')
    assert mod.is_exon_skipping()
    assert mod.exon_aggregate_method == 'min'
    assert mod.inc1.shape == exon_skip_ctx['inc1'].shape
    assert mod.inc2.shape == exon_skip_ctx['inc2'].shape
    assert mod.skip.shape == exon_skip_ctx['skip'].shape


def test_change_aggregation_to_mean(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'min')
    old_counts = mod.counts.copy()
    mod.set_exon_aggregate_method('mean')
    assert mod.exon_aggregate_method == 'mean'
    assert not np.allclose(old_counts, mod.counts)
    expected_inclusion_mean = (exon_skip_ctx['inc1'] + exon_skip_ctx['inc2']) / 2.0
    expected_total_mean = expected_inclusion_mean + exon_skip_ctx['skip']
    np.testing.assert_allclose(mod.counts, expected_inclusion_mean)
    np.testing.assert_allclose(mod.denominator, expected_total_mean)


def test_change_blocked_after_technical_fit(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'mean')
    mod.mark_technical_fit_complete()
    with pytest.raises(ValueError):
        mod.set_exon_aggregate_method('min')


def test_override_after_technical_fit(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'mean')
    mod.mark_technical_fit_complete()
    mod.set_exon_aggregate_method('min', allow_after_technical_fit=True)
    assert mod.exon_aggregate_method == 'min'
    expected_inclusion_min = np.minimum(exon_skip_ctx['inc1'], exon_skip_ctx['inc2'])
    np.testing.assert_allclose(mod.counts, expected_inclusion_min)


def test_feature_subset_preserves_exon_data(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'min')
    subset = mod.get_feature_subset([0, 1, 2])
    assert subset.is_exon_skipping()
    assert subset.inc1.shape[0] == 3
    assert subset.exon_aggregate_method == mod.exon_aggregate_method


def test_cell_subset_preserves_exon_data(exon_skip_ctx):
    mod = _make_modality(exon_skip_ctx, 'min')
    subset = mod.get_cell_subset([0, 1, 2, 3, 4])
    assert subset.inc1.shape[1] == 5


def test_regular_binomial_not_exon_skipping(exon_skip_ctx):
    mod = exon_skip_ctx['Modality'](
        name='regular_binomial',
        counts=exon_skip_ctx['inclusion_min'].copy(),
        feature_meta=exon_skip_ctx['feature_meta'],
        distribution='binomial',
        denominator=exon_skip_ctx['total_min'].copy(),
    )
    assert not mod.is_exon_skipping()
    with pytest.raises(ValueError):
        mod.set_exon_aggregate_method('mean')
