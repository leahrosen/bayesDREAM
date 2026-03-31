"""Test exon skipping aggregation functionality."""

import unittest
import numpy as np
import pandas as pd


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


class TestExonSkipAggregation(unittest.TestCase):
    """Test Modality with exon skipping inc1/inc2/skip arrays."""

    @classmethod
    def setUpClass(cls):
        from bayesDREAM import Modality
        cls.Modality = Modality
        cls.inc1, cls.inc2, cls.skip, cls.feature_meta = _make_exon_skip_data()
        cls.inclusion_min = np.minimum(cls.inc1, cls.inc2)
        cls.total_min = cls.inclusion_min + cls.skip

    def _make_modality(self, method='min'):
        return self.Modality(
            name='exon_skip_test',
            counts=self.inclusion_min.copy(),
            feature_meta=self.feature_meta,
            distribution='binomial',
            denominator=self.total_min.copy(),
            inc1=self.inc1,
            inc2=self.inc2,
            skip=self.skip,
            exon_aggregate_method=method,
        )

    def test_create_with_min_aggregation(self):
        mod = self._make_modality('min')
        self.assertTrue(mod.is_exon_skipping())
        self.assertEqual(mod.exon_aggregate_method, 'min')
        self.assertEqual(mod.inc1.shape, self.inc1.shape)
        self.assertEqual(mod.inc2.shape, self.inc2.shape)
        self.assertEqual(mod.skip.shape, self.skip.shape)

    def test_change_aggregation_to_mean(self):
        mod = self._make_modality('min')
        old_counts = mod.counts.copy()
        mod.set_exon_aggregate_method('mean')
        self.assertEqual(mod.exon_aggregate_method, 'mean')
        # Counts should have changed
        self.assertFalse(np.allclose(old_counts, mod.counts))
        expected_inclusion_mean = (self.inc1 + self.inc2) / 2.0
        expected_total_mean = expected_inclusion_mean + self.skip
        np.testing.assert_allclose(mod.counts, expected_inclusion_mean)
        np.testing.assert_allclose(mod.denominator, expected_total_mean)

    def test_change_blocked_after_technical_fit(self):
        mod = self._make_modality('mean')
        mod.mark_technical_fit_complete()
        with self.assertRaises(ValueError):
            mod.set_exon_aggregate_method('min')

    def test_override_after_technical_fit(self):
        mod = self._make_modality('mean')
        mod.mark_technical_fit_complete()
        mod.set_exon_aggregate_method('min', allow_after_technical_fit=True)
        self.assertEqual(mod.exon_aggregate_method, 'min')
        expected_inclusion_min = np.minimum(self.inc1, self.inc2)
        np.testing.assert_allclose(mod.counts, expected_inclusion_min)

    def test_feature_subset_preserves_exon_data(self):
        mod = self._make_modality('min')
        subset = mod.get_feature_subset([0, 1, 2])
        self.assertTrue(subset.is_exon_skipping())
        self.assertEqual(subset.inc1.shape[0], 3)
        self.assertEqual(subset.exon_aggregate_method, mod.exon_aggregate_method)

    def test_cell_subset_preserves_exon_data(self):
        mod = self._make_modality('min')
        subset = mod.get_cell_subset([0, 1, 2, 3, 4])
        self.assertEqual(subset.inc1.shape[1], 5)

    def test_regular_binomial_not_exon_skipping(self):
        mod = self.Modality(
            name='regular_binomial',
            counts=self.inclusion_min.copy(),
            feature_meta=self.feature_meta,
            distribution='binomial',
            denominator=self.total_min.copy(),
        )
        self.assertFalse(mod.is_exon_skipping())
        with self.assertRaises(ValueError):
            mod.set_exon_aggregate_method('mean')


if __name__ == '__main__':
    unittest.main()
