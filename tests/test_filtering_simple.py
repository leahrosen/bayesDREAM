"""Test distribution-specific filtering at modality creation."""

import unittest
import numpy as np
import pandas as pd


def _make_model():
    from bayesDREAM import bayesDREAM
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(1, 11)],
        'guide': ['g1', 'g2', 'g3', 'g4', 'g5', 'ntc1', 'ntc2', 'ntc3', 'ntc4', 'ntc5'],
        'target': ['GFI1B'] * 5 + ['ntc'] * 5,
        'cell_line': ['A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'B'],
        'sum_factor': [1.0] * 10,
    })
    gene_counts = pd.DataFrame(
        {f'cell{i}': [10 + i, 20 + i, 30 + i, 40 + i, 50 + i,
                      100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 1000 + i]
         for i in range(1, 11)},
        index=['GFI1B', 'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5',
               'GENE6', 'GENE7', 'GENE8', 'GENE9', 'GENE10'],
    )
    return bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='filter_test',
    )


class TestFilteringSimple(unittest.TestCase):
    """Verify constant-feature filtering for each distribution type."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_model()

    def test_binomial_filters_constant_ratio(self):
        custom_counts = np.array([
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],   # variable ratio
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],   # constant ratio
        ])
        custom_denom = np.array([
            [100] * 10,                                   # constant denom → variable ratio
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # proportional → constant ratio
        ])
        custom_meta = pd.DataFrame({'feature': ['feat1_variable', 'feat2_constant']})
        self.model.add_custom_modality(
            name='custom_binomial',
            counts=custom_counts,
            feature_meta=custom_meta,
            distribution='binomial',
            denominator=custom_denom,
        )
        mod = self.model.get_modality('custom_binomial')
        self.assertEqual(mod.dims['n_features'], 1,
                         "Constant-ratio binomial feature should be filtered out")

    def test_multinomial_filters_constant_ratios(self):
        multinomial_counts = np.array([
            [[10, 10], [20, 10], [30, 10], [40, 10], [50, 10],
             [60, 10], [70, 10], [80, 10], [90, 10], [100, 10]],   # variable
            [[5, 5]] * 10,                                            # constant [0.5, 0.5]
        ])
        multinomial_meta = pd.DataFrame({'feature': ['feat1_variable', 'feat2_constant']})
        self.model.add_custom_modality(
            name='custom_multinomial',
            counts=multinomial_counts,
            feature_meta=multinomial_meta,
            distribution='multinomial',
        )
        mod = self.model.get_modality('custom_multinomial')
        self.assertEqual(mod.dims['n_features'], 1,
                         "Constant-ratio multinomial feature should be filtered out")

    def test_normal_filters_zero_variance(self):
        normal_counts = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # variable
            [5.0] * 10,                                                # constant → filter
        ])
        normal_meta = pd.DataFrame({'feature': ['feat1_variable', 'feat2_constant']})
        self.model.add_custom_modality(
            name='custom_normal',
            counts=normal_counts,
            feature_meta=normal_meta,
            distribution='normal',
        )
        mod = self.model.get_modality('custom_normal')
        self.assertEqual(mod.dims['n_features'], 1,
                         "Zero-variance normal feature should be filtered out")


if __name__ == '__main__':
    unittest.main()
