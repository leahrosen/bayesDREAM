"""Test cell_names parameter for add_custom_modality with numpy arrays."""

import unittest
import numpy as np
import pandas as pd


def _make_base_data(n_genes=10, n_cells=50, n_guides=5, seed=42):
    np.random.seed(seed)
    cell_names = [f'cell_{i}' for i in range(n_cells)]
    meta = pd.DataFrame({
        'cell': cell_names,
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['K562', 'HEL'], n_cells),
        'target': ['GFI1B'] * 30 + ['ntc'] * 20,
        'sum_factor': np.random.lognormal(0, 0.2, n_cells),
    })
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts_df = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, (len(genes), n_cells)),
        index=genes,
        columns=cell_names,
    )
    return meta, gene_counts_df, cell_names


class TestCellNamesNumpy(unittest.TestCase):
    """Verify cell_names behaviour in add_custom_modality."""

    @classmethod
    def setUpClass(cls):
        from bayesDREAM import bayesDREAM
        cls.meta, cls.gene_counts_df, cls.cell_names = _make_base_data()
        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=cls.gene_counts_df,
            cis_gene='GFI1B',
            output_dir='./test_output/cell_names_test',
            label='cell_names_test',
            device='cpu',
        )
        cls.n_cells = len(cls.cell_names)

    def test_numpy_array_with_explicit_cell_names(self):
        custom_counts = np.random.randn(15, self.n_cells)
        custom_meta = pd.DataFrame({'feature': [f'custom_feature_{i}' for i in range(15)]})
        self.model.add_custom_modality(
            name='custom_array',
            counts=custom_counts,
            feature_meta=custom_meta,
            distribution='normal',
            cell_names=self.cell_names,
            overwrite=True,
        )
        mod = self.model.get_modality('custom_array')
        self.assertIsNotNone(mod.cell_names)
        self.assertEqual(len(mod.cell_names), self.n_cells)
        self.assertEqual(mod.cell_names, self.cell_names)

    def test_dataframe_auto_extracts_cell_names(self):
        custom_counts_df = pd.DataFrame(
            np.random.randn(10, self.n_cells),
            index=[f'df_feature_{i}' for i in range(10)],
            columns=self.cell_names,
        )
        custom_meta = pd.DataFrame({'feature': [f'df_feature_{i}' for i in range(10)]})
        self.model.add_custom_modality(
            name='custom_dataframe',
            counts=custom_counts_df,
            feature_meta=custom_meta,
            distribution='normal',
        )
        mod = self.model.get_modality('custom_dataframe')
        self.assertIsNotNone(mod.cell_names)
        self.assertEqual(mod.cell_names, self.cell_names)

    def test_cell_subset_preserves_cell_names(self):
        # Ensure custom_array modality exists
        if 'custom_array' not in self.model.modalities:
            self.test_numpy_array_with_explicit_cell_names()
        mod = self.model.get_modality('custom_array')
        subset_cells = self.cell_names[:20]
        subset = mod.get_cell_subset(subset_cells)
        self.assertIsNotNone(subset.cell_names)
        self.assertEqual(len(subset.cell_names), 20)
        self.assertEqual(subset.cell_names, subset_cells)

    def test_no_cell_names_legacy_behavior(self):
        custom_counts = np.random.randn(8, self.n_cells)
        custom_meta = pd.DataFrame({'feature': [f'no_names_feature_{i}' for i in range(8)]})
        self.model.add_custom_modality(
            name='custom_no_names',
            counts=custom_counts,
            feature_meta=custom_meta,
            distribution='normal',
        )
        mod = self.model.get_modality('custom_no_names')
        self.assertIsNone(mod.cell_names)


if __name__ == '__main__':
    unittest.main()
