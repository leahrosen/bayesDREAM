"""Test gene metadata handling in bayesDREAM."""

import numpy as np
import pandas as pd

import pytest


def _make_base_data():
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(1, 21)],
        'guide': ['g1', 'g2', 'g3', 'g4', 'g5'] * 4,
        'target': ['GFI1B'] * 10 + ['ntc'] * 10,
        'cell_line': ['A'] * 10 + ['B'] * 10,
        'sum_factor': [1.0] * 20,
    })
    gene_counts = pd.DataFrame(
        np.random.randint(10, 100, (10, 20)),
        index=[f'GENE{i}' for i in range(10)],
        columns=[f'cell{i}' for i in range(1, 21)],
    )
    gene_counts.loc['GFI1B'] = np.random.randint(50, 150, 20)
    return meta, gene_counts


class TestGeneMeta(unittest.TestCase):
    """Verify gene metadata handling at model initialisation.

    After the refactoring, gene metadata lives in the gene modality's
    feature_meta (model.get_modality('gene').feature_meta), not on the
    model object directly.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.meta, cls.gene_counts = _make_base_data()

    def _make_model(self, **kwargs):
        from bayesDREAM import bayesDREAM
        return bayesDREAM(
            meta=self.meta,
            counts=self.gene_counts,
            cis_gene='GFI1B',
            output_dir='./test_output',
            **kwargs,
        )

    def _gene_feature_meta(self, model):
        return model.get_modality('gene').feature_meta

    def test_no_gene_meta_creates_minimal_metadata(self):
        model = self._make_model(label='test_no_meta')
        fm = self._gene_feature_meta(model)
        self.assertIsNotNone(fm)
        self.assertGreater(fm.shape[0], 0)

    def test_full_gene_meta_accepted(self):
        gene_meta = pd.DataFrame({
            'gene': [f'GENE{i}' for i in range(10)] + ['GFI1B'],
            'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
            'gene_id': [f'ENSG{i:08d}' for i in range(11)],
            'chromosome': ['chr1'] * 11,
            'biotype': ['protein_coding'] * 11,
        }, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])
        model = self._make_model(feature_meta=gene_meta, label='test_with_meta')
        self.assertIn('gene', self._gene_feature_meta(model).columns)

    def test_gene_meta_with_gene_name_only(self):
        gene_meta_simple = pd.DataFrame({
            'gene_name': [f'GENE{i}' for i in range(10)] + ['GFI1B'],
        }, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])
        model = self._make_model(feature_meta=gene_meta_simple, label='test_simple_meta')
        fm = self._gene_feature_meta(model)
        self.assertIn('gene_name', fm.columns, "'gene_name' column should be present")

    def test_gene_meta_index_becomes_gene_column(self):
        gene_meta_indexed = pd.DataFrame({
            'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
            'gene_id': [f'ENSG{i:08d}' for i in range(11)],
        })
        gene_meta_indexed.index = [f'GENE{i}' for i in range(10)] + ['GFI1B']
        gene_meta_indexed.index.name = 'gene_symbol'
        model = self._make_model(feature_meta=gene_meta_indexed, label='test_indexed_meta')
        fm = self._gene_feature_meta(model)
        self.assertIn('gene_name', fm.columns, "'gene_name' column should be present")


if __name__ == '__main__':
    unittest.main()
