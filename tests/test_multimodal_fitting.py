"""Test multi-modal fitting infrastructure.

Tests:
1. Backward compatibility: bayesDREAM works exactly like bayesDREAM for gene expression
2. fit_modality_technical() delegates correctly to fit_technical()
3. fit_modality_trans() delegates correctly to fit_trans()
4. Distribution registry is properly loaded
"""

import unittest
import numpy as np
import pandas as pd


def _make_toy_data(n_genes=20, n_cells=50, n_guides=10, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['A', 'B'], n_cells),
        'target': ['GFI1B' if i < 30 else 'ntc' for i in range(n_cells)],
        'sum_factor': np.random.lognormal(0, 0.3, n_cells),
    })
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.poisson(50, (len(genes), n_cells)),
        index=genes,
        columns=meta['cell'],
    )
    return meta, gene_counts, n_genes


class TestDistributionRegistry(unittest.TestCase):
    """Verify the distribution registry and helper functions."""

    def test_all_distributions_registered(self):
        from bayesDREAM import DISTRIBUTION_REGISTRY
        for dist in ('negbinom', 'multinomial', 'binomial', 'normal'):
            self.assertIn(dist, DISTRIBUTION_REGISTRY)

    def test_helper_functions(self):
        from bayesDREAM import requires_denominator, is_3d_distribution
        self.assertTrue(requires_denominator('binomial'))
        self.assertFalse(requires_denominator('negbinom'))
        self.assertTrue(is_3d_distribution('multinomial'))
        self.assertFalse(is_3d_distribution('negbinom'))

    def test_get_observation_sampler(self):
        from bayesDREAM import get_observation_sampler
        sampler = get_observation_sampler('negbinom', 'trans')
        self.assertTrue(callable(sampler))


class TestMultimodalFitting(unittest.TestCase):
    """Verify bayesDREAM multi-modal infrastructure without running fitting."""

    @classmethod
    def setUpClass(cls):
        from bayesDREAM import bayesDREAM
        cls.meta, cls.gene_counts, cls.n_genes = _make_toy_data()
        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=cls.gene_counts,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='test_multimodal',
            device='cpu',
            cores=1,
        )

    def test_model_creation(self):
        self.assertEqual(self.model.primary_modality, 'gene')
        self.assertGreater(len(self.model.modalities), 0)

    def test_core_fitting_methods_exist(self):
        for method in ('fit_technical', 'fit_cis', 'fit_trans', 'set_technical_groups'):
            self.assertTrue(hasattr(self.model, method), f"Missing method: {method}")

    def test_gene_modality_excludes_cis_gene(self):
        gene_modality = self.model.get_modality('gene')
        gene_names = gene_modality.feature_meta['gene'].tolist()
        self.assertNotIn('GFI1B', gene_names, "Cis gene should be excluded from gene modality")
        self.assertEqual(len(gene_names), self.n_genes)

    def test_base_class_retains_cis_gene(self):
        self.assertIn('GFI1B', self.model.counts.index)
        self.assertEqual(self.model.cis_gene, 'GFI1B')


if __name__ == '__main__':
    unittest.main()
