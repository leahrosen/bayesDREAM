"""Test backward compatibility of fit_technical with negbinom distribution."""

import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_toy_data(n_cells=40, n_genes=15, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['ntc'] * 20 + ['gRNA1'] * 20,
        'target': ['ntc'] * 20 + ['GFI1B'] * 20,
        'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
        'cell_line': (['K562'] * 10 + ['HEK293T'] * 10) * 2,
    })
    gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
    counts = pd.DataFrame(
        np.random.poisson(200, (n_genes, n_cells)),
        index=gene_names,
        columns=meta['cell'],
    )
    return meta, counts


class TestTechnicalCompat(unittest.TestCase):
    """Ensure fit_technical still works with negbinom (backward-compat check)."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM
        meta, counts = _make_toy_data()
        cls.model = bayesDREAM(
            meta=meta,
            counts=counts,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='technical_compat_test',
        )
        cls.model.set_technical_groups(['cell_line'])

    def test_fit_technical_negbinom_runs(self):
        self.model.fit_technical(
            sum_factor_col='sum_factor',
            distribution='negbinom',
            niters=50,
            nsamples=10,
            lr=1e-2,
        )

    def test_alpha_y_prefit_set_in_modality(self):
        gene_modality = self.model.get_modality('gene')
        self.assertTrue(hasattr(gene_modality, 'alpha_y_prefit'))
        self.assertIsNotNone(gene_modality.alpha_y_prefit)

    def test_alpha_y_prefit_correct_shape(self):
        gene_modality = self.model.get_modality('gene')
        # Shape should be (n_samples, n_groups, n_genes)
        self.assertEqual(len(gene_modality.alpha_y_prefit.shape), 3)


if __name__ == '__main__':
    unittest.main()
