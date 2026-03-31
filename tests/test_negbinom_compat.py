"""Test backward compatibility with negbinom distribution for fit_trans."""

import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_toy_data(n_cells=30, n_genes=15, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['ntc'] * 8 + ['gRNA1'] * 7 + ['ntc'] * 7 + ['gRNA1'] * 8,
        'target': ['ntc'] * 8 + ['GFI1B'] * 7 + ['ntc'] * 7 + ['GFI1B'] * 8,
        'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
        'cell_line': ['K562'] * 15 + ['Jurkat'] * 15,
    })
    gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
    counts = pd.DataFrame(
        np.random.poisson(50, (n_genes, n_cells)),
        index=gene_names,
        columns=meta['cell'],
    )
    return meta, counts


class TestNegbinomCompat(unittest.TestCase):
    """Ensure fit_trans with negbinom still works after refactoring."""

    @classmethod
    def setUpClass(cls):
        import torch
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM
        meta, counts = _make_toy_data()
        cls.model = bayesDREAM(
            meta=meta,
            counts=counts,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='negbinom_compat_test',
        )
        cls.model.set_technical_groups(['cell_line'])
        cls.model.fit_technical(sum_factor_col='sum_factor', niters=10, nsamples=10)

        # Set dummy x_true for trans testing
        n_guides = len(cls.model.meta)
        cls.model.x_true = torch.ones(n_guides, device=cls.model.device)
        cls.model.x_true_type = 'point'
        cls.model.log2_x_true = torch.log2(cls.model.x_true)
        cls.model.log2_x_true_type = 'point'

    def test_fit_trans_negbinom_runs(self):
        self.model.fit_trans(
            sum_factor_col='sum_factor',
            distribution='negbinom',
            function_type='single_hill',
            niters=50,
            lr=1e-2,
            p0=0.01,
            gamma_threshold=0.01,
            nsamples=10,
        )

    def test_posterior_samples_created(self):
        gene_modality = self.model.get_modality('gene')
        self.assertTrue(hasattr(gene_modality, 'posterior_samples_trans'))
        self.assertIsNotNone(gene_modality.posterior_samples_trans)


if __name__ == '__main__':
    unittest.main()
