"""Test high MOI (multiple guides per cell) functionality.

Verifies:
1. Initialization with guide_assignment matrix and guide_meta
2. Backward compatibility (single-guide mode still works)
3. Additive guide effects in fit_cis
4. Proper handling of NTC cells
"""

import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_high_moi_data(n_cells=100, n_genes=50, n_guides=6, seed=42):
    import torch
    import pyro
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'cell_line': np.random.choice(['K562', 'MOLM13'], n_cells),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)],
    )
    guide_assignment = np.zeros((n_cells, n_guides), dtype=int)
    guide_assignment[0:30, 0] = 1
    guide_assignment[0:30, 1] = 1
    guide_assignment[30:60, 2] = 1
    guide_assignment[60:80, 3] = 1
    guide_assignment[60:80, 4] = 1
    guide_assignment[80:100, 5] = 1
    guide_meta = pd.DataFrame({
        'guide': ['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'],
        'target': ['GFI1B', 'GFI1B', 'MYB', 'MYB', 'MYB', 'ntc'],
    }, index=['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'])
    return meta, counts, guide_assignment, guide_meta, n_guides


def _make_single_guide_data(n_cells=50, n_genes=30, seed=42):
    import torch
    import pyro
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['guide_A'] * 25 + ['ntc'] * 25,
        'target': ['GFI1B'] * 25 + ['ntc'] * 25,
        'cell_line': ['K562'] * n_cells,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)],
    )
    return meta, counts


class TestHighMOIInitialization(unittest.TestCase):
    """Verify high MOI model initialization."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM
        meta, counts, guide_assignment, guide_meta, n_guides = _make_high_moi_data()
        cls.n_guides = n_guides
        cls.model = bayesDREAM(
            meta=meta,
            counts=counts,
            guide_assignment=guide_assignment,
            guide_meta=guide_meta,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='test_high_moi',
            device='cpu',
        )

    def test_high_moi_mode_active(self):
        self.assertTrue(self.model.is_high_moi)

    def test_guide_assignment_attribute(self):
        self.assertTrue(hasattr(self.model, 'guide_assignment'))
        self.assertTrue(hasattr(self.model, 'guide_meta'))
        self.assertTrue(hasattr(self.model, 'guide_assignment_tensor'))

    def test_guide_meta_length(self):
        self.assertEqual(len(self.model.guide_meta), self.n_guides)

    def test_target_column_created(self):
        self.assertIn('target', self.model.meta.columns)

    def test_ntc_cell_count(self):
        ntc_cells = (self.model.meta['target'] == 'ntc').sum()
        self.assertEqual(ntc_cells, 20)

    def test_targeting_cell_count(self):
        targeting_cells = (self.model.meta['target'] == 'GFI1B').sum()
        self.assertEqual(targeting_cells, 30)

    def test_guide_code_is_placeholder(self):
        self.assertTrue((self.model.meta['guide_code'] == -1).all())


class TestHighMOIBackwardCompat(unittest.TestCase):
    """Verify single-guide mode is unaffected by high-MOI changes."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM
        meta, counts = _make_single_guide_data()
        cls.model = bayesDREAM(
            meta=meta,
            counts=counts,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='test_single_guide',
            device='cpu',
        )

    def test_not_high_moi_mode(self):
        self.assertFalse(self.model.is_high_moi)

    def test_no_guide_assignment_attribute(self):
        self.assertFalse(hasattr(self.model, 'guide_assignment'))

    def test_guide_code_not_placeholder(self):
        self.assertIn('guide_code', self.model.meta.columns)
        self.assertFalse((self.model.meta['guide_code'] == -1).all())


class TestHighMOICisFitting(unittest.TestCase):
    """Run fit_cis in high-MOI mode and verify output shapes."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM
        meta, counts, guide_assignment, guide_meta, _ = _make_high_moi_data()
        cls.model = bayesDREAM(
            meta=meta,
            counts=counts,
            guide_assignment=guide_assignment,
            guide_meta=guide_meta,
            cis_gene='GFI1B',
            output_dir='./test_output',
            label='test_high_moi_cis',
            device='cpu',
        )
        cls.model.fit_cis(
            sum_factor_col='sum_factor',
            lr=1e-2,
            niters=500,
            nsamples=10,
            tolerance=1e-3,
        )

    def test_posterior_samples_cis_exists(self):
        self.assertTrue(hasattr(self.model, 'posterior_samples_cis'))

    def test_x_true_in_posterior(self):
        self.assertIn('x_true', self.model.posterior_samples_cis)

    def test_x_eff_g_in_posterior(self):
        self.assertIn('x_eff_g', self.model.posterior_samples_cis)

    def test_x_true_cell_count(self):
        x_true = self.model.posterior_samples_cis['x_true']
        self.assertEqual(x_true.shape[1], len(self.model.meta))

    def test_x_eff_g_guide_count(self):
        x_eff_g = self.model.posterior_samples_cis['x_eff_g']
        self.assertEqual(x_eff_g.shape[1], self.model.guide_assignment.shape[1])


if __name__ == '__main__':
    unittest.main()
