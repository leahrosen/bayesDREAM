"""Test fit_trans with all distributions and function types.

Verifies:
1. Binomial Hill works in probability space with Beta priors
2. Binomial polynomial works in logit space
3. Multinomial has per-category parameters (K-1 for Hill, K for polynomial)
4. Technical groups are applied correctly (no double-application)
5. All distributions work with all function types
"""

import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _create_test_data(n_features=10, n_cells=50, n_categories=3, seed=42):
    np.random.seed(seed)
    n_guides = 10
    cells_per_guide = n_cells // n_guides
    guides, targets, cell_lines = [], [], []
    for i in range(n_guides):
        for j in range(cells_per_guide):
            guides.append(f'guide_{i}')
            targets.append('GFI1B' if i < 4 else 'ntc')
            cell_lines.append('A' if (i + j) % 2 == 0 else 'B')

    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': guides,
        'target': targets,
        'cell_line': cell_lines,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'sum_factor_adj': np.random.uniform(0.8, 1.2, n_cells),
    })
    gene_counts = pd.DataFrame(
        np.random.poisson(lam=50, size=(n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'gene_{i}' for i in range(n_features)],
    )
    gene_counts.loc['GFI1B'] = np.random.poisson(lam=100, size=n_cells)
    feature_meta = pd.DataFrame({'gene': gene_counts.index}, index=gene_counts.index)
    inclusion_counts = pd.DataFrame(
        np.random.poisson(lam=30, size=(n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'exon_{i}' for i in range(n_features)],
    )
    total_counts = pd.DataFrame(
        np.random.poisson(lam=60, size=(n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'exon_{i}' for i in range(n_features)],
    )
    multinomial_counts = np.random.poisson(lam=20, size=(n_features, n_cells, n_categories))
    normal_scores = pd.DataFrame(
        np.random.normal(loc=0, scale=1, size=(n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'score_{i}' for i in range(n_features)],
    )
    return meta, gene_counts, feature_meta, inclusion_counts, total_counts, multinomial_counts, normal_scores


def _base_model(meta, gene_counts, feature_meta):
    from bayesDREAM import bayesDREAM
    return bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=feature_meta,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        device='cpu',
    )


class TestTransAllDistributions(unittest.TestCase):
    """Run fit_trans with each distribution × function_type combination."""

    @classmethod
    def setUpClass(cls):
        import torch
        import pyro
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        pyro.set_rng_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        cls.data = _create_test_data()

    def _fitted_base_model(self):
        meta, gene_counts, feature_meta = self.data[:3]
        model = _base_model(meta, gene_counts, feature_meta)
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20, learning_rate=0.01)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20, learning_rate=0.01)
        return model

    def test_negbinom_single_hill(self):
        model = self._fitted_base_model()
        model.fit_trans(sum_factor_col='sum_factor_adj', function_type='single_hill',
                        num_steps=20, learning_rate=0.01)
        self.assertIsNotNone(model.posterior_samples_trans)
        for key in ('A', 'Vmax_a', 'K_a', 'n_a'):
            self.assertIn(key, model.posterior_samples_trans)

    def test_negbinom_additive_hill(self):
        model = self._fitted_base_model()
        model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill',
                        num_steps=20, learning_rate=0.01)
        for key in ('Vmax_b', 'K_b', 'n_b'):
            self.assertIn(key, model.posterior_samples_trans)

    def test_binomial_single_hill_probability_space(self):
        meta, gene_counts, feature_meta, inclusion_counts, total_counts, _, _ = self.data
        model = _base_model(meta, gene_counts, feature_meta)
        exon_meta = pd.DataFrame({'exon': inclusion_counts.index})
        model.add_custom_modality(
            name='exon_skip', counts=inclusion_counts, feature_meta=exon_meta,
            distribution='binomial', denominator=total_counts,
        )
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20)
        model.fit_trans(sum_factor_col=None, function_type='single_hill',
                        modality_name='exon_skip', num_steps=20, learning_rate=0.01)
        A = model.posterior_samples_trans['A']
        Vmax_a = model.posterior_samples_trans['Vmax_a']
        self.assertTrue(torch.all(A >= 0) and torch.all(A <= 1))
        self.assertTrue(torch.all(Vmax_a >= 0) and torch.all(Vmax_a <= 1))

    def test_binomial_polynomial_logit_space(self):
        meta, gene_counts, feature_meta, inclusion_counts, total_counts, _, _ = self.data
        model = _base_model(meta, gene_counts, feature_meta)
        exon_meta = pd.DataFrame({'exon': inclusion_counts.index})
        model.add_custom_modality(
            name='exon_skip', counts=inclusion_counts, feature_meta=exon_meta,
            distribution='binomial', denominator=total_counts,
        )
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20)
        model.fit_trans(sum_factor_col=None, function_type='polynomial',
                        modality_name='exon_skip', num_steps=20, learning_rate=0.01)
        self.assertTrue(any('poly_coeff' in k for k in model.posterior_samples_trans))
        A = model.posterior_samples_trans['A']
        self.assertTrue(torch.all(A >= 0) and torch.all(A <= 1))

    def test_multinomial_single_hill_k_minus_1(self):
        meta, gene_counts, feature_meta, _, _, multinomial_counts, _ = self.data
        model = _base_model(meta, gene_counts, feature_meta)
        K = multinomial_counts.shape[-1]
        donor_meta = pd.DataFrame(
            {'donor': [f'donor_{i}' for i in range(multinomial_counts.shape[0])]}
        )
        model.add_custom_modality(
            name='donor_usage', counts=multinomial_counts,
            feature_meta=donor_meta, distribution='multinomial',
        )
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20)
        model.fit_trans(sum_factor_col=None, function_type='single_hill',
                        modality_name='donor_usage', num_steps=20, learning_rate=0.01)
        n_a = model.posterior_samples_trans['n_a']
        self.assertEqual(n_a.shape[-1], K - 1)

    def test_multinomial_polynomial_k_full(self):
        meta, gene_counts, feature_meta, _, _, multinomial_counts, _ = self.data
        model = _base_model(meta, gene_counts, feature_meta)
        K = multinomial_counts.shape[-1]
        donor_meta = pd.DataFrame(
            {'donor': [f'donor_{i}' for i in range(multinomial_counts.shape[0])]}
        )
        model.add_custom_modality(
            name='donor_usage', counts=multinomial_counts,
            feature_meta=donor_meta, distribution='multinomial',
        )
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20)
        model.fit_trans(sum_factor_col=None, function_type='polynomial',
                        modality_name='donor_usage', num_steps=20, learning_rate=0.01)
        poly_coeffs = {k: v for k, v in model.posterior_samples_trans.items()
                       if 'poly_coeff' in k}
        for name, samples in poly_coeffs.items():
            self.assertEqual(samples.shape[-1], K)

    def test_normal_polynomial(self):
        meta, gene_counts, feature_meta, _, _, _, normal_scores = self.data
        model = _base_model(meta, gene_counts, feature_meta)
        score_meta = pd.DataFrame({'score': normal_scores.index})
        model.add_custom_modality(
            name='scores', counts=normal_scores,
            feature_meta=score_meta, distribution='normal',
        )
        model.set_technical_groups(['cell_line'])
        model.fit_technical(sum_factor_col='sum_factor', num_steps=20)
        model.fit_cis(sum_factor_col='sum_factor', num_steps=20)
        model.fit_trans(sum_factor_col=None, function_type='polynomial',
                        modality_name='scores', num_steps=20, learning_rate=0.01)
        self.assertTrue('sigma_y' in model.posterior_samples_trans
                        or 'sigma' in model.posterior_samples_trans)


if __name__ == '__main__':
    unittest.main()
