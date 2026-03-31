"""Test summary export functionality using mock posterior data.

Validates all summary export methods without requiring a full fitting pipeline.
"""

import os
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd


def _make_toy_data(n_genes=10, n_cells=100, n_guides=6, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['K562', 'HEL'], n_cells),
        'target': ['GFI1B'] * 60 + ['ntc'] * 40,
        'sum_factor': np.random.lognormal(0, 0.2, n_cells),
        'sum_factor_adj': np.random.lognormal(0, 0.2, n_cells),
    })
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, (len(genes), n_cells)),
        index=genes,
        columns=meta['cell'],
    )
    return meta, gene_counts


class TestSummaryExportSimple(unittest.TestCase):
    """Verify save_technical_summary, save_cis_summary, and save_trans_summary."""

    @classmethod
    def setUpClass(cls):
        import torch
        from bayesDREAM import bayesDREAM
        cls.torch = torch
        cls.tmpdir = tempfile.mkdtemp()
        cls.meta, cls.gene_counts = _make_toy_data()
        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=cls.gene_counts,
            cis_gene='GFI1B',
            guide_covariates=['cell_line'],
            output_dir=cls.tmpdir,
            label='summary_test_simple',
            device='cpu',
        )
        cls.model.set_technical_groups(['cell_line'])

        # Inject mock technical fit posteriors
        gene_mod = cls.model.modalities['gene']
        n_features = gene_mod.dims['n_features']
        n_groups = 2
        gene_mod.alpha_y_prefit = torch.randn((100, n_groups, n_features), device=cls.model.device)
        gene_mod.alpha_y_type = 'posterior'

        # Inject mock cis fit posteriors (x_true is cell-level: [n_samples, n_cells])
        n_cells_total = len(cls.model.meta)
        cls.model.x_true = torch.randn(n_cells_total, device=cls.model.device) + 5
        cls.model.posterior_samples_cis = {
            'x_true': torch.randn((100, n_cells_total), device=cls.model.device) + 5
        }

        # Inject mock trans fit posteriors (additive_hill)
        n_features_trans = cls.model.modalities['gene'].dims['n_features']
        n_samples = 100
        cls.model.function_type = 'additive_hill'
        cls.model.posterior_samples_trans = {
            'Vmax_a': torch.randn((n_samples, n_features_trans), device=cls.model.device),
            'K_a': torch.randn((n_samples, n_features_trans), device=cls.model.device) + 5,
            'n_a': torch.abs(torch.randn((n_samples, n_features_trans), device=cls.model.device)) + 1.5,
            'Vmax_b': torch.randn((n_samples, n_features_trans), device=cls.model.device),
            'K_b': torch.randn((n_samples, n_features_trans), device=cls.model.device) + 5,
            'n_b': torch.abs(torch.randn((n_samples, n_features_trans), device=cls.model.device)) + 1.5,
            'pi_y': torch.rand((n_samples, n_features_trans), device=cls.model.device) * 0.5 + 0.5,
            'alpha': torch.rand((n_features_trans,), device=cls.model.device),
            'beta': torch.rand((n_features_trans,), device=cls.model.device),
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_technical_summary_columns(self):
        tech_df = self.model.save_technical_summary()
        for col in ('feature', 'modality', 'distribution'):
            self.assertIn(col, tech_df.columns)
        self.assertTrue(any('alpha_y' in col for col in tech_df.columns))

    def test_technical_summary_csv_created(self):
        self.model.save_technical_summary()
        csv_path = os.path.join(self.tmpdir, 'summary_test_simple', 'technical_feature_summary_gene.csv')
        self.assertTrue(os.path.exists(csv_path))

    def test_cis_summary_columns(self):
        guide_df, cell_df = self.model.save_cis_summary()
        for col in ('guide', 'target', 'x_true_mean', 'x_true_lower', 'x_true_upper'):
            self.assertIn(col, guide_df.columns)

    def test_cis_summary_csvs_created(self):
        self.model.save_cis_summary()
        outdir = os.path.join(self.tmpdir, 'summary_test_simple')
        self.assertTrue(os.path.exists(os.path.join(outdir, 'cis_guide_summary.csv')))
        self.assertTrue(os.path.exists(os.path.join(outdir, 'cis_cell_summary.csv')))

    def test_trans_summary_additive_hill_columns(self):
        trans_df = self.model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
        for col in ('feature', 'function_type', 'observed_log2fc', 'Vmax_a_mean',
                    'EC50_a_mean', 'inflection_a_mean', 'full_log2fc_mean'):
            self.assertIn(col, trans_df.columns)

    def test_trans_summary_csv_created(self):
        self.model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
        outdir = os.path.join(self.tmpdir, 'summary_test_simple')
        self.assertTrue(os.path.exists(os.path.join(outdir, 'trans_feature_summary_gene.csv')))

    def test_trans_summary_polynomial_columns(self):
        import torch
        n_features_trans = self.model.modalities['gene'].dims['n_features']
        n_samples = 100
        degree = 6
        self.model.function_type = 'polynomial'
        self.model.posterior_samples_trans = {
            'poly_coefs': torch.randn((n_samples, n_features_trans, degree),
                                      device=self.model.device)
        }
        poly_df = self.model.save_trans_summary(compute_inflection=False, compute_full_log2fc=True)
        self.assertIn('coef_0_mean', poly_df.columns)
        self.assertIn('full_log2fc_mean', poly_df.columns)


if __name__ == '__main__':
    unittest.main()
