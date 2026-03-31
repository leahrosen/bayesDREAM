"""Test summary export functionality with a full fitting pipeline.

Runs technical → cis → trans on toy CSV data and validates that all
summary export methods produce R-friendly CSV files with the expected columns.
"""

import os
import tempfile
import shutil
import unittest
import pandas as pd
import numpy as np

import pytest
pytestmark = pytest.mark.slow

TOYDATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'toydata')


@pytest.mark.skipif(
    not os.path.exists(os.path.join(TOYDATA_DIR, 'cell_meta.csv')),
    reason="toydata not found",
)
class TestSummaryExport(unittest.TestCase):
    """Run full pipeline and validate summary CSV outputs."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM

        cls.tmpdir = tempfile.mkdtemp()

        meta = pd.read_csv(os.path.join(TOYDATA_DIR, 'cell_meta.csv'))
        gene_counts = pd.read_csv(os.path.join(TOYDATA_DIR, 'gene_counts.csv'), index_col=0)
        gene_meta = pd.read_csv(os.path.join(TOYDATA_DIR, 'gene_meta.csv'))
        gene_meta.set_index('gene_name', inplace=True)

        cls.model = bayesDREAM(
            meta=meta,
            counts=gene_counts,
            feature_meta=gene_meta,
            cis_gene='GFI1B',
            guide_covariates=['cell_line'],
            output_dir=cls.tmpdir,
            label='summary_test',
            device='cpu',
        )
        cls.model.set_technical_groups(['cell_line'])
        cls.model.fit_technical(sum_factor_col='sum_factor', niters=5000, lr=0.001)
        cls.model.fit_cis(sum_factor_col='sum_factor', n_steps=500, lr=0.01)
        cls.model.fit_trans(
            sum_factor_col='sum_factor_adj',
            function_type='additive_hill',
            n_steps=500,
            lr=0.01,
        )
        cls.outdir = os.path.join(cls.tmpdir, 'summary_test')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_technical_summary_structure(self):
        tech_df = self.model.save_technical_summary()
        for col in ('feature', 'modality', 'distribution'):
            self.assertIn(col, tech_df.columns)
        alpha_cols = [c for c in tech_df.columns if 'alpha_y' in c]
        self.assertGreater(len(alpha_cols), 0)

    def test_technical_summary_csv(self):
        self.model.save_technical_summary()
        self.assertTrue(os.path.exists(
            os.path.join(self.outdir, 'technical_feature_summary_gene.csv')
        ))

    def test_cis_summary_guide_level_columns(self):
        guide_df, _ = self.model.save_cis_summary(include_cell_level=True)
        for col in ('guide', 'target', 'n_cells', 'x_true_mean', 'x_true_lower', 'x_true_upper'):
            self.assertIn(col, guide_df.columns)

    def test_cis_summary_cell_level_columns(self):
        _, cell_df = self.model.save_cis_summary(include_cell_level=True)
        for col in ('cell', 'guide', 'x_true_mean'):
            self.assertIn(col, cell_df.columns)

    def test_cis_summary_csvs(self):
        self.model.save_cis_summary()
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'cis_guide_summary.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'cis_cell_summary.csv')))

    def test_trans_summary_additive_hill_columns(self):
        trans_df = self.model.save_trans_summary(
            compute_inflection=True, compute_full_log2fc=True
        )
        for col in ('feature', 'modality', 'distribution', 'function_type',
                    'observed_log2fc', 'observed_log2fc_se', 'B_pos_mean', 'K_pos_mean',
                    'EC50_pos_mean', 'B_neg_mean', 'K_neg_mean', 'IC50_neg_mean',
                    'inflection_pos_mean', 'inflection_neg_mean', 'full_log2fc_mean'):
            self.assertIn(col, trans_df.columns)

    def test_trans_summary_csv(self):
        self.model.save_trans_summary(compute_inflection=True, compute_full_log2fc=True)
        self.assertTrue(os.path.exists(
            os.path.join(self.outdir, 'trans_feature_summary_gene.csv')
        ))

    def test_trans_summary_polynomial_columns(self):
        self.model.fit_trans(
            sum_factor_col='sum_factor_adj',
            function_type='polynomial',
            n_steps=500,
            lr=0.01,
        )
        poly_df = self.model.save_trans_summary(compute_inflection=False, compute_full_log2fc=True)
        coef_cols = [c for c in poly_df.columns if 'coef_' in c]
        self.assertGreater(len(coef_cols), 0)
        self.assertIn('full_log2fc_mean', poly_df.columns)


if __name__ == '__main__':
    unittest.main()
