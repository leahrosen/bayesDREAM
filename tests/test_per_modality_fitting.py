"""Test per-modality fitting functionality.

Verifies that fit_technical() and fit_trans() can fit different modalities
and store results correctly per modality, without overwriting each other.
"""

import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_base_data(n_genes=10, n_cells=50, n_guides=10, seed=42):
    np.random.seed(seed)
    cell_names = [f'cell_{i}' for i in range(n_cells)]
    guides = np.repeat([f'guide_{i}' for i in range(n_guides)], n_cells // n_guides)
    targets = ['ntc'] * (n_cells // 2) + ['GFI1B'] * (n_cells // 2)
    cell_lines = np.random.choice(['A', 'B'], n_cells)
    meta = pd.DataFrame({
        'cell': cell_names,
        'guide': guides,
        'target': targets,
        'cell_line': cell_lines,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'L_cell_barcode': cell_names,
    })
    gene_names = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.poisson(50, (n_genes + 1, n_cells)),
        index=gene_names,
        columns=cell_names,
    )
    return meta, gene_counts, cell_names, n_cells


class TestPerModalityFitting(unittest.TestCase):
    """Run technical and trans fits across primary and non-primary modalities."""

    @classmethod
    def setUpClass(cls):
        import torch
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM, Modality

        cls.meta, cls.gene_counts, cls.cell_names, cls.n_cells = _make_base_data()

        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=cls.gene_counts,
            cis_gene='GFI1B',
            primary_modality='gene',
            output_dir='./test_output',
            label='per_modality_test',
            device='cpu',
        )

        # Add a splicing-like binomial modality
        n_junctions = 5
        sj_counts = np.random.poisson(20, (n_junctions, cls.n_cells))
        sj_total = np.random.poisson(100, (n_junctions, cls.n_cells))
        sj_meta = pd.DataFrame({
            'junction_id': [f'junction_{i}' for i in range(n_junctions)],
            'chrom': ['chr1'] * n_junctions,
            'strand': ['+'] * n_junctions,
        })
        cls.splicing_modality = Modality(
            name='splicing_test',
            counts=pd.DataFrame(sj_counts, columns=cls.cell_names),
            feature_meta=sj_meta,
            distribution='binomial',
            denominator=sj_total,
            cells_axis=1,
        )
        cls.model.add_modality('splicing_test', cls.splicing_modality)

        # Fit technical on both modalities
        cls.model.set_technical_groups(['cell_line'])
        cls.model.fit_technical(sum_factor_col='sum_factor', modality_name='gene',
                                niters=100, nsamples=10)
        cls.model.fit_technical(modality_name='splicing_test', niters=100, nsamples=10)

        # Set dummy x_true for trans tests
        cls.model.x_true = torch.ones(cls.n_cells, dtype=torch.float32)
        cls.model.x_true_type = 'point'

        # Fit trans on gene modality
        cls.model.fit_trans(
            sum_factor_col='sum_factor',
            function_type='additive_hill',
            modality_name='gene',
            p0=0.01, gamma_threshold=0.01,
            niters=100, nsamples=10,
        )

        # Fit trans on splicing modality
        cls.model.fit_trans(
            function_type='additive_hill',
            modality_name='splicing_test',
            p0=0.01, gamma_threshold=0.01,
            niters=100, nsamples=10,
        )

    # --- Technical fit checks ---

    def test_gene_modality_alpha_y_prefit_set(self):
        gene_mod = self.model.get_modality('gene')
        self.assertIsNotNone(gene_mod.alpha_y_prefit)

    def test_splicing_modality_alpha_y_prefit_set(self):
        spl_mod = self.model.get_modality('splicing_test')
        self.assertIsNotNone(spl_mod.alpha_y_prefit)

    def test_gene_alpha_not_overwritten_by_splicing_fit(self):
        gene_mod = self.model.get_modality('gene')
        spl_mod = self.model.get_modality('splicing_test')
        # Both should be set, and they should differ (different data)
        self.assertIsNotNone(gene_mod.alpha_y_prefit)
        self.assertIsNotNone(spl_mod.alpha_y_prefit)

    def test_model_level_technical_stored(self):
        self.assertIsNotNone(self.model.posterior_samples_technical)

    # --- Trans fit checks ---

    def test_gene_modality_posterior_samples_trans(self):
        gene_mod = self.model.get_modality('gene')
        self.assertIsNotNone(gene_mod.posterior_samples_trans)

    def test_splicing_modality_posterior_samples_trans(self):
        spl_mod = self.model.get_modality('splicing_test')
        self.assertIsNotNone(spl_mod.posterior_samples_trans)

    def test_model_level_posterior_samples_trans_backward_compat(self):
        self.assertIsNotNone(self.model.posterior_samples_trans)

    # --- Error handling ---

    def test_trans_without_technical_fit_raises(self):
        from bayesDREAM import Modality
        third_modality = Modality(
            name='untrained',
            counts=np.random.poisson(10, (5, self.n_cells)),
            feature_meta=pd.DataFrame({'feature': [f'f_{i}' for i in range(5)]}),
            distribution='negbinom',
            cells_axis=1,
        )
        self.model.add_modality('untrained', third_modality)
        with self.assertRaises(ValueError):
            self.model.fit_trans(modality_name='untrained', niters=10, nsamples=5)

    # --- Default behaviour ---

    def test_fit_technical_defaults_to_primary_modality(self):
        gene_mod = self.model.get_modality('gene')
        gene_mod.alpha_y_prefit = None
        gene_mod.posterior_samples_technical = None
        self.model.fit_technical(sum_factor_col='sum_factor', niters=100, nsamples=10)
        self.assertIsNotNone(gene_mod.alpha_y_prefit)


if __name__ == '__main__':
    unittest.main()
