"""Test modality-specific save/load functionality.

Verifies:
1. Users can save specific modalities only
2. Primary modality is NOT saved by default - must be in the list
3. save_model_level flag works correctly
4. Loading specific modalities works correctly
"""

import os
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd

import pytest
pytestmark = pytest.mark.slow


def _make_test_data(n_cells=60, seed=42):
    np.random.seed(seed)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['NTC'] * 30 + ['guide1'] * 30,
        'cell_line': ['A'] * 20 + ['B'] * 20 + ['C'] * 20,
        'target': ['ntc'] * 30 + ['GFI1B'] * 30,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'sum_factor_adj': np.random.uniform(0.8, 1.2, n_cells),
    })
    genes = [f'gene_{i}' for i in range(10)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.poisson(100, (11, n_cells)),
        index=genes,
        columns=meta['cell'],
    )
    atac_counts = np.random.poisson(50, (5, n_cells))
    atac_meta = pd.DataFrame({
        'region': [f'region_{i}' for i in range(5)],
        'chrom': 'chr1',
        'start': np.arange(5) * 1000,
        'end': np.arange(5) * 1000 + 500,
    })
    splicing_counts = np.random.poisson(20, (3, n_cells))
    splicing_denom = np.random.poisson(100, (3, n_cells))
    splicing_meta = pd.DataFrame({
        'junction': [f'junction_{i}' for i in range(3)],
        'gene': ['gene_0', 'gene_1', 'gene_2'],
    })
    return meta, gene_counts, atac_counts, atac_meta, splicing_counts, splicing_denom, splicing_meta


class TestModalitySaveLoad(unittest.TestCase):
    """Verify per-modality save/load respects modality selection and flags."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip('torch')
        pytest.importorskip('pyro')
        from bayesDREAM import bayesDREAM, Modality

        cls.tmpdir = tempfile.mkdtemp()
        (cls.meta, cls.gene_counts, cls.atac_counts, cls.atac_meta,
         cls.splicing_counts, cls.splicing_denom, cls.splicing_meta) = _make_test_data()

        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=cls.gene_counts,
            cis_gene='GFI1B',
            output_dir=os.path.join(cls.tmpdir, 'test1'),
            label='modality_test',
            primary_modality='gene',
        )

        cls.atac_modality = Modality(
            name='atac',
            counts=cls.atac_counts,
            feature_meta=cls.atac_meta,
            cell_names=cls.meta['cell'].values,
            distribution='negbinom',
        )
        cls.model.modalities['atac'] = cls.atac_modality

        cls.splicing_modality = Modality(
            name='splicing',
            counts=cls.splicing_counts,
            feature_meta=cls.splicing_meta,
            cell_names=cls.meta['cell'].values,
            distribution='binomial',
            denominator=cls.splicing_denom,
        )
        cls.model.modalities['splicing'] = cls.splicing_modality

        cls.model.set_technical_groups(['cell_line'])
        for mod_name in ['gene', 'atac', 'splicing']:
            cls.model.fit_technical(modality_name=mod_name, sum_factor_col='sum_factor', niters=10)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_save_excludes_primary_by_default(self):
        outdir = os.path.join(self.tmpdir, 'test_excl_primary')
        os.makedirs(outdir, exist_ok=True)
        self.model.save_technical_fit(
            output_dir=outdir,
            modalities=['atac', 'splicing'],
            save_model_level=False,
        )
        self.assertFalse(os.path.exists(os.path.join(outdir, 'alpha_y_prefit_gene.pt')),
                         "Primary 'gene' should NOT be saved when not listed")
        self.assertTrue(os.path.exists(os.path.join(outdir, 'alpha_y_prefit_atac.pt')))
        self.assertTrue(os.path.exists(os.path.join(outdir, 'alpha_y_prefit_splicing.pt')))

    def test_save_model_level_false_skips_model_params(self):
        outdir = os.path.join(self.tmpdir, 'test_no_model_level')
        os.makedirs(outdir, exist_ok=True)
        self.model.save_technical_fit(
            output_dir=outdir,
            modalities=['atac'],
            save_model_level=False,
        )
        self.assertFalse(os.path.exists(os.path.join(outdir, 'alpha_x_prefit.pt')))
        self.assertFalse(os.path.exists(os.path.join(outdir, 'alpha_y_prefit.pt')))

    def test_save_primary_explicitly(self):
        outdir = os.path.join(self.tmpdir, 'test_explicit_primary')
        os.makedirs(outdir, exist_ok=True)
        self.model.save_technical_fit(
            output_dir=outdir,
            modalities=['gene'],
            save_model_level=True,
        )
        self.assertTrue(os.path.exists(os.path.join(outdir, 'alpha_y_prefit_gene.pt')))
        self.assertFalse(os.path.exists(os.path.join(outdir, 'alpha_y_prefit_atac.pt')))
        self.assertTrue(os.path.exists(os.path.join(outdir, 'alpha_y_prefit.pt')))

    def test_load_specific_modalities(self):
        from bayesDREAM import bayesDREAM

        # Save atac + splicing only
        outdir = os.path.join(self.tmpdir, 'test_load_specific')
        os.makedirs(outdir, exist_ok=True)
        self.model.save_technical_fit(
            output_dir=outdir,
            modalities=['atac', 'splicing'],
            save_model_level=False,
        )

        model2 = bayesDREAM(
            meta=self.meta,
            counts=self.gene_counts,
            cis_gene='GFI1B',
            output_dir=outdir,
            label='modality_test',
            primary_modality='gene',
        )
        model2.modalities['atac'] = self.atac_modality
        model2.modalities['splicing'] = self.splicing_modality

        model2.load_technical_fit(modalities=['atac', 'splicing'], load_model_level=False)

        gene_mod = model2.modalities['gene']
        atac_mod = model2.modalities['atac']
        spl_mod = model2.modalities['splicing']

        gene_loaded = hasattr(gene_mod, 'alpha_y_prefit') and gene_mod.alpha_y_prefit is not None
        self.assertFalse(gene_loaded, "gene should NOT be loaded")
        self.assertIsNotNone(atac_mod.alpha_y_prefit)
        self.assertIsNotNone(spl_mod.alpha_y_prefit)

    def test_default_save_includes_all_modalities(self):
        outdir = os.path.join(self.tmpdir, 'test_save_all')
        os.makedirs(outdir, exist_ok=True)
        self.model.save_technical_fit(output_dir=outdir)
        for name in ['gene', 'atac', 'splicing']:
            self.assertTrue(
                os.path.exists(os.path.join(outdir, f'alpha_y_prefit_{name}.pt')),
                f"alpha_y_prefit_{name}.pt should exist when modalities=None",
            )


if __name__ == '__main__':
    unittest.main()
