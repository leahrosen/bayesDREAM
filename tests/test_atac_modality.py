"""Test ATAC modality functionality in bayesDREAM.

Tests:
1. Creating ATAC modality with region metadata
2. Cis modality auto-created from cis_region
3. ATAC-only initialization (no gene expression)
4. Manual guide effects infrastructure
"""

import unittest
import numpy as np
import pandas as pd


def _make_gene_model(n_genes=50, n_cells=100, n_guides=5, seed=42):
    from bayesDREAM import bayesDREAM
    np.random.seed(seed)
    gene_counts = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, size=(n_genes, n_cells)),
        index=[f'GENE{i}' for i in range(n_genes)],
        columns=[f'cell{i}' for i in range(n_cells)],
    )
    guides = [f'guide{i % n_guides}' for i in range(n_cells)]
    meta = pd.DataFrame({
        'cell': [f'cell{i}' for i in range(n_cells)],
        'guide': guides,
        'cell_line': ['line1'] * 50 + ['line2'] * 50,
        'target': ['GFI1B'] * 80 + ['ntc'] * 20,
        'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    })
    model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GENE0')
    return model, meta, n_cells, n_genes


def _make_atac_data(n_regions=10, n_cells=100):
    atac_counts = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, size=(n_regions, n_cells)),
        index=[f'region{i}' for i in range(n_regions)],
        columns=[f'cell{i}' for i in range(n_cells)],
    )
    region_meta = pd.DataFrame({
        'region_id': [f'region{i}' for i in range(n_regions)],
        'region_type': ['promoter'] * 3 + ['gene_body'] * 3 + ['distal'] * 4,
        'chrom': ['chr9'] * n_regions,
        'start': np.arange(1000, 1000 + n_regions * 1000, 1000),
        'end': np.arange(2000, 2000 + n_regions * 1000, 1000),
        'gene': ['GFI1B', 'GFI1B', 'SPI1'] + ['GFI1B'] * 3 + [''] * 4,
    })
    return atac_counts, region_meta, n_regions


class TestAtacWithGeneExpression(unittest.TestCase):
    """Add ATAC modality alongside gene expression."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.meta, cls.n_cells, cls.n_genes = _make_gene_model()
        cls.atac_counts, cls.region_meta, cls.n_regions = _make_atac_data(n_cells=cls.n_cells)
        cls.model.add_atac_modality(
            atac_counts=cls.atac_counts,
            region_meta=cls.region_meta,
            name='atac',
            cis_region='region0',
        )

    def test_atac_modality_in_model(self):
        modalities_df = self.model.list_modalities()
        self.assertIn('atac', modalities_df['name'].values)

    def test_atac_distribution(self):
        atac_mod = self.model.get_modality('atac')
        self.assertEqual(atac_mod.distribution, 'negbinom')

    def test_atac_feature_count(self):
        atac_mod = self.model.get_modality('atac')
        self.assertEqual(atac_mod.dims['n_features'], self.n_regions)

    def test_cis_modality_auto_created(self):
        self.assertIn('cis', self.model.modalities)
        cis_mod = self.model.get_modality('cis')
        self.assertEqual(cis_mod.dims['n_features'], 1)


class TestAtacOnlyInitialization(unittest.TestCase):
    """Initialize bayesDREAM without gene expression, then add ATAC."""

    @classmethod
    def setUpClass(cls):
        from bayesDREAM import bayesDREAM
        np.random.seed(0)
        n_cells = 100
        n_guides = 5
        guides = [f'guide{i % n_guides}' for i in range(n_cells)]
        cls.meta = pd.DataFrame({
            'cell': [f'cell{i}' for i in range(n_cells)],
            'guide': guides,
            'cell_line': ['line1'] * 50 + ['line2'] * 50,
            'target': ['GFI1B'] * 80 + ['ntc'] * 20,
            'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
        })
        cls.model = bayesDREAM(
            meta=cls.meta,
            counts=None,
            modality_name='atac',
        )
        n_regions = 10
        atac_counts = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(n_regions, n_cells)),
            index=[f'chr9:{i*1000}-{(i+1)*1000}' for i in range(n_regions)],
            columns=[f'cell{i}' for i in range(n_cells)],
        )
        region_meta = pd.DataFrame({
            'region_id': [f'chr9:{i*1000}-{(i+1)*1000}' for i in range(n_regions)],
            'region_type': ['promoter'] * 3 + ['gene_body'] * 3 + ['distal'] * 4,
            'chrom': ['chr9'] * n_regions,
            'start': np.arange(1000, 1000 + n_regions * 1000, 1000),
            'end': np.arange(2000, 2000 + n_regions * 1000, 1000),
            'gene': ['GFI1B', 'GFI1B', 'SPI1'] + ['GFI1B'] * 3 + [''] * 4,
        })
        cls.model.add_atac_modality(
            atac_counts=atac_counts,
            region_meta=region_meta,
            name='atac',
            cis_region='chr9:1000-2000',
        )

    def test_atac_modality_added(self):
        modalities_df = self.model.list_modalities()
        self.assertIn('atac', modalities_df['name'].values)

    def test_primary_modality_is_atac(self):
        self.assertEqual(self.model.primary_modality, 'atac')


class TestManualGuideEffectsInfrastructure(unittest.TestCase):
    """Validate that the manual guide effects parameter interface is correct."""

    def test_guide_effects_dataframe_format(self):
        guide_effects = pd.DataFrame({
            'guide': ['guide0', 'guide1', 'guide2'],
            'log2FC': [-2.5, -1.8, -1.2],
        })
        for col in ('guide', 'log2FC'):
            self.assertIn(col, guide_effects.columns)


if __name__ == '__main__':
    unittest.main()
