"""adjust_ntc_sum_factor(pool_batches=True), the default: one log-space offset
per guide (NTC guides included), pooled across covariate groups, so a guide
with only 1-2 cells per batch keeps each cell's own depth instead of having it
replaced by the batch's NTC mean (the pool_batches=False behaviour).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('pyro')


def _low_moi_model(seed=0, n_batches=40, guide_offsets=None, batch_of=None):
    """Cells: per batch, 1-2 cells of each guide; log sf = batch + guide + noise."""
    from bayesDREAM import bayesDREAM
    rng = np.random.default_rng(seed)
    guide_offsets = guide_offsets or {'ntc_1': 0.0, 'ntc_2': 0.2, 'ntc_3': -0.1, 'guide_T': -0.5}
    batch_eff = rng.normal(0, 0.4, n_batches)
    rows = []
    for b in range(n_batches):
        for g, off in guide_offsets.items():
            if batch_of is not None and b not in batch_of.get(g, range(n_batches)):
                continue
            for _ in range(rng.integers(1, 3)):
                rows.append((g, 'GFI1B' if g == 'guide_T' else 'ntc', f'b{b}',
                             np.exp(batch_eff[b] + off + rng.normal(0, 0.3))))
    meta = pd.DataFrame(rows, columns=['guide', 'target', 'batch', 'sum_factor'])
    meta['cell'] = [f'c{i}' for i in range(len(meta))]
    meta['cell_line'] = 'K562'
    genes = ['GFI1B'] + [f'g{i}' for i in range(5)]
    counts = pd.DataFrame(rng.poisson(5, (len(genes), len(meta))), index=genes, columns=meta['cell'])
    return bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B', output_dir='./test_output',
                      label='test_pool_sf', device='cpu')


def _sf(model, col):
    sf = model.get_modality(model.primary_modality).sum_factors
    return sf.loc[model.meta['cell'].values, col].values


def test_pooled_is_default_and_one_offset_per_guide():
    m = _low_moi_model()
    m.adjust_ntc_sum_factor(covariates=['batch'])                 # pool_batches=True by default
    ratio = np.log(_sf(m, 'sum_factor_adj') / _sf(m, 'sum_factor'))
    per_guide = pd.Series(ratio).groupby(m.meta['guide'].values)
    assert per_guide.std().max() < 1e-10                           # same factor for every cell of a guide
    # offsets relative to the NTC cells: target ~ -0.5, so it is scaled UP by ~0.5
    assert -per_guide.mean()['guide_T'] == pytest.approx(-0.5, abs=0.15)
    # NTC guides are corrected like any other guide (not left at 1)
    assert -per_guide.mean()['ntc_2'] == pytest.approx(0.2, abs=0.15)
    assert abs(per_guide.mean()['ntc_2']) > 0.05


def test_pooled_keeps_each_cells_depth_within_guide():
    m = _low_moi_model()
    m.adjust_ntc_sum_factor(covariates=['batch'])
    t = (m.meta['guide'] == 'guide_T').values
    r = np.corrcoef(np.log(_sf(m, 'sum_factor')[t]), np.log(_sf(m, 'sum_factor_adj')[t]))[0, 1]
    assert r == pytest.approx(1.0)


def test_legacy_per_group_replaces_single_cell_by_batch_ntc_mean():
    m = _low_moi_model()
    m.adjust_ntc_sum_factor(covariates=['batch'], pool_batches=False)
    meta = m.meta.assign(sf=_sf(m, 'sum_factor'), adj=_sf(m, 'sum_factor_adj'))
    ntc_mean = meta[meta['target'] == 'ntc'].groupby('batch')['sf'].mean()
    n = meta.groupby(['batch', 'guide'])['cell'].transform('size')
    single = meta[(n == 1) & (meta['guide'] == 'guide_T')]
    assert len(single) > 0
    np.testing.assert_allclose(single['adj'].values, single['batch'].map(ntc_mean).values)


def test_pooled_raises_for_guide_only_in_groups_without_ntc():
    # guide_T only in batches 0-4; NTC guides only in batches 5+
    batch_of = {'guide_T': range(0, 5), 'ntc_1': range(5, 40), 'ntc_2': range(5, 40), 'ntc_3': range(5, 40)}
    m = _low_moi_model(batch_of=batch_of)
    with pytest.raises(ValueError, match='without NTC cells'):
        m.adjust_ntc_sum_factor(covariates=['batch'])


def test_pooled_high_moi_one_offset_per_guide_combination():
    from bayesDREAM import bayesDREAM
    rng = np.random.default_rng(1)
    n_cells = 120
    guide_assignment = np.zeros((n_cells, 4), dtype=int)
    guide_assignment[0:40, 0] = 1                 # guide_A
    guide_assignment[20:60, 1] = 1                # guide_B (20-39 carry A+B)
    guide_assignment[60:120, 2] = 1               # ntc_1
    guide_assignment[90:120, 3] = 1               # ntc_2 (90-119 carry ntc_1+ntc_2)
    guide_meta = pd.DataFrame({'guide': ['guide_A', 'guide_B', 'ntc_1', 'ntc_2'],
                               'target': ['GFI1B', 'GFI1B', 'ntc', 'ntc']},
                              index=['guide_A', 'guide_B', 'ntc_1', 'ntc_2'])
    meta = pd.DataFrame({'cell': [f'c{i}' for i in range(n_cells)],
                         'cell_line': rng.choice(['L1', 'L2', 'L3'], n_cells),
                         'sum_factor': np.exp(rng.normal(0, 0.3, n_cells))})
    genes = ['GFI1B'] + [f'g{i}' for i in range(5)]
    counts = pd.DataFrame(rng.poisson(5, (len(genes), n_cells)), index=genes, columns=meta['cell'])
    m = bayesDREAM(meta=meta, counts=counts, guide_assignment=guide_assignment, guide_meta=guide_meta,
                   cis_gene='GFI1B', output_dir='./test_output', label='test_pool_sf_hm', device='cpu')
    m.adjust_ntc_sum_factor(covariates=['cell_line'])
    ratio = np.log(_sf(m, 'sum_factor_adj') / _sf(m, 'sum_factor'))
    combo = [tuple(r) for r in np.asarray(m.guide_assignment)]
    per_combo = pd.Series(ratio).groupby(pd.Series(combo).astype(str).values).std()
    assert np.isfinite(ratio).all()
    assert per_combo.max() < 1e-10                 # one correction per guide combination, across cell lines
