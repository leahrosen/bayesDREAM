"""Per-group mu_target prior for fit_cis(independent_mu_sigma=True).

Each mu/sigma group's mu_target_i prior is built from that group's own guides
(CisFitter._group_mu_prior), not from all guides pooled -- otherwise a cis
group with one or two guides gets pulled toward NTC when there are many NTC
guides (Replogle all-NTC variant: ~500 NTC guides + 1 targeting guide pair).
The prior's width stays the shared all-guide SD.
"""

import numpy as np
import pandas as pd
import pytest

from bayesDREAM.fitting.cis import CisFitter

prior = CisFitter._group_mu_prior


# ------------------------------------------------------------------ unit tests
def test_single_guide_group_gets_own_centre_and_all_guide_scale():
    # group 0: 4 NTC guides around 2**0 = 1; group 1: ONE targeting guide at 2**-2
    sums = np.array([100., 110., 90., 100., 25.])
    ns = np.array([100., 100., 100., 100., 100.])
    means = np.log2(sums / ns)
    loc, scale = prior(sums, ns, means, np.array([0, 0, 0, 0, 1]), fallback_center=0.0)
    assert loc[1] == pytest.approx(-2.0, abs=1e-6)       # own centre, not the NTC level
    assert loc[0] == pytest.approx(np.log2(400 / 400), abs=1e-6)
    assert scale == pytest.approx(np.full(2, np.std(means, ddof=1)), rel=1e-5)   # never NaN
    assert np.all(np.isfinite(loc)) and np.all(np.isfinite(scale))


def test_two_guide_group_uses_all_guide_scale_not_its_own_sd():
    # two nearly identical NTC guides: their own SD ~0 would make a point-mass prior
    sums = np.array([100., 100.5, 25.])
    ns = np.array([100., 100., 100.])
    means = np.log2(sums / ns)
    loc, scale = prior(sums, ns, means, np.array([0, 0, 1]), fallback_center=0.0)
    assert loc[0] == pytest.approx(np.log2(200.5 / 200), abs=1e-6)
    assert scale[0] == pytest.approx(np.std(means, ddof=1), rel=1e-5)
    assert scale[0] > 0.5                                 # spans the NTC-vs-target gap


def test_scale_matches_shared_prior_width():
    # same SD the shared (independent_mu_sigma=False) prior uses: torch.std, unbiased
    import torch
    rng = np.random.default_rng(0)
    means = rng.normal(0, 0.7, 30)
    sums, ns = 2.0 ** means * 10, np.full(30, 10.)
    _, scale = prior(sums, ns, means, rng.integers(0, 3, 30), fallback_center=0.0)
    assert scale[0] == pytest.approx(float(torch.std(torch.tensor(means))), rel=1e-5)


def test_all_zero_guide_does_not_drag_centre_to_epsilon():
    # one of two targeting guides has no counts at all: per-guide log2 mean ~ log2(1e-6)
    sums = np.array([100., 100., 100., 50., 0.])
    ns = np.array([100., 100., 100., 50., 50.])
    means = np.log2(np.clip(sums / ns, 1e-6, None))
    loc, _ = prior(sums, ns, means, np.array([0, 0, 0, 1, 1]), fallback_center=0.0)
    assert loc[1] == pytest.approx(np.log2(50 / 100), abs=1e-6)   # cell-pooled, not mean of log2s


def test_group_without_counts_falls_back():
    sums = np.array([100., 0.])
    ns = np.array([100., 30.])
    loc, scale = prior(sums, ns, np.log2(np.clip(sums / ns, 1e-6, None)), np.array([0, 1]),
                       fallback_center=-0.5)
    assert loc[1] == pytest.approx(-0.5)
    assert np.all(scale > 0)


def test_identical_guides_give_tiny_but_positive_scale():
    sums = np.array([10., 10.])
    ns = np.array([10., 10.])
    _, scale = prior(sums, ns, np.zeros(2), np.array([0, 1]), fallback_center=0.0)
    assert np.all(scale == pytest.approx(1e-3))


# --------------------------------------------------------------- end to end
def _one_target_many_ntc(seed=0, n_ntc_guides=40, cells_per_guide=40, true_lfc=-2.0, base=20.0):
    rng = np.random.default_rng(seed)
    rows = [(f'ntc_{g}', 'ntc') for g in range(n_ntc_guides) for _ in range(cells_per_guide)]
    rows += [('guide_T', 'GFI1B')] * cells_per_guide
    meta = pd.DataFrame(rows, columns=['guide', 'target'])
    meta['cell'] = [f'cell_{i}' for i in range(len(meta))]
    meta['cell_line'] = 'K562'
    meta['sum_factor'] = rng.uniform(0.8, 1.2, len(meta))
    genes = ['GFI1B'] + [f'gene_{i}' for i in range(10)]
    mu = np.full((len(genes), len(meta)), base) * meta['sum_factor'].values
    mu[0, (meta['target'] == 'GFI1B').values] *= 2.0 ** true_lfc
    counts = pd.DataFrame(rng.negative_binomial(10, 10 / (10 + mu)), index=genes,
                          columns=meta['cell'].tolist())
    return meta, counts


@pytest.mark.slow
def test_single_targeting_guide_not_shrunk_toward_many_ntc():
    """Low expression (base 0.5 counts/cell, ~5 counts in the targeting guide
    in total) + 150 NTC guides, i.e. Replogle-like: the old all-guide prior
    gave ~-0.4 here (seeds 0/1) for a true -2.0; the per-group-centred prior
    gives -2.5 / -1.7 / -1.8 (seeds 0/1/2). The bound is loose because one
    guide with ~5 counts is genuinely noisy, but it cleanly separates the two."""
    from bayesDREAM import bayesDREAM
    true_lfc = -2.0
    meta, counts = _one_target_many_ntc(n_ntc_guides=150, cells_per_guide=40, base=0.5,
                                        true_lfc=true_lfc, seed=1)
    model = bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B', output_dir='./test_output',
                       label='test_group_mu_prior', device='cpu')
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=500, nsamples=20)
    model.fit_cis(sum_factor_col='sum_factor', niters=5000, nsamples=50, force=True,
                  independent_mu_sigma=True)

    x_eff = np.log2(model.posterior_samples_cis['x_eff_g'].median(dim=0).values.numpy().ravel())
    labels = model.cis_guide_labels
    t = [i for i, g in enumerate(labels) if g.startswith('guide_T')]
    n = [i for i, g in enumerate(labels) if g.startswith('ntc_')]
    lfc = x_eff[t].mean() - np.median(x_eff[n])
    assert -3.5 < lfc < -1.2, f"estimated log2FC {lfc:.2f} vs true {true_lfc} (old prior gave ~-0.4)"
