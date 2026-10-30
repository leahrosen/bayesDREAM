"""Tests for guide_covariates / guide_covariates_ntc semantics and the
mu/sigma grouping used by fit_cis(independent_mu_sigma=True).

- guide_used: NTC guides split only by guide_covariates_ntc, non-NTC guides
  only by guide_covariates (the two lists are never combined).
- fit_cis(independent_mu_sigma=True): groups = (NTC | cis) x covariate combo,
  defaulting to guide_covariates_ntc / guide_covariates, overridable with
  mu_sigma_covariates_ntc / mu_sigma_covariates.
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _make_data(seed=0, n_per_guide=20):
    """CRISPRi/CRISPRa arms ('cell_line') and two lanes ('lane').

    guide_A: CRISPRi cells (both lanes), guide_B: CRISPRa cells (both lanes),
    ntc_1 / ntc_2: cells in both arms and both lanes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for guide, target, arms in [
        ('guide_A', 'GFI1B', ['CRISPRi']),
        ('guide_B', 'GFI1B', ['CRISPRa']),
        ('ntc_1', 'ntc', ['CRISPRi', 'CRISPRa']),
        ('ntc_2', 'ntc', ['CRISPRi', 'CRISPRa']),
    ]:
        for arm in arms:
            for lane in ['L1', 'L2']:
                for _ in range(n_per_guide // 2):
                    rows.append((guide, target, arm, lane))
    meta = pd.DataFrame(rows, columns=['guide', 'target', 'cell_line', 'lane'])
    meta['cell'] = [f'cell_{i}' for i in range(len(meta))]
    meta['sum_factor'] = rng.uniform(0.8, 1.2, len(meta))
    genes = [f'gene_{i}' for i in range(20)]
    genes[0] = 'GFI1B'
    counts = pd.DataFrame(
        rng.negative_binomial(20, 0.3, (len(genes), len(meta))),
        index=genes, columns=meta['cell'].tolist(),
    )
    return meta, counts


def _make_model(**kwargs):
    from bayesDREAM import bayesDREAM
    meta, counts = _make_data()
    return bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B',
        output_dir='./test_output', label='test_mu_sigma_groups', device='cpu',
        **kwargs,
    )


def _groups(model, **kw):
    G = model.meta['guide_code'].nunique()
    model._cis_fitter._build_mu_sigma_groups(
        G, None, kw.get('mu_sigma_covariates'), kw.get('mu_sigma_covariates_ntc'))
    return model.mu_sigma_group_labels


# ---------------------------------------------------------------- guide_used
def test_guide_used_ntc_and_non_ntc_use_separate_covariate_lists():
    model = _make_model(guide_covariates=['cell_line'], guide_covariates_ntc=['lane'])
    used = model.meta.groupby('guide')['guide_used'].agg(lambda s: set(s))
    # non-NTC: only cell_line (no lane); NTC: only lane (no cell_line)
    assert used['guide_A'] == {'guide_A_CRISPRi'}
    assert used['guide_B'] == {'guide_B_CRISPRa'}
    assert used['ntc_1'] == {'ntc_1_L1', 'ntc_1_L2'}
    assert model.meta['guide_used'].nunique() == 1 + 1 + 2 + 2


def test_guide_used_defaults_do_not_split():
    model = _make_model()
    assert set(model.meta['guide_used'].unique()) == {
        'guide_A_', 'guide_B_', 'ntc_1_', 'ntc_2_'}


# ------------------------------------------------------- mu/sigma group build
def test_default_groups_split_cis_by_guide_covariates_ntc_by_ntc_covariates():
    model = _make_model(guide_covariates=['cell_line'])  # guide_covariates_ntc=[]
    labels = _groups(model)
    assert sorted(labels) == sorted([
        'GFI1B|cell_line=CRISPRi', 'GFI1B|cell_line=CRISPRa', 'ntc'])


def test_ntc_split_by_guide_covariates_ntc_by_default():
    model = _make_model(guide_covariates=['cell_line'], guide_covariates_ntc=['lane'])
    labels = _groups(model)
    assert 'ntc|lane=L1' in labels and 'ntc|lane=L2' in labels
    assert not any(l.startswith('ntc') and 'cell_line' in l for l in labels)


def test_explicit_overrides():
    model = _make_model(guide_covariates=['cell_line'])
    # merge CRISPRi/CRISPRa into one cis group
    assert sorted(_groups(model, mu_sigma_covariates=[])) == ['GFI1B', 'ntc']
    # NTC guides are not split by cell_line, so it varies within an NTC guide
    # effect -> must raise
    with pytest.raises(ValueError, match="cell_line"):
        _groups(model, mu_sigma_covariates_ntc=['cell_line'])


def test_covariate_varying_within_guide_raises():
    model = _make_model(guide_covariates=['cell_line'])
    with pytest.raises(ValueError, match="lane"):
        _groups(model, mu_sigma_covariates=['lane'])


def test_ntc_split_by_arm_when_guides_split_by_arm():
    model = _make_model(guide_covariates=['cell_line'], guide_covariates_ntc=['cell_line'])
    assert sorted(_groups(model)) == sorted([
        'GFI1B|cell_line=CRISPRi', 'GFI1B|cell_line=CRISPRa',
        'ntc|cell_line=CRISPRi', 'ntc|cell_line=CRISPRa'])


def test_missing_covariate_column_raises():
    model = _make_model(guide_covariates=['cell_line'])
    with pytest.raises(ValueError, match="not found"):
        _groups(model, mu_sigma_covariates=['nope'])


def test_high_moi_groups():
    pytest.importorskip('torch')
    pytest.importorskip('pyro')
    from bayesDREAM import bayesDREAM
    from test_high_moi import _make_high_moi_covariate_data

    meta, counts, guide_assignment, guide_meta, _ = _make_high_moi_covariate_data()
    model = bayesDREAM(
        meta=meta, counts=counts, guide_assignment=guide_assignment,
        guide_meta=guide_meta, cis_gene='GFI1B',
        guide_covariates=['lane'], guide_covariates_ntc=[],
        output_dir='./test_output', label='test_mu_sigma_high_moi', device='cpu',
    )
    G = model.guide_assignment.shape[1]
    model._cis_fitter._build_mu_sigma_groups(G, None, None, None)
    assert sorted(model.mu_sigma_group_labels) == sorted([
        'GFI1B|lane=L1', 'GFI1B|lane=L2', 'ntc'])


# --------------------------------------------------------------- end to end
def _fitted(**fit_kwargs):
    model = _make_model(guide_covariates=['cell_line'])
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=100, nsamples=20)
    model.fit_cis(sum_factor_col='sum_factor', niters=50, nsamples=20,
                  force=True, **fit_kwargs)
    return model


def test_fit_cis_independent_mu_sigma_sites():
    model = _fitted(independent_mu_sigma=True)
    ps = model.posterior_samples_cis
    K = len(model.mu_sigma_group_labels)
    assert K == 3
    for i in range(K):
        assert f'mu_target_{i}' in ps and f'sigma_target_{i}' in ps
    assert 'mu' not in ps


def test_fit_cis_default_is_shared_mu_sigma():
    model = _fitted()
    assert 'mu' in model.posterior_samples_cis
    assert model.mu_sigma_group_labels is None


# ------------------------------------------------- save/load: guide-axis labels
def _fitted_model(meta_filter=None, guide_covariates=('cell_line',), **fit_kwargs):
    from bayesDREAM import bayesDREAM
    meta, counts = _make_data()
    if meta_filter is not None:
        meta = meta[meta_filter(meta)].reset_index(drop=True)
        counts = counts[meta['cell'].tolist()]
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B',
        output_dir='./test_output', label='test_mu_sigma_groups', device='cpu',
        guide_covariates=list(guide_covariates),
    )
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=100, nsamples=20)
    model.fit_cis(sum_factor_col='sum_factor', niters=50, nsamples=20, force=True, **fit_kwargs)
    return model


def _fresh_model(meta_filter=None, guide_covariates=('cell_line',)):
    from bayesDREAM import bayesDREAM
    meta, counts = _make_data()
    if meta_filter is not None:
        meta = meta[meta_filter(meta)].reset_index(drop=True)
        counts = counts[meta['cell'].tolist()]
    model = bayesDREAM(
        meta=meta, counts=counts, cis_gene='GFI1B',
        output_dir='./test_output', label='test_mu_sigma_groups', device='cpu',
        guide_covariates=list(guide_covariates),
    )
    model.set_technical_groups(['cell_line'])
    return model


@pytest.fixture(scope='module')
def saved_fit(tmp_path_factory):
    d = str(tmp_path_factory.mktemp('cis_fit'))
    model = _fitted_model(independent_mu_sigma=True)
    model.save_cis_fit(output_dir=d)
    return model, d


def test_save_records_labels(saved_fit):
    import torch
    model, d = saved_fit
    payload = torch.load(f'{d}/posterior_samples_cis.pt', weights_only=False)
    assert payload['guide_axis_labels'] == model.cis_guide_labels
    assert payload['mu_sigma_group_labels'] == model.mu_sigma_group_labels
    assert len(payload['mu_sigma_guide_groups']) == len(model.cis_guide_labels)
    lean = torch.load(f'{d}/posterior_samples_cis_lean.pt', weights_only=False)
    assert lean['mu_sigma_group_labels'] == model.mu_sigma_group_labels


def test_roundtrip_restores_labels(saved_fit):
    model, d = saved_fit
    m2 = _fresh_model()
    m2.load_cis_fit(input_dir=d)
    assert m2.mu_sigma_group_labels == model.mu_sigma_group_labels
    assert m2.mu_sigma_guide_groups == model.mu_sigma_guide_groups
    assert m2.cis_guide_labels == model.cis_guide_labels


def test_load_aligns_reordered_guide_axis_by_name(saved_fit, tmp_path):
    import torch, shutil
    model, d = saved_fit
    d2 = str(tmp_path / 'reordered')
    shutil.copytree(d, d2)
    payload = torch.load(f'{d2}/posterior_samples_cis.pt', weights_only=False)
    G = len(payload['guide_axis_labels'])
    perm = list(reversed(range(G)))
    payload['guide_axis_labels'] = [payload['guide_axis_labels'][i] for i in perm]
    payload['mu_sigma_guide_groups'] = [payload['mu_sigma_guide_groups'][i] for i in perm]
    for k in ('x_eff_g', 'sigma_eff', 'eps_x_eff_g'):
        payload['posterior_samples'][k] = payload['posterior_samples'][k][..., perm]
    torch.save(payload, f'{d2}/posterior_samples_cis.pt')

    m2 = _fresh_model()
    m2.load_cis_fit(input_dir=d2)
    for k in ('x_eff_g', 'sigma_eff'):
        assert torch.allclose(m2.posterior_samples_cis[k], model.posterior_samples_cis[k])
    assert m2.mu_sigma_guide_groups == model.mu_sigma_guide_groups


def test_load_drops_guides_absent_from_current_model(saved_fit):
    import torch
    model, d = saved_fit
    m2 = _fresh_model(meta_filter=lambda m: m['guide'] != 'ntc_2')
    m2.load_cis_fit(input_dir=d)
    assert len(m2.cis_guide_labels) == len(model.cis_guide_labels) - 1  # ntc_2 (one effect: guide_covariates_ntc=[])
    assert 'ntc_2_' not in ''.join(m2.cis_guide_labels)
    keep = [model.cis_guide_labels.index(n) for n in m2.cis_guide_labels]
    assert torch.allclose(m2.posterior_samples_cis['x_eff_g'],
                          model.posterior_samples_cis['x_eff_g'][..., keep])
    # guide_code re-compacted to index the aligned axis
    assert sorted(m2.meta['guide_code'].unique()) == list(range(len(m2.cis_guide_labels)))
    assert m2.mu_sigma_group_labels == model.mu_sigma_group_labels


def test_load_raises_on_guide_not_in_fit(saved_fit):
    model, d = saved_fit
    m2 = _fresh_model(guide_covariates=())  # guide_used labels differ ('guide_A_' vs 'guide_A_CRISPRi')
    with pytest.raises(ValueError, match="not in the saved cis fit"):
        m2.load_cis_fit(input_dir=d)


def test_legacy_fit_without_labels_loads_with_warning(saved_fit, tmp_path):
    import torch, shutil
    model, d = saved_fit
    d2 = str(tmp_path / 'legacy')
    shutil.copytree(d, d2)
    payload = torch.load(f'{d2}/posterior_samples_cis.pt', weights_only=False)
    for k in ('guide_axis_labels', 'mu_sigma_group_labels', 'mu_sigma_guide_groups'):
        payload.pop(k)
    torch.save(payload, f'{d2}/posterior_samples_cis.pt')
    m2 = _fresh_model()
    with pytest.warns(UserWarning, match="predates saved guide-axis labels"):
        m2.load_cis_fit(input_dir=d2)
    assert m2.mu_sigma_group_labels is None and m2.cis_guide_labels is None
