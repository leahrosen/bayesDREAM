"""Test that all required dependencies can be imported."""


def test_numpy():
    import numpy  # noqa: F401


def test_scipy():
    import scipy  # noqa: F401


def test_pandas():
    import pandas  # noqa: F401


def test_scikit_learn():
    import sklearn  # noqa: F401
    from sklearn.preprocessing import SplineTransformer  # noqa: F401
    from sklearn.linear_model import Ridge  # noqa: F401
    from sklearn.pipeline import make_pipeline  # noqa: F401


def test_torch():
    import torch  # noqa: F401


def test_pyro():
    import pyro  # noqa: F401


def test_matplotlib():
    import matplotlib  # noqa: F401


def test_seaborn():
    import seaborn  # noqa: F401


def test_h5py():
    import h5py  # noqa: F401


def test_bayesdream_package():
    from bayesDREAM import bayesDREAM, Modality  # noqa: F401


def test_bayesdream_distribution_registry():
    from bayesDREAM import (
        get_observation_sampler,
        requires_denominator,
        is_3d_distribution,
        DISTRIBUTION_REGISTRY,
    )
    assert 'negbinom' in DISTRIBUTION_REGISTRY
    assert 'multinomial' in DISTRIBUTION_REGISTRY
    assert 'binomial' in DISTRIBUTION_REGISTRY
    assert 'normal' in DISTRIBUTION_REGISTRY
    assert requires_denominator('binomial')
    assert not requires_denominator('negbinom')
    assert is_3d_distribution('multinomial')
    assert not is_3d_distribution('negbinom')
    sampler = get_observation_sampler('negbinom', 'trans')
    assert callable(sampler)
