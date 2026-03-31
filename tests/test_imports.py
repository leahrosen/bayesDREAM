"""Test that all required dependencies can be imported."""

import unittest


class TestImports(unittest.TestCase):
    """Verify all required packages are importable."""

    def test_numpy(self):
        import numpy  # noqa: F401

    def test_scipy(self):
        import scipy  # noqa: F401

    def test_pandas(self):
        import pandas  # noqa: F401

    def test_scikit_learn(self):
        import sklearn  # noqa: F401
        from sklearn.preprocessing import SplineTransformer  # noqa: F401
        from sklearn.linear_model import Ridge  # noqa: F401
        from sklearn.pipeline import make_pipeline  # noqa: F401

    def test_torch(self):
        import torch  # noqa: F401

    def test_pyro(self):
        import pyro  # noqa: F401

    def test_matplotlib(self):
        import matplotlib  # noqa: F401

    def test_seaborn(self):
        import seaborn  # noqa: F401

    def test_h5py(self):
        import h5py  # noqa: F401

    def test_bayesdream_package(self):
        from bayesDREAM import bayesDREAM, Modality  # noqa: F401

    def test_bayesdream_distribution_registry(self):
        from bayesDREAM import (
            get_observation_sampler,
            requires_denominator,
            is_3d_distribution,
            DISTRIBUTION_REGISTRY,
        )
        self.assertIn('negbinom', DISTRIBUTION_REGISTRY)
        self.assertIn('multinomial', DISTRIBUTION_REGISTRY)
        self.assertIn('binomial', DISTRIBUTION_REGISTRY)
        self.assertIn('normal', DISTRIBUTION_REGISTRY)
        self.assertTrue(requires_denominator('binomial'))
        self.assertFalse(requires_denominator('negbinom'))
        self.assertTrue(is_3d_distribution('multinomial'))
        self.assertFalse(is_3d_distribution('negbinom'))
        sampler = get_observation_sampler('negbinom', 'trans')
        self.assertTrue(callable(sampler))


if __name__ == '__main__':
    unittest.main()
