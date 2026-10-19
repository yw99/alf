"""Numerical checks for encoder-input block comparisons."""
import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform
from analyze_actor_encoder_inputs import block_metrics


class EncoderInputMetricsTest(unittest.TestCase):

    def test_agrees_with_independent_distances(self):
        x=np.random.default_rng(9).normal(size=(10,5,7))
        m=block_metrics(x)
        flat=x.reshape(10,-1)
        np.testing.assert_allclose(m['l2'],squareform(pdist(flat)),atol=1e-12)
        np.testing.assert_allclose(m['rms'],squareform(pdist(flat))/np.sqrt(35),atol=1e-12)
        np.testing.assert_allclose(m['cosine'],squareform(pdist(flat,'cosine')),atol=1e-12)

    def test_combined_rms_weights_by_block_width(self):
        rng=np.random.default_rng(2)
        hidden=rng.normal(size=(10,4,8))
        action=rng.normal(size=(10,2,8))
        combined=block_metrics(np.concatenate([hidden,action],axis=1))['rms']**2
        expected=(4*block_metrics(hidden)['rms']**2+2*block_metrics(action)['rms']**2)/6
        np.testing.assert_allclose(combined,expected,atol=1e-12)

    def test_shared_position_does_not_change_euclidean_distance(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(10,3,8))
        shared=rng.normal(size=(1,3,8))
        np.testing.assert_allclose(block_metrics(x)['rms'],block_metrics(x+shared)['rms'],atol=1e-12)

    def test_zero_vectors_have_undefined_cosine(self):
        m=block_metrics(np.zeros((10,4)))
        self.assertTrue(np.isnan(m['cosine']).all())
        np.testing.assert_array_equal(m['rms'],0)
        np.testing.assert_array_equal(m['actor_zero_fraction'],1)


if __name__=='__main__':unittest.main()
