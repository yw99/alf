"""Check actor-ID geometry against independent distance calculations."""
import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from plot_control_a_embeddings import geometry


class EmbeddingGeometryTest(unittest.TestCase):

    def test_matches_scipy(self):
        z = np.random.default_rng(3).normal(size=(10, 32))
        l2, cosine, norms, rank = geometry(z)
        np.testing.assert_allclose(l2, squareform(pdist(z)), atol=1e-12)
        np.testing.assert_allclose(cosine, squareform(pdist(z, 'cosine')), atol=1e-12)
        np.testing.assert_allclose(norms, np.sqrt((z*z).sum(axis=1)))
        self.assertGreater(rank, 1)
        self.assertLessEqual(rank, 9 + 1e-12)

    def test_scale_and_translation(self):
        z = np.random.default_rng(4).normal(size=(10, 32))
        l2, cosine, norms, rank = geometry(z)
        scaled = geometry(5*z)
        np.testing.assert_allclose(scaled[0], 5*l2, atol=1e-12)
        np.testing.assert_allclose(scaled[1], cosine, atol=1e-12)
        self.assertAlmostEqual(scaled[3], rank)
        translated = geometry(z+7)
        np.testing.assert_allclose(translated[0], l2, atol=1e-12)
        self.assertAlmostEqual(translated[3], rank)

    def test_identical_and_orthogonal(self):
        l2, cosine, _, rank = geometry(np.ones((10, 12)))
        np.testing.assert_allclose(l2, 0)
        np.testing.assert_allclose(cosine, 0, atol=1e-12)
        self.assertEqual(rank, 0)
        self.assertAlmostEqual(geometry(np.eye(10))[3], 9)


if __name__ == '__main__':
    unittest.main()
