"""Numerical checks for offline actor-pair diagnostics (no ALF import needed)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'encoding_distances', Path(__file__).with_name('plot_actor_encoding_distances.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairwiseMetricsTest(unittest.TestCase):

    def test_known_geometry_and_action_axis(self):
        z = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        actions = np.array([[[0., 0.], [3., 4.], [0., 0.]],
                            [[0., 0.], [0., 0.], [0., 0.]]])
        metrics = module.pairwise_metrics(z, actions)
        np.testing.assert_allclose(metrics['l2'][0], [0, np.sqrt(2), 2])
        np.testing.assert_allclose(metrics['cosine'][0], [0, 1, 2])
        self.assertEqual(metrics['action_rms'][0, 1], 2.5)
        self.assertEqual(metrics['action_rms'][0, 2], 0.)
        for matrix in metrics.values():
            np.testing.assert_allclose(matrix, matrix.T)
            np.testing.assert_allclose(matrix.diagonal(), 0)

    def test_scaling_changes_l2_but_not_cosine(self):
        z = np.array([[1., 2.], [-2., 1.]])
        actions = np.zeros((4, 2, 3))
        first = module.pairwise_metrics(z, actions)
        scaled = module.pairwise_metrics(z * 7, actions)
        np.testing.assert_allclose(scaled['l2'], first['l2'] * 7)
        np.testing.assert_allclose(scaled['cosine'], first['cosine'], atol=1e-15)

    def test_zero_encoding_is_undefined_cosine(self):
        result = module.pairwise_metrics([[0., 0.], [1., 0.]],
                                         np.zeros((2, 2, 1)))
        self.assertTrue(np.isnan(result['cosine'][0]).all())
        self.assertEqual(result['l2'][0, 1], 1.)

    def test_bad_shape_and_nonfinite_rejected(self):
        for z, actions in [(np.ones((3, 2)), np.zeros((3, 4, 2))),
                           (np.array([[np.nan, 1.]]), np.zeros((2, 1, 1)))]:
            with self.assertRaises(ValueError):
                module.pairwise_metrics(z, actions)


if __name__ == '__main__':
    unittest.main()
