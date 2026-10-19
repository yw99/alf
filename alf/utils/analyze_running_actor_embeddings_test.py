"""Numerical geometry checks for collapse diagnostics."""
import unittest
import numpy as np
from alf.utils.analyze_running_actor_embeddings import geometry


class GeometryTest(unittest.TestCase):
    def test_identical_nonzero_embeddings(self):
        m=geometry(np.ones((10,12)))
        np.testing.assert_equal(m['l2'],0)
        np.testing.assert_allclose(m['cosine'],0,atol=1e-15)
        self.assertEqual(float(m['effective_rank']),0)
        self.assertEqual(float(m['centered_energy_fraction']),0)

    def test_centered_simplex_has_nine_equal_directions(self):
        m=geometry(np.eye(10))
        np.testing.assert_allclose(m['l2'],np.sqrt(2))
        self.assertAlmostEqual(float(m['effective_rank']),9)
        self.assertAlmostEqual(float(m['centered_energy_fraction']),.9)
        self.assertAlmostEqual(float(m['leading_centered_variance_fraction']),1/9)

    def test_common_offset_does_not_change_centered_separation(self):
        first=geometry(np.eye(10));offset=geometry(np.eye(10)+100)
        np.testing.assert_allclose(first['l2'],offset['l2'])
        self.assertAlmostEqual(float(offset['effective_rank']),9)
        self.assertLess(float(offset['centered_energy_fraction']),1e-5)
        self.assertLess(offset['cosine'].mean(),1e-4)

    def test_scale_invariant_metrics_and_one_dimensional_variation(self):
        x=np.arange(10.)[:,None]*np.ones((1,5))+1
        a,b=geometry(x),geometry(x*7)
        np.testing.assert_allclose(b['l2'],7*a['l2'])
        np.testing.assert_allclose(a['relative_l2'],b['relative_l2'])
        np.testing.assert_allclose(a['cosine'],b['cosine'],atol=1e-15)
        self.assertAlmostEqual(float(a['effective_rank']),1)
        self.assertAlmostEqual(float(b['effective_rank']),1)


if __name__=='__main__':unittest.main()
