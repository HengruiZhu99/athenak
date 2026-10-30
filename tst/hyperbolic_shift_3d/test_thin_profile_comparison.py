"""Check physical tensor contractions and same-location reference interpolation."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('comparison', ROOT/'docs/hyperbolic-shift-3d/reproduction/thin-overlap-v1/compare_profiles.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ProfileComparison(unittest.TestCase):
    def test_spherical_tensor_cartesian_components(self):
        rng = np.random.default_rng(143)
        normal = rng.normal(size=(50, 3)); normal /= np.linalg.norm(normal, axis=1)[:, None]
        radius = np.linspace(.025, .4, len(normal))
        data = np.zeros(len(normal), dtype=[(key, float) for key in ['r', 'x', 'y', 'z']+[f'v{i}' for i in range(31)]])
        data['r'] = radius
        for i, key in enumerate(('x', 'y', 'z')):
            data[key] = radius*normal[:, i]
            data[f'v{22+i}'] = (.4+radius)*normal[:, i]
        data['v0'] = .3
        data['v18'] = .7-(1+radius**2)/2
        data['v14'] = .02
        # Distinct radial/tangential eigenvalues expose missing off-diagonal factors.
        for f, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            data[f'v{1+f}'] = 2*(i == j)+2*normal[:, i]*normal[:, j]-(i == j)
            data[f'v{7+f}'] = -.25*(i == j)+.75*normal[:, i]*normal[:, j]
        got = comparison.radial_fields(data)
        for key, value in dict(chi=.3, alpha=.7, beta_r=.4, g_rr=4, A_rr=.5, Theta=.02).items():
            np.testing.assert_allclose(got[key], value, rtol=0, atol=3e-15)

    def test_polynomial_reference_and_no_extrapolation(self):
        x = np.linspace(0, 1, 513); samples = np.linspace(.025, .4, 76)
        polynomial = lambda x: .1+x-2*x*x+.4*x**5
        np.testing.assert_allclose(comparison.interpolate_six(x, polynomial(x), samples), polynomial(samples), rtol=0, atol=3e-16)
        with self.assertRaises(AssertionError):
            comparison.interpolate_six(x, x, np.array([1.001]))


if __name__ == '__main__':
    unittest.main()
