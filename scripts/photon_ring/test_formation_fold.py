"""Exercise the formation diagnostics on a manufactured fold of rotating slices."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from dense_fold import dense_fold
from formation_witness import fit_witness
from onset import continued_outer, qualify


class FormationFoldTest(unittest.TestCase):
    radius = 2.0
    birth = 1.001
    rotation = -0.012

    def frame(self, t, cycle):
        # Euclidean metric. On a sphere Theta+ = 2/r - 2k. The radial-
        # azimuthal part of K gives a nonzero signed J without changing Theta+.
        x, z = np.meshgrid(np.linspace(0, 4, 65), np.linspace(0, 4, 65))
        k = (
            2
            * self.radius
            * (1 + 0.1 * (t - self.birth))
            / (x * x + z * z + self.radius**2)
        )
        fields = {}
        for name in ("xx", "xy", "xz", "yy", "yz", "zz"):
            diagonal = name[0] == name[1]
            g = np.ones_like(x) if diagonal else np.zeros_like(x)
            curvature = k * g
            if name == "xz":
                curvature = self.rotation * x * x
            if name == "yz":
                curvature = self.rotation * x * z
            fields["adm_g" + name] = [g[None, ...]]
            fields["adm_K" + name] = [curvature[None, ...]]
        return dict(
            time=t,
            cycle=cycle,
            header=["<mesh>", "ix2_bc = reflect"],
            mb_geometry=[[0, 4, 0, 4]],
            mb_data=fields,
        )

    def test_signed_charges_and_branch_birth(self):
        rows = [
            (t, 5 * i, Path(f"frame-{i}"))
            for i, t in enumerate(np.arange(0.97, 1.071, 0.01))
        ]
        frames = {p: self.frame(t, c) for t, c, p in rows}
        t, c, p = rows[-1]
        q = 1 + 0.1 * (t - self.birth)
        seeds = [[self.radius * (q + sign * np.sqrt(q * q - 1))] for sign in (1, -1)]
        with (
            patch("onset.read_binary", side_effect=frames.__getitem__),
            patch("dense_fold.read_binary", side_effect=frames.__getitem__),
        ):
            pair = qualify(p, seeds, 1.0)
            self.assertIsNotNone(pair)
            for name, (r,) in zip(("outer", "inner"), seeds):
                fit = pair[name]
                self.assertAlmostEqual(fit["area"], 4 * np.pi * r * r, delta=3e-4)
                self.assertAlmostEqual(fit["J"], self.rotation * r**5 / 3, delta=3e-6)
                self.assertAlmostEqual(
                    fit["zeta"], 2 * self.rotation * r**3 / 3, delta=2e-6
                )
                lam = (
                    -2 / r**2 + 8 * self.radius * q * r / (r * r + self.radius**2) ** 2
                )
                self.assertAlmostEqual(fit["lambda0"], lam, delta=3e-5)
            w = fit_witness(frames[p], pair["outer"]["coeff"], pair["inner"]["coeff"])
            self.assertTrue(w["accepted"])
            self.assertIsNone(continued_outer(rows[0][2], pair["outer"]["coeff"]))
            result = dense_fold(rows, [pair], 1.0)
        self.assertEqual(result["status"], "reconstructed", result)
        for v in result["variants"]:
            f = v["fold"]
            self.assertAlmostEqual(f["t_M"], self.birth, delta=0.0002)
            self.assertAlmostEqual(
                f["zeta"], 2 * self.rotation * self.radius**3 / 3, delta=0.0001
            )
            self.assertLess(abs(f["lambda_M2"]), 1e-7)
            self.assertGreater(f["d2t_dmean2"], 0)
            lo, hi = v["adjacent_evolved_times_M"]
            self.assertLess(lo, self.birth)
            self.assertGreater(hi, self.birth)


if __name__ == "__main__":
    unittest.main()
