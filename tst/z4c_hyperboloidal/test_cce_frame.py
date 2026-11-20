"""Recover the rest frame of a translated, boosted, supertranslated Schwarzschild cut."""

import unittest

import numpy as np
import scri
from athenak_cce.diagnostics import fit_frame, parameters, summarize_cut


class FrameTest(unittest.TestCase):
    def test_nonspinning_superrest(self):
        abd = scri.AsymptoticBondiData(time=np.linspace(0, 20, 101), ell_max=4)
        abd.psi2[:, 0] = -np.sqrt(4 * np.pi)
        alpha = np.zeros(25, dtype=complex)
        alpha[2], alpha[6] = 0.06, 0.12
        moved = abd.transform(supertranslation=alpha, boost_velocity=[0.001, 0, 0])
        before = summarize_cut(moved, 12, 3)
        self.assertGreater(before["proper_supermomentum"], 0.01)
        self.assertGreater(before["center_M0"], 0.01)
        self.assertGreater(before["speed"], 0.0009)
        transform = fit_frame(moved, 12, 3, keep_axes=True)
        restored = moved.transform(**parameters(transform))
        after = summarize_cut(restored, 12, 3)
        for key in ("speed", "proper_supermomentum", "center_M0"):
            self.assertLess(after[key], 1e-6)
        self.assertAlmostEqual(after["mass_M0"], 1, places=7)


if __name__ == "__main__":
    unittest.main()
