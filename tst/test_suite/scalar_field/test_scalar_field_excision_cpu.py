"""Smooth black-hole-interior excision regression for the scalar module."""

import math
from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_excision.athinput"
_BASENAME = "scalar_field_excision"
_TLIM = 0.05
_DAMPING_TIME = 0.2


def _read(variable):
    """Read one final scalar table output."""
    filename = f"tab/{_BASENAME}.{variable}.00001.tab"
    return athena_read.tab(filename)


def test_smooth_scalar_black_hole_excision():
    """Interior fields relax smoothly without contaminating the far exterior."""
    history_file = Path(f"{_BASENAME}.scalar.hst")
    history_file.unlink(missing_ok=True)
    try:
        assert testutils.run(_INPUT_FILE)

        phi = _read("sf_phi0")
        pi = _read("sf_pi0")
        energy = _read("sf_energy")
        x1 = np.asarray(phi["x1v"])
        phi_values = np.asarray(phi["sf_phi0"])
        pi_values = np.asarray(pi["sf_pi0"])
        energy_values = np.asarray(energy["sf_energy"])

        for values in (x1, phi_values, pi_values, energy_values):
            assert np.all(np.isfinite(values))

        deep_interior = np.abs(x1) < 0.5
        far_exterior = np.abs(x1) > 1.5
        expected_interior = math.exp(-_TLIM/_DAMPING_TIME)
        np.testing.assert_allclose(
            phi_values[deep_interior],
            expected_interior,
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            pi_values[deep_interior], 0.0, rtol=0.0, atol=2.0e-13
        )
        np.testing.assert_allclose(
            phi_values[far_exterior], 1.0, rtol=0.0, atol=2.0e-12
        )
        np.testing.assert_allclose(
            pi_values[far_exterior], 0.0, rtol=0.0, atol=2.0e-12
        )

        history = athena_read.hst(history_file)
        assert np.all(np.isfinite(history["sf-energy"]))
    finally:
        history_file.unlink(missing_ok=True)
        testutils.cleanup()
