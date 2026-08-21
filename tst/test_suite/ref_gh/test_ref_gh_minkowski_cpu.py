"""CPU dynamic Minkowski regression for reference-frame FO-GH."""

import test_suite.testutils as testutils
import math
import numpy as np


def test_ref_gh_minkowski():
    testutils.run("inputs/ref_gh_minkowski.athinput")


def test_ref_gh_time_dependent_reference():
    """Verify that each RK stage refreshes a genuinely time-dependent provider."""
    try:
        assert testutils.run("inputs/ref_gh_time_dependent_reference.athinput")
    finally:
        testutils.cleanup()


def test_ref_gh_linear_wave_convergence():
    errors = []
    try:
        for resolution in (8, 16, 32):
            assert testutils.run(
                "inputs/ref_gh_linear_wave.athinput",
                [f"mesh/nx1={resolution}", f"meshblock/nx1={resolution}"],
            )
            errors.append(float(np.loadtxt("ref_gh_linear_wave-errors.dat")[4]))
        orders = [math.log(errors[i]/errors[i + 1], 2.0) for i in range(2)]
        assert min(orders) > 3.6, f"errors={errors}, orders={orders}"
    finally:
        testutils.cleanup()
