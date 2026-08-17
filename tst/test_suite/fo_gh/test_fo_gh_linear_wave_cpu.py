"""Fourth-order CPU linear-wave convergence for regularized vacuum FO-GH."""

import math
import numpy as np
import pytest
import test_suite.testutils as testutils


def test_fo_gh_linear_wave_convergence():
    errors = []
    try:
        for resolution in (8, 16, 32):
            result = testutils.run(
                "inputs/fo_gh_linear_wave.athinput",
                [
                    f"mesh/nx1={resolution}",
                    f"meshblock/nx1={resolution}",
                    "time/tlim=0.05",
                    "problem/amp=1.0e-6",
                ],
            )
            assert result, f"FO-GH linear-wave run failed at nx1={resolution}."
            data = np.loadtxt("fo_gh_linear_wave-errors.dat")
            errors.append(float(data[4]))
        orders = [math.log(errors[n]/errors[n + 1], 2.0) for n in range(2)]
        if min(orders) < 3.7:
            pytest.fail(
                f"FO-GH fourth-order wave converged too slowly: "
                f"errors={errors}, orders={orders}"
            )
        if errors[-1] > 7.0e-13:
            pytest.fail(f"FO-GH fine-grid wave error too large: {errors[-1]}")
    finally:
        testutils.cleanup()


def test_fo_gh_linear_wave_smr_convergence():
    errors = []
    try:
        for resolution in (16, 32):
            result = testutils.run(
                "inputs/fo_gh_linear_wave.athinput",
                [
                    f"mesh/nx1={resolution}",
                    f"meshblock/nx1={resolution // 2}",
                    "mesh_refinement/refinement=static",
                    "time/tlim=0.002",
                    "problem/amp=1.0e-6",
                ],
            )
            assert result, f"FO-GH SMR wave run failed at nx1={resolution}."
            data = np.loadtxt("fo_gh_linear_wave-errors.dat")
            errors.append(float(data[4]))
        order = math.log(errors[0]/errors[1], 2.0)
        if order < 1.2:
            pytest.fail(
                f"FO-GH SMR wave did not converge: errors={errors}, order={order}"
            )
    finally:
        testutils.cleanup()
