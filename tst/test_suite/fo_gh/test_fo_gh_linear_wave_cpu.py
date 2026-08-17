"""Fourth-order CPU linear-wave convergence for regularized vacuum FO-GH."""

import math
from pathlib import Path
import shutil
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


def test_fo_gh_restart_equivalence():
    """A checkpoint boundary must reproduce uninterrupted FO-GH evolution exactly."""
    shutil.rmtree("rst", ignore_errors=True)
    try:
        assert testutils.run(
            "inputs/fo_gh_linear_wave.athinput",
            ["job/basename=fo_gh_direct", "time/nlim=2"],
        )
        direct = np.array(np.loadtxt("fo_gh_direct-errors.dat"), copy=True)
        testutils.cleanup()

        assert testutils.run(
            "inputs/fo_gh_linear_wave.athinput",
            [
                "job/basename=fo_gh_split",
                "time/nlim=1",
                "output1/dcycle=1",
            ],
        )
        restart = Path("rst/fo_gh_split.00001.rst")
        assert restart.exists()
        testutils.cleanup()

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(restart),
                "job/basename=fo_gh_resumed",
                "time/nlim=2",
                "output1/dcycle=0",
            ]
        )
        resumed = np.loadtxt("fo_gh_resumed-errors.dat")
        np.testing.assert_array_equal(resumed, direct)
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        testutils.cleanup()


def test_fo_gh_dynamic_regrid_gradient_repair():
    """A real regrid must restore every compatible first-order gradient."""
    try:
        assert testutils.run(
            "inputs/fo_gh_linear_wave_amr.athinput",
            ["mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8"],
        )
        data = np.loadtxt("fo_gh_linear_wave_amr-errors.dat")
        assert int(data[7]) == 8
        assert int(data[8]) == 7
        assert math.isfinite(data[6]) and data[6] < 5.0e-13
    finally:
        testutils.cleanup()
