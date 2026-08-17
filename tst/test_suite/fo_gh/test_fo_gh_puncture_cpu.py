"""Static four-resolution puncture boundedness test for vacuum FO-GH."""

from pathlib import Path
import shutil

import numpy as np
import pytest
import test_suite.testutils as testutils


def test_fo_gh_puncture_refinement():
    rhs_maxima = []
    try:
        for resolution in (8, 12, 16, 20):
            result = testutils.run(
                "inputs/fo_gh_puncture.athinput",
                [
                    f"mesh/nx1={resolution}",
                    f"mesh/nx2={resolution}",
                    f"mesh/nx3={resolution}",
                    f"meshblock/nx1={resolution}",
                    f"meshblock/nx2={resolution}",
                    f"meshblock/nx3={resolution}",
                ],
            )
            assert result, f"FO-GH puncture probe failed at N={resolution}."
            data = np.loadtxt("fo_gh_puncture-puncture.dat")
            checkpoint = np.loadtxt("fo_gh_puncture-checkpoint.dat")
            assert int(data[8]) == 0
            assert data[9] < 1.0e-13
            assert checkpoint.shape == (41,)
            assert int(checkpoint[3]) == 1
            assert np.all(np.isfinite(checkpoint))
            assert Path("bin/fo_gh_puncture.fo_gh.00000.bin").exists()
            assert Path("bin/fo_gh_puncture.adm.00000.bin").exists()
            assert Path("bin/fo_gh_puncture.fo_gh_con.00000.bin").exists()
            rhs_maxima.append(float(data[7]))
        if rhs_maxima[-1] > 2.0 * max(rhs_maxima[:-1]):
            pytest.fail(f"Puncture RHS appears unbounded: {rhs_maxima}")
    finally:
        shutil.rmtree("bin", ignore_errors=True)
        testutils.cleanup()
