"""Z4c-parity robust Minkowski stability tests for vacuum FO-GH."""

import numpy as np
import pytest
import test_suite.testutils as testutils


def _check_history():
    data = np.loadtxt("fo_gh_stability.user.hst")
    linf = data[:, 2]
    if not np.all(np.isfinite(data)):
        pytest.fail("FO-GH robust stability history contains nonfinite values.")
    if linf[-1] > 1.5 * linf[0]:
        pytest.fail(
            f"FO-GH robust perturbation grew too much: "
            f"initial={linf[0]}, final={linf[-1]}"
        )


def test_fo_gh_stability_uniform():
    try:
        result = testutils.run("inputs/fo_gh_stability.athinput")
        assert result, "Uniform FO-GH robust stability run failed."
        _check_history()
    finally:
        testutils.cleanup()


def test_fo_gh_stability_smr():
    try:
        result = testutils.run(
            "inputs/fo_gh_stability.athinput",
            [
                "mesh/nx1=16",
                "meshblock/nx1=8",
                "mesh_refinement/refinement=static",
                "time/tlim=0.01",
            ],
        )
        assert result, "SMR FO-GH robust stability run failed."
        _check_history()
    finally:
        testutils.cleanup()
