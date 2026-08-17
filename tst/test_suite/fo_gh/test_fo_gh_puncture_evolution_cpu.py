"""Short no-floor/no-excision FO-GH puncture evolution smoke test."""

from pathlib import Path
import shutil

import numpy as np
import test_suite.testutils as testutils


def constraint_family_norms(history):
    """Return masked proper-volume L2 norms for H, M, GH, and reduction families."""
    component_norms = np.sqrt(history[:, 2:14] / history[:, -1, None])
    return np.column_stack(
        (
            component_norms[:, 0],
            np.linalg.norm(component_norms[:, 1:4], axis=1),
            np.linalg.norm(component_norms[:, 4:8], axis=1),
            np.linalg.norm(component_norms[:, 8:12], axis=1),
        )
    )


def test_fo_gh_puncture_evolution():
    history = Path("fo_gh_puncture_evolution.fo_gh.hst")
    try:
        testutils.run("inputs/fo_gh_puncture_evolution.athinput")
        data = np.loadtxt(history)
        checkpoint = np.loadtxt("fo_gh_puncture_evolution-checkpoint.dat")
        assert data.shape[1] == 15
        assert np.all(np.isfinite(data))
        assert checkpoint.shape == (41,)
        assert int(checkpoint[3]) == 1
        assert np.all(np.isfinite(checkpoint))
    finally:
        history.unlink(missing_ok=True)
        testutils.cleanup()


def test_fo_gh_puncture_constraint_convergence():
    """The lapse-excised history norms must improve on a resolved three-grid ladder."""
    history = Path("fo_gh_puncture_convergence.fo_gh.hst")
    resolutions = (16, 24, 32)
    final_norms = []
    try:
        for resolution in resolutions:
            history.unlink(missing_ok=True)
            testutils.run(
                "inputs/fo_gh_puncture_evolution.athinput",
                [
                    "job/basename=fo_gh_puncture_convergence",
                    "time/tlim=0.01",
                    "output1/dt=0.01",
                    f"mesh/nx1={resolution}",
                    f"mesh/nx2={resolution}",
                    f"mesh/nx3={resolution}",
                    f"meshblock/nx1={resolution}",
                    f"meshblock/nx2={resolution}",
                    f"meshblock/nx3={resolution}",
                ],
            )
            data = np.atleast_2d(np.loadtxt(history))
            assert np.all(np.isfinite(data))
            assert data[-1, 0] == 0.01
            final_norms.append(constraint_family_norms(data)[-1])

        final_norms = np.asarray(final_norms)
        assert np.all(final_norms[2] < final_norms[1])
        fine_orders = np.log(final_norms[1] / final_norms[2]) / np.log(32.0 / 24.0)
        assert np.all(fine_orders > 1.5), (
            f"Masked puncture constraints did not converge: norms={final_norms}, "
            f"fine_orders={fine_orders}"
        )
    finally:
        history.unlink(missing_ok=True)
        testutils.cleanup()


def test_fo_gh_puncture_bounded_time_ladder():
    """A modest local time extension must remain finite with bounded masked norms."""
    history = Path("fo_gh_puncture_bounded.fo_gh.hst")
    try:
        testutils.run(
            "inputs/fo_gh_puncture_evolution.athinput",
            [
                "job/basename=fo_gh_puncture_bounded",
                "time/tlim=0.2",
                "output1/dt=0.02",
            ],
        )
        data = np.atleast_2d(np.loadtxt(history))
        checkpoint = np.loadtxt("fo_gh_puncture_bounded-checkpoint.dat")
        norms = constraint_family_norms(data)
        assert data[-1, 0] == 0.2
        assert np.all(np.isfinite(data))
        assert np.all(np.isfinite(checkpoint))
        assert int(checkpoint[3]) == 1
        assert norms[-1, 0] < 1.05 * norms[0, 0]
        assert norms[-1, 1] < 1.0e-2
        assert norms[-1, 2] < 1.0e-2
        assert norms[-1, 3] < 1.05 * norms[0, 3]
    finally:
        history.unlink(missing_ok=True)
        testutils.cleanup()


def test_fo_gh_puncture_restart_equivalence():
    """An identical-data puncture checkpoint must resume bit-for-bit at two cycles."""
    shutil.rmtree("rst", ignore_errors=True)
    try:
        testutils.run(
            "inputs/fo_gh_puncture_evolution.athinput",
            ["job/basename=fo_gh_puncture_direct", "time/nlim=2", "time/tlim=1.0"],
        )
        direct = np.array(np.loadtxt("fo_gh_puncture_direct-checkpoint.dat"), copy=True)
        testutils.cleanup()

        testutils.run(
            "inputs/fo_gh_puncture_evolution.athinput",
            [
                "job/basename=fo_gh_puncture_split",
                "time/nlim=1",
                "time/tlim=1.0",
                "output2/dcycle=1",
            ],
        )
        restart = Path("rst/fo_gh_puncture_split.00001.rst")
        assert restart.exists()
        testutils.cleanup()

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(restart),
                "job/basename=fo_gh_puncture_resumed",
                "time/nlim=2",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ]
        )
        resumed = np.loadtxt("fo_gh_puncture_resumed-checkpoint.dat")
        np.testing.assert_array_equal(resumed, direct)
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for basename in (
            "fo_gh_puncture_direct",
            "fo_gh_puncture_split",
            "fo_gh_puncture_resumed",
        ):
            Path(f"{basename}.fo_gh.hst").unlink(missing_ok=True)
        testutils.cleanup()
