"""Short no-floor/no-excision FO-GH puncture evolution smoke test."""

from pathlib import Path
import shutil

import numpy as np
import test_suite.testutils as testutils


def constraint_family_norms(history):
    """Return masked proper-volume L2 norms for H, M, GH, and reduction families."""
    component_norms = np.sqrt(history[:, 2:10] / history[:, -2, None])
    return np.column_stack(
        (
            component_norms[:, 0],
            component_norms[:, 1],
            np.linalg.norm(component_norms[:, 2:4], axis=1),
            np.linalg.norm(component_norms[:, 4:8], axis=1),
        )
    )


def curl_family_norms(history):
    """Return the combined masked proper-volume curl-constraint L2 norm."""
    return np.sqrt(history[:, 10] / history[:, -2])


def near_constraint_family_norms(history):
    """Return fixed-radius masked H, M, GH, and reduction/curl L2 norms."""
    return np.sqrt(history[:, 16:20] / history[:, -1, None])


def test_fo_gh_puncture_evolution():
    history = Path("fo_gh_puncture_evolution.fo_gh.hst")
    try:
        testutils.run("inputs/fo_gh_puncture_evolution.athinput")
        data = np.loadtxt(history)
        checkpoint = np.loadtxt("fo_gh_puncture_evolution-checkpoint.dat")
        assert data.shape[1] == 22
        assert np.all(np.isfinite(data))
        assert np.all(np.isfinite(curl_family_norms(data)))
        assert np.all(np.isfinite(near_constraint_family_norms(data)))
        assert checkpoint.shape == (43,)
        assert int(checkpoint[3]) == 1
        assert np.all(np.isfinite(checkpoint))
        final = data[-1]
        volume = final[-2]
        history_norms = np.array(
            [
                np.sqrt(final[2] / volume),
                np.sqrt(final[3] / volume),
                np.sqrt(np.sum(final[4:6]) / volume),
                np.sqrt(np.sum(final[6:11]) / volume),
            ]
        )
        checkpoint_norms = checkpoint[[18, 21, 24, 27]]
        np.testing.assert_allclose(checkpoint_norms, history_norms, rtol=2.0e-14)
    finally:
        history.unlink(missing_ok=True)
        testutils.cleanup()


def test_fo_gh_common_adm_fixed_region_history():
    """Common unmasked ADM norms must cover fixed regions and use finite maxima."""
    basename = "fo_gh_common_adm"
    histories = [Path(f"{basename}.adm_common{index}.hst") for index in range(6)]
    try:
        testutils.run(
            "inputs/fo_gh_puncture_evolution.athinput",
            [
                f"job/basename={basename}",
                "time/tlim=0.0",
                "time/nlim=0",
                "problem/common_adm_history=true",
                "problem/common_adm_fd_order=4",
            ],
        )
        data = [np.atleast_2d(np.loadtxt(history)) for history in histories]
        assert data[0].shape[1] == 19
        assert all(chunk.shape[1] == 16 for chunk in data[1:])
        assert all(np.all(np.isfinite(chunk)) for chunk in data)
        # On the uniform [-4,4]^3 N=16 mesh, 1/dx is exactly two.
        assert data[0][-1, 16] == 2.0
        # The fixed regions r<2, 2<=r<4, 4<=r<8, and r>=8 partition the domain.
        np.testing.assert_allclose(
            data[0][-1, 8], data[1][-1, 8] + data[1][-1, 15]
            + data[2][-1, 8] + data[2][-1, 15], rtol=2.0e-14
        )
    finally:
        for history in histories:
            history.unlink(missing_ok=True)
        Path(f"{basename}.fo_gh.hst").unlink(missing_ok=True)
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
