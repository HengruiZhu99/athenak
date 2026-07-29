"""Bitwise-sensitive continuation test for complex scalar restart data."""

from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_oscillator.athinput"
_VARIABLES = ("sf_phi0", "sf_pi0", "sf_phi1", "sf_pi1")
_FINAL_CYCLE = 12
_MID_CYCLE = _FINAL_CYCLE // 2
_ROUND_OFF_TOL = 32.0*np.finfo(np.float64).eps


def _run_arguments(basename, nlim):
    """Return common complex-oscillator arguments for a cycle-limited leg."""
    return [
        f"job/basename={basename}",
        "problem/test_case=complex_free",
        "problem/amplitude=0.2",
        "scalar_field/field_type=complex",
        "scalar_field/potential=free",
        "scalar_field/mass=0.7",
        "scalar_field/lambda=0.0",
        "time/integrator=rk4",
        "time/cfl_number=0.4",
        "time/tlim=10.0",
        f"time/nlim={nlim}",
    ]


def _tab_intervals(interval):
    """Enable or disable all four direct scalar table outputs together."""
    return [f"output{index}/dt={interval}" for index in range(1, 5)]


def _read_state(basename, file_number):
    """Read every canonical complex-field component from final table dumps."""
    state = {}
    for variable in _VARIABLES:
        filename = f"tab/{basename}.{variable}.{file_number:05d}.tab"
        table = athena_read.tab(filename)
        state[variable] = np.asarray(table[variable], dtype=np.float64)
    return state


def _read_diagnostic(basename):
    """Read the single oscillator diagnostic written at finalization."""
    data = athena_read.error_dat(f"{basename}-errs.dat")
    assert data.shape == (1, 11)
    return data[0]


def _remove_restart_files(*basenames):
    """Remove only restart artifacts owned by this regression."""
    restart_dir = Path("rst")
    for basename in basenames:
        for filename in restart_dir.glob(f"{basename}.*.rst"):
            filename.unlink()
    if restart_dir.exists() and not any(restart_dir.iterdir()):
        restart_dir.rmdir()


def test_complex_scalar_restart_continuation():
    """Restarted and uninterrupted runs must agree to double precision."""
    full_basename = "scalar_field_restart_full"
    split_basename = "scalar_field_restart_split"
    resumed_basename = "scalar_field_restart_resumed"
    owned_basenames = (full_basename, split_basename, resumed_basename)
    _remove_restart_files(*owned_basenames)

    try:
        full_arguments = [
            *_run_arguments(full_basename, _FINAL_CYCLE),
            *_tab_intervals(10.0),
            "output5/dt=0.0",
        ]
        assert testutils.run(_INPUT_FILE, full_arguments)
        full_state = _read_state(full_basename, 1)
        full_diagnostic = _read_diagnostic(full_basename)

        split_arguments = [
            *_run_arguments(split_basename, _MID_CYCLE),
            *_tab_intervals(0.0),
            "output5/dt=10.0",
        ]
        assert testutils.run(_INPUT_FILE, split_arguments)
        restart_file = Path(
            f"rst/{split_basename}.00001.rst"
        )
        assert restart_file.is_file()

        restart_arguments = [
            "./athena",
            "-r",
            str(restart_file),
            f"job/basename={resumed_basename}",
            f"time/nlim={_FINAL_CYCLE}",
            "time/tlim=10.0",
            *_tab_intervals(10.0),
            "output5/dt=0.0",
        ]
        assert testutils.run_command(restart_arguments)
        resumed_state = _read_state(resumed_basename, 0)
        resumed_diagnostic = _read_diagnostic(resumed_basename)

        for variable in _VARIABLES:
            assert full_state[variable].shape == resumed_state[variable].shape
            np.testing.assert_allclose(
                resumed_state[variable],
                full_state[variable],
                rtol=0.0,
                atol=_ROUND_OFF_TOL,
                err_msg=f"{variable} changed across restart",
            )

        # Columns 3:7 independently cover the four component L1 errors.  Columns
        # 8:10 are the homogeneous scalar energy and U(1) charge drifts.
        np.testing.assert_allclose(
            resumed_diagnostic[3:7],
            full_diagnostic[3:7],
            rtol=0.0,
            atol=_ROUND_OFF_TOL,
        )
        np.testing.assert_allclose(
            resumed_diagnostic[8:10],
            full_diagnostic[8:10],
            rtol=0.0,
            atol=_ROUND_OFF_TOL,
        )
    finally:
        _remove_restart_files(*owned_basenames)
        testutils.cleanup()
