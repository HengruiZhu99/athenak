"""Restart continuation test covering coupled Z4c and complex scalar state."""

from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_z4c_restart.athinput"
_FINAL_CYCLE = 8
_MID_CYCLE = _FINAL_CYCLE // 2
_TOLERANCE = 64.0*np.finfo(np.float64).eps


def _intervals(interval):
    """Set both state-table intervals together."""
    return [f"output{index}/dt={interval}" for index in (1, 2)]


def _read_state(basename, file_number):
    """Read all evolved scalar and Z4c table columns."""
    state = {}
    for group in ("sf", "z4c"):
        filename = f"tab/{basename}.{group}.{file_number:05d}.tab"
        table = athena_read.tab(filename)
        for name, values in table.items():
            if name.startswith("sf_") or name.startswith("z4c_"):
                state[name] = np.asarray(values, dtype=np.float64)
    return state


def _remove_restart(basename):
    """Remove one restart artifact owned by this test."""
    filename = Path(f"rst/{basename}.00001.rst")
    filename.unlink(missing_ok=True)
    restart_dir = filename.parent
    if restart_dir.exists() and not any(restart_dir.iterdir()):
        restart_dir.rmdir()


def test_z4c_complex_scalar_restart_continuation():
    """A split run must reproduce every evolved Z4c and scalar component."""
    full_name = "scalar_field_z4c_restart_full"
    split_name = "scalar_field_z4c_restart_split"
    resumed_name = "scalar_field_z4c_restart_resumed"
    _remove_restart(split_name)

    try:
        assert testutils.run(
            _INPUT_FILE,
            [
                f"job/basename={full_name}",
                f"time/nlim={_FINAL_CYCLE}",
                *_intervals(10.0),
                "output3/dt=0.0",
            ],
        )
        full = _read_state(full_name, 1)

        assert testutils.run(
            _INPUT_FILE,
            [
                f"job/basename={split_name}",
                f"time/nlim={_MID_CYCLE}",
                *_intervals(0.0),
                "output3/dt=10.0",
            ],
        )
        restart_file = Path(f"rst/{split_name}.00001.rst")
        assert restart_file.is_file()

        command = [
            "./athena",
            "-r",
            str(restart_file),
            f"job/basename={resumed_name}",
            f"time/nlim={_FINAL_CYCLE}",
            "time/tlim=10.0",
            *_intervals(10.0),
            "output3/dt=0.0",
        ]
        assert testutils.run_command(command)
        resumed = _read_state(resumed_name, 0)

        assert set(resumed) == set(full)
        assert {"sf_phi0", "sf_pi0", "sf_phi1", "sf_pi1"} <= set(full)
        assert len([name for name in full if name.startswith("z4c_")]) == 25
        assert np.max(np.abs(full["z4c_Khat"])) > 0.0
        assert np.max(np.abs(full["z4c_Theta"])) > 0.0
        for variable in full:
            np.testing.assert_allclose(
                resumed[variable],
                full[variable],
                rtol=0.0,
                atol=_TOLERANCE,
                err_msg=f"{variable} changed across Z4c+scalar restart",
            )
    finally:
        _remove_restart(split_name)
        testutils.cleanup()
