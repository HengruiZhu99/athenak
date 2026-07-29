"""Two-rank per-rank-file restart continuation for a complex scalar."""

from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_oscillator.athinput"
_FINAL_CYCLE = 12
_MID_CYCLE = _FINAL_CYCLE // 2
_TOLERANCE = 64.0*np.finfo(np.float64).eps


def _arguments(basename, nlim):
    """Return common complex oscillator arguments."""
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


def _diagnostic(basename):
    """Read scalar component, phase, energy, and charge errors."""
    row = athena_read.error_dat(f"{basename}-errs.dat")[0]
    assert row.shape == (11,)
    return row


def _restart_file(rank, basename):
    """Return one rank's restart path."""
    return Path(
        f"rst/rank_{rank:08d}/{basename}.00001.rst"
    )


def _remove_restart_files(basename):
    """Remove only per-rank restart artifacts owned by this test."""
    for rank in range(2):
        filename = _restart_file(rank, basename)
        filename.unlink(missing_ok=True)
        directory = filename.parent
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    restart_dir = Path("rst")
    if restart_dir.exists() and not any(restart_dir.iterdir()):
        restart_dir.rmdir()


def test_complex_scalar_per_rank_restart():
    """Every rank must resume its own four-component scalar state."""
    full_name = "scalar_field_rank_restart_full"
    split_name = "scalar_field_rank_restart_split"
    resumed_name = "scalar_field_rank_restart_resumed"
    _remove_restart_files(split_name)

    try:
        assert testutils.mpi_run(
            _INPUT_FILE,
            [
                *_arguments(full_name, _FINAL_CYCLE),
                "output5/dt=0.0",
            ],
            threads=2,
        )
        full = _diagnostic(full_name)

        assert testutils.mpi_run(
            _INPUT_FILE,
            [
                *_arguments(split_name, _MID_CYCLE),
                "output5/dt=10.0",
                "output5/single_file_per_rank=true",
            ],
            threads=2,
        )
        for rank in range(2):
            restart_file = _restart_file(rank, split_name)
            assert restart_file.is_file()
            assert restart_file.stat().st_size > 0

        command = [
            "mpirun",
            "-np",
            "2",
            "./athena",
            "-r",
            str(_restart_file(0, split_name)),
            f"job/basename={resumed_name}",
            f"time/nlim={_FINAL_CYCLE}",
            "time/tlim=10.0",
            "output5/dt=0.0",
        ]
        assert testutils.run_command(command)
        resumed = _diagnostic(resumed_name)

        np.testing.assert_allclose(
            resumed[2:10], full[2:10], rtol=0.0, atol=_TOLERANCE
        )
    finally:
        _remove_restart_files(split_name)
        testutils.cleanup()
