"""Two-rank equivalence regression for adaptive scalar-field AMR."""

import math
from pathlib import Path

import athena_read
import numpy as np
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_amr_adaptive.athinput"
_ERROR_SLICE = slice(5, 8)
_MESH_SLICE = slice(8, 12)


def _read_row(basename):
    """Read one scalar-wave diagnostic row."""
    data = athena_read.error_dat(Path(f"{basename}-errs.dat"))
    assert data.shape == (1, 12)
    return data[0]


def _arguments(basename):
    """Use a short run that still performs and evolves after a regrid."""
    return [f"job/basename={basename}", "time/tlim=0.08"]


def _remove_history(basename):
    """Remove the scalar history artifact not covered by common cleanup."""
    Path(f"{basename}.scalar.hst").unlink(missing_ok=True)


def test_scalar_field_adaptive_amr_two_rank_equivalence():
    """Compare one-rank and two-rank AMR solutions and mesh histories."""
    serial_name = "scalar_field_amr_serial"
    mpi_name = "scalar_field_amr_mpi"
    _remove_history(serial_name)
    _remove_history(mpi_name)
    try:
        assert testutils.run(_INPUT_FILE, _arguments(serial_name))
        serial = _read_row(serial_name)

        assert testutils.mpi_run(
            _INPUT_FILE, _arguments(mpi_name), threads=2
        )
        parallel = _read_row(mpi_name)

        for value in np.concatenate(
            (serial[_ERROR_SLICE], parallel[_ERROR_SLICE])
        ):
            assert math.isfinite(value) and value > 0.0

        assert int(serial[9] + serial[10]) > 0
        assert int(parallel[9] + parallel[10]) > 0
        np.testing.assert_allclose(
            parallel[_ERROR_SLICE],
            serial[_ERROR_SLICE],
            rtol=5.0e-10,
            atol=2.0e-14,
        )
        np.testing.assert_array_equal(
            parallel[_MESH_SLICE], serial[_MESH_SLICE]
        )
        serial_energy = athena_read.hst(
            f"{serial_name}.scalar.hst"
        )["sf-energy"]
        parallel_energy = athena_read.hst(
            f"{mpi_name}.scalar.hst"
        )["sf-energy"]
        np.testing.assert_allclose(
            parallel_energy, serial_energy, rtol=5.0e-12, atol=2.0e-14
        )
    finally:
        _remove_history(serial_name)
        _remove_history(mpi_name)
        testutils.cleanup()
