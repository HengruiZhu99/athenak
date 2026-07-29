"""Homogeneous real, complex, and quartic scalar oscillator tests."""

import math

import test_suite.testutils as testutils
import athena_read


_INPUT_FILE = "inputs/scalar_field_oscillator.athinput"


def _read_single_row(basename):
    """Read and validate one deterministic oscillator diagnostic row."""
    data = athena_read.error_dat(f"{basename}-errs.dat")
    assert data.shape == (1, 11)
    assert all(math.isfinite(value) for value in data[0])
    return data[0]


def test_real_massive_oscillator():
    """Check real-oscillator state, phase, energy, and analytic Tmunu."""
    basename = "scalar_field_real_oscillator"
    try:
        testutils.run(
            _INPUT_FILE,
            [
                f"job/basename={basename}",
                "problem/test_case=real_free",
                "problem/amplitude=0.2",
                "scalar_field/field_type=real",
                "scalar_field/potential=free",
                "scalar_field/mass=0.7",
                "scalar_field/lambda=0.0",
                "time/tlim=1.1",
            ],
        )
        row = _read_single_row(basename)
        assert max(row[3], row[4]) < 1.0e-7
        assert row[7] < 1.0e-6
        assert row[8] < 1.0e-8
        assert row[9] == 0.0
        assert row[10] < 1.0e-7
    finally:
        testutils.cleanup()


def test_complex_phase_rotation():
    """Check phase, energy, charge, and time-independent complex Tmunu."""
    basename = "scalar_field_complex_oscillator"
    try:
        testutils.run(
            _INPUT_FILE,
            [
                f"job/basename={basename}",
                "problem/test_case=complex_free",
                "problem/amplitude=0.2",
                "scalar_field/field_type=complex",
                "scalar_field/potential=free",
                "scalar_field/mass=0.7",
                "scalar_field/lambda=0.0",
                "time/tlim=1.3",
            ],
        )
        row = _read_single_row(basename)
        assert max(row[3:7]) < 1.0e-7
        assert row[7] < 1.0e-6
        assert row[8] < 1.0e-8
        assert row[9] < 1.0e-8
        assert row[10] < 1.0e-7
    finally:
        testutils.cleanup()


def test_quartic_ode_convergence():
    """Verify fourth-order convergence to the mass-quartic ODE reference."""
    basename = "scalar_field_quartic_oscillator"
    resolutions = (8, 16)
    try:
        for resolution in resolutions:
            testutils.run(
                _INPUT_FILE,
                [
                    f"job/basename={basename}",
                    f"mesh/nx1={resolution}",
                    "meshblock/nx1=4",
                    "problem/test_case=real_quartic",
                    "problem/amplitude=0.4",
                    "scalar_field/field_type=real",
                    "scalar_field/potential=mass_quartic",
                    "scalar_field/mass=0.7",
                    "scalar_field/lambda=1.3",
                    "time/cfl_number=0.4",
                    "time/tlim=0.5",
                ],
            )

        data = athena_read.error_dat(f"{basename}-errs.dat")
        assert data.shape == (2, 11)
        assert all(math.isfinite(value) for row in data for value in row)
        for column, name in ((3, "phi"), (4, "Pi")):
            coarse_error = data[0, column]
            fine_error = data[1, column]
            assert coarse_error > 0.0
            assert fine_error > 0.0
            order = math.log(coarse_error/fine_error, 2.0)
            assert order >= 3.5, (
                f"quartic {name} observed order {order:.6g} is below 3.5; "
                f"errors were {coarse_error:.6g}, {fine_error:.6g}"
            )
        assert data[1, 8] < data[0, 8]
        assert data[1, 10] < data[0, 10]
    finally:
        testutils.cleanup()
