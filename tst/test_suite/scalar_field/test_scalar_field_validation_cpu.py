"""Focused scalar-field input, boundary, and timestep validation."""

import math
from pathlib import Path

import athena_read
import pytest
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_validation.athinput"
_MISSING_GEOMETRY_INPUT = "inputs/scalar_field_missing_geometry.athinput"
_WAVE_INPUT = "inputs/scalar_field_wave.athinput"
_Z4C_INPUT = "inputs/scalar_field_coupling.athinput"


def _new_log_text(offset):
    """Return test-log output written after ``offset``."""
    log_path = Path(testutils.LOG_FILE_PATH)
    with log_path.open("rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8", errors="replace")


@pytest.mark.parametrize(
    ("input_file", "arguments", "expected_message"),
    (
        (
            _INPUT_FILE,
            ["scalar_field/field_type=quaternion"],
            "field_type must be real or complex",
        ),
        (
            _INPUT_FILE,
            ["scalar_field/potential=cubic"],
            "potential must be free or mass_quartic",
        ),
        (
            _INPUT_FILE,
            ["scalar_field/mass=-0.1"],
            "mass must be nonnegative",
        ),
        (
            _INPUT_FILE,
            ["scalar_field/lambda=-0.1"],
            "lambda must be nonnegative",
        ),
        (
            _MISSING_GEOMETRY_INPUT,
            [],
            "requires either an <adm> or <z4c> block",
        ),
        (
            _INPUT_FILE,
            ["mesh/nghost=3", "scalar_field/spatial_order=6"],
            "spatial_order requires at least 4 ghost cells",
        ),
        (
            _Z4C_INPUT,
            ["scalar_field/spatial_order=4"],
            "spatial_order must match <z4c>/spatial_order",
        ),
        (
            _INPUT_FILE,
            [
                "coord/minkowski=false",
                "coord/excise=true",
                "coord/smooth_excision=true",
                "scalar_field/excision_tdamp=0.0",
            ],
            "excision_tdamp must be finite and positive",
        ),
        (
            _INPUT_FILE,
            ["scalar_field/backreaction=true"],
            "backreaction=true requires Z4c",
        ),
        (
            _INPUT_FILE,
            [
                "mesh/nghost=6",
                "mesh_refinement/refinement=adaptive",
            ],
            "multilevel scalar evolution requires nghost=2 or 4",
        ),
        (
            _INPUT_FILE,
            ["mesh_refinement/refinement=adaptive"],
            "at least 2*nghost cells per active MeshBlock dimension",
        ),
        (
            _WAVE_INPUT,
            [
                "mesh/ix1_bc=shear_periodic",
                "mesh/ox1_bc=shear_periodic",
            ],
            "does not support shear-periodic boundaries",
        ),
        (
            _INPUT_FILE,
            ["coord/minkowski=false", "coord/a=0.0", "coord/excise=true"],
            "requires <coord>/smooth_excision=true",
        ),
    ),
    ids=(
        "unknown-field-type",
        "unknown-potential",
        "negative-mass",
        "negative-lambda",
        "missing-geometry",
        "insufficient-ghosts",
        "z4c-order-mismatch",
        "invalid-excision-damping",
        "fixed-adm-backreaction",
        "unsupported-multilevel-ghosts",
        "undersized-multilevel-block",
        "shear-periodic-boundaries",
        "nonsmooth-black-hole-excision",
    ),
)
def test_invalid_scalar_inputs(input_file, arguments, expected_message):
    """Every unsupported configuration must fail early with a useful reason."""
    log_path = Path(testutils.LOG_FILE_PATH)
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    try:
        with pytest.raises(RuntimeError):
            testutils.run(input_file, arguments)
        assert expected_message in _new_log_text(log_offset)
    finally:
        testutils.cleanup()


def _run_one_cycle(basename, arguments):
    """Run one oscillator cycle and return its deterministic diagnostic row."""
    flags = [
        f"job/basename={basename}",
        "time/nlim=1",
        "time/tlim=100.0",
        *arguments,
    ]
    assert testutils.run(_INPUT_FILE, flags)
    data = athena_read.error_dat(f"{basename}-errs.dat")
    assert data.shape == (1, 11)
    assert all(math.isfinite(value) for value in data[0])
    return data[0]


def test_mass_frequency_restricts_timestep():
    """A stiff massive field must reduce the timestep below the light-cone limit."""
    try:
        light = _run_one_cycle(
            "scalar_field_dt_light",
            ["scalar_field/mass=0.7"],
        )
        stiff = _run_one_cycle(
            "scalar_field_dt_stiff",
            ["scalar_field/mass=100.0"],
        )
        stiff_lapse = _run_one_cycle(
            "scalar_field_dt_stiff_lapse",
            [
                "scalar_field/mass=100.0",
                "problem/lapse=2.5",
            ],
        )

        assert light[1] == stiff[1] == stiff_lapse[1] == 1
        assert light[2] == pytest.approx(0.4/8.0, rel=2.0e-14)
        assert stiff[2] == pytest.approx(0.4/100.0, rel=2.0e-14)
        assert stiff_lapse[2] == pytest.approx(
            0.4/(2.5*100.0), rel=2.0e-14
        )
        assert stiff[2] < light[2]/10.0
        assert stiff_lapse[2] < stiff[2]
    finally:
        testutils.cleanup()


def test_scalar_physical_boundaries():
    """Even reflection and extrapolating boundaries preserve homogeneous data."""
    rows = {}
    try:
        preserving_boundaries = ("reflect", "outflow", "diode", "vacuum")
        for boundary in (*preserving_boundaries, "inflow"):
            rows[boundary] = _run_one_cycle(
                f"scalar_field_bc_{boundary}",
                [
                    f"mesh/ix1_bc={boundary}",
                    f"mesh/ox1_bc={boundary}",
                    "scalar_field/mass=0.7",
                ],
            )

        for boundary in preserving_boundaries:
            assert rows[boundary][3] < 1.0e-9
            assert rows[boundary][4] < 1.0e-9

        inflow_error = max(rows["inflow"][3], rows["inflow"][4])
        preserving_error = max(
            rows[boundary][3:5].max() for boundary in preserving_boundaries
        )
        assert inflow_error > 1.0e-6
        assert inflow_error > 100.0*preserving_error
    finally:
        testutils.cleanup()
