"""One-step scalar-field backreaction regression for Z4c."""

import math
from pathlib import Path

import athena_read
import pytest
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_coupling.athinput"
_DEFAULT_INPUT_FILE = "inputs/scalar_field_coupling_default.athinput"
_MHD_INPUT_FILE = "inputs/scalar_field_mhd_coupling.athinput"
_MASS = 0.7
_AMPLITUDE = 0.2
_GAMMA = 1.5
_DENSITY = 1.25
_PRESSURE = 0.3
_MAGNETIC_FIELD = (0.4, -0.2, 0.1)
_REL_TOL = 5.0e-12
_ABS_TOL = 5.0e-14

_CYCLE = 0
_TIME = 1
_DT = 2
_BACKREACTION = 3
_HAS_TMUNU = 4
_KHAT = 5
_THETA = 6
_SXX = 7
_SXY = 8
_SXZ = 9
_SYY = 10
_SYZ = 11
_SZZ = 12
_ENERGY = 13
_SX = 14
_SY = 15
_SZ = 16
_TMUNU_COLUMNS = (_SXX, _SXY, _SXZ, _SYY, _SYZ, _SZZ, _ENERGY, _SX, _SY, _SZ)


def test_scalar_field_backreaction_defaults_off():
    """An omitted backreaction switch must leave Z4c unsourced."""
    basename = "scalar_field_coupling_default"
    diagnostic = Path(f"{basename}-coupling.dat")
    diagnostic.unlink(missing_ok=True)

    try:
        assert testutils.run(
            _DEFAULT_INPUT_FILE,
            [f"job/basename={basename}"],
        )
        row = athena_read.error_dat(diagnostic)[0]
        assert int(row[_BACKREACTION]) == 0
        for column in (_KHAT, _THETA) + _TMUNU_COLUMNS:
            assert row[column] == pytest.approx(0.0, abs=_ABS_TOL)
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("backreaction", (False, True), ids=("off", "on"))
def test_scalar_field_coupling(backreaction):
    """Check the scalar stress tensor and its one-step Z4c source."""
    suffix = "on" if backreaction else "off"
    basename = f"scalar_field_coupling_{suffix}"
    diagnostic = Path(f"{basename}-coupling.dat")
    diagnostic.unlink(missing_ok=True)
    arguments = [
        f"job/basename={basename}",
        f"scalar_field/backreaction={str(backreaction).lower()}",
    ]

    try:
        assert testutils.run(_INPUT_FILE, arguments)
        data = athena_read.error_dat(diagnostic)
        assert data.shape == (1, 17)
        row = data[0]
        assert all(math.isfinite(value) for value in row)
        assert int(row[_CYCLE]) == 1
        assert row[_TIME] == pytest.approx(row[_DT], rel=_REL_TOL, abs=_ABS_TOL)
        assert bool(int(row[_BACKREACTION])) is backreaction

        if backreaction:
            potential = 0.5 * _MASS * _MASS * _AMPLITUDE * _AMPLITUDE
            z4c_source = -8.0 * math.pi * potential * row[_DT]
            final_pi = row[_DT] * _MASS * _MASS * _AMPLITUDE
            final_kinetic = 0.5 * final_pi * final_pi
            final_energy = potential + final_kinetic
            final_pressure = -potential + final_kinetic
            assert int(row[_HAS_TMUNU]) == 1
            assert row[_KHAT] == pytest.approx(z4c_source, rel=_REL_TOL, abs=_ABS_TOL)
            assert row[_THETA] == pytest.approx(z4c_source, rel=_REL_TOL, abs=_ABS_TOL)
            assert row[_ENERGY] == pytest.approx(
                final_energy, rel=_REL_TOL, abs=_ABS_TOL
            )
            for column in (_SXX, _SYY, _SZZ):
                assert row[column] == pytest.approx(
                    final_pressure, rel=_REL_TOL, abs=_ABS_TOL
                )
            for column in (_SXY, _SXZ, _SYZ, _SX, _SY, _SZ):
                assert row[column] == pytest.approx(0.0, abs=_ABS_TOL)
        else:
            assert int(row[_HAS_TMUNU]) in (0, 1)
            for column in (_KHAT, _THETA) + _TMUNU_COLUMNS:
                assert row[column] == pytest.approx(0.0, abs=_ABS_TOL)
    finally:
        testutils.cleanup()


def test_scalar_field_mhd_tmunu_additivity():
    """Require the shared Tmunu accumulator to contain MHD plus scalar matter."""
    basename = "scalar_field_mhd_coupling"
    diagnostic = Path(f"{basename}-coupling.dat")
    diagnostic.unlink(missing_ok=True)

    try:
        assert testutils.run(_MHD_INPUT_FILE, [f"job/basename={basename}"])
        data = athena_read.error_dat(diagnostic)
        assert data.shape == (1, 17)
        row = data[0]
        assert all(math.isfinite(value) for value in row)
        assert int(row[_CYCLE]) == 1
        assert int(row[_BACKREACTION]) == 1
        assert int(row[_HAS_TMUNU]) == 1

        magnetic_squared = sum(component * component for component in _MAGNETIC_FIELD)
        internal_energy = _PRESSURE / (_GAMMA - 1.0)
        mhd_energy = _DENSITY + internal_energy + 0.5 * magnetic_squared
        scalar_potential = 0.5 * _MASS * _MASS * _AMPLITUDE * _AMPLITUDE
        final_pi = row[_DT] * _MASS * _MASS * _AMPLITUDE
        scalar_kinetic = 0.5 * final_pi * final_pi
        total_energy = mhd_energy + scalar_potential + scalar_kinetic

        isotropic_mhd_stress = _PRESSURE + 0.5 * magnetic_squared
        mhd_stress = (
            isotropic_mhd_stress - _MAGNETIC_FIELD[0] ** 2,
            -_MAGNETIC_FIELD[0] * _MAGNETIC_FIELD[1],
            -_MAGNETIC_FIELD[0] * _MAGNETIC_FIELD[2],
            isotropic_mhd_stress - _MAGNETIC_FIELD[1] ** 2,
            -_MAGNETIC_FIELD[1] * _MAGNETIC_FIELD[2],
            isotropic_mhd_stress - _MAGNETIC_FIELD[2] ** 2,
        )
        scalar_stress = (
            -scalar_potential + scalar_kinetic,
            0.0,
            0.0,
            -scalar_potential + scalar_kinetic,
            0.0,
            -scalar_potential + scalar_kinetic,
        )
        total_stress = tuple(
            mhd + scalar for mhd, scalar in zip(mhd_stress, scalar_stress)
        )

        assert row[_ENERGY] == pytest.approx(total_energy, rel=_REL_TOL, abs=_ABS_TOL)
        for column, expected in zip((_SXX, _SXY, _SXZ, _SYY, _SYZ, _SZZ), total_stress):
            assert row[column] == pytest.approx(expected, rel=_REL_TOL, abs=_ABS_TOL)
        for column in (_SX, _SY, _SZ):
            assert row[column] == pytest.approx(0.0, abs=_ABS_TOL)

        # Either producer overwriting the other would leave one of these values.
        assert abs(row[_ENERGY] - mhd_energy) > 100.0 * _ABS_TOL
        assert abs(row[_ENERGY] - scalar_potential) > 100.0 * _ABS_TOL

        initial_stress_trace = sum(mhd_stress[index] for index in (0, 3, 5))
        initial_stress_trace -= 3.0*scalar_potential
        initial_total_energy = mhd_energy + scalar_potential
        khat_source = 4.0 * math.pi * (
            initial_stress_trace + initial_total_energy
        )
        theta_source = -8.0 * math.pi * initial_total_energy
        assert row[_KHAT] == pytest.approx(
            khat_source * row[_DT], rel=_REL_TOL, abs=_ABS_TOL
        )
        assert row[_THETA] == pytest.approx(
            theta_source * row[_DT], rel=_REL_TOL, abs=_ABS_TOL
        )
    finally:
        testutils.cleanup()
