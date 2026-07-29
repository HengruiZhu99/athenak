"""Direct, derived, and history output tests for the canonical scalar field."""

import math
from pathlib import Path
import shutil

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_outputs.athinput"
_WAVE_INPUT_FILE = "inputs/scalar_field_wave.athinput"
_BASENAME = "scalar_field_outputs"
_AMPLITUDE = 0.2
_MASS = 0.7
_TLIM = 0.6
_NCELLS = 24
_DOMAIN_VOLUME = (2.0 - (-1.0))*(1.5 - (-0.5))*(2.5 - (-1.5))
_DIRECT_VARIABLES = ("sf_phi0", "sf_pi0", "sf_phi1", "sf_pi1")
_SINGLE_VARIABLES = _DIRECT_VARIABLES + (
    "sf_amplitude",
    "sf_energy",
    "sf_charge",
)


def _expected_values(time):
    """Return the exact homogeneous rotating-field values at one time."""
    phase = _MASS*time
    return {
        "sf_phi0": _AMPLITUDE*math.cos(phase),
        "sf_pi0": _AMPLITUDE*_MASS*math.sin(phase),
        "sf_phi1": _AMPLITUDE*math.sin(phase),
        "sf_pi1": -_AMPLITUDE*_MASS*math.cos(phase),
        "sf_amplitude": _AMPLITUDE,
        "sf_energy": _MASS*_MASS*_AMPLITUDE*_AMPLITUDE,
        "sf_charge": _MASS*_AMPLITUDE*_AMPLITUDE,
    }


def _read_scalar_table(variable, file_number):
    """Read one table and retain only its stable scalar data columns."""
    filename = f"tab/{_BASENAME}.{variable}.{file_number:05d}.tab"
    data = athena_read.tab(filename)
    scalar_data = {
        name: values for name, values in data.items() if name.startswith("sf_")
    }
    return data, scalar_data


def _remove_history(filename):
    """Remove the history artifact that the common cleanup helper does not cover."""
    Path(filename).unlink(missing_ok=True)


def test_complex_field_and_history_outputs():
    """Validate direct/grouped fields, derived densities, and volume integrals."""
    history_file = f"{_BASENAME}.scalar.hst"
    _remove_history(history_file)
    try:
        assert testutils.run(_INPUT_FILE)

        for file_number, expected_time in ((0, 0.0), (1, _TLIM)):
            expected = _expected_values(expected_time)
            direct_data = {}

            for variable in _SINGLE_VARIABLES:
                table, scalar_data = _read_scalar_table(variable, file_number)
                assert table["time"] == pytest.approx(expected_time, abs=1.0e-14)
                assert set(scalar_data) == {variable}
                assert scalar_data[variable].shape == (_NCELLS,)
                np.testing.assert_allclose(
                    scalar_data[variable], expected[variable], rtol=0.0, atol=1.0e-7
                )
                if variable in _DIRECT_VARIABLES:
                    direct_data[variable] = scalar_data[variable]

            table, grouped_data = _read_scalar_table("sf", file_number)
            assert table["time"] == pytest.approx(expected_time, abs=1.0e-14)
            assert set(grouped_data) == set(_DIRECT_VARIABLES)
            for variable in _DIRECT_VARIABLES:
                np.testing.assert_array_equal(
                    grouped_data[variable], direct_data[variable]
                )

        history = athena_read.hst(history_file)
        assert list(history) == ["time", "dt", "sf-energy", "sf-charge"]
        np.testing.assert_allclose(history["time"], [0.0, _TLIM], rtol=0.0, atol=1.0e-14)
        expected_energy = _DOMAIN_VOLUME*_MASS*_MASS*_AMPLITUDE*_AMPLITUDE
        expected_charge = _DOMAIN_VOLUME*_MASS*_AMPLITUDE*_AMPLITUDE
        np.testing.assert_allclose(
            history["sf-energy"], expected_energy, rtol=0.0, atol=5.0e-7
        )
        np.testing.assert_allclose(
            history["sf-charge"], expected_charge, rtol=0.0, atol=5.0e-7
        )
    finally:
        _remove_history(history_file)
        testutils.cleanup()


def test_real_field_rejects_imaginary_output():
    """An imaginary-component request for a real field must fail clearly."""
    disabled_outputs = [f"output{number}/dt=0.0" for number in range(2, 10)]
    arguments = [
        "job/basename=scalar_field_invalid_real_output",
        "problem/test_case=real_free",
        "scalar_field/field_type=real",
        "output1/variable=sf_phi1",
        "time/nlim=0",
        *disabled_outputs,
    ]
    log_path = Path(testutils.LOG_FILE_PATH)
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    try:
        with pytest.raises(RuntimeError):
            testutils.run(_INPUT_FILE, arguments)
        with log_path.open("rb") as log_file:
            log_file.seek(log_offset)
            log_text = log_file.read().decode("utf-8", errors="replace")
        assert "sf_phi1" in log_text
        assert "real" in log_text.lower()
    finally:
        testutils.cleanup()


def test_scalar_history_uses_proper_volume():
    """A nonunit metric determinant must scale the integrated scalar energy."""
    basename = "scalar_field_proper_volume"
    history_file = f"{basename}.scalar.hst"
    _remove_history(history_file)
    try:
        assert testutils.run(
            _WAVE_INPUT_FILE,
            [
                f"job/basename={basename}",
                "time/nlim=0",
                "output1/dt=1.0",
            ],
        )
        history = athena_read.hst(history_file)
        assert list(history) == ["time", "dt", "sf-energy"]
        assert history["time"] == pytest.approx([0.0], abs=1.0e-14)

        determinant = 0.9375
        inverse_xx = 16.0/15.0
        inverse_xy = -4.0/15.0
        inverse_yy = 16.0/15.0
        wave_number = 2.0*math.pi
        wave_number_squared = (
            inverse_xx*wave_number**2
            + 2.0*inverse_xy*wave_number**2
            + inverse_yy*wave_number**2
        )
        resolution = 16
        discrete_wave_number = (
            math.sin(wave_number/resolution)*resolution
        )
        discrete_wave_number_squared = (
            inverse_xx*discrete_wave_number**2
            + 2.0*inverse_xy*discrete_wave_number**2
            + inverse_yy*discrete_wave_number**2
        )
        amplitude = 1.0e-3
        mass = 0.7
        expected_energy = (
            math.sqrt(determinant)
            * 0.25
            * amplitude**2
            * (
                wave_number_squared
                + discrete_wave_number_squared
                + 2.0*mass**2
            )
        )
        assert history["sf-energy"] == pytest.approx(
            [expected_energy], rel=5.0e-13, abs=1.0e-15
        )
    finally:
        _remove_history(history_file)
        testutils.cleanup()


def test_scalar_only_two_dimensional_pdf_output():
    """Unweighted scalar PDFs must not require an unrelated fluid state."""
    disabled_outputs = [f"output{number}/dt=0.0" for number in range(1, 10)]
    pdf_directory = Path("pdf_sf_pdf_sf_pi0")
    shutil.rmtree(pdf_directory, ignore_errors=True)
    try:
        assert testutils.run(
            _INPUT_FILE,
            [
                "job/basename=scalar_field_pdf",
                "time/nlim=0",
                "output10/dt=1.0",
                *disabled_outputs,
            ],
        )
        bins = pdf_directory / "scalar_field_pdf.bins.pdf"
        values = pdf_directory / "scalar_field_pdf.00000.pdf"
        assert bins.is_file() and bins.stat().st_size > 0
        assert values.is_file() and values.stat().st_size > 0
    finally:
        shutil.rmtree(pdf_directory, ignore_errors=True)
        testutils.cleanup()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (
            [
                "problem/test_case=real_free",
                "scalar_field/field_type=real",
                "output10/variable_2=sf_phi1",
            ],
            "requires a complex scalar field",
        ),
        (
            ["output10/variable=sf"],
            "single variable",
        ),
        (
            ["output1/variable=tmunu_E", "output1/dt=1.0"],
            "no Tmunu object",
        ),
    ),
    ids=("pdf-secondary-imaginary", "pdf-scalar-group", "missing-tmunu"),
)
def test_invalid_scalar_output_combinations(arguments, expected):
    """Invalid scalar/Tmunu output requests must fail before dereferencing storage."""
    disabled_outputs = [f"output{number}/dt=0.0" for number in range(1, 10)]
    log_path = Path(testutils.LOG_FILE_PATH)
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    try:
        with pytest.raises(RuntimeError):
            testutils.run(
                _INPUT_FILE,
                [
                    "job/basename=scalar_field_invalid_output",
                    "time/nlim=0",
                    "output10/dt=1.0",
                    *disabled_outputs,
                    *arguments,
                ],
            )
        with log_path.open("rb") as log_file:
            log_file.seek(log_offset)
            log_text = log_file.read().decode("utf-8", errors="replace")
        assert expected in log_text
    finally:
        testutils.cleanup()
