"""Static and adaptive AMR regressions for the canonical scalar field."""

import math
from pathlib import Path

import athena_read
import numpy as np
import test_suite.testutils as testutils


_STATIC_INPUT = "inputs/scalar_field_amr_static.athinput"
_ADAPTIVE_INPUT = "inputs/scalar_field_amr_adaptive.athinput"

_PHI_L1 = 5
_PI_L1 = 6
_MAX_LINF = 7
_NMB = 8
_NCREATED = 9
_NDELETED = 10
_NLEVELS = 11


def _run_and_read(input_file, basename):
    """Run one scalar AMR case and return its single diagnostic row."""
    diagnostic = Path(f"{basename}-errs.dat")
    diagnostic.unlink(missing_ok=True)
    assert testutils.run(input_file, [f"job/basename={basename}"])
    data = athena_read.error_dat(diagnostic)
    assert data.shape == (1, 12)
    return data[0]


def _remove_history(basename):
    """Remove the scalar history artifact not covered by common cleanup."""
    Path(f"{basename}.scalar.hst").unlink(missing_ok=True)


def _assert_finite_accurate(row, phi_limit, pi_limit, linf_limit):
    """Require evolved, finite, quantitatively accurate scalar data."""
    assert int(row[3]) > 0
    assert row[4] > 0.0
    for value in row[_PHI_L1:_MAX_LINF + 1]:
        assert math.isfinite(value) and value > 0.0
    assert row[_PHI_L1] < phi_limit
    assert row[_PI_L1] < pi_limit
    assert row[_MAX_LINF] < linf_limit


def test_scalar_field_static_refinement_interfaces():
    """Propagate a fourth-order wave across a fixed mixed-level mesh."""
    try:
        row = _run_and_read(_STATIC_INPUT, "scalar_field_amr_static")
        _assert_finite_accurate(row, 2.0e-6, 2.0e-5, 1.0e-4)

        root_meshblocks = (32 // 8) * (32 // 8)
        assert int(row[_NMB]) > root_meshblocks
        assert int(row[_NMB]) < 4 * root_meshblocks
        assert int(row[_NCREATED]) == 0
        assert int(row[_NDELETED]) == 0
        assert int(row[_NLEVELS]) == 2
    finally:
        testutils.cleanup()


def test_scalar_field_adaptive_regrid():
    """Demand a real regrid and accurate scalar data after redistribution."""
    basename = "scalar_field_amr_adaptive"
    _remove_history(basename)
    try:
        row = _run_and_read(_ADAPTIVE_INPUT, basename)
        _assert_finite_accurate(row, 2.0e-5, 2.0e-4, 1.0e-3)

        mesh_events = int(row[_NCREATED]) + int(row[_NDELETED])
        assert mesh_events > 0, "adaptive run did not create or delete MeshBlocks"
        assert int(row[_NMB]) > 0
        assert int(row[_NLEVELS]) == 2

        history = athena_read.hst(f"{basename}.scalar.hst")
        energy = np.asarray(history["sf-energy"])
        assert energy.shape == (2,)
        assert np.isfinite(energy).all() and (energy > 0.0).all()
        relative_energy_change = abs(energy[-1] - energy[0])/energy[0]
        assert relative_energy_change < 5.0e-3
    finally:
        _remove_history(basename)
        testutils.cleanup()
