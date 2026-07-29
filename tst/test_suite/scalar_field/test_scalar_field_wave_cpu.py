"""Convergence of a real scalar plane wave on fixed ADM."""

import math
from pathlib import Path

import athena_read
import pytest
import test_suite.testutils as testutils


_SPATIAL_ORDERS = (2, 4, 6)
_RESOLUTIONS = {2: (16, 32), 4: (32, 64), 6: (16, 32)}
_INPUT_FILE = "inputs/scalar_field_wave.athinput"
# Order four uses 32/64 because Pi is still pre-asymptotic at 16/32. The
# threshold allows modest variation while demanding clear high-order behavior.
# A measured pre-asymptotic sixth-order rate at 16/32 does not imply that RK4
# can sustain sixth-order end-to-end convergence: with dt proportional to dx,
# the asymptotic temporal ceiling remains fourth order.
_MIN_OBSERVED_ORDER = {2: 1.8, 4: 3.5, 6: 3.5}


def _basename(spatial_order):
    """Return an order-specific basename so error tables cannot mix."""
    return f"scalar_field_wave_o{spatial_order}"


def _arguments(resolution, spatial_order):
    """Set equal x-y resolution while retaining several periodic MeshBlocks."""
    return [
        f"job/basename={_basename(spatial_order)}",
        f"mesh/nx1={resolution}",
        f"mesh/nx2={resolution}",
        "mesh/nx3=1",
        f"mesh/nghost={spatial_order // 2 + 1}",
        "meshblock/nx1=8",
        "meshblock/nx2=8",
        "meshblock/nx3=1",
        f"scalar_field/spatial_order={spatial_order}",
    ]


@pytest.mark.parametrize("spatial_order", _SPATIAL_ORDERS)
def test_scalar_field_plane_wave(spatial_order):
    """Check finite phi/Pi errors and the expected convergence floor."""
    error_file = Path(f"{_basename(spatial_order)}-errs.dat")
    error_file.unlink(missing_ok=True)
    try:
        resolutions = _RESOLUTIONS[spatial_order]
        for resolution in resolutions:
            assert testutils.run(_INPUT_FILE, _arguments(resolution, spatial_order))

        data = athena_read.error_dat(error_file)
        assert data.shape[0] == len(resolutions)
        assert tuple(data[:, 0].astype(int)) == resolutions
        phi_errors = data[:, 5]
        pi_errors = data[:, 6]

        for name, errors in (("phi", phi_errors), ("Pi", pi_errors)):
            assert all(math.isfinite(error) and error > 0.0 for error in errors)
            order = math.log(errors[0] / errors[1], 2.0)
            minimum_order = _MIN_OBSERVED_ORDER[spatial_order]
            assert order >= minimum_order, (
                f"order-{spatial_order} {name} observed order {order:.6g} "
                f"is below {minimum_order}; "
                f"errors were {errors[0]:.6g}, {errors[1]:.6g}"
            )
    finally:
        testutils.cleanup()
