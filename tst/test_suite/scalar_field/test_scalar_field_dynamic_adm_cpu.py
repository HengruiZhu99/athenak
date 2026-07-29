"""Stage-time regression for a scalar field on a prescribed dynamic ADM metric."""

import math
from pathlib import Path

import numpy as np

import athena_read
import test_suite.testutils as testutils


_INPUT_FILE = "inputs/scalar_field_oscillator.athinput"
_TLIM = 0.4


def _run(cfl):
    """Evolve the analytic homogeneous FLRW solution at one timestep scale."""
    basename = f"scalar_field_dynamic_adm_{cfl:.3f}".replace(".", "p")
    arguments = [
        f"job/basename={basename}",
        "adm/dynamic=true",
        "problem/test_case=dynamic_flrw",
        "problem/amplitude=0.3",
        "problem/initial_pi=0.2",
        "problem/expansion_rate=0.6",
        "scalar_field/field_type=real",
        "scalar_field/potential=free",
        "scalar_field/mass=0.0",
        "scalar_field/lambda=0.0",
        f"time/cfl_number={cfl}",
        f"time/tlim={_TLIM}",
    ]
    assert testutils.run(_INPUT_FILE, arguments)
    row = athena_read.error_dat(Path(f"{basename}-errs.dat"))[0]
    assert row[2] == _TLIM
    return math.hypot(row[3], row[4])


def test_dynamic_adm_uses_rk_stage_time():
    """Prescribed ADM data must be refreshed at every RK RHS abscissa."""
    try:
        errors = np.asarray([_run(cfl) for cfl in (0.4, 0.2, 0.1)])
        assert np.all(np.isfinite(errors))
        assert np.all(errors > 0.0)

        orders = np.log2(errors[:-1]/errors[1:])
        assert np.min(orders) > 3.7
        assert errors[-1] < 2.0e-8
    finally:
        testutils.cleanup()
