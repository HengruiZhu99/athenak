"""Regressions for BBH gauge/diagnostic integration with upstream features."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def run_case(directory, input_name, flags, success=True):
    """Run the test-suite executable in an isolated output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    input_file = directory / "test.athinput"
    text = (ROOT / input_name).read_text()
    for flag in flags:
        block, parameter = flag.split("/", 1)
        text += f"\n<{block}>\n{parameter}\n"
    input_file.write_text(text)
    result = subprocess.run(
        [str(Path("athena").resolve()), "-i", str(input_file)],
        cwd=directory, capture_output=True, text=True, timeout=120,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result.stdout + result.stderr


def wave_flags():
    """Small flat-space evolution, with enough ghost zones for every FD order."""
    return [
        "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8", "mesh/nghost=4",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
        "mesh_refinement/refinement=none", "problem/amp=0",
        "time/nlim=3", "time/tlim=0.01", "time/cfl_number=0.01",
        "output1/file_type=tab", "output1/variable=z4c",
        "output1/dt=0.0001", "output1/slice_x2=0.5",
        "output1/slice_x3=0.5",
    ]


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("variety", ["oscillator", "pid", "relaxation", "dob", "bdob"])
def test_drift_control_with_bbh_gauge_cpu(tmp_path, order, variety):
    """Drift control must reach the split gauge kernel for every FD stencil."""
    common = wave_flags() + [
        f"z4c/spatial_order={order}", f"z4c/dc_variety={variety}",
        "z4c/co_0_type=BH", "z4c/co_0_x=0.25",
        "z4c/co_0_y=0.5", "z4c/co_0_z=0.5",
        "z4c/dc_fixed_y=0.5", "z4c/dc_fixed_z=0.5",
        "z4c/dc_damping_scale=1000000", "z4c/dc_damping_time=1",
        "z4c/slow_start_lapse=true", "z4c/telegraph_lapse=true",
        "z4c/sss_damping_amp=0.2", "z4c/roll_kappa=true",
        "z4c/target_kappa1=0.02",
    ]
    shifts = []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        run_case(directory, "tst/inputs/lwave_z4c.athinput",
                 common + [f"z4c/enable_driftcontrol={str(enabled).lower()}"])
        files = sorted((directory / "tab").glob("*.tab"))
        assert len(files) >= 2
        data = np.loadtxt(files[-1])
        assert np.isfinite(data).all()
        # tab begins with gid,i,x1, then the Z4c state; beta_x is state index 19.
        shifts.append(data[:, 3 + 19])
    assert np.max(np.abs(shifts[0])) < 1e-14
    assert np.min(shifts[1]) > 1e-10


def test_gravity_output_guard_cpu(tmp_path):
    """grav_phi without gravity must give a diagnostic, not dereference null."""
    output = run_case(
        tmp_path, "tst/inputs/lwave_z4c.athinput",
        wave_flags() + ["output1/variable=grav_phi"], success=False,
    )
    assert "gravity object not constructed" in output


def test_gravity_output_cpu(tmp_path):
    """The appended gravity output choice must remain usable with gravity."""
    run_case(tmp_path, "tst/inputs/selfgravity.athinput", [
        "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
        "time/nlim=1", "output1/file_type=tab",
        "output1/variable=grav_phi", "output1/dt=0.01",
        "output1/slice_x2=0", "output1/slice_x3=0",
    ])
    files = sorted((tmp_path / "tab").glob("*.tab"))
    assert files
    assert np.isfinite(np.loadtxt(files[-1])).all()


def test_z4c_diagnostics_cpu(tmp_path):
    """Flat-space curvature diagnostics remain available beside grav_phi."""
    run_case(tmp_path, "tst/inputs/lwave_z4c.athinput",
             wave_flags() + ["output1/variable=z4c_diag"])
    files = sorted((tmp_path / "tab").glob("*.tab"))
    assert files
    for output in files:
        data = np.loadtxt(output)
        assert data.shape[1] == 19  # block, cell index, coordinate, 16 diagnostics
        assert np.isfinite(data).all()
        assert np.max(np.abs(data[:, 3:])) < 1e-14


@pytest.mark.parametrize("per_rank", [False, True])
@pytest.mark.parametrize("variety", ["oscillator", "pid", "relaxation", "dob", "bdob"])
def test_z4c_restart_cpu(tmp_path, per_rank, variety):
    """Merged Z4c restart metadata must preserve the evolved fields."""
    flags = wave_flags() + [
        "problem/amp=1e-8", "output2/file_type=rst", "output2/dt=0.0001",
        f"output2/single_file_per_rank={str(per_rank).lower()}",
        "z4c/co_0_type=BH", "z4c/co_0_x=0.25",
        "z4c/co_0_y=0.5", "z4c/co_0_z=0.5",
        "z4c/enable_driftcontrol=true", f"z4c/dc_variety={variety}",
        "z4c/telegraph_lapse=true", "z4c/spatial_order=4",
        "output1/data_format=%24.16e",
    ]
    full, split = tmp_path / "full", tmp_path / "split"
    run_case(full, "tst/inputs/lwave_z4c.athinput", flags)
    run_case(split, "tst/inputs/lwave_z4c.athinput", flags + ["time/nlim=1"])
    restarts = sorted(split.rglob("*.rst"))
    assert restarts
    result = subprocess.run(
        [str(Path("athena").resolve()), "-r", str(restarts[-1]), "time/nlim=3"],
        cwd=split, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    final = []
    for directory in (full, split):
        output = sorted((directory / "tab").glob("*.tab"))[-1]
        data = np.loadtxt(output)
        assert np.isfinite(data).all()
        final.append(data)
    np.testing.assert_allclose(final[0], final[1], rtol=1e-12, atol=1e-14)
