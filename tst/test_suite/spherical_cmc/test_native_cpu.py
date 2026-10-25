"""Native constrained spherical module: fields, convergence, restart and guards."""
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXE = Path(os.environ.get("ATHENA_CMC_EXE", ROOT / "build-cmc/src/athena")).resolve()
INPUT = ROOT / "inputs/spherical_cmc/minkowski.athinput"


def run(directory, *overrides, checkpoint=None, success=True, input_file=INPUT):
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(EXE), "-r" if checkpoint else "-i", str(checkpoint or input_file)]
    result = subprocess.run(command + list(overrides), cwd=directory, text=True,
                            capture_output=True, timeout=120)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def fields(directory, first=False):
    paths = sorted((directory / "tab").glob("*.tab"))
    path = paths[0] if first else paths[-1]
    names = path.read_text().splitlines()[1].lstrip("#").split()
    # The existing tab header lists inactive dimensions; 1D rows omit them.
    names = [v for v in names if v not in ("j", "x2v", "k", "x3v")]
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape[1] == len(names)
    assert np.isfinite(data).all()
    return dict(zip(names, data.T))


def history(directory):
    result = np.atleast_2d(np.loadtxt(directory / "cmc.spherical_cmc.hst"))
    assert np.isfinite(result).all()
    return result


def test_reference(tmp_path):
    run(tmp_path, "mesh/nx1=32", "meshblock/nx1=32", "time/tlim=0.1",
        "spherical_cmc/metric_pulse=0")
    q = fields(tmp_path)
    np.testing.assert_array_equal(q["scmc_a"], 1)
    np.testing.assert_array_equal(q["scmc_chi"], 1)
    np.testing.assert_allclose(q["scmc_alpha"], (1 + q["x1v"]**2) / 2, atol=1e-14)
    np.testing.assert_allclose(q["scmc_beta"], -q["x1v"], atol=1e-14)
    np.testing.assert_array_equal(q["scmc_Khat"], -3)
    np.testing.assert_array_equal(q["scmc_Theta"], 0)
    assert abs(q["scmc_H"]).max() < 1e-10
    assert abs(history(tmp_path)[:, 7]).max() == 0  # exact-scri null residual


def test_dirac_decay_and_constraint_convergence(tmp_path):
    errors = []
    for n in (32, 64):
        d = tmp_path / str(n)
        run(d, f"mesh/nx1={n}", f"meshblock/nx1={n}", "time/tlim=0.5")
        q0, q = fields(d, True), fields(d)
        # The exact continuum condition is V(t,r)=exp(-eta*t) V(0,r).
        hist = history(d)
        t = hist[-1, 0]
        decay = abs(q["scmc_Gamma"] - np.exp(-t)*q0["scmc_Gamma"]).max()
        assert decay < 0.03*abs(q0["scmc_Gamma"]).max()
        assert hist[-1, 2] < hist[0, 2]
        assert abs(hist[:, 7]).max() == 0
        errors.append(abs(q["scmc_H"]).max())
    assert errors[0]/errors[1] > 2.5, errors


def test_schwarzschild_excision(tmp_path):
    errors = []
    for n in (64, 128):
        d = tmp_path / str(n)
        run(d, f"mesh/nx1={n}", f"meshblock/nx1={n}", "mesh/x1min=0.05",
            "spherical_cmc/mass=0.5", "spherical_cmc/dirac_damping=0",
            "spherical_cmc/metric_pulse=0", "time/tlim=0.05")
        q = fields(d)
        np.testing.assert_array_equal(q["scmc_a"], 1)
        assert abs(q["scmc_Arr"]).max() > 0
        assert abs(history(d)[:, 7]).max() == 0
        errors.append(abs(q["scmc_H"]).max())
    assert errors[0]/errors[1] > 2.5, errors


@pytest.mark.parametrize("black_hole", [False, True])
@pytest.mark.parametrize("per_rank", [False, True])
def test_restart(tmp_path, black_hole, per_rank):
    extra = ["mesh/x1min=0.05", "spherical_cmc/mass=0.5"] if black_hole else []
    common = ["mesh/nx1=32", "meshblock/nx1=32", "time/tlim=1",
              f"output3/single_file_per_rank={str(per_rank).lower()}"] + extra
    run(tmp_path / "full", *common, "time/nlim=10")
    run(tmp_path / "split", *common, "time/nlim=5")
    checkpoint = sorted((tmp_path / "split/rst").rglob("*.rst"))[-1]
    run(tmp_path / "resume", "time/nlim=10", checkpoint=checkpoint)
    full, resumed = fields(tmp_path / "full"), fields(tmp_path / "resume")
    for name in full:
        np.testing.assert_array_equal(full[name], resumed[name], err_msg=name)


@pytest.mark.parametrize("overrides, message", [
    (["time/integrator=rk4"], "rk3"),
    (["meshblock/nx1=32"], "one uniform"),
    (["spherical_cmc/metric_pulse=nan"], "metric_pulse"),
    (["spherical_cmc/dirac_damping=-1"], "eta"),
    (["spherical_cmc/mass=0.5"], "rmin"),
    (["mesh/x1min=0.8", "spherical_cmc/mass=0.5"], "inside the horizon"),
    (["hydro/eos=ideal"], "exclusive vacuum"),
    (["output1/ghost_zones=true"], "interior cells"),
    (["spherical_cmc/state_version=2"], "state version"),
])
def test_guards(tmp_path, overrides, message):
    source = INPUT
    if overrides == ["hydro/eos=ideal"]:
        source = tmp_path / "invalid.athinput"
        source.write_text(INPUT.read_text() + "\n<hydro>\neos=ideal\n")
        overrides = []
    result = run(tmp_path, *overrides, success=False, input_file=source)
    assert message in result.stdout + result.stderr


def test_black_hole_gauge_evolution(tmp_path):
    errors = []
    for n in (64, 128):
        d = tmp_path / str(n)
        run(d, f"mesh/nx1={n}", f"meshblock/nx1={n}", "mesh/x1min=0.05",
            "spherical_cmc/mass=0.5", "time/tlim=2")
        q0, q = fields(d, True), fields(d)
        assert abs(history(d)[:, 7]).max() == 0
        errors.append(abs(q["scmc_H"]).max())
        # Test the independent tensor constraints, not just a stationary metric.
        assert abs(q["scmc_a"] - q0["scmc_a"]).max() > 1e-3
        assert abs(q["scmc_Gamma"]).max() < 0.2*abs(q0["scmc_Gamma"]).max()
    assert errors[0]/errors[1] > 3, errors


def test_binary_output(tmp_path):
    pytest.importorskip("h5py")  # Dependency of AthenaK's existing binary reader.
    spec = importlib.util.spec_from_file_location(
        "cmc_bin_convert", ROOT / "vis/python/bin_convert.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    run(tmp_path / "tab", "time/tlim=0.01")
    run(tmp_path / "bin", "time/tlim=0.01", "output1/file_type=bin")
    expected = fields(tmp_path / "tab")
    path = sorted((tmp_path / "bin/bin").glob("*.bin"))[-1]
    actual = converter.read_binary(str(path))
    assert actual["nvars"] == 13
    for name in actual["var_names"]:
        np.testing.assert_allclose(np.asarray(actual["mb_data"][name]).ravel(),
                                   expected[name], rtol=2e-7, atol=1e-12)
