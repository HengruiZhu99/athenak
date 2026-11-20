"""Exercise the production executable, including real regrids and restart.

Usage: python evolution.py /absolute/path/to/athena [--ranks 2] [--turn]
Uses the repository's binary reader (NumPy and h5py). --turn adds three
angular resolutions and compares a turning pulse with exact straight rays.
"""

import argparse
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vis/python"))
from bin_convert import read_binary


def run(executable, ranks, directory, changes, restart=None, binary_dt=None):
    directory.mkdir()
    template = (ROOT / "inputs/boltzmann/ring.athinput").read_text()
    for section, values in changes.items():
        block = re.search(rf"(?ms)^<{section}>\n(.*?)(?=^<|\Z)", template)
        body = block[1]
        for key, value in values.items():
            pattern = rf"(?m)^{key}\s*=.*$"
            if re.search(pattern, body):
                body = re.sub(pattern, f"{key} = {value}", body)
            else:
                body += f"{key} = {value}\n"
        template = template[: block.start(1)] + body + template[block.end(1) :]
    if binary_dt:
        template += f"\n<output3>\nfile_type = bin\nvariable = tmunu\ndt = {binary_dt}\n"
    path = directory / "input"
    path.write_text(template)
    args = ["mpiexec", "-n", str(ranks)] if ranks > 1 else []
    args += [executable, "-d", str(directory)]
    args += ["-i", str(path)] if restart is None else ["-r", str(restart), "time/nlim=6"]
    result = subprocess.run(args, cwd=directory, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:] + result.stderr[-4000:])
    return np.loadtxt(directory / "ring.user.hst", ndmin=2), result.stdout


def check_energy(rows, ranks, phase_cells):
    assert np.all(rows[:, 8] >= 0), np.min(rows[:, 8])
    energy = rows[:, 2]
    residual = energy + rows[:, 4] - rows[:, 6] - energy[0]
    # Tree reductions, one RK sum per accepted step, and summation over bins.
    depth = math.ceil(math.log2(max(2, ranks * phase_cells)))
    steps = 1 + np.sum(
        np.diff(rows[:, 0]) / np.maximum(rows[1:, 1], np.finfo(float).tiny),
        where=rows[1:, 1] > 0,
    )
    flux_scale = np.sum(abs(np.diff(rows[:, 4]))) + np.sum(abs(np.diff(rows[:, 6])))
    scale = abs(energy[0]) + flux_scale
    tolerance = (4 * depth + 2 * steps) * np.finfo(float).eps * scale
    relative = np.max(abs(residual)) / scale
    assert relative < tolerance / scale, (relative, tolerance / scale)
    return relative


def restart_amr(exe, ranks, root):
    changes = {
        "problem": {
            "exercise_deterministic_amr": "true",
            "ring_radius": 15,
            "angular_width": 0.65,
        },
        "boltzmann": {"sign": "both", "backreaction": "false"},
        "time": {"nlim": 6},
        "fastflow": {"num_horizons": 0},
        "amr_criterion0": {"method": "user"},
        "mesh_refinement": {"ncycle_check": 1, "refinement_interval": 1},
        "output1": {"dcycle": 1},
        "output2": {"dcycle": 3},
    }
    direct, log = run(exe, ranks, root / "direct", changes)
    assert re.search(r"3 MeshBlocks created, 3 deleted", log), log[-800:]
    if ranks == 2:
        assert re.search(r"[1-9]\d* communicated for load balancing", log)
    residual = check_energy(direct, ranks, 32 * 32 * 4 * 12 * 8)
    rst = root / "direct/rst/ring.00001.rst"
    resumed, _ = run(exe, 1, root / "resumed", changes, restart=rst)
    np.testing.assert_allclose(resumed[-1], direct[-1], rtol=3e-12, atol=5e-16)
    assert direct[-1, 4] > 1.0e-8 * direct[0, 2]
    # Final topology has its four original blocks. Compare every active
    # ordinate in the block records, independently of history reduction order.
    nb, angular, cells, vertices = 4, 2 * 4 * 12, 16 + 8, 17 + 8
    stride = angular * cells**2 + 25 * vertices**2

    def ordinates(directory):
        checkpoint = sorted((directory / "rst").glob("*.rst"))[-1].read_bytes()
        data = np.frombuffer(checkpoint[-nb * stride * 8 :], dtype=np.float64)
        bins = data.reshape(nb, stride)[:, : angular * cells**2]
        return bins.reshape(nb, angular, cells, cells)[:, :, 4:-4, 4:-4]

    np.testing.assert_allclose(
        ordinates(root / "resumed"), ordinates(root / "direct"), rtol=2e-13, atol=1e-17
    )
    print(f"refine, derefine, migration and restart: energy residual {residual:.3g}")


def cells(snapshot, field):
    for bounds, array in zip(snapshot["mb_geometry"], snapshot["mb_data"][field]):
        zmin, zmax = bounds[2:4]
        rmin, rmax = bounds[:2]
        nz, nr = array.shape[1:]
        dr, dz = (rmax - rmin) / nr, (zmax - zmin) / nz
        r, z = np.meshgrid(rmin + (np.arange(nr) + 0.5) * dr, zmin + (np.arange(nz) + 0.5) * dz)
        yield r, z, array[0], dr, dz


def expansion(exe, ranks, root):
    changes = {
        "problem": {"pgen_name": "boltzmann_flrw", "hubble": 0.2},
        "boltzmann": {"sign": "both"},
        "time": {"tlim": 0.5},
        "z4c": {"lapse_oplog": 0, "shift_Gamma": 0},
        "mesh_refinement": {"refinement": "none"},
        "fastflow": {"num_horizons": 0},
    }
    run(exe, ranks, root / "flrw", changes, binary_dt=0.5)
    snapshot = read_binary(sorted((root / "flrw/bin").glob("*.bin"))[-1])
    exact = 3 * 0.2**2 / (8 * np.pi) / (1 + 2 * 0.2 * snapshot["time"]) ** 2
    errors = []
    for r, z, v, _, _ in cells(snapshot, "tmunu_E"):
        mask = (r > 5) & (r < 10) & (abs(z) < 2)
        errors.extend(abs(v[mask] / exact - 1))
    err = np.max(errors)
    assert err < 1.0e-4, err
    print(f"radiation FLRW interior max relative error {err:.3g}")


def signs(exe, ranks, root):
    rows = []
    for sign, tilt in [("prograde", 0.25), ("retrograde", -0.25)]:
        changes = {
            "boltzmann": {"backreaction": "false", "fixed_metric": "true", "sign": sign},
            "problem": {"ring_mu": tilt},
            "time": {"tlim": 0.75},
            "fastflow": {"num_horizons": 0},
            "output1": {"dcycle": 4},
        }
        data, _ = run(exe, ranks, root / sign, changes)
        check_energy(data, ranks, 64 * 64 * 4 * 12)
        rows.append(data)
    parity = np.array([1, 1, 1, -1, 1, -1, 1, -1, 1])
    np.testing.assert_allclose(rows[0], rows[1] * parity, rtol=3.0e-12, atol=5.0e-16)
    print("prograde and retrograde AMR histories obey reflection symmetry")


def kerr(exe, ranks, root):
    from kerr_beam import stress

    errors = []
    for nm, ns in [(4, 12), (8, 20)]:
        changes = {
            "boltzmann": {
                "backreaction": "false",
                "fixed_metric": "true",
                "npsi": ns,
                "nmu_half": nm,
            },
            "problem": {
                "pgen_name": "boltzmann_kerr_beam",
                "M": 1,
                "chi": 0.6,
                "z_h": -12,
                "initial_gauge": "stationary",
            },
            "time": {"tlim": 9},
            "fastflow": {"num_horizons": 0},
            "mesh_refinement": {"refinement": "none"},
        }
        directory = root / f"kerr-{nm}"
        run(exe, ranks, directory, changes, binary_dt=3)
        snapshots = sorted((directory / "bin").glob("*.bin"))
        values = []
        for idx in [1, 3]:
            snapshot = read_binary(snapshots[idx])
            actual = [
                float(v[(r == 3.25) & (z == 0.25)][0])
                for field in ["tmunu_E", "tmunu_Sx"]
                for r, z, v, _, _ in cells(snapshot, field)
                if ((r == 3.25) & (z == 0.25)).any()
            ]
            exact = np.asarray(stress(3.25, 0.25, snapshot["time"]))
            values.extend(abs(actual - exact) / abs(exact))
            if nm == 8:
                assert actual[1] * exact[1] > 0, (actual, exact)
        print(
            f"Kerr {nm} x {ns} relative errors (E, radial momentum at ingress/egress): {values}"
        )
        errors.append(values)
    assert np.all(np.array(errors[1]) < errors[0]), errors


def flux_profile(snapshot):
    # Compare radial current on the sampling cylinder to the continuum distribution
    # backtraced to the initial annulus. Include the sign of ingress/egress.
    mu, weight = np.polynomial.legendre.leggauss(48)
    mu = (1 + mu[:, None]) / 2
    psi = (np.arange(192) + 0.5) * 2 * np.pi / 192
    nr = np.sqrt(1 - mu * mu) * np.cos(psi)
    nz = np.sqrt(1 - mu * mu) * np.sin(psi)
    t = snapshot["time"]
    numerator = denominator = integrated = exact_integrated = 0
    for rs, zs, values, dr, dz in cells(snapshot, "tmunu_Sx"):
        idx = np.where((rs >= 3) & (rs < 3 + dr) & (abs(zs) < 4))
        for r, z, value in zip(rs[idx], zs[idx], values[idx]):
            rstart = np.hypot(r - t * nr, t * mu)
            dot = (
                0.25 * r * mu - math.sqrt(1 - 0.25**2) * (r * nr - t * (nr * nr + mu * mu))
            ) / rstart
            intensity = 0.002 * np.exp(
                -0.5 * ((rstart - 6) ** 2 + (z - t * nz) ** 2) + (dot - 1) / 0.3**2
            )
            true = np.sum(intensity * nr * weight[:, None]) * np.pi / 192
            integrated += value * dz
            exact_integrated += true * dz
            numerator += abs(true - value) * dz
            denominator += abs(true) * dz
    return numerator, denominator, integrated, exact_integrated


def turning(exe, ranks, root):
    errors = []
    for nm, ns in [(2, 8), (4, 12), (8, 20)]:
        changes = {
            "boltzmann": {
                "backreaction": "false",
                "fixed_metric": "true",
                "npsi": ns,
                "nmu_half": nm,
            },
            "fastflow": {"num_horizons": 0},
            "time": {"tlim": 8.0},
        }
        directory = root / f"turn-{nm}"
        rows, _ = run(exe, ranks, directory, changes, binary_dt=0.5)
        balance = check_energy(rows, ranks, 64 * 64 * nm * ns)
        gaps = abs(rows[-1, 3] + rows[-1, 5] - rows[0, 3]) / abs(rows[0, 3])
        samples = sorted((directory / "bin").glob("*.bin"))[1:]
        profiles = [read_binary(p) for p in samples]
        times = np.array([p["time"] for p in profiles])
        sums = np.array([flux_profile(p) for p in profiles])
        err = sums[:, 0].sum() / sums[:, 1].sum()

        # Arrival time of the ingoing pulse: centroid of inward radial flux.
        def arrival(signal):
            weight = np.maximum(-signal, 0)

            def integral(y):
                return np.sum(np.diff(times) * (y[1:] + y[:-1]) / 2)

            return integral(times * weight) / integral(weight)

        timing = abs(arrival(sums[:, 2]) - arrival(sums[:, 3]))
        print(
            f"{nm} x {ns}: Jz residual {gaps:.5g}, radial flux profile L1 {err:.5g}, arrival error {timing:.5g}, energy {balance:.3g}"
        )
        errors.append((gaps, err, timing))
    assert np.all(np.diff(np.array(errors), axis=0) < 0), errors


def joint_refinement(exe, ranks, root):
    residuals = []
    for zones, nm, ns in [(16, 2, 8), (32, 4, 12), (64, 8, 20)]:
        changes = {
            "mesh": {"nx1": zones, "nx2": zones},
            "mesh_refinement": {"refinement": "none"},
            "boltzmann": {
                "backreaction": "false",
                "fixed_metric": "true",
                "nmu_half": nm,
                "npsi": ns,
            },
            "fastflow": {"num_horizons": 0},
            "time": {"tlim": 4},
        }
        processes = min(ranks, (zones // 16) ** 2)
        rows, _ = run(exe, processes, root / f"joint-{zones}", changes)
        balance = check_energy(rows, processes, zones**2 * nm * ns)
        residual = abs(rows[-1, 3] + rows[-1, 5] - rows[0, 3]) / abs(rows[0, 3])
        residuals.append(residual)
        print(
            f"joint {zones} x {zones}, {nm} x {ns}: Jz residual {residual:.5g}, energy {balance:.3g}"
        )
    assert np.all(np.diff(residuals) < 0), residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("athena", type=Path)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--kerr", action="store_true")
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="athenak-boltzmann-") as tmp:
        root = Path(tmp)
        restart_amr(str(args.athena.resolve()), args.ranks, root)
        expansion(str(args.athena.resolve()), args.ranks, root)
        signs(str(args.athena.resolve()), args.ranks, root)
        if args.joint:
            joint_refinement(str(args.athena.resolve()), args.ranks, root)
        if args.kerr:
            kerr(str(args.athena.resolve()), args.ranks, root)
        if args.turn:
            turning(str(args.athena.resolve()), args.ranks, root)
