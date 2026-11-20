"""Run and independently measure the native photon_torus handoff matrix."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[3]


def measure(folder, meta):
    data = np.loadtxt(folder / "initial.vertices.0")
    # Merge block-boundary copies by lattice index, including sub-ulp differences.
    h = data[1, 0] - data[0, 0]
    i, j = np.rint((data[:, :2] - data[:, :2].min(axis=0)) / h).astype(int).T
    rho = data[:, 0].min() + np.arange(i.max() + 1) * h
    z = data[:, 1].min() + np.arange(j.max() + 1) * h
    mesh = np.full((len(rho), len(z), data.shape[1] - 2), np.nan)
    mesh[i, j] = data[:, 2:]
    assert np.isfinite(mesh).all()
    psi = mesh[:, :, 0] ** 0.25
    h = rho[1] - rho[0]
    dv = 2 * np.pi * rho[:, None] * h**2 * psi**6
    dv[[-1]] *= 0.5
    dv[:, [0, -1]] *= 0.5
    row = {}
    row["psi_at_6_0p8"] = float(RectBivariateSpline(rho, z, psi).ev(6.0, 0.8))
    row["A_rp_at_6_0p8"] = float(RectBivariateSpline(rho, z, psi**2 * mesh[:, :, 3]).ev(6.0, 0.8))
    for n, key in enumerate(["H", "Mr", "Mz", "Mp"]):
        val = mesh[:, :, 7 + n]
        row[key + "_L2"] = np.sqrt(np.sum(dv * val**2))
        row[key + "_max"] = abs(val).max()
    q, w = roots_legendre(128)
    for radius in [8, 9, 10, 11]:
        s = np.sqrt(1 - q * q)

        def extract(values, **deriv):
            return RectBivariateSpline(rho, z, values).ev(radius * s, radius * q, **deriv)

        gradient = s * extract(psi, dx=1) + q * extract(psi, dy=1)
        row["M" + str(radius)] = -(radius**2) * (w @ gradient)
        # K_ij phi^i n^j dS = rho psi^2 K_pn r^2 dOmega.
        row["J" + str(radius)] = (
            radius**3
            / 4
            * (
                w
                @ (
                    s
                    * extract(psi) ** 2
                    * (s * extract(mesh[:, :, 3]) + q * extract(mesh[:, :, 5]))
                )
            )
        )
    cell = np.loadtxt(folder / "initial.cells.0")
    r, zc, e, jr, jz, jp, bins, effective = cell[:, :8].T
    factor = RectBivariateSpline(rho, z, psi).ev(r, zc) ** 2
    vol = 2 * np.pi * r * h * h * factor**3
    ep = vol * e
    pop = e > 0
    row.update(
        E=ep.sum(),
        J=np.sum(vol * r * jp),
        R0=ep @ r / ep.sum() if pop.any() else 0.0,
        min_bins=bins[pop].min() if pop.any() else 0,
        min_effective_bins=effective[pop].min() if pop.any() else 0,
        mean_nr=(vol @ (jr / factor)) / ep.sum() if pop.any() else 0,
        max_vertical_fraction=np.max(abs(jz[pop]) / (factor[pop] * e[pop])) if pop.any() else 0,
    )
    if cell.shape[1] >= 10:
        szz, spp = cell[:, 8:10].T
        row["sigma_b"] = (
            np.sqrt(max(0.0, (vol @ (r * r * spp)) / ep.sum() - (row["J"] / row["E"]) ** 2))
            if pop.any()
            else 0
        )
        row["sigma_nu"] = np.sqrt((vol @ (szz / factor**2)) / ep.sum()) if pop.any() else 0
    hist = np.loadtxt(folder / "torus.user.hst", ndmin=2)
    if cell.shape[1] >= 13:
        raw = 2 * np.pi * h * h * cell[:, 10:12].sum(axis=0)
        row.update(raw_E=raw[0], raw_J=raw[1], raw_moment_max=cell[:, 12].max())
        np.testing.assert_allclose(raw, hist[0, 2:4], rtol=1e-12, atol=1e-15)
        assert row["raw_moment_max"] < 1e-11
    row.update(
        hist_E=hist[0, 2],
        hist_J=hist[0, 3],
        end_t=hist[-1, 0],
        end_E=hist[-1, 2],
        end_J=hist[-1, 3],
        min_U=hist[:, 8].min(),
    )
    if pop.any():
        np.testing.assert_allclose(hist[0, 3] / hist[0, 2], meta["mean_b"], rtol=5e-12)
        assert np.all(jr[pop] < 0) and row["max_vertical_fraction"] < 1e-12
    return row


def run(exe, family, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, case, n, mu, sectors, evolve in [
        ("space48", "strong", 48, 24, 32, True),
        ("space96", "strong", 96, 24, 32, True),
        ("space144", "strong", 144, 24, 32, False),
        ("angle_coarse", "strong", 96, 12, 16, False),
        ("angle_fine", "strong", 96, 48, 64, False),
        ("polygon", "strong", 96, 24, 64, False),
        ("vertical", "strong", 96, 48, 128, False),
        ("elliptic", "medium", 96, 24, 32, False),
        ("negative", "negative", 96, 24, 32, True),
        ("weak", "weak", 96, 24, 32, True),
        ("narrow", "narrow", 96, 24, 32, True),
        ("vacuum", "vacuum", 48, 24, 32, False),
    ]:
        folder = out / label
        folder.mkdir(exist_ok=False)
        inp = ROOT / "inputs/boltzmann/photon_torus.athinput"
        args = [
            str(exe),
            "-i",
            str(inp),
            "problem/torus_file=" + str(family / (case + ".adm")),
            f"mesh/nx1={n}",
            f"mesh/nx2={n}",
            f"boltzmann/nmu_half={mu}",
            f"boltzmann/npsi={sectors}",
        ]
        if case == "negative":
            args.append("boltzmann/sign=retrograde")
        if not evolve:
            args.append("time/nlim=0")
        (folder / "command.json").write_text(json.dumps(args, indent=2) + "\n")
        with (folder / "run.log").open("w") as log:
            subprocess.run(args, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = measure(folder, json.loads((family / (case + ".adm.json")).read_text()))
        rows.append(dict(case=label, grid=n, nmu=mu, npsi=sectors, **row))
        print(label, row["H_L2"], row["J10"], row["end_t"], flush=True)
    with (out / "native.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for key in ["H_L2", "Mr_L2", "Mz_L2", "Mp_L2"]:
        assert rows[2][key] < rows[0][key], (key, rows[0][key], rows[2][key])
    assert rows[0]["end_t"] >= 0.03 and rows[0]["min_U"] >= 0
    assert rows[-1]["H_max"] < 1e-10 and rows[-1]["J"] == 0
    plot_native(rows, out)


def plot_native(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, quantity in zip(axes, ["L2", "max"]):
        for key in ["H", "Mr", "Mz", "Mp"]:
            ax.loglog(
                [r["grid"] for r in rows[:3]],
                [r[f"{key}_{quantity}"] for r in rows[:3]],
                "o-",
                label=key,
            )
        ax.set(
            xlabel="Cells per spatial direction",
            ylabel="Proper-volume L2" if quantity == "L2" else "Maximum absolute residual",
        )
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.savefig(out / "native_constraints.png", dpi=160)
    plt.close(fig)


def compare_backends(cpu, gpu):
    differences = {}
    for suffix, count in [
        ("initial.vertices.0", 13),
        ("initial.cells.0", 8),
        ("torus.user.hst", 9),
    ]:
        a = np.loadtxt(cpu / suffix)[:, :count]
        b = np.loadtxt(gpu / suffix)[:, :count]
        np.testing.assert_allclose(a, b, rtol=3e-11, atol=1e-11)
        differences[suffix] = np.max(abs(a - b), axis=0).tolist()
    return differences


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["exe", "family", "out"]:
        cli.add_argument(name, type=Path)
    cli.add_argument("--compare-gpu", type=Path, help="matched 48-cell coupled GPU run")
    p = cli.parse_args()
    run(p.exe.resolve(), p.family.resolve(), p.out.resolve())
    if p.compare_gpu:
        delta = compare_backends(p.out / "space48", p.compare_gpu)
        (p.out / "backend_differences.json").write_text(json.dumps(delta, indent=2) + "\n")
