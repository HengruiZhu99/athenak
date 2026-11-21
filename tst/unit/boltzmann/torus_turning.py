"""Compare native transport snapshots with the independent IrisK straight-ray integral."""

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from evolution import read_binary, cells
from ring_reflection import reflected_equator


def profiles(directory, rays, metrics, band):
    stats, profiles = [], []
    for file in sorted((directory / "bin").glob("*.bin")):
        frame = read_binary(file)
        t = frame["time"]
        inputs = {}
        for line in frame["header"]:
            if "=" in line:
                key, val = line.split("=", 1)
                inputs[key.strip()] = val.strip()
        columns = {}
        copies = 2 if reflected_equator(frame) else 1
        height_second = 0.0
        for name in ["E", "Sx"]:
            field = "tmunu_" + name
            if field not in frame["mb_data"]:
                field = "tmunu_S_x" if name == "Sx" else field
            bins = {}
            mid = {}
            for r, z, v, dr, dz in cells(frame, field):
                if name == "E":
                    height_second += copies * 2 * np.pi * dr * dz * np.sum(r * z * z * v, dtype=np.float64)
                for k, rho in enumerate(r[0]):
                    idx = round(rho / dr - 0.5)
                    bins[idx] = bins.get(idx, 0) + copies * np.sum(v[:, k]) * dz
                    mid[idx] = mid.get(idx, 0) + copies * 0.5 * np.sum(v[abs(z[:, k]) < 0.51 * dz, k])
            columns[name] = np.array([bins[k] for k in sorted(bins)])
            columns["mid" + name] = np.array([mid[k] for k in sorted(mid)])
        e, f = columns["E"], columns["Sx"]
        r = (np.arange(len(e)) + 0.5) * dr
        # Exact cylindrical shell averages of the height-integrated profiles.
        eq, fq = np.zeros_like(r), np.zeros_like(r)
        for x, wt in zip(*roots_legendre(6)):
            rs = r + 0.5 * dr * x
            er, fr = rays.forward(rs, t)
            eq += 0.5 * wt * rs / r * er
            fq += 0.5 * wt * rs / r * fr
        em, fm = rays.forward(r, t, 0.0)
        # The entire physical range of single-ray turning radii.
        lo = np.maximum(band[0], r - dr / 2)
        hi = np.minimum(band[1], r + dr / 2)
        vol = np.pi * np.maximum(hi**2 - lo**2, 0)
        native = metrics(r, e, f, dr, band)
        exact = metrics(r, eq, fq, dr, band)
        row = dict(
            time=t,
            nx=len(e),
            nz=frame["Nx2"],
            nmu=int(inputs["nmu_half"]),
            npsi=int(inputs["npsi"]),
            **native,
        )
        energy = 2 * np.pi * dr * (r @ e)
        inside_edge = r + dr / 2 <= band[0]
        row.update(
            total_E=energy,
            E_inside_bmin=2 * np.pi * dr * (r[inside_edge] @ e[inside_edge]),
            mean_rho=((r * r) @ e) / (r @ e),
            ref_mean_rho=((r * r) @ eq) / (r @ eq),
            z_rms=np.sqrt(height_second / energy),
            ref_z_rms=rays.height_rms(t),
        )
        row.update({"ref_" + k: v for k, v in exact.items()})
        # Scale both profiles by reference inner energy, well conditioned through flux reversal.
        scale = vol @ eq
        row.update(
            density_error=(vol @ abs(e - eq)) / scale if scale > 1e-8 else np.nan,
            flux_error=(vol @ abs(f - fq)) / scale if scale > 1e-8 else np.nan,
            mid_peak=columns["midE"].max(),
            ref_mid_peak=em.max(),
            mid_FWHM=metrics(r, columns["midE"], columns["midSx"], dr, band)["FWHM"],
            ref_mid_FWHM=metrics(r, em, fm, dr, band)["FWHM"],
        )
        stats.append(row)
        profiles.extend(
            zip(np.full(len(r), t), r, e, f, eq, fq, columns["midE"], columns["midSx"], em, fm)
        )
    if not stats or stats[-1]["time"] < 6.49:
        raise ValueError("incomplete transport snapshots in " + str(directory))
    return stats, profiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["irisk", "flat_json", "output"]:
        p.add_argument(key, type=Path)
    p.add_argument("runs", nargs="+", help="case=run_directory")
    args = p.parse_args()
    sys.path.insert(0, str(args.irisk / "scripts/photon_torus"))
    from turning import Rays, metrics, turning_band

    rays = Rays(json.loads(args.flat_json.read_text()), 64)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained", sharex=True)
    at_turn = []
    panels = {}
    for spec in args.runs:
        label, folder = spec.split("=", 1)
        stats, values = profiles(Path(folder), rays, metrics, turning_band(rays.m))
        rows.extend(dict(case=label, **row) for row in stats)
        sample = min(stats, key=lambda row: abs(row["time"] - 5.5))
        at_turn.append(sample)
        values = np.array(values)
        np.savetxt(
            args.output / (label + "-profiles.csv"),
            values,
            delimiter=",",
            header="t,rho,column,flux,exact_column,exact_flux,mid_E,mid_Sr,exact_mid_E,exact_mid_Sr",
            comments="",
        )
        key = (sample["nx"], sample["nmu"], sample["npsi"])
        panels[key] = []
        for t in [5.0, 6.5]:
            nearest = min(stats, key=lambda row: abs(row["time"] - t))["time"]
            panels[key].append(values[values[:, 0] == nearest])
    ordered = sorted(panels)
    middle_grid = np.median(sorted({k[0] for k in ordered}))
    middle_sky = np.median(sorted({k[1] for k in ordered}))
    middle = min(ordered, key=lambda k: (abs(k[0] - middle_grid), abs(k[1] - middle_sky)))
    selected = ordered if len(ordered) <= 3 else sorted({ordered[0], middle, ordered[-1]})
    for k, t in enumerate([5.0, 6.5]):
        rr = np.linspace(0.15, 5, 320)
        reference = rays.forward(rr, t)
        for j, name in enumerate(
            ["Height-integrated energy density", "Height-integrated radial flux"]
        ):
            ax = axes[j, k]
            ax.plot(rr, reference[j], "k-", lw=2, label="Straight rays")
            for key in selected:
                cut = panels[key][k]
                name_grid = f"{key[0]} cells, {key[1]} × {key[2]} sky"
                ax.plot(cut[:, 1], cut[:, 2 + j], label=name_grid, lw=1.3)
            ax.set(xlim=(0.15, 5), xlabel="Cylindrical radius", ylabel=name, title=f"t = {t:g}")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    fig.savefig(args.output / "turning.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, subset, key, title in [
        (axes[0], [r for r in at_turn if r["nmu"] == 24], "nx", "Spatial refinement, 24 × 32 sky"),
        (
            axes[1],
            [r for r in at_turn if r["nx"] == 72],
            "nmu",
            "Angular refinement, 72 × 36 cells",
        ),
    ]:
        subset.sort(key=lambda r: r[key])
        xx = [r[key] for r in subset]
        for var, label in [
            ("peak", "Peak column"),
            ("FWHM", "FWHM"),
            ("inner_E", "Energy in turning band"),
        ]:
            ax.plot(
                xx, [100 * abs(r[var] / r["ref_" + var] - 1) for r in subset], "o-", label=label
            )
        ax.plot(
            xx, [100 * r["flux_error"] for r in subset], "s--", label="Radial flux profile / energy"
        )
        ax.set(
            xlabel="Radial cells" if key == "nx" else "Cosine bins per hemisphere",
            ylabel="Error at t ≈ 5.5 (%)",
            title=title,
            xticks=xx,
        )
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize="small")
    fig.savefig(args.output / "turning_refinement.png", dpi=160)
    plt.close(fig)
    with (args.output / "transport.csv").open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
