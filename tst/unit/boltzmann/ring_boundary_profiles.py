"""Reduce ring snapshots to coordinate radii and central concentration diagnostics."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vis/python"))
from bin_convert import read_binary


def profile(path, mass=1.0):
    frame = read_binary(path)
    data = frame["mb_data"]
    geometry = frame["mb_geometry"]
    adm_path = path.with_name(path.name.replace(".tmunu.", ".adm."))
    adm = read_binary(adm_path) if adm_path.exists() else None
    if adm is not None:
        np.testing.assert_array_equal(geometry, adm["mb_geometry"])
        assert frame["cycle"] == adm["cycle"]
    weights, distances, circumferential = [], [], []
    flow, peak = 0.0, 0.0
    for block, bounds in enumerate(geometry):
        e = data["tmunu_E"][block][0].astype(float)
        jz = data["tmunu_Sy"][block][0]
        jr = data["tmunu_Sx"][block][0]
        nz, nr = e.shape
        dr, dz = (bounds[1] - bounds[0]) / nr, (bounds[3] - bounds[2]) / nz
        rho, z = np.meshgrid(bounds[0] + (np.arange(nr) + 0.5) * dr,
                             bounds[2] + (np.arange(nz) + 0.5) * dz)
        rad = np.hypot(rho, z)
        vol = 2 * np.pi * rho * dr * dz
        if adm is not None:
            g = [adm["mb_data"]["adm_g" + k][block][0].astype(float)
                 for k in ("xx", "xy", "xz", "yy", "yz", "zz")]
            if g[0].shape == (nz + 1, nr + 1):
                g = [(v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]) / 4
                     for v in g]
            xx, xy, xz, yy, yz, zz = g
            # In the modified-Cartoon meridional plane the suppressed Cartesian
            # component zz carries the physical azimuthal metric divided by rho^2.
            circumferential.append((rho*np.sqrt(zz)).ravel())
            vol *= np.sqrt(xx*yy*zz + 2*xy*xz*yz - xx*yz**2 - yy*xz**2 - zz*xy**2)
        weights.append((vol * e).ravel())
        distances.append(rad.ravel())
        flow += np.sum(vol * (rho * jr + z * jz) / rad)
        peak = max(peak, np.max(e))
    weight, radius = np.concatenate(weights), np.concatenate(distances)
    total = weight.sum()
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    quantile = np.interp([0.5, 0.9], cumulative / total, radius[order])
    # The step inverse of the cell-weighted empirical CDF is independent of
    # the ordering of cells at identical radii.
    median_cell = radius[order[np.searchsorted(cumulative, .5*total)]]
    median_circ = None
    if circumferential:
        circ = np.concatenate(circumferential)
        order_circ = np.argsort(circ)
        median_circ = circ[order_circ[np.searchsorted(np.cumsum(weight[order_circ]), .5*total)]]
    return dict(time=frame["time"], time_M=frame["time"]/mass, proper_volume=int(adm is not None),
                E_integral=total, E_peak=peak, median_radius=quantile[0],
                median_circumferential_radius=median_circ, cell_median_radius=median_cell, radius90=quantile[1], fraction_r2M=weight[radius < 2*mass].sum() / total,
                mean_radial_covector=flow / total)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--mass", type=float, default=1.0)
    args = cli.parse_args()
    rows = [profile(path, args.mass) for path in sorted(args.directory.glob("bin/*.tmunu.*.bin"))]
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
