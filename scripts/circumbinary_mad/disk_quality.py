"""Estimate vertical MRI resolution and density thickness on matched physical slices."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
from bin_convert import read_binary

RADII = (100, 128, 180, 240, 300, 400)


def weighted_median(x, w):
    order = np.argsort(x)
    x, w = x[order], w[order]
    return float(x[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def measure(midplane, meridional):
    snapshots = [read_binary(str(f)) for f in (midplane, meridional)]
    if snapshots[0]["cycle"] != snapshots[1]["cycle"]:
        raise ValueError("Disk slices are from different accepted steps")
    bins = {r: {"q": [], "w": [], "m": 0.0, "mz2": 0.0} for r in RADII}
    gamma = 13 / 9
    for plane, h in enumerate(snapshots):
        b = h["mb_data"]
        for m, (xl, xu, yl, yu, zl, zu) in enumerate(h["mb_geometry"]):
            rho = b["dens"][m].astype(float)
            nz, ny, nx = rho.shape
            dx = (xu - xl) / nx
            x = xl + (np.arange(nx) + 0.5) * dx
            if plane == 0:
                dy = (yu - yl) / ny
                y = yl + (np.arange(ny) + 0.5) * dy
                R = np.hypot(x[None, :], y[:, None])
                rho = rho[0]
                p = b["press"][m][0]
                bz = b["bcc3"][m][0]
                # The physical cells are cubic even though the blocks are slabs.
                va = np.abs(bz) / np.sqrt(rho + gamma / (gamma - 1) * p + bz**2)
                q = 2 * np.pi * va * R**1.5 / dx
                weight = rho * dx * dy
            else:
                dz = (zu - zl) / nz
                z = zl + (np.arange(nz) + 0.5) * dz
                rho = rho[:, 0, :]
                R = np.broadcast_to(np.abs(x), rho.shape)
                height = np.broadcast_to(z[:, None], rho.shape)
                weight = rho * dx * dz
            for r, result in bins.items():
                inside = (R >= 0.95 * r) & (R < 1.05 * r)
                if plane:
                    inside &= np.abs(height) < 0.5 * r
                if not np.any(inside):
                    continue
                w = weight[inside]
                if np.any(~np.isfinite(w)) or np.any(w <= 0):
                    raise ValueError("Invalid evolved density in disk-resolution sample")
                if plane == 0:
                    if not np.isfinite(q[inside]).all():
                        raise ValueError("Invalid field or pressure in disk-resolution sample")
                    result["q"].append(q[inside])
                    result["w"].append(w)
                else:
                    result["m"] += w.sum()
                    result["mz2"] += (w * height[inside] ** 2).sum()
    out = {}
    for r, b in bins.items():
        if not b["q"] or b["m"] <= 0:
            raise ValueError(f"No valid disk slice at R={r}")
        q, w = (np.concatenate(b[k]) for k in ("q", "w"))
        out[r] = {
            "mass_weighted_median_Qz": weighted_median(q, w),
            "mass_fraction_Qz_below_6": float(w[q < 6].sum() / w.sum()),
            "meridional_H_over_R": math.sqrt(b["mz2"] / b["m"]) / r,
        }
    return {"time": snapshots[0]["time"], "radii": out}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("midplane", type=Path)
    p.add_argument("meridional", type=Path)
    a = p.parse_args()
    print(json.dumps(measure(a.midplane, a.meridional), indent=2, allow_nan=False))
