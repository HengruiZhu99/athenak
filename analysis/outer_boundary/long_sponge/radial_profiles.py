#!/usr/bin/env python3
"""Active-cell regional profiles for uniform evolved Minkowski residual controls.

Run separately from the all-payload/ghost SPD validator. No metric is repaired.
Saved maxima are state diagnostics, not an initial-injection locator. The
returned core amplitude is not by itself a reflection coefficient.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "gpu"))
from check_minkowski_checkpoint import supported, boolean, checkpoint_header, cohort


def extract(run, ranks, cycle, start_radius=512., end_radius=1792., bin_width=64.):
    first, records = cohort(run, ranks, cycle)
    q = records[0]
    raw = first.read_bytes()
    end = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
    params = {}
    for line in raw[:end].decode().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<"):
            section = line[1:-1]
            params[section] = {}
        elif "=" in line:
            k, v = line.split("=", 1)
            params[section][k.strip()] = v.strip()
    supported(params)
    assert all(float(params["problem"].get(f"bh_center_x{a}", 0)) == 0 for a in (1, 2, 3)), "Profile requires a centered background/sponge"
    sponge_enabled = boolean(params["problem"], "outer_sponge_enabled")
    if sponge_enabled:
        assert params["problem"].get("outer_sponge_geometry") == "radial"
        assert float(params["problem"]["outer_sponge_start_radius"]) == start_radius
        assert float(params["problem"]["outer_sponge_ramp_width"]) == end_radius-start_radius
    ng, nx, ny, nz = struct.unpack_from("<19i", raw, end + 8 + 72 + 76)[:4]
    loc_start = end + 8 + 72 + 2 * 76 + 20
    locs = [struct.unpack_from("<4i", raw, loc_start + 16*i) for i in range(q["total"])]
    assert all(loc[3] == q["level"] for loc in locs)
    mesh = params["mesh"]
    lo = np.array([float(mesh[f"x{a}min"]) for a in (1, 2, 3)])
    hi = np.array([float(mesh[f"x{a}max"]) for a in (1, 2, 3)])
    ns = np.array([int(mesh[f"nx{a}"]) for a in (1, 2, 3)])
    dx = (hi-lo)/ns
    assert np.all(lo == -hi), "Centered cube/profile only"
    assert np.all(dx == dx[0]), "Isotropic uniform grid only"
    bins = np.arange(0, np.linalg.norm(hi) + bin_width, bin_width)
    sums = {n: np.zeros(len(bins)-1) for n in ("proper_volume", "Theta2", "Theta_signed", "alpha2", "beta2", "chi2", "metric2")}
    radial_max = np.zeros(len(bins)-1)
    region_names = ("protected_core", "smooth_ramp", "outer_plateau", "within_256M_of_physical_face", "whole_domain")
    regions = {n: {"cells": 0, "proper_volume": 0., "Theta2": 0., "Theta_max": 0., "alpha_res_max": 0., "beta_res_max": 0., "chi_res_max": 0., "conformal_metric_res_max": 0.} for n in region_names}
    peaks = {}
    gid = 0
    for rank, rec in enumerate(records):
        for values in rec["state"]:
            u = np.asarray(values).reshape(25, nz+2*ng, ny+2*ng, nx+2*ng)[:, ng:ng+nz, ng:ng+ny, ng:ng+nx]
            assert np.isfinite(u).all()
            loc = locs[gid]
            axes = [lo[a] + (loc[a]*n + np.arange(n)+.5)*dx[a] for a, n in enumerate((nx, ny, nz))]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
            r = np.sqrt(x*x + y*y + z*z)
            dist = np.minimum.reduce([x-lo[0], hi[0]-x, y-lo[1], hi[1]-y, z-lo[2], hi[2]-z])
            xx, xy, xz, yy, yz, zz = 1+u[1], u[2], u[3], 1+u[4], u[5], 1+u[6]
            detg = xx*yy*zz + 2*xy*xz*yz - xx*yz*yz - yy*xz*xz - zz*xy*xy
            chi = 1+u[0]
            assert np.all(chi > 0) and np.all(detg > 0)
            volume = np.prod(dx) * np.sqrt(detg) / chi**1.5
            theta, alpha, beta = u[17], u[18], np.sqrt(np.sum(u[19:22]**2, axis=0))
            metric_res = np.sqrt(u[1]**2+u[4]**2+u[6]**2+2*(u[2]**2+u[3]**2+u[5]**2))
            curvature_res = np.sqrt(u[8]**2+u[11]**2+u[13]**2+2*(u[9]**2+u[10]**2+u[12]**2))
            index = np.searchsorted(bins, r.ravel(), side="right")-1
            for key, vals in (("proper_volume", volume), ("Theta2", volume*theta**2), ("Theta_signed", volume*theta), ("alpha2", volume*alpha**2), ("beta2", volume*beta**2), ("chi2", volume*u[0]**2), ("metric2", volume*metric_res**2)):
                sums[key] += np.bincount(index, weights=vals.ravel(), minlength=len(bins)-1)
            np.maximum.at(radial_max, index, abs(theta).ravel())
            masks = (r <= start_radius, (r > start_radius) & (r < end_radius), r >= end_radius, dist <= 256., np.ones(r.shape, bool))
            for name, mask in zip(region_names, masks):
                item = regions[name]
                item["cells"] += int(mask.sum())
                item["proper_volume"] += float(volume[mask].sum())
                item["Theta2"] += float((volume*theta**2)[mask].sum())
                if mask.any():
                    for key, vals in (("Theta_max", theta), ("alpha_res_max", alpha), ("beta_res_max", beta), ("chi_res_max", u[0]), ("conformal_metric_res_max", metric_res)):
                        item[key] = max(item[key], float(abs(vals[mask]).max()))
            for name, vals in (("Theta", theta), ("alpha_res", alpha), ("beta_res", beta), ("chi_res", u[0]), ("conformal_metric_res", metric_res), ("A_res", curvature_res), ("Khat", u[7]), ("Gamma_res", np.sqrt(np.sum(u[14:17]**2, axis=0)))):
                k, j, i = np.unravel_index(np.argmax(abs(vals)), vals.shape)
                value = float(vals[k, j, i])
                if name not in peaks or abs(value) > abs(peaks[name]["value"]):
                    peaks[name] = {"value": value, "rank": rank, "gid": gid, "relative_level": 0, "xyz": [float(x[k,j,i]), float(y[k,j,i]), float(z[k,j,i])], "radius": float(r[k,j,i]), "distance_to_nearest_face": float(dist[k,j,i]), "in_sponge": bool(sponge_enabled and r[k,j,i] > start_radius), "active_cell": True, "zero_tie_location_arbitrary": value == 0}
            gid += 1
    assert gid == q["total"]
    total = regions["whole_domain"]["Theta2"]
    for item in regions.values():
        item["Theta_RMS"] = float(np.sqrt(item["Theta2"]/item["proper_volume"])) if item["proper_volume"] else None
        item["fraction_Theta2"] = item["Theta2"]/total if total else None
    def means(key, square_root=False):
        arr = np.divide(sums[key], sums["proper_volume"], out=np.zeros_like(sums[key]), where=sums["proper_volume"] > 0)
        if square_root:
            arr = np.sqrt(arr)
        return [float(v) if vol else None for v, vol in zip(arr, sums["proper_volume"])]
    return {"time_code": q["time"], "cycle": q["cycle"], "ranks": ranks, "blocks": q["total"], "dx": dx.tolist(), "sponge_enabled": sponge_enabled, "profile_start_radius": start_radius, "profile_end_radius": end_radius, "regions": regions, "active_peaks": peaks, "radial_bins": {"edges": bins.tolist(), "proper_volume": sums["proper_volume"].tolist(), "Theta_RMS": means("Theta2", True), "Theta_mean_signed": means("Theta_signed"), "Theta_max": radial_max.tolist(), "alpha_res_RMS": means("alpha2", True), "beta_res_RMS": means("beta2", True), "chi_res_RMS": means("chi2", True), "conformal_metric_res_RMS": means("metric2", True)}, "scope": "Active cells only. First three regions partition the domain; physical-face band overlaps them. For a disabled sponge these are matched geometric zones, not a real damping layer. Beta/Gamma amplitudes use coordinate-Euclidean vector norms; metric/A amplitudes use coordinate-Frobenius norms, not physical-metric norms or maximum components. Gamma_res is evolved Gamma residual, not Gamma-minus-metric constraint. Zero-field argmax locations are arbitrary; noise-scale peaks cannot locate a resolved mode. No origin-time or reflection-coefficient inference."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--ranks", required=True, type=int)
    p.add_argument("--cycle", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cycles = [args.cycle] if args.cycle is not None else sorted({checkpoint_header(f)["cycle"] for f in (args.run/"rst/rank_00000000").glob("*.rst")})
    assert cycles
    result = [extract(args.run, args.ranks, cycle) for cycle in cycles]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"profiles": len(result), "first_time": result[0]["time_code"], "last_time": result[-1]["time_code"], "output": str(args.output)}))
