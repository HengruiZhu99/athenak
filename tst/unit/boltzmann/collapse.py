"""Run the coupled trial and retain its histories and accepted horizon candidates.

Example: python collapse.py build/src/athena results/base --ranks 4
Use --radius 10 for a launch-radius test with matched discrete photon E and Jz.
Use --zones-per-unit 4 or --nmu 6 --npsi 24 for independent refinements.
"""

import argparse
from pathlib import Path

import numpy as np

from evolution import run


def domain(radius, zones):
    return 16 * int(np.ceil((radius + 10) * zones / 16)) / zones


def charges(radius, zones, nm, ns, tilt):
    extent = domain(radius, zones)
    edges = np.arange(int(extent * zones)) + 0.5
    r, z = edges / zones, edges / zones - extent / 2
    mu = (np.arange(nm)[:, None] + 0.5) / nm
    psi = (np.arange(ns) + 0.5) * 2 * np.pi / ns
    alignment = tilt * mu - np.sqrt((1 - tilt**2) * (1 - mu**2)) * np.cos(psi)
    intensity = np.exp((alignment - 1) / 0.3**2)
    weight = (2 * np.pi) ** 2 / (ns * nm * zones**2) * np.exp(-z * z / 2).sum()
    shell = r * np.exp(-((r - radius) ** 2) / 2)
    return weight * np.array(
        [shell.sum() * intensity.sum(), (r * shell).sum() * (mu * intensity).sum()]
    )


def matched_data(radius, zones, nm, ns):
    target = 0.045 * charges(6, 2, 3, 12, 0.25)
    lo, hi = -0.99, 0.99
    for _ in range(60):
        tilt = (lo + hi) / 2
        energy, angular = charges(radius, zones, nm, ns, tilt)
        if angular / energy < target[1] / target[0]:
            lo = tilt
        else:
            hi = tilt
    if abs(angular / energy - target[1] / target[0]) > 1.0e-9:
        raise ValueError("angular mesh cannot represent requested specific angular momentum")
    return tilt, target[0] / energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("athena", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--zones-per-unit", type=int, default=2)
    parser.add_argument("--nmu", type=int, default=3)
    parser.add_argument("--npsi", type=int, default=12)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--tlim", type=float)
    opt = parser.parse_args()
    extent = domain(opt.radius, opt.zones_per_unit)
    zones = int(extent * opt.zones_per_unit)
    tilt, amp = matched_data(opt.radius, opt.zones_per_unit, opt.nmu, opt.npsi)
    changes = {
        "mesh": {
            "nx1": zones,
            "nx2": zones,
            "x1max": extent,
            "x2min": -extent / 2,
            "x2max": extent / 2,
        },
        "meshblock": {"nx1": 16, "nx2": 16},
        "boltzmann": {"npsi": opt.npsi, "nmu_half": opt.nmu},
        "problem": {"ring_radius": opt.radius, "ring_mu": tilt, "intensity": amp},
        "time": {"tlim": opt.radius + 10 if opt.tlim is None else opt.tlim},
        "fastflow": {
            "find_interval_0": 30,
            "initial_radius_0": 2,
            "lmax": opt.lmax,
            "ntheta": max(12, 2 * opt.lmax + 2),
        },
    }
    out = opt.output.resolve()
    matter, log = run(str(opt.athena.resolve()), opt.ranks, out, changes)
    (out / "stdout.log").write_text(log)
    constraints = np.loadtxt(out / "ring.z4c.user.hst", ndmin=2)
    print(
        "initial E, Jz",
        matter[0, 2:4],
        "initial/final H-norm2, M-norm2",
        constraints[[0, -1], 3:5],
    )
    print("final exterior Jz, escaped Jz", matter[-1, [7, 5]])
    candidates = list(out.glob("*.cartoon_m0_horizon_0.txt"))
    accepted = []
    for path in candidates:
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                continue
            row = line.split()
            if row[3] == "1":
                area, spin = float(row[7]), float(row[10])
                accepted.append((row[1], row[2], area, spin, 1 - 8 * np.pi * abs(spin) / area))
    print(
        "last accepted candidate (t, branch, A, J_H, gap)", accepted[-1] if accepted else "none"
    )
