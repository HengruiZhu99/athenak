"""Measure the initial midplane MRI wavelength on the actual finest-zone hierarchy."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from inputs import read_header, write_blocks

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
from bin_convert import read_binary


def midplane_probe(exe, deck, work):
    # Recover the equilibrium profiles independently of the global mesh. Include
    # the inner edge and the pressure maximum in the field normalization.
    b = read_header(deck)
    for axis, cells, lo, hi, block in ((1, 512, 32, 544, 32), (2, 8, -4, 4, 8), (3, 16, -8, 8, 8)):
        b["mesh"].update({f"nx{axis}": cells, f"x{axis}min": lo, f"x{axis}max": hi})
        b["meshblock"][f"nx{axis}"] = block
    b["problem"].update(
        flux_diagnostics="false",
        flux_horizon1="false",
        flux_horizon2="false",
        flux_monitor_ct="false",
    )
    path = work / "probe.athinput"
    path.write_text(write_blocks(b))
    with (work / "probe.log").open("w") as log:
        subprocess.run(
            [exe, "-i", path, "-d", work], stdout=log, stderr=subprocess.STDOUT, check=True
        )
    h = read_binary(str(next((work / "bin").glob("*.00000.bin"))))
    gamma = float(b["mhd"]["gamma"])
    profiles = []
    for m, (x0, x1, y0, y1, z0, z1) in enumerate(h["mb_geometry"]):
        if z0 != 0:
            continue
        d = h["mb_data"]
        for i in range(32):
            r = math.hypot(x0 + i + 0.5, 0.5)
            rho, p, bz = [d[k][m][0, 4, i] for k in ("dens", "press", "bcc3")]
            # Newtonian midplane scale estimates at R >> M, in the inertial frame.
            enthalpy = rho + gamma * p / (gamma - 1)
            alfven = abs(bz) / math.sqrt(enthalpy + bz * bz)
            sound = math.sqrt(gamma * p / enthalpy)
            profiles.append([r, 2 * math.pi * alfven * r**1.5, sound * r**1.5])
    return np.array(sorted(profiles))


def audit(mesh, profiles, block_nx1):
    boxes = []
    for block in mesh.read_text().split("#MeshBlock ")[1:]:
        xyz = np.fromstring(" ".join(block.splitlines()[2:]), sep=" ").reshape(-1, 3)
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        if lo[2] <= 0 < hi[2]:
            boxes.append((*lo[:2], *hi[:2], (hi[0] - lo[0]) / block_nx1))
    boxes = np.array(boxes)
    result = []
    for r in (64, 80, 100, 128, 150, 180, 200, 240, 272, 300, 320, 360, 400, 480, 520):
        wavelength, height = [np.interp(r, profiles[:, 0], profiles[:, k]) for k in (1, 2)]
        spacing = []
        for angle in (np.arange(128) + 0.5) * 2 * np.pi / 128:
            x, y = r * np.cos(angle), r * np.sin(angle)
            cells = boxes[
                (boxes[:, 0] <= x) & (x < boxes[:, 2]) & (boxes[:, 1] <= y) & (y < boxes[:, 3])
            ]
            if len(cells) != 1:
                raise ValueError(f"Expected a unique leaf at {(x,y)}")
            spacing.append(cells[0, 4])
        result.append(
            {
                "R": r,
                "dx_min": min(spacing),
                "dx_max": max(spacing),
                "lambda_MRI": wavelength,
                "H_sound": height,
                "Qz_min": wavelength / max(spacing),
                "Qz_median": wavelength / np.median(spacing),
                "H_over_dx_min": height / max(spacing),
            }
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("inertial_ic", type=Path)
    p.add_argument("production_input", type=Path)
    p.add_argument("mesh_structure", type=Path)
    p.add_argument("work", type=Path)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    profiles = midplane_probe(a.executable.resolve(), a.inertial_ic, a.work.resolve())
    cells = int(read_header(a.production_input)["meshblock"]["nx1"])
    print(json.dumps(audit(a.mesh_structure, profiles, cells), indent=2))
