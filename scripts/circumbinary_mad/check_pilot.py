"""Verify real AMR disk footprints, horizon qualifications and CT in an MPI pilot."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

from ct_guard import check

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
import athena_read
from bin_convert import read_binary


def disk_spacings(snapshot):
    x0, x1, y0, y1 = snapshot["mb_geometry"][:, :4].T
    spacing = (x1 - x0) / snapshot["nx1_out_mb"]
    phase = (np.arange(128) + 0.5) * 2 * np.pi / 128
    result = {}
    for r in (100, 128, 150, 180, 240, 300, 400):
        samples = []
        for x, y in zip(r * np.cos(phase), r * np.sin(phase)):
            v = spacing[(x0 <= x) & (x < x1) & (y0 <= y) & (y < y1)]
            if len(v) != 1:
                raise ValueError(f"Missing or overlapping midplane cells at {(x, y)}")
            samples.append(v.item())
        result[r] = samples
    return result


def validate(path, sep=25):
    history = path / f"mad{sep}.user.hst"
    check(history)
    h = athena_read.hst(history)
    zones = set(h["zoom_zone"].astype(int))
    returns = sorted(set(h["zoom_event"][h["zoom_zone"] == 0].astype(int)) - {0})
    if zones != set(range(3)) or len(returns) < 2:
        raise ValueError("Pilot must complete multiple full zoom cycles")
    for key in ("valid_BH1_Kerr", "valid_BH2_Kerr"):
        if not np.all(h[key][h["zoom_zone"] == 0] == 1):
            raise ValueError(f"Unqualified finest-zone horizon {key}")
    for key in ("valid_H1", "valid_H2"):
        if not np.all(h[key] == 1):
            raise ValueError(f"Unqualified 5M transfer surface {key}")
    baseline = None
    audited = set()
    for file in sorted((path / "bin").glob("*mesh_audit*.bin")):
        s = read_binary(str(file))
        i = int(np.argmin(np.abs(h["time"] - s["time"])))
        if abs(h["time"][i] - s["time"]) > 1e-4:
            raise ValueError("Slice missing from pilot history")
        zone = int(h["zoom_zone"][i])
        if zone in audited:
            continue
        spacing = disk_spacings(s)
        if baseline is None:
            if zone != 0:
                raise ValueError("Missing finest initial disk slice")
            baseline = spacing
        if spacing != baseline:
            raise ValueError(f"Inner zoom coarsened an active disk annulus in zone {zone}")
        audited.add(zone)
    if audited != zones:
        raise ValueError(f"Disk slices cover zones {audited}, expected {zones}")
    return {
        "full_returns": len(returns),
        "max_relative_divB": max(h["ct_divB_relative"]),
        "peak_face_B": max(h["ct_face_B_max"]),
        "disk_zones": sorted(audited),
        "max_dx_by_radius": {r: max(v) for r, v in baseline.items()},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilot", type=Path)
    p.add_argument("--separation", type=int, default=25)
    args = p.parse_args()
    print(json.dumps(validate(args.pilot, args.separation), indent=2))
