#!/usr/bin/env python3
"""Reduce AMR physical fields for the Ref-GH wormhole-to-trumpet test.

The input is Athena binary-output v1.1 data on leaf MeshBlocks.  Every cell in
the conservative three-cell puncture-stencil cube is discarded.  The remaining
physical lapse, determinant conformal factor, and radial shift are compared
directly with both isotropic-wormhole and stationary n=2 trumpet profiles.
"""

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


IDS = ("physical_alpha", "physical_psi4", "physical_betax",
       "physical_betay", "physical_betaz")
REGIONS = (
    ("whole", 0.0, math.inf),
    ("r_lt_1", 0.0, 1.0),
    ("r_1_2", 1.0, 2.0),
    ("r_2_4", 2.0, 4.0),
    ("r_4_8", 4.0, 8.0),
    ("r_ge_8", 8.0, math.inf),
)
PROFILE_EDGES = np.asarray(
    (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
     3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, math.inf),
    dtype=np.float64)


def header_value(line):
    return line.decode("ascii").split("=", 1)[1].strip()


def read_cbin(path):
    """Read scalar CBin data and retain each leaf MeshBlock separately."""
    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"Athena binary output version=1.1":
            raise ValueError("{}: unsupported header {!r}".format(path, first))
        metadata = {}
        variables = []
        input_bytes = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("{}: truncated text header".format(path))
            stripped = line.strip()
            if stripped.startswith(b"time="):
                metadata["time"] = float(header_value(stripped))
            elif stripped.startswith(b"cycle="):
                metadata["cycle"] = int(header_value(stripped))
            elif stripped.startswith(b"size of location="):
                metadata["location_size"] = int(header_value(stripped))
            elif stripped.startswith(b"size of variable="):
                metadata["variable_size"] = int(header_value(stripped))
            elif stripped.startswith(b"number of variables="):
                metadata["nvar"] = int(header_value(stripped))
            elif stripped.startswith(b"variables:"):
                variables = stripped.split(b":", 1)[1].decode("ascii").split()
            elif stripped.startswith(b"header offset="):
                input_bytes = int(header_value(stripped))
                break
        if input_bytes is None:
            raise ValueError("{}: missing input header size".format(path))
        stream.read(input_bytes)
        if metadata.get("location_size") != 8:
            raise ValueError("{}: locations are not binary64".format(path))
        if metadata.get("variable_size") != 8:
            raise ValueError("{}: field is not binary64".format(path))
        if metadata.get("nvar") != 1 or len(variables) != 1:
            raise ValueError("{}: expected exactly one scalar".format(path))

        blocks = {}
        prefix_struct = struct.Struct("<10i6d")
        while True:
            prefix = stream.read(prefix_struct.size)
            if not prefix:
                break
            if len(prefix) != prefix_struct.size:
                raise ValueError("{}: truncated MeshBlock prefix".format(path))
            values = prefix_struct.unpack(prefix)
            ois, oie, ojs, oje, oks, oke, lx1, lx2, lx3, level = values[:10]
            bounds = tuple(values[10:])
            shape = (oke - oks + 1, oje - ojs + 1, oie - ois + 1)
            count = shape[0]*shape[1]*shape[2]
            raw = stream.read(8*count)
            if len(raw) != 8*count:
                raise ValueError("{}: truncated MeshBlock data".format(path))
            key = (lx1, lx2, lx3, level, shape, bounds)
            if key in blocks:
                raise ValueError("{}: duplicate MeshBlock".format(path))
            blocks[key] = np.frombuffer(raw, dtype="<f8").reshape(shape).copy()
    if not blocks:
        raise ValueError("{}: no MeshBlocks".format(path))
    metadata.update(path=str(path), variable=variables[0], blocks=blocks)
    return metadata


def read_header_time(path):
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Athena binary output version=1.1":
            raise ValueError("{}: unsupported header".format(path))
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("{}: missing time".format(path))
            if line.strip().startswith(b"time="):
                return float(header_value(line.strip()))


def discover(run_dir):
    """Return time -> diagnostic-id -> CBin path."""
    by_id = {}
    for identifier in IDS:
        candidates = sorted(run_dir.rglob("*{}*.cbin".format(identifier)))
        if not candidates:
            raise ValueError("no CBin files for {}".format(identifier))
        timed = {}
        for path in candidates:
            time = read_header_time(path)
            if time in timed:
                raise ValueError("duplicate {} snapshot at {}".format(
                    identifier, time))
            timed[time] = path
        by_id[identifier] = timed
    common = set.intersection(*(set(values) for values in by_id.values()))
    if not common:
        raise ValueError("physical field outputs have no common times")
    for identifier, timed in by_id.items():
        if set(timed) != common:
            raise ValueError("{} snapshot times differ".format(identifier))
    return {
        time: {identifier: by_id[identifier][time] for identifier in IDS}
        for time in sorted(common)
    }


def parse_constant(text, name):
    match = re.search(r"\b{}\s*=\s*([^;]+);".format(name), text)
    if match is None:
        raise ValueError("missing {} in trumpet table".format(name))
    return float(match.group(1))


def parse_array(text, name):
    match = re.search(
        r"\b{}\[kTrumpetTableSize\]\s*=\s*\{{(.*?)\}};".format(name),
        text, re.DOTALL)
    if match is None:
        raise ValueError("missing {} in trumpet table".format(name))
    return np.fromstring(match.group(1).replace(",", " "), sep=" ")


def load_trumpet_table(source_root):
    text = (source_root / "src/ref_gh/trumpet_table_generated.hpp").read_text(
        encoding="utf-8")
    size = int(parse_constant(text, "kTrumpetTableSize"))
    table = {
        "log_min": parse_constant(text, "kTrumpetLogRMin"),
        "spacing": parse_constant(text, "kTrumpetLogRSpacing"),
    }
    for profile in ("Alpha", "ArealRadius", "ShiftQ"):
        coefficients = [
            parse_array(text, "kTrumpet{}A{}".format(profile, degree))
            for degree in range(6)
        ]
        if any(array.size != size for array in coefficients):
            raise ValueError("wrong {} coefficient size".format(profile))
        table[profile] = coefficients
    return table


def interpolate_profile(table, name, radius):
    u = (np.log(radius) - table["log_min"])/table["spacing"]
    size = table[name][0].size
    index = np.clip(np.floor(u).astype(np.int64), 0, size - 2)
    s = u - index
    a0, a1, a2, a3, a4, a5 = (
        coefficient[index] for coefficient in table[name])
    return a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))


def targets(table, radius):
    wormhole_psi = 1.0 + 0.5/radius
    areal = interpolate_profile(table, "ArealRadius", radius)
    shift_q = interpolate_profile(table, "ShiftQ", radius)
    return {
        "wormhole": {
            "alpha": wormhole_psi**-2,
            "psi4": wormhole_psi**4,
            "beta_radial": np.zeros(radius.shape, dtype=np.float64),
        },
        "trumpet": {
            "alpha": interpolate_profile(table, "Alpha", radius),
            "psi4": (areal/radius)**2,
            "beta_radial": shift_q*radius,
        },
    }


def weighted_summary(values, weights):
    if values.size == 0:
        return {
            "cells": 0, "volume": 0.0, "mean": None, "RMS": None,
            "minimum": None, "maximum": None,
        }
    volume = float(np.sum(weights))
    return {
        "cells": int(values.size),
        "volume": volume,
        "mean": float(np.sum(weights*values)/volume),
        "RMS": float(np.sqrt(np.sum(weights*values*values)/volume)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def error_summary(observed, expected, weights):
    difference = observed - expected
    conditioned = difference/np.maximum(1.0, np.abs(expected))
    result = weighted_summary(difference, weights)
    result["conditioned_RMS"] = weighted_summary(
        conditioned, weights)["RMS"]
    result["Linf"] = float(np.max(np.abs(difference)))
    result["conditioned_Linf"] = float(np.max(np.abs(conditioned)))
    return result


def summarize_snapshot(paths, table, stencil_radius):
    fields = {identifier: read_cbin(path)
              for identifier, path in paths.items()}
    times = {payload["time"] for payload in fields.values()}
    if len(times) != 1:
        raise ValueError("snapshot field times differ")
    key_sets = [set(payload["blocks"]) for payload in fields.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("snapshot MeshBlock trees differ")

    region_data = {
        region[0]: {
            "alpha": [], "psi4": [], "beta_radial": [],
            "wormhole_alpha": [], "wormhole_psi4": [],
            "wormhole_beta_radial": [], "trumpet_alpha": [],
            "trumpet_psi4": [], "trumpet_beta_radial": [], "weights": [],
        } for region in REGIONS
    }
    bins = [{
        "radius": [], "alpha": [], "psi4": [], "beta_radial": [],
        "wormhole_alpha": [], "wormhole_psi4": [],
        "wormhole_beta_radial": [], "trumpet_alpha": [],
        "trumpet_psi4": [], "trumpet_beta_radial": [], "weights": [],
    } for _ in range(PROFILE_EDGES.size - 1)]
    excluded_cells = 0
    retained_cells = 0
    level_counts = {}

    for key in sorted(key_sets[0], key=lambda item: item[:4]):
        _, _, _, level, shape, bounds = key
        nz, ny, nx = shape
        xlo, xhi, ylo, yhi, zlo, zhi = bounds
        dx = (xhi - xlo)/nx
        dy = (yhi - ylo)/ny
        dz = (zhi - zlo)/nz
        x = xlo + (np.arange(nx, dtype=np.float64) + 0.5)*dx
        y = ylo + (np.arange(ny, dtype=np.float64) + 0.5)*dy
        z = zlo + (np.arange(nz, dtype=np.float64) + 0.5)*dz
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        radius = np.sqrt(xx*xx + yy*yy + zz*zz)
        clear = ((np.abs(xx) > stencil_radius*dx)
                 | (np.abs(yy) > stencil_radius*dy)
                 | (np.abs(zz) > stencil_radius*dz))
        excluded_cells += int(np.count_nonzero(~clear))
        retained_cells += int(np.count_nonzero(clear))
        level_counts[level] = level_counts.get(level, 0) + int(radius.size)
        if not np.any(clear):
            continue

        alpha = fields["physical_alpha"]["blocks"][key][clear]
        psi4 = fields["physical_psi4"]["blocks"][key][clear]
        bx = fields["physical_betax"]["blocks"][key][clear]
        by = fields["physical_betay"]["blocks"][key][clear]
        bz = fields["physical_betaz"]["blocks"][key][clear]
        rr = radius[clear]
        beta_radial = (
            bx*xx[clear] + by*yy[clear] + bz*zz[clear])/rr
        expected = targets(table, rr)
        weight = np.full(rr.shape, dx*dy*dz, dtype=np.float64)

        values = {
            "alpha": alpha, "psi4": psi4, "beta_radial": beta_radial,
            "wormhole_alpha": expected["wormhole"]["alpha"],
            "wormhole_psi4": expected["wormhole"]["psi4"],
            "wormhole_beta_radial": expected["wormhole"]["beta_radial"],
            "trumpet_alpha": expected["trumpet"]["alpha"],
            "trumpet_psi4": expected["trumpet"]["psi4"],
            "trumpet_beta_radial": expected["trumpet"]["beta_radial"],
        }
        for name, lower, upper in REGIONS:
            selected = (rr >= lower) & (rr < upper)
            if not np.any(selected):
                continue
            for field, array in values.items():
                region_data[name][field].append(array[selected])
            region_data[name]["weights"].append(weight[selected])
        indices = np.searchsorted(PROFILE_EDGES, rr, side="right") - 1
        for index in np.unique(indices):
            if index < 0 or index >= len(bins):
                continue
            selected = indices == index
            bins[index]["radius"].append(rr[selected])
            for field, array in values.items():
                bins[index][field].append(array[selected])
            bins[index]["weights"].append(weight[selected])

    def combine(items):
        return np.concatenate(items) if items else np.empty(0, dtype=np.float64)

    regions = {}
    for name, _, _ in REGIONS:
        data = region_data[name]
        weights = combine(data["weights"])
        observed = {
            field: combine(data[field])
            for field in ("alpha", "psi4", "beta_radial")
        }
        payload = {
            "cells": int(weights.size),
            "volume": float(np.sum(weights)) if weights.size else 0.0,
            "observed": {
                field: weighted_summary(values, weights)
                for field, values in observed.items()
            },
            "errors": {},
        }
        if weights.size:
            for reference in ("wormhole", "trumpet"):
                payload["errors"][reference] = {
                    field: error_summary(
                        observed[field], combine(data[
                            "{}_{}".format(reference, field)]), weights)
                    for field in observed
                }
        regions[name] = payload

    profiles = []
    for index, data in enumerate(bins):
        weights = combine(data["weights"])
        if not weights.size:
            continue
        payload = {
            "r_min": float(PROFILE_EDGES[index]),
            "r_max": (None if not math.isfinite(PROFILE_EDGES[index + 1])
                      else float(PROFILE_EDGES[index + 1])),
            "cells": int(weights.size),
            "volume": float(np.sum(weights)),
        }
        for field in ("radius", "alpha", "psi4", "beta_radial",
                      "wormhole_alpha", "wormhole_psi4",
                      "wormhole_beta_radial", "trumpet_alpha",
                      "trumpet_psi4", "trumpet_beta_radial"):
            payload[field] = weighted_summary(
                combine(data[field]), weights)["mean"]
        profiles.append(payload)

    finite = all(
        np.isfinite(payload["blocks"][key]).all()
        for payload in fields.values() for key in payload["blocks"])
    return {
        "time": float(next(iter(times))),
        "cycle": int(next(iter(fields.values()))["cycle"]),
        "files": {name: str(path) for name, path in paths.items()},
        "meshblocks": len(key_sets[0]),
        "level_cell_counts": {
            str(level): count for level, count in sorted(level_counts.items())
        },
        "stencil_radius_cells": stencil_radius,
        "excluded_puncture_stencil_cells": excluded_cells,
        "retained_cells": retained_cells,
        "all_physical_fields_finite": finite,
        "regions": regions,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target", action="append", type=float, default=[])
    parser.add_argument("--target-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--stencil-radius", type=int, default=3)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.stencil_radius < 2:
        raise ValueError("stencil-radius must cover at least the FD4 radius")

    discovered = discover(args.run_dir)
    targets_requested = args.target or sorted(discovered)
    selected = {}
    for target in targets_requested:
        nearest = min(discovered, key=lambda time: abs(time - target))
        if abs(nearest - target) > args.target_tolerance:
            raise ValueError("no physical snapshot near {}".format(target))
        selected[nearest] = discovered[nearest]
    table = load_trumpet_table(args.source_root)
    snapshots = [
        summarize_snapshot(selected[time], table, args.stencil_radius)
        for time in sorted(selected)
    ]

    movement = {}
    if len(snapshots) >= 2:
        initial = snapshots[0]["regions"]
        final = snapshots[-1]["regions"]
        for region in initial:
            movement[region] = {}
            for field in ("alpha", "psi4", "beta_radial"):
                initial_mean = initial[region]["observed"][field]["mean"]
                final_mean = final[region]["observed"][field]["mean"]
                movement[region][field] = {
                    "initial_mean": initial_mean,
                    "final_mean": final_mean,
                    "final_minus_initial": (
                        final_mean - initial_mean
                        if initial_mean is not None and final_mean is not None
                        else None),
                }
            ratios = {}
            for field in ("alpha", "psi4", "beta_radial"):
                wormhole = final[region]["errors"].get(
                    "wormhole", {}).get(field)
                trumpet = final[region]["errors"].get(
                    "trumpet", {}).get(field)
                denominator = (
                    wormhole["conditioned_RMS"] if wormhole is not None
                    else None)
                ratios[field] = (
                    trumpet["conditioned_RMS"]/denominator
                    if trumpet is not None and denominator is not None
                    and denominator > 0.0 else None)
            movement[region][
                "final_error_ratio_trumpet_over_wormhole"] = ratios

    payload = {
        "schema": "ref-gh-relative-damped-wormhole-physical-profile-v1",
        "claim_boundary": (
            "AMR leaf-cell physical profiles with a conservative three-cell "
            "puncture-stencil exclusion; no horizon or asymptotic claim"),
        "snapshots": snapshots,
        "movement": movement,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(args.output_prefix) + ".json")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "time\tr_min\tr_max\tcells\talpha\twormhole_alpha\t"
            "trumpet_alpha\tpsi4\twormhole_psi4\ttrumpet_psi4\t"
            "beta_radial\twormhole_beta_radial\ttrumpet_beta_radial\n")
        for snapshot in snapshots:
            for profile in snapshot["profiles"]:
                stream.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
                             "{}\t{}\t{}\n".format(
                    snapshot["time"], profile["r_min"], profile["r_max"],
                    profile["cells"], profile["alpha"],
                    profile["wormhole_alpha"], profile["trumpet_alpha"],
                    profile["psi4"], profile["wormhole_psi4"],
                    profile["trumpet_psi4"], profile["beta_radial"],
                    profile["wormhole_beta_radial"],
                    profile["trumpet_beta_radial"]))
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
