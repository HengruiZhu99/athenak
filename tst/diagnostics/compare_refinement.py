"""Analyze matched fixed-time histories, including small and nonmonotone signals."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
import athena_read


def order(samples):
    a, b, c = samples
    d1, d2 = a - b, b - c
    floor = 1e-12 * max(1.0, abs(a), abs(b), abs(c))
    if min(abs(d1), abs(d2)) <= floor:
        return None
    return math.log2(abs(d1 / d2))


def refinement(histories, time):
    sizes = sorted(histories)
    if not sizes:
        raise ValueError("No matching run directories")
    finest = sizes[-1]
    levels = (finest // 4, finest // 2, finest)
    if finest % 4 or any(n not in histories for n in levels):
        raise ValueError(f"Need factor-of-two triplet {levels}")
    triplets = [
        (n // 4, n // 2, n)
        for n in sizes
        if n % 4 == 0 and n // 2 in histories and n // 4 in histories
    ]
    report = {}
    for n, h in histories.items():
        np.testing.assert_allclose(h["time"][-1], time, atol=1e-14, rtol=0)
        np.testing.assert_allclose(h["time"][0], 0, atol=1e-14, rtol=0)
        for key in h:
            if key.startswith("shell_") and key.endswith("_interval_dt"):
                np.testing.assert_allclose(
                    h[key][1:], np.diff(h["time"]), atol=1e-14, rtol=1e-11,
                    err_msg=f"Missing accepted step for {key} at {n}",
                )
            if key.startswith("valid_") or key.endswith("_volume_valid"):
                np.testing.assert_array_equal(
                    h[key], np.ones_like(h[key]), err_msg=f"{key} at {n}"
                )
    h = histories[finest]
    faces = [
        key[6:] for key in h if key.startswith("valid_") and "reason_" + key[6:] in h
    ]
    physical = {
        name + "_" + face
        for face in faces
        for name in (
            "mdot",
            "edot_f",
            "edot_em",
            "phiB",
            "area",
            "M0_net_in_moving",
            "M0_in_moving",
            "M0_out_moving",
        )
    }
    for face in faces:
        for sector in ("f", "em"):
            for axis in "xyz":
                for prefix in ("p", "l"):
                    physical.add(f"{prefix}{axis}dot_{sector}_{face}")
                for prefix in ("P", "J"):
                    physical.add(f"{prefix}{axis}_{sector}_in_moving_{face}")
            for energy in ("Et", "Eeu"):
                physical.add(f"{energy}_{sector}_in_moving_{face}")
    physical.update(
        k
        for k in h
        if k.startswith("shell_")
        and "_partial_" not in k
        and k.endswith(("_stored", "_source", "_boundary_in", "_sampled_volume"))
        and "_interval_" not in k
    )
    physical.update(k for k in h if k.startswith("M0_qualified_cumulative_"))
    for key in sorted(physical):
        values = [float(histories[n][key][-1]) for n in sizes]
        assert np.isfinite(values).all(), key
        endpoint = {n: v for n, v in zip(sizes, values)}
        triplet_orders = [order([endpoint[n] for n in t]) for t in triplets]
        a, b, c = (endpoint[n] for n in levels)
        initial = [float(histories[n][key][0]) for n in levels]
        report[key] = {
            "values": values,
            "order": triplet_orders[-1],
            "orders_by_triplet": triplet_orders,
            "reversal": (a - b) * (b - c) < 0,
            "initial_triplet": initial,
            "initial_order": order(initial),
        }
    residual = {}
    for key in h:
        if key.startswith("shell_") and key.endswith("_interval_residual"):
            prefix = key.removesuffix("residual")
            rates, fractions = [], []
            for n in sizes:
                runs = histories[n]
                error = np.abs(runs[key][1:]).sum()
                assert np.isfinite(error)
                scale = sum(
                    np.abs(runs[prefix + t][1:]).sum()
                    for t in ("boundary", "source", "other")
                )
                rates.append(float(error / time))
                fractions.append(float(error / scale) if scale > 1e-12 else None)
            residual[key] = {"L1_rate": rates, "fraction": fractions}
    return {
        "sizes": sizes,
        "order_triplet": levels,
        "triplets": triplets,
        "observables": report,
        "balances": residual,
    }


def compare_frames(rot, inert, omega, time):
    """Compare geometric generators on matched slices at the common endpoint."""
    theta = omega * time
    rotation = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    differences = {}

    def record(label, measured, reference, scale=None):
        absolute = float(np.linalg.norm(measured - reference))
        norm = float(np.linalg.norm(reference)) if scale is None else scale
        differences[label] = {
            "absolute": absolute,
            "relative": absolute / norm if norm else None,
        }

    def vector(history, names):
        return np.array([history[k][-1] for k in names])

    for name in rot:
        if name.startswith("valid_") and "reason_" + name[6:] in rot:
            face = name[6:]
            for part in ("f", "em"):
                for q in ("P", "J"):
                    keys = [f"{q}{axis}_{part}_in_moving_{face}" for axis in "xyz"]
                    record(
                        f"{q}_{part}_{face}",
                        vector(rot, keys),
                        rotation @ vector(inert, keys),
                    )
                key = f"Jz_{part}_in_moving_{face}"
                record(key, rot[key][-1], inert[key][-1])
            for q in ("area", "M0_net_in_moving"):
                key = f"{q}_{face}"
                record(key, rot[key][-1], inert[key][-1])
        if name.startswith("shell_") and name.endswith("_volume_valid"):
            shell = name.removesuffix("_volume_valid")
            stored = [f"{shell}_P{axis}_stored" for axis in "xyz"]
            for kind in ("stored", "boundary_in", "source"):
                keys = [f"{shell}_P{axis}_{kind}" for axis in "xyz"]
                expected = rotation @ vector(inert, keys)
                if kind == "source":
                    expected -= np.cross(
                        [0, 0, omega], rotation @ vector(inert, stored)
                    )
                record(f"{shell}_P_{kind}", vector(rot, keys), expected)
                key = f"{shell}_Jz_{kind}"
                scale = abs(inert[key][-1])
                if kind == "source":
                    scale += abs(inert[f"{shell}_Jz_boundary_in"][-1])
                record(key, rot[key][-1], inert[key][-1], scale)
    return differences


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Parents of rot-N and inert-N directories",
    )
    p.add_argument("--time", type=float, default=0.02)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--omega", type=float, default=0.125, help="Binary frame angular velocity"
    )
    args = p.parse_args()
    result = {}
    histories = {}
    for mode in ("rot", "inert"):
        series = {}
        for directory in args.directories:
            for run in directory.glob(mode + "-*"):
                n = int(run.name.split("-")[-1])
                if n in series:
                    raise ValueError(f"Duplicate {mode}-{n}")
                series[n] = athena_read.hst(run / "matched_bondi.user.hst")
        result[mode] = refinement(series, args.time)
        histories[mode] = series
    if histories["rot"].keys() != histories["inert"].keys():
        raise ValueError("Frame comparison requires matched resolutions")
    result["frame_differences"] = {
        n: compare_frames(
            histories["rot"][n], histories["inert"][n], args.omega, args.time
        )
        for n in sorted(histories["rot"])
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
