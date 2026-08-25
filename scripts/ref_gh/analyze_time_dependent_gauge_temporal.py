#!/usr/bin/env python3
"""Analyze variable-step RK4 self-convergence of Ref-GH gauge subtraction."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from analyze_perturbed_trumpet_convergence import read_cbin


TIME_RE = re.compile(
    r"cycle=(?P<cycle>[0-9]+)\s+time=(?P<time>[-+0-9.eE]+)\s+dt=")


def accepted_steps(path: Path) -> list[float]:
    """Return accepted step sizes from AthenaK cycle/time progress lines."""
    cycle_times: dict[int, float] = {}
    for match in TIME_RE.finditer(path.read_text(encoding="utf-8")):
        cycle_times[int(match.group("cycle"))] = float(match.group("time"))
    if len(cycle_times) < 2 or min(cycle_times) != 0:
        raise ValueError(f"{path}: need progress lines from cycle 0 through final")
    cycles = sorted(cycle_times)
    if cycles != list(range(cycles[-1] + 1)):
        raise ValueError(f"{path}: non-contiguous progress cycles {cycles}")
    steps = [cycle_times[cycle] - cycle_times[cycle - 1]
             for cycle in cycles[1:]]
    if any(not math.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError(f"{path}: invalid accepted steps {steps}")
    return steps


def leading_weight(steps: list[float], order: float) -> float:
    """Global variable-step order-p error weight for local error p+1."""
    return sum(step**(order + 1.0) for step in steps)


def predicted_ratio(step_sets: list[list[float]], order: float) -> float:
    weights = [leading_weight(steps, order) for steps in step_sets]
    denominator = weights[1] - weights[2]
    return (weights[0] - weights[1])/denominator


def effective_order(ratio: float, step_sets: list[list[float]]) -> float:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return math.nan
    lo, hi = 0.01, 12.0
    flo = predicted_ratio(step_sets, lo) - ratio
    fhi = predicted_ratio(step_sets, hi) - ratio
    if flo*fhi > 0.0:
        return math.nan
    for _ in range(100):
        mid = 0.5*(lo + hi)
        fmid = predicted_ratio(step_sets, mid) - ratio
        if flo*fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5*(lo + hi)


def puncture_clear_mask(metadata: dict, stencil_radius: int,
                        puncture: tuple[float, float, float]) -> np.ndarray:
    """Select native cells whose full axis-aligned stencil box is puncture-free."""
    nz, ny, nx = metadata["data"].shape[-3:]
    coordinates = []
    for (lower, upper), count in zip(metadata["bounds"], (nx, ny, nz)):
        coordinates.append(
            lower + (np.arange(count) + 0.5)*(upper - lower)/count)
    z, y, x = np.meshgrid(coordinates[2], coordinates[1], coordinates[0],
                          indexing="ij")
    spacing = [(upper - lower)/count
               for (lower, upper), count in zip(
                   metadata["bounds"], (nx, ny, nz))]
    overlaps = ((np.abs(x - puncture[0]) <= stencil_radius*spacing[0])
                & (np.abs(y - puncture[1]) <= stencil_radius*spacing[1])
                & (np.abs(z - puncture[2]) <= stencil_radius*spacing[2]))
    return ~overlaps


def norm_result(arrays: list[np.ndarray], component_slice: slice,
                mask: np.ndarray, step_sets: list[list[float]]) -> dict:
    first = (arrays[0] - arrays[1])[component_slice][:, mask].ravel()
    second = (arrays[1] - arrays[2])[component_slice][:, mask].ravel()
    first_l2 = float(np.sqrt(np.mean(first*first)))
    second_l2 = float(np.sqrt(np.mean(second*second)))
    first_linf = float(np.max(np.abs(first)))
    second_linf = float(np.max(np.abs(second)))
    ratio_l2 = first_l2/second_l2
    ratio_linf = first_linf/second_linf
    expected = predicted_ratio(step_sets, 4.0)
    return {
        "coarse_medium_L2": first_l2,
        "medium_fine_L2": second_l2,
        "ratio_L2": ratio_l2,
        "variable_step_order_L2": effective_order(ratio_l2, step_sets),
        "ratio_L2_over_RK4_prediction": ratio_l2/expected,
        "coarse_medium_Linf": first_linf,
        "medium_fine_Linf": second_linf,
        "ratio_Linf": ratio_linf,
        "variable_step_order_Linf": effective_order(ratio_linf, step_sets),
        "ratio_Linf_over_RK4_prediction": ratio_linf/expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", nargs=3, type=Path, required=True,
                        metavar=("COARSE", "MEDIUM", "FINE"))
    parser.add_argument("--log", nargs=3, type=Path, required=True,
                        metavar=("COARSE", "MEDIUM", "FINE"))
    parser.add_argument(
        "--fd-stencil-radius", type=int, required=True,
        help=("maximum native-cell radius of every evolution operator; for "
              "fourth-order Ref-GH with nonzero KO dissipation this is 3"))
    parser.add_argument("--puncture", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fd_stencil_radius < 1:
        raise ValueError("fd-stencil-radius must be positive")

    loaded = [read_cbin(path) for path in args.field]
    variables = loaded[0]["variables"]
    shape = loaded[0]["data"].shape
    for item in loaded[1:]:
        if (item["variables"] != variables or item["data"].shape != shape
                or item["bounds"] != loaded[0]["bounds"]
                or item["variable_size"] != loaded[0]["variable_size"]):
            raise ValueError(
                "field labels, shapes, bounds, or precision differ across ladder")
    times = [float(item["time"]) for item in loaded]
    if max(times) - min(times) > 1.0e-14:
        raise ValueError(f"final output times differ: {times}")
    try:
        gauge_offset = variables.index("ref_gh_Hhat0")
    except ValueError as error:
        raise ValueError("field output lacks ref_gh_Hhat0") from error

    step_sets = [accepted_steps(path) for path in args.log]
    final_step_times = [sum(steps) for steps in step_sets]
    if any(abs(time - times[0]) > 2.0e-9 for time in final_step_times):
        raise ValueError(
            f"log final times {final_step_times} disagree with output {times[0]}")
    rk4_prediction = predicted_ratio(step_sets, 4.0)
    arrays = [item["data"] for item in loaded]
    mask = puncture_clear_mask(
        loaded[0], args.fd_stencil_radius, tuple(args.puncture))
    if not np.any(mask):
        raise ValueError("puncture-stencil mask removes every native cell")
    result = {
        "method": ("three-level self-convergence normalized by the actual "
                   "variable-step RK4 leading-error weight sum(dt^5)"),
        "final_time": times[0],
        "shape": shape,
        "variables": len(variables),
        "gauge_component_offset": gauge_offset,
        "analysis_sample_count": int(np.count_nonzero(mask)),
        "total_native_cell_count": int(mask.size),
        "puncture_stencil_mask": {
            "enabled": True,
            "fd_stencil_radius": args.fd_stencil_radius,
            "puncture": args.puncture,
            "rule": ("reject a native cell when its full axis-aligned "
                     "finite-difference support box contains the puncture"),
            "scope": ("caller-supplied maximum evolution footprint, including "
                      "Kreiss-Oliger dissipation"),
        },
        "accepted_steps": step_sets,
        "rk4_leading_weights": [leading_weight(steps, 4.0)
                                 for steps in step_sets],
        "rk4_predicted_self_difference_ratio": rk4_prediction,
        "all": norm_result(arrays, slice(None), mask, step_sets),
        "einstein": norm_result(
            arrays, slice(0, gauge_offset), mask, step_sets),
        "gauge": norm_result(
            arrays, slice(gauge_offset, None), mask, step_sets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
