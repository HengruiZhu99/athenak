#!/usr/bin/env python3
"""Prospective three-grid qualification for the half-plane spin-0.5 Kerr case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


CASES = ("h32", "h48", "h64")
H = {"h32": 1.0 / 32.0, "h48": 1.0 / 48.0, "h64": 1.0 / 64.0}
FAMILIES = ("C", "H", "M", "Z")
REGIONS = ("global", "axis", "off_axis", "layer0", "layer1", "layer2",
           "layer3", "layer4", "linf")
ANALYTIC = {
    "area": 8.0 * math.pi * (1.0 + math.sqrt(0.75)),
    "irreducible_mass": math.sqrt(
        8.0 * math.pi * (1.0 + math.sqrt(0.75)) / (16.0 * math.pi)),
    "horizon_mass": 1.0,
    "spin_z": 0.5,
    "center_z": 0.0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def parse_history(path: Path) -> list[dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#  [1]=")]
    require(len(headers) == 1, f"{path}: expected exactly one indexed header")
    labels = re.findall(r"\[\d+\]=([^ ]+)", headers[0])
    require(len(labels) == len(set(labels)), f"{path}: duplicate history label")
    required = {"time", "Volume", "ax-N", "off-Vol"}
    for family in FAMILIES:
        required |= {f"{family}-norm2", f"ax-{family}2", f"off-{family}2",
                     f"{family}-Linf"}
        required |= {f"L{layer}-{family}2" for layer in range(5)}
    required |= {f"L{layer}-N" for layer in range(5)}
    require(required <= set(labels), f"{path}: incomplete Cartoon history inventory")

    by_time: dict[float, dict[str, float]] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = [float(token) for token in line.split()]
        require(len(values) == len(labels), f"{path}: row width differs from header")
        require(all(math.isfinite(value) for value in values),
                f"{path}: nonfinite history value")
        row = dict(zip(labels, values, strict=True))
        by_time[row["time"]] = row
    rows = [by_time[time] for time in sorted(by_time)]
    require(len(rows) >= 2 and all(rows[n]["time"] < rows[n + 1]["time"]
                                   for n in range(len(rows) - 1)),
            f"{path}: insufficient strictly increasing history")
    return rows


def rms(row: dict[str, float], family: str, region: str) -> float:
    if region == "global":
        numerator, denominator = row[f"{family}-norm2"], row["Volume"]
    elif region == "axis":
        numerator, denominator = row[f"ax-{family}2"], row["ax-N"]
    elif region == "off_axis":
        numerator, denominator = row[f"off-{family}2"], row["off-Vol"]
    elif region.startswith("layer"):
        layer = int(region.removeprefix("layer"))
        numerator, denominator = row[f"L{layer}-{family}2"], row[f"L{layer}-N"]
    elif region == "linf":
        return row[f"{family}-Linf"]
    else:
        raise RuntimeError(f"unknown region {region}")
    require(numerator >= 0.0 and denominator > 0.0,
            f"invalid {family}/{region} norm inputs")
    return math.sqrt(numerator / denominator)


def legendre(order: int, x: float) -> float:
    if order == 0:
        return 1.0
    if order == 1:
        return x
    lower, current = 1.0, x
    for ell in range(1, order):
        upper = ((2 * ell + 1) * x * current - ell * lower) / (ell + 1)
        lower, current = current, upper
    return current


def reflection_residual(coefficients: list[float], mean_radius: float) -> float:
    require(mean_radius > 0.0, "horizon mean radius is nonpositive")
    maximum = 0.0
    for x in np.linspace(-1.0, 1.0, 257):
        positive = 0.0
        reflected = 0.0
        for ell, coefficient in enumerate(coefficients):
            normalization = math.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi))
            positive += coefficient * normalization * legendre(ell, float(x))
            reflected += coefficient * normalization * legendre(ell, float(-x))
        maximum = max(maximum, abs(positive - reflected))
    return maximum / mean_radius


def parse_horizon(path: Path) -> list[dict[str, float | list[float]]]:
    by_time: dict[float, dict[str, float | list[float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        require(len(fields) >= 17, f"{path}: short horizon row")
        if fields[2] != "origin" or fields[3] != "1":
            continue
        values = [float(value) for value in fields[4:15]]
        coefficients = [float(value) for value in fields[16:]]
        require(coefficients and all(math.isfinite(value) for value in
                                     (*values, *coefficients)),
                f"{path}: nonfinite accepted horizon row")
        names = ("center_z", "axis_extremum_z", "center_lapse", "area",
                 "irreducible_mass", "horizon_mass", "spin_z", "mean_radius",
                 "minimum_radius", "direct_residual", "flow_residual")
        row: dict[str, float | list[float]] = dict(zip(names, values, strict=True))
        row["cycle"] = float(fields[0])
        row["time"] = float(fields[1])
        row["coefficients"] = coefficients
        row["reflection_residual"] = reflection_residual(
            coefficients, float(row["mean_radius"]))
        by_time[float(row["time"])] = row
    rows = [by_time[time] for time in sorted(by_time)]
    require(rows and all(float(rows[n]["time"]) < float(rows[n + 1]["time"])
                         for n in range(len(rows) - 1)),
            f"{path}: no strictly increasing accepted origin horizon series")
    return rows


def interpolate(times: np.ndarray, values: np.ndarray,
                targets: np.ndarray) -> np.ndarray:
    require(targets[0] >= times[0] and targets[-1] <= times[-1],
            "interpolation target outside common coverage")
    return np.interp(targets, times, values)


def pair_orders(errors: dict[str, np.ndarray]) -> dict[str, list[float | None]]:
    result = {"p32_48": [], "p48_64": []}
    for coarse, fine, key in (("h32", "h48", "p32_48"),
                              ("h48", "h64", "p48_64")):
        for left, right in zip(errors[coarse], errors[fine], strict=True):
            order = None
            if left > 0.0 and right > 0.0:
                order = math.log(float(left / right)) / math.log(H[coarse] / H[fine])
            result[key].append(finite_number(order) if order is not None else None)
    return result


def unequal_order(q1: float, q2: float, q3: float) -> float | None:
    denominator = q2 - q3
    if denominator == 0.0:
        return None
    ratio = (q1 - q2) / denominator
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None

    def residual(order: float) -> float:
        expected = ((H["h32"] ** order - H["h48"] ** order) /
                    (H["h48"] ** order - H["h64"] ** order))
        return expected - ratio

    grid = np.linspace(0.01, 12.0, 1200)
    values = [residual(float(order)) for order in grid]
    for index in range(len(grid) - 1):
        if values[index] == 0.0:
            return float(grid[index])
        if values[index] * values[index + 1] > 0.0:
            continue
        lower, upper = float(grid[index]), float(grid[index + 1])
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if residual(lower) * residual(middle) <= 0.0:
                upper = middle
            else:
                lower = middle
        return 0.5 * (lower + upper)
    return None


def common_targets(series: dict[str, list[dict[str, object]]],
                   start: float = 0.5, spacing: float = 0.25) -> np.ndarray:
    end = min(float(rows[-1]["time"]) for rows in series.values())
    require(end >= start, "common time coverage ends before the qualification window")
    values = list(np.arange(start, end + 1.0e-12, spacing))
    if end - values[-1] > 1.0e-10:
        values.append(end)
    return np.asarray(values)


def monotone_fraction(series: dict[str, np.ndarray]) -> float:
    passes = ((series["h32"] > series["h48"]) &
              (series["h48"] > series["h64"]))
    return float(np.mean(passes))


def analyze_gauge(run_root: Path, gauge: str) -> dict[str, object]:
    histories: dict[str, list[dict[str, float]]] = {}
    horizons: dict[str, list[dict[str, float | list[float]]]] = {}
    statuses: dict[str, int] = {}
    for case in CASES:
        case_root = run_root / gauge / case
        basename = f"kerr_half_plane_{case}_{gauge}"
        statuses[case] = int((case_root / "status.txt").read_text().strip())
        histories[case] = parse_history(case_root / f"{basename}.z4c.user.hst")
        horizons[case] = parse_horizon(
            case_root / f"{basename}.cartoon_m0_horizon_0.txt")

    history_targets = common_targets(histories)
    horizon_targets = common_targets(horizons)
    constraints: dict[str, object] = {}
    constraint_gate = True
    for family in FAMILIES:
        constraints[family] = {}
        for region in REGIONS:
            values = {
                case: interpolate(
                    np.asarray([row["time"] for row in histories[case]]),
                    np.asarray([rms(row, family, region) for row in histories[case]]),
                    history_targets)
                for case in CASES
            }
            orders = pair_orders(values)
            fraction = monotone_fraction(values)
            positive_fractions = {
                key: float(np.mean([value is not None and value > 0.0
                                    for value in sequence]))
                for key, sequence in orders.items()
            }
            passes = (fraction >= 0.9 and
                      all(value >= 0.8 for value in positive_fractions.values()) and
                      values["h32"][-1] > values["h48"][-1] > values["h64"][-1])
            constraint_gate &= passes
            constraints[family][region] = {
                "times": history_targets.tolist(),
                "values": {case: values[case].tolist() for case in CASES},
                "orders": orders,
                "monotone_fraction": fraction,
                "positive_order_fraction": positive_fractions,
                "passes": bool(passes),
            }

    horizon_quantities = ("area", "irreducible_mass", "horizon_mass", "spin_z",
                          "center_z", "mean_radius", "minimum_radius",
                          "direct_residual", "flow_residual", "reflection_residual")
    horizon_result: dict[str, object] = {}
    horizon_gate = True
    for quantity in horizon_quantities:
        values = {
            case: interpolate(
                np.asarray([float(row["time"]) for row in horizons[case]]),
                np.asarray([float(row[quantity]) for row in horizons[case]]),
                horizon_targets)
            for case in CASES
        }
        entry: dict[str, object] = {
            "times": horizon_targets.tolist(),
            "values": {case: values[case].tolist() for case in CASES},
        }
        if quantity in ANALYTIC:
            errors = {case: np.abs(values[case] - ANALYTIC[quantity]) for case in CASES}
            entry["absolute_errors"] = {case: errors[case].tolist() for case in CASES}
            entry["orders"] = pair_orders(errors)
            entry["error_monotone_fraction"] = monotone_fraction(errors)
        elif quantity in ("mean_radius", "minimum_radius"):
            entry["three_grid_orders"] = [
                finite_number(unequal_order(values["h32"][index],
                                            values["h48"][index],
                                            values["h64"][index]))
                for index in range(len(horizon_targets))
            ]
        horizon_result[quantity] = entry

    direct_ok = all(max(float(row["direct_residual"]) for row in horizons[case]) <= 0.03
                    for case in CASES)
    flow_ok = all(max(float(row["flow_residual"]) for row in horizons[case]) <= 0.03
                  for case in CASES)
    reflection_ok = all(max(float(row["reflection_residual"])
                            for row in horizons[case]) <= 0.05 for case in CASES)
    invariant_ok = all(float(horizon_result[name]["error_monotone_fraction"]) >= 0.8
                       for name in ("area", "horizon_mass", "spin_z"))
    horizon_gate &= direct_ok and flow_ok and reflection_ok and invariant_ok

    reached = all(statuses[case] == 0 and histories[case][-1]["time"] >= 5.0 - 1e-9
                  for case in CASES)
    horizon_coverage = all(float(horizons[case][-1]["time"]) >= 4.8 for case in CASES)
    return {
        "status": statuses,
        "coverage": {
            case: {"history_end": histories[case][-1]["time"],
                   "horizon_end": horizons[case][-1]["time"]}
            for case in CASES
        },
        "constraints": constraints,
        "horizon": horizon_result,
        "gates": {
            "all_runs_reached_5M": reached,
            "horizon_coverage_through_4p8M": horizon_coverage,
            "constraint_time_dependent_convergence": bool(constraint_gate),
            "horizon_invariant_time_dependent_convergence": invariant_ok,
            "direct_residual_at_most_0p03": direct_ok,
            "flow_residual_at_most_0p03": flow_ok,
            "reflection_residual_at_most_0p05": reflection_ok,
            "gauge_qualification": bool(reached and horizon_coverage and
                                        constraint_gate and horizon_gate),
        },
    }


def self_test() -> None:
    require(finite_number(None) is None,
            "unresolved convergence order was not preserved as JSON null")
    require(unequal_order(1.0, 1.0, 1.0) is None,
            "degenerate three-grid values unexpectedly produced an order")
    require(json.loads(json.dumps({"gate": bool(np.bool_(True))})) == {"gate": True},
            "NumPy-derived gate was not normalized to a strict JSON boolean")
    synthetic = {
        case: np.asarray([H[case] ** 4, 2.0 * H[case] ** 4]) for case in CASES
    }
    orders = pair_orders(synthetic)
    require(all(math.isclose(float(value), 4.0, rel_tol=1e-12, abs_tol=1e-12)
                for values in orders.values() for value in values),
            "known-exact pairwise-order self-test failed")
    require(math.isclose(float(unequal_order(
        3.0 + H["h32"] ** 5,
        3.0 + H["h48"] ** 5,
        3.0 + H["h64"] ** 5)), 5.0, rel_tol=2e-6, abs_tol=2e-6),
            "unknown-limit unequal-spacing order self-test failed")
    require(reflection_residual([math.sqrt(4.0 * math.pi), 0.0, 0.1], 1.0) < 1e-14,
            "even horizon coefficients lost reflection symmetry")
    require(reflection_residual([math.sqrt(4.0 * math.pi), 0.1], 1.0) > 0.09,
            "odd horizon coefficient did not register a reflection residual")
    print("Cartoon half-plane Kerr convergence analyzer self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gauges", nargs="+", choices=("moving_puncture",),
                        default=("moving_puncture",))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.run_root is not None and args.output is not None,
            "--run-root and --output are required outside --self-test")
    result = {
        "schema": "athenak_cartoon_half_plane_kerr_convergence_v2",
        "claim_scope": "fresh_three_grid_time_dependent_constraint_and_horizon_qualification",
        "mass": 1.0,
        "dimensionless_spin": 0.5,
        "finest_spacing_M": H,
        "gauge": {
            "name": "athenak_default_moving_puncture",
            "lapse": "advective_1_plus_log",
            "shift": "advective_Gamma_driver",
        },
        "prospective_gates": {
            "constraint_monotone_fraction": 0.9,
            "positive_order_fraction": 0.8,
            "horizon_invariant_error_monotone_fraction": 0.8,
            "direct_and_flow_residual_max": 0.03,
            "reflection_residual_max": 0.05,
        },
        "gauges": {gauge: analyze_gauge(args.run_root, gauge)
                   for gauge in args.gauges},
    }
    result["qualification_claim"] = all(
        gauge["gates"]["gauge_qualification"] for gauge in result["gauges"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"qualification_claim": result["qualification_claim"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
