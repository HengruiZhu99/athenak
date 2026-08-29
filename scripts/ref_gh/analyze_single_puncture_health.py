#!/usr/bin/env python3
"""Extract accepted-state health telemetry for the frozen Ref-GH campaign.

The production executable is deliberately unchanged.  This tool reconstructs
the relative-metric signature and the compact q estimator from binary64 cbin
snapshots, then joins them to the existing native and user history streams.
Every finite-difference norm excludes the complete FD4+KO stencil support box
around the puncture.  Pointwise signature checks cover every active cell.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from analyze_perturbed_trumpet_convergence import read_cbin


REF_COLUMNS = (
    "time", "dt", "GH-L2sq", "Reduction-L2sq", "Curl-L2sq",
    "physical-g-error-L2sq", "physical-alpha-error-L2sq",
    "physical-beta-error-L2sq", "GHnear-L2sq",
    "ReductionNear-L2sq", "CurlNear-L2sq", "Volume", "alpha-max",
    "minus-alpha-min", "regular-max", "G-condition-max",
    "coordinate-g-max", "char-speed-max", "effective-CFL",
    "minus-detg-margin", "NearVolume", "bad-state", "Q-Linf",
    "Delta-Linf", "frame-Ricci-Linf", "coordinate-Ricci-Linf",
    "source-curvature-Linf", "source-QQ-Linf",
    "source-DeltaDelta-Linf", "source-damping-Linf",
    "source-frame-correction-Linf",
    "controller-q", "controller-q-dot", "controller-q-est",
)

USER_COLUMNS = (
    "time", "dt", "q", "q-dot", "q-ddot", "q-est", "q-analytic",
    "qest-minus-analytic", "q-variance", "q-effective-samples", "q-min",
    "q-max", "q-cells", "epsilon-G-mean", "epsilon-G-variance",
    "q-shell-valid", "q-generation", "q-frozen", "prescribed-q",
    "prescribed-q-dot", "prescribed-q-ddot", "prescribed-q-error",
    "prescribed-qdot-error",
)

PSI_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3),
)

# Frozen before the campaign.  These stop only grossly unusable trajectories;
# convergence and secular-drift decisions remain separate scientific gates.
CATASTROPHIC_RMS = 1.0
CATASTROPHIC_LINF = 10.0
Q_MIN = 0.5
Q_MAX = 2.5
QDOT_MAX = 0.25


def read_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, float]]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        values = list(map(float, raw.split()))
        if len(values) != len(columns):
            raise ValueError(
                f"{path}:{number}: expected {len(columns)} columns, got {len(values)}")
        rows.append(dict(zip(columns, values, strict=True)))
    if not rows:
        raise ValueError(f"{path}: no numerical rows")
    return rows


def closest(rows: list[dict[str, float]], time: float) -> dict[str, float]:
    return min(rows, key=lambda row: abs(row["time"] - time))


def variable(data: dict, name: str) -> np.ndarray:
    try:
        index = data["variables"].index(name)
    except ValueError as error:
        raise ValueError(f"missing variable {name!r}") from error
    return data["data"][index]


def relative_metric(state: dict) -> np.ndarray:
    shape = state["data"].shape[1:]
    metric = np.zeros(shape + (4, 4), dtype=np.float64)
    for a, b in PSI_PAIRS:
        values = variable(state, f"ref_gh_Psi{a}{b}")
        metric[..., a, b] = values
        metric[..., b, a] = values
    return metric


def coordinates(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nz, ny, nx = data["data"].shape[1:]
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = data["bounds"]
    x = xmin + (np.arange(nx) + 0.5)*(xmax - xmin)/nx
    y = ymin + (np.arange(ny) + 0.5)*(ymax - ymin)/ny
    z = zmin + (np.arange(nz) + 0.5)*(zmax - zmin)/nz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    h = min((xmax - xmin)/nx, (ymax - ymin)/ny, (zmax - zmin)/nz)
    return xx, yy, zz, h


def relative_geometry_health(metric: np.ndarray) -> dict[str, float | bool]:
    flat = metric.reshape((-1, 4, 4))
    lambda_min = math.inf
    lambda_max = -math.inf
    condition_max = 0.0
    relative_lapse_min = math.inf
    relative_lapse_max = -math.inf
    psi_max = float(np.max(np.abs(flat)))
    inverse_psi_max = 0.0
    finite = bool(np.isfinite(flat).all())
    signature_valid = finite

    for start in range(0, flat.shape[0], 100_000):
        points = flat[start:start + 100_000]
        spatial = points[:, 1:, 1:]
        try:
            eigenvalues = np.linalg.eigvalsh(spatial)
            spatial_inverse = np.linalg.inv(spatial)
        except np.linalg.LinAlgError:
            signature_valid = False
            continue
        local_min = eigenvalues[:, 0]
        local_max = eigenvalues[:, -1]
        lambda_min = min(lambda_min, float(np.min(local_min)))
        lambda_max = max(lambda_max, float(np.max(local_max)))
        positive = local_min > 0.0
        signature_valid = signature_valid and bool(np.all(positive))
        if np.any(positive):
            condition_max = max(
                condition_max,
                float(np.max(local_max[positive]/local_min[positive])))

        mixed = points[:, 0, 1:]
        raised = np.einsum("nij,nj->ni", spatial_inverse, mixed, optimize=True)
        alpha2 = -points[:, 0, 0] + np.einsum(
            "ni,ni->n", mixed, raised, optimize=True)
        valid_lapse = alpha2 > 0.0
        signature_valid = signature_valid and bool(np.all(valid_lapse))
        if np.any(valid_lapse):
            lapse = np.sqrt(alpha2[valid_lapse])
            relative_lapse_min = min(relative_lapse_min, float(np.min(lapse)))
            relative_lapse_max = max(relative_lapse_max, float(np.max(lapse)))

        safe_alpha2 = np.where(valid_lapse, alpha2, np.nan)
        inverse00 = -1.0/safe_alpha2
        inverse0i = raised/safe_alpha2[:, None]
        inverses = np.empty_like(points)
        inverses[:, 0, 0] = inverse00
        inverses[:, 0, 1:] = inverse0i
        inverses[:, 1:, 0] = inverse0i
        inverses[:, 1:, 1:] = (
            spatial_inverse
            - raised[:, :, None]*raised[:, None, :]/safe_alpha2[:, None, None])
        if np.isfinite(inverses).any():
            inverse_psi_max = max(
                inverse_psi_max, float(np.nanmax(np.abs(inverses))))

    return {
        "finite": finite,
        "signature_valid": signature_valid,
        "relative_G_lambda_min": lambda_min,
        "relative_G_lambda_max": lambda_max,
        "relative_G_condition_max": condition_max,
        "relative_lapse_min": relative_lapse_min,
        "relative_lapse_max": relative_lapse_max,
        "Psi_abs_max": psi_max,
        "inverse_Psi_abs_max": inverse_psi_max,
    }


def native_constraint_health(
        constraints: dict, clear: np.ndarray) -> dict[str, float]:
    gh2 = np.zeros(clear.shape, dtype=np.float64)
    for component in range(4):
        value = variable(constraints, f"ref_gh_C{component}")
        gh2 += value*value
    fields = {
        "GH": np.sqrt(gh2),
        "reduction": np.abs(variable(constraints, "ref_gh_reduction")),
        "curl": np.abs(variable(constraints, "ref_gh_curl")),
    }
    result = {}
    for name, values in fields.items():
        selected = values[clear]
        result[f"{name}_RMS"] = float(np.sqrt(np.mean(selected*selected)))
        result[f"{name}_Linf"] = float(np.max(selected))
    return result


def parse_constant(text: str, name: str) -> float:
    match = re.search(rf"\b{name}\s*=\s*([^;]+);", text)
    if match is None:
        raise ValueError(f"missing {name} in trumpet table")
    return float(match.group(1))


def parse_array(text: str, name: str) -> np.ndarray:
    match = re.search(
        rf"\b{name}\[kTrumpetTableSize\]\s*=\s*\{{(.*?)\}};",
        text, re.DOTALL)
    if match is None:
        raise ValueError(f"missing {name} in trumpet table")
    return np.fromstring(match.group(1).replace(",", " "), sep=" ")


def trumpet_l_and_q(source_root: Path, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    text = (source_root / "src/ref_gh/trumpet_table_generated.hpp").read_text(
        encoding="utf-8")
    table_size = int(parse_constant(text, "kTrumpetTableSize"))
    log_min = parse_constant(text, "kTrumpetLogRMin")
    spacing = parse_constant(text, "kTrumpetLogRSpacing")
    coefficients = [
        parse_array(text, f"kTrumpetArealRadiusA{degree}")
        for degree in range(6)
    ]
    if any(array.size != table_size for array in coefficients):
        raise ValueError("trumpet coefficient array has wrong size")
    u = (np.log(rho) - log_min)/spacing
    index = np.clip(np.floor(u).astype(np.int64), 0, table_size - 2)
    s = u - index
    a0, a1, a2, a3, a4, a5 = (array[index] for array in coefficients)
    areal = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))
    dy = (a1 + s*(2.0*a2 + s*(3.0*a3 + s*(4.0*a4 + s*5.0*a5))))/spacing
    areal_r = dy/rho
    trumpet_l = areal/rho
    trumpet_l_r = areal_r/rho - areal/(rho*rho)
    q_trumpet = -rho*trumpet_l_r/trumpet_l
    return trumpet_l, q_trumpet


def offline_q_health(
        state: dict, metric: np.ndarray, xx: np.ndarray, yy: np.ndarray,
        zz: np.ndarray, h: float, q: float, source_root: Path) -> dict[str, float | bool]:
    radius = np.sqrt(xx*xx + yy*yy + zz*zz)
    shell = (radius >= 2.0*h) & (radius < 8.0*h)
    clear = (np.abs(xx) > 3.0*h) | (np.abs(yy) > 3.0*h) | (np.abs(zz) > 3.0*h)
    selected = shell & clear
    if np.count_nonzero(selected) < 8:
        return {"shell_valid": False, "cell_count": int(np.count_nonzero(selected))}

    spatial = metric[..., 1:, 1:][selected]
    spatial_inverse = np.linalg.inv(spatial)
    rho = radius[selected]
    trumpet_l, q_analytic = trumpet_l_and_q(source_root, rho)
    window = np.exp(-(rho/3.0)**2)
    u = window*np.log(rho)
    u_r = window*(1.0/rho - 2.0*rho*np.log(rho)/9.0)
    reference_l = trumpet_l*np.exp(-(q - 1.0)*u)

    phi = np.empty((np.count_nonzero(selected), 3, 3, 3), dtype=np.float64)
    for direction in range(3):
        for a in range(3):
            for b in range(a, 3):
                name = f"ref_gh_Phi{direction + 1}{a + 1}{b + 1}"
                values = variable(state, name)[selected]
                phi[:, direction, a, b] = values
                phi[:, direction, b, a] = values
    displacement = np.stack((xx[selected], yy[selected], zz[selected]), axis=1)
    contraction = np.einsum(
        "nk,nij,nkij->n", displacement, spatial_inverse, phi, optimize=True)
    epsilon = -reference_l*contraction/6.0
    q_reference = q_analytic + (q - 1.0)*rho*u_r
    q_local = q_reference + epsilon
    weights = (2.0*h/rho)**3
    weight_sum = float(np.sum(weights))

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(weights*values)/weight_sum)

    q_est = weighted_mean(q_local)
    q_analytic_mean = weighted_mean(q_analytic)
    epsilon_mean = weighted_mean(epsilon)
    return {
        "shell_valid": bool(
            np.isfinite(q_local).all() and np.isfinite(q_analytic).all()),
        "cell_count": int(q_local.size),
        "effective_sample_size": float(weight_sum*weight_sum/np.sum(weights*weights)),
        "q_est": q_est,
        "q_analytic": q_analytic_mean,
        "q_est_minus_q_analytic": q_est - q_analytic_mean,
        "q_variance": weighted_mean(q_local*q_local) - q_est*q_est,
        "epsilon_G_mean": epsilon_mean,
        "epsilon_G_variance": weighted_mean(epsilon*epsilon) - epsilon_mean*epsilon_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--ref-history", type=Path, required=True)
    parser.add_argument("--user-history", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = read_cbin(args.state)
    constraints = read_cbin(args.constraints)
    if state["time"] != constraints["time"]:
        raise ValueError("state/constraint snapshot times differ")
    time = float(state["time"])
    ref = closest(read_rows(args.ref_history, REF_COLUMNS), time)
    user = closest(read_rows(args.user_history, USER_COLUMNS), time)
    metric = relative_metric(state)
    xx, yy, zz, h = coordinates(state)
    clear = (np.abs(xx) > 3.0*h) | (np.abs(yy) > 3.0*h) | (np.abs(zz) > 3.0*h)
    geometry = relative_geometry_health(metric)
    native = native_constraint_health(constraints, clear)
    q_offline = offline_q_health(
        state, metric, xx, yy, zz, h, user["q"], args.source_root)
    physical = {
        "metric_error_RMS": math.sqrt(
            ref["physical-g-error-L2sq"]/ref["Volume"]),
        "lapse_error_RMS": math.sqrt(
            ref["physical-alpha-error-L2sq"]/ref["Volume"]),
        "shift_error_RMS": math.sqrt(
            ref["physical-beta-error-L2sq"]/ref["Volume"]),
        "physical_lapse_min": -ref["minus-alpha-min"],
        "physical_lapse_max": ref["alpha-max"],
    }
    controller = {
        name: user[name] for name in (
            "q", "q-dot", "q-ddot", "q-est", "q-analytic",
            "qest-minus-analytic", "q-variance", "q-effective-samples",
            "q-cells", "epsilon-G-mean", "epsilon-G-variance",
            "q-shell-valid", "q-generation", "q-frozen")
    }
    finite = all(math.isfinite(value) for value in native.values())
    catastrophic = any(
        native[f"{name}_RMS"] > CATASTROPHIC_RMS
        or native[f"{name}_Linf"] > CATASTROPHIC_LINF
        for name in ("GH", "reduction", "curl"))
    checks = {
        "finite_state_and_telemetry": bool(
            geometry["finite"] and finite and ref["bad-state"] == 0.0),
        "relative_geometry_SPD_and_Lorentzian": bool(geometry["signature_valid"]),
        "q_within_hard_bounds": Q_MIN <= user["q"] <= Q_MAX,
        "qdot_within_hard_bound": abs(user["q-dot"]) <= QDOT_MAX,
        "q_shell_valid": bool(q_offline["shell_valid"]),
        "below_predeclared_catastrophic_constraint_threshold": not catastrophic,
    }
    result = {
        "schema": "ref-gh-single-puncture-health-v1",
        "time": time,
        "cycle": int(state["cycle"]),
        "h_over_M": h,
        "puncture_stencil_radius_cells": 3,
        "constraint_thresholds": {
            "RMS": CATASTROPHIC_RMS, "Linf": CATASTROPHIC_LINF},
        "geometry": geometry,
        "physical": physical,
        "native_constraints": native,
        "controller_history": controller,
        "q_offline": q_offline,
        "checks": checks,
        "pass": all(checks.values()),
        "inputs": {
            "state": str(args.state), "constraints": str(args.constraints),
            "ref_history": str(args.ref_history),
            "user_history": str(args.user_history),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
