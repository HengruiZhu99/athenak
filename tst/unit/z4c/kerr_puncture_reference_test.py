#!/usr/bin/env python3
"""Compile and verify the Kerr-puncture evaluator against an independent oracle."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PACKED = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def check_python_subprocess_contract() -> None:
    """Reject bare nested Python commands and verify interpreter identity."""
    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "subprocess"):
            continue
        command = node.args[0]
        if not isinstance(command, (ast.List, ast.Tuple)) or not command.elts:
            continue
        executable = command.elts[0]
        if (isinstance(executable, ast.Constant)
                and isinstance(executable.value, str)
                and Path(executable.value).name in {"python", "python3"}):
            raise AssertionError(
                f"line {node.lineno}: nested bare Python command bypasses "
                "the CTest interpreter")

    probe = subprocess.run(
        [sys.executable, "-c",
         "import json,sys; print(json.dumps({"
         "'executable': sys.executable, 'version': list(sys.version_info[:3])}))"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    identity = json.loads(probe.stdout)
    expected_executable = Path(sys.executable).resolve(strict=True)
    actual_executable = Path(identity["executable"]).resolve(strict=True)
    if actual_executable != expected_executable:
        raise AssertionError(
            "nested Python interpreter changed: "
            f"{actual_executable} != {expected_executable}")
    if identity["version"] != list(sys.version_info[:3]):
        raise AssertionError(
            f"nested Python version changed: {identity['version']} != "
            f"{list(sys.version_info[:3])}")


def unpack(values: list[float]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for value, (i, j) in zip(values, PACKED):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def parse_output(output: str) -> dict[str, object]:
    fields = output.split()
    if len(fields) != 39:
        raise AssertionError(f"expected 39 driver fields, received {len(fields)}")
    values = [float(value) for value in fields]
    return {
        "valid": bool(int(values[0])),
        "physical_adm_available": bool(int(values[1])),
        "at_puncture": bool(int(values[2])),
        "isotropic_radius": values[3],
        "boyer_lindquist_radius": values[4],
        "r_plus": values[5],
        "r_minus": values[6],
        "horizon_radius": values[7],
        "lapse": values[8],
        "shift": values[9:12],
        "psi4": values[12],
        "spatial_metric": values[13:19],
        "extrinsic_curvature": values[19:25],
        "conformal_chi": values[25],
        "conformal_metric": values[26:32],
        "trace_extrinsic_curvature": values[32],
        "conformal_tracefree_curvature": values[33:39],
    }


def run_point(driver: Path, specification: dict[str, object]) -> dict[str, object]:
    command = [str(driver), str(specification["map"]),
               str(specification["gauge"]), str(specification["M"]),
               str(specification["chi"]), str(specification["z_h"])]
    command.extend(str(value) for value in specification["coordinates"])
    result = subprocess.run(command, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return parse_output(result.stdout)


def assert_close(actual: float, expected: float, label: str,
                 relative: float = 3.0e-12, absolute: float = 3.0e-13) -> None:
    if not math.isfinite(actual):
        raise AssertionError(f"{label}: nonfinite actual value {actual}")
    tolerance = absolute + relative * abs(expected)
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: actual={actual:.17e} expected={expected:.17e} "
            f"tolerance={tolerance:.3e}")


def compare_reference(actual: dict[str, object], expected: dict[str, object],
                      case_id: str) -> None:
    for key in ("valid", "physical_adm_available", "at_puncture"):
        if actual[key] != expected[key]:
            raise AssertionError(f"{case_id}.{key}: {actual[key]} != {expected[key]}")
    scalar_keys = (
        "isotropic_radius", "boyer_lindquist_radius", "r_plus", "r_minus",
        "horizon_radius", "lapse", "psi4", "conformal_chi",
        "trace_extrinsic_curvature",
    )
    vector_keys = (
        "shift", "spatial_metric", "extrinsic_curvature", "conformal_metric",
        "conformal_tracefree_curvature",
    )
    for key in scalar_keys:
        assert_close(float(actual[key]), float(expected[key]), f"{case_id}.{key}")
    for key in vector_keys:
        for index, (actual_value, expected_value) in enumerate(
                zip(actual[key], expected[key])):
            assert_close(float(actual_value), float(expected_value),
                         f"{case_id}.{key}[{index}]")


def check_scientific_invariants(records: dict[str, dict[str, object]]) -> None:
    schwarzschild = records["schwarzschild"]
    specification = schwarzschild["input"]
    actual = schwarzschild["actual"]
    coordinates = np.asarray(specification["coordinates"], dtype=float)
    radius = np.linalg.norm(coordinates)
    psi4 = (1.0 + float(specification["M"]) / (2.0 * radius)) ** 4
    metric = unpack(actual["spatial_metric"])
    np.testing.assert_allclose(metric, psi4 * np.identity(3), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(actual["extrinsic_curvature"], 0.0, atol=3e-14)
    assert_close(actual["lapse"], psi4 ** -0.5, "Schwarzschild pre-collapsed lapse")

    plus = records["high_spin_plus"]["actual"]
    minus = records["high_spin_minus"]["actual"]
    for key in ("spatial_metric", "conformal_metric"):
        np.testing.assert_allclose(plus[key], minus[key], rtol=2e-13, atol=2e-13)
    for key in ("extrinsic_curvature", "conformal_tracefree_curvature", "shift"):
        np.testing.assert_allclose(plus[key], -np.asarray(minus[key]),
                                   rtol=2e-13, atol=2e-13)
    assert_close(plus["lapse"], minus["lapse"], "spin-sign lapse")

    positive = records["cartoon_positive_rho"]["actual"]
    negative = records["cartoon_negative_rho"]["actual"]
    parity = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    vector_parity = np.asarray([-1.0, 1.0, -1.0])
    for key in ("spatial_metric", "extrinsic_curvature", "conformal_metric",
                "conformal_tracefree_curvature"):
        np.testing.assert_allclose(negative[key], parity * np.asarray(positive[key]),
                                   rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(negative["shift"],
                               vector_parity * np.asarray(positive["shift"]),
                               rtol=3e-13, atol=3e-13)

    cartesian = records["component_map_cartesian"]["actual"]
    permutation = np.asarray([0, 2, 1])
    for key in ("spatial_metric", "extrinsic_curvature", "conformal_metric",
                "conformal_tracefree_curvature"):
        source = unpack(cartesian[key])
        mapped = source[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(unpack(positive[key]), mapped,
                                   rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(positive["shift"],
                               np.asarray(cartesian["shift"])[permutation],
                               rtol=3e-13, atol=3e-13)

    horizon = records["horizon_equator"]
    assert_close(float(horizon["input"]["coordinates"][0]),
                 horizon["actual"]["horizon_radius"], "horizon r_plus/4",
                 relative=3e-15, absolute=3e-15)
    assert_close(records["horizon_stationary"]["actual"]["lapse"], 0.0,
                 "stationary horizon lapse", relative=0.0, absolute=1.0e-15)

    for name, expected_lapse in (("puncture_precollapsed", 0.0),
                                 ("puncture_stationary", 1.0)):
        puncture = records[name]["actual"]
        if puncture["physical_adm_available"] or not puncture["at_puncture"]:
            raise AssertionError(f"{name}: puncture availability contract violated")
        assert_close(puncture["lapse"], expected_lapse, f"{name}.lapse")
        assert_close(puncture["conformal_chi"], 0.0, f"{name}.chi")
        np.testing.assert_allclose(unpack(puncture["conformal_metric"]),
                                   np.identity(3), rtol=0, atol=0)

    asymptotic = records["asymptotic"]["actual"]
    np.testing.assert_allclose(unpack(asymptotic["spatial_metric"]), np.identity(3),
                               rtol=0.025, atol=0.025)
    if abs(asymptotic["lapse"] - 1.0) > 0.025:
        raise AssertionError("asymptotic lapse is not approaching one")

    near = records["near_puncture_limit"]["actual"]
    if near["conformal_chi"] <= 0.0 or near["conformal_chi"] > 1.0e-12:
        raise AssertionError("near-puncture chi does not approach its r^4 limit")
    np.testing.assert_allclose(unpack(near["conformal_metric"]), np.identity(3),
                               rtol=2.0e-3, atol=2.0e-3)
    if np.linalg.norm(near["conformal_tracefree_curvature"]) > 1.0e-8:
        raise AssertionError("near-puncture conformal A does not approach zero")


def check_invalid_domain(driver: Path) -> None:
    base = {"map": "cartesian", "gauge": "precollapsed", "M": "1",
            "chi": "0", "z_h": "0", "coordinates": ["1", "2", "3"]}
    for label, replacement in (
            ("zero_mass", {"M": "0"}),
            ("extremal_spin", {"chi": "1"}),
            ("negative_extremal_spin", {"chi": "-1"}),
            ("nonfinite_center", {"z_h": "inf"})):
        specification = dict(base)
        specification.update(replacement)
        actual = run_point(driver, specification)
        if actual["valid"]:
            raise AssertionError(f"{label}: invalid parameter domain accepted")
        for key, value in actual.items():
            if key in ("valid", "physical_adm_available", "at_puncture"):
                continue
            values = value if isinstance(value, list) else [value]
            if not all(math.isfinite(item) for item in values):
                raise AssertionError(f"{label}.{key}: invalid result is nonfinite")


def metric_and_curvature(driver: Path, point: np.ndarray,
                         mass: float, chi: float) -> tuple[np.ndarray, np.ndarray]:
    result = run_point(driver, {
        "map": "cartesian", "gauge": "precollapsed", "M": mass,
        "chi": chi, "z_h": 0.0, "coordinates": point.tolist(),
    })
    return unpack(result["spatial_metric"]), unpack(result["extrinsic_curvature"])


def vacuum_constraint_residual(driver: Path) -> tuple[float, float]:
    point = np.asarray([2.3, -1.7, 1.1], dtype=float)
    mass = 1.0
    chi = 0.8
    step = 2.0e-4
    metric0, curvature0 = metric_and_curvature(driver, point, mass, chi)
    metric_inv = np.linalg.inv(metric0)
    dmetric = np.zeros((3, 3, 3))
    dcurvature = np.zeros((3, 3, 3))
    ddmetric = np.zeros((3, 3, 3, 3))
    for direction in range(3):
        offset = np.zeros(3)
        offset[direction] = step
        metric_plus, curvature_plus = metric_and_curvature(
            driver, point + offset, mass, chi)
        metric_minus, curvature_minus = metric_and_curvature(
            driver, point - offset, mass, chi)
        dmetric[direction] = (metric_plus - metric_minus) / (2.0 * step)
        dcurvature[direction] = (curvature_plus - curvature_minus) / (2.0 * step)
        ddmetric[direction, direction] = (
            metric_plus - 2.0 * metric0 + metric_minus) / (step * step)
        for other in range(direction):
            offset_other = np.zeros(3)
            offset_other[other] = step
            metric_pp, _ = metric_and_curvature(
                driver, point + offset + offset_other, mass, chi)
            metric_pm, _ = metric_and_curvature(
                driver, point + offset - offset_other, mass, chi)
            metric_mp, _ = metric_and_curvature(
                driver, point - offset + offset_other, mass, chi)
            metric_mm, _ = metric_and_curvature(
                driver, point - offset - offset_other, mass, chi)
            mixed = (metric_pp - metric_pm - metric_mp + metric_mm) / (4.0 * step ** 2)
            ddmetric[direction, other] = mixed
            ddmetric[other, direction] = mixed

    dmetric_inv = np.zeros((3, 3, 3))
    for direction in range(3):
        dmetric_inv[direction] = -metric_inv @ dmetric[direction] @ metric_inv
    christoffel = np.zeros((3, 3, 3))
    dchristoffel = np.zeros((3, 3, 3, 3))
    for upper in range(3):
        for first in range(3):
            for second in range(3):
                for lower in range(3):
                    combination = (dmetric[first, lower, second]
                                   + dmetric[second, lower, first]
                                   - dmetric[lower, first, second])
                    christoffel[upper, first, second] += (
                        0.5 * metric_inv[upper, lower] * combination)
                    for derivative in range(3):
                        derivative_combination = (
                            ddmetric[derivative, first, lower, second]
                            + ddmetric[derivative, second, lower, first]
                            - ddmetric[derivative, lower, first, second])
                        dchristoffel[derivative, upper, first, second] += 0.5 * (
                            dmetric_inv[derivative, upper, lower] * combination
                            + metric_inv[upper, lower] * derivative_combination)
    ricci = np.zeros((3, 3))
    for first in range(3):
        for second in range(3):
            for k in range(3):
                ricci[first, second] += (
                    dchristoffel[k, k, first, second]
                    - dchristoffel[second, k, first, k])
                for ell in range(3):
                    ricci[first, second] += (
                        christoffel[k, first, second] * christoffel[ell, k, ell]
                        - christoffel[ell, first, k] * christoffel[k, second, ell])
    scalar_curvature = np.einsum("ij,ij", metric_inv, ricci)
    trace_k = np.einsum("ij,ij", metric_inv, curvature0)
    k_squared = np.einsum("ia,jb,ij,ab", metric_inv, metric_inv,
                          curvature0, curvature0)
    hamiltonian = scalar_curvature + trace_k ** 2 - k_squared

    mixed_k = metric_inv @ curvature0
    d_mixed_k = np.zeros((3, 3, 3))
    d_trace_k = np.zeros(3)
    for derivative in range(3):
        d_mixed_k[derivative] = (
            dmetric_inv[derivative] @ curvature0
            + metric_inv @ dcurvature[derivative])
        d_trace_k[derivative] = (
            np.einsum("ij,ij", dmetric_inv[derivative], curvature0)
            + np.einsum("ij,ij", metric_inv, dcurvature[derivative]))
    momentum = np.zeros(3)
    for lower in range(3):
        for j in range(3):
            momentum[lower] += d_mixed_k[j, j, lower]
            for ell in range(3):
                momentum[lower] += (
                    christoffel[j, j, ell] * mixed_k[ell, lower]
                    - christoffel[ell, j, lower] * mixed_k[j, ell])
        momentum[lower] -= d_trace_k[lower]
    return abs(float(hamiltonian)), float(np.linalg.norm(momentum))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    check_python_subprocess_contract()
    reference = source / "tst/unit/z4c/kerr_puncture_reference.json"
    generator = source / "tst/unit/z4c/generate_kerr_puncture_reference.py"
    subprocess.run([sys.executable, str(generator), "--check", "--output",
                    str(reference)], check=True)
    payload = json.loads(reference.read_text(),
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValueError(f"nonfinite JSON token {value}")))
    if payload.get("schema") != "athenak_kerr_puncture_reference_v1":
        raise AssertionError("unexpected reference schema")
    with tempfile.TemporaryDirectory(prefix="athenak-kerr-puncture-") as directory:
        driver = Path(directory) / "kerr_puncture_point_driver"
        subprocess.run([
            "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{source / 'src'}",
            str(source / "tst/unit/z4c/kerr_puncture_point_driver.cpp"),
            "-o", str(driver),
        ], check=True)
        records: dict[str, dict[str, object]] = {}
        for record in payload["records"]:
            specification = record["input"]
            actual = run_point(driver, specification)
            compare_reference(actual, record["expected"], specification["id"])
            records[specification["id"]] = {"input": specification,
                                             "actual": actual}
        check_scientific_invariants(records)
        check_invalid_domain(driver)
        hamiltonian, momentum = vacuum_constraint_residual(driver)
        if hamiltonian > 2.0e-6 or momentum > 2.0e-7:
            raise AssertionError(
                "finite-difference vacuum constraint sample failed: "
                f"H={hamiltonian:.6e} M={momentum:.6e}")
        print("kerr_puncture_reference=PASS "
              f"records={len(records)} H={hamiltonian:.6e} M={momentum:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
