"""Synthetic-only independent checks; no scientific file is an input."""

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings

import mpmath as mp
import numpy as np

from pade13_einsum import COEFFICIENTS, THETA13, expm_pade13, mm


HERE = Path(__file__).resolve().parent
TOLERANCES = {"forward": 2e-11, "solve": 2e-14,
              "consistency": 5e-11, "oracle": "1e-85"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def error(a, b):
    return peak(a - b) / max(1.0, peak(b))


def mp_matrix(a):
    return mp.matrix([[mp.mpf(float(v)) for v in row] for row in a])


def mp_oracle(a, digits):
    with mp.workdps(digits):
        out = mp.expm(mp_matrix(a))
        return [[mp.nstr(out[i, j], digits) for j in range(a.shape[1])]
                for i in range(a.shape[0])]


def oracle_difference(left, right):
    with mp.workdps(140):
        lv = [mp.mpf(v) for row in left for v in row]
        rv = [mp.mpf(v) for row in right for v in row]
        return max((abs(a - b) for a, b in zip(lv, rv)), default=mp.mpf(0)) / max(
            mp.mpf(1), max((abs(v) for v in rv), default=mp.mpf(0)))


def exact_coefficient_gate():
    coeff = [Fraction(math.factorial(26 - k) * math.factorial(13),
                      math.factorial(26) * math.factorial(k)
                      * math.factorial(13 - k)) for k in range(14)]
    integer = [v / coeff[-1] for v in coeff]
    assert integer == list(COEFFICIENTS)
    for n in range(27):
        value = sum((Fraction((-1) ** k * COEFFICIENTS[k],
                              math.factorial(n - k))
                     for k in range(min(13, n) + 1)), Fraction(0))
        if n <= 13:
            value -= COEFFICIENTS[n]
        assert value == 0, (n, value)
    tree = ast.parse((HERE / "pade13_einsum.py").read_text())
    assert not any(isinstance(node, ast.MatMult) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("matmul", "dot") for node in calls)
    einsums = [node for node in calls if isinstance(node.func, ast.Attribute)
               and node.func.attr == "einsum"]
    assert len(einsums) == 1
    assert any(kw.arg == "optimize" and isinstance(kw.value, ast.Constant)
               and kw.value.value is False for kw in einsums[0].keywords)
    imports = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 2
    return {"factorial_coefficients_exact": True,
            "taylor_matching_exact_coefficients": 27,
            "literal_unoptimized_einsum_only": True}


def cases():
    rows = []

    def add(name, a, closed=None, high_precision=True):
        rows.append({"name": name, "a": a, "closed": closed,
                     "high_precision": high_precision})

    add("empty", np.empty((0, 0)), np.empty((0, 0)), False)
    add("zero3", np.zeros((3, 3)), np.eye(3))
    add("diagonal_growth_decay", np.diag([-12.0, -0.5, 0.0, 0.75, 8.0]),
        np.diag(np.exp([-12.0, -0.5, 0.0, 0.75, 8.0])))
    n = np.zeros((4, 4))
    n[0, 1], n[1, 2], n[2, 3] = 3.0, -2.0, 5.0
    add("nilpotent4", n, np.eye(4) + n + mm(n, n) / 2
        + mm(mm(n, n), n) / 6)
    add("shifted_jordan4", n - 0.75 * np.eye(4),
        math.exp(-0.75) * (np.eye(4) + n + mm(n, n) / 2
                          + mm(mm(n, n), n) / 6))
    omega = 20.0
    add("rotation20", np.array([[0.0, -omega], [omega, 0.0]]),
        np.array([[math.cos(omega), -math.sin(omega)],
                  [math.sin(omega), math.cos(omega)]]))
    add("dense_fixed4", np.array([[0.5, 4.0, -3.0, 1.0],
                                  [-2.0, -1.0, 0.75, 2.0],
                                  [1.25, 0.0, -0.5, -4.0],
                                  [0.0, 1.0, 3.0, -0.25]]))
    s = np.array([[1.0, 8.0, -3.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
    d = np.diag([-1.0, 0.5, 2.0])
    a = np.linalg.solve(s.T, mm(s, d).T).T
    closed = np.linalg.solve(s.T, mm(s, np.diag(np.exp(np.diag(d)))).T).T
    add("nonnormal_exact_similarity", a, closed)
    lower = np.array([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0],
                      [-0.25, 0.75, 1.0]])
    d = np.diag([-2.0, 0.25, 1.5])
    j = np.linalg.solve(lower.T, mm(d, lower.T))
    closed = np.linalg.solve(lower.T, mm(np.diag(np.exp(np.diag(d))), lower.T))
    add("spd_energy_similarity_J", j, closed)
    transformed = np.linalg.solve(lower, mm(lower.T, j).T).T
    add("spd_energy_similarity_A", transformed, np.diag(np.exp(np.diag(d))))
    for factor in (1.0, 2.0, 32.0):
        boundary = factor * THETA13
        for suffix, v in (("below", np.nextafter(boundary, 0.0)),
                          ("at", boundary),
                          ("above", np.nextafter(boundary, math.inf))):
            # Nilpotent: norm can cross large scaling boundaries without
            # exponent overflow or any dependence on spectral conditioning.
            a = np.array([[0.0, v], [0.0, 0.0]])
            add("scale_{}_{}".format(int(factor), suffix), a, np.eye(2) + a)
    add("zero64", np.zeros((64, 64)), np.eye(64), False)
    add("identity64", np.eye(64), math.e * np.eye(64), False)
    diag = np.linspace(-3.0, 2.0, 64)
    add("diagonal64", np.diag(diag), np.diag(np.exp(diag)), False)
    a = np.zeros((64, 64))
    for i in range(63):
        a[i, i + 1] = 3.0
    closed = np.zeros((64, 64))
    for i in range(64):
        for k in range(64 - i):
            closed[i, i + k] = 3.0 ** k / math.factorial(k)
    add("nilpotent_jordan64", a, closed, False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    started = time.monotonic()
    source_pins = {name: sha(HERE / name) for name in
                   ("pade13_einsum.py", "check_synthetic.py", "PLAN.md")}
    result = {"status": "RUNNING", "scope": "synthetic only; no saved scientific operator",
              "source_sha256": source_pins, "thresholds": TOLERANCES,
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "numpy": np.__version__,
              "mpmath": mp.__version__, "numpy_error_mode": np.geterr(),
              "exact": exact_coefficient_gate(), "cases": []}
    arrays, oracles = {}, {}
    for case in cases():
        name, a = case["name"], case["a"]
        value, info = expm_pade13(a, return_info=True)
        assert np.all(np.isfinite(value))
        assert info["scaled_one_norm"] <= THETA13
        assert info["rational_solve_residual"] <= TOLERANCES["solve"]
        row = {"name": name, "shape": list(a.shape), "info": info}
        if case["closed"] is not None:
            row["closed_form_error"] = error(value, case["closed"])
            assert row["closed_form_error"] <= TOLERANCES["forward"]
            arrays[name + "_closed"] = case["closed"]
        if case["high_precision"]:
            first, second = mp_oracle(a, 100), mp_oracle(a, 130)
            difference = oracle_difference(first, second)
            assert difference <= mp.mpf(TOLERANCES["oracle"])
            reference = np.array(second, dtype=np.float64)
            row["mpmath_100_130_difference"] = mp.nstr(difference, 20)
            row["mpmath_forward_error"] = error(value, reference)
            assert row["mpmath_forward_error"] <= TOLERANCES["forward"]
            arrays[name + "_oracle"] = reference
            oracles[name] = {"digits100": first, "digits130": second}
        half = expm_pade13(0.5 * a)
        row["half_composition_error"] = error(mm(half, half), value)
        inverse = expm_pade13(-a)
        row["inverse_error"] = error(mm(value, inverse), np.eye(a.shape[0]))
        assert row["half_composition_error"] <= TOLERANCES["consistency"]
        assert row["inverse_error"] <= TOLERANCES["consistency"]
        arrays[name + "_argument"] = a
        arrays[name + "_exponential"] = value
        result["cases"].append(row)
    bad = [("nonsquare", np.zeros((2, 3)), ValueError),
           ("rankone", np.zeros(3), ValueError),
           ("complex", np.eye(2, dtype=complex), TypeError),
           ("nan", np.array([[math.nan]]), FloatingPointError),
           ("infinity", np.array([[math.inf]]), FloatingPointError)]
    result["rejections"] = []
    for name, a, expected in bad:
        try:
            expm_pade13(a)
        except expected as exc:
            result["rejections"].append({"name": name, "exception": type(exc).__name__})
        else:
            raise AssertionError("invalid input accepted: " + name)
    result.update(status="PASS", seconds=time.monotonic() - started,
                  case_count=len(result["cases"]),
                  oracle_case_count=len(oracles),
                  final_source_sha256={name: sha(HERE / name) for name in source_pins})
    assert result["source_sha256"] == result["final_source_sha256"]
    np.savez(out / "synthetic-arrays.npz", **arrays)
    (out / "high-precision-oracles.json").write_text(json.dumps(oracles, indent=2) + "\n")
    result["output_sha256"] = {name: sha(out / name) for name in
                               ("synthetic-arrays.npz", "high-precision-oracles.json")}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cases": result["case_count"],
                      "oracle_cases": result["oracle_case_count"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
