"""One separately authorized actual t6 finite-matrix accuracy reference."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import warnings

import mpmath as mp
import numpy as np

from pade13_einsum import expm_pade13, mm


HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mp_matrix(a):
    return mp.matrix([[mp.mpf(float(v)) for v in row] for row in a])


def as_strings(a, digits):
    return [[mp.nstr(a[i, j], digits) for j in range(a.cols)] for i in range(a.rows)]


def as_float(a):
    return np.array([[float(a[i, j]) for j in range(a.cols)] for i in range(a.rows)])


def compare(a, b):
    delta = a - b
    absolute = math.hypot(*(float(v) for v in delta.flat))
    scale = max(1.0, math.hypot(*(float(v) for v in a.flat)),
                math.hypot(*(float(v) for v in b.flat)))
    peak = max((abs(float(v)) for v in delta.flat), default=0.0)
    peakscale = max(1.0, max((abs(float(v)) for v in b.flat), default=0.0))
    return {"absolute_frobenius": absolute, "scaled_frobenius": absolute / scale,
            "absolute_peak": peak, "scaled_peak": peak / peakscale}


def mp_relative(a, b):
    delta = max((abs(a[i, j] - b[i, j]) for i in range(a.rows)
                 for j in range(a.cols)), default=mp.mpf(0))
    scale = max(mp.mpf(1), max((abs(b[i, j]) for i in range(b.rows)
                              for j in range(b.cols)), default=mp.mpf(0)))
    return delta / scale


def triangular_back_substitution(lower, state):
    n, count = state.rows, state.cols
    out = mp.matrix(n, count)
    for column in range(count):
        for i in range(n - 1, -1, -1):
            coupling = mp.fsum(lower[k, i] * out[k, column] for k in range(i + 1, n))
            out[i, column] = (state[i, column] - coupling) / lower[i, i]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    start = time.monotonic()
    prepared = HERE / "prepared-001/receipt.json"
    inputs = HERE / "prepared-001/inputs.npz"
    binding = json.loads(prepared.read_text())
    assert binding["status"] == "PASS_input_binding_only"
    assert sha(inputs) == binding["inputs_sha256"]
    pins = {str(inputs): sha(inputs), str(prepared): sha(prepared),
            str(HERE / "PLAN.md"): sha(HERE / "PLAN.md"),
            str(HERE / "pade13_einsum.py"): sha(HERE / "pade13_einsum.py"),
            str(Path(__file__)): sha(__file__)}
    assert pins[str(HERE / "pade13_einsum.py")] == "0654c3ba8884dc467aa1c4665fc4758a3abddf58189e2702f7e8cfc90d2e79b7"
    with np.load(inputs, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    A, L, Z = values["A"], values["L"], values["initial_energy_seed"]
    assert A.shape == L.shape == (64, 64) and Z.shape == (64, 8)
    assert np.array_equal(values["floating_argument"], 6.0 * A)
    result = {"status": "RUNNING", "time": 6, "J": 0, "N": 8, "rb": .98,
              "scope": "one actual finite-matrix accuracy check; no PDE/native/nonlinear/scri/BH acceptance",
              "inputs_before": pins, "numpy_error_mode": np.geterr(),
              "python": sys.version, "executable": sys.executable,
              "numpy": np.__version__, "mpmath": mp.__version__,
              "mpmath_expm_method": "taylor", "mpmath_expm_source": {
                  "path": inspect.getsourcefile(mp.expm),
                  "sha256": sha(inspect.getsourcefile(mp.expm))},
              "thresholds": {"oracle_agreement": "1e-80", "float_scaled_comparison": 2e-7},
              "generator_eigensolve": False, "PDE_queries": False,
              "both_ordinary_FD_attempts_and_original_Scipy_attempt_remain_failed": True,
              "general_nongauge_continuum_comparator_unresolved": True}
    (out / "launch.json").write_text(json.dumps(result, indent=2) + "\n")
    oracles = {}
    for digits in (100, 130):
        (out / "active-stage.json").write_text(json.dumps({"stage": "mpmath_taylor_expm", "digits": digits}) + "\n")
        stage_start = time.monotonic()
        with mp.workdps(digits):
            # Exact binary64 A entries multiplied by exact integer6 here.
            argument = 6 * mp_matrix(A)
            G = mp.expm(argument, method="taylor")
            energy = G * mp_matrix(Z)
            modal = triangular_back_substitution(mp_matrix(L), energy)
            payload = {"digits": digits, "time": 6,
                       "argument_definition": "exact6 times exact binary64 A entries",
                       "energy_propagator": as_strings(G, digits),
                       "seed_energy_states": as_strings(energy, digits),
                       "seed_modal_states": as_strings(modal, digits)}
            (out / ("oracle-" + str(digits) + ".json")).write_text(json.dumps(payload, indent=2) + "\n")
        oracles[digits] = (G, energy, modal)
        result["precision_" + str(digits) + "_seconds"] = time.monotonic() - stage_start
        print(json.dumps({"finished_precision": digits,
                          "seconds": result["precision_" + str(digits) + "_seconds"]}), flush=True)
    with mp.workdps(150):
        agreement = [mp_relative(a, b) for a, b in zip(oracles[100], oracles[130])]
        assert all(v <= mp.mpf("1e-80") for v in agreement)
        rounding_delta = mp_matrix(values["floating_argument"]) - 6 * mp_matrix(A)
        result["float_argument_rounding_peak"] = mp.nstr(max(abs(v) for v in rounding_delta), 40)
        result["oracle_100_130_relative_peak"] = [mp.nstr(v, 40) for v in agreement]
    reference, energy_reference, modal_reference = map(as_float, oracles[130])
    (out / "active-stage.json").write_text(json.dumps({"stage": "floating_helper_and_saved_state_comparison"}) + "\n")
    floating, info = expm_pade13(values["floating_argument"], return_info=True)
    result["floating_helper_info"] = info
    result["full_propagator_comparison"] = compare(floating, reference)
    energy_float = mm(floating, Z)
    result["energy_seed_comparison"] = compare(energy_float, energy_reference)
    result["saved_modal_seed_comparison"] = compare(values["saved_t6_modal"], modal_reference)
    result["per_column"] = []
    for column in range(8):
        result["per_column"].append({"column": column,
            "energy": compare(energy_float[:, column], energy_reference[:, column]),
            "saved_modal": compare(values["saved_t6_modal"][:, column], modal_reference[:, column])})
    checks = [result[key] for key in ("full_propagator_comparison", "energy_seed_comparison", "saved_modal_seed_comparison")]
    checks += [v[key] for v in result["per_column"] for key in ("energy", "saved_modal")]
    assert all(v["scaled_frobenius"] <= 2e-7 and v["scaled_peak"] <= 2e-7 for v in checks)
    np.savez(out / "comparisons.npz", high_precision_propagator_rounded=reference,
             high_precision_energy_seeds_rounded=energy_reference,
             high_precision_modal_seeds_rounded=modal_reference,
             floating_helper_propagator=floating, floating_energy_seeds=energy_float)
    result.update(status="PASS_finite_matrix_accuracy_only", seconds=time.monotonic() - start,
                  inputs_after={p: sha(p) for p in pins},
                  output_sha256={name: sha(out / name) for name in
                                 ("oracle-100.json", "oracle-130.json", "comparisons.npz")})
    assert result["inputs_after"] == pins
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "seconds": result["seconds"],
                      "full_propagator_comparison": result["full_propagator_comparison"],
                      "saved_modal_seed_comparison": result["saved_modal_seed_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
