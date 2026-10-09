"""Saved algebra and source/input review only; no spectrum or propagation."""
import ast
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.linalg import solve_triangular


HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
SOURCE = R / "continuum/finite-rb-limited-matrix-growth-pade13-20261009"
PINS = {
    8: "a7a08c5387fec735840e8c76db82624f75518403a96f23782d054920cc26472a",
    12: "d807cb2e76583ff9c2c367dd71ccc6a5f39f3cb7e3048a9bc36d0bb178419d2a",
    16: "ecb441e376637cd289b17a513fa07735781385ca062f4e08c4a1a8e582fb8414",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(a):
    return math.hypot(*(float(v) for v in a.real.flat), *(float(v) for v in a.imag.flat))


def difference(a, b):
    return norm(a - b) / max(1.0, norm(a), norm(b))


def main():
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    analyzer = SOURCE / "analyze_growth.py"
    assert sha(analyzer) == "cf51853026bd8647ff497559f2b741db17a8cb4c7653fa164f9096523099f2a0"
    tree = ast.parse(analyzer.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("mm", "energy_transform")]
    namespace = {"np": np, "solve_triangular": solve_triangular}
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "two-reviewed-helper-ASTs", "exec"), namespace)
    mm, transform = namespace["mm"], namespace["energy_transform"]
    reviews, all_pins = [], {}
    for N, expected in PINS.items():
        run = SOURCE / ("J0-N{}-rb98-growth001".format(N))
        rp = run / "receipt.json"
        assert sha(rp) == expected
        data = json.loads(rp.read_text())
        assert data["passed_finite_ODE_numerical_checks"] is True and data["error"] is None
        assert (data["J"], data["N"], data["rb"]) == (0, N, .98)
        assert data["input_pins_unchanged"] is True
        assert data["inputs_before"] == data["inputs_after"]
        for p, expected_input in data["inputs_before"].items():
            assert sha(p) == expected_input, p
            all_pins[p] = expected_input
        all_pins[str(rp)] = expected
        assert data["original_SciPy_expm_attempt_remains_failed"] is True
        assert data["both_original_FD_attempts_remain_failed"] is True
        assert data["general_nongauge_continuum_comparator_unresolved"] is True
        assert data["original_full_projection_defect_gate_passed"] is False
        assert (SOURCE / (run.name + ".stderr")).read_bytes() == b""
        payload = run / "growth-payload.npz"
        assert sha(payload) == data["payload_sha256"]
        all_pins[str(payload)] = sha(payload)
        matrix = Path(data["command"][data["command"].index("--matrix") + 1])
        if not matrix.is_absolute():
            matrix = R.parent / matrix
        # CLI uses repository-relative paths; R is build-layer-research.
        if not matrix.exists():
            matrix = Path(data["command"][data["command"].index("--matrix") + 1]).resolve()
        assert data["inputs_before"][str(matrix.resolve())] == sha(matrix)
        with np.load(matrix, allow_pickle=False) as saved:
            E, J = saved["E"].copy(), saved["Jbulk"] + saved["Jsat"]
        L = np.linalg.cholesky(E)
        A = transform(L, J)
        with np.load(payload, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        assert all(np.all(np.isfinite(v)) for v in arrays.values())
        dim = 8 * N
        eig, V = arrays["eigenvalues"], arrays["energy_eigenvectors"]
        assert eig.shape == (dim,) and V.shape == (dim, dim)
        residual = mm(A, V) - V * eig[None, :]
        residual_saved_difference = difference(residual, arrays["eigenvector_residuals"])
        assert residual_saved_difference <= 2e-9
        absolute = np.array([norm(residual[:, column]) for column in range(dim)])
        vnorm = np.array([norm(V[:, column]) for column in range(dim)])
        denominator = (data["generator_energy_norm_2"] + np.abs(eig)) * vnorm
        backward = absolute / denominator
        assert abs(float(np.max(absolute)) - data["max_eigenvector_absolute_residual"]) <= 2e-9
        assert abs(float(np.max(backward)) - data["max_eigenvector_relative_backward_residual"]) <= 2e-9
        assert float(np.max(eig.real)) == data["finite_matrix_spectral_abscissa"]
        order = np.lexsort((-eig.imag, -eig.real))
        selected = np.array([int(i) for i in order if eig[i].imag >= 0][:4])
        assert np.array_equal(selected, arrays["selected_indices"])
        actual_Jv_difference = difference(mm(J, arrays["selected_modal_modes"]), arrays["selected_actual_Jv"])
        lift_difference = difference(mm(L.T, arrays["selected_modal_modes"]), V[:, selected])
        assert max(actual_Jv_difference, lift_difference) <= 2e-9
        X = arrays["physical_seed_modal"]
        Z = mm(L.T, X)
        initial = np.array([norm(Z[:, column]) for column in range(8)])
        assert all(v > 0 for v in initial)
        assert len(data["seeds"]) == 8
        for column, seed in enumerate(data["seeds"]):
            assert abs(initial[column] - seed["initial_energy_norm"]) <= 2e-9
            assert seed["amplitude"] == (0.1 if seed["field"] == "alpha" else 0.02 if seed["field"] == "beta" else 0.01)
        assert np.array_equal(arrays["propagation_times"], [0., .25, .5, 1., 2., 4., 6.])
        assert arrays["seed_states_modal"].shape == (7, dim, 8)
        seed_checks = []
        for saved_state, record in zip(arrays["seed_states_modal"], data["propagation"]):
            energy_state = mm(L.T, saved_state)
            recorded_norms = np.array(record["seed_energy_norms"])
            reconstructed = np.array([norm(energy_state[:, column]) for column in range(8)])
            amplifications = reconstructed / initial
            row = {"time": record["time"], "norm_difference": difference(reconstructed, recorded_norms),
                   "amplification_difference": difference(amplifications, np.array(record["seed_energy_amplifications"])),
                   "half_time_consistency": record["half_time_product_check"]}
            assert max(row["norm_difference"], row["amplification_difference"]) <= 2e-9
            seed_checks.append(row)
        evaluations = data["exponential_evaluations"]
        assert all(v["status"] == "passed_rational_solve_check" and
                   v["rational_solve_residual"] <= 2e-9 for v in evaluations)
        reviews.append({"N": N, "receipt_sha256": sha(rp), "payload_sha256": sha(payload),
            "source_input_pin_count": len(data["inputs_before"]),
            "spectral_abscissa_saved": data["finite_matrix_spectral_abscissa"],
            "selected_saved_eigenvalues": [[float(v.real), float(v.imag)] for v in eig[selected]],
            "recorded_generator_energy_norm_2": data["generator_energy_norm_2"],
            "absolute_eigenvector_residual_max_readback": float(np.max(absolute)),
            "relative_backward_residual_max_readback": float(np.max(backward)),
            "eigenvector_residual_payload_difference": residual_saved_difference,
            "actual_Jv_difference": actual_Jv_difference, "selected_lift_difference": lift_difference,
            "seed_state_and_amplification_readback": seed_checks,
            "t6_operator_norm_saved_not_recomputed": data["propagation"][-1]["energy_operator_norm"],
            "t6_amplifications_saved": data["propagation"][-1]["seed_energy_amplifications"],
            "RK3_saved": data["RK3"], "exponential_call_count": len(evaluations)})
    hp_index = R / "continuum/immutable-J0-N8-t6-expm-high-precision-20261009/index.json"
    assert sha(hp_index) == "1884f8808e2f189fceb0476a61ea1291458a3e02f8863ef6089755af0226f386"
    hpi = json.loads(hp_index.read_text())
    for entry in hpi["files"]:
        assert sha(hp_index.parent / entry["path"]) == entry["sha256"]
    hp = json.loads((hp_index.parent / "reference-001/results/receipt.json").read_text())
    assert hp["status"] == "PASS_finite_matrix_accuracy_only"
    assert hp["inputs_before"] == hp["inputs_after"]
    assert all(Decimal(v) <= Decimal("1e-80") for v in hp["oracle_100_130_relative_peak"])
    assert hp["full_propagator_comparison"]["scaled_frobenius"] <= 2e-7
    assert hp["saved_modal_seed_comparison"]["scaled_frobenius"] <= 2e-7
    all_pins[str(hp_index)] = sha(hp_index)
    result = {"status": "PASS_source_pins_and_saved_algebra_review", "review_source_sha256": sha(__file__),
        "cases": reviews, "high_precision_index_sha256": sha(hp_index),
        "high_precision_scope": "Independent100/130-digit Taylor t6 accuracy check only for N8; not a general nonnormal bound or N12/N16 oracle.",
        "high_precision_full_propagator_scaled_error": hp["full_propagator_comparison"]["scaled_frobenius"],
        "high_precision_saved_modal_scaled_error": hp["saved_modal_seed_comparison"]["scaled_frobenius"],
        "input_sha256": all_pins,
        "generator_eigensolves_or_exponentials_or_propagation_or_PDE_queries": False,
        "source_executed": "Only exact mm/energy_transform function ASTs; no analyze() import/call.",
        "scope_findings": [
            "Saved finite eigenvectors and their absolute/backward residuals are accurately reported; no eigenvalue perturbation bound or continuum eigenvalue follows.",
            "The relative residual denominator uses the recorded finite generator norm; absolute residuals remain explicit.",
            "Amplifications are sqrt(X(t)^T E X(t))/sqrt(X(0)^T E X(0)) for fixed physical modal seeds, not primitive field amplitudes.",
            "Seeds are N-dependent interpolants of fixed scalar envelopes; no pure spatial order/convergence follows from the mixed degree results.",
            "Bitwise-zero half-time products reuse nested fixed scaling/squaring and are internal consistency only.",
            "Rational solve residuals are not exponential forward error certificates; independent high-precision evidence covers N8 at t6 only.",
            "Finite rb=.98, J0, adjoint incoming SAT and unresolved nongauge continuum comparator prevent PDE/CPBC/exact-scri/native/nonlinear/BH acceptance.",
            "All previous failed ordinary-FD/SciPy classifications remain unchanged; the successful finite-matrix computation is separately sourced."
        ], "corrections": []}
    (HERE / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "receipt_sha256": sha(HERE / "receipt.json")}))


if __name__ == "__main__":
    main()
