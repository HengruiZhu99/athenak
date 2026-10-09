"""Bind exact existing binary64 energy-generator input without propagation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.linalg import solve_triangular


HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
OP = R / "boundary/total-j-finite-rb-control-20261009/J0-segmented-Q64-sector-readback001/operator.npz"
GROWTH = R / "continuum/finite-rb-limited-matrix-growth-pade13-20261009"
RUN = GROWTH / "J0-N8-rb98-growth001"
ISOLATED = R / "continuum/finite-rb-expm-diagnosis-20261009/isolated-scipy-001/isolated-input.npz"
PINS = {
    str(OP): "8b4b9a5b33151d86359aa9e35f0ae44dc436ef4ab2d6c17796e77983f9422a27",
    str(RUN / "receipt.json"): "a7a08c5387fec735840e8c76db82624f75518403a96f23782d054920cc26472a",
    str(GROWTH / "analyze_growth.py"): "cf51853026bd8647ff497559f2b741db17a8cb4c7653fa164f9096523099f2a0",
    str(GROWTH / "pade13_einsum.py"): "0654c3ba8884dc467aa1c4665fc4758a3abddf58189e2702f7e8cfc90d2e79b7",
    str(ISOLATED): "4ced570e79ea509b4d502b9544d4d0272a4b313b761088ad61e46ca21d11a7f1",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    out = HERE / "prepared-001"
    out.mkdir(exist_ok=False)
    for path, expected in PINS.items():
        assert sha(path) == expected, path
    receipt = json.loads((RUN / "receipt.json").read_text())
    assert receipt["passed_finite_ODE_numerical_checks"] is True
    assert (receipt["J"], receipt["N"], receipt["rb"]) == (0, 8, .98)
    payload = RUN / "growth-payload.npz"
    assert sha(payload) == receipt["payload_sha256"]
    pins = {**PINS, str(payload): sha(payload), str(Path(__file__)): sha(__file__),
            str(HERE / "PLAN.md"): sha(HERE / "PLAN.md")}
    tree = ast.parse((GROWTH / "analyze_growth.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("mm", "energy_transform")]
    assert len(nodes) == 2
    namespace = {"np": np, "solve_triangular": solve_triangular}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewed-helper-asts", "exec"), namespace)
    with np.load(OP, allow_pickle=False) as data:
        E = data["E"].copy()
        J = data["Jbulk"] + data["Jsat"]
    L = np.linalg.cholesky(E)
    A = namespace["energy_transform"](L, J)
    with np.load(ISOLATED, allow_pickle=False) as prior:
        for name, value in (("A", A), ("L", L), ("J", J)):
            assert np.array_equal(prior[name], value), name
    with np.load(payload, allow_pickle=False) as data:
        X = data["physical_seed_modal"].copy()
        times = data["propagation_times"].copy()
        assert np.array_equal(times, [0., .25, .5, 1., 2., 4., 6.])
        saved_t6 = data["seed_states_modal"][-1].copy()
    Z = namespace["mm"](L.T, X)
    np.savez(out / "inputs.npz", A=A, J=J, E=E, L=L,
             physical_seed_modal=X, initial_energy_seed=Z,
             saved_t6_modal=saved_t6, propagation_times=times,
             floating_argument=6.0 * A)
    result = {"status": "PASS_input_binding_only", "source_sha256": pins,
              "A_L_J_bitwise_match_prior_snapshot": True,
              "extracted_helpers_only": [n.name for n in nodes],
              "analyzer_executed": False, "generator_spectrum_computed": False,
              "exponential_evaluated": False,
              "numpy_error_mode": np.geterr(), "python": sys.version,
              "inputs_sha256": sha(out / "inputs.npz")}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "inputs_sha256": result["inputs_sha256"]}))


if __name__ == "__main__":
    main()
