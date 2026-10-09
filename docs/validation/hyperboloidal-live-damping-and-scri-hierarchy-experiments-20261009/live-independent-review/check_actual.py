"""Independent actual-matrix delta and native-grid frequency/limit metadata audit."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "live-damping-control"
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(x):
    a = np.asarray(x)
    return a[..., 0]+1j*a[..., 1]


def main():
    rows = json.loads((CONTROL / "full20.json").read_text())
    assert len(rows) == 1064
    pairs = {}
    for row in rows:
        key = tuple(row[k] for k in ("a", "perturb", "r", "Omega", "k", "oblique"))
        pairs.setdefault(key, {})[row["profile"]] = row
    maximum, changed_beta = 0., 0
    for pair in pairs.values():
        assert set(pair) == {0, 1}
        row = pair[1]
        expected = np.zeros((20, 20))
        expected[[2, 3], 3] = -10*row["kappa2"]/row["Omega"]
        beta = -2*row["V"]*row["Theta"]*np.asarray(row["Omega_gradient"])/row["Omega"]
        expected[2, 4:7] = expected[3, 4:7] = beta
        observed = matrix(row["L"])-matrix(pair[0]["L"])
        maximum = max(maximum, float(abs(observed-expected).max()/(1+abs(expected).max())))
        assert np.count_nonzero(observed[np.r_[0:2, 4:20]]) == 0
        changed_beta += int(np.any(beta))
    assert maximum < 1e-11 and changed_beta == 247 and len(pairs) == 532
    native_input = ROOT / "build-layer-research/spatial-norm-native-controls/N36-t0.2/finite-angular-long-N36/layer.athinput"
    section, parameters = "", {}
    for raw in native_input.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("<"):
            section = line.strip("<>")
        elif section == "mesh" and "=" in line:
            key, value = line.split("=", 1)
            parameters[key.strip()] = value.strip()
    span = float(parameters["x1max"])-float(parameters["x1min"])
    assert span == 2.1
    rk = json.loads((CONTROL / "full20-RK-all-a.json").read_text())
    assert len(rk) == 192
    omega_error, frequency_error, excess = 0., 0., 0.
    positives, cases = 0, set()
    for row in rk:
        N, a = row["N"], row["a"]
        h = span/N
        coordinate = float(parameters["x1min"])+(np.arange(N)+.5)*h
        r2 = sum(x*x for x in np.meshgrid(coordinate, coordinate, coordinate, indexing="ij"))
        expected = (1-r2[r2 < 1].max())/(2*a)
        omega_error = max(omega_error, abs(row["Omega"]-expected))
        target_k = 256 if row["frequency"] else np.pi/h
        frequency_error = max(frequency_error, abs(row["k"]-target_k))
        assert abs(row["dt"]-.03*expected) < 1e-14
        assert abs(row["r"]**2-r2[r2 < 1].max()) < 2e-14
        L = matrix(row["L"])
        weight = np.r_[np.ones(12), np.full(8, 1/max(row["k"], 1))]
        ev = np.linalg.eigvals(weight[:, None]*L/weight[None, :])
        positives += int(np.any(ev.real > 0))
        z = row["dt"]*ev[ev.real <= 0]
        excess = max(excess, float(max(0, np.max(abs(1+z+z*z/2+z*z*z/6))-1)))
        cases.add((a, N, row["profile"], row["perturb"], row["oblique"], row["frequency"]))
    assert omega_error < 2e-14 and frequency_error < 2e-13 and excess < 1e-12
    assert len(cases) == 192
    report = {
        "status": "PASS", "scope": "Independent read-only actualfull20 value-Jacobian delta and192 all-a native-grid frequency/Omega_min scalar-eigenvalue checks; finiteOmega only",
        "full20_rows": len(rows), "matched_pairs": len(pairs),
        "finite_Theta_beta_pairs": changed_beta, "relative_delta_error": maximum,
        "all_other_rows_bitwise_unchanged": True,
        "native_span": span, "RK_cases": len(rk), "Omega_min_max_error": omega_error,
        "frequency_max_error": frequency_error, "nonpositive_root_RK3_max_excess": excess,
        "RK_cases_with_excluded_positive_roots": positives,
        "limitations": ["Continuum frozen matrices sampled at native pi/h, not native finite-difference symbols.",
                        "Excludes positive roots and does not bound nonnormal propagators, global RK stability or a regularity manifold.",
                        "Full nonlinear coefficient/constraint closure and evolution preservation of kappa2 admissibility remain unproved."],
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in
                    (Path(__file__), native_input, CONTROL / "full20.json", CONTROL / "full20-RK-all-a.json")}}
    (HERE / "actual-review.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
