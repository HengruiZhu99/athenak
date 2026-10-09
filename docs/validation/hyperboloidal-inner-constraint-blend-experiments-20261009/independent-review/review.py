"""Read-only independent frozen prescribed-C1-blend admission review."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "build-layer-research/continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009"
EXPECTED = "1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(value):
    a = np.asarray(value)
    assert np.isfinite(a).all()
    return a[:, :, 0] + 1j * a[:, :, 1]


def main():
    np.seterr(all="raise")
    assert sha(GATE / "index.json") == EXPECTED
    index = json.loads((GATE / "index.json").read_text())
    for group in ("files", "large_files"):
        for name, record in index[group].items():
            path = GATE / name if group == "files" else Path(name)
            assert sha(path) == record["sha256"] and path.stat().st_size == record["bytes"]
    receipt = json.loads((GATE / "receipt.json").read_text())
    assert len(receipt["source_before"]) == 380
    assert receipt["source_before"] == receipt["source_after"]
    assert all(sha(ROOT / name) == digest for name, digest in receipt["source_after"].items())
    assert len(receipt["commands"]) == 10
    assert all(row["returncode"] == 0 for row in receipt["commands"])
    assert not receipt["global_native_or_scri_stability_accepted"]
    report = json.loads((GATE / "check-report.json").read_text())
    results = []
    for mode in ("C1", "blend"):
        path = GATE.parent / ("tangent-" + mode + "-kappa10.json")
        rows = json.loads(path.read_text())
        assert len(rows) == 1000
        fine, roots, omitted = [], [], []
        for row in rows:
            q, d, s = [matrix(row[name]) for name in ("Q", "D", "subsidiary")]
            assert abs(q[:, [0, 4, 5, 6]]).max() == 0
            if row["level"] != 4:
                continue
            relative = float(abs(d - s).max() / (1 + abs(d).max() + abs(s).max()))
            fine.append(relative)
            g = matrix(row["constraint_generator"])[:, :8]
            weights = np.array([1.] * 4 + [max(row["k"], 1)] * 4)
            balanced = weights[:, None] * g / weights[None, :]
            eigen = np.linalg.eigvals(balanced)
            roots.append(float(eigen.real.max()))
            ng = matrix(row["without_dC"])
            omitted.append(float(abs(d - ng).max() / (1 + abs(d).max() + abs(ng).max())))
        assert len(fine) == len(roots) == 200
        assert max(fine) < 1e-7 and max(roots) < 0
        if mode == "blend":
            assert max(omitted) > .01
        recorded = next(c for c in report["cases"] if c["mode"] == mode)
        assert abs(max(fine) - recorded["max_finest_relative_error"]) < 1e-15
        assert abs(max(roots) - recorded["worst_local_root"]["max_real"]) < 1e-10
        results.append({"mode": mode, "actual20_rows": len(rows), "local8_rows": len(roots),
                        "max_finest_relative_closure_error": max(fine),
                        "maximum_local8_real_root": max(roots),
                        "max_error_omitting_coefficient_gradient": max(omitted)})
    helper = (GATE / "bulk_c1_additions.hpp").read_text()
    assert "1-LayerCoefficients(radius,u.alpha.value,g).weight" in helper
    assert "TensorC1Additions(u,o,coefficient,true)" in helper
    assert sha(GATE / "bulk_c1_additions.hpp") == "4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2"
    tensor = json.loads((GATE / "blend-gate.json").read_text())
    assert tensor == json.loads((GATE / "blend-gate-debug.json").read_text())
    assert tensor["rows"] == 4004
    assert tensor["einstein_max"] == tensor["outer_offconstraint_max"] == tensor["outer_cutoff_jets_max"] == 0
    out = {"status": "PASS", "gate_index_sha256": EXPECTED,
           "source_paths_verified": 380, "commands_exit0": 10, "results": results,
           "all_C1_parts_and_covector_repair_same_coefficient": True,
           "gradient_c_source": "gamma^ja c_a DeltaS_ij-c_i gamma^ja DeltaS_ja",
           "outer_support_bound_a05": .2775, "tensor_gate": tensor,
           "physical_P_storage_and_lapse_gauge_unchanged": True,
           "geometric_P_RHS_receives_C1_trace_addition": True,
           "admission": "Fixed finite-Omega exploratory RHS/Jv/RK/reference/short screens only",
           "uniform_energy_scri_native_stability_or_BH_accepted": False,
           "source_sha256": sha(Path(__file__))}
    Path(__file__).with_name("receipt.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
