"""Independent hash, scope and numerical reproduction of frozen C1 preflight."""
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("independent_small", HERE / "review_small_omega.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
CANDIDATE = HERE.parent / "covariant-z4-candidate"
FROZEN = CANDIDATE / "immutable-C1-stiffness-20261009"
EXPECTED = "321439833808976b1b6ebfc99e443b0f61e0018b4a0aea257986ae55f0e846a8"
V2 = CANDIDATE / "immutable-C1-stiffness-v2-20261009"
EXPECTED_V2 = "d8d137e4422dec83ec685f0fee45fc80d3363cbf6a3252fec8b8d41c53c485ed"


def key(row):
    return tuple(row[x] for x in (
        "a", "kappa", "form", "norm_gauge", "perturb", "r", "Omega", "k", "oblique"))


def error(a, b):
    return abs(a - b) / max(1, abs(a), abs(b))


def main():
    np.seterr(all="raise")
    index_path = FROZEN / "index.json"
    assert review.sha(index_path) == EXPECTED
    index = json.loads(index_path.read_text())
    for group in ("files", "large_files"):
        for name, record in index[group].items():
            path = FROZEN / name if group == "files" else Path(name)
            assert review.sha(path) == record["sha256"]
            assert path.stat().st_size == record["bytes"]
    identity = index["identity_index"]
    assert review.sha(Path(identity["path"])) == identity["sha256"]
    assert review.sha(V2 / "index.json") == EXPECTED_V2
    v2 = json.loads((V2 / "index.json").read_text())
    assert v2["supersedes_prose_only"]["index_sha256"] == EXPECTED
    for name, record in v2["files"].items():
        assert review.sha(V2 / name) == record["sha256"]
        assert (V2 / name).stat().st_size == record["bytes"]
    assert v2["large_files"] == index["large_files"]
    for name in index["files"]:
        if name != "REPORT.md":
            assert index["files"][name] == v2["files"][name]
    receipt = json.loads((FROZEN / "receipt.json").read_text())
    assert receipt["sources_unchanged"]
    assert receipt["source_before"] == receipt["source_after"]
    assert len(receipt["source_before"]) == 370
    for path, digest in receipt["source_after"].items():
        assert review.sha(ROOT / path) == digest
    assert len(receipt["commands"]) == 4
    assert all(row["returncode"] == 0 and not row["stderr"]
               for row in receipt["commands"])
    result = json.loads((CANDIDATE / "check-report.json").read_text())
    assert not result["native_stability_accepted"]
    assert not result["nonlinear_scri_closure_accepted"]
    local = json.loads((HERE / "receipt.json").read_text())
    small = json.loads((CANDIDATE / "small-Omega.json").read_text())
    large = json.loads((CANDIDATE / "Fourier.json").read_text())
    own = {tuple(r[k] for k in (
        "a", "kappa", "form", "norm_gauge", "perturb", "r", "Omega")): r
        for group in local["groups"] for r in [group["smallest_Omega_row"]]}
    small_errors = []
    for row in result["small_Omega"]:
        short = tuple(row[k] for k in (
            "a", "kappa", "form", "norm_gauge", "perturb", "r", "Omega"))
        if short not in own:
            continue
        o = own[short]
        for old, new in [
                ("balanced_L_norm", "balanced_generator_norm2"),
                ("raw_rk_norm", "raw_RK3_norm2"),
                ("balanced_rk_norm", "balanced_RK3_norm2"),
                ("balanced_exact_norm", "balanced_exact_norm2"),
                ("omega_spectral_radius", "scaled_spectral_radius")]:
            small_errors.append(error(row[old], o[new]))
    assert len(small_errors) == 96 * 5 and max(small_errors) < 1e-10
    matrices = {key(row): row for row in large}
    scalar_excess, scalar_error = 0., 0.
    for row in result["native_RK"]:
        source = matrices[key(row)]
        om = source["Omega"]
        weights = np.ones(20)
        weights[12:] = om
        b = om * weights[:, None] * review.matrix(source) / weights[None, :]
        eig = np.linalg.eigvals(b) / om
        z = row["dt"] * eig
        polynomial = abs(1 + z + z * z / 2 + z * z * z / 6)
        damped = eig.real <= 0
        scalar_excess = max(scalar_excess, float(max(0, polynomial[damped].max() - 1)))
        scalar_error = max(scalar_error, error(float(polynomial.max()), row["rk_radius"]))
    assert len(result["native_RK"]) == 5040 and scalar_excess < 2e-10
    assert scalar_error < 1e-10
    propagator_error = 0.
    for row in result["propagators"]:
        source = matrices[key(row)]
        om = source["Omega"]
        weights = np.ones(20)
        weights[12:] = om
        b = om * weights[:, None] * review.matrix(source) / weights[None, :]
        exponential = review.expm(row["time_over_Omega"] * b)
        raw = exponential / weights[:, None] * weights[None, :]
        for value, recorded in [
                (np.linalg.norm(exponential, 2), row["balanced_exact_norm"]),
                (np.linalg.norm(raw, 2), row["raw_exact_norm"])]:
            propagator_error = max(propagator_error, error(value, recorded))
    assert len(result["propagators"]) == 810 and propagator_error < 2e-10
    out = {
        "status": "PASS", "frozen_index_sha256": EXPECTED,
        "v2_prose_correction_index_sha256": EXPECTED_V2,
        "v1_preserved_numerical_sources_and_receipt_identical_v2": True,
        "frozen_receipt_sha256": review.sha(FROZEN / "receipt.json"),
        "370_input_sources_unchanged": True, "four_commands_exit0": True,
        "96_smallest_Omega_rows_independent_norm_and_spectrum_comparisons": True,
        "smallest_Omega_max_relative_difference": max(small_errors),
        "native_local_scalar_RK_cases": len(result["native_RK"]),
        "damped_RK_excess_independent": scalar_excess,
        "scalar_RK_max_relative_difference": scalar_error,
        "independent_exact_propagators": len(result["propagators"]),
        "propagator_max_relative_difference": propagator_error,
        "scope_review": "Finite-Omega local preflight only; no uniform raw norm, native stability, scri closure or imposed falloff",
        "review_source_sha256": review.sha(Path(__file__)),
    }
    (HERE / "frozen-candidate-review.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
