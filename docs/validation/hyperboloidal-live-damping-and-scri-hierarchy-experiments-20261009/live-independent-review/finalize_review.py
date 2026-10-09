"""Verify final immutable local gate and freeze independent review separately."""
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTROL = HERE.parent / "live-damping-control"
GATE = CONTROL / "immutable-live-damping-local-20261009"
DEST = HERE / "immutable-independent-live-review-20261009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    assert sha(GATE / "index.json") == "fcfd4a740fc25e999d0417598015608c009bf61de032aff5266f1f843e2d6b59"
    index = json.loads((GATE / "index.json").read_text())
    for name, digest in index["files"].items():
        assert sha(GATE / name) == digest
    for entry in index["large_outputs_outside_snapshot"].values():
        path = ROOT / entry["original_repo_path"]
        assert sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
    receipt = json.loads((GATE / "receipt.json").read_text())
    assert sha(GATE / "receipt.json") == "e038769458c15e7cd1c5a956ef75bd4508b777c981d7ac7da9594c273dcdf229"
    assert len(receipt["source_before"]) == 384
    assert receipt["source_before"] == receipt["source_after"]
    for name, digest in receipt["source_after"].items():
        assert sha(ROOT / name) == digest
    assert len(receipt["commands"]) == 17
    assert all(row["returncode"] == 0 and row["stderr"] == "" for row in receipt["commands"])
    assert receipt["baseline_rerun_byte_identical"]
    report = json.loads((GATE / "check-report.json").read_text())
    assert report["passed_finite_Omega_local_gate"]
    assert not report["global_native_or_scri_stability_accepted"]
    independent = json.loads((HERE / "actual-review.json").read_text())
    assert independent["relative_delta_error"] == report["full20"]["relative_exact_delta_error"]
    assert independent["RK_cases_with_excluded_positive_roots"] == 94
    assert report["principal"]["passed_kernel_cases"] == 360
    assert report["subsidiary"]["live"]["worst_omitted_dkappa2_error"] > .1
    review = {
        "status": "PASS", "scope": "Independent mathematical/source/provenance and actual-matrix review of frozen finiteOmega local live-damping gate only",
        "gate_index_sha256": sha(GATE / "index.json"),
        "gate_receipt_sha256": sha(GATE / "receipt.json"),
        "gate_small_files_verified": len(index["files"]),
        "gate_large_hashes_verified": len(index["large_outputs_outside_snapshot"]),
        "gate_unchanged_inputs_verified": 384, "gate_zero_commands_verified": 17,
        "math_review_sha256": sha(HERE / "math-review.json"),
        "actual_review_sha256": sha(HERE / "actual-review.json"),
        "reviewed_limits": ["All 16 poles directly evaluated; exact nullity/Hurwitz statements concern eight rationally reconstructed reference matrices only.",
                            "Positive primitive roots and bounded innerk0 subsidiary roots are retained; scalarRK excludes positive roots in94of192 matrices.",
                            "No actual native finite-difference symbol, nonnormal propagator, global spectrum, energy, nonlinear invariant bound, scri closure or BH gate.",
                            "The earlier uncut positive-kappa2 witness and current a.5 initial bounds are preserved separately; all-a local gates do not extend those initial-data inequalities."],
        "source_and_dimensions_review": "kappa_input and beta.Omega_i have dimension1/length; kappa2,V,Omega are dimensionless; m_i dimension1/length^2. kappa_input is spatially constant and V=.15-.3 is prescribed for S1, distinct from geometry/gauge cutoffs.",
        "corrections_required": []}
    (HERE / "final-review.json").write_text(json.dumps(review, indent=2, allow_nan=False)+"\n")
    DEST.mkdir()
    for name in ("review.py", "check_actual.py", "finalize_review.py", "math-review.json", "actual-review.json",
                 "final-review.json", "run.log", "actual.log"):
        shutil.copy2(HERE / name, DEST / name)
    files = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(DEST.iterdir())}
    frozen = {"scope": review["scope"], "files": files,
              "external_gate_index_sha256": sha(GATE / "index.json"),
              "file_count": len(files), "bytes": sum(value["bytes"] for value in files.values())}
    (DEST / "index.json").write_text(json.dumps(frozen, indent=2, allow_nan=False)+"\n")
    print("PASS frozen independent live review", str(DEST.relative_to(ROOT)), sha(DEST / "index.json"))
    print(len(files), "small files", frozen["bytes"], "bytes")


if __name__ == "__main__":
    main()
