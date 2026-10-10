"""Saved finite JSON/hash/rational readback; never imports or reruns targets."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib
import json
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "build-layer-research/continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009"
RELEASE = ROOT / "build-layer-research/Gaussian-third-jet-oracle-v3-root-release-20261009"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(p):
    p = Path(p)
    require(p.suffix not in (".jsonl", ".npz", ".npy") and p.stat().st_size <= 1048576,
            "scientific stream/array decoding is not admitted")
    return json.loads(p.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def captured(p):
    return HERE / "captured" / Path(p).relative_to(ROOT)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "explicit unoptimized -I -B required")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "explicit optimization environment required")
    out = HERE / "saved-readback.json"
    require(not out.exists(), "one-shot fresh saved readback required")
    manifest = load(HERE / "capture.json")
    json_count = 0
    for row in manifest["inputs"]:
        for key in ("origin", "copy"):
            require(sha(row[key]) == row["sha256"] and Path(row[key]).stat().st_size == row["bytes"], "capture drift")
        if Path(row["copy"]).suffix == ".json":
            load(row["copy"])
            json_count += 1
    childdir = OWNER / "attempts/units001"
    child = load(captured(childdir / "receipt.json"))
    report = load(captured(childdir / "result.json"))
    outer = load(captured(RELEASE / "units-outer001/receipt.json"))
    root = load(captured(RELEASE / "units-invocation001/receipt.json"))
    require(child["stage"] == "units" and all(child[k] is True for k in ("completed", "passed", "sources_unchanged", "scientific_stage_authorized")), "actual child admission")
    require(child["source_index_sha256"] == "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a", "actual child source index")
    require(child["recipe_sha256"] == "847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf", "actual child recipe")
    require(report == {"passed": True, "records": 0, "checks": 318, "failed": [], "failed_total": 0}, "actual result summary")
    for receipt in (outer, root):
        require(receipt["returncode"] == 0 and all(receipt[k] is True for k in ("completed", "accepted_stage", "inputs_unchanged")), "enclosing completion")
        require(receipt["child_receipt_sha256"] == sha(childdir / "receipt.json") and receipt["result_sha256"] == sha(childdir / "result.json"), "enclosing output binding")
    require(root["outer_receipt_sha256"] == sha(RELEASE / "units-outer001/receipt.json"), "root/outer binding")
    require(root["root_process_group_cap_seconds"] == 60 and root["elapsed_seconds"] < 60, "actual root process cap")
    before = load(captured(childdir / "source-before.json"))
    after = load(captured(childdir / "source-after.json"))
    require(before == after, "child before/after maps")
    combined = dict(before)
    release = load(captured(RELEASE / "units-release.json"))
    require(release["required_base_pins"] == 437 and release["stage"] == "units", "root base/stage binding")
    for p, h in release["pins"].items():
        if p in combined:
            require(combined[p] == h, "overlapping pin mismatch")
        combined[p] = h
    for p, h in combined.items():
        require(sha(p) == h, "current protected input drift: " + p)
    for rel, h in child["output_hashes"].items():
        require(sha(childdir / rel) == h, "actual child output drift")
    for row in root["output_inventory"]:
        require(sha(row["path"]) == row["sha256"] and Path(row["path"]).stat().st_size == row["bytes"], "root output inventory drift")
    auth = load(captured(RELEASE / "units-authorization.json"))
    require(sha(RELEASE / "units-authorization.json") == release["authorization_sha256"], "exact authorization pin")
    require(auth["Gaussian_third_jet_oracle_stage_authorized"] == "units", "actual stage authorization")
    require(all(auth[key] == child[key] for key in ("source_index_sha256", "recipe_sha256", "driver_sha256")), "actual authorization/source binding")
    for folder in ("units-invocation001", "units-outer001"):
        invocation = load(captured(RELEASE / folder / "invocation.json"))
        require("-I" in invocation["command"] and "-B" in invocation["command"], "actual isolated bytecode-free route")
        env = invocation.get("fixed_environment", invocation.get("environment", {}))
        require(env.get("PYTHONOPTIMIZE") == "0", "actual unoptimized environment")
        for stream in ("stdout.log", "stderr.log"):
            require((RELEASE / folder / stream).stat().st_size == 0, "nonempty actual run log")
    rows = load(captured(childdir / "unit-checks.json"))
    require(len(rows) == 318 and len({row["name"] for row in rows}) == 318, "fixed unique unit rows")
    groups = Counter(row["name"].split("/")[0] for row in rows)
    max_scaled = Fraction(0)
    max_saved_error = Fraction(0)
    max_name = None
    for row in rows:
        require(row["admission_gate"] is True, "unit row is not gated")
        terms = list(map(Fraction, row["terms"]))
        signed, absolute, total, scaled = map(Fraction, (row["signed"], row["absolute"], row["term_sum"], row["scaled"]))
        require(absolute >= 0 and total >= 0 and scaled >= 0 and absolute == abs(signed), "invalid stored signed/absolute fields")
        require(scaled <= Fraction("1e-55"), "unchanged scientific tolerance failed")
        exact_sum = sum(terms, Fraction(0))
        exact_total = sum(map(abs, terms), Fraction(0))
        error = max(abs(exact_sum - signed), abs(exact_total - total)) / max(Fraction(1), exact_total)
        # Serialization readback allowance only; no target is recomputed or retoleranced.
        require(error <= Fraction("1e-75"), "saved operand-sum serialization mismatch")
        denominator = Fraction(row["component_scale"]) if "component_scale" in row else max(Fraction(1), total)
        exact_scaled = absolute / denominator
        require(abs(exact_scaled - scaled) <= Fraction("1e-75") * max(Fraction(1), exact_scaled), "saved scaled residual mismatch")
        max_saved_error = max(max_saved_error, error)
        if scaled > max_scaled:
            max_scaled, max_name = scaled, row["name"]
    for row in manifest["inputs"]:
        require(sha(row["origin"]) == row["sha256"] and sha(row["copy"]) == row["sha256"], "postreview capture drift")
    for p, h in combined.items():
        require(sha(p) == h, "postreview protected input drift")
    output = {"passed": True, "saved_only": True, "inputs_unchanged": True,
              "source_index_sha256": child["source_index_sha256"], "recipe_sha256": child["recipe_sha256"], "driver_sha256": child["driver_sha256"],
              "child_receipt_sha256": sha(childdir / "receipt.json"), "result_sha256": sha(childdir / "result.json"),
              "unit_checks_sha256": sha(childdir / "unit-checks.json"),
              "outer_receipt_sha256": sha(RELEASE / "units-outer001/receipt.json"),
              "root_receipt_sha256": sha(RELEASE / "units-invocation001/receipt.json"),
              "checks": 318, "failed": 0, "groups": dict(sorted(groups.items())),
              "original_scientific_tolerance": "1e-55", "saved_scalar_serialization_allowance": "1e-75; not a target tolerance change",
              "maximum_saved_scaled_fraction": str(max_scaled), "maximum_scaled_row": max_name,
              "maximum_saved_operand_sum_error_fraction": str(max_saved_error),
              "child_protected_pins": len(before), "unique_child_plus_root_protected_pins": len(combined),
              "captured_inputs": len(manifest["inputs"]), "finite_captured_JSON": json_count,
              "child_seconds": child["elapsed_seconds"], "root_seconds": root["elapsed_seconds"], "root_group_cap_seconds": 60,
              "unchanged_80_digit_result_and_unit_rows_equal_v2_hashes": sha(childdir / "result.json") == "c569c2e3d8fe884e3a4f14723fc91ce7c25c8cf447b577cff6ddb8244c98db89" and sha(childdir / "unit-checks.json") == "871110768b57abc0e742b87d68cacef240a392b498bed6e0755f774430cdf781",
              "candidate_imports_or_targets_rerun": False, "scientific_JSONL_NPZ_NPY_decoded": False,
              "timing_full_native_admission": False,
              "command": ["python3", "-I", "-B", str(Path(__file__).resolve())],
              "runtime": {"executable": sys.executable, "resolved_executable": str(Path(sys.executable).resolve()),
                          "executable_sha256": sha(sys.executable), "version": sys.version, "flags": str(sys.flags),
                          "environment": {k: os.environ.get(k) for k in ("PYTHONOPTIMIZE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}}
    out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in ("passed", "checks", "failed", "child_protected_pins", "unique_child_plus_root_protected_pins", "groups", "root_seconds", "child_seconds")}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        failure = HERE / "review-failure.txt"
        if not failure.exists():
            failure.write_text(traceback.format_exc())
        raise
