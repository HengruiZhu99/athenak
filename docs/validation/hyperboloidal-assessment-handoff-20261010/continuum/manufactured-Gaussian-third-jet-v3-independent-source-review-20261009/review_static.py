"""Independent metadata/text/AST review; never imports the held oracle."""
import ast
import difflib
import hashlib
import json
import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
EXPECTED_INDEX = "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a"
EXPECTED_RECIPE = "847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    path = pathlib.Path(path)
    require(path.suffix not in (".jsonl", ".npz", ".npy") and path.stat().st_size <= 1048576,
            "scientific payload decoding is forbidden in this source review")
    return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def save(path, value):
    with pathlib.Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def syntax(path):
    return ast.dump(ast.parse(pathlib.Path(path).read_text()), include_attributes=False)


def verify(pins):
    for name, expected in pins.items():
        require(sha(name) == expected, "input drift: " + name)


def capture(path, target):
    path, target = pathlib.Path(path), pathlib.Path(target)
    require(path.suffix not in (".jsonl", ".npz", ".npy") and path.stat().st_size <= 1048576,
            "only compact UTF-8 history evidence may be copied")
    data = path.read_bytes()
    data.decode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "fresh evidence copy required")
    target.write_bytes(data)
    require(sha(path) == sha(target), "history copy mismatch")
    return {"origin": str(path), "copy": str(target), "bytes": len(data), "sha256": sha(path)}


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "source reviewer requires explicit unoptimized -I -B")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "optimization environment must be explicit")
    require(not (ROOT / "static-readback.json").exists(), "one-shot fresh review required")
    capture_manifest = load(ROOT / "capture.json")
    require(capture_manifest["reviewed_source_index_sha256"] == EXPECTED_INDEX,
            "source capture mismatch")
    require(sha(INPUT / "source-index.json") == EXPECTED_INDEX and sha(INPUT / "recipe.json") == EXPECTED_RECIPE,
            "literal source/recipe binding mismatch")
    index, recipe = load(INPUT / "source-index.json"), load(INPUT / "recipe.json")
    original = INPUT / "source-history/v2-final"
    original_index, original_recipe = load(original / "source-index.json"), load(original / "recipe.json")
    require(len(index["files"]) == index["file_count"] == 58 and len(recipe["protected_inputs"]) == 378,
            "unexpected fixed source/pin counts")
    all_pins = dict(recipe["protected_inputs"])
    for row in capture_manifest["files"]:
        require(row["origin"] not in all_pins or all_pins[row["origin"]] == row["sha256"], "conflicting pin")
        all_pins[row["origin"]] = row["sha256"]
        require(sha(row["copy"]) == row["sha256"], "captured input changed")
    verify(all_pins)
    save(ROOT / "pins-before.json", all_pins)

    # Verify the actual original v2 source histories, not only owner preparation claims.
    original_folder = pathlib.Path(recipe["v3_precision_lineage"]["v2_source_index"]["path"]).parent
    require(sha(original / "source-index.json") == "92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3",
            "preserved v2 index mismatch")
    for name, expected in original_index["files"].items():
        rel = pathlib.Path(name).relative_to(original_folder)
        require(sha(original / rel) == expected and sha(name) == expected, "original v2 source/history mismatch")

    identical = []
    names = ("taylor3.py", "reference3.py", "gaussian3.py", "geometry.py", "oracle.py", "units.py",
             "values_context.py", "diagnostics.py", "outer_once.py")
    for name in names:
        require((INPUT / name).read_bytes() == (original / name).read_bytes(), "runtime body changed: " + name)
        require(syntax(INPUT / name) == syntax(original / name), "runtime AST changed: " + name)
        identical.append({"file": name, "sha256": sha(INPUT / name), "bytes_and_AST_equal_v2": True})
    require((INPUT / "authorization-schema.json").read_bytes() == (original / "authorization-schema.json").read_bytes(),
            "authorization schema changed")

    new_driver, old_driver = (INPUT / "run_oracle.py").read_text(), (original / "run_oracle.py").read_text()
    require(new_driver.count("180_vs_220_compact_component") == 1 and old_driver.count("110_vs_150_compact_component") == 1,
            "precision label uniqueness mismatch")
    restored_driver = new_driver.replace("180_vs_220_compact_component", "110_vs_150_compact_component")
    require(restored_driver.encode() == (original / "run_oracle.py").read_bytes(), "driver reverse bytes mismatch")
    require(ast.dump(ast.parse(restored_driver), include_attributes=False) == syntax(original / "run_oracle.py"),
            "driver reverse AST mismatch")

    expected_levels = [
        {"digits": 180, "roots": {"absolute": "1e-155", "width": "1e-160", "newton": 16, "bisection": 768}},
        {"digits": 220, "roots": {"absolute": "1e-195", "width": "1e-200", "newton": 16, "bisection": 768}}]
    require(recipe["levels"] == expected_levels, "precision/root settings mismatch")
    restored_recipe = dict(recipe)
    for name in ("status", "levels", "protected_inputs"):
        restored_recipe[name] = original_recipe[name]
    restored_recipe.pop("v3_precision_lineage")
    require(restored_recipe == original_recipe, "nonprecision scientific recipe change")
    require((json.dumps(restored_recipe, indent=2, allow_nan=False) + "\n").encode() == (original / "recipe.json").read_bytes(),
            "recipe reverse bytes mismatch")
    require(recipe["unit_digits"] == 80 and recipe["unit_roots"] == original_recipe["unit_roots"], "unit recipe changed")
    require(recipe["identity_tolerance"] == recipe["component_tolerance"] == "1e-55", "identity/component gate changed")
    require(recipe["height_tolerance"] == "1e-30", "height context gate changed")
    require(recipe["execution_admitted"] is False and index["execution_admitted"] is False,
            "held source must not self-authorize")

    new_plan = (INPUT / "PLAN.md").read_text()
    marker = "\n\n## Fresh v3 precision-only sibling and preserved failure\n"
    require(new_plan.count(marker) == 1, "new plan addendum marker mismatch")
    base_plan, addendum = new_plan.split(marker)
    require(base_plan.replace("and5,010 at180/220 digits.", "and5,010 at110/150 digits.").encode() == (original / "PLAN.md").read_bytes(),
            "PLAN reverse bytes mismatch")
    require((INPUT / "SCHEMA.md").read_text().replace("Values180/220 are retained", "Values110/150 are retained").encode()
            == (original / "SCHEMA.md").read_bytes(), "SCHEMA reverse bytes mismatch")
    require((INPUT / "measured-timing-review-schema.json").read_text().replace("exact current v3 source-index hash", "exact current v2 source-index hash").encode()
            == (original / "measured-timing-review-schema.json").read_bytes(), "timing schema reverse bytes mismatch")

    changed = ("run_oracle.py", "recipe.json", "PLAN.md", "SCHEMA.md", "measured-timing-review-schema.json")
    expected_diff = "".join("".join(difflib.unified_diff((original / name).read_text().splitlines(True),
                    (INPUT / name).read_text().splitlines(True), fromfile="v2/" + name, tofile="v3/" + name))
                    for name in changed)
    require(expected_diff.encode() == (INPUT / "v2-to-v3.diff").read_bytes(), "published full diff differs from actual source")

    # Capture compact historical receipts before reviewing their fields. No scientific stream is decoded.
    lineage = recipe["v3_precision_lineage"]
    history_inputs = [pathlib.Path(lineage[key]["path"]) for key in ("v2_units_receipt", "v2_timing_receipt")]
    history_inputs += [history_inputs[0].parent / "result.json", history_inputs[1].parent / "result.json"]
    history_inputs += [pathlib.Path(lineage["v2_failure_saved_review_runtime_addendum"]["path"]).parent / "ADDENDUM.json"]
    records = [capture(path, ROOT / "history-evidence" / ("%02d-" % n + path.parent.name + "-" + path.name))
               for n, path in enumerate(history_inputs)]
    save(ROOT / "history-capture.json", {"files": records, "source_and_metadata_only": True})
    unit, timing, unit_result, timing_result, runtime_addendum = [load(row["copy"]) for row in records]
    require(unit["source_index_sha256"] == timing["source_index_sha256"] == lineage["v2_source_index"]["sha256"],
            "historical receipt candidate mismatch")
    require(unit["completed"] is True and unit["passed"] is True and unit["sources_unchanged"] is True
            and unit["stage"] == "units" and unit_result["passed"] is True and unit_result["checks"] == 318
            and unit_result["failed_total"] == 0, "historical unit classification mismatch")
    require(timing["completed"] is True and timing["passed"] is False and timing["sources_unchanged"] is True
            and timing["stage"] == "timing" and timing_result["passed"] is False and timing_result["records"] == 20
            and timing_result["failed_total"] == 6, "historical timing failure classification mismatch")
    require(runtime_addendum["explicit_I_B_flags"] is False
            and runtime_addendum["historical_reviewer_interpreter_resolved_path_recorded"] is False,
            "historical reviewer runtime limitation omitted")
    require(lineage == load(INPUT / "historical-v2-context.json"), "historical context recipe mismatch")
    for key in ("v2_source_index", "v2_units_receipt", "v2_timing_receipt", "v2_failure_saved_review",
                "v2_failure_saved_review_runtime_addendum", "precision_mechanism_proposal"):
        pin = lineage[key]
        require(recipe["protected_inputs"].get(pin["path"]) == pin["sha256"], "unprotected lineage pin: " + key)

    # Text review checks bind the inherited gate route, not a substitute implementation or a runtime pass.
    outer = (INPUT / "outer_once.py").read_text()
    for text in ('sys.flags.optimize!=0', 'sys.flags.isolated!=1', 'sys.flags.dont_write_bytecode!=1',
                 'unit prerequisite is from another candidate', 'timing prerequisite is from another candidate',
                 'full_stage_source_review_passed', 'full_stage_cost_admission', 'timing_result_sha256',
                 '**auth["review_pins"]', 'verify(pins);before={p:sha(p) for p in pins}',
                 'return 0 if accepted and receipt["sources_unchanged"] else 1'):
        require(text in new_driver, "expected inherited child guard absent: " + text)
    require("'-I','-B'" in outer and "report.get('passed')" in outer and "before!=pins" in outer,
            "outer exact-route/result/pin guards missing")

    verify(all_pins)
    for row in capture_manifest["files"] + records:
        require(sha(row["origin"]) == row["sha256"] and sha(row["copy"]) == row["sha256"], "postreview drift")
    save(ROOT / "pins-after.json", all_pins)
    result = {"passed": True, "source_review_only": True, "inputs_unchanged": True,
              "reviewed_source_index_sha256": EXPECTED_INDEX, "recipe_sha256": EXPECTED_RECIPE,
              "driver_sha256": sha(INPUT / "run_oracle.py"), "diff_sha256": sha(INPUT / "v2-to-v3.diff"),
              "captured_source_files": len(capture_manifest["files"]), "protected_external_pins": len(recipe["protected_inputs"]),
              "unique_total_input_pins": len(all_pins), "unchanged_runtime_modules": identical,
              "driver_reverse_bytes_and_AST": True, "recipe_reverse_bytes_and_structure": True,
              "published_full_diff_matches_actual": True, "PLAN_SCHEMA_timing_schema_reverse_bytes": True,
              "science_recipe_unchanged_except_precision_root_settings": True,
              "history": {"v2_units": "completed PASS 318; unchanged inputs", "v2_timing": "completed FAIL 20 records / 6 identities; unchanged inputs",
                          "v2_history_does_not_satisfy_same_source_v3_prerequisites": True,
                          "historical_reviewer_runtime_limit_preserved": True},
              "runtime": {"command": ["python3", "-I", "-B", str(pathlib.Path(__file__).resolve())],
                          "executable": sys.executable, "resolved_executable": str(pathlib.Path(sys.executable).resolve()),
                          "executable_sha256": sha(sys.executable), "version": sys.version, "flags": str(sys.flags),
                          "environment": {k: os.environ.get(k) for k in ["PYTHONOPTIMIZE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}},
              "candidate_imports": False, "target_evaluation": False, "payload_decoding": False,
              "execution_authorized_by_this_review": False,
              "limitations": ["Units return before the later child soft time-cap check; retain a separate 60-second root process-group cap as in v2.",
                              "Higher precision and finite bisection budget provide no uniform connection forward-error theorem or promised numerical pass.",
                              "Fresh same-index units and timing plus exact timing-bound measured-cost/source approval remain required for full stage."]}
    save(ROOT / "static-readback.json", result)
    print(json.dumps({"passed": True, "unique_pins": len(all_pins), "unchanged_runtime_modules": len(identical),
                      "reviewed_source_index_sha256": EXPECTED_INDEX}))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        failure = ROOT / "review-failure.txt"
        if not failure.exists():
            failure.write_text(traceback.format_exc())
        raise
