"""One-shot stdlib-only precision sibling preparation; no oracle imports."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import os
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "build-layer-research/continuum"
OLD = BASE / "manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009"
ASSESSMENT = BASE / "manufactured-Gaussian-third-jet-v2-timing-mechanism-assessment-20261009"
SOURCE_REVIEW = BASE / "manufactured-Gaussian-third-jet-v2-independent-source-review-20261009"
UNIT_REVIEW = BASE / "manufactured-Gaussian-third-jet-v2-units-independent-saved-review-20261009"
TIMING_REVIEW = BASE / "manufactured-Gaussian-third-jet-v2-timing-failure-independent-saved-review-20261009"
RUNTIME_ADDENDUM = BASE / "manufactured-Gaussian-third-jet-v2-timing-review-runtime-addendum-20261009"
LITERAL_PINS = {
    str(OLD / "source-index.json"): "92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3",
    str(OLD / "recipe.json"): "b3fdabcb50016d8bba414f7a4999ff51438d3a1e8e00f6b8a15ca20e1cd76ea2",
    str(OLD / "attempts/units001/receipt.json"): "2cac8cc11cdef7993ea8cb18e2af82d14cfa062d42b8df3c1b0c3ba4e4fed774",
    str(OLD / "attempts/timing001/receipt.json"): "368ffb6f2435efedefac5e24d43ca6550ffe5b7737fcfeab055af1557358c0a1",
    str(SOURCE_REVIEW / "index.json"): "d69f8df5f2b40e1f8434684711edf789affbfa9b93a04930f8a22bf932eda979",
    str(UNIT_REVIEW / "index.json"): "225745a0f7ce9a4414936d15f37b3d77739dfd58d52b8ebbf3a9fe87ebf3441f",
    str(TIMING_REVIEW / "index.json"): "a8d0797e9707c4108fbbc5b228eb9749921daed645abf0140212367115bc8448",
    str(RUNTIME_ADDENDUM / "index.json"): "7ee4afcf83dc458049cc72b876e4a61a2bc56b924cc48d539fb0ef12638e01b2",
    str(ASSESSMENT / "index.json"): "60d6c2b1178193981e5314ea3292ddda4e194f82650e5327f75bf936e1929ca1",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    path = Path(path)
    require(path.suffix not in (".jsonl", ".npz", ".npy") and path.stat().st_size <= 1048576,
            "large payload decoding is outside source preparation")
    return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def replace_once(text, old, new):
    require(text.count(old) == 1, "unexpected replacement anchor: " + old)
    return text.replace(old, new)


def ast_text(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def indexed_pins(folder):
    rows = load(folder / "index.json")["files"]
    if isinstance(rows, dict):
        return {str(Path(name) if Path(name).is_absolute() else folder / name):
                (item if isinstance(item, str) else item["sha256"]) for name, item in rows.items()}
    return {str(folder / item["path"]): item["sha256"] for item in rows}


def verify(pins):
    for name, digest in pins.items():
        require(sha(name) == digest, "protected input drift: " + name)


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "source preparation requires unoptimized isolated -I -B")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "explicit preparation optimization setting required")
    require(not (HERE / "source-index.json").exists(), "source candidate already frozen")
    require(not (HERE / "source-preparation-v3.json").exists(), "one-shot preparation cannot be repeated")
    verify(LITERAL_PINS)
    original_index = load(OLD / "source-index.json")
    original_recipe = load(OLD / "recipe.json")
    protected = {**original_recipe["protected_inputs"], **original_index["files"], **LITERAL_PINS}
    for folder in (ASSESSMENT, SOURCE_REVIEW, UNIT_REVIEW, TIMING_REVIEW, RUNTIME_ADDENDUM):
        protected.update(indexed_pins(folder))
        protected[str(folder / "index.json")] = sha(folder / "index.json")
    units = load(OLD / "attempts/units001/receipt.json")
    timing = load(OLD / "attempts/timing001/receipt.json")
    for stage, receipt in (("units", units), ("timing", timing)):
        require(receipt["stage"] == stage and receipt["completed"] is True and receipt["sources_unchanged"] is True,
                "historical completion/provenance mismatch")
        require(receipt["source_index_sha256"] == LITERAL_PINS[str(OLD / "source-index.json")], "historical candidate mismatch")
        folder = OLD / "attempts" / (stage + "001")
        protected.update({str(folder / name): digest for name, digest in receipt["output_hashes"].items()})
    unit_result = load(OLD / "attempts/units001/result.json")
    timing_result = load(OLD / "attempts/timing001/result.json")
    require(units["passed"] is True and unit_result["passed"] is True and unit_result["checks"] == 318
            and unit_result["failed_total"] == 0, "actual v2 unit PASS prerequisite history mismatch")
    require(timing["passed"] is False and timing_result["passed"] is False
            and (timing_result["records"], timing_result["failed_total"]) == (20, 6), "actual v2 timing FAIL history mismatch")
    addendum = load(RUNTIME_ADDENDUM / "ADDENDUM.json")
    require(addendum["explicit_I_B_flags"] is False and addendum["historical_reviewer_interpreter_resolved_path_recorded"] is False,
            "historical reviewer-runtime limitation must be retained")
    verify(protected)
    before = dict(protected)

    # Preserve the complete immutable v2 source capsule, not its scientific payloads.
    history = HERE / "source-history/v2-final"
    for name, digest in original_index["files"].items():
        old_path = Path(name)
        relative = old_path.relative_to(OLD)
        destination = history / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_path, destination)
        require(sha(destination) == digest, "history copy drift")
    shutil.copyfile(OLD / "source-index.json", history / "source-index.json")

    numerical = ("taylor3.py", "reference3.py", "gaussian3.py", "geometry.py", "oracle.py",
                 "units.py", "values_context.py", "diagnostics.py", "outer_once.py")
    unchanged = []
    for name in numerical:
        shutil.copyfile(OLD / name, HERE / name)
        require((HERE / name).read_bytes() == (OLD / name).read_bytes(), "unchanged numerical source mismatch")
        unchanged.append({"path": name, "sha256": sha(HERE / name), "exact_bytes_v2": True,
                          "AST_equal_v2": ast_text((HERE / name).read_text()) == ast_text((OLD / name).read_text())})

    old_driver = (OLD / "run_oracle.py").read_text()
    old_label, new_label = "110_vs_150_compact_component", "180_vs_220_compact_component"
    new_driver = replace_once(old_driver, old_label, new_label)
    (HERE / "run_oracle.py").write_text(new_driver)
    restored_driver = replace_once(new_driver, new_label, old_label)
    require(restored_driver == old_driver and ast_text(restored_driver) == ast_text(old_driver), "driver reverse byte/AST failure")

    recipe = json.loads(json.dumps(original_recipe))
    recipe["status"] = "HELD source-only Gaussian v3 precision sibling: 180/220 digits, unchanged formulas/gates/registry"
    recipe["levels"] = [
        {"digits": 180, "roots": {"absolute": "1e-155", "width": "1e-160", "newton": 16, "bisection": 768}},
        {"digits": 220, "roots": {"absolute": "1e-195", "width": "1e-200", "newton": 16, "bisection": 768}}]
    recipe["protected_inputs"] = protected
    lineage = {
        "v2_source_index": {"path": str(OLD / "source-index.json"), "sha256": sha(OLD / "source-index.json")},
        "v2_units_receipt": {"path": str(OLD / "attempts/units001/receipt.json"), "sha256": sha(OLD / "attempts/units001/receipt.json"), "classification": "PASS_318"},
        "v2_timing_receipt": {"path": str(OLD / "attempts/timing001/receipt.json"), "sha256": sha(OLD / "attempts/timing001/receipt.json"), "classification": "COMPLETE_FAIL_20_RECORDS_6_IDENTITIES"},
        "v2_failure_saved_review": {"path": str(TIMING_REVIEW / "index.json"), "sha256": sha(TIMING_REVIEW / "index.json")},
        "v2_failure_saved_review_runtime_addendum": {"path": str(RUNTIME_ADDENDUM / "index.json"), "sha256": sha(RUNTIME_ADDENDUM / "index.json")},
        "precision_mechanism_proposal": {"path": str(ASSESSMENT / "index.json"), "sha256": sha(ASSESSMENT / "index.json")},
        "historical_timing_reviewer_isolated_route_not_recorded": True,
        "historical_v2_unit_pass_does_not_authorize_v3_timing": True,
        "fresh_same_index_units_timing_and_measured_cost_review_required": True}
    recipe["v3_precision_lineage"] = lineage
    save(HERE / "recipe.json", recipe)
    restored_recipe = json.loads((HERE / "recipe.json").read_text())
    for name in ("status", "levels", "protected_inputs"):
        restored_recipe[name] = original_recipe[name]
    restored_recipe.pop("v3_precision_lineage")
    require(restored_recipe == original_recipe, "nonprecision scientific recipe change")
    original_encoded = json.dumps(original_recipe, indent=2, allow_nan=False) + "\n"
    require((OLD / "recipe.json").read_text() == original_encoded, "unexpected original recipe serialization")
    require(json.dumps(restored_recipe, indent=2, allow_nan=False) + "\n" == (OLD / "recipe.json").read_text(),
            "recipe reverse exact bytes failure")

    plan_original = (OLD / "PLAN.md").read_text()
    plan_precision = replace_once(plan_original, "and5,010 at110/150 digits.", "and5,010 at180/220 digits.")
    plan_addendum = """

## Fresh v3 precision-only sibling and preserved failure

The frozen v2 actual units passed all 318 checks. Its subsequent timing stage completed 20 records but FAILED six conformal-connection identities, all introduced at 110 digits for nominal r=1-10^-18 and no new failures at 150 digits. The original receipts/results/source capsule and independent saved audit remain immutable. The separately pinned runtime addendum states that the historical saved reviewer did not record an isolated invocation, resolved interpreter or environment; this sibling does not retroactively infer those facts.

V3 changes only the two geometry levels to 180/220 digits, their scalar residual gates to 1e-155/1e-195 and numerical widths to 1e-160/1e-200, and their bisection budgets to 768. Newton remains 16. The compact precision branch label is truthfully 180_vs_220_compact_component. All nine other runtime mathematical/outer modules are byte-identical; the driver's only mathematical-output change is that label. The full registry, nominal admission, all counts, 1e-55 identity/component thresholds, 80-digit unit recipe, unit roots, height 128/256 and separate 1e-30 context, resource/payload caps and raw physical-reference RWM scope are unchanged.

The pinned source assessment estimates up to four inverse powers of Omega of raw embedding-connection conditioning; it is not a uniform forward-error theorem or a promised PASS. No connection comparator is replaced or disabled. Original failed rows remain gated. A 768-step pure-bisection fallback supports the tighter numerical widths for initial spans below 2, without relying on Newton success. Endpoint signs/residuals and width remain numerical checks, not interval enclosures.

The old v2 units/timing are provenance, not same-index admission for v3. Fresh v3 units and timing need separate exact root releases and source/independent review. The full stage remains held until actual same-index units and timing PASS plus an explicit timing-receipt/result-bound source and measured-cost approval record. Preparing this sibling performs stdlib hashing/AST/text bookkeeping only; no oracle import, units, timing, full registry, native query, inverse coverage, BH adoption or equation replacement occurs.
"""
    (HERE / "PLAN.md").write_text(plan_precision + plan_addendum)
    restored_plan = replace_once((HERE / "PLAN.md").read_text(), plan_addendum, "")
    restored_plan = replace_once(restored_plan, "and5,010 at180/220 digits.", "and5,010 at110/150 digits.")
    require(restored_plan == plan_original, "PLAN reverse exact bytes failure")
    schema_original = (OLD / "SCHEMA.md").read_text()
    schema_new = replace_once(schema_original, "Values110/150 are retained", "Values180/220 are retained")
    (HERE / "SCHEMA.md").write_text(schema_new)
    require(replace_once(schema_new, "Values180/220 are retained", "Values110/150 are retained") == schema_original,
            "SCHEMA reverse exact bytes failure")
    measured_original = (OLD / "measured-timing-review-schema.json").read_text()
    measured_new = replace_once(measured_original, "exact current v2 source-index hash", "exact current v3 source-index hash")
    (HERE / "measured-timing-review-schema.json").write_text(measured_new)
    require(replace_once(measured_new, "exact current v3 source-index hash", "exact current v2 source-index hash") == measured_original,
            "measured schema reverse exact bytes failure")
    shutil.copyfile(OLD / "authorization-schema.json", HERE / "authorization-schema.json")
    save(HERE / "historical-v2-context.json", lineage)

    changed = ("run_oracle.py", "recipe.json", "PLAN.md", "SCHEMA.md", "measured-timing-review-schema.json")
    diff = "".join("".join(difflib.unified_diff((OLD / name).read_text().splitlines(True),
                    (HERE / name).read_text().splitlines(True), fromfile="v2/" + name, tofile="v3/" + name))
                    for name in changed)
    (HERE / "v2-to-v3.diff").write_text(diff)
    modules = []
    for path in sorted(HERE.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
        modules.append({"path": str(path), "sha256": sha(path), "AST_parse_only": True})
    verify(before)
    save(HERE / "source-preparation-v3.json", {
        "source_only": True, "candidate_imports": False, "arithmetic_or_units_calls": False,
        "compiler_or_kernel_calls": False, "payload_decode": False,
        "command": sys.argv, "preparation_interpreter": str(Path(sys.executable).resolve()),
        "preparation_interpreter_sha256": sha(Path(sys.executable).resolve()),
        "preparation_flags": {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode,
                              "optimize": sys.flags.optimize},
        "preparation_environment": {"PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")},
        "launch_HEAD": "284b4c21e09077ab86f0d0cbbbb5b3a11503cf58",
        "v2_source_index_sha256": sha(OLD / "source-index.json"),
        "protected_external_inputs": len(before), "inputs_unchanged": True,
        "driver_sole_precision_label_change": {"exact_forward_bytes": True, "exact_reverse_bytes": True,
             "reverse_AST": True, "old_sha256": sha(OLD / "run_oracle.py"), "new_sha256": sha(HERE / "run_oracle.py")},
        "unchanged_runtime_numerical_and_outer_sources": unchanged,
        "recipe_reverse_exact_bytes": True, "recipe_reverse_JSON_structure": True,
        "recipe_changes_only_status_levels_provenance_pins": True,
        "PLAN_SCHEMA_measured_schema_exact_reverse_bytes": True,
        "unit_recipe_thresholds_registry_counts_caps_unchanged": True,
        "all_levels_bisection_768_newton_16": True,
        "historical_reviewer_runtime_limit_pinned": True,
        "old_v2_actual_unit_PASS_and_timing_FAIL_preserved": True,
        "AST_only_modules": modules, "execution_admitted": False})
    files = sorted(path for path in HERE.rglob("*") if path.is_file())
    mapping = {str(path): sha(path) for path in files}
    save(HERE / "source-index.json", {"status": "immutable SOURCE-ONLY held Gaussian v3 precision sibling",
         "source_only": True, "execution_admitted": False, "file_count": len(mapping), "files": mapping})
    verify({**before, **mapping})
    print(json.dumps({"source_index_sha256": sha(HERE / "source-index.json"), "recipe_sha256": sha(HERE / "recipe.json"),
          "driver_sha256": sha(HERE / "run_oracle.py"), "outer_sha256": sha(HERE / "outer_once.py"),
          "diff_sha256": sha(HERE / "v2-to-v3.diff"), "files": len(mapping), "protected_external_inputs": len(before),
          "candidate_imports": False, "scientific_execution": False}))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        failure = HERE / "source-preparation-v3-failure.txt"
        if not failure.exists():
            failure.write_text(traceback.format_exc())
        raise
