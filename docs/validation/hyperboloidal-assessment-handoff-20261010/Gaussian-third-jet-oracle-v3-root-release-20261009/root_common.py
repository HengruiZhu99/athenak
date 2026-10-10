"""Root stdlib-only pins/proofs for the held Gaussian v3 units release."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OWNER = BASE / "continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009"
OLD = BASE / "continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009"
IDENTITIES = {
    "source_index_sha256": "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a",
    "recipe_sha256": "847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf",
    "driver_sha256": "de0b83ba44f1a837c2b477d6d60f5a8f2a947b8a885159a638bbbaeee94e7915",
    "outer_sha256": "68706de821c990be9302848aeea41d16d774c506bb8dea71221b1e2a675cea04",
    "v2_source_index_sha256": "92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3",
}
EXPECTED_BASE_PINS = 437
UNCHANGED = ("taylor3.py", "reference3.py", "gaussian3.py", "geometry.py", "oracle.py",
             "units.py", "values_context.py", "diagnostics.py", "outer_once.py")


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
            "scientific payload decoding is forbidden in root source preparation")
    return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def guard():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "root scripts require unoptimized isolated -I -B")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "root scripts require explicit PYTHONOPTIMIZE=0")


def merge(destination, source):
    for name, digest in source.items():
        name = str(Path(name).resolve())
        require(name not in destination or destination[name] == digest, "conflicting pin: " + name)
        destination[name] = digest


def verify(pins):
    for name, digest in pins.items():
        require(sha(name) == digest, "protected input drift: " + name)


def base_pins():
    for name, key in (("source-index.json", "source_index_sha256"), ("recipe.json", "recipe_sha256"),
                      ("run_oracle.py", "driver_sha256"), ("outer_once.py", "outer_sha256")):
        require(sha(OWNER / name) == IDENTITIES[key], "typed Gaussian v3 identity mismatch: " + name)
    index, recipe = load(OWNER / "source-index.json"), load(OWNER / "recipe.json")
    require(index["source_only"] is True and index["execution_admitted"] is False, "owner source-only history changed")
    require(len(index["files"]) == 58 and len(recipe["protected_inputs"]) == 378, "fixed source/runtime inventory count differs")
    pins = {str(OWNER / "source-index.json"): IDENTITIES["source_index_sha256"]}
    merge(pins, index["files"])
    merge(pins, recipe["protected_inputs"])
    require(len(pins) == EXPECTED_BASE_PINS, "437 base source/runtime/history pins are required")
    verify(pins)
    return pins, recipe


def scripts_pins(expected_index_sha):
    path = HERE / "root-scripts-source-index001.json"
    require(sha(path) == expected_index_sha, "root scripts index admission mismatch")
    index = load(path)
    pins = {str(path): expected_index_sha}
    for name, digest in index["files"].items():
        target = Path(name).resolve()
        require(target.parent == HERE, "root source index escapes its fixed prefix")
        pins[str(target)] = digest
    required = {"root_common.py", "prepare_review001.py", "finalize_review001.py", "prepare_units001.py", "launch_units.py", "PREPARED-PLAN.md", "prepared-readiness001.json"}
    require({Path(name).name for name in index["files"]} == required, "root source script inventory differs")
    verify(pins)
    return pins


def ast_text(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def source_proofs(recipe):
    require(sha(OLD / "source-index.json") == IDENTITIES["v2_source_index_sha256"], "historical v2 index changed")
    unchanged = []
    for name in UNCHANGED:
        left, right = (OWNER / name).read_bytes(), (OLD / name).read_bytes()
        require(left == right and ast_text(left) == ast_text(right), "nine-body byte/AST mismatch: " + name)
        unchanged.append({"path": name, "sha256": sha(OWNER / name), "byte_identical": True, "AST_identical": True})
    old_text, new_text = (OLD / "run_oracle.py").read_text(), (OWNER / "run_oracle.py").read_text()
    old_label, new_label = "110_vs_150_compact_component", "180_vs_220_compact_component"
    require(old_text.count(old_label) == 1 and new_text.count(new_label) == 1, "precision label anchors differ")
    require(old_text.replace(old_label, new_label) == new_text, "driver changed beyond precision output label")
    restored = new_text.replace(new_label, old_label)
    require(restored == old_text and ast_text(restored) == ast_text(old_text), "driver exact reverse bytes/AST mismatch")
    old_recipe = load(OLD / "recipe.json")
    require(recipe["levels"] == [
        {"digits": 180, "roots": {"absolute": "1e-155", "width": "1e-160", "newton": 16, "bisection": 768}},
        {"digits": 220, "roots": {"absolute": "1e-195", "width": "1e-200", "newton": 16, "bisection": 768}}],
        "approved precision/root values differ")
    restored_recipe = json.loads(json.dumps(recipe))
    for name in ("status", "levels", "protected_inputs"):
        restored_recipe[name] = old_recipe[name]
    restored_recipe.pop("v3_precision_lineage")
    require(restored_recipe == old_recipe, "recipe changed beyond approved precision and history metadata")
    require(json.dumps(restored_recipe, indent=2, allow_nan=False) + "\n" == (OLD / "recipe.json").read_text(),
            "recipe reverse exact bytes mismatch")
    require(recipe["unit_digits"] == 80 and recipe["unit_height_order"] == 256 and
            recipe["unit_roots"] == {"absolute": "1e-60", "width": "1e-65", "newton": 16, "bisection": 512},
            "unchanged 80-digit unit recipe required")
    require(recipe["identity_tolerance"] == recipe["component_tolerance"] == "1e-55", "threshold drift")
    require(recipe["expected_counts"]["units"] == 318 and recipe["expected_counts"]["timing_records"] == 20
            and recipe["expected_counts"]["full_records"] == 5010 and recipe["expected_counts"]["full_identity_checks"] == 3330320
            and recipe["expected_counts"]["full_component_checks"] == 470752, "fixed scientific counts differ")
    require(recipe["stage_seconds"] == {"units": 60, "timing": 600, "full": 14400}
            and recipe["outer_seconds"] == {"units": 120, "timing": 660, "full": 14460}, "owner resource caps drift")
    lineage = recipe["v3_precision_lineage"]
    u = load(lineage["v2_units_receipt"]["path"])
    t = load(lineage["v2_timing_receipt"]["path"])
    require(u["completed"] is True and u["passed"] is True and u["sources_unchanged"] is True, "historical v2 unit PASS drift")
    require(t["completed"] is True and t["passed"] is False and t["sources_unchanged"] is True, "historical v2 timing FAIL drift")
    require(lineage["historical_timing_reviewer_isolated_route_not_recorded"] is True, "review runtime addendum lost")
    prepared = load(OWNER / "source-preparation-v3.json")
    require(prepared["driver_sole_precision_label_change"]["exact_reverse_bytes"] is True and
            prepared["driver_sole_precision_label_change"]["reverse_AST"] is True and
            prepared["unit_recipe_thresholds_registry_counts_caps_unchanged"] is True, "owner proof scope changed")
    return {"nine_unchanged_bodies": unchanged, "driver_reverse_exact_bytes_AST": True,
            "recipe_reverse_exact_bytes_JSON": True, "only_approved_precision_root_and_metadata_changes": True,
            "unit_recipe_registry_thresholds_counts_caps_unchanged": True,
            "v2_units_PASS_timing_FAIL_and_reviewer_runtime_limit_preserved": True}


def root_review():
    preparation = load(HERE / "review-preparation001.json")
    review = load(HERE / "source-review001.json")
    require(review["passed"] is True and review["root_full_source_math_and_admission_review"] is True,
            "root full static review prerequisite absent")
    require(review["source_index_sha256"] == IDENTITIES["source_index_sha256"] and review["protected_base_pins"] == 437,
            "root review candidate/inventory mismatch")
    return preparation, review


def independent_review(folder, expected_index, expected_receipt):
    folder = Path(folder).resolve()
    require(sha(folder / "index.json") == expected_index and sha(folder / "receipt.json") == expected_receipt,
            "independent review admission pin mismatch")
    receipt, index = load(folder / "receipt.json"), load(folder / "index.json")
    require(receipt.get("passed") is True and receipt.get("reviewed_source_index_sha256") == IDENTITIES["source_index_sha256"],
            "independent source/math/admission PASS for exact v3 required")
    pins = {str(folder / "index.json"): expected_index, str(folder / "receipt.json"): expected_receipt}
    rows = index["files"]
    if isinstance(rows, dict):
        iterable = [(name, item if isinstance(item, str) else item["sha256"]) for name, item in rows.items()]
    else:
        iterable = [(item["path"], item["sha256"]) for item in rows]
    for name, digest in iterable:
        path = Path(name)
        path = path if path.is_absolute() else folder / path
        require(path.resolve().is_relative_to(folder), "independent review index escapes its prefix")
        pins[str(path.resolve())] = digest
    verify(pins)
    return pins


def metadata_phase(name, function):
    output = HERE / name
    output.mkdir(exist_ok=False)
    receipt = {"completed": False, "passed": False, "metadata_only": True, "candidate_imports": False,
               "numeric_calls": False, "command": sys.argv, "interpreter": str(Path(sys.executable).resolve()),
               "flags": {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode, "optimize": sys.flags.optimize},
               "environment": {"PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")}}
    pins = {}
    try:
        guard()
        result = function(output, pins)
        verify(pins)
        receipt.update(completed=True, passed=True, protected_inputs=len(pins), inputs_unchanged=True, result=result)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
        try:
            verify(pins)
            receipt["inputs_unchanged"] = bool(pins)
        except BaseException as post_error:
            receipt.update(inputs_unchanged=False, post_error=str(post_error))
    write(output / "receipt.json", receipt)
    print(json.dumps(receipt, allow_nan=False))
    return 0 if receipt["passed"] else 1
