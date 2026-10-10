"""Stdlib-only timing bookkeeping; imports only the pinned old root stdlib helper."""
from pathlib import Path
import hashlib
import importlib.util
import os
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
UNIT_ROOT = BASE / "Gaussian-third-jet-oracle-v3-root-release-20261009"
UNIT_REVIEW = BASE / "continuum/manufactured-Gaussian-third-jet-v3-units-independent-saved-review-20261009"
SOURCE_REVIEW = BASE / "continuum/manufactured-Gaussian-third-jet-v3-independent-source-review-20261009"
COMMON = UNIT_ROOT / "root_common.py"
COMMON_SHA = "83fcc9cfc666c11a9001812b6b5ce6a3c5d363a2372bc8797d6637fcc085aad7"
if hashlib.sha256(COMMON.read_bytes()).hexdigest() != COMMON_SHA:
    raise RuntimeError("pinned read-only units-root stdlib helper changed before import")
spec = importlib.util.spec_from_file_location("gaussian_v3_frozen_units_root_common", COMMON)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

SOURCE_INDEX = "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a"
UNIT_RECEIPT = "d41b648021152f62c7972c483be08c2f70d6b949ad01cecabe7b621787f34e16"
UNIT_RESULT = "c569c2e3d8fe884e3a4f14723fc91ce7c25c8cf447b577cff6ddb8244c98db89"
UNIT_OUTER = "79e973af0a68a41db0da0b462d940ba9118c6ac90a05fc07d9deeff73388cd18"
UNIT_ENCLOSING = "6889815e4b18dea584e2fba46a094f334b7e45e068910540ab48aab4afb875a4"
UNIT_REVIEW_INDEX = "151feb06c55bf3ed3aad70c6dc3e8a8adee5a3020cfae3772f58e5b46f068b1b"
UNIT_REVIEW_RECEIPT = "4120ae1d1d07ee107c9c8bb741fb3f12b18657037b372605e75f0742f93cbfb9"
SOURCE_REVIEW_INDEX = "6773bbe3899c06db8e1486acf7a6cc964448983b950936462c0b017cf64f8988"
SOURCE_REVIEW_RECEIPT = "bce50b583db35f3a9fb9728abf91213174554c23ada0324fd79662dd2330828e"
ROOT_PYTHON = Path("/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9")
ROOT_PYTHON_SHA = "6e7ae61f68a3838094fc56590f84c52069a97d7816f6bb79e0e85995b340e464"


def guard():
    c.guard()
    c.require(Path(sys.executable).resolve() == ROOT_PYTHON and c.sha(Path(sys.executable).resolve()) == ROOT_PYTHON_SHA,
              "exact pinned CLT root interpreter required")


def source_pins(expected_index):
    path = HERE / "timing-root-source-index001.json"
    c.require(c.sha(path) == expected_index, "exact held timing source-index hash required")
    index = c.load(path)
    c.require(index["source_only"] is True and index["execution_admitted"] is False,
              "held source history must remain source only")
    pins = {str(path): expected_index}
    for name, digest in index["files"].items():
        p = Path(name).resolve()
        c.require(p.is_relative_to(HERE) and p.is_file(), "timing source index escapes its prefix")
        c.merge(pins, {str(p): digest})
    required = {"timing_support.py", "prepare_timing001.py", "launch_timing.py", "PREPARED-PLAN.md", "prepared-readiness001.json"}
    c.require(required <= {Path(name).name for name in index["files"]}, "timing source inventory incomplete")
    c.verify(pins)
    return pins


def prerequisites(review_folder, review_index, review_receipt):
    c.require(Path(review_folder).resolve() == UNIT_REVIEW and review_index == UNIT_REVIEW_INDEX
              and review_receipt == UNIT_REVIEW_RECEIPT, "exact completed v3 saved-unit review is required")
    pins, recipe = c.base_pins()
    c.require(len(pins) == 437 and c.load(UNIT_ROOT / "source-pins001.json") == pins,
              "exact 437 root-reviewed base pins required")
    c.source_proofs(recipe)
    prior_release = c.load(UNIT_ROOT / "units-release.json")
    c.require(prior_release["stage"] == "units" and prior_release["required_base_pins"] == 437,
              "historical units-only release required")
    c.merge(pins, prior_release["pins"])
    c.merge(pins, c.independent_review(SOURCE_REVIEW, SOURCE_REVIEW_INDEX, SOURCE_REVIEW_RECEIPT))
    c.merge(pins, c.independent_review(UNIT_REVIEW, UNIT_REVIEW_INDEX, UNIT_REVIEW_RECEIPT))
    unit_path = c.OWNER / "attempts/units001/receipt.json"
    literal = {str(unit_path): UNIT_RECEIPT, str(unit_path.parent / "result.json"): UNIT_RESULT,
               str(UNIT_ROOT / "units-outer001/receipt.json"): UNIT_OUTER,
               str(UNIT_ROOT / "units-invocation001/receipt.json"): UNIT_ENCLOSING,
               str(COMMON): COMMON_SHA}
    c.merge(pins, literal)
    c.verify(pins)
    unit = c.load(unit_path)
    result = c.load(unit_path.parent / "result.json")
    root = c.load(UNIT_ROOT / "units-invocation001/receipt.json")
    outer = c.load(UNIT_ROOT / "units-outer001/receipt.json")
    audit = c.load(UNIT_REVIEW / "receipt.json")
    c.require(unit["stage"] == "units" and all(unit[k] is True for k in ("completed", "passed", "sources_unchanged"))
              and unit["source_index_sha256"] == SOURCE_INDEX, "same-source actual unit PASS required")
    c.require(result == {"passed": True, "records": 0, "checks": 318, "failed": [], "failed_total": 0},
              "fixed actual318 unit result required")
    for enclosing in (root, outer):
        c.require(enclosing["completed"] is True and enclosing["accepted_stage"] is True and enclosing["inputs_unchanged"] is True
                  and type(enclosing["returncode"]) is int and enclosing["returncode"] == 0,
                  "actual enclosing unit PASS required")
        c.require(enclosing["child_receipt_sha256"] == UNIT_RECEIPT and enclosing["result_sha256"] == UNIT_RESULT,
                  "actual enclosing unit result binding")
    c.require(root["root_process_group_cap_seconds"] == 60 and root["elapsed_seconds"] < 60
              and root["outer_receipt_sha256"] == UNIT_OUTER, "actual unit group cap/outer binding")
    c.require(audit["passed"] is True and audit["saved_only"] is True and audit["inputs_unchanged"] is True
              and audit["checks"] == 318 and audit["failed_total"] == 0 and audit["root_process_group_cap_seconds"] == 60,
              "independent saved318 audit required")
    expected_audit = {"reviewed_source_index_sha256": SOURCE_INDEX, "child_receipt_sha256": UNIT_RECEIPT,
                      "result_sha256": UNIT_RESULT, "outer_receipt_sha256": UNIT_OUTER, "root_receipt_sha256": UNIT_ENCLOSING}
    c.require(all(audit.get(key) == value for key, value in expected_audit.items()), "saved-unit review exact provenance mismatch")
    for folder in (UNIT_ROOT, unit_path.parent):
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                c.merge(pins, {str(p): c.sha(p)})
    for rel, digest in unit["output_hashes"].items():
        c.merge(pins, {str(unit_path.parent / rel): digest})
    for row in root["output_inventory"]:
        c.require(Path(row["path"]).stat().st_size == row["bytes"] and c.sha(row["path"]) == row["sha256"], "unit inventory drift")
        c.merge(pins, {row["path"]: row["sha256"]})
    c.require(len(recipe["timing_keys"]) == 10 and recipe["expected_counts"]["timing_records"] == 20
              and recipe["stage_seconds"]["timing"] == 600 and recipe["outer_seconds"]["timing"] == 660,
              "fixed timing keys/count/caps changed")
    c.verify(pins)
    return pins, recipe, unit_path
