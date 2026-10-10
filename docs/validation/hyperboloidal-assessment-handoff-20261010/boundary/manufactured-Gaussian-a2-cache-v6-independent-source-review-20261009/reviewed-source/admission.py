"""Standard-library admission only. Every mathematical import follows checks."""
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    def reject(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def verify_pin(entry):
    p = Path(entry["path"])
    if not p.is_file() or p.stat().st_size != entry["bytes"] or digest(p) != entry["sha256"]:
        raise RuntimeError("pin mismatch: " + str(p))


def verify_receipt(pin, stage, index_sha):
    verify_pin(pin)
    data = load(pin["path"])
    if not (data.get("completed") is True and type(data.get("returncode")) is int and data.get("returncode") == 0
            and data.get("passed") is True and data.get("inputs_unchanged") is True
            and data.get("stage") == stage and data.get("source_index_sha256") == index_sha):
        raise RuntimeError("unsuccessful/wrong prerequisite receipt")
    return data


def check(recipe_path, authorization_path, output_path, stage):
    root = Path(__file__).resolve().parent
    if Path(recipe_path).resolve() != root / "recipe.json":
        raise RuntimeError("only the exact local recipe is admitted")
    if not (sys.flags.isolated == 1 and sys.dont_write_bytecode
            and sys.flags.optimize == 0):
        raise RuntimeError("require -I -B and optimization disabled")
    if os.environ.get("PYTHONOPTIMIZE") != "0":
        raise RuntimeError("require explicit PYTHONOPTIMIZE=0")
    authorization = load(authorization_path)
    index_path = root / "source-index.json"
    index_sha = digest(index_path)
    recipe_sha = digest(root / "recipe.json")
    output = Path(output_path).resolve()
    if not (authorization.get("allow_execution") is True
            and authorization.get("stage") == stage
            and authorization.get("source_index_sha256") == index_sha
            and authorization.get("recipe_sha256") == recipe_sha
            and authorization.get("output") == str(output)):
        raise RuntimeError("missing exact external stage authorization")
    if output.parent != root / "attempts":
        raise RuntimeError("output must be an immediate fresh attempts child")
    index, recipe = load(index_path), load(recipe_path)
    for entry in index["files"]:
        verify_pin(entry)
    for entry in recipe["protected_inputs"]:
        verify_pin(entry)
    verify_pin(recipe["python"])
    if Path(sys.executable).resolve() != Path(recipe["python"]["path"]).resolve():
        raise RuntimeError("wrong interpreter")
    if recipe["source_only"] is not True or stage not in recipe["stage_timeouts"]:
        raise RuntimeError("invalid held recipe")
    if stage != "certificate":
        raise RuntimeError("cache v6 instrumentation admits only a fresh diagnostic producer")
    # Admission/history only: the old cached producer remains failed.
    prior_pin = recipe["cached_v4_timeout_receipt"]
    if authorization.get("prior_timeout_receipt") != prior_pin:
        raise RuntimeError("exact failed cached-v4 receipt pin required")
    verify_pin(prior_pin)
    prior = load(prior_pin["path"])
    if not (prior.get("completed") is False and prior.get("passed") is False
            and type(prior.get("returncode")) is int and prior.get("returncode") == 1
            and prior.get("inputs_unchanged") is True and prior.get("stage") == "certificate"
            and prior.get("source_index_sha256") == recipe["cached_v4_source_index_sha256"]
            and prior.get("recipe_sha256") == recipe["cached_v4_recipe_sha256"]):
        raise RuntimeError("cached-v4 timeout history classification differs")
    for pin in (recipe["cached_v4_progress"], recipe["cached_v4_stderr"],
                recipe["cached_v4_command"], recipe["cached_v4_partial_certificate_metadata"]):
        if pin not in prior.get("outputs", []):
            raise RuntimeError("failed cached-v4 receipt does not bind history payload")
    for pin in (recipe["cached_v4_progress"], recipe["cached_v4_stderr"], recipe["cached_v4_command"]):
        verify_pin(pin)
    if "UNRESOLVED: declared domain time limit" not in Path(recipe["cached_v4_stderr"]["path"]).read_text():
        raise RuntimeError("cached-v4 failure was not the declared time cap")
    if (Path(prior_pin["path"]).parent / "report.json").exists():
        raise RuntimeError("failed cached-v4 unexpectedly has a producer report")
    # The partial tree is bound as metadata only; no decode or resume.
    # BEGIN_OBSERVATION_ONLY
    v5_pin = recipe["cached_v5_timeout_receipt"]
    if authorization.get("prior_v5_timeout_receipt") != v5_pin:
        raise RuntimeError("exact failed cached-v5 receipt pin required")
    verify_pin(v5_pin)
    v5 = load(v5_pin["path"])
    if not (v5.get("completed") is False and v5.get("passed") is False
            and type(v5.get("returncode")) is int and v5.get("returncode") == 1
            and v5.get("inputs_unchanged") is True and v5.get("stage") == "certificate"
            and v5.get("source_index_sha256") == recipe["cached_v5_source_index_sha256"]
            and v5.get("recipe_sha256") == recipe["cached_v5_recipe_sha256"]):
        raise RuntimeError("cached-v5 timeout history classification differs")
    for entry in (recipe["cached_v5_progress"], recipe["cached_v5_stderr"],
                  recipe["cached_v5_command"], recipe["cached_v5_partial_certificate_metadata"]):
        if entry not in v5.get("outputs", []):
            raise RuntimeError("v5 failed receipt does not bind history output")
    for entry in (recipe["cached_v5_progress"], recipe["cached_v5_stderr"], recipe["cached_v5_command"]):
        verify_pin(entry)
    if "UNRESOLVED: declared domain time limit" not in Path(recipe["cached_v5_stderr"]["path"]).read_text():
        raise RuntimeError("v5 failure is not the declared time cap")
    if (Path(v5_pin["path"]).parent / "report.json").exists():
        raise RuntimeError("failed v5 unexpectedly has a producer report")
    # END_OBSERVATION_ONLY
    dependencies = {"prior_timeout": prior}
    if stage in ("certificate", "replay"):
        unit_pin = recipe["cached_v3_units_receipt"]
        if authorization.get("units_receipt") != unit_pin:
            raise RuntimeError("exact successful cached v3 units pin required")
        units = verify_receipt(unit_pin, "units", recipe["cached_v3_source_index_sha256"])
        if units.get("recipe_sha256") != recipe["cached_v3_recipe_sha256"]:
            raise RuntimeError("wrong cached v3 unit recipe")
        report_pin, cache_pin = recipe["cached_v3_units_report"], recipe["cached_v3_cache_report"]
        for entry in (report_pin, cache_pin, recipe["cached_v3_root_source_review"]):
            verify_pin(entry)
        if report_pin not in units["outputs"] or cache_pin not in units["outputs"]:
            raise RuntimeError("successful v3 receipt does not bind both reports")
        report, cache = load(report_pin["path"]), load(cache_pin["path"])
        review = load(recipe["cached_v3_root_source_review"]["path"])
        if not (report.get("passed") is True and report.get("stage") == "units"
                and report.get("source_index_sha256") == recipe["cached_v3_source_index_sha256"]
                and report.get("case_count") == 144 and report.get("cache_unit_count") == 35
                and report.get("combined_unit_count") == 179 and report.get("cache_units_passed") is True
                and cache.get("passed") is True and cache.get("case_count") == 35
                and len(cache.get("cases", [])) == 35 and all(x.get("passed") is True for x in cache["cases"])
                and review.get("passed") is True
                and review.get("root_full_cache_source_math_and_admission_review") is True
                and review.get("original_uncached_endpoint_body_AST_identical") is True):
            raise RuntimeError("cached v3 unit/report/root-review prerequisite failed")
        dependencies["units"] = units
    if stage == "replay":
        dependencies["certificate"] = verify_receipt(authorization["certificate_receipt"], "certificate", index_sha)
        certificate_pin = authorization["certificate_payload"]
        verify_pin(certificate_pin)
        outputs = dependencies["certificate"]["outputs"]
        if certificate_pin not in outputs:
            raise RuntimeError("certificate is not a bound successful output")
    return root, recipe, authorization, index_sha, dependencies


def unchanged(root, recipe, index_sha):
    if digest(root / "source-index.json") != index_sha:
        raise RuntimeError("source index changed")
    for entry in load(root / "source-index.json")["files"] + recipe["protected_inputs"]:
        verify_pin(entry)
    verify_pin(recipe["python"])
    return True
