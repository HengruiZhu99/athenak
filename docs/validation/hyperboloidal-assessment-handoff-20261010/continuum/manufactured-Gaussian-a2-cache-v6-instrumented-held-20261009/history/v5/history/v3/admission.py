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
    if stage != "units":
        raise RuntimeError("cache v3 admits units only; certificate/replay remain held")
    dependencies = {}
    if stage in ("certificate", "replay"):
        dependencies["units"] = verify_receipt(authorization["units_receipt"], "units", index_sha)
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
