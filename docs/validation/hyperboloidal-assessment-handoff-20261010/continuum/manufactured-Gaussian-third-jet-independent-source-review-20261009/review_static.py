"""Saved source/metadata readback only; never imports any reviewed candidate."""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
CAPTURED = HERE / "captured"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def main():
    output = HERE / "static-readback.json"
    if output.exists():
        raise RuntimeError("one-shot fresh saved-source readback only")
    result = {"saved_source_metadata_only": True, "candidate_imports": False,
              "target_arithmetic": False, "CAS_compile_queries_eigen_evolution": False,
              "argv": sys.argv, "source_sha256": sha(__file__)}
    frozen = load(HERE / "inputs-before-review.json")
    originals = {}
    for item in frozen["inputs"]:
        for kind in ("original", "captured"):
            pin = item[kind]
            if Path(pin["path"]).stat().st_size != pin["bytes"] or sha(pin["path"]) != pin["sha256"]:
                raise RuntimeError("changed pre-read pin: " + pin["path"])
        originals[item["original"]["path"]] = item["original"]["sha256"]
    index = load(CAPTURED / "source-index.json")
    recipe = load(CAPTURED / "recipe.json")
    if sha(CAPTURED / "source-index.json") != "2348bbf4391444606dc2f946067a3a75eb55d472cc0c738795d8f0617f161d88":
        raise RuntimeError("wrong reviewed index")
    if len(index["files"]) != index["file_count"] or len(originals) != index["file_count"] + 1:
        raise RuntimeError("source closure count mismatch")
    protected = {**index["files"], **recipe["protected_inputs"]}
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError("protected input drift: " + path)
    parsed = []
    for path in sorted(CAPTURED.rglob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
        parsed.append(str(path.relative_to(CAPTURED)))
    radii = [Fraction(r) for r in recipe["radii"]]
    if len(radii) != len(set(radii)) or len(radii) != 25:
        raise RuntimeError("radius registry mismatch")
    directions = len(recipe["angular_p"]) + 4
    per_epsilon = len(recipe["times"]) * (1 + (len(radii)-1)*directions)
    valid_per_level = len(recipe["epsilon"]) * per_epsilon
    component_per_valid = 11*10 + 11*4 + 22 + 10 + 2
    native_per_epsilon = len(recipe["times"]) * (1 + (sum(r <= Fraction(".98") for r in radii)-1)*directions)
    counts = {"directions": directions, "per_epsilon": per_epsilon,
              "valid_per_level": valid_per_level, "full_records": 2*(valid_per_level+1),
              "compact_components_per_valid": component_per_valid,
              "full_component_checks": valid_per_level*component_per_valid,
              "full_identity_checks": 2*valid_per_level*654 + 2*per_epsilon*22,
              "nominal_native_eligible_valid_cases": len(recipe["epsilon"])*native_per_epsilon}
    for key in ("full_records", "full_component_checks", "full_identity_checks", "nominal_native_eligible_valid_cases"):
        if counts[key] != recipe["expected_counts"][key]:
            raise RuntimeError("integer registry count mismatch: " + key)
    result.update(metadata_readback_passed=True, source_files=index["file_count"],
                  captured_files=len(originals), external_pins=len(recipe["protected_inputs"]),
                  unique_source_external_pins=len(protected), AST_parsed=parsed,
                  integer_registry_counts=counts, all_original_and_captured_bytes_unchanged=True,
                  scientific_source_admission_passed=False,
                  blockers=["physical/conformal reference connection mismatch",
                            "reconstructed radius used for nominal exact-rational graph gate",
                            "separate measured-timing review not enforced for full admission"])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metadata_readback_passed": True, "source_admission_passed": False,
                      "source_files": index["file_count"], "external_pins": len(recipe["protected_inputs"]),
                      "result_sha256": sha(output)}, sort_keys=True))

if __name__ == "__main__":
    main()
