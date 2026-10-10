"""Root seals its manual source/math review; never imports the candidate."""
from pathlib import Path
import argparse
import hashlib
import importlib.util

HERE = Path(__file__).resolve().parent
COMMON_SHA = "83fcc9cfc666c11a9001812b6b5ce6a3c5d363a2372bc8797d6637fcc085aad7"
common = HERE / "root_common.py"
if hashlib.sha256(common.read_bytes()).hexdigest() != COMMON_SHA:
    raise RuntimeError("pinned root stdlib helper changed before import")
spec = importlib.util.spec_from_file_location("gaussian_v3_root_common", common)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-source-math-reviewed", action="store_true", required=True)
    args = parser.parse_args()

    def finalize(output, pins):
        preparation = c.load(c.HERE / "review-preparation001.json")
        c.require(preparation["metadata_passed"] is True and preparation["protected_base_pins"] == 437,
                  "successful metadata preparation prerequisite required")
        c.merge(pins, c.scripts_pins(preparation["root_scripts_index_sha256"]))
        original_pins = c.load(c.HERE / "source-pins001.json")
        base, recipe = c.base_pins()
        c.require(original_pins == base, "437 source/runtime/history pins changed since preparation")
        c.merge(pins, base)
        c.merge(pins, {str(c.HERE / name): c.sha(c.HERE / name)
                       for name in ("source-pins001.json", "review-preparation001.json")})
        c.write(output / "pins-before.json", pins)
        proof = c.source_proofs(recipe)
        c.require(args.root_source_math_reviewed is True, "explicit root manual source/math review required")
        c.verify(pins)
        review = {"passed": True, "root_full_source_math_and_admission_review": True,
            "manual_review_explicit_flag": True,
            "source_index_sha256": c.IDENTITIES["source_index_sha256"],
            "recipe_sha256": c.IDENTITIES["recipe_sha256"],
            "driver_sha256": c.IDENTITIES["driver_sha256"],
            "root_scripts_index_sha256": preparation["root_scripts_index_sha256"],
            "protected_base_pins": 437, "inputs_unchanged": True,
            "static_equivalence_proof": proof,
            "candidate_imports": False, "numeric_calls": False,
            "independent_v3_source_math_admission_PASS_required_before_units": True,
            "units_timing_full_separate_releases_required": True,
            "v2_timing_FAIL_remains_preserved": True,
            "historical_saved_reviewer_runtime_limitation_preserved": True,
            "no_native_RHS_inverse_coverage_global_slicing_or_BH_admission": True}
        c.write(c.HERE / "source-review001.json", review)
        c.write(output / "pins-after.json", {name: c.sha(name) for name in pins})
        return review

    return c.metadata_phase("finalize-review-invocation001", finalize)


if __name__ == "__main__":
    raise SystemExit(main())
