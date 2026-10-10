"""Capture all 437 Gaussian v3 pins and independent static equivalences only."""
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
    parser.add_argument("--scripts-index-sha256", required=True)
    args = parser.parse_args()

    def prepare(output, pins):
        scripts = c.scripts_pins(args.scripts_index_sha256)
        c.merge(pins, scripts)
        base, recipe = c.base_pins()
        c.merge(pins, base)
        c.write(output / "pins-before.json", pins)
        proof = c.source_proofs(recipe)
        inventory = [{"path": str(path), "bytes": path.stat().st_size, "sha256": c.sha(path)}
                     for path in sorted(c.OWNER.glob("*.py"))]
        c.verify(pins)
        c.write(c.HERE / "source-pins001.json", base)
        c.write(c.HERE / "review-preparation001.json", {
            "metadata_passed": True, "mathematical_source_review_complete": False,
            "source_index_sha256": c.IDENTITIES["source_index_sha256"],
            "root_scripts_index_sha256": args.scripts_index_sha256,
            "protected_base_pins": len(base), "protected_preparation_pins": len(pins),
            "static_equivalence_proof": proof, "inventory": inventory,
            "candidate_imports": False, "numeric_calls": False,
            "full_source_math_review_and_independent_v3_review_required_before_units": True,
            "scope": "Physical-reference RWM manufactured analytic oracle; no compound inner BM"})
        c.write(output / "pins-after.json", {name: c.sha(name) for name in pins})
        return {"prepared": True, "base_pins": len(base), "static_proofs_passed": True,
                "no_execution_admission": True}

    return c.metadata_phase("review-preparation-invocation001", prepare)


if __name__ == "__main__":
    raise SystemExit(main())
