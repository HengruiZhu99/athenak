"""Prepare only exact fresh 318-unit admission after root and independent review."""
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
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-index-sha256", required=True)
    parser.add_argument("--review-receipt-sha256", required=True)
    args = parser.parse_args()

    def prepare(output, pins):
        preparation, root_review = c.root_review()
        scripts = c.scripts_pins(preparation["root_scripts_index_sha256"])
        c.merge(pins, scripts)
        base, recipe = c.base_pins()
        c.require(c.load(c.HERE / "source-pins001.json") == base, "437 source pins differ from root reviewed set")
        c.merge(pins, base)
        peer = c.independent_review(args.review, args.review_index_sha256, args.review_receipt_sha256)
        c.merge(pins, peer)
        reviews = dict(peer)
        c.merge(reviews, scripts)
        for name in ("source-pins001.json", "review-preparation001.json", "source-review001.json"):
            reviews[str(c.HERE / name)] = c.sha(c.HERE / name)
        for folder in ("review-preparation-invocation001", "finalize-review-invocation001"):
            receipt = c.load(c.HERE / folder / "receipt.json")
            c.require(receipt["completed"] is True and receipt["passed"] is True and receipt["inputs_unchanged"] is True,
                      "root static metadata phase did not pass")
            for path in sorted((c.HERE / folder).rglob("*")):
                if path.is_file():
                    reviews[str(path)] = c.sha(path)
        c.merge(pins, reviews)
        c.write(output / "pins-before.json", pins)
        c.source_proofs(recipe)
        child_output = c.OWNER / "attempts/units001"
        outer_output = c.HERE / "units-outer001"
        invocation = c.HERE / "units-invocation001"
        c.require(not any(path.exists() for path in (child_output, outer_output, invocation)),
                  "all unit output paths must be fresh")
        c.verify(pins)
        authorization = {
            "Gaussian_third_jet_oracle_stage_authorized": "units",
            "source_index_sha256": c.IDENTITIES["source_index_sha256"],
            "recipe_sha256": c.IDENTITIES["recipe_sha256"],
            "driver_sha256": c.IDENTITIES["driver_sha256"],
            "output": str(child_output), "outer_output": str(outer_output), "review_pins": reviews,
            "scope": "Fresh318 v3 analytic units only; root60-second process-group cap. Timing/full/native remain held."}
        c.write(c.HERE / "units-authorization.json", authorization)
        c.write(c.HERE / "units-release.json", {
            "owner": str(c.OWNER), "stage": "units", "pins": pins,
            "root_scripts_index_sha256": preparation["root_scripts_index_sha256"],
            "authorization_sha256": c.sha(c.HERE / "units-authorization.json"),
            "output": str(child_output), "outer_output": str(outer_output), "invocation": str(invocation),
            "required_base_pins": 437, "root_process_group_cap_seconds": 60,
            "independent_review_index_sha256": args.review_index_sha256,
            "independent_review_receipt_sha256": args.review_receipt_sha256})
        c.write(output / "pins-after.json", {name: c.sha(name) for name in pins})
        return {"prepared": True, "stage": "units", "required_base_pins": 437, "protected_pins": len(pins),
                "authorization_sha256": c.sha(c.HERE / "units-authorization.json"),
                "units_release_sha256": c.sha(c.HERE / "units-release.json"), "no_execution": True}

    return c.metadata_phase("units-preparation-invocation001", prepare)


if __name__ == "__main__":
    raise SystemExit(main())
