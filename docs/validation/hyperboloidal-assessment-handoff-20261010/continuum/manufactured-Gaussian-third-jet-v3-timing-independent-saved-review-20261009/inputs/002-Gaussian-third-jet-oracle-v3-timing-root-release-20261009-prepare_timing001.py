"""HELD metadata-only preparation; root must review and invoke it separately."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
SUPPORT_SHA = "6ba37e3ea43ae9244ad1cfd283b470bbf937c080ff5e17a1562f53e60df6b154"
support = HERE / "timing_support.py"
if hashlib.sha256(support.read_bytes()).hexdigest() != SUPPORT_SHA:
    raise RuntimeError("held timing support changed before stdlib-only import")
spec = importlib.util.spec_from_file_location("gaussian_v3_timing_support", support)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
c = s.c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-index-sha256", required=True)
    parser.add_argument("--review-receipt-sha256", required=True)
    args = parser.parse_args()
    output = HERE / "timing-preparation-invocation001"
    output.mkdir(exist_ok=False)
    receipt = {"completed": False, "passed": False, "metadata_only": True, "candidate_imports": False,
               "numeric_target_calls": False, "command": sys.argv, "interpreter": str(Path(sys.executable).resolve()),
               "flags": {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode, "optimize": sys.flags.optimize},
               "environment": {"PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")}}
    pins, before = {}, {}
    try:
        s.guard()
        c.merge(pins, s.source_pins(args.source_index_sha256))
        prior, recipe, unit_path = s.prerequisites(args.review, args.review_index_sha256, args.review_receipt_sha256)
        c.merge(pins, prior)
        c.verify(pins)
        before = {name: c.sha(name) for name in pins}
        c.write(output / "pins-before.json", before)
        child = c.OWNER / "attempts/timing001"
        outer = HERE / "timing-outer001"
        invocation = HERE / "timing-invocation001"
        c.require(not any(path.exists() for path in (child, outer, invocation, HERE / "timing-authorization.json", HERE / "timing-release.json")),
                  "all timing destinations must be fresh; no rerun into partial paths")
        review_pins = dict(pins)
        authorization = {"Gaussian_third_jet_oracle_stage_authorized": "timing",
                         "source_index_sha256": s.SOURCE_INDEX, "recipe_sha256": c.IDENTITIES["recipe_sha256"],
                         "driver_sha256": c.IDENTITIES["driver_sha256"], "output": str(child), "outer_output": str(outer),
                         "unit_receipt": {"path": str(unit_path), "sha256": s.UNIT_RECEIPT}, "review_pins": review_pins,
                         "scope": "Only fixed20 v3 timing records,600s child/660s outer/690s root process-group cap. Full/native remain held."}
        c.write(HERE / "timing-authorization.json", authorization)
        release = {"owner": str(c.OWNER), "stage": "timing", "pins": pins,
                   "timing_root_source_index_sha256": args.source_index_sha256,
                   "authorization_sha256": c.sha(HERE / "timing-authorization.json"),
                   "output": str(child), "outer_output": str(outer), "invocation": str(invocation),
                   "required_base_pins": 437, "root_process_group_cap_seconds": 690,
                   "independent_unit_review_index_sha256": args.review_index_sha256,
                   "independent_unit_review_receipt_sha256": args.review_receipt_sha256,
                   "actual_unit_receipt_sha256": s.UNIT_RECEIPT, "actual_unit_result_sha256": s.UNIT_RESULT}
        c.write(HERE / "timing-release.json", release)
        receipt.update(completed=True, passed=True, result={"prepared": True, "stage": "timing", "records_only": 20,
                       "protected_pins": len(pins), "required_base_pins": 437, "no_execution": True,
                       "authorization_sha256": release["authorization_sha256"], "timing_release_sha256": c.sha(HERE / "timing-release.json")})
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        after = {}
        for name in pins:
            try:
                after[name] = c.sha(name)
            except OSError as error:
                after[name] = {"error": str(error)}
        receipt["inputs_unchanged"] = bool(before) and before == after
        if not receipt["inputs_unchanged"]:
            receipt["passed"] = False
        c.write(output / "pins-after.json", after)
        c.write(output / "receipt.json", receipt)
    print(json.dumps(receipt, allow_nan=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
