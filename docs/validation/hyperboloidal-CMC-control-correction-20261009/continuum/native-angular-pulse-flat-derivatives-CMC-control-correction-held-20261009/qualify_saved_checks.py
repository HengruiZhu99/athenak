"""HELD stdlib saved-check qualification, excluding the entire control slice."""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import argparse
import ast
import json
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    # This local module imports stdlib only; its numerical imports are inside
    # its separate admitted main and are never reached by this saved reader.
    from controls_only import admission, sha, write
    protected = {}
    receipt = {"kind": "Saved noncontrol check qualification", "original_full_gate_passed": False,
               "scientific_recomputation": False, "large_jet_payloads_decoded": False}
    try:
        recipe, protected = admission(Path(args.recipe).resolve(), Path(args.authorization).resolve(), output,
                                      "saved_check_qualification_execution_admitted")
        if str(output) != recipe["fresh_saved_qualification_output"]:
            raise PermissionError("wrong fixed saved-qualification output")
        original = json.loads(Path(recipe["original_full_receipt"]).read_text())
        checks = json.loads(Path(recipe["original_checks"]).read_text())
        if len(checks) != 2360 or len({row["name"] for row in checks}) != 2360:
            raise RuntimeError("unexpected original count or duplicate names")
        for row in checks:
            e, t = Decimal(row["error"]), Decimal(row["tolerance"])
            if not e.is_finite() or not t.is_finite() or e < 0 or t < 0 or type(row["passed"]) is not bool or row["passed"] != (e <= t):
                raise RuntimeError("invalid saved operands/flag")
        failed = [row for row in checks if not row["passed"]]
        if len(failed) != 156 or failed != original["failed_checks"]:
            raise RuntimeError("original failed list changed")
        is_control = lambda row: row["name"].startswith(("control/", "exact/", "wave/control/"))
        controls = [row for row in checks if is_control(row)]
        retained = [row for row in checks if not is_control(row)]
        settings = recipe["scientific_settings"]
        expected_control_names = set()
        for dps in settings["precisions"]:
            for level in settings["levels"]:
                for case in settings["controls"]:
                    for boost in settings["control_boosts"]:
                        prefix = "control/%s/%s/%s/%s/" % (dps, level["name"], case["name"], boost)
                        expected_control_names.update(prefix + key for key in ["root", "null", "first_graph", "second_graph", "factor_quotient", "K_jet", "D_source_jet", "Lorentz_matrix"])
                        if level["name"] == settings["final_level"]:
                            expected_control_names.update("exact/%s/%s/%s/%s" % (dps, case["name"], boost, block) for block in ["u", "gradient", "hessian"])
                            expected_control_names.add("wave/control/%s/%s/%s/%s" % (dps, level["name"], case["name"], boost))
        if len(controls) != 880 or {row["name"] for row in controls} != expected_control_names:
            raise RuntimeError("old control slice does not match fixed original registry")
        if len(retained) != 1480 or not all(row["passed"] for row in retained):
            raise RuntimeError("unexpected failure in the retained saved slice")
        if original["source_before"] != original["source_after"] or not original["sources_unchanged"]:
            raise RuntimeError("original source preservation missing")
        old = (HERE / "source-history/derivative_core-original.py").read_bytes()
        new = (HERE / "derivative_core.py").read_bytes()
        oldline = b'            b=omega*q/self.a\n'
        newline = b'            b=omega*(sumjet(t*t for t in Y)**mp.mpf(".5"))/self.a\n'
        if old.count(oldline) != 1 or new.count(newline) != 1 or new.replace(newline, oldline) != old:
            raise RuntimeError("correction exceeds the one CMC control line")
        if ast.dump(ast.parse(new.replace(newline, oldline)), include_attributes=False) != ast.dump(ast.parse(old), include_attributes=False):
            raise RuntimeError("reverse AST mismatch")
        write(output / "retained-saved-checks.json", retained)
        report = {"passed_saved_noncontrol_check_qualification": True, "original_total": 2360,
                  "original_failed": 156, "entire_control_slice_deferred": 880,
                  "retained_saved_checks": 1480, "retained_saved_failures": 0,
                  "retained_by_family": dict(Counter(row["name"].split("/")[0] for row in retained)),
                  "one_CMC_line_is_only_source_change": True,
                  "native_initial_ray_and_jet_functions_byte_unchanged": True,
                  "original_full_gate_remains_failed": True, "corrected_controls_not_evaluated_here": True,
                  "large_jet_payloads_decoded": False,
                  "scope": "Qualification of recorded checks under their original exact source/input pins, excluding all 880 control checks. No new derivative targets, native integrals, inverse, coordinate regularity or PDE acceptance."}
        write(output / "report.json", report)
        receipt.update(report)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        receipt.update({"passed_saved_noncontrol_check_qualification": False, "error": str(exc),
                        "exception_type": type(exc).__name__, "traceback": traceback.format_exc()})
        write(output / "failure.json", receipt)
    finally:
        after = {path: sha(path) for path in protected}
        receipt.update({"source_before": protected, "source_after": after, "sources_unchanged": after == protected,
                        "seconds": time.monotonic() - start, "command": sys.argv,
                        "output_pins": {str(path): sha(path) for path in output.iterdir() if path.is_file()}})
        if after != protected:
            receipt["passed_saved_noncontrol_check_qualification"] = False
        write(output / "receipt.json", receipt)
    if not receipt.get("passed_saved_noncontrol_check_qualification") or not receipt.get("sources_unchanged"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
