"""Stdlib-only saved-check classification; no scalar/jet target computation."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import argparse
import hashlib
import json
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {"kind": "Saved derivative-check diagnosis only", "scientific_recomputation": False,
               "original_full_gate_passed": False, "large_jet_payloads_decoded": False}
    protected = {}
    try:
        if sys.flags.optimize != 0:
            raise PermissionError("optimization must remain zero")
        rp = Path(args.recipe).resolve()
        if rp != HERE / "recipe.json":
            raise PermissionError("recipe must be the fixed local recipe")
        recipe = json.loads(rp.read_text())
        protected = {str(rp): sha(rp), str(Path(__file__).resolve()): sha(__file__)}
        for path, expected in recipe["pins"].items():
            actual = sha(path)
            if actual != expected:
                raise RuntimeError("input changed: " + path)
            protected[path] = actual
        original = json.loads(Path(recipe["receipt"]).read_text())
        checks = json.loads(Path(recipe["checks"]).read_text())
        if original["source_before"] != original["source_after"] or not original["sources_unchanged"]:
            raise RuntimeError("original source preservation is not established")
        if original["passed_analytic_scalar_derivative_gate"] is not False:
            raise RuntimeError("original attempt must remain failed")
        if len(checks) != 2360 or original["checks"] != 2360:
            raise RuntimeError("unexpected original fixed check count")
        if len({x["name"] for x in checks}) != len(checks):
            raise RuntimeError("duplicate original check names")
        for row in checks:
            error, tolerance = Decimal(row["error"]), Decimal(row["tolerance"])
            if not error.is_finite() or not tolerance.is_finite() or error < 0 or tolerance < 0:
                raise RuntimeError("invalid saved check operands")
            if type(row["passed"]) is not bool or row["passed"] != (error <= tolerance):
                raise RuntimeError("saved boolean disagrees with saved operands: " + row["name"])
        failed = [x for x in checks if not x["passed"]]
        if len(failed) != 156 or failed != original["failed_checks"]:
            raise RuntimeError("unexpected original failed-check list")
        if original["root_progress"]["completed_root_calls"] != 493568:
            raise RuntimeError("unexpected original root-call count")
        for path, expected in original["source_before"].items():
            actual = sha(path)
            if actual != expected:
                raise RuntimeError("original dependency changed: " + path)
            protected[path] = actual
        payloads = []
        for path, expected in original["output_pins"].items():
            actual = sha(path)
            if actual != expected:
                raise RuntimeError("saved output changed: " + path)
            protected[path] = actual
            payloads.append({"path": path, "sha256": actual, "bytes": Path(path).stat().st_size,
                             "decoded": path == recipe["checks"],
                             "role": "source_or_receipt" if path == recipe["checks"] or Path(path).name == "before.json" else "large_payload"})
        family = Counter(x["name"].split("/")[0] for x in failed)
        if dict(family) != {"control": 120, "exact": 24, "wave": 12}:
            raise RuntimeError("unexpected failure families")
        if any(not any("/" + name + "/" in x["name"] for name in ["CMC_l0", "CMC_l1", "CMC_l2"]) for x in failed):
            raise RuntimeError("failure outside the CMC controls")
        categories = {}
        for name in sorted({x["name"].split("/")[0] for x in checks}):
            rows = [x for x in checks if x["name"].split("/")[0] == name]
            worst = max(rows, key=lambda x: Decimal(x["error"]))
            categories[name] = {"checks": len(rows), "failed": sum(not x["passed"] for x in rows),
                                "max_saved_error_row": worst}
        groups = {}
        for row in failed:
            parts = row["name"].split("/")
            group = "/".join(parts[:-1]) if parts[0] != "wave" else row["name"]
            groups.setdefault(group, []).append(row)
        result = {"classification_passed": True, "checks": len(checks), "failed_checks": len(failed),
                  "passed_checks": len(checks) - len(failed), "failed_by_family": dict(family),
                  "failed_local_metric": dict(Counter(x["name"].split("/")[-1] for x in failed if x["name"].startswith("control/"))),
                  "all_failed_checks_are_CMC_controls": True,
                  "native_rows": original["native_rows"], "control_rows": original["control_rows"],
                  "initial_rows": original["initial_rows"], "root_progress": original["root_progress"],
                  "original_child_seconds": original["seconds"], "categories": categories,
                  "failed_groups": groups, "first_failed_checks": failed[:6],
                  "unchanged_original_full_failure": True,
                  "scope": "Saved operands/flags/counts only. No ray integration, jet decode, target evaluation, repaired-control pass, inverse, native or PDE claim."}
        write(out / "report.json", result)
        write(out / "failed-checks.json", failed)
        write(out / "payload-metadata.json", payloads)
        after = {p: sha(p) for p in protected}
        if after != protected:
            raise RuntimeError("protected bytes changed during saved readback")
        receipt.update({"passed_saved_check_classification": True, "before": protected, "after": after,
                        "inputs_unchanged": True, "result_sha256": sha(out / "report.json")})
        print(json.dumps({k: result[k] for k in ["checks", "failed_checks", "passed_checks", "failed_by_family", "all_failed_checks_are_CMC_controls"]}), flush=True)
    except Exception as exc:
        receipt.update({"passed_saved_check_classification": False, "error": str(exc),
                        "exception_type": type(exc).__name__, "traceback": traceback.format_exc(),
                        "before": protected, "after": {p: sha(p) for p in protected},
                        "inputs_unchanged": all(sha(p) == h for p, h in protected.items())})
        write(out / "failure.json", receipt)
        raise
    finally:
        receipt["seconds"] = time.monotonic() - start
        receipt["outputs"] = {str(p): sha(p) for p in out.iterdir() if p.is_file()}
        write(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
