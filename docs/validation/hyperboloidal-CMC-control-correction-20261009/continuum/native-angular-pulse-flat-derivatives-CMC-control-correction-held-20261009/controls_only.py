"""HELD original 880-check control slice with one corrected CMC radius jet."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def admission(recipe_path, authorization_path, output_path, key):
    if sys.flags.optimize != 0:
        raise PermissionError("optimization must be zero")
    if recipe_path != HERE / "control-recipe.json":
        raise PermissionError("consumed recipe must be the exact local control recipe")
    recipe = json.loads(recipe_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    index_path = HERE / "source-index.json"
    if authorization.get(key) is not True:
        raise PermissionError("missing fresh root release for this separate stage")
    if authorization.get("source_index") != {"path": str(index_path), "sha256": sha(index_path)}:
        raise PermissionError("authorization does not bind exact source index")
    if authorization.get("fresh_output_path") != str(output_path):
        raise PermissionError("unreleased output path")
    index = json.loads(index_path.read_text())
    protected = {str(index_path): sha(index_path), str(authorization_path): sha(authorization_path)}
    for item in index["files"]:
        path = HERE / item["path"]
        if sha(path) != item["sha256"]:
            raise RuntimeError("indexed source changed: " + str(path))
        protected[str(path)] = item["sha256"]
    for path, expected in {**recipe["protected_pins"], **recipe["mpmath_python_pins"]}.items():
        if sha(path) != expected:
            raise RuntimeError("protected dependency changed: " + path)
        protected[path] = expected
    if Path(sys.executable).resolve() != Path(recipe["runtime_path"]).resolve() or sha(sys.executable) != recipe["runtime_sha256"]:
        raise PermissionError("unreleased interpreter")
    protected[str(Path(sys.executable).resolve())] = recipe["runtime_sha256"]
    for key, expected in recipe["required_environment"].items():
        if os.environ.get(key) != expected:
            raise PermissionError("wrong required environment: " + key)
    if any(os.environ.get(key) for key in ["PYTHONPATH", "PYTHONHOME", "PYTHONWARNINGS"]):
        raise PermissionError("unreleased Python environment injection")
    original = json.loads(Path(recipe["original_full_receipt"]).read_text())
    diagnosis = json.loads(Path(recipe["diagnosis_report"]).read_text())
    if original["passed_analytic_scalar_derivative_gate"] is not False or original["checks"] != 2360 or len(original["failed_checks"]) != 156:
        raise PermissionError("original failed classification changed")
    if diagnosis.get("classification_passed") is not True or diagnosis["failed_checks"] != 156 or not diagnosis["all_failed_checks_are_CMC_controls"]:
        raise PermissionError("missing independent original-check classification")
    old_recipe = json.loads(Path(recipe["original_full_recipe"]["path"]).read_text())
    if sha(recipe["original_full_recipe"]["path"]) != recipe["original_full_recipe"]["sha256"]:
        raise RuntimeError("original full recipe changed")
    if any(value != old_recipe[key] for key, value in recipe["scientific_settings"].items()):
        raise RuntimeError("original control settings changed")
    return recipe, protected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protected = {}
    receipt = {"kind": "Corrected CMC control-only attempt", "original_full_gate_passed": False,
               "native_integrals_rerun": False, "inverse_attempted": False}
    try:
        recipe, protected = admission(Path(args.recipe).resolve(), Path(args.authorization).resolve(), output,
                                      "corrected_control_gate_execution_admitted")
        if str(output) != recipe["fresh_control_output"]:
            raise PermissionError("wrong fixed control output")
        write(output / "before.json", protected)
        # All candidate/scientific imports occur after exact admission.
        import mpmath as mp
        import derivative_core as dc
        import values_context as vc
        if sha(mp.__file__) != recipe["mpmath_python_pins"].get(str(Path(mp.__file__).resolve())):
            raise RuntimeError("consumed mpmath is not pinned")
        settings = recipe["scientific_settings"]
        rows, checks = [], []
        roots = 0

        class CountingControlGraph(dc.ControlGraph):
            def root(self, T, X, k, initial=False):
                nonlocal roots
                value = super().root(T, X, k, initial)
                roots += 1
                if roots % 2048 == 0:
                    print("control-root-progress", roots, "of", recipe["fixed_roots"], flush=True)
                return value

        def check(name, error, tolerance):
            checks.append({"name": name, "error": vc.number(error), "tolerance": tolerance,
                           "passed": bool(error <= mp.mpf(tolerance))})

        def add_metrics(prefix, metrics):
            for key in ["root", "null", "first_graph", "second_graph", "factor_quotient", "K_jet", "D_source_jet", "Lorentz_matrix"]:
                tolerance = settings["tolerances"]["root"] if key == "root" else settings["tolerances"]["local_identities"]
                check(prefix + "/" + key, metrics[key], tolerance)
            if not metrics["minimum_D"] > 0:
                raise ArithmeticError("nonpositive quadrature ray denominator")

        for dps in settings["precisions"]:
            mp.mp.dps = dps
            for level in settings["levels"]:
                for ctrl in settings["controls"]:
                    graph = CountingControlGraph(ctrl["kind"], ctrl.get("degree", 0))
                    X = list(map(mp.mpf, ctrl["X"]))
                    T = graph.height(X) + mp.mpf(ctrl["tau"])
                    exact = [jet.flat() for jet in graph.exact(T, X)]
                    for mode in settings["control_boosts"]:
                        B = dc.fixed_boost(X, mp.mpf(1), dc.ZERO(), mode)
                        result, metrics = dc.integrate_ray(graph, T, X, B, level["polar_order"], level["azimuth_order"])
                        rows.append({"name": ctrl["name"], "dps": dps, "level": level["name"], "boost": mode,
                                     "jet": [dc.strings(row) for row in result], "exact": [dc.strings(row) for row in exact],
                                     "metrics": {key: vc.number(value) for key, value in metrics.items()}})
                        add_metrics("control/%s/%s/%s/%s" % (dps, level["name"], ctrl["name"], mode), metrics)
                        if level["name"] == settings["final_level"]:
                            for block, numer in dc.jet_blocks(result).items():
                                check("exact/%s/%s/%s/%s" % (dps, ctrl["name"], mode, block),
                                      vc.scaled(numer, dc.jet_blocks(exact)[block]), settings["tolerances"]["convergence"])
                        rows[-1]["wave_trace_scaled_error"] = vc.number(dc.trace_error(result))
                        if level["name"] == settings["final_level"]:
                            check("wave/control/%s/%s/%s/%s" % (dps, level["name"], ctrl["name"], mode),
                                  dc.trace_error(result), settings["tolerances"]["wave_trace"])
                        write(output / "partial-controls.json", rows)
                        write(output / "partial-checks.json", checks)
                        print("control-complete", dps, level["name"], ctrl["name"], mode, flush=True)
        if (roots, len(rows), len(checks)) != (recipe["fixed_roots"], recipe["fixed_rows"], recipe["fixed_checks"]):
            raise RuntimeError("fixed control count mismatch")
        write(output / "controls.json", rows)
        write(output / "checks.json", checks)
        result = {"passed_corrected_control_gate": all(row["passed"] for row in checks),
                  "checks": len(checks), "failed_checks": [row for row in checks if not row["passed"]],
                  "control_rows": len(rows), "completed_control_roots": roots,
                  "original_full_gate_remains_failed": True,
                  "scope": "Original flat/CMC control slice only. Saved native/initial/coarea qualification is separate; no inverse, native, coordinate-regularity or PDE claim."}
        write(output / "result.json", result)
        receipt.update(result)
        if not result["passed_corrected_control_gate"]:
            raise ArithmeticError("corrected control-only gate failed; no retry or tolerance change")
    except Exception as exc:
        receipt.update({"passed_corrected_control_gate": False, "error": str(exc),
                        "exception_type": type(exc).__name__, "traceback": traceback.format_exc()})
        write(output / "failure.json", receipt)
    finally:
        after = {path: sha(path) for path in protected}
        receipt.update({"source_before": protected, "source_after": after, "sources_unchanged": after == protected,
                        "seconds": time.monotonic() - start, "command": sys.argv,
                        "output_pins": {str(path): sha(path) for path in output.iterdir() if path.is_file()}})
        if after != protected:
            receipt["passed_corrected_control_gate"] = False
            receipt["source_drift_failure"] = True
        write(output / "receipt.json", receipt)
    if not receipt.get("passed_corrected_control_gate") or not receipt.get("sources_unchanged"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
