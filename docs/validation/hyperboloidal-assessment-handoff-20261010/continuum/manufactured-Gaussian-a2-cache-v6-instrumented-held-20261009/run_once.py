"""One-shot standard-library stage wrapper; false authorization template by default."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from admission import check, digest, load, unchanged, verify_pin
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["units", "certificate", "replay"])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    # An existing destination is never written. The caller must preserve this
    # outer refusal/trace in its own fresh invocation log.
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"stage": args.stage, "completed": False, "passed": False,
               "returncode": None, "inputs_unchanged": False,
               "started_utc": datetime.now(timezone.utc).isoformat(), "outputs": []}
    returncode = 1
    recipe = None
    index_sha = None
    dynamic_pins = []
    try:
        _, recipe, authorization, index_sha, _ = check(args.recipe, args.authorization, out, args.stage)
        receipt["source_index_sha256"] = index_sha
        receipt["recipe_sha256"] = digest(args.recipe)
        receipt["authorization_sha256"] = digest(args.authorization)
        auth_path = Path(args.authorization).resolve()
        dynamic_pins = [{"path": str(auth_path), "bytes": auth_path.stat().st_size,
                         "sha256": receipt["authorization_sha256"]}]
        for key in ("units_receipt", "certificate_receipt", "certificate_payload"):
            if key in authorization:
                dynamic_pins.append(authorization[key])
        receipt["dynamic_input_pins"] = dynamic_pins
        script = {"units": "unit_stage.py", "certificate": "certificate_stage.py", "replay": "replay_stage.py"}[args.stage]
        command = [str(Path(sys.executable).resolve()), "-I", "-B", str(root / script),
                   "--recipe", str(root / "recipe.json"),
                   "--authorization", str(Path(args.authorization).resolve()), "--output", str(out)]
        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONWARNINGS"):
            env.pop(key, None)
        env.update({"PYTHONOPTIMIZE": "0", "PYTHONDONTWRITEBYTECODE": "1"})
        (out / "command.json").write_text(json.dumps({"argv": command,
            "environment_changes": {"PYTHONOPTIMIZE": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            "environment_removed": ["PYTHONHOME", "PYTHONPATH", "PYTHONWARNINGS"],
            "timeout_seconds": recipe["stage_timeouts"][args.stage]}, indent=2, allow_nan=False) + "\n")
        with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                    timeout=recipe["stage_timeouts"][args.stage], check=False)
        returncode = result.returncode
        receipt["returncode"] = returncode
        report = load(out / "report.json") if (out / "report.json").exists() else None
        receipt["inputs_unchanged"] = unchanged(root, recipe, index_sha)
        for entry in dynamic_pins:
            verify_pin(entry)
        if returncode != 0 or report is None or report.get("passed") is not True:
            raise RuntimeError("stage failed or produced no accepted report")
        if report.get("stage") != args.stage or report.get("source_index_sha256") != index_sha:
            raise RuntimeError("report source/stage binding mismatch")
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "receipt.json":
                receipt["outputs"].append({"path": str(path), "bytes": path.stat().st_size,
                    "sha256": digest(path)})
        receipt.update({"completed": True, "passed": True, "returncode": 0})
        returncode = 0
    except BaseException as error:
        receipt["error_type"], receipt["error"] = type(error).__name__, str(error)
        (out / "wrapper-failure.txt").write_text(traceback.format_exc())
        if recipe is not None:
            try:
                receipt["inputs_unchanged"] = unchanged(root, recipe, index_sha)
                for entry in dynamic_pins:
                    verify_pin(entry)
            except BaseException as drift:
                receipt["inputs_unchanged"] = False
                receipt["input_drift_error"] = str(drift)
        if receipt["returncode"] is None:
            receipt["returncode"] = 1
        returncode = returncode if returncode else 1
        # Failure payloads and partial certificate remain at their exact paths.
        receipt["outputs"] = [{"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path)} for path in sorted(out.iterdir())
            if path.is_file() and path.name != "receipt.json"]
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": receipt["passed"], "receipt": str(out / "receipt.json")}, allow_nan=False))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
