"""Execute only reviewed v3 analytic units under one enclosing 60-second group cap."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
COMMON_SHA = "83fcc9cfc666c11a9001812b6b5ce6a3c5d363a2372bc8797d6637fcc085aad7"
common = HERE / "root_common.py"
if hashlib.sha256(common.read_bytes()).hexdigest() != COMMON_SHA:
    raise RuntimeError("pinned root stdlib helper changed before import")
spec = importlib.util.spec_from_file_location("gaussian_v3_root_common", common)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    output = HERE / "units-invocation001"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = {"completed": False, "accepted_stage": False, "stage": "units", "returncode": None,
              "root_process_group_cap_seconds": 60, "candidate_imports_in_root": False,
              "root_command": sys.argv, "root_interpreter": str(Path(sys.executable).resolve())}
    pins, before, protected_outputs = {}, {}, []
    process = None
    child_output = c.OWNER / "attempts/units001"
    outer_output = HERE / "units-outer001"
    try:
        c.guard()
        release_path = HERE / "units-release.json"
        c.require(c.sha(release_path) == args.release_sha256, "exact unit release hash required")
        release = c.load(release_path)
        c.require(release["owner"] == str(c.OWNER) and release["stage"] == "units" and
                  release["root_process_group_cap_seconds"] == 60 and release["required_base_pins"] == 437,
                  "release owner/stage/cap differs")
        c.require(release["output"] == str(child_output) and release["outer_output"] == str(outer_output)
                  and release["invocation"] == str(output), "release paths differ from fixed fresh units001")
        c.require(not child_output.exists() and not outer_output.exists(), "original or partial unit outputs cannot be reused")
        c.merge(pins, release["pins"])
        c.merge(pins, c.scripts_pins(release["root_scripts_index_sha256"]))
        c.merge(pins, {str(release_path): args.release_sha256})
        authorization_path = HERE / "units-authorization.json"
        c.merge(pins, {str(authorization_path): release["authorization_sha256"]})
        base, recipe = c.base_pins()
        c.require(all(release["pins"].get(name) == digest for name, digest in base.items()),
                  "release does not protect all exact 437 source/runtime/history pins")
        c.merge(pins, base)
        authorization = c.load(authorization_path)
        c.require(authorization["Gaussian_third_jet_oracle_stage_authorized"] == "units" and
                  authorization["output"] == str(child_output) and authorization["outer_output"] == str(outer_output),
                  "authorization stage/path mismatch")
        for key in ("source_index_sha256", "recipe_sha256", "driver_sha256"):
            c.require(authorization[key] == c.IDENTITIES[key], "authorization exact candidate mismatch: " + key)
        for folder in ("review-preparation-invocation001", "finalize-review-invocation001", "units-preparation-invocation001"):
            meta_path = HERE / folder / "receipt.json"
            meta = c.load(meta_path)
            c.require(meta["completed"] is True and meta["passed"] is True and meta["inputs_unchanged"] is True,
                      "successful static root preparation prerequisite required")
            c.merge(pins, {str(meta_path): c.sha(meta_path)})
        prepared = c.load(HERE / "units-preparation-invocation001/receipt.json")
        c.require(prepared["result"]["units_release_sha256"] == args.release_sha256 and
                  prepared["result"]["authorization_sha256"] == release["authorization_sha256"],
                  "prepared release/authorization hash binding differs")
        c.verify(pins)
        before = {name: c.sha(name) for name in pins}
        c.write(output / "pins-before.json", before)
        environment = os.environ.copy()
        removed = ("PYTHONHOME", "PYTHONPATH", "PYTHONWARNINGS", "PYTHONSTARTUP", "PYTHONUSERBASE")
        for name in removed:
            environment.pop(name, None)
        environment.update(recipe["environment"])
        c.require(environment["PYTHONOPTIMIZE"] == "0" and environment["PYTHONDONTWRITEBYTECODE"] == "1"
                  and all(environment[name] == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")),
                  "fixed scientific environment differs")
        command = [recipe["python_runtime_path"], "-I", "-B", str(c.OWNER / "outer_once.py"),
                   "--authorization", str(authorization_path), "--authorization-sha256", release["authorization_sha256"],
                   "--invocation", str(outer_output)]
        c.write(output / "invocation.json", {"command": command, "cwd": str(c.OWNER),
                "fixed_environment": recipe["environment"], "removed_environment_keys": list(removed),
                "root_process_group_cap_seconds": 60, "required_base_pins": 437,
                "source_index_sha256": c.IDENTITIES["source_index_sha256"],
                "release_sha256": args.release_sha256, "authorization_sha256": release["authorization_sha256"]})
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=c.OWNER, env=environment, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            record.update(pid=process.pid, process_group=process.pid)
            try:
                record["returncode"] = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                record["returncode"] = kill_group(process)
                record["root_group_timeout"] = True
                raise TimeoutError("root60-second unit group cap; partial outputs preserved")
        child = c.load(child_output / "receipt.json")
        result = c.load(child_output / "result.json")
        outer = c.load(outer_output / "receipt.json")
        for name, digest in child["output_hashes"].items():
            c.require(c.sha(child_output / name) == digest, "actual child output hash mismatch: " + name)
        accepted = (type(record["returncode"]) is int and record["returncode"] == 0 and
                    child.get("completed") is True and child.get("passed") is True and child.get("sources_unchanged") is True and
                    child.get("stage") == "units" and child.get("source_index_sha256") == c.IDENTITIES["source_index_sha256"] and
                    child.get("recipe_sha256") == c.IDENTITIES["recipe_sha256"] and child.get("driver_sha256") == c.IDENTITIES["driver_sha256"] and
                    result.get("passed") is True and type(result.get("checks")) is int and result["checks"] == 318 and
                    type(result.get("failed_total")) is int and result["failed_total"] == 0 and
                    outer.get("completed") is True and outer.get("accepted_stage") is True and outer.get("inputs_unchanged") is True)
        record.update(completed=True, accepted_stage=accepted,
                      child_receipt_sha256=c.sha(child_output / "receipt.json"), result_sha256=c.sha(child_output / "result.json"),
                      outer_receipt_sha256=c.sha(outer_output / "receipt.json"))
        c.require(accepted, "actual same-source 318-unit child/outer acceptance required")
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error), accepted_stage=False)
        if process is not None and process.poll() is None:
            record["returncode"] = kill_group(process)
            record["exception_process_group_cleanup"] = True
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        after = {}
        for name in pins:
            try:
                after[name] = c.sha(name)
            except OSError as error:
                after[name] = {"error": str(error)}
        unchanged = bool(before) and before == after
        record["inputs_unchanged"] = unchanged
        if not unchanged:
            record["accepted_stage"] = False
        c.write(output / "pins-after.json", after)
        record["elapsed_seconds"] = time.monotonic() - started
        record["timing_full_native_or_BH_admission"] = False
        for folder in (output, outer_output, child_output):
            if folder.exists():
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        protected_outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": c.sha(path),
                            "policy": "large_payload" if path.suffix.lower() in (".jsonl", ".npy", ".npz") or path.stat().st_size > 1048576 else "source_or_receipt"})
        record["output_inventory"] = protected_outputs
        c.write(output / "receipt.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "output_inventory"}, allow_nan=False))
    return 0 if record["accepted_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
