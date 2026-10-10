"""HELD timing-only root launcher; never call without separate root release."""
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
SUPPORT_SHA = "6ba37e3ea43ae9244ad1cfd283b470bbf937c080ff5e17a1562f53e60df6b154"
support = HERE / "timing_support.py"
if hashlib.sha256(support.read_bytes()).hexdigest() != SUPPORT_SHA:
    raise RuntimeError("held timing support changed before stdlib-only import")
spec = importlib.util.spec_from_file_location("gaussian_v3_timing_support", support)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
c = s.c


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
    output = HERE / "timing-invocation001"
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    record = {"completed": False, "accepted_stage": False, "stage": "timing", "returncode": None,
              "root_process_group_cap_seconds": 690, "candidate_imports_in_root": False,
              "root_command": sys.argv, "root_interpreter": str(Path(sys.executable).resolve())}
    process = None
    pins, before = {}, {}
    child, outer = c.OWNER / "attempts/timing001", HERE / "timing-outer001"
    try:
        s.guard()
        release_path = HERE / "timing-release.json"
        c.require(c.sha(release_path) == args.release_sha256, "exact root timing release hash required")
        release = c.load(release_path)
        c.require(release["owner"] == str(c.OWNER) and release["stage"] == "timing"
                  and release["root_process_group_cap_seconds"] == 690 and release["required_base_pins"] == 437,
                  "release owner/stage/caps changed")
        c.require(release["output"] == str(child) and release["outer_output"] == str(outer) and release["invocation"] == str(output),
                  "release paths differ from exact fresh timing001")
        c.require(not child.exists() and not outer.exists(), "no reuse of completed or partial timing outputs")
        c.merge(pins, release["pins"])
        c.merge(pins, s.source_pins(release["timing_root_source_index_sha256"]))
        prior, recipe, unit_path = s.prerequisites(s.UNIT_REVIEW, release["independent_unit_review_index_sha256"],
                                                 release["independent_unit_review_receipt_sha256"])
        c.require(all(release["pins"].get(name) == digest for name, digest in prior.items()), "release lacks exact source/unit/review closure")
        c.merge(pins, prior)
        authorization_path = HERE / "timing-authorization.json"
        c.merge(pins, {str(release_path): args.release_sha256, str(authorization_path): release["authorization_sha256"]})
        authorization = c.load(authorization_path)
        c.require(authorization["Gaussian_third_jet_oracle_stage_authorized"] == "timing"
                  and authorization["output"] == str(child) and authorization["outer_output"] == str(outer)
                  and authorization["unit_receipt"] == {"path": str(unit_path), "sha256": s.UNIT_RECEIPT},
                  "exact timing authorization/unit/output binding required")
        for key in ("source_index_sha256", "recipe_sha256", "driver_sha256"):
            c.require(authorization[key] == c.IDENTITIES[key], "authorization candidate mismatch: " + key)
        c.require(all(authorization["review_pins"].get(name) == digest for name, digest in release["pins"].items()),
                  "child and outer must protect every release source/review input")
        prep_path = HERE / "timing-preparation-invocation001/receipt.json"
        prep = c.load(prep_path)
        c.require(prep["completed"] is True and prep["passed"] is True and prep["inputs_unchanged"] is True
                  and prep["result"]["timing_release_sha256"] == args.release_sha256
                  and prep["result"]["authorization_sha256"] == release["authorization_sha256"],
                  "successful exact separate metadata preparation required")
        for p in sorted(prep_path.parent.rglob("*")):
            if p.is_file():
                c.merge(pins, {str(p): c.sha(p)})
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
                  "fixed isolated single-thread environment required")
        command = [recipe["python_runtime_path"], "-I", "-B", str(c.OWNER / "outer_once.py"),
                   "--authorization", str(authorization_path), "--authorization-sha256", release["authorization_sha256"],
                   "--invocation", str(outer)]
        c.write(output / "invocation.json", {"command": command, "cwd": str(c.OWNER),
                "fixed_environment": recipe["environment"], "removed_environment_keys": list(removed),
                "root_process_group_cap_seconds": 690, "required_base_pins": 437,
                "source_index_sha256": s.SOURCE_INDEX, "release_sha256": args.release_sha256,
                "authorization_sha256": release["authorization_sha256"]})
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=c.OWNER, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(pid=process.pid, process_group=process.pid)
            try:
                record["returncode"] = process.wait(timeout=690)
            except subprocess.TimeoutExpired:
                record["returncode"] = kill_group(process)
                record["root_group_timeout"] = True
                raise TimeoutError("fixed690-second timing group cap; partial evidence preserved")
        actual, result, enclosing = c.load(child / "receipt.json"), c.load(child / "result.json"), c.load(outer / "receipt.json")
        for rel, digest in actual["output_hashes"].items():
            c.require(c.sha(child / rel) == digest, "actual child output drift: " + rel)
        accepted = (type(record["returncode"]) is int and record["returncode"] == 0
                    and all(actual.get(key) is True for key in ("completed", "passed", "sources_unchanged"))
                    and actual.get("stage") == "timing" and all(actual.get(key) == c.IDENTITIES[key] for key in ("source_index_sha256", "recipe_sha256", "driver_sha256"))
                    and result.get("passed") is True and type(result.get("records")) is int and result["records"] == 20
                    and type(result.get("failed_total")) is int and result["failed_total"] == 0
                    and all(enclosing.get(key) is True for key in ("completed", "accepted_stage", "inputs_unchanged")))
        record.update(completed=True, accepted_stage=accepted, child_receipt_sha256=c.sha(child / "receipt.json"),
                      result_sha256=c.sha(child / "result.json"), outer_receipt_sha256=c.sha(outer / "receipt.json"))
        c.require(accepted, "actual same-source20-record timing acceptance required")
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
        record["inputs_unchanged"] = bool(before) and before == after
        if not record["inputs_unchanged"]:
            record["accepted_stage"] = False
        c.write(output / "pins-after.json", after)
        record["elapsed_seconds"] = time.monotonic() - start
        record["full_native_or_BH_admission"] = False
        record["output_inventory"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": c.sha(p),
                                       "policy": "large_payload" if p.suffix.lower() in (".jsonl", ".npz", ".npy") or p.stat().st_size > 1048576 else "source_or_receipt"}
                                      for folder in (output, outer, child) if folder.exists()
                                      for p in sorted(folder.rglob("*")) if p.is_file()]
        c.write(output / "receipt.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "output_inventory"}, allow_nan=False))
    return 0 if record["accepted_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
