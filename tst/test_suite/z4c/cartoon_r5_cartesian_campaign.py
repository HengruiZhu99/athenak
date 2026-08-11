#!/usr/bin/env python3
"""Prepare, run, and analyze the bounded Cartesian Z4c R5 campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any


STATE_SCHEMA = "athenak_cartoon_r5_cartesian_campaign_v1"
ANALYSIS_SCHEMA = "athenak_cartoon_r5_cartesian_analysis_v1"
BASELINE_SCHEMA = "athenak_cartoon_r5_cuda_baseline_v1"
EXPECTED_BASELINE_SOURCE_COMMIT = "fc37f3a51aebce3187375eba701398c4c910f2af"
EXPECTED_KOKKOS_COMMIT = "6739bc623081648af9e752b616d9671527922cbf"
BACKEND = "Cuda"
RANKS = 4
PERFORMANCE_REGRESSION_LIMIT = 0.05
TIMING_REPEATS = 5
INPUTS = {
    "linear": ("tst/inputs/lwave_z4c.athinput",
               "9f56b4c89aecb73dee09762ecfb8da835aa0fb9f62fe742f0e937b2530c52b1f"),
    "boosted": ("tst/inputs/z4c_boosted.athinput",
                "8e7da544c3e5f6cecb4312a4a21819bf6c77bdb3ed6372f81ac2ce72d1ec05a8"),
}
RANK_WRAPPER = ("tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py",
                "694c1fb4d27ef4b09e59214ffe21bc213fc540d470bfe5bbdf949cd281a5e7d6")
LINEAR_THRESHOLDS = {"o2_n64_rms_l1": 3.5e-11,
                     "o2_error_ratio": 0.25,
                     "o6_n64_rms_l1": 6.0e-12}
BOOSTED_THRESHOLDS = {
    "C-norm2": 1.85e-2, "H-norm2": 4.7e-3, "M-norm2": 1.4e-3,
    "Z-norm2": 3.2e-3, "Mx-norm2": 1.1e-3, "My-norm2": 4.5e-4,
    "Mz-norm2": 4.5e-4, "Theta-norm": 3.1e-5, "hrms": 3.0e-2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_hash(value: Any, label: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label} is not a SHA-256 digest")


def require_git_oid(value: Any, label: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            f"{label} is not a full Git object ID")


def require_finite(value: Any, label: str = "record") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} contains a nonfinite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), f"{label} contains a non-string key")
            require_finite(item, f"{label}.{key}")
        return
    raise RuntimeError(f"{label} contains unsupported type {type(value).__name__}")


def strict_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"nonfinite JSON token {token}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def strict_write(path: Path, payload: Any) -> None:
    require_finite(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def git_value(source: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", expression], text=True).strip()


def validate_source(source: Path, expected_commit: str, expected_tree: str,
                    require_clean: bool = True) -> dict[str, Any]:
    require_git_oid(expected_commit, "expected source commit")
    require_git_oid(expected_tree, "expected source tree")
    require(source.is_dir(), f"source directory is missing: {source}")
    commit = git_value(source, "HEAD")
    tree = git_value(source, "HEAD^{tree}")
    kokkos = git_value(source / "kokkos", "HEAD")
    require(commit == expected_commit, "R5 source commit differs from the declared identity")
    require(tree == expected_tree, "R5 source tree differs from the declared identity")
    require(kokkos == EXPECTED_KOKKOS_COMMIT, "R5 Kokkos commit is not frozen")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--short"], text=True)
    if require_clean:
        require(not status, "R5 source worktree is dirty")
    references: dict[str, dict[str, str]] = {}
    for name, (relative, expected) in INPUTS.items():
        path = source / relative
        require(path.is_file() and sha256(path) == expected,
                f"frozen {name} input differs from its reviewed hash")
        references[name] = {"path": str(path.resolve()), "sha256": expected}
    wrapper_path = source / RANK_WRAPPER[0]
    require(wrapper_path.is_file() and sha256(wrapper_path) == RANK_WRAPPER[1],
            "established rank/GPU wrapper differs from its reviewed hash")
    return {"path": str(source.resolve()), "commit": commit, "tree": tree,
            "kokkos_commit": kokkos, "inputs": references,
            "rank_wrapper": {"path": str(wrapper_path.resolve()),
                             "sha256": RANK_WRAPPER[1]}}


def median_mad(samples: list[float]) -> tuple[float, float]:
    require(samples and all(math.isfinite(item) and item > 0.0 for item in samples),
            "timing samples must be finite and positive")
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(item - median) for item in samples))
    return median, mad


def validate_baseline(payload: Any) -> dict[str, Any]:
    require(isinstance(payload, dict), "baseline is not an object")
    expected = {"schema", "source_commit", "executable_sha256", "input_sha256",
                "backend", "ranks", "gpu_name", "metric", "warmup_sha256",
                "samples", "sample_evidence_sha256"}
    require(set(payload) == expected, "baseline inventory is not exact")
    require(payload["schema"] == BASELINE_SCHEMA, "baseline schema changed")
    require(payload["source_commit"] == EXPECTED_BASELINE_SOURCE_COMMIT,
            "baseline is not the declared pre-integration source")
    require_hash(payload["executable_sha256"], "baseline executable")
    require(payload["input_sha256"] == INPUTS["linear"][1],
            "baseline input differs from the frozen O6 input")
    require(payload["backend"] == BACKEND and payload["ranks"] == RANKS,
            "baseline backend/rank contract changed")
    require(isinstance(payload["gpu_name"], str) and payload["gpu_name"],
            "baseline GPU name is empty")
    require(payload["metric"] == "zone_cycles_per_second",
            "baseline performance metric changed")
    require_hash(payload["warmup_sha256"], "baseline warmup evidence")
    require(isinstance(payload["samples"], list) and
            len(payload["samples"]) == TIMING_REPEATS,
            "baseline must contain exactly five measured samples")
    require(isinstance(payload["sample_evidence_sha256"], list) and
            len(payload["sample_evidence_sha256"]) == TIMING_REPEATS,
            "baseline sample evidence inventory changed")
    for index, digest in enumerate(payload["sample_evidence_sha256"]):
        require_hash(digest, f"baseline sample {index} evidence")
    samples = [float(item) for item in payload["samples"]]
    median, mad = median_mad(samples)
    result = copy.deepcopy(payload)
    result["samples"] = samples
    result["median"] = median
    result["mad"] = mad
    require_finite(result, "baseline")
    return result


def linear_overrides(name: str, resolution: int, nghost: int) -> list[str]:
    block = resolution // 8
    return [f"job/basename={name}", f"mesh/nghost={nghost}",
            f"mesh/nx1={resolution}", f"mesh/nx2={resolution}",
            f"mesh/nx3={resolution}", f"meshblock/nx1={block}",
            f"meshblock/nx2={block}", f"meshblock/nx3={block}",
            "mesh_refinement/max_nmb_per_rank=4096",
            "problem/kx1=1", "problem/kx2=1", "problem/kx3=1"]


def case_inventory() -> list[dict[str, Any]]:
    cases = [
        {"name": "lwave_o2_n32", "kind": "linear", "order": 2,
         "resolution": 32, "role": "functional",
         "overrides": linear_overrides("r5_lwave_o2_n32", 32, 2)},
        {"name": "lwave_o2_n64", "kind": "linear", "order": 2,
         "resolution": 64, "role": "functional",
         "overrides": linear_overrides("r5_lwave_o2_n64", 64, 2)},
        {"name": "lwave_o6_n64", "kind": "linear", "order": 6,
         "resolution": 64, "role": "functional",
         "overrides": linear_overrides("r5_lwave_o6_n64", 64, 4)},
        {"name": "boosted_n128", "kind": "boosted", "order": 6,
         "resolution": 128, "role": "functional",
         "overrides": ["job/basename=boosted"]},
        {"name": "lwave_o6_n64_warmup", "kind": "linear", "order": 6,
         "resolution": 64, "role": "warmup",
         "overrides": linear_overrides("r5_lwave_o6_n64_warmup", 64, 4)},
    ]
    for repeat in range(1, TIMING_REPEATS + 1):
        name = f"lwave_o6_n64_timing_{repeat}"
        cases.append({"name": name, "kind": "linear", "order": 6,
                      "resolution": 64, "role": "timing",
                      "repeat": repeat,
                      "overrides": linear_overrides(f"r5_o6_timing_{repeat}", 64, 4)})
    return cases


def validate_state(state: Any) -> None:
    require(isinstance(state, dict), "campaign state is not an object")
    expected = {"schema", "backend", "ranks", "source", "executable",
                "python", "tooling", "baseline", "thresholds", "case_order",
                "cases"}
    optional = {"analysis"}
    require(set(state).issubset(expected | optional) and expected.issubset(state),
            "campaign state inventory is not exact")
    require(state["schema"] == STATE_SCHEMA and state["backend"] == BACKEND and
            state["ranks"] == RANKS, "campaign execution contract changed")
    require_git_oid(state["source"]["commit"], "campaign source commit")
    require_git_oid(state["source"]["tree"], "campaign source tree")
    require(state["source"]["kokkos_commit"] == EXPECTED_KOKKOS_COMMIT,
            "campaign Kokkos provenance changed")
    require_hash(state["executable"]["sha256"], "campaign executable")
    require_hash(state["python"]["sha256"], "campaign Python")
    require_hash(state["tooling"]["sha256"], "campaign tooling")
    require_hash(state["baseline"]["sha256"], "campaign baseline")
    names = [case["name"] for case in case_inventory()]
    require(state["case_order"] == names and set(state["cases"]) == set(names),
            "campaign case inventory/order changed")
    roles = [state["cases"][name]["role"] for name in names]
    require(roles.count("warmup") == 1 and roles.count("timing") == TIMING_REPEATS,
            "campaign must retain one warmup and exactly five timing repeats")
    for name in names:
        case = state["cases"][name]
        require(case["status"] in {"pending", "running", "complete", "failed"},
                f"invalid status for {name}")
        case_keys = {"name", "kind", "order", "resolution", "role", "overrides",
                     "status", "exit_code", "input", "executable_sha256"}
        if case["role"] == "timing":
            case_keys.add("repeat")
        if case["status"] != "pending":
            case_keys.add("command")
        if case["status"] in {"complete", "failed"}:
            case_keys.add("evidence")
            require(isinstance(case["exit_code"], int),
                    f"completed {name} lacks an integer exit code")
        else:
            require(case["exit_code"] is None,
                    f"unfinished {name} has an exit code")
        require(set(case) == case_keys, f"case inventory is not exact for {name}")
    require_finite(state, "campaign state")


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    source = validate_source(args.source.resolve(), args.expected_source_commit,
                             args.expected_source_tree)
    executable = args.executable.resolve()
    require(executable.is_file() and os.access(executable, os.X_OK),
            "frozen Athena executable is missing or not executable")
    executable_hash = sha256(executable)
    require(executable_hash == args.executable_sha256,
            "Athena executable hash differs from the supplied frozen hash")
    baseline_path = args.baseline.resolve()
    require(baseline_path.is_file() and sha256(baseline_path) == args.baseline_sha256,
            "immutable baseline path/hash mismatch")
    baseline = validate_baseline(strict_load(baseline_path))
    python_path = Path(sys.executable).resolve()
    tooling_path = Path(__file__).resolve()
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    copied_inputs: dict[str, dict[str, str]] = {}
    for name, record in source["inputs"].items():
        destination = output / "inputs" / Path(record["path"]).name
        shutil.copyfile(record["path"], destination)
        require(sha256(destination) == record["sha256"], "copied input hash changed")
        copied_inputs[name] = {"path": str(destination), "sha256": record["sha256"]}
    baseline_copy = output / "baseline.json"
    shutil.copyfile(baseline_path, baseline_copy)
    cases: dict[str, dict[str, Any]] = {}
    inventory = case_inventory()
    for spec in inventory:
        case = copy.deepcopy(spec)
        case.update({"status": "pending", "exit_code": None,
                     "input": copied_inputs[spec["kind"]],
                     "executable_sha256": executable_hash})
        cases[spec["name"]] = case
    state = {"schema": STATE_SCHEMA, "backend": BACKEND, "ranks": RANKS,
             "source": source,
             "executable": {"path": str(executable), "sha256": executable_hash},
             "python": {"path": str(python_path), "sha256": sha256(python_path)},
             "tooling": {"path": str(tooling_path),
                         "sha256": sha256(tooling_path)},
             "baseline": {"path": str(baseline_copy),
                          "sha256": args.baseline_sha256,
                          "summary": {"source_commit": baseline["source_commit"],
                                      "gpu_name": baseline["gpu_name"],
                                      "median": baseline["median"],
                                      "mad": baseline["mad"]}},
             "thresholds": {"linear": LINEAR_THRESHOLDS,
                            "boosted": BOOSTED_THRESHOLDS,
                            "performance_regression_limit":
                                PERFORMANCE_REGRESSION_LIMIT},
             "case_order": [spec["name"] for spec in inventory], "cases": cases}
    validate_state(state)
    strict_write(output / "campaign_state.json", state)


def verify_bound_files(state: dict[str, Any]) -> None:
    current_source = validate_source(Path(state["source"]["path"]),
                                     state["source"]["commit"],
                                     state["source"]["tree"])
    require(current_source == state["source"],
            "source, Kokkos, input, or rank-wrapper binding changed after prepare")
    require(sha256(Path(state["executable"]["path"])) ==
            state["executable"]["sha256"], "executable changed after prepare")
    require(sha256(Path(state["python"]["path"])) == state["python"]["sha256"],
            "Python interpreter changed after prepare")
    require(sha256(Path(state["tooling"]["path"])) == state["tooling"]["sha256"],
            "campaign tooling changed after prepare")
    wrapper = state["source"]["rank_wrapper"]
    require(sha256(Path(wrapper["path"])) == wrapper["sha256"],
            "rank wrapper changed after prepare")
    for name, record in state["source"]["inputs"].items():
        require(sha256(Path(record["path"])) == record["sha256"],
                f"source {name} input changed after prepare")


def run_case(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    validate_state(state)
    require(args.case in state["cases"], f"unknown case {args.case}")
    index = state["case_order"].index(args.case)
    require(all(state["cases"][name]["status"] == "complete"
                for name in state["case_order"][:index]),
            "earlier campaign cases are incomplete")
    case = state["cases"][args.case]
    require(case["status"] == "pending", f"case {args.case} is not pending")
    root = state_path.parent
    run_dir = root / "runs" / args.case
    require(not run_dir.exists(), f"case output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    bindings = run_dir / "bindings"
    verify_bound_files(state)
    executable = Path(state["executable"]["path"])
    input_path = Path(case["input"]["path"])
    wrapper = Path(state["source"]["rank_wrapper"]["path"])
    require(sha256(input_path) == case["input"]["sha256"],
            "input changed after prepare")
    athena = [str(executable), "-i", str(input_path), "-d", str(run_dir),
              *case["overrides"]]
    command = ["srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
               "--cpus-per-task=8", "--gpus-per-task=1",
               "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
               "--kill-on-bad-exit=1", state["python"]["path"], str(wrapper),
               "--evidence-dir", str(bindings), "--require-cuda", "--", *athena]
    strict_write(run_dir / "command.json", {"command": command,
                 "source_commit": state["source"]["commit"],
                 "executable_sha256": state["executable"]["sha256"],
                 "input_sha256": case["input"]["sha256"]})
    case["status"] = "running"
    case["command"] = command
    strict_write(state_path, state)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "8", "KOKKOS_NUM_THREADS": "8",
                        "MPICH_GPU_SUPPORT_ENABLED": "1",
                        "MPICH_GPU_IPC_ENABLED": "0",
                        "MPICH_OFI_NIC_POLICY": "GPU"})
    log = run_dir / "run.log"
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    case["exit_code"] = result.returncode
    case["status"] = "complete" if result.returncode == 0 else "failed"
    case["evidence"] = {"command.json": sha256(run_dir / "command.json"),
                        "run.log": sha256(log)}
    strict_write(state_path, state)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def one_match(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    require(len(matches) == 1, f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def runtime_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    wall = float(one_match(rf"^cpu time used\s*=\s*({NUMBER})\s*$", text,
                           "driver wall time"))
    throughput = float(one_match(
        rf"^zone-cycles/cpu_second\s*=\s*({NUMBER})\s*$", text,
        "zone-cycle throughput"))
    block_cycles = int(one_match(r"^MeshBlock-cycles\s*=\s*(\d+)\s*$", text,
                                 "MeshBlock-cycle count"))
    require("Terminating on time limit" in text or "Terminating on cycle limit" in text,
            "run did not terminate on its declared scientific limit")
    result = {"driver_wall_seconds": wall,
              "zone_cycles_per_second": throughput,
              "meshblock_cycles": block_cycles}
    require_finite(result, "runtime metrics")
    require(wall > 0.0 and throughput > 0.0 and block_cycles > 0,
            "runtime metrics are not positive")
    return result


def parse_table(path: Path) -> tuple[list[str], list[list[float]]]:
    require(path.is_file(), f"required table is missing: {path}")
    header: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            header.append(line)
            continue
        try:
            row = [float(token) for token in line.split()]
        except ValueError as error:
            raise RuntimeError(f"malformed numeric row in {path}") from error
        require(row and all(math.isfinite(value) for value in row),
                f"nonfinite or empty row in {path}")
        rows.append(row)
    require(rows, f"numeric table is empty: {path}")
    require(len({len(row) for row in rows}) == 1, f"ragged table: {path}")
    return header, rows


def linear_error(run_dir: Path) -> dict[str, Any]:
    matches = list(run_dir.glob("*-errs.dat"))
    require(len(matches) == 1, "linear-wave error-file inventory is not exact")
    header, rows = parse_table(matches[0])
    require(len(rows) == 1 and len(rows[0]) >= 6,
            "linear-wave error table must contain one complete row")
    require(any("RMS-L1" in line for line in header),
            "linear-wave error header is missing RMS-L1")
    return {"path": str(matches[0]), "sha256": sha256(matches[0]),
            "resolution": [int(rows[0][0]), int(rows[0][1]), int(rows[0][2])],
            "cycle": int(rows[0][3]), "rms_l1": rows[0][4],
            "linfty": rows[0][5]}


def history(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#  [1]=")]
    require(len(headers) == 1, "history header inventory is not exact")
    names = re.findall(r"\[\d+\]=(\S+)", headers[0])
    require(len(names) == len(set(names)), "history contains duplicate labels")
    rows = [[float(token) for token in line.split()] for line in lines
            if line.strip() and not line.startswith("#")]
    require(rows and all(len(row) == len(names) for row in rows),
            "history rows do not match the exact header")
    require(all(math.isfinite(value) for row in rows for value in row),
            "history contains nonfinite values")
    return {name: [row[index] for row in rows] for index, name in enumerate(names)}


def boosted_metrics(run_dir: Path) -> dict[str, Any]:
    histories = list(run_dir.glob("*.z4c.user.hst"))
    horizons = list(run_dir.glob("*.horizon_summary_0.txt"))
    require(len(histories) == 1 and len(horizons) == 1,
            "boosted output inventory is not exact")
    hst = history(histories[0])
    required = tuple(key for key in BOOSTED_THRESHOLDS if key != "hrms")
    require(all(key in hst and len(hst[key]) >= 4 for key in required),
            "boosted history lacks the established fourth-sample constraint inventory")
    horizon_header, horizon_rows = parse_table(horizons[0])
    require(len(horizon_header) == 1, "horizon header inventory is not exact")
    labels = re.findall(r"\d+:([^\s]+)", horizon_header[0])
    require("hrms" in labels and all(len(row) == len(labels) for row in horizon_rows),
            "horizon summary inventory is malformed")
    values = {key: hst[key][3] for key in required}
    values["hrms"] = horizon_rows[-1][labels.index("hrms")]
    require_finite(values, "boosted metrics")
    return {"values": values,
            "history": {"path": str(histories[0]), "sha256": sha256(histories[0])},
            "horizon": {"path": str(horizons[0]), "sha256": sha256(horizons[0])}}


def binding_records(run_dir: Path, expected_gpu: str) -> list[dict[str, Any]]:
    files = sorted((run_dir / "bindings").glob("rank_binding_*.json"))
    require(len(files) == RANKS, "rank binding inventory is not exactly four")
    records = [strict_load(path) for path in files]
    expected_keys = {"rank", "local_rank", "hostname", "cuda_visible_devices",
                     "visible_device_token", "selected_uuid", "gpu_name",
                     "binding_verified"}
    require(all(isinstance(record, dict) and set(record) == expected_keys
                for record in records), "rank binding record inventory is not exact")
    require([record.get("rank") for record in records] == list(range(RANKS)),
            "rank binding IDs are missing, duplicate, or out of order")
    uuids = [record.get("selected_uuid") for record in records]
    require(len(set(uuids)) == RANKS and all(uuids),
            "CUDA ranks are not bound to four unique GPUs")
    require(all(record.get("binding_verified") is True and
                record.get("gpu_name") == expected_gpu for record in records),
            "current GPU binding differs from the immutable baseline hardware")
    for path, record in zip(files, records):
        record["path"] = str(path)
        record["sha256"] = sha256(path)
    return records


def analyze_state(state_path: Path, verify_files: bool = True) -> dict[str, Any]:
    state = strict_load(state_path)
    validate_state(state)
    if verify_files:
        verify_bound_files(state)
    require(all(case["status"] == "complete" and case["exit_code"] == 0
                for case in state["cases"].values()),
            "all campaign cases must complete before analysis")
    root = state_path.parent
    baseline_path = Path(state["baseline"]["path"])
    require(sha256(baseline_path) == state["baseline"]["sha256"],
            "baseline changed after prepare")
    baseline = validate_baseline(strict_load(baseline_path))
    observations: dict[str, Any] = {}
    for name in state["case_order"]:
        case = state["cases"][name]
        run_dir = root / "runs" / name
        log = run_dir / "run.log"
        require(log.is_file() and sha256(log) == case["evidence"]["run.log"],
                f"raw log changed for {name}")
        command_path = run_dir / "command.json"
        require(command_path.is_file() and
                sha256(command_path) == case["evidence"]["command.json"],
                f"command evidence changed for {name}")
        command_record = strict_load(command_path)
        require(isinstance(command_record, dict) and
                set(command_record) == {"command", "source_commit",
                                        "executable_sha256", "input_sha256"},
                f"command evidence inventory changed for {name}")
        require(command_record["command"] == case["command"] and
                command_record["source_commit"] == state["source"]["commit"] and
                command_record["executable_sha256"] ==
                    state["executable"]["sha256"] and
                command_record["input_sha256"] == case["input"]["sha256"] and
                command_record["command"].count(state["executable"]["path"]) == 1,
                f"command/source/executable/input binding changed for {name}")
        record: dict[str, Any] = {
            "role": case["role"], "kind": case["kind"],
            "order": case["order"], "resolution": case["resolution"],
            "runtime": runtime_metrics(log),
            "bindings": binding_records(run_dir, baseline["gpu_name"]),
            "raw_log": {"path": str(log), "sha256": sha256(log)},
            "command": {"path": str(command_path), "sha256": sha256(command_path)},
        }
        if case["kind"] == "linear":
            record["linear_error"] = linear_error(run_dir)
        else:
            record["boosted"] = boosted_metrics(run_dir)
        observations[name] = record
    coarse = observations["lwave_o2_n32"]["linear_error"]["rms_l1"]
    fine = observations["lwave_o2_n64"]["linear_error"]["rms_l1"]
    sixth = observations["lwave_o6_n64"]["linear_error"]["rms_l1"]
    require(coarse > 0.0 and fine > 0.0 and sixth >= 0.0,
            "linear-wave errors are not physically valid")
    ratio = fine / coarse
    linear_checks = {"o2_n64_rms_l1": fine,
                     "o2_error_ratio": ratio, "o6_n64_rms_l1": sixth}
    o6_errors = {name: observations[name]["linear_error"]["rms_l1"]
                 for name in state["case_order"]
                 if state["cases"][name]["kind"] == "linear" and
                 state["cases"][name]["order"] == 6}
    linear_pass = (all(linear_checks[key] <= LINEAR_THRESHOLDS[key]
                       for key in linear_checks) and
                   all(value <= LINEAR_THRESHOLDS["o6_n64_rms_l1"]
                       for value in o6_errors.values()))
    boosted = observations["boosted_n128"]["boosted"]["values"]
    boosted_pass = all(boosted[key] <= limit
                       for key, limit in BOOSTED_THRESHOLDS.items())
    timing_names = [name for name in state["case_order"]
                    if state["cases"][name]["role"] == "timing"]
    require(timing_names == [f"lwave_o6_n64_timing_{index}"
                             for index in range(1, TIMING_REPEATS + 1)],
            "measured timing inventory/order changed")
    samples = [observations[name]["runtime"]["zone_cycles_per_second"]
               for name in timing_names]
    current_median, current_mad = median_mad(samples)
    ratio_to_baseline = current_median / baseline["median"]
    regression = 1.0 - ratio_to_baseline
    performance_pass = regression <= PERFORMANCE_REGRESSION_LIMIT
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "verdict": "pass" if linear_pass and boosted_pass and performance_pass else "fail",
        "source_commit": state["source"]["commit"],
        "executable_sha256": state["executable"]["sha256"],
        "linear": {"values": linear_checks, "o6_all_run_errors": o6_errors,
                   "thresholds": LINEAR_THRESHOLDS, "pass": linear_pass},
        "boosted": {"values": boosted, "thresholds": BOOSTED_THRESHOLDS,
                    "pass": boosted_pass},
        "performance": {
            "metric": "zone_cycles_per_second", "warmup_count": 1,
            "measured_repeat_count": len(samples), "samples": samples,
            "median": current_median, "mad": current_mad,
            "baseline_source_commit": baseline["source_commit"],
            "baseline_samples": baseline["samples"],
            "baseline_median": baseline["median"], "baseline_mad": baseline["mad"],
            "ratio_to_baseline": ratio_to_baseline,
            "regression_fraction": regression,
            "regression_limit": PERFORMANCE_REGRESSION_LIMIT,
            "pass": performance_pass,
        },
        "observations": observations,
    }
    require_finite(summary, "analysis")
    return summary


def analyze(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    summary = analyze_state(state_path)
    analysis_path = state_path.parent / "r5_analysis.json"
    strict_write(analysis_path, summary)
    state = strict_load(state_path)
    state["analysis"] = {"path": str(analysis_path), "sha256": sha256(analysis_path),
                         "verdict": summary["verdict"]}
    validate_state(state)
    strict_write(state_path, state)
    if summary["verdict"] != "pass":
        raise SystemExit("R5 Cartesian regression/performance gate failed")


def synthetic_baseline(gpu_name: str = "NVIDIA A100-SXM4-40GB") -> dict[str, Any]:
    return {"schema": BASELINE_SCHEMA,
            "source_commit": EXPECTED_BASELINE_SOURCE_COMMIT,
            "executable_sha256": "1" * 64, "input_sha256": INPUTS["linear"][1],
            "backend": BACKEND, "ranks": RANKS, "gpu_name": gpu_name,
            "metric": "zone_cycles_per_second", "warmup_sha256": "2" * 64,
            "samples": [1000.0, 995.0, 1005.0, 1002.0, 998.0],
            "sample_evidence_sha256": [f"{index:x}" * 64 for index in range(3, 8)]}


def self_test(source: Path) -> None:
    source = source.resolve()
    current_commit = git_value(source, "HEAD")
    current_tree = git_value(source, "HEAD^{tree}")
    source_record = validate_source(source, current_commit, current_tree,
                                    require_clean=False)
    try:
        validate_source(source, "0" * 40, current_tree, require_clean=False)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("mismatched expected source commit was accepted")
    try:
        validate_source(source, current_commit, "0" * 40, require_clean=False)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("mismatched expected source tree was accepted")
    try:
        validate_source(source, "", current_tree, require_clean=False)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("missing expected source identity was accepted")
    baseline = validate_baseline(synthetic_baseline())
    require(baseline["median"] == 1000.0 and baseline["mad"] == 2.0,
            "median/MAD oracle failed")
    require(len(case_inventory()) == 10 and
            sum(case["role"] == "timing" for case in case_inventory()) == 5,
            "campaign inventory fixture changed")
    try:
        require_finite({"bad": float("inf")})
    except RuntimeError:
        pass
    else:
        raise RuntimeError("strict finite gate accepted Infinity")
    malformed = synthetic_baseline()
    malformed["samples"] = malformed["samples"][:4]
    try:
        validate_baseline(malformed)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("baseline accepted fewer than five samples")
    with tempfile.TemporaryDirectory(prefix="cartoon-r5-selftest-") as directory:
        root = Path(directory)
        baseline_path = root / "baseline.json"
        strict_write(baseline_path, synthetic_baseline())
        state = {"schema": STATE_SCHEMA, "backend": BACKEND, "ranks": RANKS,
                 "source": source_record,
                 "executable": {"path": "/synthetic/athena", "sha256": "8" * 64},
                 "python": {"path": sys.executable, "sha256": "9" * 64},
                 "tooling": {"path": str(Path(__file__).resolve()),
                             "sha256": sha256(Path(__file__).resolve())},
                 "baseline": {"path": str(baseline_path),
                              "sha256": sha256(baseline_path), "summary": {}},
                 "thresholds": {"linear": LINEAR_THRESHOLDS,
                                "boosted": BOOSTED_THRESHOLDS,
                                "performance_regression_limit":
                                    PERFORMANCE_REGRESSION_LIMIT},
                 "case_order": [], "cases": {}}
        for spec in case_inventory():
            case = copy.deepcopy(spec)
            case.update({"status": "complete", "exit_code": 0,
                         "input": {"path": "/synthetic/input",
                                   "sha256": INPUTS[spec["kind"]][1]},
                         "executable_sha256": "8" * 64})
            state["case_order"].append(spec["name"])
            state["cases"][spec["name"]] = case
            run_dir = root / "runs" / spec["name"]
            (run_dir / "bindings").mkdir(parents=True)
            throughput = 960.0 + (spec.get("repeat", 0) - 3) * 2.0
            log = run_dir / "run.log"
            log.write_text("Terminating on time limit\nMeshBlock-cycles = 100\n"
                           "cpu time used  = 1.0\n"
                           f"zone-cycles/cpu_second = {throughput}\n",
                           encoding="utf-8")
            command = run_dir / "command.json"
            synthetic_command = ["synthetic", state["executable"]["path"]]
            case["command"] = synthetic_command
            strict_write(command, {"command": synthetic_command,
                                   "source_commit": state["source"]["commit"],
                                   "executable_sha256":
                                       state["executable"]["sha256"],
                                   "input_sha256": case["input"]["sha256"]})
            for rank in range(RANKS):
                strict_write(run_dir / "bindings" / f"rank_binding_{rank:04d}.json",
                             {"rank": rank, "local_rank": rank,
                              "hostname": "synthetic", "cuda_visible_devices": str(rank),
                              "visible_device_token": str(rank),
                              "selected_uuid": f"GPU-{rank}",
                              "gpu_name": baseline["gpu_name"],
                              "binding_verified": True})
            case["evidence"] = {"run.log": sha256(log),
                                "command.json": sha256(command)}
            if spec["kind"] == "linear":
                error = {"lwave_o2_n32": 1.2e-10, "lwave_o2_n64": 3.0e-11,
                         "lwave_o6_n64": 5.0e-12}.get(spec["name"], 5.0e-12)
                (run_dir / f"{spec['name']}-errs.dat").write_text(
                    "# Nx1 Nx2 Nx3 Ncycle RMS-L1 L-infty\n"
                    f"{spec['resolution']} {spec['resolution']} {spec['resolution']} "
                    f"10 {error} {error}\n", encoding="utf-8")
            else:
                labels = list(BOOSTED_THRESHOLDS)[:-1]
                header = "#  " + " ".join(f"[{i + 1}]={name}"
                                             for i, name in enumerate(labels)) + "\n"
                rows = "\n".join(" ".join("1e-6" for _ in labels)
                                   for _ in range(4)) + "\n"
                (run_dir / "boosted.z4c.user.hst").write_text(header + rows,
                                                               encoding="utf-8")
                (run_dir / "boosted.horizon_summary_0.txt").write_text(
                    "# 1:iter 2:hrms\n1 1e-3\n", encoding="utf-8")
        strict_write(root / "campaign_state.json", state)
        summary = analyze_state(root / "campaign_state.json", verify_files=False)
        require(summary["verdict"] == "pass" and
                summary["performance"]["measured_repeat_count"] == 5,
                "synthetic passing campaign did not pass")
        slower = strict_load(root / "campaign_state.json")
        timing = "lwave_o6_n64_timing_1"
        log = root / "runs" / timing / "run.log"
        log.write_text(log.read_text(encoding="utf-8").replace("956.0", "900.0"),
                       encoding="utf-8")
        slower["cases"][timing]["evidence"]["run.log"] = sha256(log)
        for repeat in range(2, 6):
            name = f"lwave_o6_n64_timing_{repeat}"
            path = root / "runs" / name / "run.log"
            path.write_text(re.sub(r"zone-cycles/cpu_second = \S+",
                                   "zone-cycles/cpu_second = 900.0",
                                   path.read_text(encoding="utf-8")), encoding="utf-8")
            slower["cases"][name]["evidence"]["run.log"] = sha256(path)
        strict_write(root / "campaign_state.json", slower)
        failed = analyze_state(root / "campaign_state.json", verify_files=False)
        require(failed["verdict"] == "fail" and
                not failed["performance"]["pass"],
                "performance regression was not flagged")
    print("Cartoon R5 Cartesian campaign tooling self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--expected-source-commit", required=True)
    prepare_parser.add_argument("--expected-source-tree", required=True)
    prepare_parser.add_argument("--executable", type=Path, required=True)
    prepare_parser.add_argument("--executable-sha256", required=True)
    prepare_parser.add_argument("--baseline", type=Path, required=True)
    prepare_parser.add_argument("--baseline-sha256", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run-case")
    run_parser.add_argument("--state", type=Path, required=True)
    run_parser.add_argument("--case", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--state", type=Path, required=True)
    self_parser = subparsers.add_parser("self-test")
    self_parser.add_argument("--source", type=Path, default=Path(__file__).parents[3])
    return result


def main() -> int:
    args = parser().parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run-case":
        run_case(args)
    elif args.action == "analyze":
        analyze(args)
    else:
        self_test(args.source)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
