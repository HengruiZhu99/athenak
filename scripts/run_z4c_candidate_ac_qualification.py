#!/usr/bin/env python3
"""Run the immutable AthenaK Candidate-A/C puncture qualification campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


SCHEMA = "ATHENAK_Z4C_CANDIDATE_AC_QUALIFICATION"
VERSION = 1
GAUGES = ("standard", "candidate_a", "candidate_c")
RESOLUTIONS = {
    "R0": {"nx": 8, "dx_min": 20.0 / 8.0 / 16.0},
    "R1": {"nx": 12, "dx_min": 20.0 / 12.0 / 16.0},
    "R2": {"nx": 16, "dx_min": 20.0 / 16.0 / 16.0},
}
TARGETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
TIME_RE = re.compile(r"^time=([0-9.eE+-]+) cycle=([0-9]+)$", re.MULTILINE)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_write(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_text(command: list[str], cwd: pathlib.Path) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def repository_identity(repo: pathlib.Path) -> dict[str, object]:
    status = run_text(["git", "status", "--porcelain=v1"], repo)
    if status:
        raise RuntimeError("campaign source worktree is not clean")
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"], repo).strip(),
        "branch": run_text(["git", "branch", "--show-current"], repo).strip(),
        "remote": run_text(["git", "remote", "get-url", "hengrui"], repo).strip(),
        "submodules": run_text(["git", "submodule", "status", "--recursive"], repo).strip(),
    }


def read_last_numeric_row(path: pathlib.Path) -> list[str]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    if not rows:
        raise RuntimeError(f"no numeric rows in {path}")
    return rows[-1]


def read_last_csv_row(path: pathlib.Path) -> list[str]:
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    if not rows:
        raise RuntimeError(f"no data rows in {path}")
    return rows[-1]


def exact_time(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 64.0 * math.ulp(max(1.0, abs(expected)))


def validate_segment(case_dir: pathlib.Path, basename: str, gauge: str,
                     target: float, stdout_path: pathlib.Path) -> dict[str, object]:
    stdout = stdout_path.read_text(encoding="utf-8")
    matches = TIME_RE.findall(stdout)
    if not matches:
        raise RuntimeError("application stdout has no terminal time/cycle")
    accepted_time = float(matches[-1][0])
    accepted_cycle = int(matches[-1][1])
    if not exact_time(accepted_time, target):
        raise RuntimeError(f"terminal time {accepted_time} does not equal {target}")

    history = case_dir / f"{basename}.user.hst"
    constraints = case_dir / f"{basename}.z4c.user.hst"
    gauge_csv = case_dir / "gauge_source_diagnostics.csv"
    horizon = case_dir / f"{basename}.horizon_summary_0.txt"
    for required in (history, constraints, gauge_csv, horizon):
        if not required.is_file() or required.stat().st_size == 0:
            raise RuntimeError(f"missing required evidence {required}")

    history_row = [float(value) for value in read_last_numeric_row(history)]
    constraint_row = [float(value) for value in read_last_numeric_row(constraints)]
    gauge_row = read_last_csv_row(gauge_csv)
    horizon_row = [float(value) for value in read_last_numeric_row(horizon)]
    if not (exact_time(history_row[0], target) and exact_time(constraint_row[0], target)
            and exact_time(float(gauge_row[0]), target) and exact_time(horizon_row[1], target)):
        raise RuntimeError("accepted diagnostics and horizon are not at the target time")
    if gauge_row[2] != gauge:
        raise RuntimeError(f"gauge telemetry says {gauge_row[2]}, expected {gauge}")
    finite_values = history_row + constraint_row + horizon_row
    finite_values += [float(value) for index, value in enumerate(gauge_row)
                      if index != 2 and value.lower() not in {"nan", "-nan"}]
    if not all(math.isfinite(value) for value in finite_values):
        raise RuntimeError("nonfinite accepted diagnostic or horizon value")
    if float(gauge_row[3]) <= 0.0 or float(gauge_row[4]) <= 0.0:
        raise RuntimeError("nonpositive accepted lapse or raw chi")
    if float(gauge_row[5]) <= 0.0 or int(float(gauge_row[8])) != 0:
        raise RuntimeError("inadmissible metric or invalid accepted point")

    restarts = sorted((case_dir / "rst").glob(f"{basename}.*.rst"))
    if not restarts:
        raise RuntimeError("missing terminal restart")
    checkpoint = restarts[-1]
    return {
        "accepted_time": accepted_time,
        "accepted_cycle": accepted_cycle,
        "checkpoint": str(checkpoint.relative_to(case_dir)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "history_sha256": sha256(history),
        "constraints_sha256": sha256(constraints),
        "gauge_diagnostics_sha256": sha256(gauge_csv),
        "horizon_sha256": sha256(horizon),
        "minimum_lapse": float(gauge_row[3]),
        "minimum_chi": float(gauge_row[4]),
        "minimum_metric_minor": float(gauge_row[5]),
        "maximum_shift": float(gauge_row[6]),
        "maximum_conformal_gamma": float(gauge_row[7]),
        "invalid_points": int(float(gauge_row[8])),
        "horizon_mass": horizon_row[2],
        "horizon_area": horizon_row[7],
        "horizon_hrms": horizon_row[8],
        "horizon_mean_radius": horizon_row[10],
        "horizon_minimum_radius": horizon_row[11],
    }


def create_manifest(args: argparse.Namespace, repo: pathlib.Path, run_root: pathlib.Path,
                    binary: pathlib.Path, input_path: pathlib.Path) -> dict[str, object]:
    if run_root.exists():
        raise RuntimeError(f"refusing existing run root {run_root}")
    run_root.mkdir(parents=True)
    identity = repository_identity(repo)
    manifest = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "repository": identity,
        "binary": {"path": str(binary), "bytes": binary.stat().st_size,
                   "sha256": sha256(binary)},
        "input": {"path": str(input_path.relative_to(repo)),
                  "sha256": sha256(input_path)},
        "compiler": run_text(["c++", "--version"], repo).splitlines()[0],
        "configuration": run_text([str(binary), "-c"], repo),
        "resources": {"mpi_ranks": 1, "openmp_threads": 8,
                      "omp_proc_bind": "false"},
        "fixed_policy": {
            "mass": 1.0, "domain": [-10.0, 10.0], "meshblock_points": 4,
            "nghost": 4, "static_refinement_levels": 5,
            "inner_refined_cube": [-0.6, 0.6], "rk_integrator": "rk4",
            "cfl_number": 0.1, "ko_dissipation": 0.02,
            "constraint_damping_kappa1": 0.02, "shift_eta": 2.0,
            "raw_chi_div_floor": -1000.0, "chi_floor_enabled": False,
        },
        "gauges": list(GAUGES), "resolutions": RESOLUTIONS,
        "targets": list(TARGETS), "schedule": [],
    }
    sequence = 0
    for target in TARGETS:
        for resolution in RESOLUTIONS:
            for gauge in GAUGES:
                manifest["schedule"].append({"sequence": sequence, "target": target,
                                             "resolution": resolution, "gauge": gauge})
                sequence += 1
    canonical_write(run_root / "manifest.json", manifest)
    return manifest


def validate_frozen(manifest: dict[str, object], repo: pathlib.Path,
                    binary: pathlib.Path, input_path: pathlib.Path) -> None:
    identity = repository_identity(repo)
    if identity != manifest["repository"]:
        raise RuntimeError("repository identity drift")
    if sha256(binary) != manifest["binary"]["sha256"]:
        raise RuntimeError("binary hash drift")
    if sha256(input_path) != manifest["input"]["sha256"]:
        raise RuntimeError("input hash drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    run_root = args.run_root.resolve()
    input_path = repo / "inputs/z4c/onepuncture/z4c_candidate_ac_qualification.athinput"
    lock_path = pathlib.Path("/tmp/athenak_z4c_candidate_ac_qualification.lock")
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another AthenaK Candidate-A/C campaign owns the lock") from error

    if args.initialize:
        manifest = create_manifest(args, repo, run_root, binary, input_path)
    else:
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != VERSION:
        raise RuntimeError("manifest schema mismatch")
    outcomes_path = run_root / "outcomes.json"
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8")) if outcomes_path.exists() else []

    while len(outcomes) < len(manifest["schedule"]):
        validate_frozen(manifest, repo, binary, input_path)
        item = manifest["schedule"][len(outcomes)]
        gauge = item["gauge"]
        resolution = item["resolution"]
        target = float(item["target"])
        case_id = f"{gauge}_{resolution.lower()}"
        basename = f"z4c_{case_id}"
        case_dir = run_root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        segment_dir = case_dir / "segments" / f"segment_{item['sequence']:03d}_t{target:g}"
        segment_dir.mkdir(parents=True)
        previous = [entry for entry in outcomes if entry.get("case_id") == case_id
                    and entry.get("classification") == "complete"]
        command = [str(binary)]
        if previous:
            restart = case_dir / previous[-1]["validation"]["checkpoint"]
            if sha256(restart) != previous[-1]["validation"]["checkpoint_sha256"]:
                raise RuntimeError("restart checkpoint hash drift")
            command += ["-r", str(restart)]
        else:
            command += ["-i", str(input_path)]
        command += [
            f"job/basename={basename}", f"z4c/shift_gauge={gauge}",
            f"mesh/nx1={RESOLUTIONS[resolution]['nx']}",
            f"mesh/nx2={RESOLUTIONS[resolution]['nx']}",
            f"mesh/nx3={RESOLUTIONS[resolution]['nx']}",
            "meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4",
            f"time/tlim={target:.17g}", "time/nlim=-1",
            f"fastflow/start_time_0={target:.17g}",
            f"fastflow/stop_time_0={target:.17g}",
            "problem/final_horizon=true",
            "problem/gauge_diagnostics_file=gauge_source_diagnostics.csv",
            "output1/dt=0.1", "output2/dt=1000", "output3/dt=1000",
            "output4/dt=1000", "output5/dt=0.1", "output6/dt=1000",
        ]
        (segment_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "8", "OMP_PROC_BIND": "false"})
        stdout_path = segment_dir / "stdout.log"
        stderr_path = segment_dir / "stderr.log"
        resource_path = segment_dir / "resource.txt"
        start = time.monotonic()
        print(f"START sequence={item['sequence']} case={case_id} target={target:g}", flush=True)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(["/usr/bin/time", "-v", "-o", str(resource_path),
                                        *command], cwd=case_dir, env=env,
                                       stdout=stdout, stderr=stderr)
        outcome: dict[str, object] = {
            "sequence": item["sequence"], "case_id": case_id, "gauge": gauge,
            "resolution": resolution, "target": target,
            "command": str((segment_dir / "command.txt").relative_to(run_root)),
            "stdout": str(stdout_path.relative_to(run_root)),
            "stderr": str(stderr_path.relative_to(run_root)),
            "resource": str(resource_path.relative_to(run_root)),
            "exit_status": completed.returncode,
            "wall_seconds": time.monotonic() - start,
        }
        try:
            if completed.returncode != 0:
                raise RuntimeError(f"application exit status {completed.returncode}")
            outcome["validation"] = validate_segment(case_dir, basename, gauge, target,
                                                       stdout_path)
            outcome["classification"] = "complete"
        except Exception as error:  # preserve an exact numerical/evidence terminal
            outcome["classification"] = "failed"
            outcome["terminal_reason"] = str(error)
        for key in ("command", "stdout", "stderr", "resource"):
            artifact = run_root / outcome[key]
            outcome[f"{key}_sha256"] = sha256(artifact)
        outcomes.append(outcome)
        canonical_write(outcomes_path, outcomes)
        print(f"END sequence={item['sequence']} case={case_id} target={target:g} "
              f"classification={outcome['classification']} wall={outcome['wall_seconds']:.1f}s",
              flush=True)
        if outcome["classification"] != "complete":
            break
        if not args.all:
            break

    terminal_status = "complete" if len(outcomes) == len(manifest["schedule"]) and all(
        item["classification"] == "complete" for item in outcomes) else "partial_or_failed"
    canonical_write(run_root / "terminal.json", {
        "schema": SCHEMA + "_TERMINAL", "schema_version": VERSION,
        "status": terminal_status, "completed_schedule_items": len(outcomes),
        "total_schedule_items": len(manifest["schedule"]),
        "outcomes_sha256": sha256(outcomes_path) if outcomes_path.exists() else None,
    })
    return 0 if terminal_status == "complete" or not args.all else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise
