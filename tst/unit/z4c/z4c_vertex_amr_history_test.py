#!/usr/bin/env python3
"""Native-VC AMR history authority, replay, restart, and compatibility regression."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fresh(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def execute(command: list[str], cwd: Path, expect_success: bool = True) -> str:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OMP_PROC_BIND", "false")
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True, check=False)
    (cwd / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cwd / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if expect_success:
        require(completed.returncode == 0,
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                f"{completed.stdout}\n{completed.stderr}")
    else:
        require(completed.returncode != 0,
                f"command unexpectedly passed: {' '.join(command)}")
    return completed.stdout + completed.stderr


def command(athena: str, input_path: str, basename: str, mode: str,
            history: Path, overrides: list[str] | None = None) -> list[str]:
    return [
        athena, "-i", input_path,
        f"job/basename={basename}",
        f"mesh_refinement/amr_history_mode={mode}",
        f"mesh_refinement/amr_history_file={history}",
        *(overrides or []),
    ]


def history_records(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines()]
    require(len(records) == 4, "fixture must record initial/refine/derefine events")
    return records[0], records[1:]


def replay_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines()]
    require(len(records) == 2, "replay must accept exactly two topology events")
    require(all(record["action"] == "replay" and record["exact_match"]
                and record["ulp_difference"] == 0 for record in records),
            "replay ledger lacks exact hierarchy/time acceptance")
    return records


def fnv1a64(text: str) -> str:
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def legacy_cell_authority(source: Path, destination: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema"] = 1
    del header["grid_centering"]
    del header["centering_schema"]
    del header["checksum"]
    base = json.dumps(header, separators=(",", ":"))
    require(base.endswith("}"), "legacy history header is malformed")
    header["checksum"] = fnv1a64(base[:-1])
    lines[0] = json.dumps(header, separators=(",", ":"))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_same_authority(events: list[dict], ledger: list[dict]) -> None:
    for event, accepted in zip(events[1:], ledger):
        require(event["event"] == accepted["event"] and
                event["time_hex"] == accepted["authority_time_hex"] and
                event["tree_checksum"] == accepted["tree_checksum"] and
                event["leaf_count"] == accepted["leaves"],
                "replay event differs from the native VC authority")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mpiexec")
    parser.add_argument("--np-flag", default="-n")
    args = parser.parse_args()

    work = Path(args.work_dir)
    fresh(work)
    roots = {name: work / name for name in (
        "record", "replay", "high", "restart_prefix", "restart_continue",
        "cell_record", "bridge_reject", "bridge_accept", "mpi_replay")}
    for root in roots.values():
        fresh(root)

    authority = roots["record"] / "vc-history.jsonl"
    execute(command(args.athena, args.input, "vc_record", "record", authority),
            roots["record"])
    header, events = history_records(authority)
    require(header["schema"] == 2 and header["grid_centering"] == "vertex" and
            header["centering_schema"] == 1,
            "native VC authority lacks centering provenance")
    require([event["event"] for event in events] == [0, 1, 2],
            "native VC event indices are not contiguous")

    replay_log = execute(command(
        args.athena, args.input, "vc_replay", "replay", authority,
        ["time/cfl_number=0.07", "time/nlim=6"]), roots["replay"])
    require("AMR_HISTORY_TIMESTEP_CLIP" in replay_log,
            "different-CFL VC replay did not clip to an authority time")
    replay = replay_records(roots["replay"] / "vc_replay.amr_history_replay.jsonl")
    require_same_authority(events, replay)

    high_log = execute(command(
        args.athena, args.input, "vc_high", "replay", authority,
        ["mesh/nx1=64", "mesh/nx2=64", "meshblock/nx1=32",
         "meshblock/nx2=32", "time/cfl_number=0.07", "time/nlim=6"]),
        roots["high"])
    require("AMR_HISTORY_REPLAY" in high_log,
            "2x-cells-per-block VC replay emitted no acceptance evidence")
    high = replay_records(roots["high"] / "vc_high.amr_history_replay.jsonl")
    require_same_authority(events, high)

    execute(command(
        args.athena, args.input, "vc_restart", "replay", authority,
        ["time/cfl_number=0.07", "time/nlim=2"]), roots["restart_prefix"])
    restart = sorted((roots["restart_prefix"] / "rst").glob("vc_restart.*.rst"))[-1]
    continued_log = execute(
        [args.athena, "-r", str(restart), "time/nlim=6"],
        roots["restart_continue"])
    require("AMR_HISTORY_REPLAY event=2 " in continued_log,
            "VC replay restart skipped or repeated the remaining authority event")

    cell_authority = roots["cell_record"] / "cell-history.jsonl"
    execute(command(
        args.athena, args.input, "cell_record", "record", cell_authority,
        ["z4c/grid_centering=cell"]), roots["cell_record"])
    cell_header, cell_events = history_records(cell_authority)
    require(cell_header["grid_centering"] == "cell",
            "cell authority was not recorded as cell centered")
    legacy_authority = roots["cell_record"] / "legacy-cell-history.jsonl"
    legacy_cell_authority(cell_authority, legacy_authority)
    rejected = execute(command(
        args.athena, args.input, "bridge_reject", "replay", legacy_authority),
        roots["bridge_reject"], expect_success=False)
    require("history schema mismatch" in rejected,
            "cross-centering replay did not fail closed by default")
    bridge_log = execute(command(
        args.athena, args.input, "bridge_accept", "replay", legacy_authority,
        ["mesh_refinement/amr_history_topology_only_centering_compatibility="
         "cell_to_vertex"]), roots["bridge_accept"])
    require("AMR_HISTORY_CENTERING_COMPATIBILITY" in bridge_log and
            "scope=topology_only native_optimal=false explicit_match=true" in bridge_log,
            "explicit cell-to-vertex topology-only evidence is missing")
    bridge = replay_records(
        roots["bridge_accept"] / "bridge_accept.amr_history_replay.jsonl")
    require_same_authority(cell_events, bridge)

    if args.mpiexec:
        mpi_command = [
            args.mpiexec, args.np_flag, "2",
            *command(args.athena, args.input, "vc_mpi", "replay", authority,
                     ["time/cfl_number=0.07", "time/nlim=6"]),
        ]
        mpi_log = execute(mpi_command, roots["mpi_replay"])
        require("AMR_HISTORY_REPLAY event=2 " in mpi_log,
                "MPI-decomposition VC replay did not finish the authority")
        mpi = replay_records(
            roots["mpi_replay"] / "vc_mpi.amr_history_replay.jsonl")
        require_same_authority(events, mpi)

    print("PASS: native VC AMR history record/replay, 2x cells, restart, "
          "MPI decomposition, and explicit CC-topology bridge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
