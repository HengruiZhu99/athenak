#!/usr/bin/env python3
"""Bounded production-path AMR history record/replay integration test."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command, cwd, log_name="run.log"):
    result = subprocess.run(command, cwd=cwd, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    (cwd / log_name).write_text(result.stdout, encoding="utf-8")
    (cwd / (log_name + ".err")).write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0,
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout


def run_fails(command, cwd, expected, log_name="failed.log"):
    result = subprocess.run(command, cwd=cwd, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    (cwd / log_name).write_text(result.stdout, encoding="utf-8")
    (cwd / (log_name + ".err")).write_text(result.stderr, encoding="utf-8")
    require(result.returncode != 0, f"command unexpectedly passed: {' '.join(command)}")
    require(expected in result.stdout + result.stderr,
            f"expected failure text missing: {expected}\n{result.stdout}\n{result.stderr}")


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def binary_payload(path):
    raw = path.read_bytes()
    marker = b"<par_end>\n"
    require(marker in raw, f"binary output lacks parameter marker: {path}")
    return raw.split(marker, 1)[1]


def outputs(root, basename):
    files = sorted((root / "bin").glob(f"{basename}.state.*.bin"))
    require(files, f"no binary outputs for {basename}")
    return files


def compare_payloads(a_root, a_name, b_root, b_name):
    a = outputs(a_root, a_name)
    b = outputs(b_root, b_name)
    require(len(a) == len(b), f"output count mismatch {len(a)} != {len(b)}")
    for left, right in zip(a, b):
        require(binary_payload(left) == binary_payload(right),
                f"numerical payload mismatch: {left.name} vs {right.name}")


def compare_final_payload(a_root, a_name, b_root, b_name):
    left = outputs(a_root, a_name)[-1]
    right = outputs(b_root, b_name)[-1]
    require(binary_payload(left) == binary_payload(right),
            f"final numerical payload mismatch: {left.name} vs {right.name}")


def load_history(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) >= 2 and all(lines), "history is empty or truncated")
    records = [json.loads(line) for line in lines]
    require(records[0]["type"] == "header", "history header missing")
    require(all(item["type"] == "event" for item in records[1:]), "non-event record")
    return records[0], records[1:]


def event_lines(log):
    return [line for line in log.splitlines() if line.startswith("AMR_HISTORY_REPLAY")]


def require_shadow_diagnostics(log, count):
    lines = event_lines(log)
    require(len(lines) == count, "unexpected replay event count")
    require(all(" shadow_refine=" in line and " shadow_derefine=" in line
                for line in lines), "replay criterion shadow diagnostics missing")


def replace(text, old, new):
    require(old in text, f"input mutation source not found: {old}")
    return text.replace(old, new, 1)


def command(exe, input_path, mode, history, basename, extra=None):
    cmd = [str(exe), "-i", str(input_path),
           f"mesh_refinement/amr_history_mode={mode}",
           f"mesh_refinement/amr_history_file={history}",
           f"job/basename={basename}"]
    return cmd + list(extra or [])


def serial_suite(args, work):
    base_text = args.input.read_text(encoding="utf-8")
    absent_text = "\n".join(line for line in base_text.splitlines()
                             if not line.startswith("amr_history_mode")
                             and not line.startswith("amr_history_file")) + "\n"
    absent_input = work / "absent.athinput"
    absent_input.write_text(absent_text, encoding="utf-8")

    roots = {name: work / name for name in
             ("absent", "off", "record", "replay", "cfl", "high", "record_restart",
              "replay_prefix", "replay_restart", "carrier_mutation", "fresh_injection",
              "existing_record")}
    for root in roots.values():
        clean_dir(root)

    run([str(args.athena), "-i", str(absent_input), "job/basename=absent"], roots["absent"])
    run(command(args.athena, args.input, "off", "unused.jsonl", "off"), roots["off"])
    history = roots["record"] / "history.jsonl"
    record_log = run(command(args.athena, args.input, "record", history, "record"),
                     roots["record"])
    header, events = load_history(history)
    require(len(events) >= 3, "fixture did not create at least two topology events")
    require([event["event"] for event in events] == list(range(len(events))),
            "event indices are not contiguous")
    require(len({event["time_hex"] for event in events}) == len(events),
            "event times are not unique")
    require("AMR_HISTORY_RECORD" in record_log, "record diagnostics missing")

    replay_log = run(command(args.athena, args.input, "replay", history, "replay"),
                     roots["replay"])
    require_shadow_diagnostics(replay_log, len(events) - 1)
    compare_payloads(roots["absent"], "absent", roots["off"], "off")
    compare_payloads(roots["off"], "off", roots["record"], "record")
    compare_payloads(roots["record"], "record", roots["replay"], "replay")

    cfl_log = run(command(args.athena, args.input, "replay", history, "cfl",
                          ["time/cfl_number=0.07", "time/nlim=6"]), roots["cfl"])
    require("AMR_HISTORY_TIMESTEP_CLIP" in cfl_log,
            "different-CFL replay did not report coordinate-time clipping")
    for event in events[1:]:
        require(any(f"event={event['event']} " in line and
                    f"time_hex={event['time_hex']} " in line for line in event_lines(cfl_log)),
                f"different-CFL replay missed exact event {event['event']}")

    high_log = run(command(
        args.athena, args.input, "replay", history, "high",
        ["mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=16",
         "meshblock/nx1=16", "meshblock/nx2=16", "meshblock/nx3=16",
         "time/nlim=5"]), roots["high"])
    require(len(event_lines(high_log)) == len(events) - 1, "2x replay event count mismatch")
    high_ledger = roots["high"] / "high.amr_history_replay.jsonl"
    require(high_ledger.is_file(), "2x replay ledger missing")
    require(header["root_blocks"] == [2, 2, 1] and
            header["cells_per_meshblock"] == [8, 8, 8], "unexpected record geometry")

    # Record restart from the initial post-event checkpoint, then require the same
    # canonical history and final numerical payload as uninterrupted record mode.
    prefix_root = roots["record_restart"]
    prefix_history = prefix_root / "history.jsonl"
    run(command(args.athena, args.input, "record", prefix_history, "record_restart",
                ["time/nlim=0"]), prefix_root, "prefix.log")
    restart0 = next((prefix_root / "rst").glob("record_restart.00000.rst"))
    run([str(args.athena), "-r", str(restart0), "time/nlim=4"], prefix_root,
        "continued.log")
    require(prefix_history.read_bytes() == history.read_bytes(),
            "record restart did not reproduce canonical history bytes")
    compare_payloads(roots["record"], "record", prefix_root, "record_restart")

    # Replay restart after the second accepted event. First stop at cycle two, then
    # continue from its post-event restart and compare the final payload.
    replay_prefix = roots["replay_prefix"]
    run(command(args.athena, args.input, "replay", history, "replay_restart",
                ["time/nlim=2"]), replay_prefix, "prefix.log")
    restart2 = sorted((replay_prefix / "rst").glob("replay_restart.*.rst"))[-1]
    replay_cont = roots["replay_restart"]
    run([str(args.athena), "-r", str(restart2), "time/nlim=4",
         "job/basename=replay_restart"], replay_cont, "continued.log")
    compare_final_payload(roots["record"], "record", replay_cont, "replay_restart")

    # Restart carriers are internal and immutable: command-line injection or
    # mutation must fail before any evolution begins.
    run_fails([str(args.athena), "-r", str(restart2),
               "amr_history_restart/next_event=0"], roots["carrier_mutation"],
              "immutable <amr_history_restart> carrier was injected or modified")
    injected_input = roots["fresh_injection"] / "injected.athinput"
    injected_input.write_text(base_text + "\n<amr_history_restart>\nschema = 1\n",
                              encoding="utf-8")
    run_fails([str(args.athena), "-i", str(injected_input)], roots["fresh_injection"],
              "Invalid <block_name> in input file")

    # A fresh record never overwrites an existing authority history.
    occupied = roots["existing_record"] / "history.jsonl"
    occupied.write_text("owned sentinel\n", encoding="utf-8")
    run_fails(command(args.athena, args.input, "record", occupied, "existing_record"),
              roots["existing_record"], "fresh record output already exists")

    return history, len(events)


def mpi_suite(args, work):
    require(args.mpiexec and args.np_flag, "MPI suite needs launcher and process flag")
    record_root = work / "mpi_record"
    replay_root = work / "mpi_replay"
    clean_dir(record_root)
    clean_dir(replay_root)
    history = record_root / "history.jsonl"
    run(command(args.athena, args.input, "record", history, "mpi_record"), record_root)
    _, events = load_history(history)
    cmd = [str(args.mpiexec), args.np_flag, "2"] + command(
        args.athena, args.input, "replay", history, "mpi_replay")
    log = run(cmd, replay_root)
    require_shadow_diagnostics(log, len(events) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mpi-only", action="store_true")
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--np-flag")
    args = parser.parse_args()
    args.athena = args.athena.resolve()
    args.input = args.input.resolve()
    clean_dir(args.work_dir)
    if args.mpi_only:
        mpi_suite(args, args.work_dir)
    else:
        serial_suite(args, args.work_dir)
    print("AMR_HISTORY_INTEGRATION_TEST_PASS")


if __name__ == "__main__":
    main()
