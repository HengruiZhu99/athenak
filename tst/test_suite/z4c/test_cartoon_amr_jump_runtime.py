#!/usr/bin/env python3
"""Host runtime, rank-decomposition, and fail-closed tests for the AMR ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


class TestFailure(RuntimeError):
    pass


def run(command: list[str], cwd: Path, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode != expect:
        raise TestFailure(f"command returned {result.returncode}, expected {expect}: "
                          f"{' '.join(command)}\n{result.stdout}")
    return result


def execute(executable: Path, input_path: Path, ranks: int,
            mpiexec: str | None, enabled: bool = True) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"z4c-amr-runtime-r{ranks}."))
    launch = root / "launch"
    run_dir = root / "run"
    launch.mkdir()
    run_dir.mkdir()
    command = [str(executable), "-i", str(input_path), "-d", str(run_dir)]
    if not enabled:
        command.append("z4c/amr_jump_diagnostic=false")
    if ranks > 1:
        if not mpiexec:
            raise TestFailure("MPI test requested without --mpiexec")
        command = [mpiexec, "-n", str(ranks)] + command
    result = run(command, launch)
    (root / "run.log").write_text(result.stdout, encoding="utf-8")
    diagnostic = run_dir / "z4c_amr_jump"
    if enabled and not diagnostic.is_dir():
        raise TestFailure("enabled run produced no diagnostic root")
    if not enabled and diagnostic.exists():
        raise TestFailure("default-off run produced diagnostic files")
    if (launch / "z4c_amr_jump").exists():
        raise TestFailure("diagnostic escaped the requested -d run directory")
    return run_dir


def aggregate_post(root: Path, ranks: int) -> list[dict[str, float]]:
    by_cycle: dict[int, dict[str, float]] = {}
    keys = ["active_cells", "coordinate_ring_volume", "proper_volume",
            "C_norm2", "H_norm2", "M_norm2", "Z_norm2"]
    for rank in range(ranks):
        path = root / "z4c_amr_jump" / f"rank{rank:04d}" / "post_event_cycles.jsonl"
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if [int(row["cycle"]) for row in rows] != [1, 2, 3]:
            raise TestFailure(f"wrong post-event window in {path}")
        for row in rows:
            cycle = int(row["cycle"])
            target = by_cycle.setdefault(cycle, {key: 0.0 for key in keys})
            for key in keys:
                target[key] += float(row[key])
    return [by_cycle[cycle] for cycle in sorted(by_cycle)]


def analyze(analyzer: Path, diagnostic: Path, output: Path, ranks: int,
            expect_success: bool) -> None:
    command = ["python3", str(analyzer), "analyze", "--diagnostic-root",
               str(diagnostic), "--output", str(output), "--ranks", str(ranks),
               "--level-before", "0", "--level-after", "1", "--post-cycles", "2",
               "--expected-cycle", "1", "--expected-old-nmb", "8",
               "--expected-new-nmb", "14"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if (result.returncode == 0) != expect_success:
        raise TestFailure(f"analyzer mutation expectation failed: {' '.join(command)}\n"
                          f"{result.stdout}")


def mutate_and_reject(analyzer: Path, source: Path, ranks: int,
                      name: str, mutation) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f"z4c-amr-mutation-{name}."))
    target = temporary / "diagnostic"
    shutil.copytree(source, target)
    mutation(target)
    analyze(analyzer, target, temporary / "analysis", ranks, False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-executable", type=Path, required=True)
    parser.add_argument("--mpi-executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    args = parser.parse_args()

    for path in [args.serial_executable, args.mpi_executable, args.input, args.analyzer]:
        if not path.resolve().is_file():
            raise TestFailure(f"missing test input {path}")
    off = execute(args.serial_executable.resolve(), args.input.resolve(), 1,
                  None, enabled=False)
    if (off / "z4c_amr_jump").exists():
        raise TestFailure("default-off diagnostic path exists")

    runs: dict[int, Path] = {}
    for ranks in (1, 2, 4):
        executable = args.serial_executable if ranks == 1 else args.mpi_executable
        runs[ranks] = execute(executable.resolve(), args.input.resolve(), ranks,
                              args.mpiexec if ranks > 1 else None)
        analyze(args.analyzer.resolve(), runs[ranks] / "z4c_amr_jump",
                runs[ranks] / "analysis", ranks, True)
    reference = aggregate_post(runs[1], 1)
    for ranks in (2, 4):
        candidate = aggregate_post(runs[ranks], ranks)
        for reference_row, candidate_row in zip(reference, candidate):
            for key in reference_row:
                if not math.isclose(reference_row[key], candidate_row[key],
                                    rel_tol=2.0e-12, abs_tol=2.0e-12):
                    raise TestFailure(f"rank aggregation mismatch r{ranks} {key}: "
                                      f"{reference_row[key]} != {candidate_row[key]}")

    source = runs[4] / "z4c_amr_jump"
    def remove_event(root: Path) -> None:
        event = next((root / "rank0000").glob("event_c*_l0_to_l1"))
        event.rename(event.with_name(event.name + ".missing"))
    mutate_and_reject(args.analyzer, source, 4, "missing-event", remove_event)

    def remove_writer(root: Path) -> None:
        event = next((root / "rank0000").glob("event_c*_l0_to_l1"))
        shutil.rmtree(event / "t3_03_SAME_LEVEL_COARSE_REFRESH")
    mutate_and_reject(args.analyzer, source, 4, "missing-writer", remove_writer)

    def duplicate_owner(root: Path) -> None:
        event = next((root / "rank0000").glob("event_c*_l0_to_l1"))
        topology = event / "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION" / "topology.csv"
        lines = topology.read_text(encoding="utf-8").splitlines()
        topology.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    mutate_and_reject(args.analyzer, source, 4, "duplicate-owner", duplicate_owner)

    def nonfinite(root: Path) -> None:
        event = next((root / "rank0000").glob("event_c*_l0_to_l1"))
        binary = event / "t2_00_REFINE_OR_DEREFINE_TRANSFER" / "u0.bin"
        with binary.open("r+b") as stream:
            stream.seek(0)
            stream.write(struct.pack("<d", float("nan")))
    mutate_and_reject(args.analyzer, source, 4, "nonfinite", nonfinite)

    def accounting_mismatch(root: Path) -> None:
        event0 = next((root / "rank0000").glob("event_c*_l0_to_l1"))
        with (event0 / "t1_topology_proposal.csv").open(encoding="utf-8") as stream:
            import csv
            refined = next(row for row in csv.DictReader(stream)
                           if int(row["new_level"]) == int(row["old_level"]) + 1)
        gid, rank = int(refined["new_gid"]), int(refined["new_rank"])
        event = next((root / f"rank{rank:04d}").glob("event_c*_l0_to_l1"))
        phase = event / "t2_00_REFINE_OR_DEREFINE_TRANSFER"
        with (phase / "topology.csv").open(encoding="utf-8") as stream:
            row = next(row for row in csv.DictReader(stream) if int(row["gid"]) == gid)
        metadata = json.loads((phase / "phase.json").read_text(encoding="utf-8"))
        shape = metadata["u0_shape"]
        local_m = int(row["local_m"])
        bounds = metadata["active_bounds"]
        flat = (((((local_m * shape[1]) * shape[2] + bounds["ks"]) * shape[3]
                  + bounds["js"]) * shape[4]) + bounds["is"])
        binary = phase / "u0.bin"
        with binary.open("r+b") as stream:
            stream.seek(flat * 8)
            value = struct.unpack("<d", stream.read(8))[0]
            stream.seek(flat * 8)
            stream.write(struct.pack("<d", value + 1.0))
    mutate_and_reject(args.analyzer, source, 4, "transfer-accounting",
                      accounting_mismatch)

    def premature(root: Path) -> None:
        path = root / "rank0000" / "post_event_cycles.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    mutate_and_reject(args.analyzer, source, 4, "premature", premature)

    print("Cartoon AMR-jump runtime/rank/mutation tests: PASS")


if __name__ == "__main__":
    main()
