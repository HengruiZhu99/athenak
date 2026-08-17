#!/usr/bin/env python3
"""Host runtime, rank-decomposition, and fail-closed tests for the AMR ledger."""

from __future__ import annotations

import argparse
import hashlib
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
            mpiexec: str | None, enabled: bool = True,
            overrides: list[str] | None = None) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"z4c-amr-runtime-r{ranks}."))
    launch = root / "launch"
    run_dir = root / "run"
    launch.mkdir()
    run_dir.mkdir()
    command = [str(executable), "-i", str(input_path), "-d", str(run_dir)]
    if not enabled:
        command.append("z4c/amr_jump_diagnostic=false")
    if overrides:
        command.extend(overrides)
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


def check_hierarchy_control(executable: Path, input_path: Path) -> None:
    for mode in ("freeze_after_target", "buffered_freeze_after_target"):
        run_dir = execute(
            executable, input_path, 1, None,
            overrides=[f"z4c/amr_jump_hierarchy_control={mode}"],
        )
        rank = run_dir / "z4c_amr_jump" / "rank0000"
        schema = json.loads((rank / "schema.json").read_text(encoding="utf-8"))
        if schema.get("hierarchy_control") != mode:
            raise TestFailure(f"schema did not bind hierarchy control {mode}")
        with (rank / "hierarchy_control.jsonl").open(encoding="utf-8") as stream:
            controls = [json.loads(line) for line in stream if line.strip()]
        if not controls:
            raise TestFailure(f"{mode} emitted no hierarchy-control evidence")
        if mode == "buffered_freeze_after_target":
            first = controls[0]
            if int(first["cycle"]) != 1 or first["target_seen"] or \
                    int(first["buffered_refine_added"]) <= 0:
                raise TestFailure("buffered target did not expand the exact event")
        frozen = [row for row in controls if row["target_seen"]]
        if not frozen:
            raise TestFailure(f"{mode} never entered frozen state")
        snapshots = sorted((rank / "accepted_topologies").glob("*.csv"))
        if [path.stem for path in snapshots] != ["c00000001"]:
            raise TestFailure(f"{mode} changed topology after the target: {snapshots}")
        with (rank / "rk_stage_exposure.jsonl").open(encoding="utf-8") as stream:
            exposure = [json.loads(line) for line in stream if line.strip()]
        if len(exposure) != 8 or any(int(row["stage"]) not in (1, 2, 3, 4)
                                     for row in exposure):
            raise TestFailure(f"{mode} exposure is not exactly RK-stage accounting")
        cumulative = [int(row["cumulative_X_CF"]) for row in exposure]
        expected = []
        total = 0
        for row in exposure:
            total += int(row["coarse_fine_leaf_face_incidents"])
            expected.append(total)
        if cumulative != expected:
            raise TestFailure(f"{mode} cumulative X_CF is invalid")


def check_transfer_control(executable: Path, input_path: Path) -> None:
    default_run = execute(executable, input_path, 1, None)
    default_schema = json.loads(
        (default_run / "z4c_amr_jump" / "rank0000" / "schema.json").read_text(
            encoding="utf-8"))
    if default_schema.get("amr_transfer") != "high_order":
        raise TestFailure("Z4c AMR transfer did not default to high_order")

    limited_run = execute(
        executable, input_path, 1, None,
        overrides=["z4c/amr_transfer=limited_o2"],
    )
    limited_schema = json.loads(
        (limited_run / "z4c_amr_jump" / "rank0000" / "schema.json").read_text(
            encoding="utf-8"))
    if limited_schema.get("amr_transfer") != "limited_o2":
        raise TestFailure("limited_o2 transfer was not authenticated in the schema")
    log = limited_run.parent / "run.log"
    if "transfer=limited_o2" not in log.read_text(encoding="utf-8"):
        raise TestFailure("limited_o2 transfer emitted no runtime provenance")

    temporary = Path(tempfile.mkdtemp(prefix="z4c-amr-transfer-invalid."))
    command = [str(executable), "-i", str(input_path), "-d", str(temporary),
               "z4c/amr_transfer=unknown"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode == 0 or "unknown <z4c>/amr_transfer=unknown" not in result.stdout:
        raise TestFailure("invalid Z4c AMR transfer did not fail closed")


def check_zero_pde_stop(executable: Path, input_path: Path,
                        transfer: str = "high_order") -> Path:
    run_dir = execute(
        executable, input_path, 1, None,
        overrides=["z4c/amr_jump_post_cycles=0", "z4c/amr_transfer=high_order",
                   "z4c/amr_jump_target_cycle=1",
                   f"z4c/amr_jump_target_transfer={transfer}"],
    )
    rank = run_dir / "z4c_amr_jump" / "rank0000"
    schema = json.loads((rank / "schema.json").read_text(encoding="utf-8"))
    if schema.get("amr_transfer") != transfer or \
            schema.get("pre_target_amr_transfer") != "high_order" or \
            schema.get("target_transaction_only_transfer") is not True:
        raise TestFailure(f"zero-PDE target-only transfer provenance is invalid: {schema}")
    event = rank / "event_c00000001_l0_to_l1"
    lifecycle = json.loads(
        (event / "target_transfer_lifecycle.json").read_text(encoding="utf-8"))
    if lifecycle.get("target_transfer") != transfer or \
            lifecycle.get("restored_transfer") != "high_order" or \
            lifecycle.get("restored_after_t5") is not True:
        raise TestFailure(f"target transfer was not restored after T5: {lifecycle}")
    with (rank / "post_event_cycles.jsonl").open(encoding="utf-8") as stream:
        post = [json.loads(line) for line in stream if line.strip()]
    if [int(row["cycle"]) for row in post] != [1]:
        raise TestFailure(f"zero-PDE probe did not stop at target cycle: {post}")
    exposure_path = rank / "rk_stage_exposure.jsonl"
    if exposure_path.exists() and exposure_path.read_text(encoding="utf-8").strip():
        raise TestFailure("zero-PDE probe advanced into a post-event RK stage")
    snapshots = sorted((rank / "accepted_topologies").glob("*.csv"))
    if [path.stem for path in snapshots] != ["c00000001"]:
        raise TestFailure(f"zero-PDE probe accepted an unexpected later cycle: {snapshots}")
    log = run_dir.parent / "run.log"
    if "after T5 and before the next RHS" not in log.read_text(encoding="utf-8"):
        raise TestFailure("zero-PDE probe emitted no explicit stop-point evidence")
    return run_dir


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_synthetic_provenance(run_dir: Path, transfer: str) -> Path:
    diagnostic = run_dir / "z4c_amr_jump"
    manifest = run_dir / "diagnostic.SHA256SUMS"
    files = sorted(path for path in diagnostic.rglob("*") if path.is_file())
    manifest.write_text("".join(
        f"{file_sha256(path)}  {path.relative_to(run_dir)}\n" for path in files),
        encoding="utf-8")
    record = {
        "schema": "athenak_z4c_amr_zero_pde_provenance_v1",
        "qualification_claim": False,
        "amr_transfer": transfer,
        "pre_target_amr_transfer": "high_order",
        "target_transaction_only_transfer": True,
        "post_cycles": 0,
        "rank_count": 1,
        "source_commit": "synthetic-source",
        "source_tree": "synthetic-tree",
        "executable_sha256": "synthetic-executable",
        "input_sha256": "synthetic-input",
        "restart_sha256": "synthetic-restart",
        "node": "synthetic-node",
        "gpu_model": "synthetic-device",
        "hardware_binding_sha256": "synthetic-binding",
        "command_sha256": f"synthetic-command-{transfer}",
        "diagnostic_manifest_sha256": file_sha256(manifest),
    }
    path = run_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def compare_zero_pde(compare_analyzer: Path, high: Path, limited: Path) -> None:
    high_provenance = write_synthetic_provenance(high, "high_order")
    limited_provenance = write_synthetic_provenance(limited, "limited_o2")
    command = ["python3", str(compare_analyzer), "analyze",
               "--high-root", str(high / "z4c_amr_jump"),
               "--limited-root", str(limited / "z4c_amr_jump"),
               "--high-provenance", str(high_provenance),
               "--limited-provenance", str(limited_provenance),
               "--output", str(high / "comparison"), "--ranks", "1",
               "--level-before", "0", "--level-after", "1",
               "--expected-cycle", "1"]
    run(command, high)


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
    parser.add_argument("--compare-analyzer", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    args = parser.parse_args()

    for path in [args.serial_executable, args.mpi_executable, args.input,
                 args.analyzer, args.compare_analyzer]:
        if not path.resolve().is_file():
            raise TestFailure(f"missing test input {path}")
    off = execute(args.serial_executable.resolve(), args.input.resolve(), 1,
                  None, enabled=False)
    if (off / "z4c_amr_jump").exists():
        raise TestFailure("default-off diagnostic path exists")
    check_hierarchy_control(args.serial_executable.resolve(), args.input.resolve())
    check_transfer_control(args.serial_executable.resolve(), args.input.resolve())
    high_zero = check_zero_pde_stop(
        args.serial_executable.resolve(), args.input.resolve(), "high_order")
    limited_zero = check_zero_pde_stop(
        args.serial_executable.resolve(), args.input.resolve(), "limited_o2")
    compare_zero_pde(args.compare_analyzer.resolve(), high_zero, limited_zero)

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
