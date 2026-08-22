#!/usr/bin/env python3
"""Production-path native-VC restart, rank-change, and rejection regression."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


MARKER = b"<par_end>\n"
COMMAND_TIMEOUT = float(os.environ.get("ATHENA_TEST_COMMAND_TIMEOUT", "45"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], cwd: Path, success: bool, required=()) -> str:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            capture_output=True, check=False,
                            timeout=COMMAND_TIMEOUT)
    output = result.stdout + "\n" + result.stderr
    require((result.returncode == 0) == success,
            f"unexpected exit {result.returncode} for {' '.join(command)}:\n{output}")
    for diagnostic in required:
        require(diagnostic in output,
                f"missing diagnostic {diagnostic!r} for {' '.join(command)}:\n{output}")
    return output


def payload(path: Path) -> bytes:
    data = path.read_bytes()
    require(data.count(MARKER) == 1, f"invalid restart marker in {path}")
    return data.split(MARKER, 1)[1]


def remove_carrier_key(data: bytes, key: bytes) -> bytes:
    header, binary = data.split(MARKER, 1)
    lines = header.splitlines(keepends=True)
    matches = [line for line in lines if line.lstrip().startswith(key + b" ")]
    require(len(matches) == 1, f"expected one carrier line for {key!r}")
    lines.remove(matches[0])
    return b"".join(lines) + MARKER + binary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mpiexec")
    parser.add_argument("--np-flag", default="-n")
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=2)
    parser.add_argument("--active-n1", type=int, default=17)
    parser.add_argument("--active-n2", type=int, default=17)
    parser.add_argument("--active-n3", type=int, default=1)
    args = parser.parse_args()

    active = (args.active_n1, args.active_n2, args.active_n3)
    stored = tuple(value + 8 if value > 1 else 1 for value in active)
    coarse_active = tuple((value - 1) // 2 + 1 if value > 1 else 1
                          for value in active)
    coarse_stored = tuple(value + 8 if value > 1 else 1
                          for value in coarse_active)
    changed_leaves = (1 << args.dimensions) - 1

    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    fresh = root / "fresh"
    fresh.mkdir()
    run([args.athena, "-i", args.input], fresh, True,
        (f"{changed_leaves} MeshBlocks created, {changed_leaves} deleted by AMR",))

    checkpoints = fresh / "rst"
    pre_refine = checkpoints / "z4c_vc_minkowski_dynamic.00000.rst"
    post_refine = checkpoints / "z4c_vc_minkowski_dynamic.00001.rst"
    post_derefine = checkpoints / "z4c_vc_minkowski_dynamic.00002.rst"
    final = checkpoints / "z4c_vc_minkowski_dynamic.00003.rst"
    for checkpoint in (pre_refine, post_refine, post_derefine, final):
        require(checkpoint.is_file(), f"missing restart checkpoint {checkpoint}")
    header = post_refine.read_bytes().split(MARKER, 1)[0]
    records = [
        b"carrier_schema             = 2",
        b"grid_centering             = vertex",
        b"centering_schema           = 1",
        b"nghost                     = 4",
    ]
    for direction in range(3):
        records.extend((
            f"active_n{direction + 1}                  = {active[direction]}".encode(),
            f"stored_n{direction + 1}                  = {stored[direction]}".encode(),
            (f"coarse_active_n{direction + 1}           = "
             f"{coarse_active[direction]}").encode(),
            (f"coarse_stored_n{direction + 1}           = "
             f"{coarse_stored[direction]}").encode(),
        ))
    for record in records:
        require(record in header, f"VC restart omitted {record!r}")

    # Checkpoints immediately before refinement, after refinement, and after
    # derefinement must continue to the identical accepted state.
    for label, checkpoint in (("before_refine", pre_refine),
                              ("after_refine", post_refine),
                              ("after_derefine", post_derefine)):
        resumed = root / label
        resumed.mkdir()
        run([args.athena, "-r", str(checkpoint), "-d", str(resumed)],
            root, True)
        resumed_final = resumed / "rst" / "z4c_vc_minkowski_dynamic.00003.rst"
        require(payload(final) == payload(resumed_final),
                f"{label} continuation changed the final binary payload")

    if args.mpiexec:
        mpi = root / "mpi2"
        mpi.mkdir()
        run([args.mpiexec, args.np_flag, "2", args.athena, "-r",
             str(post_refine), "-d", str(mpi)], root, True,
            ("Number of parallel ranks = 2",
             f"{changed_leaves} MeshBlocks created, {changed_leaves} deleted by AMR"))
        mpi_final = mpi / "rst" / "z4c_vc_minkowski_dynamic.00003.rst"
        require(payload(final) == payload(mpi_final),
                "rank-change continuation changed the global binary payload")

    # Immutable command-line conflicts must fail before mesh/physics construction.
    conflicts = (
        ("z4c/grid_centering=cell", "<z4c>/grid_centering"),
        ("z4c_restart/grid_centering=cell", "<z4c_restart>/grid_centering"),
        ("z4c_restart/centering_schema=2", "<z4c_restart>/centering_schema"),
        ("z4c_restart/stored_n1=24", "<z4c_restart>/stored_n1"),
        ("z4c_restart/carrier_schema=1", "<z4c_restart>/carrier_schema"),
    )
    for override, diagnostic in conflicts:
        output = run([args.athena, "-r", str(post_refine), override],
                     root, False,
                     ("immutable Z4c restart validation failed", diagnostic))
        require("Root grid" not in output and "AssembleZ4cTasks" not in output,
                f"restart conflict {override} reached allocation")

    incomplete = root / "missing_stored_n1.rst"
    incomplete.write_bytes(remove_carrier_key(post_refine.read_bytes(),
                                              b"stored_n1"))
    run([args.athena, "-r", str(incomplete)], root, False,
        ("invalid restart-origin Z4c carrier", "stored_n1"))

    # A real legacy/schema-1 CC checkpoint means cell centering.  It must reject
    # a VC override rather than reinterpret its shorter payload.
    cell = root / "cell"
    cell.mkdir()
    run([args.athena, "-i", args.input, "-d", str(cell),
         "z4c/grid_centering=cell"], root, True)
    cell_restart = cell / "rst" / "z4c_vc_minkowski_dynamic.00001.rst"
    cell_header = cell_restart.read_bytes().split(MARKER, 1)[0]
    require(b"carrier_schema             = 1" in cell_header,
            "CC checkpoint did not retain legacy byte contract")
    require(b"grid_centering             =" not in cell_header,
            "legacy CC carrier unexpectedly serialized centering keys")
    output = run([args.athena, "-r", str(cell_restart),
                  "z4c/grid_centering=vertex"], root, False,
                 ("immutable Z4c restart validation failed",
                  "<z4c>/grid_centering"))
    require("Root grid" not in output, "CC-to-VC restart reached allocation")

    print("PASS: native VC restart, AMR continuation, rank change, and rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
