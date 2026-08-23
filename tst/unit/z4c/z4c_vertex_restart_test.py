#!/usr/bin/env python3
"""Production-path native-VC restart, rank-change, and rejection regression."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
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
    carrier = False
    matches = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(b"<") and stripped.endswith(b">"):
            carrier = stripped == b"<z4c_restart>"
        elif carrier and line.lstrip().startswith(key + b" "):
            matches.append(index)
    require(len(matches) == 1, f"expected one carrier line for {key!r}")
    del lines[matches[0]]
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
    parser.add_argument("--vertex-prolongation-order",
                        choices=("auto", "4", "6", "8"), default="auto")
    parser.add_argument("--spatial-order", type=int, choices=(2, 4, 6))
    args = parser.parse_args()

    input_text = Path(args.input).read_text(encoding="utf-8")
    nghost_match = re.search(r"^nghost\s*=\s*(\d+)\s*$", input_text,
                              flags=re.MULTILINE)
    order_match = re.search(r"^spatial_order\s*=\s*(\d+)\s*$", input_text,
                            flags=re.MULTILINE)
    require(nghost_match is not None and order_match is not None,
            "VC restart fixture must declare nghost and spatial_order")
    nghost = int(nghost_match.group(1))
    spatial_order = (int(order_match.group(1)) if args.spatial_order is None
                     else args.spatial_order)
    if args.spatial_order is not None:
        input_text = (input_text[:order_match.start(1)] + str(spatial_order) +
                      input_text[order_match.end(1):])
    transfer_order = ({2: 4, 4: 6, 6: 8}[spatial_order]
                      if args.vertex_prolongation_order == "auto"
                      else int(args.vertex_prolongation_order))
    coarse_nghost = max(nghost, (nghost - 1) // 2 + transfer_order // 2)
    active = (args.active_n1, args.active_n2, args.active_n3)
    stored = tuple(value + 2 * nghost if value > 1 else 1 for value in active)
    coarse_active = tuple((value - 1) // 2 + 1 if value > 1 else 1
                          for value in active)
    coarse_stored = tuple(value + 2 * coarse_nghost if value > 1 else 1
                          for value in coarse_active)
    changed_leaves = (1 << args.dimensions) - 1

    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    configured_input = root / Path(args.input).name
    if args.vertex_prolongation_order == "auto":
        configured_input.write_text(input_text, encoding="utf-8")
    else:
        marker = "grid_centering = vertex"
        require(input_text.count(marker) == 1,
                "VC restart fixture lacks a unique centering selector")
        configured_input.write_text(
            input_text.replace(
                marker, marker + "\nvertex_prolongation_order = " +
                args.vertex_prolongation_order),
            encoding="utf-8")
    fresh = root / "fresh"
    fresh.mkdir()
    run([args.athena, "-i", str(configured_input)], fresh, True,
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
        b"carrier_schema             = 3",
        b"grid_centering             = vertex",
        b"centering_schema           = 1",
        f"vertex_prolongation_order  = {transfer_order}".encode(),
        f"nghost                     = {nghost}".encode(),
        f"coarse_nghost              = {coarse_nghost}".encode(),
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

    # Schema 2 is the pre-selector native-VC carrier.  It is interpreted as the
    # historical `auto` contract and upgraded to schema 3 on the next output.
    if args.vertex_prolongation_order == "auto":
        legacy_data = remove_carrier_key(post_refine.read_bytes(),
                                         b"vertex_prolongation_order")
        old_schema = b"carrier_schema             = 3"
        require(legacy_data.count(old_schema) == 1,
                "current VC checkpoint lacks a unique carrier schema")
        legacy_data = legacy_data.replace(
            old_schema, b"carrier_schema             = 2", 1)
        legacy_restart = root / "legacy_vc_schema2.rst"
        legacy_restart.write_bytes(legacy_data)
        legacy = root / "legacy_vc_schema2"
        legacy.mkdir()
        run([args.athena, "-r", str(legacy_restart), "-d", str(legacy)],
            root, True)
        upgraded = legacy / "rst" / "z4c_vc_minkowski_dynamic.00003.rst"
        upgraded_header = upgraded.read_bytes().split(MARKER, 1)[0]
        require(b"carrier_schema             = 3" in upgraded_header and
                f"vertex_prolongation_order  = {transfer_order}".encode() in
                upgraded_header,
                "legacy VC schema 2 did not upgrade to the explicit transfer carrier")

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
    conflicting_transfer = "6" if transfer_order == 4 else "4"
    conflicts = (
        ("z4c/grid_centering=cell", "<z4c>/grid_centering"),
        (f"z4c/vertex_prolongation_order={conflicting_transfer}",
         "<z4c>/vertex_prolongation_order"),
        ("z4c_restart/grid_centering=cell", "<z4c_restart>/grid_centering"),
        (f"z4c_restart/vertex_prolongation_order={conflicting_transfer}",
         "<z4c_restart>/vertex_prolongation_order"),
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
    incomplete_transfer = root / "missing_vertex_prolongation_order.rst"
    incomplete_transfer.write_bytes(remove_carrier_key(
        post_refine.read_bytes(), b"vertex_prolongation_order"))
    run([args.athena, "-r", str(incomplete_transfer)], root, False,
        ("invalid restart-origin Z4c carrier", "vertex_prolongation_order"))

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
