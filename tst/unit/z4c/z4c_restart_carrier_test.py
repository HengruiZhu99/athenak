#!/usr/bin/env python3
"""Production-path roundtrip and override tests for the Z4c restart carrier."""

import argparse
import io
import re
import shutil
import struct
import subprocess
import tarfile
from pathlib import Path


def run(command, cwd, expected_success, required=()):
    result = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + "\n" + result.stderr
    if expected_success != (result.returncode == 0):
        raise RuntimeError(
            f"unexpected exit {result.returncode} for {' '.join(map(str, command))}:\n"
            f"{output}"
        )
    for text in required:
        if text not in output:
            raise RuntimeError(f"missing diagnostic {text!r}:\n{output}")
    return output


def rejected(athena, restart, cwd, override, block, key, stored, requested):
    output = run(
        [athena, "-r", restart, override],
        cwd,
        False,
        (
            "immutable Z4c restart validation failed",
            f"<{block}>/{key}",
            f"stored='{stored}'",
            f"requested='{requested}'",
        ),
    )
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError(f"override {override} reached mesh/physics construction:\n{output}")


def replace_once(data, old, new):
    if len(old) != len(new):
        raise RuntimeError(f"replacement changes restart length: {old!r} -> {new!r}")
    if data.count(old) != 1:
        raise RuntimeError(f"expected one restart occurrence of {old!r}")
    return data.replace(old, new, 1)


def replace_value(data, key, old, new, require_same_length=True):
    if require_same_length and len(old) != len(new):
        raise RuntimeError(f"value replacement changes length for {key}")
    pattern = re.compile(rb"(" + re.escape(key.encode()) + rb"\s*=\s*)" +
                         re.escape(old.encode()) + rb"(\s)")
    replaced, count = pattern.subn(rb"\g<1>" + new.encode() + rb"\g<2>", data, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace {key}={old} in restart")
    return replaced


def replace_block_value(data, block, key, new):
    marker = b"<par_end>\n"
    text_end = data.index(marker) + len(marker)
    prefix = data[:text_end]
    suffix = data[text_end:]
    block_start = prefix.index(f"<{block}>\n".encode())
    block_end = prefix.find(b"#-------------------------", block_start)
    if block_end < 0:
        block_end = len(prefix)
    section = prefix[block_start:block_end]
    pattern = re.compile(rb"(" + re.escape(key.encode()) + rb"\s*=\s*)\S+")
    replaced, count = pattern.subn(rb"\g<1>" + new.encode(), section, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace <{block}>/{key}")
    return prefix[:block_start] + replaced + prefix[block_end:] + suffix


def remove_legacy_fastflow_time_key(data):
    """Remove the sole canonical schema-2-only key from a legacy fixture."""
    marker = b"<par_end>\n"
    text_end = data.index(marker) + len(marker)
    prefix = data[:text_end]
    suffix = data[text_end:]
    block_marker = b"<z4c_restart>\n"
    if prefix.count(block_marker) != 1:
        raise RuntimeError("expected exactly one <z4c_restart> block")
    block_start = prefix.index(block_marker)
    block_end = prefix.find(b"#-------------------------", block_start)
    if block_end < 0:
        block_end = len(prefix)
    section = prefix[block_start:block_end]

    key = b"fastflow_time_first_found"
    key_line_pattern = re.compile(
        rb"^[ \t]*" + key + rb"\b[^\r\n]*(?:\r?\n|\Z)", re.MULTILINE)
    key_lines = key_line_pattern.findall(section)
    if len(key_lines) != 1:
        raise RuntimeError(
            "expected exactly one legacy FastFlow time key line; "
            f"found {len(key_lines)}"
        )
    valid_line_pattern = re.compile(
        rb"^[ \t]*" + key +
        rb"[ \t]*=[ \t]*-1[ \t]*(?:#[ \t]*Updated[ \t]+during[ \t]+run"
        rb"[ \t]+time)?[ \t]*(?:\r?\n|\Z)"
    )
    if valid_line_pattern.fullmatch(key_lines[0]) is None:
        raise RuntimeError(
            "legacy FastFlow time key must have value -1 and only the canonical "
            "'Updated during run time' comment"
        )
    changed = section.replace(key_lines[0], b"", 1)
    if key_line_pattern.search(changed) is not None:
        raise RuntimeError("legacy FastFlow time key removal left a duplicate key")
    return prefix[:block_start] + changed + prefix[block_end:] + suffix


def test_legacy_fastflow_time_key_removal():
    binary = b"\0binary fastflow_time_first_found = 7\0"

    def fixture(*lines):
        return (b"<z4c_restart>\nfastflow_schema = 1\n" + b"".join(lines) +
                b"fastflow_converged = 0\n"
                b"#------------------------- PAR_DUMP\n<par_end>\n" + binary)

    for line in (
        b"fastflow_time_first_found = -1\n",
        b"\tfastflow_time_first_found\t = \t-1  "
        b"#  Updated  during run time \t\r\n",
    ):
        changed = remove_legacy_fastflow_time_key(fixture(line))
        header, payload = changed.split(b"<par_end>\n", 1)
        if b"fastflow_time_first_found" in header or payload != binary:
            raise RuntimeError("legacy FastFlow time fixture removal was not isolated")

    invalid = (
        fixture(),
        fixture(b"fastflow_time_first_found = -1\n",
                b"fastflow_time_first_found = -1\n"),
        fixture(b"fastflow_time_first_found = 0\n"),
        fixture(b"fastflow_time_first_found -1\n"),
        fixture(b"fastflow_time_first_found = -1 # unexpected comment\n"),
    )
    for candidate in invalid:
        try:
            remove_legacy_fastflow_time_key(candidate)
        except RuntimeError:
            continue
        raise RuntimeError("malformed legacy FastFlow time fixture was accepted")


def assert_restart_binary_payload_unchanged(current, legacy):
    marker = b"<par_end>\n"
    current_payload = current[current.index(marker) + len(marker):]
    legacy_payload = legacy[legacy.index(marker) + len(marker):]
    # Restart writes two ints, a nine-Real RegionSize, then the root RegionIndcs.
    # Its first ten ints are defined; its nine unused coarse-index ints must be zero.
    root_indcs = 2 * struct.calcsize("=i") + 9 * struct.calcsize("=d")
    root_coarse = root_indcs + 10 * struct.calcsize("=i")
    root_coarse_end = root_coarse + 9 * struct.calcsize("=i")
    for name, payload in (("current", current_payload), ("legacy", legacy_payload)):
        if len(payload) < root_coarse_end:
            raise RuntimeError(f"{name} restart binary header is truncated")
        coarse = struct.unpack_from("=9i", payload, root_coarse)
        if any(coarse):
            raise RuntimeError(
                f"{name} restart serialized nonzero root coarse indices: {coarse}"
            )
    if len(current_payload) != len(legacy_payload):
        raise RuntimeError(
            "restart carrier changed the post-<par_end> binary payload length: "
            f"{len(current_payload)} != {len(legacy_payload)}"
        )
    if current_payload != legacy_payload:
        first = next(i for i, values in enumerate(zip(current_payload, legacy_payload))
                     if values[0] != values[1])
        raise RuntimeError(
            "restart carrier changed the post-<par_end> binary payload at "
            f"offset {first}"
        )


def test_restart_binary_payload_contract():
    marker = b"<par_end>\n"
    payload = bytearray(256)
    current = b"<current>\n" + marker + payload
    legacy = b"<legacy>\n" + marker + payload
    assert_restart_binary_payload_unchanged(current, legacy)

    root_coarse = 2 * struct.calcsize("=i") + 9 * struct.calcsize("=d") + \
        10 * struct.calcsize("=i")
    nonzero_coarse = bytearray(payload)
    struct.pack_into("=i", nonzero_coarse, root_coarse, 1)
    active_mutation = bytearray(payload)
    struct.pack_into("=i", active_mutation, root_coarse - struct.calcsize("=i"), 1)
    for candidate in (nonzero_coarse, active_mutation):
        try:
            assert_restart_binary_payload_unchanged(
                current, b"<legacy>\n" + marker + candidate)
        except RuntimeError:
            continue
        raise RuntimeError("restart binary payload mutation was accepted")


def mutate_binary_dimension(data, target, expected, replacement):
    marker = b"<par_end>\n"
    header_start = data.index(marker) + len(marker)
    # Default Athena Real is double. The restart header writes two ints, a nine-Real
    # RegionSize, then two 19-int RegionIndcs objects without padding between writes.
    global_indcs = header_start + 2 * 4 + 9 * 8
    meshblock_indcs = global_indcs + 19 * 4
    offsets = {
        "mesh/nx1": global_indcs + 4,
        "mesh/nx2": global_indcs + 8,
        "mesh/nx3": global_indcs + 12,
        "meshblock/nx1": meshblock_indcs + 4,
        "meshblock/nx2": meshblock_indcs + 8,
        "meshblock/nx3": meshblock_indcs + 12,
    }
    offset = offsets[target]
    actual = struct.unpack_from("=i", data, offset)[0]
    if actual != expected:
        raise RuntimeError(
            f"unexpected binary {target}={actual}; expected {expected} at offset {offset}"
        )
    changed = bytearray(data)
    struct.pack_into("=i", changed, offset, replacement)
    return bytes(changed)


def run_build(command, cwd, timeout):
    result = subprocess.run(
        [str(item) for item in command], cwd=cwd, check=False,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"legacy fixture build failed for {' '.join(map(str, command))}:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def build_legacy_athena(source_dir, work_dir, base_commit):
    legacy_source = work_dir / "legacy_base_source"
    legacy_build = work_dir / "legacy_base_build"
    legacy_source.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(source_dir), "archive", "--format=tar", base_commit],
        check=False, capture_output=True, timeout=30,
    )
    if archive.returncode != 0:
        raise RuntimeError(f"could not export legacy base {base_commit}: {archive.stderr!r}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
        stream.extractall(legacy_source)
    archived_kokkos = legacy_source / "kokkos"
    if archived_kokkos.exists():
        shutil.rmtree(archived_kokkos)
    archived_kokkos.symlink_to((source_dir / "kokkos").resolve(), target_is_directory=True)
    run_build(
        ["cmake", "-S", legacy_source, "-B", legacy_build,
         "-DCMAKE_BUILD_TYPE=Debug", "-DAthena_ENABLE_MPI=OFF",
         "-DAthena_ENABLE_OPENMP=OFF", "-DAthena_BUILD_UNIT_TESTS=OFF"],
        work_dir, 120,
    )
    run_build(["cmake", "--build", legacy_build, "--target", "athena", "-j2"],
              work_dir, 600)
    executable = legacy_build / "src" / "athena"
    if not executable.is_file():
        raise RuntimeError("legacy base build omitted src/athena")
    return executable


def main():
    test_legacy_fastflow_time_key_removal()
    test_restart_binary_payload_contract()
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--legacy-base", required=True)
    args = parser.parse_args()

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    initial = args.work_dir / "initial"
    initial.mkdir()
    run([args.athena, "-i", args.input, "-d", initial], args.work_dir, True,
        ("AssembleZ4cTasks", "Terminating on cycle limit"))
    restart = initial / "rst" / "z4c_restart_carrier.00000.rst"
    if not restart.is_file():
        raise RuntimeError("initial run did not produce the restart fixture")
    restart_data = restart.read_bytes()
    for key in (
        b"<z4c_restart>",
        b"carrier_schema",
        b"central_initialized",
        b"central_proper_time",
        b"central_constraint_norm",
        b"central_abs_kretschmann",
        b"fastflow_coefficients",
        b"fastflow_last_search_time",
        b"fastflow_time_first_found",
    ):
        if key not in restart_data:
            raise RuntimeError(f"restart ParameterDump omitted {key!r}")

    # An ordinary Cartesian run retains a valid, inert schema-v2 central carrier.
    # These defaults are intentionally preserved when no Cartoon state is sampled.
    for key, value in (
        (b"central_schema", b"2"),
        (b"central_initialized", b"0"),
        (b"central_proper_time", b"0"),
        (b"central_previous_lapse", b"1"),
        (b"central_constraint_norm", b"0"),
        (b"central_abs_kretschmann", b"0"),
        (b"central_sample_gid", b"-1"),
        (b"central_sample_level", b"-1"),
        (b"central_last_cycle", b"-1"),
        (b"central_last_time", b"0"),
        (b"fastflow_schema", b"2"),
        (b"fastflow_time_first_found", b"-1"),
    ):
        if not re.search(rb"\b" + key + rb"\s*=\s*" + re.escape(value) + rb"\b",
                         restart_data):
            raise RuntimeError(
                f"fresh Cartesian restart has invalid {key.decode()} default"
            )

    roundtrip = args.work_dir / "roundtrip"
    roundtrip.mkdir()
    run([args.athena, "-r", restart, "-d", roundtrip], args.work_dir, True,
        ("AssembleZ4cTasks", "Terminating on cycle limit"))

    compatible = args.work_dir / "compatible"
    compatible.mkdir()
    run(
        [args.athena, "-r", restart, "-d", compatible,
         "z4c/spatial_order=-1", "z4c_restart/central_proper_time=0"],
        args.work_dir,
        True,
        ("AssembleZ4cTasks",),
    )

    # Runtime configuration and every integration-authoritative namespace reject
    # independent CLI changes before Mesh or AddPhysics construction.
    cli_cases = (
        ("z4c/symmetry=cartoon_so2", "z4c", "symmetry", "cartesian3d", "cartoon_so2"),
        ("z4c/coordinate_map=signed_rho_z_suppressed_y_v1", "z4c", "coordinate_map",
         "cartesian_xyz", "signed_rho_z_suppressed_y_v1"),
        ("z4c/symmetry_schema=2", "z4c", "symmetry_schema", "1", "2"),
        ("z4c/restart_symmetry=cartoon_so2", "z4c", "restart_symmetry",
         "cartesian3d", "cartoon_so2"),
        ("z4c/restart_coordinate_map=signed_rho_z_suppressed_y_v1", "z4c",
         "restart_coordinate_map", "cartesian_xyz", "signed_rho_z_suppressed_y_v1"),
        ("z4c/restart_symmetry_schema=2", "z4c", "restart_symmetry_schema", "1", "2"),
        ("z4c/spatial_order=2", "z4c", "spatial_order", "-1", "2"),
        ("mesh/nghost=3", "z4c", "effective_spatial_order", "2", "4"),
        ("mesh/nx1=16", "mesh", "nx1", "8", "16"),
        ("mesh/nx2=8", "mesh", "nx2", "4", "8"),
        ("mesh/nx3=8", "mesh", "nx3", "4", "8"),
        ("meshblock/nx1=8", "meshblock", "nx1", "4", "8"),
        ("meshblock/nx2=8", "meshblock", "nx2", "4", "8"),
        ("meshblock/nx3=8", "meshblock", "nx3", "4", "8"),
        ("z4c_restart/carrier_schema=2", "z4c_restart", "carrier_schema", "1", "2"),
        ("z4c_restart/symmetry=cartoon_so2", "z4c_restart", "symmetry",
         "cartesian3d", "cartoon_so2"),
        ("z4c_restart/coordinate_map=signed_rho_z_suppressed_y_v1", "z4c_restart",
         "coordinate_map", "cartesian_xyz", "signed_rho_z_suppressed_y_v1"),
        ("z4c_restart/symmetry_schema=2", "z4c_restart", "symmetry_schema", "1", "2"),
        ("z4c_restart/requested_spatial_order=2", "z4c_restart",
         "requested_spatial_order", "-1", "2"),
        ("z4c_restart/effective_spatial_order=4", "z4c_restart",
         "effective_spatial_order", "2", "4"),
        ("z4c_restart/stencil_width=3", "z4c_restart", "stencil_width", "2", "3"),
        ("z4c_restart/central_schema=3", "z4c_restart", "central_schema", "2", "3"),
        ("z4c_restart/central_initialized=1", "z4c_restart",
         "central_initialized", "0", "1"),
        ("z4c_restart/central_proper_time=1", "z4c_restart", "central_proper_time", "0", "1"),
        ("z4c_restart/central_proper_time=garbage", "z4c_restart",
         "central_proper_time", "0", "garbage"),
        ("z4c_restart/central_previous_lapse=2", "z4c_restart", "central_previous_lapse", "1", "2"),
        ("z4c_restart/central_constraint_norm=2", "z4c_restart",
         "central_constraint_norm", "0", "2"),
        ("z4c_restart/central_abs_kretschmann=2", "z4c_restart",
         "central_abs_kretschmann", "0", "2"),
        ("z4c_restart/central_sample_gid=0", "z4c_restart",
         "central_sample_gid", "-1", "0"),
        ("z4c_restart/central_sample_level=0", "z4c_restart",
         "central_sample_level", "-1", "0"),
        ("z4c_restart/central_last_cycle=0", "z4c_restart", "central_last_cycle", "-1", "0"),
        ("z4c_restart/central_last_time=1", "z4c_restart", "central_last_time", "0", "1"),
        ("z4c_restart/fastflow_schema=1", "z4c_restart", "fastflow_schema", "2", "1"),
        ("z4c_restart/fastflow_coefficient_count=1", "z4c_restart",
         "fastflow_coefficient_count", "0", "1"),
        ("z4c_restart/fastflow_coefficients=2.5", "z4c_restart",
         "fastflow_coefficients", "none", "2.5"),
        ("z4c_restart/fastflow_surface_mode=solo", "z4c_restart",
         "fastflow_surface_mode", "none", "solo"),
        ("z4c_restart/fastflow_selected_branch=plus", "z4c_restart",
         "fastflow_selected_branch", "none", "plus"),
        ("z4c_restart/fastflow_center_count=1", "z4c_restart",
         "fastflow_center_count", "0", "1"),
        ("z4c_restart/fastflow_center_z0=1", "z4c_restart", "fastflow_center_z0", "0", "1"),
        ("z4c_restart/fastflow_center_z1=1", "z4c_restart", "fastflow_center_z1", "0", "1"),
        ("z4c_restart/fastflow_status=searching", "z4c_restart",
         "fastflow_status", "not_started", "searching"),
        ("z4c_restart/fastflow_failure_code=test", "z4c_restart",
         "fastflow_failure_code", "none", "test"),
        ("z4c_restart/fastflow_last_search_cycle=0", "z4c_restart",
         "fastflow_last_search_cycle", "-1", "0"),
        ("z4c_restart/fastflow_last_search_time=1", "z4c_restart",
         "fastflow_last_search_time", "0", "1"),
        ("z4c_restart/fastflow_time_first_found=0", "z4c_restart",
         "fastflow_time_first_found", "-1", "0"),
        ("z4c_restart/fastflow_converged=1", "z4c_restart",
         "fastflow_converged", "0", "1"),
        ("z4c_restart/fastflow_converged=maybe", "z4c_restart",
         "fastflow_converged", "0", "maybe"),
    )
    for case in cli_cases:
        rejected(args.athena, restart, args.work_dir, *case)

    # -i overlays are checked after immutable capture just like CLI overrides.
    central_overlay = args.work_dir / "central_override.athinput"
    central_overlay.write_text("<z4c_restart>\ncentral_proper_time = 7\n")
    output = run(
        [args.athena, "-r", restart, "-i", central_overlay],
        args.work_dir, False,
        ("<z4c_restart>/central_proper_time", "stored='0'", "requested='7'"),
    )
    if "Root grid" in output:
        raise RuntimeError("central -i override reached Mesh construction")
    fastflow_overlay = args.work_dir / "fastflow_override.athinput"
    fastflow_overlay.write_text("<z4c_restart>\nfastflow_center_z0 = 8\n")
    output = run(
        [args.athena, "-r", restart, "-i", fastflow_overlay],
        args.work_dir, False,
        ("<z4c_restart>/fastflow_center_z0", "stored='0'", "requested='8'"),
    )
    if "Root grid" in output:
        raise RuntimeError("FastFlow -i override reached Mesh construction")

    alias_overlays = (
        ("restart_symmetry", "cartoon_so2", "cartesian3d"),
        ("restart_coordinate_map", "signed_rho_z_suppressed_y_v1", "cartesian_xyz"),
        ("restart_symmetry_schema", "2", "1"),
    )
    for key, requested, stored in alias_overlays:
        overlay = args.work_dir / f"{key}_override.athinput"
        overlay.write_text(f"<z4c>\n{key} = {requested}\n")
        output = run(
            [args.athena, "-r", restart, "-i", overlay],
            args.work_dir, False,
            (f"<z4c>/{key}", f"stored='{stored}'", f"requested='{requested}'"),
        )
        if "Root grid" in output:
            raise RuntimeError(f"alias -i override {key} reached Mesh construction")

    for key, requested, stored in (("nx1", "16", "8"),
                                   ("nx2", "8", "4"),
                                   ("nx3", "8", "4")):
        overlay = args.work_dir / f"mesh_{key}_override.athinput"
        overlay.write_text(f"<mesh>\n{key} = {requested}\n")
        output = run(
            [args.athena, "-r", restart, "-i", overlay], args.work_dir, False,
            (f"<mesh>/{key}", f"stored='{stored}'", f"requested='{requested}'"),
        )
        if "Root grid" in output:
            raise RuntimeError(f"mesh -i override {key} reached Mesh construction")

    # Every global and MeshBlock dimension in the binary header is checked against the
    # immutable text carrier before root-grid arithmetic or tree allocation.
    for target, expected, replacement in (
        ("mesh/nx1", 8, 16),
        ("mesh/nx2", 4, 8),
        ("mesh/nx3", 4, 8),
        ("meshblock/nx1", 4, 8),
        ("meshblock/nx2", 4, 8),
        ("meshblock/nx3", 4, 8),
    ):
        corrupted = args.work_dir / f"binary_{target.replace('/', '_')}.rst"
        corrupted.write_bytes(
            mutate_binary_dimension(restart_data, target, expected, replacement)
        )
        output = run(
            [args.athena, "-r", corrupted], args.work_dir, False,
            ("immutable Z4c binary restart validation failed", f"<{target}>",
             f"stored='{expected}'", f"binary='{replacement}'"),
        )
        if "Root grid" in output or "AssembleZ4cTasks" in output:
            raise RuntimeError(f"binary {target} mismatch reached allocation:\n{output}")

    # A self-consistent collapsed text identity cannot be paired with a 3-D binary tree.
    collapsed = restart_data
    for block, key, value in (
        ("z4c", "symmetry", "cartoon_so2"),
        ("z4c", "coordinate_map", "signed_rho_z_suppressed_y_v1"),
        ("z4c", "restart_symmetry", "cartoon_so2"),
        ("z4c", "restart_coordinate_map", "signed_rho_z_suppressed_y_v1"),
        ("z4c_restart", "symmetry", "cartoon_so2"),
        ("z4c_restart", "coordinate_map", "signed_rho_z_suppressed_y_v1"),
        ("z4c_restart", "mesh_nx3", "1"),
        ("z4c_restart", "meshblock_nx3", "1"),
        ("mesh", "nx3", "1"),
        ("meshblock", "nx3", "1"),
    ):
        collapsed = replace_block_value(collapsed, block, key, value)
    collapsed_restart = args.work_dir / "collapsed_text_3d_binary.rst"
    collapsed_restart.write_bytes(collapsed)
    output = run(
        [args.athena, "-r", collapsed_restart], args.work_dir, False,
        ("immutable Z4c binary restart validation failed", "<mesh/nx3>",
         "stored='1'", "binary='4'"),
    )
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("collapsed/3-D restart mismatch reached allocation")

    # A same-length partial-carrier corruption is rejected during restart-origin capture.
    partial = args.work_dir / "partial.rst"
    partial.write_bytes(replace_once(restart_data, b"central_last_cycle",
                                     b"central_last_cyclx"))
    run([args.athena, "-r", partial], args.work_dir, False,
        ("invalid restart-origin Z4c carrier", "partial <z4c_restart> carrier",
         "<z4c_restart>/central_last_cycle"))
    partial_fastflow = args.work_dir / "partial_fastflow.rst"
    partial_fastflow.write_bytes(
        replace_once(restart_data, b"fastflow_time_first_found",
                     b"fastflow_time_first_founx"))
    run([args.athena, "-r", partial_fastflow], args.work_dir, False,
        ("invalid restart-origin Z4c carrier", "partial <z4c_restart> carrier",
         "<z4c_restart>/fastflow_time_first_found"))

    schema = args.work_dir / "schema.rst"
    schema.write_bytes(replace_value(restart_data, "carrier_schema", "1", "2"))
    run([args.athena, "-r", schema], args.work_dir, False,
        ("invalid restart-origin Z4c carrier", "<z4c_restart>/carrier_schema=2"))

    # The central schema is deliberately fail-closed: schema-1 carriers lack the
    # new constraint/curvature/owner fields and are never silently inferred.
    old_central_schema = args.work_dir / "old_central_schema.rst"
    old_central_schema.write_bytes(
        replace_block_value(restart_data, "z4c_restart", "central_schema", "1")
    )
    output = run(
        [args.athena, "-r", old_central_schema], args.work_dir, False,
        ("invalid restart-origin Z4c carrier",
         "invalid axis-central integration state in <z4c_restart>"),
    )
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("old central schema reached allocation")

    # Exact inactive schema-1 state is the sole m=0 migration path.  It carries no
    # first-found key and is upgraded to current schema 2 with an explicit -1 value.
    legacy_fastflow_data = replace_value(restart_data, "fastflow_schema", "2", "1")
    legacy_fastflow_data = remove_legacy_fastflow_time_key(legacy_fastflow_data)
    legacy_fastflow = args.work_dir / "legacy_fastflow_schema1.rst"
    legacy_fastflow.write_bytes(legacy_fastflow_data)
    legacy_fastflow_run = args.work_dir / "legacy_fastflow_schema1_run"
    legacy_fastflow_run.mkdir()
    run([args.athena, "-r", legacy_fastflow, "-d", legacy_fastflow_run],
        args.work_dir, True, ("AssembleZ4cTasks",))
    upgraded_fastflow = sorted(
        (legacy_fastflow_run / "rst").glob("*.rst"))[0].read_bytes()
    for key, value in ((b"fastflow_schema", b"2"),
                       (b"fastflow_time_first_found", b"-1")):
        if not re.search(rb"\b" + key + rb"\s*=\s*" + re.escape(value) + rb"\b",
                         upgraded_fastflow):
            raise RuntimeError(f"schema-1 FastFlow upgrade omitted {key!r}")
    active_legacy_fastflow = args.work_dir / "active_legacy_fastflow_schema1.rst"
    active_legacy_fastflow.write_bytes(
        replace_value(legacy_fastflow_data, "fastflow_last_search_cycle", "-1", "00"))
    output = run(
        [args.athena, "-r", active_legacy_fastflow], args.work_dir, False,
        ("invalid restart-origin Z4c carrier",
         "active schema-1 m=0 FastFlow state lacks fastflow_time_first_found"))
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("active schema-1 FastFlow state reached allocation")
    ambiguous_legacy_fastflow = args.work_dir / "ambiguous_legacy_fastflow_schema1.rst"
    ambiguous_legacy_fastflow.write_bytes(
        replace_value(restart_data, "fastflow_schema", "2", "1"))
    output = run(
        [args.athena, "-r", ambiguous_legacy_fastflow], args.work_dir, False,
        ("invalid restart-origin Z4c carrier",
         "fastflow_schema=1 must not contain fastflow_time_first_found"))
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("ambiguous schema-1 FastFlow state reached allocation")

    for key, replacement in (("fastflow_time_first_found", "nan"),
                             ("fastflow_time_first_found", "-2")):
        corrupted = args.work_dir / f"invalid_{key}_{replacement}.rst"
        corrupted.write_bytes(
            replace_block_value(restart_data, "z4c_restart", key, replacement))
        output = run(
            [args.athena, "-r", corrupted], args.work_dir, False,
            ("invalid restart-origin Z4c carrier",
             "invalid m=0 FastFlow integration state in <z4c_restart>"))
        if "Root grid" in output or "AssembleZ4cTasks" in output:
            raise RuntimeError(f"invalid {key} reached allocation:\n{output}")

    # Every persisted central diagnostic is strict finite metadata. Invalid values and
    # an initialized flag without owner/cycle metadata fail before Mesh construction.
    for key, replacement in (
        ("central_proper_time", "nan"),
        ("central_previous_lapse", "-1"),
        ("central_constraint_norm", "inf"),
        ("central_abs_kretschmann", "nan"),
        ("central_initialized", "1"),
    ):
        corrupted = args.work_dir / f"invalid_{key}.rst"
        corrupted.write_bytes(
            replace_block_value(restart_data, "z4c_restart", key, replacement)
        )
        output = run(
            [args.athena, "-r", corrupted], args.work_dir, False,
            ("invalid restart-origin Z4c carrier", "axis-central"),
        )
        if "Root grid" in output or "AssembleZ4cTasks" in output:
            raise RuntimeError(f"invalid {key} reached allocation:\n{output}")

    # Seed non-default central and restart-authoritative FastFlow values in the origin,
    # then verify that a compatible restart restores and reserializes every value.
    seeded_data = restart_data
    for key, old, new in (
        ("central_initialized", "0", "1"),
        ("central_proper_time", "0", "2"),
        ("central_previous_lapse", "1", "3"),
        ("central_constraint_norm", "0", "4"),
        ("central_abs_kretschmann", "0", "5"),
        ("central_sample_gid", "-1", "06"),
        ("central_sample_level", "-1", "07"),
        ("central_last_cycle", "-1", "07"),
        ("central_last_time", "0", "4"),
        ("fastflow_coefficient_count", "0", "1"),
        ("fastflow_coefficients", "none", "2.50"),
        ("fastflow_surface_mode", "none", "single"),
        ("fastflow_selected_branch", "none", "plus"),
        ("fastflow_center_count", "0", "1"),
        ("fastflow_center_z0", "0", "5"),
        ("fastflow_center_z1", "0", "0"),
        ("fastflow_status", "not_started", "accepted"),
        ("fastflow_failure_code", "none", "none"),
        ("fastflow_last_search_cycle", "-1", "08"),
        ("fastflow_last_search_time", "0", "9"),
        ("fastflow_time_first_found", "-1", "07"),
        ("fastflow_converged", "0", "1"),
    ):
        seeded_data = replace_value(
            seeded_data, key, old, new, require_same_length=False)
    seeded = args.work_dir / "seeded.rst"
    seeded.write_bytes(seeded_data)
    seeded_run = args.work_dir / "seeded_run"
    seeded_run.mkdir()
    run([args.athena, "-r", seeded, "-d", seeded_run], args.work_dir, True,
        ("AssembleZ4cTasks",))
    restored = sorted((seeded_run / "rst").glob("*.rst"))[0].read_bytes()
    for key, value in (
        (b"central_initialized", b"1"),
        (b"central_proper_time", b"2"),
        (b"central_previous_lapse", b"3"),
        (b"central_constraint_norm", b"4"),
        (b"central_abs_kretschmann", b"5"),
        (b"central_sample_gid", b"6"),
        (b"central_sample_level", b"7"),
        (b"central_last_cycle", b"7"),
        (b"central_last_time", b"4"),
        (b"fastflow_coefficient_count", b"1"),
        (b"fastflow_coefficients", b"2.5"),
        (b"fastflow_surface_mode", b"single"),
        (b"fastflow_selected_branch", b"plus"),
        (b"fastflow_center_count", b"1"),
        (b"fastflow_center_z0", b"5"),
        (b"fastflow_center_z1", b"0"),
        (b"fastflow_status", b"accepted"),
        (b"fastflow_failure_code", b"none"),
        (b"fastflow_last_search_cycle", b"8"),
        (b"fastflow_last_search_time", b"9"),
        (b"fastflow_time_first_found", b"7"),
        (b"fastflow_converged", b"1"),
    ):
        if not re.search(rb"\b" + key + rb"\s*=\s*" + re.escape(value) + rb"\b",
                         restored):
            raise RuntimeError(f"restored restart omitted {key.decode()}={value.decode()}")

    # Exercise a genuine carrier-free restart produced by the exact pre-carrier base,
    # rather than relying only on a carrier-stripped current fixture.
    legacy_athena = build_legacy_athena(args.source_dir, args.work_dir,
                                        args.legacy_base)
    base_origin = args.work_dir / "base_origin"
    base_origin.mkdir()
    run([legacy_athena, "-i", args.input, "-d", base_origin], args.work_dir, True,
        ("AssembleZ4cTasks", "Terminating on cycle limit"))
    base_restart = base_origin / "rst" / "z4c_restart_carrier.00000.rst"
    if not base_restart.is_file() or b"<z4c_restart>" in base_restart.read_bytes():
        raise RuntimeError("exact legacy base did not produce a carrier-free restart")
    mms_overlay = args.work_dir / "legacy_mms_override.athinput"
    mms_overlay.write_text(
        "<problem>\npgen_name = z4c_cartoon_derivatives\ncheck_only = true\n"
    )
    output = run(
        [args.athena, "-r", base_restart, "-i", mms_overlay], args.work_dir, False,
        ("Z4c preallocation validation failed",
         "z4c_cartoon_derivatives check_only rejects restart"),
    )
    if "AssembleZ4cTasks" in output or "Setup complete, executing task list(s)" in output:
        raise RuntimeError("genuine legacy restart MMS injection reached physics allocation")
    base_restart_data = base_restart.read_bytes()
    assert_restart_binary_payload_unchanged(restart_data, base_restart_data)
    legacy_upgrade = args.work_dir / "legacy_upgrade"
    legacy_upgrade.mkdir()
    run([args.athena, "-r", base_restart, "-d", legacy_upgrade],
        args.work_dir, True, ("AssembleZ4cTasks",))
    upgraded_restarts = sorted((legacy_upgrade / "rst").glob("*.rst"))
    if not upgraded_restarts or b"<z4c_restart>" not in upgraded_restarts[0].read_bytes():
        raise RuntimeError("compatible legacy restart was not upgraded with a carrier")

    block_start = restart_data.index(b"<z4c_restart>\n")
    block_end = restart_data.index(b"#------------------------- PAR_DUMP", block_start)
    complete_carrier = args.work_dir / "complete_carrier_override.athinput"
    complete_carrier.write_bytes(restart_data[block_start:block_end])
    output = run(
        [args.athena, "-r", base_restart, "-i", complete_carrier],
        args.work_dir, False,
        ("restart origin has no immutable <z4c_restart> carrier",
         "post-capture carrier injection is forbidden"),
    )
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("legacy -i carrier injection reached Mesh construction")
    output = run(
        [args.athena, "-r", base_restart,
         "z4c_restart/central_proper_time=9"],
        args.work_dir, False,
        ("restart origin has no immutable <z4c_restart> carrier",
         "<z4c_restart>/central_proper_time", "requested='9'"),
    )
    if "Root grid" in output or "AssembleZ4cTasks" in output:
        raise RuntimeError("legacy CLI carrier injection reached Mesh construction")

    # Keep a carrier-stripped fixture as a direct format test as well. Binary payload
    # offsets remain self-describing because ParameterInput stops at <par_end>.
    legacy = args.work_dir / "legacy.rst"
    legacy.write_bytes(restart_data[:block_start] + restart_data[block_end:])
    legacy_run = args.work_dir / "legacy_run"
    legacy_run.mkdir()
    run([args.athena, "-r", legacy, "-d", legacy_run], args.work_dir, True,
        ("AssembleZ4cTasks",))
    output = run(
        [args.athena, "-r", legacy, "-i", complete_carrier],
        args.work_dir, False,
        ("restart origin has no immutable <z4c_restart> carrier",
         "post-capture carrier injection is forbidden"),
    )
    if "Root grid" in output:
        raise RuntimeError("stripped legacy carrier injection reached Mesh construction")

    fresh_reserved = args.work_dir / "fresh_reserved.athinput"
    fresh_reserved.write_bytes(
        args.input.read_bytes() + b"\n<z4c_restart>\ncarrier_schema = 1\n"
    )
    run([args.athena, "-i", fresh_reserved], args.work_dir, False,
        ("<z4c_restart> is an internal restart-only carrier",))

    print("Z4c immutable restart carrier tests passed")


if __name__ == "__main__":
    main()
