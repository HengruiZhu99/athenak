#!/usr/bin/env python3
"""Production-path roundtrip and override tests for the Z4c restart carrier."""

import argparse
import re
import shutil
import subprocess
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


def replace_value(data, key, old, new):
    if len(old) != len(new):
        raise RuntimeError(f"value replacement changes length for {key}")
    pattern = re.compile(rb"(" + re.escape(key.encode()) + rb"\s*=\s*)" +
                         re.escape(old.encode()) + rb"(\s)")
    replaced, count = pattern.subn(rb"\g<1>" + new.encode() + rb"\g<2>", data, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace {key}={old} in restart")
    return replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
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
        b"central_proper_time",
        b"fastflow_coefficients",
        b"fastflow_last_search_time",
    ):
        if key not in restart_data:
            raise RuntimeError(f"restart ParameterDump omitted {key!r}")

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
        ("z4c/spatial_order=2", "z4c", "spatial_order", "-1", "2"),
        ("mesh/nghost=3", "z4c", "effective_spatial_order", "2", "4"),
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
        ("z4c_restart/central_schema=2", "z4c_restart", "central_schema", "1", "2"),
        ("z4c_restart/central_proper_time=1", "z4c_restart", "central_proper_time", "0", "1"),
        ("z4c_restart/central_proper_time=garbage", "z4c_restart",
         "central_proper_time", "0", "garbage"),
        ("z4c_restart/central_previous_lapse=2", "z4c_restart", "central_previous_lapse", "1", "2"),
        ("z4c_restart/central_last_cycle=0", "z4c_restart", "central_last_cycle", "-1", "0"),
        ("z4c_restart/central_last_time=1", "z4c_restart", "central_last_time", "0", "1"),
        ("z4c_restart/fastflow_schema=2", "z4c_restart", "fastflow_schema", "1", "2"),
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

    # A same-length partial-carrier corruption is rejected during restart-origin capture.
    partial = args.work_dir / "partial.rst"
    partial.write_bytes(replace_once(restart_data, b"central_last_cycle",
                                     b"central_last_cyclx"))
    run([args.athena, "-r", partial], args.work_dir, False,
        ("invalid restart-origin Z4c carrier", "partial <z4c_restart> carrier",
         "<z4c_restart>/central_last_cycle"))

    schema = args.work_dir / "schema.rst"
    schema.write_bytes(replace_value(restart_data, "carrier_schema", "1", "2"))
    run([args.athena, "-r", schema], args.work_dir, False,
        ("invalid restart-origin Z4c carrier", "<z4c_restart>/carrier_schema=2"))

    # Seed non-default central and reserved FastFlow values in the authoritative origin,
    # then verify that a compatible restart restores and reserializes every value.
    seeded_data = restart_data
    for key, old, new in (
        ("central_proper_time", "0", "2"),
        ("central_previous_lapse", "1", "3"),
        ("central_last_cycle", "-1", "07"),
        ("central_last_time", "0", "4"),
        ("fastflow_coefficient_count", "0", "1"),
        ("fastflow_coefficients", "none", "2.50"),
        ("fastflow_surface_mode", "none", "solo"),
        ("fastflow_selected_branch", "none", "plus"),
        ("fastflow_center_count", "0", "1"),
        ("fastflow_center_z0", "0", "5"),
        ("fastflow_center_z1", "0", "6"),
        ("fastflow_status", "not_started", "state_saved"),
        ("fastflow_failure_code", "none", "test"),
        ("fastflow_last_search_cycle", "-1", "08"),
        ("fastflow_last_search_time", "0", "9"),
        ("fastflow_converged", "0", "1"),
    ):
        seeded_data = replace_value(seeded_data, key, old, new)
    seeded = args.work_dir / "seeded.rst"
    seeded.write_bytes(seeded_data)
    seeded_run = args.work_dir / "seeded_run"
    seeded_run.mkdir()
    run([args.athena, "-r", seeded, "-d", seeded_run], args.work_dir, True,
        ("AssembleZ4cTasks",))
    restored = sorted((seeded_run / "rst").glob("*.rst"))[0].read_bytes()
    for key, value in (
        (b"central_proper_time", b"2"),
        (b"central_previous_lapse", b"3"),
        (b"central_last_cycle", b"7"),
        (b"central_last_time", b"4"),
        (b"fastflow_coefficient_count", b"1"),
        (b"fastflow_coefficients", b"2.5"),
        (b"fastflow_surface_mode", b"solo"),
        (b"fastflow_selected_branch", b"plus"),
        (b"fastflow_center_count", b"1"),
        (b"fastflow_center_z0", b"5"),
        (b"fastflow_center_z1", b"6"),
        (b"fastflow_status", b"state_saved"),
        (b"fastflow_failure_code", b"test"),
        (b"fastflow_last_search_cycle", b"8"),
        (b"fastflow_last_search_time", b"9"),
        (b"fastflow_converged", b"1"),
    ):
        if not re.search(rb"\b" + key + rb"\s*=\s*" + re.escape(value) + rb"\b",
                         restored):
            raise RuntimeError(f"restored restart omitted {key.decode()}={value.decode()}")

    # Removing only the text carrier emulates a legacy Cartesian restart. Binary payload
    # offsets remain self-describing because ParameterInput stops at <par_end>.
    legacy = args.work_dir / "legacy.rst"
    block_start = restart_data.index(b"<z4c_restart>\n")
    block_end = restart_data.index(b"#------------------------- PAR_DUMP", block_start)
    legacy.write_bytes(restart_data[:block_start] + restart_data[block_end:])
    legacy_run = args.work_dir / "legacy_run"
    legacy_run.mkdir()
    run([args.athena, "-r", legacy, "-d", legacy_run], args.work_dir, True,
        ("AssembleZ4cTasks",))

    fresh_reserved = args.work_dir / "fresh_reserved.athinput"
    fresh_reserved.write_bytes(
        args.input.read_bytes() + b"\n<z4c_restart>\ncarrier_schema = 1\n"
    )
    run([args.athena, "-i", fresh_reserved], args.work_dir, False,
        ("<z4c_restart> is an internal restart-only carrier",))

    print("Z4c immutable restart carrier tests passed")


if __name__ == "__main__":
    main()
