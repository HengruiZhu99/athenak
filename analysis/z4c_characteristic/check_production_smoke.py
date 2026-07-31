#!/usr/bin/env python3
"""Apply the acceptance gates for the 32-node Aurora production smoke run."""

import argparse
import math
import pathlib
import re


EXPECTED_HISTOGRAM = {
    1: 52,
    2: 84,
    3: 84,
    4: 84,
    5: 80,
    6: 108,
    7: 140,
    8: 148,
    9: 84,
    10: 84,
    11: 84,
    12: 88,
    13: 64,
}

FATAL_PATTERNS = (
    r"### FATAL",
    r"\bMPI_ABORT\b",
    r"\bMPI(?:\s+|_).*error\b",
    r"\bSYCL(?:\s+|_).*error\b",
    r"\bZE_RESULT_[A-Z_]+\b",
    r"\bPI_ERROR_[A-Z_]+\b",
    r"\b(?:out of memory|std::bad_alloc|segmentation fault)\b",
    r"\b(?:nan|inf)\b",
)


def fail(message):
    raise SystemExit(message)


def parse_metadata(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def parse_input(path):
    result = {}
    section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        section_match = re.fullmatch(r"<([^>]+)>", line)
        if section_match:
            section = section_match.group(1)
            continue
        if "=" in line and section is not None:
            key, value = line.split("=", 1)
            result[(section, key.strip())] = value.strip()
    return result


def require_input_value(values, section, key, expected):
    actual = values.get((section, key))
    if actual != str(expected):
        fail(
            "{} / {}: expected {!r}, found {!r}".format(
                section, key, str(expected), actual
            )
        )


def parse_key_values(line):
    values = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def finite_float(value, label):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        fail("{} is not a floating-point value: {!r}".format(label, value))
    if not math.isfinite(parsed):
        fail("{} is nonfinite: {!r}".format(label, value))
    return parsed


def check_histories(run_dir):
    histories = sorted(run_dir.glob("*.hst"))
    if not histories:
        fail("no history files found")
    rows_seen = 0
    for path in histories:
        lines = path.read_text(encoding="utf-8").splitlines()
        labels = {}
        for line in lines:
            if not line.lstrip().startswith("#"):
                continue
            for match in re.finditer(r"\[(\d+)\]=(\S+)", line):
                labels[match.group(2)] = int(match.group(1)) - 1
        last_time = -math.inf
        file_rows = []
        for line_number, line in enumerate(
            lines, 1
        ):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                values = [float(token) for token in line.split()]
            except ValueError:
                fail("{}:{} contains a nonnumeric history row".format(path, line_number))
            if not values or not all(math.isfinite(value) for value in values):
                fail("{}:{} contains a nonfinite history row".format(path, line_number))
            if values[0] < last_time:
                fail("{}:{} has decreasing output time".format(path, line_number))
            last_time = values[0]
            file_rows.append(values)
            rows_seen += 1
        if len(file_rows) < 2:
            fail("{} has fewer than two finite history rows".format(path))

        if (
            path.name.endswith(".user.hst")
            and not path.name.endswith(".z4c.user.hst")
        ):
            required = (
                "rho-max",
                "alpha-min",
                "chi-min",
                "detg-min",
                "bad-metric",
            )
            missing = [label for label in required if label not in labels]
            if missing:
                fail(
                    "{} lacks production metric diagnostics: {}".format(
                        path, ", ".join(missing)
                    )
                )
            for row in file_rows:
                for label in ("rho-max", "alpha-min", "chi-min", "detg-min"):
                    if row[labels[label]] <= 0.0:
                        fail("{} reports nonpositive {}".format(path, label))
                if row[labels["bad-metric"]] != 0.0:
                    fail("{} reports a nonzero bad-metric count".format(path))

        if path.name.endswith(".z4c.user.hst"):
            required = (
                "C-norm2",
                "H-norm2",
                "M-norm2",
                "Z-norm2",
                "Theta-norm",
                "Volume",
            )
            missing = [label for label in required if label not in labels]
            if missing:
                fail(
                    "{} lacks Z4c constraint diagnostics: {}".format(
                        path, ", ".join(missing)
                    )
                )
            for row in file_rows:
                for label in required:
                    if row[labels[label]] < 0.0:
                        fail("{} reports negative {}".format(path, label))
                if row[labels["Volume"]] <= 0.0:
                    fail("{} reports nonpositive diagnostic volume".format(path))
    return len(histories), rows_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--expected-blocks", type=int, default=1184)
    parser.add_argument("--expected-nodes", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=12)
    parser.add_argument("--maximum-cpbc-fraction", type=float, default=0.03)
    parser.add_argument("--warmup-samples", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    stdout_path = run_dir / "athena.stdout"
    metadata_path = run_dir / "run_metadata.txt"
    if not stdout_path.is_file() or not metadata_path.is_file():
        fail("run directory lacks athena.stdout or run_metadata.txt")

    metadata = parse_metadata(metadata_path)
    expected_metadata = {
        "nodes": str(args.expected_nodes),
        "ranks_per_node": str(args.ranks_per_node),
        "mpich_gpu_support_enabled": "1",
        "sycl_device_filter": "level_zero:gpu",
        "oneapi_device_selector": "level_zero:gpu",
        "ze_flat_device_hierarchy": "COMPOSITE",
        "omp_num_threads": "1",
        "production_smoke": "1",
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            fail(
                "metadata {}: expected {!r}, found {!r}".format(
                    key, expected, actual
                )
            )

    input_candidates = sorted(run_dir.glob("*.athinput"))
    if len(input_candidates) != 1:
        fail("expected exactly one copied Athena input")
    input_values = parse_input(input_candidates[0])
    for axis in ("x1", "x2", "x3"):
        require_input_value(input_values, "mesh", "n{}".format(axis), 128)
        require_input_value(input_values, "mesh", "{}min".format(axis), -1024.0)
        require_input_value(input_values, "mesh", "{}max".format(axis), 1024.0)
        require_input_value(input_values, "meshblock", "n{}".format(axis), 64)
    require_input_value(input_values, "mesh_refinement", "num_levels", 14)
    require_input_value(input_values, "mesh_refinement", "max_nmb_per_rank", 16)
    require_input_value(input_values, "z4c", "boundary_rhs", "characteristic_cpbc")
    require_input_value(
        input_values, "z4c", "characteristic_bc_source", "zero_rate"
    )
    require_input_value(input_values, "z4c", "extrap_order", 4)
    require_input_value(input_values, "z4c", "characteristic_bc_diagnostics", "true")
    require_input_value(input_values, "problem", "outer_sponge_geometry", "radial")
    require_input_value(input_values, "problem", "outer_sponge_start_radius", 512.0)
    require_input_value(input_values, "problem", "outer_sponge_ramp_width", 128.0)
    require_input_value(input_values, "problem", "outer_sponge_damping_time", 16.0)
    require_input_value(input_values, "problem", "amr_bh_refine_level", 8)
    require_input_value(input_values, "problem", "amr_star_refine_level", 13)
    require_input_value(input_values, "problem", "amr_star_track_rhomax", "true")

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    lowered = stdout.lower()
    for pattern in FATAL_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            fail("fatal runtime signature matched {!r}".format(pattern))

    root_match = re.search(
        r"Root grid =\s+(\d+)\s+x\s+(\d+)\s+x\s+(\d+)\s+MeshBlocks", stdout
    )
    if root_match is None or tuple(map(int, root_match.groups())) != (2, 2, 2):
        fail("production root grid is not 2 x 2 x 2 MeshBlocks")
    block_match = re.search(r"Total number of MeshBlocks =\s+(\d+)", stdout)
    if block_match is None or int(block_match.group(1)) != args.expected_blocks:
        fail("startup MeshBlock count does not equal {}".format(args.expected_blocks))
    physical_match = re.search(
        r"Number of physical levels of refinement =\s+(\d+)", stdout
    )
    if physical_match is None or int(physical_match.group(1)) != 13:
        fail("production mesh does not report physical refinement level 13")

    histogram = {
        int(level): int(count)
        for level, count in re.findall(
            r"Physical level =\s+(\d+).*?:\s+(\d+)\s+MeshBlocks", stdout
        )
    }
    if histogram != EXPECTED_HISTOGRAM:
        fail(
            "startup level histogram differs from mesh-only reference: {!r}".format(
                histogram
            )
        )

    expected_ranks = args.expected_nodes * args.ranks_per_node
    rank_blocks = [
        (int(rank), int(count))
        for rank, count in re.findall(
            r"Rank =\s+(\d+):\s+(\d+)\s+MeshBlocks", stdout
        )
    ]
    if len(rank_blocks) != expected_ranks:
        fail(
            "expected {} rank-allocation rows, found {}".format(
                expected_ranks, len(rank_blocks)
            )
        )
    if sorted(rank for rank, unused in rank_blocks) != list(range(expected_ranks)):
        fail("rank-allocation rows do not cover every MPI rank exactly once")
    maximum_rank_blocks = max(count for unused, count in rank_blocks)
    if maximum_rank_blocks > 16:
        fail("rank allocation exceeds max_nmb_per_rank=16")

    sponge_lines = [
        line for line in stdout.splitlines() if line.startswith("OUTER_SPONGE ")
    ]
    if len(sponge_lines) != 1:
        fail("expected exactly one OUTER_SPONGE setup line")
    sponge = parse_key_values(sponge_lines[0])
    if sponge.get("enabled") not in ("1", "true"):
        fail("production radial sponge is not enabled")
    if sponge.get("geometry") != "radial":
        fail("production sponge is not radial")
    for key, expected in (
        ("start", 512.0),
        ("ramp_width", 128.0),
        ("damping_time", 16.0),
        ("max_rate", 1.0 / 16.0),
        ("sigma_start", 0.0),
        ("sigma_mid", 0.5 / 16.0),
        ("sigma_end", 1.0 / 16.0),
        ("sigma_boundary", 1.0 / 16.0),
    ):
        actual = finite_float(sponge.get(key), "OUTER_SPONGE {}".format(key))
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-14):
            fail(
                "OUTER_SPONGE {}: expected {:.17g}, found {:.17g}".format(
                    key, expected, actual
                )
            )

    star_rows = []
    for line in stdout.splitlines():
        if line.startswith("STAR_TRACK "):
            values = parse_key_values(line)
            numeric = [
                finite_float(values.get(key), "STAR_TRACK {}".format(key))
                for key in ("time", "x", "y", "z", "r_bh", "rho_max")
            ]
            if values.get("valid") != "1":
                fail("STAR_TRACK reported invalid tracking")
            if numeric[4] <= 0.0 or numeric[5] <= 0.0:
                fail("STAR_TRACK reported nonpositive radius or density")
            star_rows.append(numeric)
    if len(star_rows) < 2 or star_rows[-1][0] <= 0.0:
        fail("production smoke made insufficient finite star-tracking progress")

    timing_rows = []
    boundary_blocks = set()
    for line in stdout.splitlines():
        if not line.startswith("Z4C_CHARACTERISTIC_CPBC "):
            continue
        values = parse_key_values(line)
        if values.get("incoming_modes") != "10":
            fail("CPBC did not report ten incoming propagating modes")
        block_token = values.get("boundary_blocks_max")
        try:
            block_counts = tuple(int(item) for item in block_token.split(","))
        except (AttributeError, ValueError):
            fail("malformed CPBC boundary_blocks_max diagnostic")
        if len(block_counts) != 3 or any(count <= 0 for count in block_counts):
            fail("CPBC did not process physical boundary blocks in every orientation")
        boundary_blocks.add(block_counts)
        for key in (
            "gauge",
            "constraint",
            "radiation",
            "enforcement",
            "correction",
            "diagnostic_kernel_seconds",
        ):
            value = finite_float(values.get(key), "CPBC {}".format(key))
            if value < 0.0:
                fail("CPBC {} is negative".format(key))
        if values.get("performance_valid") == "1":
            kernel = finite_float(values.get("kernel_seconds"), "CPBC kernel_seconds")
            rhs = finite_float(
                values.get("volume_rhs_seconds"), "CPBC volume_rhs_seconds"
            )
            if kernel < 0.0 or rhs <= 0.0:
                fail("invalid CPBC performance diagnostic")
            timing_rows.append(kernel / rhs)
    minimum_rows = args.warmup_samples + 2
    if len(timing_rows) < minimum_rows:
        fail(
            "need at least {} valid CPBC timing rows, found {}".format(
                minimum_rows, len(timing_rows)
            )
        )
    measured = timing_rows[args.warmup_samples :]
    maximum_fraction = max(measured)
    if maximum_fraction >= args.maximum_cpbc_fraction:
        fail(
            "CPBC critical-path fraction {:.8e} exceeds {:.8e}".format(
                maximum_fraction, args.maximum_cpbc_fraction
            )
        )

    history_files, history_rows = check_histories(run_dir)
    print(
        "PRODUCTION_SMOKE PASS blocks={} levels=0-13 ranks={} "
        "max_blocks_per_rank={} star_rows={} final_star_time={:.8e} "
        "history_files={} history_rows={} timing_rows={} "
        "max_cpbc_fraction={:.8e} boundary_blocks={}".format(
            args.expected_blocks,
            expected_ranks,
            maximum_rank_blocks,
            len(star_rows),
            star_rows[-1][0],
            history_files,
            history_rows,
            len(timing_rows),
            maximum_fraction,
            sorted(boundary_blocks),
        )
    )


if __name__ == "__main__":
    main()
