#!/usr/bin/env python3
"""Measure all incoming and outgoing Z4c characteristics in one TDE run."""

import argparse
import math
import pathlib
import re

import numpy as np

import compare_tde_boundaries as comparator


MODE_NAMES = tuple(
    mode
    for modes in comparator.CHARACTERISTIC_GROUPS.values()
    for mode in modes
)
TRACKED_FIELDS = (
    "z4c_Theta", "z4c_Khat",
    "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
    "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz",
)
PROBLEM_HISTORY_LABELS = (
    "rho-max", "Theta-max", "Khat-max", "alpha-res", "beta-res",
    "Gam-res", "bad-metric",
)
CONSTRAINT_HISTORY_LABELS = (
    "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
    "Mx-norm2", "My-norm2", "Mz-norm2", "Theta-norm",
)
ERROR_PATTERN = re.compile(
    r"FATAL|MPI[^\n]*(?:error|abort)|SYCL[^\n]*error|ZE_RESULT|"
    r"PI_ERROR|out of memory|bad_alloc|segmentation fault|"
    r"non[- ]?finite|bad[- ]?metric|(?:GPU|device)[^\n]*"
    r"(?:page fault|lost)",
    re.IGNORECASE,
)


def physical_coordinates(column, row, arguments):
    axes = comparator.PLANE_AXES[arguments.slice_plane]
    coordinates = [arguments.slice_fixed_coordinate] * 3
    coordinates[axes[0]] = column
    coordinates[axes[1]] = row
    return tuple(coordinates)


def row_bounds(arguments):
    if arguments.slice_row_min is not None:
        return arguments.slice_row_min, arguments.slice_row_max
    return -arguments.slice_y_half_width, arguments.slice_y_half_width


def update_global(summary, key, time, rms, linf, coordinates):
    record = summary[key]
    if rms > record["rms"]:
        record["rms"] = rms
        record["rms_time"] = time
    if linf > record["linf"]:
        record["linf"] = linf
        record["linf_time"] = time
        record["coordinates"] = coordinates


def read_named_history(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit("{}: incomplete history".format(path))
    labels = {
        match.group(2): int(match.group(1)) - 1
        for match in re.finditer(r"\[(\d+)\]=(\S+)", lines[1])
    }
    rows = [
        [float(value) for value in line.split()]
        for line in lines[2:]
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows or "time" not in labels:
        raise SystemExit("{}: missing history data or time column".format(path))
    data = np.asarray(rows, dtype=np.float64)
    time = data[:, labels["time"]]
    if not np.any(np.isfinite(time)):
        raise SystemExit("{}: no finite history time".format(path))
    order = np.argsort(np.where(np.isfinite(time), time, math.inf))
    return labels, data[order]


def report_history(path, requested_labels, arguments):
    labels, data = read_named_history(path)
    time = data[:, labels["time"]]
    selected = (
        np.isfinite(time)
        &
        (time + 1.0e-12 >= arguments.start_time)
        & (time <= arguments.end_time + 1.0e-12)
    )
    if not np.any(selected):
        raise SystemExit("{}: no history samples in selected interval".format(
            path))
    selected_time = time[selected]
    selected_data = data[selected]
    if not np.all(np.isfinite(selected_data)):
        raise SystemExit(
            "{}: nonfinite history value in selected interval [{}, {}]".
            format(path, arguments.start_time, arguments.end_time))
    for label in requested_labels:
        if label not in labels:
            raise SystemExit("{}: missing history column {}".format(
                path, label))
        values = selected_data[:, labels[label]]
        index = int(np.argmax(np.abs(values)))
        print(
            "HISTORY file={} label={} maximum_abs={:.16e} "
            "signed_value={:.16e} time={:.16e} first={:.16e} "
            "last={:.16e} samples={}".format(
                path.name, label, abs(float(values[index])),
                float(values[index]), float(selected_time[index]),
                float(values[0]), float(values[-1]), len(values)))
    return labels, selected_data


def report_run_diagnostics(arguments):
    problem_histories = [
        path for path in sorted(arguments.run.glob("*.user.hst"))
        if not path.name.endswith(".z4c.user.hst")
    ]
    constraint_histories = sorted(arguments.run.glob("*.z4c.user.hst"))
    if len(problem_histories) != 1 or len(constraint_histories) != 1:
        raise SystemExit(
            "{}: expected one problem and one Z4c history".format(
                arguments.run))
    problem_labels, problem_data = report_history(
        problem_histories[0], PROBLEM_HISTORY_LABELS, arguments)
    report_history(
        constraint_histories[0], CONSTRAINT_HISTORY_LABELS, arguments)
    bad_metric = problem_data[:, problem_labels["bad-metric"]]
    if np.any(bad_metric != 0.0):
        raise SystemExit("{}: nonzero bad-metric history value".format(
            problem_histories[0]))

    track_time, track_position, stellar_radius = comparator.load_track(
        arguments.run)
    selected = (
        (track_time + 1.0e-12 >= arguments.start_time)
        & (track_time <= arguments.end_time + 1.0e-12)
    )
    if np.count_nonzero(selected) < 2:
        raise SystemExit("{}: fewer than two selected STAR_TRACK rows".format(
            arguments.run))
    times = track_time[selected]
    positions = track_position[selected]
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    index = int(np.argmax(displacement))
    print(
        "STAR_TRACK samples={} stellar_radius={:.16e} "
        "maximum_selected_displacement={:.16e} time={:.16e} "
        "x={:.16e} y={:.16e} z={:.16e}".format(
            len(times), stellar_radius, float(displacement[index]),
            float(times[index]), float(positions[index, 0]),
            float(positions[index, 1]), float(positions[index, 2])))

    stdout = arguments.run / "athena.stdout"
    if not stdout.exists():
        stdout = arguments.run / "stdout.log"
    if not stdout.exists():
        raise SystemExit("{}: missing Athena stdout".format(arguments.run))
    error_matches = 0
    with stdout.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if ERROR_PATTERN.search(line) is not None:
                error_matches += 1
    print("ERROR_SCAN file={} matches={}".format(
        stdout.name, error_matches))
    if error_matches:
        raise SystemExit("{}: fatal/MPI/SYCL error signature found".format(
            stdout))


def analyze_slice(path, arguments, summary, field_summary):
    time, blocks = comparator.read_slice_binary(path, arguments)
    totals = {
        (direction, mode): {"sum_squares": 0.0, "points": 0, "linf": 0.0,
                            "coordinates": None}
        for direction in ("incoming", "outgoing")
        for mode in MODE_NAMES
    }
    field_totals = {
        name: {"sum_squares": 0.0, "points": 0, "linf": 0.0,
               "coordinates": None}
        for name in TRACKED_FIELDS
    }
    selected_blocks = 0
    row_min, row_max = row_bounds(arguments)
    for block in blocks:
        if not (
            block["column_max"] > arguments.slice_x_min
            and block["column_min"] < arguments.slice_x_max
            and block["row_max"] > row_min
            and block["row_min"] < row_max
        ):
            continue
        incoming, outgoing, grid_column, grid_row = (
            comparator.characteristic_fields(
                block, arguments, include_outgoing=True))
        if set(incoming) != set(MODE_NAMES) or set(outgoing) != set(MODE_NAMES):
            raise SystemExit("{}: characteristic mode set is incomplete".format(
                path))
        mask = (
            (grid_column >= arguments.slice_x_min)
            & (grid_column < arguments.slice_x_max)
            & (grid_row >= row_min)
            & (grid_row < row_max)
        )
        margin = arguments.characteristic_block_margin
        if margin > 0:
            mask[:margin, :] = False
            mask[-margin:, :] = False
            mask[:, :margin] = False
            mask[:, -margin:] = False
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        selected_blocks += 1
        for direction, modes in (
            ("incoming", incoming), ("outgoing", outgoing)
        ):
            for mode in MODE_NAMES:
                values = modes[mode]
                selected = values[mask]
                if not np.all(np.isfinite(selected)):
                    raise SystemExit(
                        "{}: nonfinite {} {} characteristic".format(
                            path, direction, mode))
                metric = totals[(direction, mode)]
                metric["sum_squares"] += float(np.sum(selected * selected))
                metric["points"] += count
                local_index = int(np.argmax(np.abs(selected)))
                local_linf = float(abs(selected[local_index]))
                if (
                    metric["coordinates"] is None
                    or local_linf > metric["linf"]
                ):
                    candidate_indices = np.argwhere(mask)
                    row_index, column_index = candidate_indices[local_index]
                    metric["linf"] = local_linf
                    metric["coordinates"] = physical_coordinates(
                        float(grid_column[row_index, column_index]),
                        float(grid_row[row_index, column_index]),
                        arguments,
                    )
        for name in TRACKED_FIELDS:
            values = np.asarray(block["fields"][name], dtype=np.float64)
            selected = values[mask]
            if not np.all(np.isfinite(selected)):
                raise SystemExit(
                    "{}: nonfinite residual field {}".format(path, name))
            metric = field_totals[name]
            metric["sum_squares"] += float(np.sum(selected * selected))
            metric["points"] += count
            local_index = int(np.argmax(np.abs(selected)))
            local_linf = float(abs(selected[local_index]))
            if (
                metric["coordinates"] is None
                or local_linf > metric["linf"]
            ):
                candidate_indices = np.argwhere(mask)
                row_index, column_index = candidate_indices[local_index]
                metric["linf"] = local_linf
                metric["coordinates"] = physical_coordinates(
                    float(grid_column[row_index, column_index]),
                    float(grid_row[row_index, column_index]),
                    arguments,
                )
    if selected_blocks == 0:
        raise SystemExit("{}: no characteristic sampling points".format(path))

    minimum_points = min(metric["points"] for metric in totals.values())
    print(
        "SLICE time={:.16e} file={} blocks={} points={}".format(
            time, path.name, selected_blocks, minimum_points))
    for direction in ("incoming", "outgoing"):
        for mode in MODE_NAMES:
            metric = totals[(direction, mode)]
            rms = math.sqrt(metric["sum_squares"] / metric["points"])
            coordinates = metric["coordinates"]
            print(
                "MODE time={:.16e} direction={} mode={} rms={:.16e} "
                "linf={:.16e} x={:.16e} y={:.16e} z={:.16e}".format(
                    time, direction, mode, rms, metric["linf"],
                    coordinates[0], coordinates[1], coordinates[2]))
            update_global(
                summary, (direction, mode), time, rms, metric["linf"],
                coordinates)
    for name in TRACKED_FIELDS:
        metric = field_totals[name]
        rms = math.sqrt(metric["sum_squares"] / metric["points"])
        coordinates = metric["coordinates"]
        print(
            "FIELD time={:.16e} name={} rms={:.16e} linf={:.16e} "
            "x={:.16e} y={:.16e} z={:.16e}".format(
                time, name, rms, metric["linf"],
                coordinates[0], coordinates[1], coordinates[2]))
        update_global(
            field_summary, name, time, rms, metric["linf"], coordinates)
    return time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=math.inf)
    parser.add_argument("--slice-x-min", type=float, default=32.0)
    parser.add_argument("--slice-x-max", type=float, default=45.0)
    parser.add_argument("--slice-y-half-width", type=float, default=6.0)
    parser.add_argument("--slice-row-min", type=float)
    parser.add_argument("--slice-row-max", type=float)
    parser.add_argument("--slice-id", default="xz_z4c")
    parser.add_argument(
        "--slice-plane", choices=tuple(comparator.PLANE_AXES), default="xz")
    parser.add_argument("--slice-fixed-coordinate", type=float, default=0.0)
    parser.add_argument("--background-mass", type=float, default=1.0)
    parser.add_argument("--background-spin", type=float, default=0.0)
    parser.add_argument("--background-center-x", type=float, default=0.0)
    parser.add_argument("--background-center-y", type=float, default=0.0)
    parser.add_argument("--background-center-z", type=float, default=0.0)
    parser.add_argument(
        "--boundary-axis", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--boundary-side", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--residual-lapse-f", type=float, default=1.0)
    parser.add_argument("--lapse-oplog", type=float, default=2.0)
    parser.add_argument("--lapse-harmonicf", type=float, default=1.0)
    parser.add_argument("--lapse-harmonic", type=float, default=0.0)
    parser.add_argument("--shift-driver", type=float, default=1.0)
    parser.add_argument("--characteristic-block-margin", type=int, default=2)
    parser.add_argument(
        "--maximum-unresolved-normal", type=float, default=1.0e-7)
    args = parser.parse_args()

    finite_parameters = (
        args.start_time, args.slice_x_min, args.slice_x_max,
        args.slice_y_half_width, args.slice_fixed_coordinate,
        args.background_mass, args.background_spin,
        args.background_center_x, args.background_center_y,
        args.background_center_z, args.residual_lapse_f,
        args.lapse_oplog, args.lapse_harmonicf, args.lapse_harmonic,
        args.shift_driver, args.maximum_unresolved_normal,
    )
    if (
        not all(math.isfinite(value) for value in finite_parameters)
        or math.isnan(args.end_time)
        or args.end_time < args.start_time
        or not args.slice_x_min < args.slice_x_max
        or args.slice_y_half_width <= 0.0
        or args.background_mass < 0.0
        or abs(args.background_spin) > args.background_mass
        or args.residual_lapse_f <= 0.0
        or args.shift_driver <= 0.0
        or args.characteristic_block_margin < 1
        or args.maximum_unresolved_normal <= 0.0
    ):
        raise SystemExit("invalid boundary-series analysis parameter")
    if (args.slice_row_min is None) != (args.slice_row_max is None):
        raise SystemExit(
            "slice-row-min and slice-row-max must be provided together")
    if args.slice_row_min is not None and (
        not math.isfinite(args.slice_row_min)
        or not math.isfinite(args.slice_row_max)
        or not args.slice_row_min < args.slice_row_max
    ):
        raise SystemExit("invalid slice row bounds")
    if args.boundary_axis - 1 not in comparator.PLANE_AXES[args.slice_plane]:
        raise SystemExit(
            "boundary axis {} is not resolved by the {} slice".format(
                args.boundary_axis, args.slice_plane))

    series = [
        (time, path)
        for time, path in comparator.slice_series(args.run, args)
        if time + 1.0e-12 >= args.start_time
        and time <= args.end_time + 1.0e-12
    ]
    if not series:
        raise SystemExit(
            "{}: no residual slices in [{}, {}]".format(
                args.run, args.start_time, args.end_time))
    summary = {
        (direction, mode): {
            "rms": -1.0, "rms_time": math.nan,
            "linf": -1.0, "linf_time": math.nan,
            "coordinates": None,
        }
        for direction in ("incoming", "outgoing")
        for mode in MODE_NAMES
    }
    field_summary = {
        name: {
            "rms": -1.0, "rms_time": math.nan,
            "linf": -1.0, "linf_time": math.nan,
            "coordinates": None,
        }
        for name in TRACKED_FIELDS
    }
    observed_times = [
        analyze_slice(path, args, summary, field_summary)
        for _, path in series
    ]
    print(
        "SERIES run={} samples={} first_time={:.16e} last_time={:.16e} "
        "modes={} directions=2".format(
            args.run, len(observed_times), observed_times[0],
            observed_times[-1], len(MODE_NAMES)))
    for direction in ("incoming", "outgoing"):
        for mode in MODE_NAMES:
            record = summary[(direction, mode)]
            coordinates = record["coordinates"]
            print(
                "SUMMARY direction={} mode={} maximum_rms={:.16e} "
                "rms_time={:.16e} maximum_linf={:.16e} "
                "linf_time={:.16e} x={:.16e} y={:.16e} z={:.16e}".format(
                    direction, mode, record["rms"], record["rms_time"],
                    record["linf"], record["linf_time"],
                    coordinates[0], coordinates[1], coordinates[2]))
    for name in TRACKED_FIELDS:
        record = field_summary[name]
        coordinates = record["coordinates"]
        print(
            "FIELD_SUMMARY name={} maximum_rms={:.16e} "
            "rms_time={:.16e} maximum_linf={:.16e} "
            "linf_time={:.16e} x={:.16e} y={:.16e} z={:.16e}".format(
                name, record["rms"], record["rms_time"],
                record["linf"], record["linf_time"],
                coordinates[0], coordinates[1], coordinates[2]))
    report_run_diagnostics(args)


if __name__ == "__main__":
    main()
