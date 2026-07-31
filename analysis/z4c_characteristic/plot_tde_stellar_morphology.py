#!/usr/bin/env python3
"""Plot and measure the orbital-plane density morphology of the TDE star.

The Athena ``xy_mhd`` files contain two-dimensional AMR leaf blocks.  This
script measures the half-maximum core directly on those leaves (with cell-area
weights), and only rasterizes onto a uniform, peak-centered grid for plotting
and the explicitly labelled normalized-shape L2 diagnostic.
"""

import argparse
import csv
import math
import pathlib
import shutil
import struct
import subprocess
import tempfile

import numpy as np


def preamble(stream, path):
    if stream.readline().decode("ascii") != "Athena binary output version=1.1\n":
        raise SystemExit("{}: unsupported binary format".format(path))
    stream.readline()
    time = float(stream.readline().decode("ascii").split("=", 1)[1])
    stream.readline()
    location_size = int(
        stream.readline().decode("ascii").split("=", 1)[1])
    variable_size = int(
        stream.readline().decode("ascii").split("=", 1)[1])
    stream.readline()
    names = stream.readline().decode("ascii").split(":", 1)[1].split()
    header_offset = int(
        stream.readline().decode("ascii").split("=", 1)[1])
    return time, location_size, variable_size, names, header_offset


def binary_time(path):
    with path.open("rb") as stream:
        return preamble(stream, path)[0]


def read_xy_density(path):
    """Return time and independent xy AMR leaves containing density."""
    blocks = []
    with path.open("rb") as stream:
        time, location_size, variable_size, names, header_offset = preamble(
            stream, path)
        if "dens" not in names:
            raise SystemExit("{}: missing dens".format(path))
        density_index = names.index("dens")
        location_code = "f" if location_size == 4 else "d"
        variable_dtype = np.dtype("=f4" if variable_size == 4 else "=f8")
        data_start = stream.tell() + header_offset
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(data_start)
        while stream.tell() < file_size:
            raw_indices = stream.read(24)
            if not raw_indices:
                break
            if len(raw_indices) != 24:
                raise SystemExit("{}: truncated indices".format(path))
            indices = struct.unpack("=6i", raw_indices)
            logical = struct.unpack("=4i", stream.read(16))
            limits = struct.unpack(
                "=6" + location_code, stream.read(6 * location_size))
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            if nx <= 0 or ny <= 0 or nz != 1:
                raise SystemExit("{}: not an xy slice".format(path))
            count = len(names) * nx * ny
            raw = stream.read(count * variable_size)
            if len(raw) != count * variable_size:
                raise SystemExit("{}: truncated data".format(path))
            data = np.frombuffer(raw, dtype=variable_dtype).reshape(
                (len(names), 1, ny, nx))
            density = np.asarray(data[density_index, 0], dtype=np.float64)
            if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
                raise SystemExit("{}: invalid density".format(path))
            blocks.append({
                "level": logical[3],
                "xmin": limits[0], "xmax": limits[1],
                "ymin": limits[2], "ymax": limits[3],
                "density": density,
            })
    if not blocks:
        raise SystemExit("{}: no blocks".format(path))
    return time, blocks


def block_coordinates(block):
    ny, nx = block["density"].shape
    dx = (block["xmax"] - block["xmin"]) / nx
    dy = (block["ymax"] - block["ymin"]) / ny
    x = block["xmin"] + (np.arange(nx) + 0.5) * dx
    y = block["ymin"] + (np.arange(ny) + 0.5) * dy
    return x, y, dx, dy


def morphology(time, blocks):
    peak = -math.inf
    peak_x = peak_y = math.nan
    for block in blocks:
        local = int(np.argmax(block["density"]))
        iy, ix = np.unravel_index(local, block["density"].shape)
        value = float(block["density"][iy, ix])
        if value > peak:
            x, y, _, _ = block_coordinates(block)
            peak, peak_x, peak_y = value, float(x[ix]), float(y[iy])
    threshold = 0.5 * peak
    area = mass = 0.0
    sx = sy = 0.0
    selected = []
    for block in blocks:
        x, y, dx, dy = block_coordinates(block)
        xx, yy = np.meshgrid(x, y)
        mask = block["density"] >= threshold
        if not np.any(mask):
            continue
        cell_area = dx * dy
        rho = block["density"][mask]
        weights = rho * cell_area
        area += int(np.count_nonzero(mask)) * cell_area
        mass += float(np.sum(weights))
        sx += float(np.sum(weights * xx[mask]))
        sy += float(np.sum(weights * yy[mask]))
        selected.append((xx[mask], yy[mask], weights))
    cx, cy = sx / mass, sy / mass
    cxx = cyy = cxy = 0.0
    for x, y, weights in selected:
        cxx += float(np.sum(weights * (x - cx) ** 2))
        cyy += float(np.sum(weights * (y - cy) ** 2))
        cxy += float(np.sum(weights * (x - cx) * (y - cy)))
    covariance = np.array([[cxx, cxy], [cxy, cyy]]) / mass
    eigenvalues = np.linalg.eigvalsh(covariance)
    axis_ratio = math.sqrt(max(eigenvalues) / min(eigenvalues))
    effective_radii = {}
    for fraction in (0.1, 0.01):
        fraction_area = 0.0
        for block in blocks:
            _, _, dx, dy = block_coordinates(block)
            fraction_area += (
                int(np.count_nonzero(block["density"] >= fraction * peak))
                * dx * dy)
        effective_radii[fraction] = math.sqrt(fraction_area / math.pi)
    return {
        "time": time, "rho_peak": peak,
        "peak_x": peak_x, "peak_y": peak_y,
        "centroid_x": cx, "centroid_y": cy,
        "halfmax_area": area,
        "halfmax_effective_radius": math.sqrt(area / math.pi),
        "tenthmax_effective_radius": effective_radii[0.1],
        "onepercent_effective_radius": effective_radii[0.01],
        "halfmax_axis_ratio": axis_ratio,
    }


def raster(blocks, center, half_width, spacing):
    coordinates = np.arange(
        -half_width + 0.5 * spacing, half_width, spacing)
    uu, vv = np.meshgrid(coordinates, coordinates)
    xx, yy = uu + center[0], vv + center[1]
    result = np.full_like(xx, np.nan, dtype=np.float64)
    for block in sorted(blocks, key=lambda item: item["level"]):
        density = block["density"]
        ny, nx = density.shape
        dx = (block["xmax"] - block["xmin"]) / nx
        dy = (block["ymax"] - block["ymin"]) / ny
        inside = (
            (xx >= block["xmin"]) & (xx < block["xmax"])
            & (yy >= block["ymin"]) & (yy < block["ymax"])
        )
        if not np.any(inside):
            continue
        ix = np.floor((xx[inside] - block["xmin"]) / dx).astype(int)
        iy = np.floor((yy[inside] - block["ymin"]) / dy).astype(int)
        result[inside] = density[iy, ix]
    if np.any(~np.isfinite(result)):
        raise SystemExit("centered raster extends outside slice coverage")
    return coordinates, result


def shape_l2(value, reference):
    a = value / np.max(value)
    b = reference / np.max(reference)
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))


def choose(series, target):
    return min(series, key=lambda item: abs(item[0] - target))


def write_grid(path, coordinates, density):
    with path.open("w", encoding="utf-8") as stream:
        for iy, y in enumerate(coordinates):
            for ix, x in enumerate(coordinates):
                stream.write("{:.9e} {:.9e} {:.9e}\n".format(
                    x, y, density[iy, ix]))
            stream.write("\n")


def write_fractional_contours(path, coordinates, density, peak):
    """Write unconnected marching-square segments for 50/10/1% contours."""
    with path.open("w", encoding="utf-8") as stream:
        for fraction in (0.5, 0.1, 0.01):
            level = fraction * peak
            for iy in range(len(coordinates) - 1):
                for ix in range(len(coordinates) - 1):
                    points = (
                        (coordinates[ix], coordinates[iy]),
                        (coordinates[ix + 1], coordinates[iy]),
                        (coordinates[ix + 1], coordinates[iy + 1]),
                        (coordinates[ix], coordinates[iy + 1]),
                    )
                    values = (
                        density[iy, ix], density[iy, ix + 1],
                        density[iy + 1, ix + 1], density[iy + 1, ix],
                    )
                    crossings = []
                    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                        v0, v1 = values[first], values[second]
                        if (v0 < level) == (v1 < level) or v0 == v1:
                            continue
                        weight = (level - v0) / (v1 - v0)
                        x0, y0 = points[first]
                        x1, y1 = points[second]
                        crossings.append((
                            x0 + weight * (x1 - x0),
                            y0 + weight * (y1 - y0),
                        ))
                    for index in range(0, len(crossings) - 1, 2):
                        stream.write("{:.9e} {:.9e}\n".format(
                            *crossings[index]))
                        stream.write("{:.9e} {:.9e}\n\n".format(
                            *crossings[index + 1]))


def gnuplot_quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpbc-run", type=pathlib.Path, required=True)
    parser.add_argument("--sommerfeld-run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metrics-csv", type=pathlib.Path, required=True)
    parser.add_argument("--half-width", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=0.0234375)
    arguments = parser.parse_args()
    cpbc_paths = sorted((arguments.cpbc_run / "bin").glob("*.xy_mhd.*.bin"))
    sommerfeld_paths = sorted(
        (arguments.sommerfeld_run / "bin").glob("*.xy_mhd.*.bin"))
    if not cpbc_paths or not sommerfeld_paths:
        raise SystemExit("missing xy_mhd slices")
    cpbc_series = sorted((binary_time(path), path) for path in cpbc_paths)
    sommerfeld_series = sorted(
        (binary_time(path), path) for path in sommerfeld_paths)
    selected_cpbc = [choose(cpbc_series, target) for target in (0, 7.003125, 15, 30)]
    selected_sommerfeld = choose(sommerfeld_series, 7.003125)

    metrics_rows = []
    selected = {}
    initial_raster = None
    with tempfile.TemporaryDirectory(prefix="tde-morphology-") as directory:
        temporary = pathlib.Path(directory)
        # All CPBC slices make "throughout 30 M" a measured time series.
        for time, path in cpbc_series:
            read_time, blocks = read_xy_density(path)
            metric = morphology(read_time, blocks)
            coordinates, density = raster(
                blocks, (metric["peak_x"], metric["peak_y"]),
                arguments.half_width, arguments.spacing)
            if initial_raster is None:
                initial_raster = density
            metric["normalized_centered_shape_l2_vs_cpbc_t0"] = shape_l2(
                density, initial_raster)
            metric["case"] = "CPBC"
            metrics_rows.append(metric)
            if any(path == candidate[1] for candidate in selected_cpbc):
                selected[("CPBC", path)] = (
                    metric, coordinates, density.copy())

        time, path = selected_sommerfeld
        read_time, blocks = read_xy_density(path)
        metric = morphology(read_time, blocks)
        coordinates, density = raster(
            blocks, (metric["peak_x"], metric["peak_y"]),
            arguments.half_width, arguments.spacing)
        metric["normalized_centered_shape_l2_vs_cpbc_t0"] = shape_l2(
            density, initial_raster)
        metric["case"] = "Sommerfeld"
        # No CPBC density slice exists at the selected Sommerfeld time.
        # Retain the nearest stored CPBC slice and report its exact time.
        cpbc_match_path = choose(cpbc_series, read_time)[1]
        cpbc_match = selected[("CPBC", cpbc_match_path)][2]
        metric["normalized_centered_shape_l2_vs_nearest_cpbc"] = shape_l2(
            density, cpbc_match)
        metrics_rows.append(metric)
        selected[("Sommerfeld", path)] = (
            metric, coordinates, density.copy())

        fieldnames = (
            "case", "time", "rho_peak", "peak_x", "peak_y",
            "centroid_x", "centroid_y", "halfmax_area",
            "halfmax_effective_radius", "halfmax_axis_ratio",
            "tenthmax_effective_radius", "onepercent_effective_radius",
            "normalized_centered_shape_l2_vs_cpbc_t0",
            "normalized_centered_shape_l2_vs_nearest_cpbc",
        )
        arguments.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with arguments.metrics_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

        plot_order = [
            ("CPBC", selected_cpbc[0][1], "CPBC"),
            ("CPBC", selected_cpbc[1][1], "CPBC"),
            ("Sommerfeld", selected_sommerfeld[1], "Sommerfeld"),
            ("CPBC", selected_cpbc[2][1], "CPBC"),
            ("CPBC", selected_cpbc[3][1], "CPBC"),
        ]
        data_paths = []
        maximum = max(float(np.max(selected[(case, path)][2]))
                      for case, path, _ in plot_order)
        minimum = maximum * 1.0e-5
        for index, (case, path, title) in enumerate(plot_order):
            metric_value, coordinates, density = selected[(case, path)]
            data_path = temporary / "panel{}.dat".format(index)
            contour_path = temporary / "contour{}.dat".format(index)
            write_grid(data_path, coordinates, density)
            write_fractional_contours(
                contour_path, coordinates, density, metric_value["rho_peak"])
            data_paths.append((data_path, contour_path, title, metric_value))
        history_path = temporary / "history.dat"
        cpbc_rows = [row for row in metrics_rows if row["case"] == "CPBC"]
        with history_path.open("w", encoding="utf-8") as stream:
            for row in cpbc_rows:
                stream.write(
                    "{time:.9e} {rho_peak:.9e} "
                    "{halfmax_effective_radius:.9e} "
                    "{halfmax_axis_ratio:.9e} "
                    "{normalized_centered_shape_l2_vs_cpbc_t0:.9e}\n".
                    format(**row))

        if shutil.which("gnuplot") is None:
            raise SystemExit("gnuplot not found (load the gnuplot module)")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        commands = [
            "set terminal pdfcairo enhanced color font 'Helvetica,9' "
            "size 11in,7.5in",
            "set output {}".format(gnuplot_quote(arguments.output)),
            "set multiplot layout 2,3 margins 0.06,0.98,0.08,0.95 "
            "spacing 0.06,0.11",
            "set size ratio -1",
            "set xrange [-1:1]",
            "set yrange [-1:1]",
            "set xlabel 'x-x_peak [M]'",
            "set ylabel 'y-y_peak [M]'",
            "set logscale cb",
            "set format cb '10^{%L}'",
            "set cbrange [{:.12e}:{:.12e}]".format(minimum, maximum),
            "set palette defined (0 '#081d58', 0.35 '#225ea8', "
            "0.65 '#41b6c4', 0.82 '#c7e9b4', 1 '#ffffcc')",
        ]
        for index, (data_path, contour_path, title, metric_value) in enumerate(
                data_paths):
            commands.extend([
                "set title '{} t={:.5f} M; "
                "rho_max={:.3e}, R_1/2={:.3f}, a/b={:.3f}'".format(
                    title, metric_value["time"], metric_value["rho_peak"],
                    metric_value["halfmax_effective_radius"],
                    metric_value["halfmax_axis_ratio"]),
                ("set colorbox vertical" if index == 4
                 else "unset colorbox"),
                "plot {} using 1:2:3 with image notitle, "
                "{} using 1:2 with lines lc rgb 'white' lw 0.7 notitle".format(
                    gnuplot_quote(data_path), gnuplot_quote(contour_path)),
            ])
        commands.extend([
            "unset logscale cb",
            "unset colorbox",
            "set size ratio 0.5",
            "set title 'CPBC orbital-plane morphology through 30 M'",
            "set xrange [0:30]",
            "set yrange [0:1.2]",
            "set xlabel 't [M]'",
            "set ylabel 'normalized metric'",
            "set key top right",
            "r0={:.12e}".format(cpbc_rows[0]["halfmax_effective_radius"]),
            "plot {} using 1:($3/r0) with lines lw 2 title 'R_1/2/R_1/2(0)', "
            "{} using 1:4 with lines lw 2 title 'axis ratio a/b', "
            "{} using 1:5 with lines lw 2 title 'centered shape L2 vs t=0'".
            format(*(gnuplot_quote(history_path),) * 3),
            "unset multiplot",
        ])
        script_path = temporary / "plot.gnuplot"
        script_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
        subprocess.run(["gnuplot", str(script_path)], check=True)

    for row in metrics_rows:
        if row["case"] == "Sommerfeld" or any(
                abs(row["time"] - target) < 1.0e-4
                for target in (0, 7.003125, 15, 30)):
            print("METRIC " + " ".join(
                "{}={}".format(name, row.get(name, ""))
                for name in fieldnames))


if __name__ == "__main__":
    main()
