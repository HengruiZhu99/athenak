#!/usr/bin/env python3
"""Compare small-boundary TDE runs against a far-boundary reference.

The acceptance observable is the full-metric, background-shift-adapted
projection of all eight incoming gauge and constraint characteristics on
matched leaf cells after the configured fastest-gauge round-trip time.
Direct residual-component differences on a fixed physical slice, global
histories, density, and STAR_TRACK trajectories are retained as independent
secondary diagnostics.
"""

import argparse
import math
import pathlib
import re
import struct

import numpy as np


TRACK_PATTERN = re.compile(
    r"STAR_TRACK time=(?P<time>\S+) "
    r"x=(?P<x>\S+) y=(?P<y>\S+) z=(?P<z>\S+) .*?"
    r"rho_max=(?P<rho>\S+) valid=(?P<valid>\S+)"
)
ISOTROPIC_RADIUS_PATTERN = re.compile(
    r"Radius \(Isotropic\):\s*(?P<radius>\S+)"
)
REQUIRED_RETURN_LABELS = (
    "Theta-max",
    "alpha-res",
    "beta-res",
    "Gam-res",
)
OPTIONAL_RETURN_LABELS = ("Khat-res", "res-inner")
Z4C_VARIABLES = (
    "z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy",
    "z4c_gyz", "z4c_gzz", "z4c_Khat", "z4c_Axx", "z4c_Axy",
    "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz", "z4c_Gamx",
    "z4c_Gamy", "z4c_Gamz", "z4c_Theta", "z4c_alpha",
    "z4c_betax", "z4c_betay", "z4c_betaz", "z4c_Bx", "z4c_By",
    "z4c_Bz",
)
SLICE_GROUPS = {
    "gauge": (
        "z4c_Khat", "z4c_alpha", "z4c_betax", "z4c_betay",
        "z4c_betaz", "z4c_Bx", "z4c_By", "z4c_Bz",
    ),
    # Theta is a Z4 constraint field.  The evolved conformal connection
    # variables also carry the Z_i/connection-constraint sector.
    "constraint": (
        "z4c_Theta", "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
    ),
}
CHARACTERISTIC_GROUPS = {
    "gauge": (
        "lapse", "shift_longitudinal",
        "shift_transverse_1", "shift_transverse_2",
    ),
    "constraint": (
        "constraint_scalar_theta", "constraint_scalar_z",
        "constraint_transverse_1", "constraint_transverse_2",
    ),
}


def symmetric_name(prefix, first, second):
    labels = ("x", "y", "z")
    if first > second:
        first, second = second, first
    return prefix + labels[first] + labels[second]


def load_history(root):
    matches = [
        path for path in sorted(root.glob("*.user.hst"))
        if not path.name.endswith(".z4c.user.hst")
    ]
    if len(matches) != 1:
        raise SystemExit(
            "{}: expected one problem-generator user history, found {}".format(
                root, len(matches)))
    path = matches[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit("{}: incomplete history".format(path))
    labels = {}
    for match in re.finditer(r"\[(\d+)\]=(\S+)", lines[1]):
        labels[match.group(2)] = int(match.group(1)) - 1
    required = ("time", "rho-max") + REQUIRED_RETURN_LABELS
    missing = [label for label in required if label not in labels]
    if missing:
        raise SystemExit(
            "{}: missing history columns {}".format(
                path, ", ".join(missing)))
    rows = []
    for line in lines[2:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = [float(value) for value in line.split()]
        if not all(math.isfinite(value) for value in row):
            raise SystemExit("{}: nonfinite history row".format(path))
        rows.append(row)
    if not rows:
        raise SystemExit("{}: no history rows".format(path))
    data = np.asarray(rows)
    order = np.argsort(data[:, labels["time"]])
    return {
        label: data[order, index] for label, index in labels.items()
    }


def load_track(root):
    path = root / "athena.stdout"
    if not path.exists():
        path = root / "stdout.log"
    if not path.exists():
        raise SystemExit("{}: missing Athena stdout".format(root))
    text = path.read_text(encoding="utf-8")
    radius_matches = ISOTROPIC_RADIUS_PATTERN.findall(text)
    if len(radius_matches) != 1:
        raise SystemExit(
            "{}: expected one initialized isotropic stellar radius".format(path))
    stellar_radius = float(radius_matches[0])
    if not math.isfinite(stellar_radius) or stellar_radius <= 0.0:
        raise SystemExit("{}: invalid isotropic stellar radius".format(path))
    rows = []
    for match in TRACK_PATTERN.finditer(text):
        values = {
            key: float(match.group(key))
            for key in ("time", "x", "y", "z", "rho")
        }
        valid = float(match.group("valid"))
        if not all(math.isfinite(value) for value in values.values()):
            raise SystemExit("{}: nonfinite STAR_TRACK row".format(path))
        if not math.isfinite(valid) or valid != 1.0:
            raise SystemExit("{}: invalid STAR_TRACK row".format(path))
        if values["rho"] <= 0.0:
            raise SystemExit(
                "{}: nonpositive STAR_TRACK density".format(path))
        rows.append((
            values["time"], values["x"], values["y"], values["z"],
            values["rho"],
        ))
    if len(rows) < 2:
        raise SystemExit("{}: insufficient valid STAR_TRACK rows".format(path))
    data = np.asarray(rows)
    order = np.argsort(data[:, 0])
    data = data[order]
    unique_time, unique_indices = np.unique(data[:, 0], return_index=True)
    return unique_time, data[unique_indices, 1:4], stellar_radius


def common_samples(reference_time, case_time, start):
    lower = max(float(reference_time[0]), float(case_time[0]), start)
    upper = min(float(reference_time[-1]), float(case_time[-1]))
    selected = reference_time[
        (reference_time >= lower) & (reference_time <= upper)]
    if len(selected) < 2:
        raise SystemExit(
            "insufficient common samples in [{:.6g}, {:.6g}]".format(
                lower, upper))
    return selected


def history_differences(reference, case, start, return_labels):
    samples = common_samples(reference["time"], case["time"], start)
    differences = {}
    for label in return_labels:
        ref = np.interp(samples, reference["time"], reference[label])
        value = np.interp(samples, case["time"], case[label])
        differences[label] = float(np.max(np.abs(value - ref)))
    rho_ref = np.interp(samples, reference["time"], reference["rho-max"])
    rho_case = np.interp(samples, case["time"], case["rho-max"])
    rho_scale = max(float(np.max(np.abs(rho_ref))), 1.0e-300)
    density_difference = float(np.max(np.abs(rho_case - rho_ref))) / rho_scale
    return differences, density_difference


def trajectory_difference(reference, case, start):
    ref_time, ref_position = reference[:2]
    case_time, case_position = case[:2]
    samples = common_samples(ref_time, case_time, start)
    ref = np.column_stack([
        np.interp(samples, ref_time, ref_position[:, component])
        for component in range(3)
    ])
    value = np.column_stack([
        np.interp(samples, case_time, case_position[:, component])
        for component in range(3)
    ])
    return float(np.max(np.linalg.norm(value - ref, axis=1)))


def parse_binary_preamble(stream, path):
    if stream.readline().decode("ascii") != (
            "Athena binary output version=1.1\n"):
        raise SystemExit("{}: unsupported binary format".format(path))
    stream.readline()
    time_line = stream.readline().decode("ascii")
    stream.readline()
    location_line = stream.readline().decode("ascii")
    variable_line = stream.readline().decode("ascii")
    stream.readline()
    names_line = stream.readline().decode("ascii")
    header_line = stream.readline().decode("ascii")
    try:
        time = float(time_line.split("=", 1)[1])
        location_size = int(location_line.split("=", 1)[1])
        variable_size = int(variable_line.split("=", 1)[1])
        names = names_line.split(":", 1)[1].split()
        header_offset = int(header_line.split("=", 1)[1])
    except (IndexError, ValueError) as error:
        raise SystemExit(
            "{}: malformed binary preamble: {}".format(path, error))
    if location_size not in (4, 8) or variable_size not in (4, 8):
        raise SystemExit("{}: unsupported floating-point size".format(path))
    return time, location_size, variable_size, names, header_offset


def binary_time(path):
    with path.open("rb") as stream:
        return parse_binary_preamble(stream, path)[0]


PLANE_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def read_slice_binary(path, arguments):
    """Read an AMR residual slice as independent two-dimensional leaf blocks."""
    blocks = []
    expected_axes = PLANE_AXES[arguments.slice_plane]
    fixed_axis = ({0, 1, 2} - set(expected_axes)).pop()
    with path.open("rb") as stream:
        time, location_size, variable_size, names, header_offset = (
            parse_binary_preamble(stream, path))
        missing = [name for name in Z4C_VARIABLES if name not in names]
        if missing:
            raise SystemExit(
                "{}: missing residual variables {}".format(
                    path, ", ".join(missing)))
        data_start = stream.tell() + header_offset
        location_code = "f" if location_size == 4 else "d"
        variable_dtype = np.dtype("=f4" if variable_size == 4 else "=f8")
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(data_start)
        while stream.tell() < file_size:
            raw_indices = stream.read(24)
            if not raw_indices:
                break
            if len(raw_indices) != 24:
                raise SystemExit("{}: truncated block indices".format(path))
            indices = struct.unpack("=6i", raw_indices)
            raw_location = stream.read(16)
            raw_limits = stream.read(6 * location_size)
            if (len(raw_location) != 16 or
                    len(raw_limits) != 6 * location_size):
                raise SystemExit("{}: truncated block header".format(path))
            logical_location = struct.unpack("=4i", raw_location)
            limits = struct.unpack("=6" + location_code, raw_limits)
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            sizes = (nx, ny, nz)
            if (
                any(size <= 0 for size in sizes)
                or sizes[fixed_axis] != 1
                or any(sizes[axis] <= 1 for axis in expected_axes)
            ):
                raise SystemExit(
                    "{}: expected a {} slice, got {}x{}x{}".format(
                        path, arguments.slice_plane, nx, ny, nz))
            count = len(names) * nx * ny * nz
            raw_data = stream.read(count * variable_size)
            if len(raw_data) != count * variable_size:
                raise SystemExit("{}: truncated block data".format(path))
            data = np.frombuffer(
                raw_data, dtype=variable_dtype, count=count,
            ).reshape((len(names), nz, ny, nx))
            data = np.squeeze(data, axis=3 - fixed_axis)
            if not np.all(np.isfinite(data)):
                raise SystemExit("{}: nonfinite residual slice".format(path))
            fields = {
                name: data[names.index(name)]
                for name in Z4C_VARIABLES
            }
            coordinate_min = (limits[0], limits[2], limits[4])
            coordinate_max = (limits[1], limits[3], limits[5])
            column_axis, row_axis = expected_axes
            blocks.append({
                "level": logical_location[3],
                "logical_location": logical_location,
                "plane_axes": expected_axes,
                "fixed_axis": fixed_axis,
                "column_min": coordinate_min[column_axis],
                "column_max": coordinate_max[column_axis],
                "row_min": coordinate_min[row_axis],
                "row_max": coordinate_max[row_axis],
                "ncolumn": sizes[column_axis],
                "nrow": sizes[row_axis],
                "fields": fields,
            })
    if not blocks:
        raise SystemExit("{}: no slice MeshBlocks".format(path))
    return time, blocks


def slice_series(root, arguments):
    # Production runs place binary outputs under the configured bin/
    # subdirectory, whereas the lightweight synthetic tests write directly
    # into their case directory.
    binary_root = root / "bin"
    if not binary_root.is_dir():
        binary_root = root
    paths = sorted(binary_root.glob("*.{}.*.bin".format(
        arguments.slice_id)))
    if not paths:
        raise SystemExit(
            "{}: no {} residual slices under {}".format(
                root, arguments.slice_id, binary_root))
    series = []
    for path in paths:
        time = binary_time(path)
        if not math.isfinite(time):
            raise SystemExit("{}: nonfinite output time".format(path))
        series.append((time, path))
    series.sort(key=lambda item: (item[0], item[1].name))
    duplicate_times = [
        series[index][0] for index in range(1, len(series))
        if abs(series[index][0] - series[index - 1][0]) <= 1.0e-12
    ]
    if duplicate_times:
        raise SystemExit(
            "{}: duplicate residual-slice times {}".format(
                root, duplicate_times))
    return series


def sample_points(arguments):
    if not (
        arguments.slice_x_min < arguments.slice_x_max
        and arguments.slice_y_half_width > 0.0
        and arguments.slice_spacing > 0.0
    ):
        raise SystemExit("slice sampling bounds and spacing must be valid")
    x = np.arange(
        arguments.slice_x_min + 0.5 * arguments.slice_spacing,
        arguments.slice_x_max, arguments.slice_spacing)
    second = np.arange(
        -arguments.slice_y_half_width + 0.5 * arguments.slice_spacing,
        arguments.slice_y_half_width, arguments.slice_spacing)
    if len(x) < 2 or len(second) < 2:
        raise SystemExit("slice sampling window is too small")
    grid_x, grid_second = np.meshgrid(x, second, indexing="xy")
    return grid_x.ravel(), grid_second.ravel()


def sample_slice(blocks, point_x, point_second, path):
    """Sample the finest leaf block containing each fixed physical point."""
    result = {
        name: np.full(point_x.shape, np.nan)
        for name in Z4C_VARIABLES
    }
    sampled_level = np.full(point_x.shape, -1, dtype=int)
    for block in sorted(blocks, key=lambda item: item["level"]):
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(block["column_min"]), abs(block["column_max"]),
            abs(block["row_min"]), abs(block["row_max"]))
        inside = (
            (point_x >= block["column_min"] - tolerance)
            & (point_x < block["column_max"] - tolerance)
            & (point_second >= block["row_min"] - tolerance)
            & (point_second < block["row_max"] - tolerance)
            & (block["level"] >= sampled_level)
        )
        if not np.any(inside):
            continue
        dx = (
            (block["column_max"] - block["column_min"])
            / block["ncolumn"]
        )
        dsecond = (
            (block["row_max"] - block["row_min"]) / block["nrow"]
        )
        ix = np.floor(
            (point_x[inside] - block["column_min"]) / dx).astype(int)
        irow = np.floor(
            (point_second[inside] - block["row_min"]) / dsecond
        ).astype(int)
        ix = np.clip(ix, 0, block["ncolumn"] - 1)
        irow = np.clip(irow, 0, block["nrow"] - 1)
        for name in Z4C_VARIABLES:
            result[name][inside] = block["fields"][name][irow, ix]
        sampled_level[inside] = block["level"]
    missing = sampled_level < 0
    if np.any(missing):
        raise SystemExit(
            "{}: {} of {} fixed points are not covered by the slice".
            format(path, int(np.count_nonzero(missing)), len(point_x)))
    return result, sampled_level


def kerr_schild_background(x, y, z, arguments):
    """Return analytic conformal metric, lapse, chi, and shift."""
    x = np.asarray(x) - arguments.background_center_x
    y = np.asarray(y) - arguments.background_center_y
    z = np.asarray(z) - arguments.background_center_z
    mass = arguments.background_mass
    spin = arguments.background_spin
    rho2 = x * x + y * y + z * z
    discriminant = (
        (rho2 - spin * spin) ** 2 + 4.0 * spin * spin * z * z
    )
    r2 = 0.5 * (
        rho2 - spin * spin + np.sqrt(np.maximum(discriminant, 0.0))
    )
    if np.any(r2 <= 0.0):
        raise SystemExit(
            "characteristic sampling reached the Kerr-Schild ring/interior")
    radius = np.sqrt(r2)
    denominator = r2 + spin * spin
    null = np.stack((
        (radius * x + spin * y) / denominator,
        (radius * y - spin * x) / denominator,
        z / radius,
    ), axis=-1)
    h = (
        mass * radius ** 3
        / (radius ** 4 + spin * spin * z * z)
    )
    determinant = 1.0 + 2.0 * h
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise SystemExit("invalid Kerr-Schild background determinant")
    identity = np.eye(3)
    physical_metric = (
        identity + 2.0 * h[..., None, None]
        * null[..., :, None] * null[..., None, :]
    )
    chi = determinant ** (-1.0 / 3.0)
    conformal_metric = chi[..., None, None] * physical_metric
    lapse = determinant ** (-0.5)
    shift = (
        (2.0 * h / determinant)[..., None] * null
    )
    return conformal_metric, lapse, chi, shift


def tensor_from_fields(fields, prefix):
    shape = fields[prefix + "xx"].shape
    result = np.empty(shape + (3, 3), dtype=np.float64)
    for first in range(3):
        for second in range(3):
            result[..., first, second] = fields[
                symmetric_name(prefix, first, second)
            ]
    return result


def vector_from_fields(fields, prefix):
    return np.stack((
        fields[prefix + "x"],
        fields[prefix + "y"],
        fields[prefix + "z"],
    ), axis=-1).astype(np.float64)


def metric_frame(metric, side):
    """Vectorized equivalent of MakeBoundaryFrame in z4c_Sbc.cpp."""
    inverse = np.linalg.inv(metric)
    determinant = np.linalg.det(metric)
    side = np.asarray(side, dtype=np.float64)
    norm2 = np.einsum("i,...ij,j->...", side, inverse, side)
    if (
        np.any(~np.isfinite(norm2)) or np.any(norm2 <= 0.0)
        or np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0)
    ):
        raise SystemExit("invalid reconstructed metric in characteristic frame")
    normal_d = side / np.sqrt(norm2)[..., None]
    normal_u = np.einsum("...ij,...j->...i", inverse, normal_d)

    candidates = []
    norms = []
    for axis in range(3):
        candidate = np.zeros_like(normal_u)
        candidate[..., axis] = 1.0
        candidate -= normal_u * normal_d[..., axis, None]
        candidates.append(candidate)
        norms.append(np.einsum(
            "...i,...ij,...j->...", candidate, metric, candidate))
    candidates = np.stack(candidates, axis=-2)
    norms = np.stack(norms, axis=-1)
    best = np.argmax(norms, axis=-1)
    tangent1_u = np.take_along_axis(
        candidates, best[..., None, None], axis=-2
    )[..., 0, :]
    tangent1_norm2 = np.take_along_axis(
        norms, best[..., None], axis=-1
    )[..., 0]
    if np.any(tangent1_norm2 <= 0.0):
        raise SystemExit("invalid first tangent in characteristic frame")
    tangent1_u /= np.sqrt(tangent1_norm2)[..., None]
    tangent1_d = np.einsum("...ij,...j->...i", metric, tangent1_u)
    tangent2_u = (
        np.cross(normal_d, tangent1_d)
        / np.sqrt(determinant)[..., None]
    )
    tangent2_norm2 = np.einsum(
        "...i,...ij,...j->...", tangent2_u, metric, tangent2_u)
    if np.any(tangent2_norm2 <= 0.0):
        raise SystemExit("invalid second tangent in characteristic frame")
    tangent2_u /= np.sqrt(tangent2_norm2)[..., None]
    tangent2_d = np.einsum("...ij,...j->...i", metric, tangent2_u)
    return (
        inverse, normal_d, normal_u,
        tangent1_d, tangent1_u, tangent2_d, tangent2_u,
    )


def coordinate_derivatives(values, dcolumn, drow):
    return (
        np.gradient(values, dcolumn, axis=1, edge_order=2),
        np.gradient(values, drow, axis=0, edge_order=2),
    )


def project_tensor(tensor, left, right):
    return np.einsum("...i,...ij,...j->...", left, tensor, right)


def characteristic_fields(block, arguments):
    """Project the eight incoming gauge/constraint modes on one slice block."""
    fields = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in block["fields"].items()
    }
    nrow, ncolumn = fields["z4c_chi"].shape
    plane_axes = block.get("plane_axes", (0, 1))
    fixed_axis = block.get(
        "fixed_axis", ({0, 1, 2} - set(plane_axes)).pop())
    column_axis, row_axis = plane_axes
    column_min = block.get("column_min", block.get("x1min"))
    column_max = block.get("column_max", block.get("x1max"))
    row_min = block.get("row_min", block.get("x2min"))
    row_max = block.get("row_max", block.get("x2max"))
    dcolumn = (column_max - column_min) / ncolumn
    drow = (row_max - row_min) / nrow
    column = column_min + (np.arange(ncolumn) + 0.5) * dcolumn
    row = row_min + (np.arange(nrow) + 0.5) * drow
    grid_column, grid_row = np.meshgrid(column, row, indexing="xy")
    coordinates = [
        np.full_like(
            grid_column, getattr(arguments, "slice_fixed_coordinate", 0.0))
        for _ in range(3)
    ]
    coordinates[column_axis] = grid_column
    coordinates[row_axis] = grid_row
    grid_x, grid_y, grid_z = coordinates

    background_metric, alpha_bg, chi_bg, beta_bg = (
        kerr_schild_background(grid_x, grid_y, grid_z, arguments)
    )
    residual_metric = tensor_from_fields(fields, "z4c_g")
    residual_a = tensor_from_fields(fields, "z4c_A")
    residual_gamma = vector_from_fields(fields, "z4c_Gam")
    residual_beta = vector_from_fields(fields, "z4c_beta")
    full_metric = background_metric + residual_metric
    alpha = alpha_bg + fields["z4c_alpha"]
    chi = chi_bg + fields["z4c_chi"]
    beta = beta_bg + residual_beta
    side = np.zeros(3)
    side[arguments.boundary_axis - 1] = float(arguments.boundary_side)
    (
        inverse_metric, normal_d, normal_u,
        tangent1_d, tangent1_u, tangent2_d, tangent2_u,
    ) = metric_frame(full_metric, side)
    unresolved = float(np.max(np.abs(normal_u[..., fixed_axis])))
    if unresolved > arguments.maximum_unresolved_normal:
        raise SystemExit(
            "{} slice cannot resolve normal component along axis {}: "
            "{:.8e} > {:.8e}".format(
                "".join("xyz"[axis] for axis in plane_axes),
                fixed_axis + 1, unresolved,
                arguments.maximum_unresolved_normal))

    metric_derivatives = []
    for first in range(3):
        row = []
        for second in range(3):
            column_value, row_value = coordinate_derivatives(
                residual_metric[..., first, second], dcolumn, drow)
            row.append(
                normal_u[..., column_axis] * column_value
                + normal_u[..., row_axis] * row_value)
        metric_derivatives.append(row)
    metric_derivative = np.stack(
        [np.stack(row, axis=-1) for row in metric_derivatives], axis=-2)

    def normal_derivative(value):
        column_value, row_value = coordinate_derivatives(
            value, dcolumn, drow)
        return (
            normal_u[..., column_axis] * column_value
            + normal_u[..., row_axis] * row_value
        )

    beta_derivative = np.stack(
        [normal_derivative(residual_beta[..., axis])
         for axis in range(3)],
        axis=-1,
    )
    metric_derivative_trace = np.einsum(
        "...ij,...ij->...", inverse_metric, metric_derivative)
    a_trace = np.einsum(
        "...ij,...ij->...", inverse_metric, residual_a)

    scalar_p = (
        fields["z4c_Khat"],
        fields["z4c_Theta"],
        project_tensor(residual_a, normal_u, normal_u) - a_trace / 3.0,
        np.einsum("...i,...i->...", normal_d, residual_gamma),
    )
    scalar_d = (
        normal_derivative(fields["z4c_chi"]),
        project_tensor(metric_derivative, normal_u, normal_u)
        - metric_derivative_trace / 3.0,
        normal_derivative(fields["z4c_alpha"]),
        np.einsum("...i,...i->...", normal_d, beta_derivative),
    )

    beta_full_normal = np.einsum("...i,...i->...", normal_d, beta)
    beta_bg_normal = np.einsum("...i,...i->...", normal_d, beta_bg)
    lapse_driver = (
        arguments.residual_lapse_f
        * (
            arguments.lapse_oplog * arguments.lapse_harmonicf
            + arguments.lapse_harmonic * alpha_bg
        )
        * alpha_bg
    )
    shift_driver = arguments.shift_driver
    beta_difference = beta_full_normal - beta_bg_normal
    lapse_root = 0.5 * (
        beta_full_normal + beta_bg_normal
        + np.sqrt(beta_difference ** 2 + 4.0 * chi * lapse_driver)
    )
    shift_long_root = 0.5 * (
        beta_full_normal + beta_bg_normal
        + np.sqrt(beta_difference ** 2 + (16.0 / 3.0) * shift_driver)
    )
    shift_transverse_root = 0.5 * (
        beta_full_normal + beta_bg_normal
        + np.sqrt(beta_difference ** 2 + 4.0 * shift_driver)
    )
    if (
        np.any(alpha <= 0.0) or np.any(chi <= 0.0)
        or np.any(lapse_root <= 0.0)
        or np.any(shift_long_root <= 0.0)
        or np.any(shift_transverse_root <= 0.0)
    ):
        raise SystemExit("invalid characteristic coefficient in TDE slice")

    sqrt_chi = np.sqrt(chi)
    shift_mu = shift_long_root - beta_full_normal
    shift_delta_bg = shift_long_root - beta_bg_normal
    lapse_shift_separation = (
        chi * lapse_driver - shift_mu * shift_delta_bg
    )
    light_shift_separation = chi * alpha ** 2 - shift_mu ** 2
    shift_q = (4.0 / 3.0) * shift_driver

    left_lapse_p0 = -(lapse_root - beta_bg_normal) / chi
    left_shift_p0 = (
        alpha * shift_delta_bg ** 2 * light_shift_separation
    )
    left_shift_p1 = (
        0.5 * alpha * shift_q * lapse_shift_separation
    )
    left_shift_p3 = (
        0.25 * shift_delta_bg
        * (4.0 * chi * alpha ** 2 - 3.0 * shift_mu ** 2)
        * lapse_shift_separation
    )
    left_shift_d0 = (
        0.5 * alpha ** 2 * shift_delta_bg * lapse_shift_separation
    )
    left_shift_d2 = (
        -chi * alpha * shift_delta_bg * light_shift_separation
    )
    left_shift_d3 = (
        lapse_shift_separation * light_shift_separation
    )
    result = {
        "lapse": left_lapse_p0 * scalar_p[0] + scalar_d[2],
        "shift_longitudinal": (
            left_shift_p0 * scalar_p[0]
            + left_shift_p1 * scalar_p[1]
            + left_shift_p3 * scalar_p[3]
            + left_shift_d0 * scalar_d[0]
            + left_shift_d2 * scalar_d[2]
            + left_shift_d3 * scalar_d[3]
        ),
        "constraint_scalar_theta": (
            sqrt_chi * scalar_p[1]
            + 0.5 * chi * scalar_p[3]
            + scalar_d[0]
        ),
        "constraint_scalar_z": (
            4.0 * scalar_p[0] / (3.0 * sqrt_chi)
            + 2.0 * scalar_p[1] / (3.0 * sqrt_chi)
            - 2.0 * scalar_p[2] / sqrt_chi
            - scalar_p[3]
            + scalar_d[1]
        ),
    }
    for number, (tangent_d, tangent_u) in enumerate((
        (tangent1_d, tangent1_u), (tangent2_d, tangent2_u)
    ), start=1):
        a_normal_tangent = project_tensor(
            residual_a, normal_u, tangent_u)
        gamma_tangent = np.einsum(
            "...i,...i->...", tangent_d, residual_gamma)
        metric_normal_tangent = project_tensor(
            metric_derivative, normal_u, tangent_u)
        beta_tangent = np.einsum(
            "...i,...i->...", tangent_d, beta_derivative)
        result["shift_transverse_{}".format(number)] = (
            (shift_transverse_root - beta_bg_normal) * gamma_tangent
            + beta_tangent
        )
        result["constraint_transverse_{}".format(number)] = (
            -2.0 * a_normal_tangent / sqrt_chi
            - gamma_tangent
            + metric_normal_tangent
        )
    return result, grid_column, grid_row


def block_key(block):
    return (
        block["level"], block["plane_axes"],
        block["ncolumn"], block["nrow"],
        round(block["column_min"], 12), round(block["column_max"], 12),
        round(block["row_min"], 12), round(block["row_max"], 12),
    )


def block_intersects_window(block, arguments):
    return (
        block["column_max"] > arguments.slice_x_min
        and block["column_min"] < arguments.slice_x_max
        and block["row_max"] > -arguments.slice_y_half_width
        and block["row_min"] < arguments.slice_y_half_width
    )


def matched_characteristic_differences(
        reference_root, case_root, start, arguments):
    reference_series = slice_series(reference_root, arguments)
    case_series = slice_series(case_root, arguments)
    reference_times = np.asarray([item[0] for item in reference_series])
    per_mode = {
        mode: {"rms": 0.0, "linf": 0.0}
        for modes in CHARACTERISTIC_GROUPS.values() for mode in modes
    }
    samples = 0
    minimum_points = None
    for case_time, case_path in case_series:
        if case_time + 1.0e-12 < start:
            continue
        nearest = int(np.argmin(np.abs(reference_times - case_time)))
        reference_time, reference_path = reference_series[nearest]
        time_tolerance = 1.0e-10 * max(
            1.0, abs(case_time), abs(reference_time))
        if abs(case_time - reference_time) > time_tolerance:
            raise SystemExit(
                "{}: no far-reference slice at t={:.16g}".format(
                    case_path, case_time))
        _, reference_blocks = read_slice_binary(reference_path, arguments)
        _, case_blocks = read_slice_binary(case_path, arguments)
        reference_by_key = {
            block_key(block): block for block in reference_blocks
        }
        case_by_key = {block_key(block): block for block in case_blocks}
        reference_window = {
            key for key, block in reference_by_key.items()
            if block_intersects_window(block, arguments)
        }
        case_window = {
            key for key, block in case_by_key.items()
            if block_intersects_window(block, arguments)
        }
        if reference_window != case_window:
            raise SystemExit(
                "{}: far/near characteristic blocks or AMR levels differ".
                format(case_path))

        sum_squares = {mode: 0.0 for mode in per_mode}
        maxima = {mode: 0.0 for mode in per_mode}
        points = 0
        for key in sorted(case_window):
            reference_block = reference_by_key[key]
            case_block = case_by_key[key]
            reference_modes, grid_x, grid_y = characteristic_fields(
                reference_block, arguments)
            case_modes, case_x, case_y = characteristic_fields(
                case_block, arguments)
            if not (
                np.array_equal(grid_x, case_x)
                and np.array_equal(grid_y, case_y)
            ):
                raise SystemExit(
                    "{}: far/near characteristic cell centers differ".
                    format(case_path))
            margin = arguments.characteristic_block_margin
            mask = (
                (grid_x >= arguments.slice_x_min)
                & (grid_x < arguments.slice_x_max)
                & (grid_y >= -arguments.slice_y_half_width)
                & (grid_y < arguments.slice_y_half_width)
            )
            if margin > 0:
                mask[:margin, :] = False
                mask[-margin:, :] = False
                mask[:, :margin] = False
                mask[:, -margin:] = False
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            points += count
            for mode in per_mode:
                difference = (
                    case_modes[mode][mask] - reference_modes[mode][mask]
                )
                if not np.all(np.isfinite(difference)):
                    raise SystemExit(
                        "{}: nonfinite {} characteristic".format(
                            case_path, mode))
                sum_squares[mode] += float(np.sum(difference ** 2))
                maxima[mode] = max(
                    maxima[mode], float(np.max(np.abs(difference))))
        if points == 0:
            raise SystemExit(
                "{}: no characteristic sampling points".format(case_path))
        minimum_points = (
            points if minimum_points is None else min(minimum_points, points)
        )
        for mode in per_mode:
            per_mode[mode]["rms"] = max(
                per_mode[mode]["rms"],
                math.sqrt(sum_squares[mode] / points),
            )
            per_mode[mode]["linf"] = max(
                per_mode[mode]["linf"], maxima[mode])
        samples += 1
    if samples < 2:
        raise SystemExit(
            "{}: fewer than two characteristic slices at or after t={}".
            format(case_root, start))
    groups = {
        group: {
            norm: max(per_mode[mode][norm] for mode in modes)
            for norm in ("rms", "linf")
        }
        for group, modes in CHARACTERISTIC_GROUPS.items()
    }
    return groups, per_mode, samples, minimum_points


def matched_slice_differences(reference_root, case_root, start, arguments):
    reference_series = slice_series(reference_root, arguments)
    case_series = slice_series(case_root, arguments)
    point_x, point_y = sample_points(arguments)
    reference_times = np.asarray([item[0] for item in reference_series])
    per_field = {
        name: {"rms": 0.0, "linf": 0.0}
        for name in Z4C_VARIABLES
    }
    samples = 0
    for case_time, case_path in case_series:
        if case_time + 1.0e-12 < start:
            continue
        nearest = int(np.argmin(np.abs(reference_times - case_time)))
        reference_time, reference_path = reference_series[nearest]
        time_tolerance = 1.0e-10 * max(
            1.0, abs(case_time), abs(reference_time))
        if abs(case_time - reference_time) > time_tolerance:
            raise SystemExit(
                "{}: no far-reference slice at t={:.16g}".format(
                    case_path, case_time))
        _, reference_blocks = read_slice_binary(reference_path, arguments)
        _, case_blocks = read_slice_binary(case_path, arguments)
        reference, reference_level = sample_slice(
            reference_blocks, point_x, point_y, reference_path)
        case, case_level = sample_slice(
            case_blocks, point_x, point_y, case_path)
        if not np.array_equal(reference_level, case_level):
            mismatch = int(np.count_nonzero(reference_level != case_level))
            raise SystemExit(
                "{}: {} of {} fixed points differ in AMR level from {}".
                format(case_path, mismatch, len(point_x), reference_path))
        for name in Z4C_VARIABLES:
            difference = (
                case[name].astype(np.float64)
                - reference[name].astype(np.float64)
            )
            rms = float(np.sqrt(np.mean(difference * difference)))
            linf = float(np.max(np.abs(difference)))
            per_field[name]["rms"] = max(per_field[name]["rms"], rms)
            per_field[name]["linf"] = max(per_field[name]["linf"], linf)
        samples += 1
    if samples < 2:
        raise SystemExit(
            "{}: fewer than two residual slices at or after t={}".
            format(case_root, start))
    groups = {}
    for group, names in SLICE_GROUPS.items():
        groups[group] = {
            norm: max(per_field[name][norm] for name in names)
            for norm in ("rms", "linf")
        }
    return groups, per_field, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("far", type=pathlib.Path)
    parser.add_argument("sommerfeld", type=pathlib.Path)
    parser.add_argument("cpbc", type=pathlib.Path)
    parser.add_argument("cpbc_sponge", type=pathlib.Path)
    parser.add_argument("--round-trip-time", type=float, default=11.5)
    parser.add_argument("--maximum-density-fraction", type=float, default=0.01)
    parser.add_argument(
        "--maximum-trajectory-radii", type=float, default=0.1)
    parser.add_argument("--minimum-return-improvement", type=float, default=10.0)
    parser.add_argument("--slice-x-min", type=float, default=32.0)
    parser.add_argument("--slice-x-max", type=float, default=45.0)
    parser.add_argument("--slice-y-half-width", type=float, default=6.0)
    parser.add_argument("--slice-spacing", type=float, default=0.125)
    parser.add_argument("--slice-id", default="xy_z4c")
    parser.add_argument(
        "--slice-plane", choices=tuple(PLANE_AXES), default="xy")
    parser.add_argument("--slice-fixed-coordinate", type=float, default=0.0)
    parser.add_argument(
        "--minimum-sommerfeld-slice-rms", type=float, default=1.0e-12)
    parser.add_argument("--background-mass", type=float, default=1.0)
    parser.add_argument("--background-spin", type=float, default=0.0)
    parser.add_argument("--background-center-x", type=float, default=0.0)
    parser.add_argument("--background-center-y", type=float, default=0.0)
    parser.add_argument("--background-center-z", type=float, default=0.0)
    parser.add_argument(
        "--boundary-axis", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--boundary-side", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--residual-lapse-f", type=float, default=1.0)
    parser.add_argument("--lapse-oplog", type=float, default=2.0)
    parser.add_argument("--lapse-harmonicf", type=float, default=1.0)
    parser.add_argument("--lapse-harmonic", type=float, default=0.0)
    parser.add_argument("--shift-driver", type=float, default=1.0)
    parser.add_argument("--characteristic-block-margin", type=int, default=2)
    parser.add_argument(
        "--maximum-unresolved-normal", type=float, default=1.0e-7)
    args = parser.parse_args()

    if not (
        math.isfinite(args.round_trip_time) and args.round_trip_time >= 0.0
    ):
        raise SystemExit("round-trip time must be finite and nonnegative")
    finite_parameters = (
        args.background_mass, args.background_spin,
        args.background_center_x, args.background_center_y,
        args.background_center_z, args.residual_lapse_f,
        args.lapse_oplog, args.lapse_harmonicf, args.lapse_harmonic,
        args.shift_driver, args.maximum_unresolved_normal,
        args.slice_fixed_coordinate,
    )
    if (
        not all(math.isfinite(value) for value in finite_parameters)
        or args.background_mass < 0.0
        or abs(args.background_spin) > args.background_mass
        or args.residual_lapse_f <= 0.0
        or args.shift_driver <= 0.0
        or args.characteristic_block_margin < 1
        or args.maximum_unresolved_normal <= 0.0
    ):
        raise SystemExit("invalid characteristic-projection parameter")
    if args.boundary_axis - 1 not in PLANE_AXES[args.slice_plane]:
        raise SystemExit(
            "boundary axis {} is not resolved by the {} slice".format(
                args.boundary_axis, args.slice_plane))

    roots = {
        "far": args.far,
        "sommerfeld": args.sommerfeld,
        "cpbc": args.cpbc,
        "cpbc_sponge": args.cpbc_sponge,
    }
    histories = {name: load_history(root) for name, root in roots.items()}
    tracks = {name: load_track(root) for name, root in roots.items()}
    initialized_radii = {
        name: tracks[name][2] for name in roots
    }
    far_radius = initialized_radii["far"]
    for name, radius in initialized_radii.items():
        if not math.isclose(
            radius, far_radius, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise SystemExit(
                "{} initialized stellar radius {:.16g} differs from far "
                "reference {:.16g}".format(name, radius, far_radius))
    stellar_radius = far_radius
    print(
        "initialized_stellar_radius={:.8e} trajectory_limit={:.8e}".format(
            far_radius, args.maximum_trajectory_radii * stellar_radius
        )
    )
    return_labels = REQUIRED_RETURN_LABELS + tuple(
        label for label in OPTIONAL_RETURN_LABELS
        if all(label in histories[name] for name in roots)
    )

    metrics = {}
    for name in ("sommerfeld", "cpbc", "cpbc_sponge"):
        history_fields, density = history_differences(
            histories["far"], histories[name], args.round_trip_time,
            return_labels)
        trajectory = trajectory_difference(
            tracks["far"], tracks[name], args.round_trip_time)
        slice_groups, slice_fields, slice_samples = (
            matched_slice_differences(
                args.far, roots[name], args.round_trip_time, args))
        characteristic_groups, characteristic_modes, characteristic_samples, (
            characteristic_points
        ) = matched_characteristic_differences(
            args.far, roots[name], args.round_trip_time, args)
        metrics[name] = {
            "density": density,
            "trajectory": trajectory,
            "history": history_fields,
            "slice_groups": slice_groups,
            "slice_fields": slice_fields,
            "characteristic_groups": characteristic_groups,
            "characteristic_modes": characteristic_modes,
        }
        print(
            "{} history_difference={:.8e} density_fraction={:.8e} "
            "trajectory={:.8e} history_fields={}".format(
                name, max(history_fields.values()), density, trajectory,
                ",".join(
                    "{}:{:.8e}".format(label, history_fields[label])
                    for label in return_labels),
            )
        )
        print(
            "{} slice_samples={} slice_groups={}".format(
                name, slice_samples, ",".join(
                    "{}:rms={:.8e}:linf={:.8e}".format(
                        group, slice_groups[group]["rms"],
                        slice_groups[group]["linf"])
                    for group in SLICE_GROUPS)))
        print(
            "{} slice_fields={}".format(
                name, ",".join(
                    "{}:rms={:.8e}:linf={:.8e}".format(
                        field, slice_fields[field]["rms"],
                        slice_fields[field]["linf"])
                    for field in sorted(set(
                        field for fields in SLICE_GROUPS.values()
                        for field in fields)))))
        print(
            "{} characteristic_samples={} minimum_points={} "
            "characteristic_groups={}".format(
                name, characteristic_samples, characteristic_points,
                ",".join(
                    "{}:rms={:.8e}:linf={:.8e}".format(
                        group, characteristic_groups[group]["rms"],
                        characteristic_groups[group]["linf"])
                    for group in CHARACTERISTIC_GROUPS)))
        print(
            "{} characteristic_modes={}".format(
                name, ",".join(
                    "{}:rms={:.8e}:linf={:.8e}".format(
                        mode, characteristic_modes[mode]["rms"],
                        characteristic_modes[mode]["linf"])
                    for mode in sorted(characteristic_modes))))
        if characteristic_samples != slice_samples:
            raise SystemExit(
                "{}: characteristic/raw slice sample counts differ".format(
                    name))

    failures = []
    for group in CHARACTERISTIC_GROUPS:
        sommerfeld_rms = (
            metrics["sommerfeld"]["characteristic_groups"][group]["rms"])
        if sommerfeld_rms < args.minimum_sommerfeld_slice_rms:
            failures.append(
                "Sommerfeld {} characteristic RMS {:.6g} is below the "
                "observable "
                "floor {:.6g}".format(
                    group, sommerfeld_rms,
                    args.minimum_sommerfeld_slice_rms))
    for name in ("cpbc", "cpbc_sponge"):
        density = metrics[name]["density"]
        trajectory = metrics[name]["trajectory"]
        for group in CHARACTERISTIC_GROUPS:
            sommerfeld_rms = (
                metrics["sommerfeld"]["characteristic_groups"][group]["rms"])
            case_rms = (
                metrics[name]["characteristic_groups"][group]["rms"])
            improvement = sommerfeld_rms / max(case_rms, 1.0e-300)
            print(
                "{} {}_characteristic_rms_improvement={:.8e}".format(
                    name, group, improvement))
            if improvement < args.minimum_return_improvement:
                failures.append(
                    "{} {} return improvement {:.6g} is below {:.6g}".
                    format(name, group, improvement,
                           args.minimum_return_improvement))
        for group, modes in CHARACTERISTIC_GROUPS.items():
            for mode in modes:
                sommerfeld_rms = metrics["sommerfeld"][
                    "characteristic_modes"][mode]["rms"]
                case_rms = metrics[name]["characteristic_modes"][mode]["rms"]
                improvement = sommerfeld_rms / max(case_rms, 1.0e-300)
                print(
                    "{} {}_characteristic_rms_improvement={:.8e}".format(
                        name, mode, improvement))
                if improvement < args.minimum_return_improvement:
                    failures.append(
                        "{} {} return improvement {:.6g} is below {:.6g}".
                        format(name, mode, improvement,
                               args.minimum_return_improvement))
        if density >= args.maximum_density_fraction:
            failures.append(
                "{} density difference {:.6g} exceeds {:.6g}".format(
                    name, density, args.maximum_density_fraction))
        maximum_trajectory = (
            args.maximum_trajectory_radii * stellar_radius)
        if trajectory >= maximum_trajectory:
            failures.append(
                "{} trajectory difference {:.6g} exceeds {:.6g}".format(
                    name, trajectory, maximum_trajectory))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
