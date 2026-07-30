#!/usr/bin/env python3
"""Measure a reflected characteristic pulse from Athena binary output.

The boundary diagnostic verifies the algebraic Bjørhus enforcement at the
boundary point.  This checker instead projects the saved one-dimensional
interior state: it compares the incoming-family L2 norm in the final dump with
the outgoing-family L2 norm in the initial dump.
"""

import argparse
import math
import pathlib
import struct

import numpy as np


VARIABLES = (
    "z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy",
    "z4c_gyz", "z4c_gzz", "z4c_Khat", "z4c_Axx", "z4c_Axy",
    "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz", "z4c_Gamx",
    "z4c_Gamy", "z4c_Gamz", "z4c_Theta", "z4c_alpha",
    "z4c_betax", "z4c_betay", "z4c_betaz", "z4c_Bx", "z4c_By",
    "z4c_Bz",
)


def symmetric_name(prefix, a, b):
    if a > b:
        a, b = b, a
    suffix = (("xx", "xy", "xz"), ("xy", "yy", "yz"),
              ("xz", "yz", "zz"))[a][b]
    return prefix + suffix


def read_plane_binary(path, axis):
    """Return sorted cell centers and transversely averaged residual fields."""
    path = pathlib.Path(path)
    with path.open("rb") as stream:
        if stream.readline().decode("ascii") != (
                "Athena binary output version=1.1\n"):
            raise RuntimeError("{}: unsupported binary format".format(path))
        stream.readline()
        stream.readline()
        stream.readline()
        location_size = int(stream.readline().decode("ascii")[19:])
        variable_size = int(stream.readline().decode("ascii")[19:])
        stream.readline()
        names = stream.readline().decode("ascii")[12:].split()
        header_offset = int(stream.readline().decode("ascii")[16:])
        data_start = stream.tell() + header_offset
        stream.seek(data_start)

        if location_size not in (4, 8) or variable_size not in (4, 8):
            raise RuntimeError("{}: unsupported floating-point size".format(path))
        missing = [name for name in VARIABLES if name not in names]
        if missing:
            raise RuntimeError(
                "{}: missing variables {}".format(path, ", ".join(missing)))
        location_code = "f" if location_size == 4 else "d"
        variable_dtype = np.dtype("=f4" if variable_size == 4 else "=f8")
        coordinates = []
        fields = {name: [] for name in VARIABLES}

        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(data_start)
        while stream.tell() < file_size:
            raw_indices = stream.read(24)
            if not raw_indices:
                break
            if len(raw_indices) != 24:
                raise RuntimeError("{}: truncated block indices".format(path))
            indices = struct.unpack("=6i", raw_indices)
            stream.read(16)  # logical block location and refinement level
            limits = struct.unpack(
                "=6" + location_code, stream.read(6 * location_size))
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            shape = (nz, ny, nx)
            count = len(names) * nx * ny * nz
            data = np.frombuffer(
                stream.read(count * variable_size), dtype=variable_dtype,
                count=count,
            ).reshape((len(names),) + shape)

            cell_count = (nx, ny, nz)[axis]
            lower = limits[2 * axis]
            upper = limits[2 * axis + 1]
            centers = lower + (np.arange(cell_count) + 0.5) * (
                (upper - lower) / cell_count)
            coordinates.append(centers)
            average_axes = tuple(index for index in (0, 1, 2)
                                 if index != 2 - axis)
            for name in VARIABLES:
                fields[name].append(
                    np.mean(data[names.index(name)], axis=average_axes))

    x = np.concatenate(coordinates)
    order = np.argsort(x)
    result = {name: np.concatenate(parts)[order]
              for name, parts in fields.items()}
    return x[order], result


def boundary_frame(axis, side):
    normal = np.zeros(3)
    normal[axis] = float(side)
    tangent1 = np.zeros(3)
    tangent1[0 if axis != 0 else 1] = 1.0
    tangent2 = np.cross(normal, tangent1)
    return normal, tangent1, tangent2


def derivative(values, x, side):
    result = np.gradient(values, x, edge_order=2)
    if len(values) >= 7:
        dx = np.median(np.diff(x))
        result[3:-3] = (
            -values[:-6] + 9.0 * values[1:-5] -
            45.0 * values[2:-4] + 45.0 * values[4:-2] -
            9.0 * values[5:-1] + values[6:]) / (60.0 * dx)
    return side * result


def project_vector(fields, prefix, vector):
    return sum(vector[a] * fields[prefix + ("x", "y", "z")[a]]
               for a in range(3))


def project_tensor(fields, prefix, left, right):
    value = np.zeros_like(fields[prefix + "xx"])
    for a in range(3):
        for b in range(3):
            value += (left[a] * right[b] *
                      fields[symmetric_name(prefix, a, b)])
    return value


def characteristic(fields, x, family, axis, side, incoming):
    normal, tangent1, tangent2 = boundary_frame(axis, side)
    tangents = (tangent1, tangent2)
    sign = 1.0 if incoming else -1.0

    khat = fields["z4c_Khat"]
    theta = fields["z4c_Theta"]
    gamma_n = project_vector(fields, "z4c_Gam", normal)
    metric_trace = sum(fields[symmetric_name("z4c_g", a, a)]
                       for a in range(3))
    a_trace = sum(fields[symmetric_name("z4c_A", a, a)]
                  for a in range(3))
    a_nn = project_tensor(fields, "z4c_A", normal, normal) - a_trace / 3.0
    dg_nn = derivative(
        project_tensor(fields, "z4c_g", normal, normal) - metric_trace / 3.0,
        x, side)
    dchi = derivative(fields["z4c_chi"], x, side)
    dalpha = derivative(fields["z4c_alpha"], x, side)
    beta_n = project_vector(fields, "z4c_beta", normal)
    db_n = derivative(beta_n, x, side)

    if family == "lapse":
        return -sign * math.sqrt(2.0) * khat + dalpha
    if family == "shift_longitudinal":
        return (-4.0 * khat + 4.0 * theta +
                sign * 2.0 * math.sqrt(3.0) * dchi +
                sign * 2.0 * math.sqrt(3.0) * dalpha - 2.0 * db_n)
    if family == "constraint_scalar_theta":
        return sign * theta + 0.5 * gamma_n + dchi
    if family == "constraint_scalar_z":
        return (sign * (4.0 * khat / 3.0 + 2.0 * theta / 3.0 -
                        2.0 * a_nn) - gamma_n + dg_nn)

    if family.endswith("_1"):
        tangent = tangents[0]
    elif family.endswith("_2"):
        tangent = tangents[1]
    else:
        tangent = None
    if tangent is not None:
        gamma_a = project_vector(fields, "z4c_Gam", tangent)
        db_a = derivative(project_vector(fields, "z4c_beta", tangent), x, side)
        if family.startswith("shift_transverse"):
            return sign * gamma_a + db_a
        a_na = project_tensor(fields, "z4c_A", normal, tangent)
        dg_na = derivative(
            project_tensor(fields, "z4c_g", normal, tangent), x, side)
        if family.startswith("constraint_transverse"):
            return -sign * 2.0 * a_na - gamma_a + dg_na

    if family == "tt_plus":
        a_tf = 0.5 * (
            project_tensor(fields, "z4c_A", tangent1, tangent1) -
            project_tensor(fields, "z4c_A", tangent2, tangent2))
        dg_tf = derivative(
            0.5 * (
                project_tensor(fields, "z4c_g", tangent1, tangent1) -
                project_tensor(fields, "z4c_g", tangent2, tangent2)),
            x, side)
        return -sign * 2.0 * a_tf + dg_tf
    if family == "tt_cross":
        a_tf = project_tensor(fields, "z4c_A", tangent1, tangent2)
        dg_tf = derivative(
            project_tensor(fields, "z4c_g", tangent1, tangent2), x, side)
        return -sign * 2.0 * a_tf + dg_tf
    raise RuntimeError("unsupported family {}".format(family))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial")
    parser.add_argument("final")
    parser.add_argument("family")
    parser.add_argument("axis", type=int, choices=(1, 2, 3))
    parser.add_argument("side", type=int, choices=(-1, 1))
    parser.add_argument("--maximum-ratio", type=float, default=0.02)
    parser.add_argument(
        "--maximum-initial-incoming-ratio", type=float, default=0.005)
    parser.add_argument("--boundary-cells", type=int, default=4)
    parser.add_argument(
        "--control-final",
        help=(
            "optional causally disconnected final dump at the same spacing; "
            "its incoming characteristic is subtracted before measuring "
            "boundary reflection"
        ),
    )
    args = parser.parse_args()

    x0, initial = read_plane_binary(args.initial, args.axis - 1)
    xf, final = read_plane_binary(args.final, args.axis - 1)
    if x0.shape != xf.shape or not np.allclose(x0, xf):
        raise SystemExit("initial and final grids differ")
    outgoing = characteristic(
        initial, x0, args.family, args.axis - 1, args.side, False)
    initial_incoming = characteristic(
        initial, x0, args.family, args.axis - 1, args.side, True)
    incoming_raw = characteristic(
        final, xf, args.family, args.axis - 1, args.side, True)
    incoming = incoming_raw
    control_norm = 0.0
    if args.control_final:
        xc, control = read_plane_binary(
            args.control_final, args.axis - 1)
        if len(xc) < len(xf) or xf[0] < xc[0] or xf[-1] > xc[-1]:
            raise SystemExit(
                "control grid does not contain the measured grid")
        dx = np.median(np.diff(xf))
        dxc = np.median(np.diff(xc))
        if not (
                np.allclose(np.diff(xf), dx) and
                np.allclose(np.diff(xc), dxc) and
                math.isclose(dx, dxc, rel_tol=1.0e-6, abs_tol=1.0e-12)):
            raise SystemExit(
                "control and measured grids do not have matching spacing")
        control_incoming = characteristic(
            control, xc, args.family, args.axis - 1, args.side, True)
        control_on_grid = np.interp(xf, xc, control_incoming)
        incoming = incoming_raw - control_on_grid
        control_norm = float(np.linalg.norm(control_on_grid))
    trim = args.boundary_cells
    if 2 * trim >= len(x0):
        raise SystemExit("too many boundary cells excluded")
    outgoing = outgoing[trim:-trim]
    initial_incoming = initial_incoming[trim:-trim]
    incoming = incoming[trim:-trim]
    outgoing_norm = float(np.linalg.norm(outgoing))
    initial_incoming_norm = float(np.linalg.norm(initial_incoming))
    incoming_norm = float(np.linalg.norm(incoming))
    if not (
            math.isfinite(outgoing_norm) and
            math.isfinite(initial_incoming_norm) and
            math.isfinite(incoming_norm)):
        raise SystemExit("nonfinite characteristic norm")
    if outgoing_norm <= 1.0e-12:
        raise SystemExit("outgoing pulse norm was not observed")
    ratio = incoming_norm / outgoing_norm
    initial_incoming_ratio = initial_incoming_norm / outgoing_norm
    print(
        "family={} outgoing_l2={:.8e} initial_incoming_l2={:.8e} "
        "initial_incoming_ratio={:.8e} incoming_l2={:.8e} "
        "control_l2={:.8e} ratio={:.8e}".format(
            args.family, outgoing_norm, initial_incoming_norm,
            initial_incoming_ratio, incoming_norm, control_norm, ratio))
    if initial_incoming_ratio >= args.maximum_initial_incoming_ratio:
        raise SystemExit(
            "initial incoming ratio {:.6e} exceeds {:.6e}".format(
                initial_incoming_ratio,
                args.maximum_initial_incoming_ratio))
    if ratio >= args.maximum_ratio:
        raise SystemExit(
            "reflected interior ratio {:.6e} exceeds {:.6e}".format(
                ratio, args.maximum_ratio))


if __name__ == "__main__":
    main()
