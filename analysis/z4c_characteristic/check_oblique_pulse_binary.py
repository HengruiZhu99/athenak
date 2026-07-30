#!/usr/bin/env python3
"""Measure an edge/corner response for a deterministic-ownership check.

The small-domain result is differenced against a causally disconnected run at
the same grid spacing.  Directional derivatives are reconstructed on the
uniform three-dimensional grid, so this test measures the characteristic
selected by the normalized composite normal used at an edge or corner.  The
reported ratio is a finite-response diagnostic, not a normal-incidence
absorption gate: a planar L=1 condition is not exact for an oblique wave
striking multiple faces.
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


def read_uniform_binary(path):
    """Assemble all MeshBlocks in a single-rank binary output."""
    path = pathlib.Path(path)
    blocks = []
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
        if location_size not in (4, 8) or variable_size not in (4, 8):
            raise RuntimeError("{}: unsupported floating-point size".format(path))
        missing = [name for name in VARIABLES if name not in names]
        if missing:
            raise RuntimeError(
                "{}: missing variables {}".format(path, ", ".join(missing)))
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
                raise RuntimeError("{}: truncated block indices".format(path))
            indices = struct.unpack("=6i", raw_indices)
            stream.read(16)
            limits = struct.unpack(
                "=6" + location_code, stream.read(6 * location_size))
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            count = len(names) * nx * ny * nz
            data = np.frombuffer(
                stream.read(count * variable_size), dtype=variable_dtype,
                count=count,
            ).reshape((len(names), nz, ny, nx))
            coordinates = []
            for axis, count_axis in enumerate((nx, ny, nz)):
                lower = limits[2 * axis]
                upper = limits[2 * axis + 1]
                coordinates.append(
                    lower + (np.arange(count_axis) + 0.5) *
                    ((upper - lower) / count_axis))
            blocks.append((coordinates, data))

    if not blocks:
        raise RuntimeError("{}: no MeshBlocks".format(path))
    coordinates = tuple(
        np.unique(np.concatenate([block[0][axis] for block in blocks]))
        for axis in range(3))
    shape = (len(coordinates[2]), len(coordinates[1]), len(coordinates[0]))
    fields = {name: np.full(shape, np.nan) for name in VARIABLES}
    for block_coordinates, data in blocks:
        indices = []
        for axis in range(3):
            index = np.searchsorted(coordinates[axis], block_coordinates[axis])
            if not np.allclose(
                    coordinates[axis][index], block_coordinates[axis],
                    rtol=0.0, atol=1.0e-10):
                raise RuntimeError("{}: nonuniform block coordinates".format(path))
            indices.append(index)
        target = np.ix_(indices[2], indices[1], indices[0])
        for name in VARIABLES:
            fields[name][target] = data[names.index(name)]
    if any(not np.all(np.isfinite(values)) for values in fields.values()):
        raise RuntimeError("{}: missing or nonfinite assembled data".format(path))
    return coordinates, fields


def derivative_axis(values, coordinate, array_axis):
    result = np.gradient(values, coordinate, axis=array_axis, edge_order=2)
    if len(coordinate) >= 7:
        dx = float(np.median(np.diff(coordinate)))
        target = [slice(None)] * 3
        target[array_axis] = slice(3, -3)
        terms = []
        for start, coefficient in (
                (0, -1.0), (1, 9.0), (2, -45.0),
                (4, 45.0), (5, -9.0), (6, 1.0)):
            source = [slice(None)] * 3
            source[array_axis] = slice(start, start + len(coordinate) - 6)
            terms.append(coefficient * values[tuple(source)])
        result[tuple(target)] = sum(terms) / (60.0 * dx)
    return result


def normal_derivative(values, coordinates, normal):
    # Coordinate x^a maps to NumPy axes (2, 1, 0).
    return sum(
        normal[axis] * derivative_axis(values, coordinates[axis], 2 - axis)
        for axis in range(3))


def boundary_frame(dimensions, side):
    if dimensions == 3:
        normal = side * np.ones(3) / math.sqrt(3.0)
        tangent1 = np.array((2.0, -1.0, -1.0)) / math.sqrt(6.0)
        tangent2 = side * np.array((0.0, 1.0, -1.0)) / math.sqrt(2.0)
    else:
        normal = side * np.array((1.0, 1.0, 0.0)) / math.sqrt(2.0)
        tangent1 = np.array((1.0, -1.0, 0.0)) / math.sqrt(2.0)
        tangent2 = np.array((0.0, 0.0, -float(side)))
    return normal, tangent1, tangent2


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


def characteristic(fields, coordinates, family, dimensions, side, incoming):
    normal, tangent1, tangent2 = boundary_frame(dimensions, side)
    tangents = (tangent1, tangent2)
    sign = 1.0 if incoming else -1.0
    derivative = lambda value: normal_derivative(value, coordinates, normal)

    khat = fields["z4c_Khat"]
    theta = fields["z4c_Theta"]
    gamma_n = project_vector(fields, "z4c_Gam", normal)
    metric_trace = sum(fields[symmetric_name("z4c_g", a, a)]
                       for a in range(3))
    a_trace = sum(fields[symmetric_name("z4c_A", a, a)]
                  for a in range(3))
    a_nn = project_tensor(fields, "z4c_A", normal, normal) - a_trace / 3.0
    dg_nn = derivative(
        project_tensor(fields, "z4c_g", normal, normal) - metric_trace / 3.0)
    dchi = derivative(fields["z4c_chi"])
    dalpha = derivative(fields["z4c_alpha"])
    beta_n = project_vector(fields, "z4c_beta", normal)
    db_n = derivative(beta_n)

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

    tangent = None
    if family.endswith("_1"):
        tangent = tangents[0]
    elif family.endswith("_2"):
        tangent = tangents[1]
    if tangent is not None:
        gamma_a = project_vector(fields, "z4c_Gam", tangent)
        db_a = derivative(project_vector(fields, "z4c_beta", tangent))
        if family.startswith("shift_transverse"):
            return sign * gamma_a + db_a
        a_na = project_tensor(fields, "z4c_A", normal, tangent)
        dg_na = derivative(project_tensor(fields, "z4c_g", normal, tangent))
        if family.startswith("constraint_transverse"):
            return -sign * 2.0 * a_na - gamma_a + dg_na

    if family == "tt_plus":
        a_tf = 0.5 * (
            project_tensor(fields, "z4c_A", tangent1, tangent1) -
            project_tensor(fields, "z4c_A", tangent2, tangent2))
        dg_tf = derivative(
            0.5 * (
                project_tensor(fields, "z4c_g", tangent1, tangent1) -
                project_tensor(fields, "z4c_g", tangent2, tangent2)))
        return -sign * 2.0 * a_tf + dg_tf
    if family == "tt_cross":
        a_tf = project_tensor(fields, "z4c_A", tangent1, tangent2)
        dg_tf = derivative(project_tensor(fields, "z4c_g", tangent1, tangent2))
        return -sign * 2.0 * a_tf + dg_tf
    raise RuntimeError("unsupported family {}".format(family))


def restrict_control(control_coordinates, control_fields, coordinates):
    indices = []
    for axis in range(3):
        index = np.searchsorted(control_coordinates[axis], coordinates[axis])
        if np.any(index >= len(control_coordinates[axis])) or not np.allclose(
                control_coordinates[axis][index], coordinates[axis],
                rtol=0.0, atol=1.0e-9):
            raise SystemExit("control grid does not contain measured grid")
        indices.append(index)
    target = np.ix_(indices[2], indices[1], indices[0])
    return {name: values[target] for name, values in control_fields.items()}


def trimmed_norm(values, trim):
    if any(2 * trim >= count for count in values.shape):
        raise SystemExit("too many boundary cells excluded")
    interior = values[trim:-trim, trim:-trim, trim:-trim]
    return float(np.linalg.norm(interior.ravel()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial")
    parser.add_argument("final")
    parser.add_argument("family")
    parser.add_argument("dimensions", type=int, choices=(2, 3))
    parser.add_argument("side", type=int, choices=(-1, 1))
    parser.add_argument("--control-final", required=True)
    parser.add_argument("--maximum-ratio", type=float, default=1.0)
    parser.add_argument(
        "--maximum-initial-incoming-ratio", type=float, default=0.005)
    parser.add_argument("--boundary-cells", type=int, default=4)
    args = parser.parse_args()

    coordinates0, initial = read_uniform_binary(args.initial)
    coordinatesf, final = read_uniform_binary(args.final)
    if any(not np.allclose(a, b) for a, b in zip(coordinates0, coordinatesf)):
        raise SystemExit("initial and final grids differ")
    control_coordinates, control_full = read_uniform_binary(args.control_final)
    control = restrict_control(control_coordinates, control_full, coordinatesf)

    outgoing = characteristic(
        initial, coordinates0, args.family, args.dimensions, args.side, False)
    initial_incoming = characteristic(
        initial, coordinates0, args.family, args.dimensions, args.side, True)
    incoming_raw = characteristic(
        final, coordinatesf, args.family, args.dimensions, args.side, True)
    control_incoming = characteristic(
        control, coordinatesf, args.family, args.dimensions, args.side, True)
    incoming = incoming_raw - control_incoming

    outgoing_norm = trimmed_norm(outgoing, args.boundary_cells)
    initial_incoming_norm = trimmed_norm(
        initial_incoming, args.boundary_cells)
    incoming_norm = trimmed_norm(incoming, args.boundary_cells)
    control_norm = trimmed_norm(control_incoming, args.boundary_cells)
    if outgoing_norm <= 1.0e-12:
        raise SystemExit("outgoing pulse norm was not observed")
    ratio = incoming_norm / outgoing_norm
    initial_incoming_ratio = initial_incoming_norm / outgoing_norm
    print(
        "family={} geometry={} outgoing_l2={:.8e} "
        "initial_incoming_l2={:.8e} initial_incoming_ratio={:.8e} "
        "incoming_l2={:.8e} control_l2={:.8e} ratio={:.8e}".format(
            args.family, "corner" if args.dimensions == 3 else "edge",
            outgoing_norm, initial_incoming_norm, initial_incoming_ratio,
            incoming_norm, control_norm, ratio))
    if initial_incoming_ratio >= args.maximum_initial_incoming_ratio:
        raise SystemExit(
            "initial incoming ratio {:.6e} exceeds {:.6e}".format(
                initial_incoming_ratio,
                args.maximum_initial_incoming_ratio))
    if ratio >= args.maximum_ratio:
        raise SystemExit(
            "reflected oblique ratio {:.6e} exceeds {:.6e}".format(
                ratio, args.maximum_ratio))


if __name__ == "__main__":
    main()
