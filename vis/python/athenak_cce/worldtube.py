"""Convert shared, double-precision native checkpoints to SpECTRE MetricModal HDF5."""

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np
from scipy.special import sph_harm_y

FIELDS = ("gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Shiftx", "Shifty", "Shiftz", "Lapse")


def metadata(raw):
    return dict(part.split("=", 1) for part in raw.decode().split(";")[1:] if "=" in part)


def header(path):
    with path.open("rb") as f:
        while True:
            line = f.readline(65536)
            if line == b"<par_end>\n":
                break
            if not line or f.tell() > 16 << 20:
                raise ValueError(f"{path}: missing native parameter header")
        fixed = struct.Struct("=ii9d38i2di")
        n, root, *rest = fixed.unpack(f.read(fixed.size))
        atlas = f.read(struct.unpack("=Q", f.read(8))[0])
        if atlas.split(b";")[0] not in (
            b"athenak-chart-atlas-v2",
            b"athenak-chart-atlas-v3",
        ):
            raise ValueError("unsupported native atlas")
        tree = np.frombuffer(f.read(n * 20), "=i4").reshape(n, 5)
        f.seek(n * 4, 1)
        magic, version, size, count, _, _, _, per_rank = struct.unpack("=8Q", f.read(64))
        if (magic, version, per_rank) != (0x4859504552525354, 1, 0):
            raise ValueError("expected a shared version-1 hyperboloidal checkpoint")
        identity = f.read(size)
        (stride,) = struct.unpack("=Q", f.read(8))
        if (
            count % 39
            or stride != count * 8
            or path.stat().st_size != f.tell() + n * stride
        ):
            raise ValueError(f"{path}: inconsistent record extent")
        data = np.memmap(
            path, dtype="=f8", mode="r", offset=f.tell(), shape=(n, count // 39, 39)
        )
    return rest[-3], root, tree, atlas, identity, data


def omega(r, p):
    old = (1 - r * r) / 2
    if not int(p["g.profile.flat"]):
        return old
    lo, hi, plateau = (
        float.fromhex(p["g.profile." + k])
        for k in ("flat_radius", "outer_radius", "plateau")
    )
    if r <= lo:
        return plateau
    if r >= hi:
        return old
    from scipy.special import betainc

    return plateau + betainc(7, 7, (r - lo) / (hi - lo)) * (old - plateau)


def weights(nodes, center, order):
    """Polynomial weights at center on a scaled nonuniform stencil."""
    x = nodes - center
    scale = np.max(abs(x))
    rhs = np.zeros(len(x))
    rhs[order] = 1
    return np.linalg.solve(np.vander(x / scale, increasing=True).T, rhs) / scale**order


def adm(native, x, conformal, spatial_scale):
    """Physical ADM for T=t/M and X=(y,z,x)/(M*spatial_scale)."""
    chi = native[..., 0]
    a, b, c, d, e = np.moveaxis(native[..., 1:6], -1, 0)
    a, d = a + 1, d + 1
    minor = a * d - b * b
    if np.any(chi <= 0) or np.any(a <= 0) or np.any(minor <= 0):
        raise ValueError("nonpositive spatial metric at worldtube")
    f = (1 - 2 * b * c * e + a * e * e + d * c * c) / minor
    # Cyclic permutation puts the head-on x axis at the harmonic north pole.
    g = (
        np.stack((d, e, b, f, c, a), axis=-1)
        * (spatial_scale / conformal[..., None]) ** 2
        / chi[..., None]
    )
    shift = (native[..., 22:25] - x)[..., [1, 2, 0]] / spatial_scale
    lapse = (native[..., 18] + (1 + np.sum(x * x, axis=-1)) / 2) / conformal
    if np.any(lapse <= 0):
        raise ValueError("nonpositive worldtube lapse")
    return np.concatenate((g, shift, lapse[..., None]), axis=-1)


class Shell:
    def __init__(self, root, tree, atlas, identity, radius, mass, ell):
        a, p = metadata(atlas), metadata(identity)
        if p.get("real") != "8" or p.get("fields") != "31" or p.get("chi-record") != "8":
            raise ValueError("unsupported native restart field layout")
        self.scale = omega(radius, p)
        self.radius = radius / (mass * self.scale)
        self.layout = (root, tree.tobytes(), atlas, identity)
        nx, ny, nz = map(int, p["n"].split(","))
        if nx < 2 or min(ny, nz) < 1:
            raise ValueError("insufficient native worldtube resolution")
        ov = int(p["overlap"])
        self.shape = (nz + 2 * ov, ny + 2 * ov, nx + 2 * ov)
        knots = np.array(
            [
                float.fromhex(v)
                for v in (a["inner_radius"] + "," + a["radii"]).strip(",").split(",")
            ]
        )
        widths = np.array(
            list(map(int, a["radial_blocks"].strip(",").split(",")))
            if "radial_blocks" in a
            else [1] * (len(knots) - 1)
        )
        layer = np.searchsorted(knots, radius, side="right") - 1
        if not 0 <= layer < len(widths) or (layer == 0 and knots[0] == 0):
            raise ValueError("worldtube must lie in a spherical atlas layer")
        self.blocks = []
        directions, areas = [], []
        angular = np.array(list(map(int, a["angular"].strip(",").split(","))))
        area = np.zeros(6)
        for gid, (i, j, k, level, chart) in enumerate(tree):
            if not chart:
                continue
            factor = 2 ** int(level - root)
            width = widths[layer] * factor
            local = i - int(sum(widths[:layer])) * factor
            dr = (knots[layer + 1] - knots[layer]) / width
            start = knots[layer] + local * dr
            selected = min(width - 1, int((radius - knots[layer]) / dr))
            if local != selected:
                continue
            step = np.array([ny, nz]) * angular * factor
            u, v = np.meshgrid(
                -1 + 2 * (j * ny + np.arange(ny) + 0.5) / step[0],
                -1 + 2 * (k * nz + np.arange(nz) + 0.5) / step[1],
            )
            jac = np.ones_like(u)
            if int(a["equiangular"]):
                u, v = np.tan(np.pi * u / 4), np.tan(np.pi * v / 4)
                jac = (np.pi / 4) ** 2 * (1 + u * u) * (1 + v * v)
            axis = (chart - 1) // 2
            n = np.stack((np.full_like(u, 1 if chart % 2 else -1), u, v), axis=-1)
            norm = np.linalg.norm(n, axis=-1)
            n = np.roll(n / norm[..., None], axis, axis=-1)
            areas.extend((4 * jac / (np.prod(step) * norm**3)).ravel())
            directions.extend(n.reshape(-1, 3))
            area[chart - 1] += 1 / (np.prod(angular) * factor**2)
            offset = 0 if layer == len(widths) - 1 and int(a["outer_endpoint"]) else 0.5
            nodes = start + dr * (np.arange(nx) + offset) / nx
            idx = np.sort(np.argsort(abs(nodes - radius))[: min(6, nx)])
            r = nodes[idx]
            self.blocks.append(
                (
                    gid,
                    slice(ov, ov + nz),
                    slice(ov, ov + ny),
                    idx + ov,
                    r[:, None, None, None] * n,
                    np.array([omega(z, p) for z in r]),
                    weights(r, radius, 0),
                    weights(r, radius, 1) * mass * self.scale,
                )
            )
        if not np.allclose(area, 1):
            raise ValueError("incomplete worldtube angular coverage")
        n = np.array(directions)
        self.modes = [
            (degree, m) for degree in range(ell + 1) for m in range(-degree, degree + 1)
        ]
        basis = np.column_stack(
            [
                sph_harm_y(degree, m, np.arccos(n[:, 0]), np.arctan2(n[:, 2], n[:, 1]))
                for degree, m in self.modes
            ]
        )
        weighted = basis.conj().T * np.array(areas)
        gram = weighted @ basis
        if np.linalg.cond(gram) > 1e8:
            raise ValueError("angular resolution cannot support requested modes")
        self.project = np.linalg.solve(gram, weighted)

    def sample(self, data):
        values, radial = [], []
        count = int(np.prod(self.shape))
        for gid, k, j, i, x, om, w, dw in self.blocks:
            bulk = data[gid, :count].reshape(self.shape + (39,))
            native = np.moveaxis(bulk[k, j][:, :, i], 2, 0)
            fields = adm(native, x, om[:, None, None], self.scale)
            values.extend(np.einsum("r,rkjf->kjf", w, fields).reshape(-1, 10))
            radial.extend(np.einsum("r,rkjf->kjf", dw, fields).reshape(-1, 10))
        return self.project @ values, self.project @ radial


def convert(paths, output, radius, mass, ell):
    if not np.isfinite(mass) or mass <= 0 or not 0 < radius < 1 or ell < 2:
        raise ValueError("invalid mass, compact radius or bandlimit")
    samples = {}
    shell = None
    for path in paths:
        t, root, tree, atlas, identity, data = header(path)
        if shell is None:
            shell = Shell(root, tree, atlas, identity, radius, mass, ell)
        if shell.layout != (root, tree.tobytes(), atlas, identity):
            raise ValueError("mixed identities or changing topology in checkpoint series")
        sample = shell.sample(data)
        if t in samples and not np.array_equal(samples[t], sample):
            raise ValueError(f"conflicting checkpoints at t={t}")
        samples[t] = sample
    if len(samples) < 5:
        raise ValueError("need at least five distinct checkpoints")
    times = np.array(sorted(samples)) / mass
    values, radial = np.moveaxis(np.array([samples[t] for t in sorted(samples)]), 1, 0)
    temporal = np.empty_like(values)
    for j, t in enumerate(times):
        idx = np.sort(np.argsort(abs(times - t))[:5])
        temporal[j] = np.einsum("t,tmf->mf", weights(times[idx], t, 1), values[idx])
    if not all(np.isfinite(a).all() for a in (values, radial, temporal)):
        raise ValueError("nonfinite worldtube data")
    legend = ["time"] + [
        f"{part}({degree},{m})" for degree, m in shell.modes for part in ("Re", "Im")
    ]
    with h5py.File(output, "x") as f:
        f.attrs["ExtractionRadius"] = shell.radius
        f.attrs["mass_unit"] = mass
        f.attrs["compact_radius"] = radius
        f.attrs["axes"] = "X=AthenaK y, Y=AthenaK z, Z=AthenaK x"
        for prefix, arr in (("", values), ("Dr", radial), ("Dt", temporal)):
            for i, name in enumerate(FIELDS):
                packed = np.ascontiguousarray(arr[:, :, i]).view(float)
                ds = f.create_dataset(
                    prefix + name + ".dat", data=np.column_stack((times, packed))
                )
                ds.attrs.create("Legend", legend, dtype=h5py.string_dtype("ascii"))
    return shell.radius


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--radius", type=float, required=True, help="native compact coordinate radius"
    )
    p.add_argument("--mass", type=float, required=True, help="M0 in code units")
    p.add_argument("--lmax", type=int, default=8)
    p.add_argument("checkpoints", type=Path, nargs="+")
    a = p.parse_args()
    print(
        "SpECTRE ExtractionRadius =",
        convert(a.checkpoints, a.output, a.radius, a.mass, a.lmax),
    )


if __name__ == "__main__":
    main()
