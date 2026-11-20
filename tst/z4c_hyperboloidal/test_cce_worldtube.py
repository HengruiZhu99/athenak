"""Check ADM restoration, harmonic axes and worldtube derivatives."""

import os
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from athenak_cce.worldtube import convert


def checkpoint(path, time, schwarzschild_mass=None):
    n, o = 6, 1
    atlas = (
        b"athenak-chart-atlas-v2;inner_radius=0x1p-2;radii=0x1.8p-1,;"
        b"angular=1,1,;equiangular=1;outer_endpoint=0"
    )
    identity = (
        b"native-hyperboloidal-v1;n=6,6,6;overlap=1;real=8;fields=31;"
        b"chi-record=8;g.profile.flat=0"
    )
    tree = np.array([(0, 0, 0, 0, c + 1) for c in range(6)], dtype="=i4")
    size = n + 2 * o
    slots = size**3 + n**2
    data = np.zeros((6, slots, 39))
    u = np.tan((2 * (np.arange(n) + 0.5) / n - 1) * np.pi / 4)
    v, u, r = np.meshgrid(u, u, 0.25 + 0.5 * (np.arange(n) + 0.5) / n, indexing="ij")
    for face in range(6):
        ray = np.stack((np.full_like(u, (-1) ** face), u, v), axis=-1)
        ray = np.roll(
            ray / np.linalg.norm(ray, axis=-1, keepdims=True), face // 2, axis=-1
        )
        x = r[..., None] * ray
        omega = (1 - r * r) / 2
        raw = data[face, : size**3].reshape(size, size, size, 39)[o:-o, o:-o, o:-o]
        raw[..., 0] = 1 / omega**2
        raw[..., 22:25] = x
        raw[..., 24] += 0.2 * x[..., 2]
        # x is the CCE polar axis. This lapse has analytic (0,0) and (1,0) modes.
        raw[..., 18] = omega * (1 + 0.1 * x[..., 0] * (1 + time**2)) - (1 + r * r) / 2
        if schwarzschild_mass is not None:
            # T_Schwarzschild = t + tilt*r gives a nonzero radial shift
            # and off-diagonal spatial metric in native Cartesian components.
            q = schwarzschild_mass / (2 * r)
            lapse, spatial, tilt = (1 - q) / (1 + q), (1 + q) ** 2, 0.35
            radial = spatial**2 - (lapse * tilt) ** 2
            gamma = (spatial**2)[..., None, None] * np.eye(3) - ((lapse * tilt) ** 2)[
                ..., None, None
            ] * ray[..., :, None] * ray[..., None, :]
            determinant = spatial**4 * radial
            conformal_metric = gamma / np.cbrt(determinant)[..., None, None]
            raw[..., 0] = 1 / (omega**2 * np.cbrt(determinant))
            i, j = np.triu_indices(3)
            raw[..., 1:7] = conformal_metric[..., i, j] - np.eye(3)[i, j]
            raw[..., 22:25] = x - (lapse**2 * tilt / radial)[..., None] * ray
            raw[..., 18] = omega * lapse * spatial / np.sqrt(radial) - (1 + r * r) / 2
    with path.open("wb") as f:
        f.write(b"<par_end>\n")
        f.write(struct.pack("=ii9d38i2di", 6, 0, *([0] * 47), time, 0.1, 0))
        f.write(struct.pack("=Q", len(atlas)) + atlas + tree.tobytes() + bytes(6 * 4))
        f.write(
            struct.pack(
                "=8Q", 0x4859504552525354, 1, len(identity), slots * 39, 1, 1, 0, 0
            )
        )
        f.write(identity + struct.pack("=Q", slots * 39 * 8) + data.tobytes())


class WorldtubeTest(unittest.TestCase):
    def test_metric_modal(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            times = np.array([0, 0.21, 0.35, 0.68, 0.99])
            paths = [tmp / f"{i}.rst" for i in range(len(times))]
            for path, t in zip(paths, times):
                checkpoint(path, t)
            r, mass = 0.5, 0.25
            scale = (1 - r * r) / 2
            output = tmp / "metric.h5"
            convert(paths[::-1], output, r, mass, 3)
            with h5py.File(output) as f:
                self.assertEqual(len(f), 30)
                np.testing.assert_allclose(f["Lapse.dat"][:, 0], times / mass)
                k = np.sqrt(4 * np.pi / 3) * 0.1
                for field, expected in [
                    ("Lapse", k * r * (1 + times**2)),
                    ("DrLapse", k * mass * scale * (1 + times**2)),
                    ("DtLapse", k * r * 2 * times * mass),
                ]:
                    np.testing.assert_allclose(
                        f[field + ".dat"][:, 5], expected, atol=3e-13
                    )
                np.testing.assert_allclose(
                    f["gxx.dat"][:, 1], scale**2 * np.sqrt(4 * np.pi)
                )
                np.testing.assert_allclose(f["Drgxx.dat"][:, 1:], 0, atol=3e-13)
                np.testing.assert_allclose(f["Shiftz.dat"][:, 1:], 0, atol=3e-13)
                # Native z is output Y. Its sine dipole has imaginary m=+1.
                np.testing.assert_allclose(
                    f["Shifty.dat"][:, 8], 0.2 * r / scale * np.sqrt(2 * np.pi / 3)
                )

    @unittest.skipUnless(
        os.environ.get("SPECTRE_CCE_RELEASE"), "requires SpECTRE CCE release"
    )
    def test_schwarzschild_cce(self):
        from athenak_cce.run import run

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mass = 0.03
            paths = []
            for i in range(30):
                path = tmp / f"{i}.rst"
                checkpoint(path, i * 0.1 * mass, schwarzschild_mass=mass)
                paths.append(path)
            metric = tmp / "metric.h5"
            convert(paths, metric, 0.5, mass, 4)
            result = run(
                metric, Path(os.environ["SPECTRE_CCE_RELEASE"]), tmp / "exterior"
            )
            with h5py.File(result) as f:
                group = f[next(key for key in f if key.endswith(".cce"))]
                measured = -group["Psi2"][:, 1] / np.sqrt(4 * np.pi)
                maxima = {
                    key: float(np.max(np.abs(group[key][:, 1:])))
                    for key in ("Strain", "News", "Psi0", "Psi1", "Psi3", "Psi4")
                }
                print(
                    "Schwarzschild mass range",
                    measured.min(),
                    measured.max(),
                    "maximum mode amplitudes",
                    maxima,
                )
                self.assertGreater(len(measured), 10)
                np.testing.assert_allclose(measured, 1, atol=0.002)
                for key, peak in maxima.items():
                    # The leading Psi0 coefficient carries five powers of radius.
                    # Use the SpECTRE release regression's absolute tolerance for it.
                    tolerance = 3e-7 if key == "Psi0" else 2e-8
                    self.assertLess(peak, tolerance, key)


if __name__ == "__main__":
    unittest.main()
