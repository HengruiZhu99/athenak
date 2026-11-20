"""Re-evaluate fitted surfaces with local polynomial interpolation of native ADM nodes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from ring_surface import NativePatch, measure, read_binary


class LocalPolynomial:
    """Use a centered tensor-product Lagrange stencil on the signed meridional grid."""

    def __init__(self, patch, degree):
        self.dx = patch.dx
        self.bound = patch.bound
        self.count = degree + 1
        n = round(self.bound / self.dx)
        coords = np.arange(-n, n+1) * self.dx
        # Interpolating splines recover their input at the native grid vertices.
        self.fields = np.stack([s(coords, coords) for s in patch.splines], axis=-1)
        self.offsets = np.arange(self.count)
        self.divisors = np.array([np.prod(i - self.offsets[self.offsets != i]) for i in self.offsets])

    def stencil(self, coords):
        cell = (coords + self.bound) / self.dx
        first = np.floor(cell).astype(int) - self.count//2 + 1
        first = np.clip(first, 0, len(self.fields)-self.count)
        local = cell-first
        weights = np.column_stack([np.prod(local[:, None] - self.offsets[self.offsets != i], axis=1)
                                   for i in self.offsets]) / self.divisors
        return first[:, None]+self.offsets, weights

    def tensor(self, rho, z, start):
        zi, wz = self.stencil(z)
        ri, wr = self.stencil(rho)
        nodes = self.fields[zi[:, :, None], ri[:, None, :], start:start+6]
        values = np.einsum("ni,nj,nijc->nc", wz, wr, nodes)
        return values[:, [[0, 1, 2], [1, 3, 4], [2, 4, 5]]]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("fits", type=Path, nargs="+")
    args = cli.parse_args()
    frame = read_binary(args.snapshot)
    native = NativePatch(frame)
    theta = np.linspace(1.e-5, np.pi-1.e-5, 2049)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["frame","time","dx","fit","interpolation","target","min_hThetaPlus",
                     "max_hThetaPlus","max_hThetaMinus","angular_checks"])
    for label, patch in [("cubic_spline", native), ("local_cubic", LocalPolynomial(native, 3)),
                         ("local_quintic", LocalPolynomial(native, 5))]:
        for fit in args.fits:
            r = json.loads(fit.read_text().splitlines()[0])
            h, plus, minus, _, _ = measure(patch, r["coeff"], theta)
            writer.writerow([args.snapshot.name, frame["time"], patch.dx, fit.name, label, r["target"],
                             min(h*plus), max(h*plus), max(h*minus), len(theta)])
