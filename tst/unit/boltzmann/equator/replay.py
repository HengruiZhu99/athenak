"""Check archived marginal and interior surfaces using full or reflected ADM data."""
import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ring_surface import NativePatch, measure, read_binary


def upper_only(frame):
    selected = np.asarray(frame["mb_geometry"])[:, 2] >= 0
    header = list(frame["header"])
    begin = header.index("<mesh>")
    for i in range(begin + 1, len(header)):
        if header[i].startswith("<"):
            break
        if re.match(r"ix2_bc\s*=", header[i]):
            header[i] = "ix2_bc = reflect"
    return dict(frame, header=header, mb_geometry=np.asarray(frame["mb_geometry"])[selected],
                mb_data={k: [b for b, use in zip(v, selected) if use]
                         for k, v in frame["mb_data"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("slices", type=Path, nargs="+")
    args = parser.parse_args()
    archived = json.loads(args.reference.read_text())
    theta = np.r_[0., np.linspace(1.e-4, np.pi-1.e-4, 1025), np.pi]
    x, weights = np.polynomial.legendre.leggauss(192)
    for path in sorted(args.slices):
        data = read_binary(path)
        for coverage, frame in [("full", data), ("folded", upper_only(data))]:
            patch = NativePatch(frame)
            for name, shape in [("inner", archived), ("marginal", archived["enclosing_horizon"])]:
                start = perf_counter()
                c = np.array(shape["coeff"])
                h, plus, minus, _, _ = measure(patch, c, theta)
                _, _, _, da, integrand = measure(patch, c, np.arccos(x))
                area = 2*np.pi*np.dot(weights, da/np.sqrt(1-x*x))
                angular = .25*np.dot(weights, da*integrand/np.sqrt(1-x*x))
                irr2 = area / (16*np.pi)
                mass = np.sqrt(irr2 + angular**2 / (4*irr2))
                print(json.dumps(dict(file=path.name, coverage=coverage, shape=name,
                    time=data["time"], max_out=float(max(h*plus)), min_out=float(min(h*plus)),
                    max_in=float(max(h*minus)), trapped=bool(max(plus)<0 and max(minus)<0),
                    cells=float(min(h)/patch.dx), area=area, mass=mass, spin=angular/mass**2,
                    seconds=perf_counter()-start)))
if __name__ == "__main__":
    main()
