"""Search earlier saved slices and cross-check strict expansion signs on the whole meridian."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ring_interpolation_check import LocalPolynomial
from ring_surface import NativePatch, measure, read_binary, solve_surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("snapshots", type=Path, nargs="+")
    p.add_argument("--order", default=40, type=int)
    p.add_argument("--target", default=-.002, type=float)
    args = p.parse_args()
    theta = np.r_[0., np.linspace(1e-5, np.pi-1e-5, 4097), np.pi]
    seed = json.loads(args.seed.read_text())["coeff"]
    rows = []
    for path in reversed(sorted(args.snapshots)):
        frame = read_binary(path)
        try:
            patch = NativePatch(frame)
            fit = solve_surface(patch, seed[0], args.target, args.order, 50, seed)
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(dict(file=path.name, cycle=frame["cycle"], time=frame["time"],
                             all_negative=False, failure=str(error)))
            continue
        checks = {}
        for name, interp in [("spline", patch), ("cubic", LocalPolynomial(patch,3)),
                             ("quintic", LocalPolynomial(patch,5))]:
            h, plus, minus, _, _ = measure(interp, fit["coeff"], theta)
            checks[name] = [float(np.max(h*plus)),float(np.max(h*minus))]
        passed = all(max(pair)<0 for pair in checks.values())
        rows.append(dict(file=path.name, cycle=frame["cycle"], time=frame["time"],
                         checks=checks, angles=theta.size, all_negative=passed, **fit))
        print(path.name, frame["time"], checks, passed, flush=True)
        if passed:
            seed = fit["coeff"]
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
if __name__ == "__main__":
    main()
