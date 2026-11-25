"""Calibrate an IrisK handoff along one of the two photon-ring family coordinates."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import root, root_scalar

from campaign import ROOT, angular_grid, load
from coverage import audit


def calibrate(key):
    ref = load("reference.json")
    axis = load("scan.json")["axis"]
    data = ROOT / "calibration"
    sys.path.insert(0, str(data))
    os.environ.setdefault("IRIS_LOBATTO_OPERATORS", str((data / "read.sh").resolve()))
    from measure import Slice
    from solve import Launch, solve, save

    dest = ROOT / "cases" / key
    target = float(key)
    width = target if axis == "width" else ref["w"]
    spin = ref["spin_ratio"] if axis == "width" else target
    mass = ref["M_adm"]

    def evaluate(energy, angular):
        signed = bool(abs(angular) <= energy * ref["b_halfwidth"])
        p = Launch(
            ref["R0"],
            width,
            energy,
            angular,
            ref["b_halfwidth"],
            ref["nu_halfwidth"],
            signed,
        )
        outer = p.R0 + 1.25 * p.w if axis == "width" else ref["outer"]
        meta, coeff = solve(p, ref["nr"], ref["nt"], ref["quadrature"], outer)
        geom = Slice(meta, coeff)
        m = geom.adm_mass()
        _, j = geom.surface(16, 2 * ref["nt"])
        meta.update(
            M_adm=m, J_adm=j, width=2 * width, spin_ratio=j / m**2, target_spin=spin
        )
        return meta, coeff

    if axis == "width":
        samples = [
            ref,
            *(json.loads(f.read_text()) for f in dest.parent.glob("*/ring.adm.json")),
        ]
        a = min(samples, key=lambda s: abs(s["w"] - width))
        other = [s for s in samples if s["w"] != a["w"]]
        energy = a["E_gamma"]
        if other:
            b = min(other, key=lambda s: abs(s["w"] - width))
            energy += (width - a["w"]) * (b["E_gamma"] - energy) / (b["w"] - a["w"])

        def residual(x):
            meta, _ = evaluate(*x)
            return [meta["M_adm"] / mass - 1, meta["J_adm"] / mass**2 - spin]

        fit = root(residual, [energy, ref["J_z"]], tol=1e-10)
        meta, coeff = evaluate(*fit.x)
        error = max(abs(meta["M_adm"] / mass - 1), abs(meta["spin_ratio"] - spin))
    else:
        energy = ref["E_gamma"]
        angular = 0.0
        if spin:
            guess = spin * mass**2
            fit = root_scalar(
                lambda j: evaluate(energy, j)[0]["spin_ratio"] - spin,
                x0=guess,
                x1=guess * 1.0001,
                method="secant",
                xtol=1e-10,
            )
            if not fit.converged:
                raise RuntimeError("ADM spin calibration did not converge")
            angular = fit.root
        meta, coeff = evaluate(energy, angular)
        error = abs(meta["spin_ratio"] - spin)
    if not np.isfinite(error) or error > 1e-9 or not meta["margin_lower_bound"] > 0:
        raise ValueError(f"calibration failed with residual {error}")
    report = audit(load("mesh-bounds.json"), meta)
    # coverage.json is the marker consumed by the dispatcher. Publish it last.
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        tmp = Path(tmp)
        save(tmp / "ring.adm", meta, coeff)
        sign = angular_grid(meta)
        (tmp / "settings").write_text(f"sign={sign}\n")
        (tmp / "coverage.json").write_text(json.dumps(report))
        for name in ("ring.adm", "ring.adm.json", "settings", "coverage.json"):
            (tmp / name).replace(dest / name)


if __name__ == "__main__":
    try:
        calibrate(sys.argv[1])
    except (RuntimeError, ValueError) as exc:
        if not any(
            s in str(exc) for s in ("unable to certify", "forbidden null direction")
        ):
            raise
        dest = ROOT / "cases" / sys.argv[1]
        path = dest / "invalid.tmp"
        path.write_text(json.dumps(dict(reason=str(exc), parameter=sys.argv[1])))
        path.replace(dest / "invalid.json")
