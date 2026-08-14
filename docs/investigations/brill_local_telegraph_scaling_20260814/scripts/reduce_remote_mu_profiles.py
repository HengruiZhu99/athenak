#!/usr/bin/env python3
"""Read selected AthenaK mu binaries in place and emit compact JSON profiles.

This script is intended to run on Perlmutter with PYTHONPATH pointing at the
authenticated AthenaK ``vis/python`` directory.  It never writes beside the
immutable campaign evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

# The binary reader imports h5py only for conversion routines that are unused
# here.  Perlmutter's current login-node h5py wheel is ABI-incompatible with
# its NumPy module, so keep this read-only path independent of that optional
# dependency.
sys.modules["h5py"] = types.ModuleType("h5py")
import bin_convert


ROOT = Path(
    "/pscratch/sd/h/hzhu/"
    "axisymmetric-cartoon-r4-brill-local-telegraph-mu-2a8ad80e-v2-20260814"
)
SELECTIONS = {
    # File zero is emitted before the first RHS evaluation, while the
    # diagnostic work array still has its constructor value.  File one is the
    # first evolved, physically evaluated mu profile.
    "max_domain_abs_K": (1, 11, 53),
    "local_abs_K": (1, 12, 27),
    "local_extrinsic_curvature_norm": (1, 13, 25),
    "local_chi_gradient_norm": (1, 15, 43),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(path: Path) -> dict[str, object]:
    data = bin_convert.read_binary(str(path))
    variable = "z4c_telegraph_mu"
    if list(data["var_names"]) != [variable]:
        raise RuntimeError(f"unexpected variables in {path}: {data['var_names']}")

    axis: list[tuple[float, float, float, int]] = []
    equator: list[tuple[float, float, float, int]] = []
    all_min = float("inf")
    all_max = -float("inf")
    nonfinite = 0
    negative = 0
    zero = 0
    for block, geometry in enumerate(np.asarray(data["mb_geometry"])):
        rho_lo, rho_hi, z_lo, z_hi, _y_lo, _y_hi = map(float, geometry)
        values = np.asarray(data["mb_data"][variable][block])[0]
        nz, nrho = values.shape
        drho = (rho_hi - rho_lo) / nrho
        dz = (z_hi - z_lo) / nz
        rho = rho_lo + (np.arange(nrho) + 0.5) * drho
        z = z_lo + (np.arange(nz) + 0.5) * dz
        level = int(data["mb_logical"][block, 3])

        finite = np.isfinite(values)
        nonfinite += int(values.size - np.count_nonzero(finite))
        negative += int(np.count_nonzero(values[finite] < 0.0))
        zero += int(np.count_nonzero(values[finite] == 0.0))
        if np.any(finite):
            all_min = min(all_min, float(np.min(values[finite])))
            all_max = max(all_max, float(np.max(values[finite])))

        if abs(rho_lo) <= 1.0e-12:
            for j, z_value in enumerate(z):
                axis.append((float(z_value), float(values[j, 0]), 0.5 * drho, level))
        if z_lo <= 0.0 <= z_hi:
            j = int(np.argmin(np.abs(z)))
            for i, rho_value in enumerate(rho):
                equator.append((float(rho_value), float(values[j, i]), float(z[j]), level))

    axis.sort(key=lambda row: row[0])
    equator.sort(key=lambda row: row[0])
    if not axis or not equator:
        raise RuntimeError(f"missing axis/equatorial profile in {path}")
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "time": float(data["time"]),
        "cycle": int(data["cycle"]),
        "meshblocks": int(data["n_mbs"]),
        "field_min": all_min,
        "field_max": all_max,
        "nonfinite_cells": nonfinite,
        "negative_cells": negative,
        "zero_cells": zero,
        "axis_profile": [
            {"z": z, "mu": mu, "rho_center": rho, "level": level}
            for z, mu, rho, level in axis
        ],
        "equatorial_profile": [
            {"rho": rho, "mu": mu, "z_center": z, "level": level}
            for rho, mu, z, level in equator
        ],
    }


def main() -> None:
    cases: dict[str, list[dict[str, object]]] = {}
    for prescription, indices in SELECTIONS.items():
        rank = ROOT / "run/cases" / prescription / "bin/rank_00000000"
        files = sorted(rank.glob("*.telegraph_mu.*.bin"))
        if len(files) <= max(indices):
            raise RuntimeError(
                f"{prescription}: only {len(files)} files, need index {max(indices)}"
            )
        cases[prescription] = [profile(files[index]) for index in indices]
    payload = {
        "schema": "athenak_brill_local_telegraph_mu_selected_profiles_v1",
        "qualification_claim": False,
        "remote_root": str(ROOT),
        "selections": {key: list(value) for key, value in SELECTIONS.items()},
        "cases": cases,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
