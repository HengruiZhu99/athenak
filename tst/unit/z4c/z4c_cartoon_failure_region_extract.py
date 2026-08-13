#!/usr/bin/env python3
"""Extract fail-visible Cartoon state near the historical axis failure region.

The input is AthenaK's native binary output after it has been loaded/evolved by
AthenaK.  Each selected row retains the complete Z4c and constraint carriers,
plus the two algebraic conformal constraints and K derived without modifying
the evolution state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


LAYERS = (0.5, 1.5, 2.5, 3.5, 4.5)
REQUIRED_Z4C = (
    "z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy",
    "z4c_gyz", "z4c_gzz", "z4c_Khat", "z4c_Axx", "z4c_Axy",
    "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz", "z4c_Gamx",
    "z4c_Gamy", "z4c_Gamz", "z4c_Theta", "z4c_alpha", "z4c_betax",
    "z4c_betay", "z4c_betaz", "z4c_Bx", "z4c_By", "z4c_Bz",
)
REQUIRED_CONSTRAINTS = (
    "con_C", "con_H", "con_M", "con_Z", "con_Mx", "con_My", "con_Mz",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0.0 else "-inf"
    return value


def packed_symmetric(values: dict[str, float], prefix: str):
    import numpy as np  # pylint: disable=import-outside-toplevel

    xx = values[f"{prefix}xx"]
    xy = values[f"{prefix}xy"]
    xz = values[f"{prefix}xz"]
    yy = values[f"{prefix}yy"]
    yz = values[f"{prefix}yz"]
    zz = values[f"{prefix}zz"]
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)), dtype=float)


def same_mesh(left: dict, right: dict, np) -> None:
    for key in ("time", "cycle", "n_mbs", "nx1_mb", "nx2_mb", "nx3_mb"):
        require(left[key] == right[key], f"binary carriers disagree in {key}")
    for key in ("mb_index", "mb_logical", "mb_geometry"):
        require(np.array_equal(left[key], right[key]),
                f"binary carriers disagree in {key}")


def extract_pair(z4c: dict, constraints: dict, z4c_path: Path,
                 constraints_path: Path, rank_file: int,
                 z_min: float, z_max: float, np) -> list[dict[str, object]]:
    same_mesh(z4c, constraints, np)
    for name in REQUIRED_Z4C:
        require(name in z4c["mb_data"], f"Z4c carrier omitted {name}")
    for name in REQUIRED_CONSTRAINTS:
        require(name in constraints["mb_data"], f"constraint carrier omitted {name}")

    rows: list[dict[str, object]] = []
    for block in range(z4c["n_mbs"]):
        reference = np.asarray(z4c["mb_data"]["z4c_chi"][block])
        require(reference.ndim == 3 and reference.shape[0] == 1,
                "Cartoon diagnostic expects one stored suppressed-direction plane")
        nx2, nx1 = reference.shape[1:]
        x1min, x1max, x2min, x2max, _, _ = z4c["mb_geometry"][block]
        dx1 = float(x1max - x1min) / nx1
        dx2 = float(x2max - x2min) / nx2
        require(dx1 > 0.0 and dx2 > 0.0, "invalid diagnostic cell spacing")
        for j in range(nx2):
            axial = float(x2min + (j + 0.5) * dx2)
            if not z_min <= abs(axial) <= z_max:
                continue
            for i in range(nx1):
                rho = float(x1min + (i + 0.5) * dx1)
                rho_over_h = rho / dx1
                matching = [layer for layer in LAYERS
                            if math.isclose(rho_over_h, layer,
                                            rel_tol=2e-12, abs_tol=2e-12)]
                if not matching:
                    continue
                z_values = {
                    name: float(z4c["mb_data"][name][block][0, j, i])
                    for name in REQUIRED_Z4C
                }
                con_values = {
                    name: float(constraints["mb_data"][name][block][0, j, i])
                    for name in REQUIRED_CONSTRAINTS
                }
                metric = packed_symmetric(z_values, "z4c_g")
                tracefree = packed_symmetric(z_values, "z4c_A")
                determinant_error = float(np.linalg.det(metric) - 1.0)
                try:
                    trace_a = float(np.trace(np.linalg.solve(metric, tracefree)))
                except np.linalg.LinAlgError:
                    trace_a = math.nan
                finite = all(math.isfinite(value) for value in
                             (*z_values.values(), *con_values.values(),
                              determinant_error, trace_a))
                rows.append({
                    "rank_file": rank_file,
                    "meshblock_record": block,
                    "logical_location": [int(value)
                                         for value in z4c["mb_logical"][block]],
                    "rho": rho,
                    "z": axial,
                    "h_rho": dx1,
                    "h_z": dx2,
                    "rho_over_h": matching[0],
                    "finite": finite,
                    "z4c": {name: encoded(value) for name, value in z_values.items()},
                    "constraints": {
                        name: encoded(value) for name, value in con_values.items()
                    },
                    "derived": {
                        "K": encoded(z_values["z4c_Khat"] +
                                     2.0 * z_values["z4c_Theta"]),
                        "det_conformal_metric_minus_one": encoded(determinant_error),
                        "trace_conformal_A": encoded(trace_a),
                    },
                    "sources": {
                        "z4c": str(z4c_path),
                        "constraints": str(constraints_path),
                    },
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--z4c", type=Path, nargs="+", required=True)
    parser.add_argument("--constraints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=1.3)
    args = parser.parse_args()
    require(0.0 <= args.z_min <= args.z_max, "invalid absolute-z selection")
    require(len(args.z4c) == len(args.constraints),
            "Z4c and constraint rank-file counts differ")

    sys.path.insert(0, str(args.source_dir / "vis/python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    import numpy as np  # pylint: disable=import-error,import-outside-toplevel

    rows: list[dict[str, object]] = []
    times: set[float] = set()
    cycles: set[int] = set()
    sources = []
    for rank_file, (z4c_path, constraints_path) in enumerate(
            zip(args.z4c, args.constraints, strict=True)):
        z4c = bin_convert.read_binary(str(z4c_path))
        constraints = bin_convert.read_binary(str(constraints_path))
        times.add(float(z4c["time"]))
        cycles.add(int(z4c["cycle"]))
        rows.extend(extract_pair(z4c, constraints, z4c_path, constraints_path,
                                 rank_file, args.z_min, args.z_max, np))
        sources.append({
            "z4c": {"path": str(z4c_path), "sha256": sha256(z4c_path)},
            "constraints": {"path": str(constraints_path),
                            "sha256": sha256(constraints_path)},
        })

    require(len(times) == 1 and len(cycles) == 1,
            "rank files do not describe one time slice")
    observed = {(row["rho_over_h"], 1 if row["z"] > 0.0 else -1)
                for row in rows}
    expected = {(layer, sign) for layer in LAYERS for sign in (-1, 1)}
    require(expected <= observed,
            f"former-failure-region coverage incomplete: missing {sorted(expected-observed)}")
    require(len(rows) == len({(row["rank_file"], row["meshblock_record"],
                              row["rho"], row["z"]) for row in rows}),
            "diagnostic extraction duplicated a physical cell")

    result = {
        "schema": 1,
        "claim_scope": "AthenaK_native_binary_post_load_state_not_termwise_rhs",
        "time": times.pop(),
        "cycle": cycles.pop(),
        "selection": {
            "rho_over_local_h": list(LAYERS),
            "absolute_z_interval": [args.z_min, args.z_max],
        },
        "nonfinite_encoding": ["nan", "+inf", "-inf"],
        "sources": sources,
        "rows": sorted(rows, key=lambda row: (row["rho_over_h"], row["z"],
                                               row["rank_file"],
                                               row["meshblock_record"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(f"Cartoon former-failure-region extraction passed ({len(rows)} rows)")


if __name__ == "__main__":
    main()
