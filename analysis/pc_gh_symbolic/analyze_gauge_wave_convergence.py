#!/usr/bin/env python3
"""Fail closed on the all-sector PC-GH shifted gauge-wave convergence ladder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def read_table(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(encoding="utf-8") as stream:
        header_line = stream.readline()
        if not header_line.startswith("# "):
            raise AssertionError(f"{path}: missing named header")
        header = header_line[2:].split()
        rows = [dict(zip(header, map(float, line.split()), strict=True))
                for line in stream if line.strip() and not line.startswith("#")]
    rows.sort(key=lambda row: row["nx1"])
    if len(rows) < 3:
        raise AssertionError(f"{path}: need at least three resolutions, found {len(rows)}")
    if len({row["nx1"] for row in rows}) != len(rows):
        raise AssertionError(f"{path}: duplicate nx1 values")
    return header, rows


def adjacent_orders(rows: list[dict[str, float]], key: str) -> list[float]:
    result = []
    for coarse, fine in zip(rows, rows[1:]):
        ec = coarse[key]
        ef = fine[key]
        if not (math.isfinite(ec) and math.isfinite(ef) and ec > 0.0 and ef > 0.0):
            raise AssertionError(f"{key}: nonpositive or nonfinite non-exact error")
        result.append(math.log(ec/ef)/math.log(fine["nx1"]/coarse["nx1"]))
    return result


def audit_group(name: str, keys: list[str], rows: list[dict[str, float]],
                min_order: float, exact_tol: float, allow_exact_all: bool = False) -> None:
    nontrivial: list[tuple[float, str, list[float]]] = []
    exact = 0
    for key in keys:
        values = [row[key] for row in rows]
        if max(values) <= exact_tol:
            exact += 1
            continue
        orders = adjacent_orders(rows, key)
        nontrivial.append((min(orders), key, orders))
    if not nontrivial:
        if allow_exact_all and exact == len(keys):
            print(f"{name:12s} exact_invariant components={exact} tol={exact_tol:.1e}")
            return
        raise AssertionError(f"{name}: no nontrivial component exercises this sector")
    worst_order, worst_key, orders = min(nontrivial)
    print(f"{name:12s} min_order={worst_order:.6f} worst={worst_key} "
          f"adjacent={','.join(f'{order:.6f}' for order in orders)} "
          f"exact_components={exact}")
    if worst_order < min_order:
        raise AssertionError(
            f"{name}: {worst_key} order {worst_order:.6f} < {min_order:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--min-order", type=float, default=1.8)
    parser.add_argument("--exact-tol", type=float, default=1.0e-12)
    args = parser.parse_args()
    header, rows = read_table(args.table)

    required_metadata = {
        "nx1", "nx2", "nx3", "time", "cycles", "aggregate_l1", "aggregate_linf"
    }
    if not required_metadata.issubset(header):
        raise AssertionError("missing gauge-wave metadata columns")
    if any(row["nx2"] != 1 or row["nx3"] != 1 for row in rows):
        raise AssertionError("current exact shifted-wave audit requires a one-dimensional mesh")

    state_names = [
        "chi", "gtxx", "gtxy", "gtxz", "gtyy", "gtyz", "gtzz", "K",
        "Atxx", "Atxy", "Atxz", "Atyy", "Atyz", "Atzz",
        "Lamx", "Lamy", "Lamz", "pi", "A", "betax", "betay", "betaz",
        "X1", "X2", "X3",
        "Q1xx", "Q1xy", "Q1xz", "Q1yy", "Q1yz", "Q1zz",
        "Q2xx", "Q2xy", "Q2xz", "Q2yy", "Q2yz", "Q2zz",
        "Q3xx", "Q3xy", "Q3xz", "Q3yy", "Q3yz", "Q3zz",
        "Y1", "Y2", "Y3",
        "B11", "B12", "B13", "B21", "B22", "B23", "B31", "B32", "B33",
    ]
    state_groups = {
        "primary": state_names[:22],
        "X": state_names[22:25],
        "Q": state_names[25:43],
        "Y": state_names[43:46],
        "B": state_names[46:55],
    }
    constraint_groups = {
        "GH": ["Cperp", "Zx", "Zy", "Zz"],
        "ADM": ["H", "Mhatx", "Mhaty", "Mhatz"],
        "reduction": ["red_X", "red_Q", "red_Y", "red_B"],
        "curl": ["curl_X", "curl_Q", "curl_Y", "curl_B"],
    }
    expected = []
    for names in state_groups.values():
        expected.extend(f"pcgh_{name}_{norm}" for name in names for norm in ("rms", "linf"))
    for names in constraint_groups.values():
        expected.extend(f"pcgh_{name}_{norm}" for name in names for norm in ("rms", "linf"))
    missing = sorted(set(expected) - set(header))
    if missing:
        raise AssertionError(f"missing required columns: {', '.join(missing)}")

    print("resolutions", " ".join(str(int(row["nx1"])) for row in rows))
    audit_group("aggregate", ["aggregate_l1", "aggregate_linf"], rows,
                args.min_order, args.exact_tol)
    for group, names in state_groups.items():
        audit_group(group, [f"pcgh_{name}_{norm}" for name in names
                            for norm in ("rms", "linf")],
                    rows, args.min_order, args.exact_tol)
    for group, names in constraint_groups.items():
        audit_group(group, [f"pcgh_{name}_{norm}" for name in names
                            for norm in ("rms", "linf")],
                    rows, args.min_order, args.exact_tol,
                    allow_exact_all=(group == "curl"))

    projection = [row["pcgh_projection_rms"] for row in rows]
    if not all(fine < coarse for coarse, fine in zip(projection, projection[1:])):
        raise AssertionError("algebraic projection correction does not decrease monotonically")
    print("projection   monotonically_decreasing=" + ",".join(f"{x:.6e}" for x in projection))
    print("PASS: shifted harmonic gauge wave converges in every exercised PC-GH sector")


if __name__ == "__main__":
    main()
