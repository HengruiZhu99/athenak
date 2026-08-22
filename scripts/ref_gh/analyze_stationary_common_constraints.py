#!/usr/bin/env python3
"""Compute supplementary fixed-radius ADM-constraint norms from cbin files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_perturbed_trumpet_convergence import read_cbin


def parse_case(text: str) -> tuple[str, Path]:
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--case must be LABEL=PATH")
    return label, Path(path)


def norms(values: np.ndarray) -> tuple[float, float, float]:
    return (float(np.mean(np.abs(values))),
            float(np.sqrt(np.mean(values*values))),
            float(np.max(np.abs(values))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--r-min", action="append", type=float)
    parser.add_argument("--expected-time", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    radii = tuple(args.r_min) if args.r_min else (0.0, 0.125, 0.25, 0.5)
    rows = []
    for label, path in args.case:
        loaded = read_cbin(path)
        if loaded["variable_size"] != 8:
            raise ValueError(f"{path}: supplementary analysis requires binary64")
        if loaded["variables"] != ["adm_common_H", "adm_common_M2"]:
            raise ValueError(f"{path}: unexpected variables {loaded['variables']}")
        time = loaded["time"]
        if args.expected_time is not None and abs(time - args.expected_time) > 1.0e-12:
            raise ValueError(f"{path}: time {time} != {args.expected_time}")
        nz, ny, nx = loaded["data"].shape[-3:]
        bounds = loaded["bounds"]
        x = bounds[0][0] + (np.arange(nx) + 0.5) * (
            bounds[0][1] - bounds[0][0]) / nx
        y = bounds[1][0] + (np.arange(ny) + 0.5) * (
            bounds[1][1] - bounds[1][0]) / ny
        z = bounds[2][0] + (np.arange(nz) + 0.5) * (
            bounds[2][1] - bounds[2][0]) / nz
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        radius = np.sqrt(xx*xx + yy*yy + zz*zz)
        hamiltonian = loaded["data"][0]
        momentum2 = loaded["data"][1]
        for r_min in radii:
            mask = radius >= r_min
            selected_h = hamiltonian[mask]
            selected_m2 = momentum2[mask]
            negative_m2 = int(np.count_nonzero(selected_m2 < 0.0))
            momentum = np.sqrt(np.maximum(selected_m2, 0.0))
            h_l1, h_l2, h_linf = norms(selected_h)
            m_l1, m_l2, m_linf = norms(momentum)
            rows.append({
                "case": label,
                "time": f"{time:.17g}",
                "r_min": f"{r_min:.17g}",
                "cell_count": str(int(np.count_nonzero(mask))),
                "H_L1": f"{h_l1:.17g}",
                "H_L2": f"{h_l2:.17g}",
                "H_Linf": f"{h_linf:.17g}",
                "M_L1": f"{m_l1:.17g}",
                "M_L2": f"{m_l2:.17g}",
                "M_Linf": f"{m_linf:.17g}",
                "negative_M2_count": str(negative_m2),
                "minimum_M2": f"{float(np.min(selected_m2)):.17g}",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
