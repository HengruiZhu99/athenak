#!/usr/bin/env python3
"""Analyze deterministic random-noise PC-GH Minkowski resolution ladders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


HISTORY_NAMES = [
    "time", "dt", "Cperp", "Z", "H", "Mhat",
    "redX", "redQ", "redY", "redB",
    "curlX", "curlQ", "curlY", "curlB",
    "detg", "trA", "trQ", "projection", "W", "L", "rhs", "Volume",
]
CHANNELS = HISTORY_NAMES[2:14]


def read_named_table(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        line = stream.readline()
        if not line.startswith("# "):
            raise AssertionError(f"{path}: missing named header")
        header = line[2:].split()
        return [dict(zip(header, map(float, row.split()), strict=True))
                for row in stream if row.strip() and not row.startswith("#")]


def read_history(path: Path) -> list[dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or "Athena++ history data" not in lines[0]:
        raise AssertionError(f"{path}: malformed history file")
    rows = [dict(zip(HISTORY_NAMES, map(float, line.split()), strict=True))
            for line in lines[2:] if line.strip()]
    if len(rows) < 5 or rows[0]["time"] != 0.0:
        raise AssertionError(f"{path}: history must include t=0 and at least five samples")
    if any(row["Volume"] <= 0.0 for row in rows):
        raise AssertionError(f"{path}: nonpositive diagnostic volume")
    return rows


def fit_slope(x: list[float], y: list[float]) -> float:
    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)
    denominator = sum((value - xmean)**2 for value in x)
    if denominator == 0.0:
        raise AssertionError("degenerate fit interval")
    return sum((a - xmean)*(b - ymean) for a, b in zip(x, y, strict=True))/denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_table", type=Path)
    parser.add_argument("--history", action="append", required=True,
                        help="resolution:path (repeat at least three times)")
    parser.add_argument("--max-amplification", type=float, default=10.0)
    parser.add_argument("--max-fit-growth", type=float, default=0.1)
    args = parser.parse_args()

    final_rows = read_named_table(args.final_table)
    final_by_n = {int(row["nx1"]): row for row in final_rows}
    histories: dict[int, list[dict[str, float]]] = {}
    for specification in args.history:
        resolution_text, separator, path_text = specification.partition(":")
        if not separator:
            raise AssertionError(f"invalid --history value {specification!r}")
        resolution = int(resolution_text)
        histories[resolution] = read_history(Path(path_text))
    if len(histories) < 3:
        raise AssertionError("need at least three distinct robust-Minkowski resolutions")
    if set(histories) - set(final_by_n):
        raise AssertionError("a history resolution is missing from the final-state table")

    slopes_by_channel: dict[str, list[tuple[int, float]]] = {name: [] for name in CHANNELS}
    for resolution in sorted(histories):
        rows = histories[resolution]
        final = final_by_n[resolution]
        if not (final["min_A"] > 0.0 and final["min_chi"] > 0.0
                and final["min_SPD"] > 0.0):
            raise AssertionError(f"N={resolution}: positivity or SPD failure")
        worst_peak = (0.0, "")
        worst_fit = (-math.inf, "")
        endpoint_rates = []
        for channel in CHANNELS:
            norms = [math.sqrt(row[channel]/row["Volume"]) for row in rows]
            if not all(math.isfinite(value) and value > 0.0 for value in norms):
                raise AssertionError(f"N={resolution} {channel}: invalid norm history")
            amplification = max(norms)/norms[0]
            endpoint_rate = math.log(norms[-1]/norms[0])/(rows[-1]["time"] - rows[0]["time"])
            late = rows[len(rows)//2:]
            late_norms = norms[len(rows)//2:]
            slope = fit_slope([row["time"] for row in late], [math.log(x) for x in late_norms])
            slopes_by_channel[channel].append((resolution, slope))
            worst_peak = max(worst_peak, (amplification, channel))
            worst_fit = max(worst_fit, (slope, channel))
            endpoint_rates.append(endpoint_rate)
        print(f"N={resolution:4d} peak_amp={worst_peak[0]:.6f}({worst_peak[1]}) "
              f"max_late_fit={worst_fit[0]:.6f}/M({worst_fit[1]}) "
              f"max_endpoint_rate={max(endpoint_rates):.6f}/M "
              f"state_rms={final['state_rms']:.6e} min_SPD={final['min_SPD']:.12e}")
        if worst_peak[0] > args.max_amplification:
            raise AssertionError(f"N={resolution}: amplification {worst_peak[0]} exceeds gate")
        if worst_fit[0] > args.max_fit_growth:
            raise AssertionError(f"N={resolution}: late fitted growth {worst_fit[0]} exceeds gate")

    for channel, samples in slopes_by_channel.items():
        slopes = [slope for _, slope in samples]
        if all(fine > coarse for coarse, fine in zip(slopes, slopes[1:])) \
                and slopes[-1] > args.max_fit_growth:
            raise AssertionError(f"{channel}: resolution-growing exponential rate {samples}")
    print("PASS: no resolution-growing exponential instability in robust Minkowski ladder")


if __name__ == "__main__":
    main()
