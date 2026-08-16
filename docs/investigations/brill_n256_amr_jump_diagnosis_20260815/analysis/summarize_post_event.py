#!/usr/bin/env python3
"""Aggregate recorded per-rank post-event integrals without rederiving physics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "diagnostic"
KEYS = ("C_norm2", "H_norm2", "M_norm2", "Z_norm2",
        "proper_volume", "coordinate_ring_volume")


def main() -> None:
    totals: dict[int, dict[str, float]] = {}
    for path in sorted(RAW.glob("rank????/post_event_cycles.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cycle = int(row["cycle"])
            aggregate = totals.setdefault(cycle, {key: 0.0 for key in KEYS})
            for key in KEYS:
                aggregate[key] += float(row[key])
            aggregate["time"] = float(row["time"])
    if sorted(totals) != list(range(1722, 1731)):
        raise RuntimeError("incomplete post-event cycle inventory")

    csv_path = Path(__file__).with_name("post_event_integrals.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("cycle", "time") + KEYS)
        writer.writeheader()
        for cycle in sorted(totals):
            writer.writerow({"cycle": cycle, **totals[cycle]})

    cycles = sorted(totals)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    for axis, key, title in zip(axes.flat, KEYS[:4], ("C", "H", "M", "Z")):
        axis.plot(cycles, [totals[cycle][key] for cycle in cycles], marker="o")
        axis.set_title(f"{title} squared proper-volume integral")
        axis.set_xlabel("cycle")
        axis.grid(alpha=0.3)
    fig.savefig(Path(__file__).with_name("post_event_integrals.png"), dpi=180)


if __name__ == "__main__":
    main()
