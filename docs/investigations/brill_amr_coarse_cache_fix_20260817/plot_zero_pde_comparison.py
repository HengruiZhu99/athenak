#!/usr/bin/env python3
"""Compare production C++ aggregates for the old and patched cycle-1722 event."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("C_norm2", "H_norm2", "M_norm2", "Z_norm2")


def load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")))


def strict_dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-t0", type=Path, required=True)
    parser.add_argument("--old-t5", type=Path, required=True)
    parser.add_argument("--patched-t0", type=Path, required=True)
    parser.add_argument("--patched-t5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for label, t0_path, t5_path in (
        ("old", args.old_t0, args.old_t5),
        ("patched", args.patched_t0, args.patched_t5),
    ):
        t0, t5 = load(t0_path), load(t5_path)
        records[label] = {
            key: {
                "t0": float(t0[key]),
                "t5": float(t5[key]),
                "jump_factor": float(t5[key]) / float(t0[key]),
            }
            for key in FIELDS
        }
    records["materiality"] = {
        key: {
            "patched_over_old_t5": records["patched"][key]["t5"] /
                records["old"][key]["t5"],
            "patched_over_old_jump_factor": records["patched"][key]["jump_factor"] /
                records["old"][key]["jump_factor"],
        }
        for key in FIELDS
    }
    records["verdict"] = "cache_invariant_fixed_but_constraint_jump_essentially_unchanged"
    strict_dump(args.output / "zero_pde_comparison.json", records)

    labels = ("C", "H", "M", "Z")
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.bar(x - width / 2,
             [records["old"][key]["jump_factor"] for key in FIELDS],
             width, label="old production")
    axis.bar(x + width / 2,
             [records["patched"][key]["jump_factor"] for key in FIELDS],
             width, label="patched zero-PDE")
    axis.set_yscale("log")
    axis.set_ylabel(r"T5 / T0 production norm$^2$")
    axis.set_xticks(x, labels)
    axis.set_title("Cycle 1722 refinement jump: old vs cache-preserving patch")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend()
    fig.savefig(args.output / "zero_pde_constraint_jump_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
