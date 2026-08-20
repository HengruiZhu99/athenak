#!/usr/bin/env python3
"""Create the compact, deterministic summary of the fresh Aurora N256 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_history(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 71 or data.shape[0] == 0:
        raise RuntimeError(f"unexpected history shape for {path}: {data.shape}")
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"nonfinite history value in {path}")
    return data


def strict_load(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON token {value} in {path}")

    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return result


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()

    segment0_path = root / "segment0/brill_contract_n256_scratch.z4c.user.hst"
    continuation_path = (
        root / "continuation/brill_contract_n256_scratch_cont.z4c.user.hst"
    )
    segment0 = read_history(segment0_path)
    continuation = read_history(continuation_path)
    if continuation[0, 0] <= segment0[-1, 0]:
        raise RuntimeError("continuation history does not advance beyond segment 0")
    history = np.concatenate((segment0, continuation), axis=0)

    # Athena history indices (zero based): time, dt, C/H/M/Z, max|K|, nmb,
    # max Kretschmann, relative maximum level, cycle.
    time = history[:, 0]
    target_time = 9.476710063617325

    figure, axes = plt.subplots(4, 1, figsize=(10.2, 12.0), sharex=True)
    for column, label in zip(range(2, 6), ("C", "H", "M", "Z")):
        axes[0].semilogy(time, history[:, column], label=label)
    axes[0].set_ylabel("proper-volume integral")
    axes[0].legend(ncol=4)

    max_k = np.where(history[:, 11] > 0.0, history[:, 11], np.nan)
    max_kretschmann = np.where(history[:, 13] > 0.0, history[:, 13], np.nan)
    axes[1].semilogy(time, max_k, label="max |K|")
    axes[1].semilogy(time, max_kretschmann, label="max Kretschmann")
    axes[1].set_ylabel("curvature diagnostic")
    axes[1].legend()

    axes[2].semilogy(time, history[:, 1])
    axes[2].set_ylabel("dt / M")

    axes[3].step(time, history[:, 12], where="post", label="MeshBlocks")
    level_axis = axes[3].twinx()
    level_axis.step(
        time, history[:, 14], where="post", color="tab:red", label="max rel. level"
    )
    axes[3].set_ylabel("MeshBlocks")
    level_axis.set_ylabel("max relative level", color="tab:red")
    axes[3].set_xlabel("t / M")

    for axis in axes:
        axis.axvline(target_time, color="black", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
    figure.suptitle("Fresh N256 Brill trajectory (cycle-1800 restart is a run segment boundary)")
    figure.tight_layout()
    figure.savefig(root / "fresh_n256_history.png", dpi=180)
    plt.close(figure)

    verdict = strict_load(root / "analysis/target_event/verdict.json")
    parent = strict_load(root / "analysis/parent_state/parent_state_audit.json")
    native = verdict["native_integrals"]
    old = native["t0_00_ACCEPTED_OLD_STATE"]
    new = native["t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION"]
    family = {row["family"]: row for row in parent["families"]}

    summary = {
        "schema": "athenak_brill_fresh_n256_aurora_summary_v1",
        "qualification_claim": False,
        "source_commit": "e8781f4057c73a0e97f5802413aefde899e24123",
        "source_tree": "f7a0d222b7f08c035a1cf4c2a783c2d4ff8f8154",
        "athena_executable_sha256": (
            "8d9035cc22f4406788792f585648243e4dc043e484ee5ab6f54ac14009a063ed"
        ),
        "aurora_jobs": {"segment0_from_t0": 8768636, "continuation": 8768689},
        "segment_boundary": {
            "cycle": 1800,
            "time": float(segment0[-1, 0]),
            "restart_sha256": (
                "8771db9d9cf7ed6973513d4f81cd6fe3cc7f8edb58538e21014dd5b804fd058a"
            ),
        },
        "terminal": {
            "cycle": int(round(continuation[-1, 18])),
            "time": float(continuation[-1, 0]),
            "dt": float(continuation[-1, 1]),
            "C_norm2": float(continuation[-1, 2]),
            "H_norm2": float(continuation[-1, 3]),
            "M_norm2": float(continuation[-1, 4]),
            "Z_norm2": float(continuation[-1, 5]),
            "max_abs_K": float(continuation[-1, 11]),
            "nmb_total": int(round(continuation[-1, 12])),
            "max_kretschmann": float(continuation[-1, 13]),
            "max_relative_level": int(round(continuation[-1, 14])),
        },
        "target_event": {
            "cycle": int(verdict["cycle"]),
            "time": float(verdict["time"]),
            "explicit_refined_parent_gids": [28, 45],
            "physical_patch": {"rho": [5.0, 6.0], "z": [-1.0, 1.0]},
            "old_nmb": int(verdict["old_nmb"]),
            "new_nmb": int(verdict["new_nmb"]),
            "coordinate_ring_volume_relative_change": (
                float(new["coordinate_ring_volume"] / old["coordinate_ring_volume"] - 1.0)
            ),
            "proper_volume_relative_change": (
                float(new["proper_volume"] / old["proper_volume"] - 1.0)
            ),
            "native_integral_ratios_T5_over_T0": {
                name: float(new[name] / old[name])
                for name in ("C_norm2", "H_norm2", "M_norm2", "Z_norm2")
            },
            "projection_fraction_of_fixed_lattice_constraint_stage_l2": float(
                verdict["fixed_child_lattice_constraint_stage_l2"][
                    "ALGEBRAIC_PROJECTION"
                ]
                / (
                    verdict["fixed_child_lattice_constraint_stage_l2"][
                        "T0_TO_T3_REPRESENTATION_AND_BOUNDARY"
                    ]
                    + verdict["fixed_child_lattice_constraint_stage_l2"][
                        "ALGEBRAIC_PROJECTION"
                    ]
                )
            ),
            "worst_fixed_lattice_C_change": verdict["worst_fixed_lattice_C_change"],
        },
        "parent_state_selected": {
            name: {
                "PR_relative_l2": float(family[name]["PR_relative_l2"]),
                "PR_edge_band_relative_l2": float(
                    family[name]["PR_edge_band_relative_l2"]
                ),
                "PR_interior_relative_l2": float(
                    family[name]["PR_interior_relative_l2"]
                ),
                "D2_O6_O4_relative_l2": float(
                    family[name]["D2_O6_O4_relative_l2"]
                ),
                "block_nyquist_max": float(family[name]["block_nyquist_max"]),
            }
            for name in ("chi", "K", "Atilde", "Gammatilde", "gammatilde")
        },
        "history_inputs": [
            {"path": str(segment0_path.relative_to(root)), "sha256": digest(segment0_path)},
            {
                "path": str(continuation_path.relative_to(root)),
                "sha256": digest(continuation_path),
            },
        ],
        "limitations": [
            "The run is one N256 trajectory, not a convergence study.",
            "The source contracts and diagnostics are qualified; no unique production AMR operator correction is established.",
            "The cycle-1800 restart is an authenticated segmentation of the same from-t0 trajectory, not independent initial data.",
        ],
    }
    json.dumps(summary, allow_nan=False)
    with (root / "SUMMARY.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
