#!/usr/bin/env python3
"""Compare matched baseline/optimized N512 performance-run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PARAMETER_END = b"<par_end>\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restart_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    if PARAMETER_END not in raw:
        raise RuntimeError(f"restart lacks parameter terminator: {path}")
    return raw.split(PARAMETER_END, 1)[1]


def history_rows(path: Path) -> tuple[bytes, list[list[float]]]:
    raw = path.read_bytes()
    rows = [
        [float(value) for value in line.split()]
        for line in raw.decode("ascii").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows:
        raise RuntimeError(f"history contains no rows: {path}")
    return raw, rows


def final_restart(root: Path) -> Path:
    paths = sorted((root / "rst").glob("*.rst"))
    if not paths:
        raise RuntimeError(f"run has no restart: {root}")
    return paths[-1]


def performance(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "meshblock_cycles": r"MeshBlock-cycles = ([0-9]+)",
        "execution_seconds": r"cpu time used\s*= ([0-9.eE+-]+)",
        "zone_cycles_per_second": r"zone-cycles/cpu_second = ([0-9.eE+-]+)",
        "output_seconds": r"output wall time = ([0-9.eE+-]+)",
    }
    parsed: dict[str, float] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing {key} in {path}")
        parsed[key] = float(matches[-1])
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_history_path = next(args.baseline.glob("*.z4c.user.hst"))
    optimized_history_path = next(args.optimized.glob("*.z4c.user.hst"))
    baseline_history, baseline_rows = history_rows(baseline_history_path)
    optimized_history, optimized_rows = history_rows(optimized_history_path)

    baseline_restart_path = final_restart(args.baseline)
    optimized_restart_path = final_restart(args.optimized)
    baseline_payload = restart_payload(baseline_restart_path)
    optimized_payload = restart_payload(optimized_restart_path)

    baseline_perf = performance(args.baseline / "stdout.log")
    optimized_perf = performance(args.optimized / "stdout.log")
    result = {
        "schema": "z4c_vc_n512_performance_equivalence_v1",
        "baseline": {
            "root": str(args.baseline),
            "history_sha256": sha256(baseline_history),
            "restart": str(baseline_restart_path),
            "restart_payload_sha256": sha256(baseline_payload),
            "performance": baseline_perf,
        },
        "optimized": {
            "root": str(args.optimized),
            "history_sha256": sha256(optimized_history),
            "restart": str(optimized_restart_path),
            "restart_payload_sha256": sha256(optimized_payload),
            "performance": optimized_perf,
        },
        "history_bytes_exact": baseline_history == optimized_history,
        "history_rows_exact": baseline_rows == optimized_rows,
        "restart_payload_exact": baseline_payload == optimized_payload,
        "same_history_shape": (
            len(baseline_rows) == len(optimized_rows)
            and len(baseline_rows[0]) == len(optimized_rows[0])
        ),
        "same_final_time": baseline_rows[-1][0] == optimized_rows[-1][0],
        "same_final_cycle": baseline_rows[-1][18] == optimized_rows[-1][18],
        "same_final_meshblocks": baseline_rows[-1][12] == optimized_rows[-1][12],
        "same_final_max_level": baseline_rows[-1][14] == optimized_rows[-1][14],
        "speedup": (
            optimized_perf["zone_cycles_per_second"]
            / baseline_perf["zone_cycles_per_second"]
        ),
    }
    exact = (
        result["history_bytes_exact"]
        and result["restart_payload_exact"]
        and result["same_final_time"]
        and result["same_final_cycle"]
        and result["same_final_meshblocks"]
        and result["same_final_max_level"]
    )
    result["verdict"] = "BITWISE_EVOLVED_STATE" if exact else "MISMATCH"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "speedup": result["speedup"]}, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
