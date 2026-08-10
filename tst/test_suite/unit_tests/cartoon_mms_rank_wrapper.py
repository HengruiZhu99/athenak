#!/usr/bin/env python3
"""Record rank-local GPU binding, then replace this process with Athena."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def rank_value() -> int:
    for name in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def local_rank_value() -> int:
    for name in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def selected_gpu() -> tuple[str | None, str | None]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.startswith("GPU-"):
        selected = visible.split(",")[0]
    else:
        selected = None
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return selected, None
    if not lines:
        return selected, None
    records = [tuple(part.strip() for part in line.split(",", 1)) for line in lines]
    if selected:
        match = next((record for record in records if record[0] == selected), None)
        return match if match else (selected, None)
    if visible:
        first = visible.split(",")[0]
        if first.isdigit() and int(first) < len(records):
            return records[int(first)]
    local_rank = local_rank_value()
    return records[local_rank % len(records)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing Athena command")
    rank = rank_value()
    uuid, gpu_name = selected_gpu()
    if args.require_cuda and not uuid:
        raise SystemExit("CUDA qualification requires a rank-local GPU UUID")
    record = {"rank": rank, "local_rank": local_rank_value(),
              "hostname": socket.gethostname(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "selected_uuid": uuid, "gpu_name": gpu_name}
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    path = args.evidence_dir / f"rank_binding_{rank:04d}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    os.execv(args.command[0], args.command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
