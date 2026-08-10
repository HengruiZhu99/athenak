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


def selected_gpu() -> tuple[str, str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is unset")
    devices = [item.strip() for item in visible.split(",")]
    if len(devices) != 1 or not devices[0] or devices[0] in {"-1", "NoDevFiles"}:
        raise RuntimeError(
            "CUDA qualification requires exactly one visible device per rank")
    selected = devices[0]
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", f"--id={selected}", "--query-gpu=uuid,name",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot resolve visible CUDA device {selected!r} to a UUID") from error
    records = [tuple(part.strip() for part in line.split(",", 1)) for line in lines]
    if len(records) != 1 or len(records[0]) != 2 or not all(records[0]):
        raise RuntimeError(
            f"visible CUDA device {selected!r} did not resolve to exactly one UUID/name")
    return selected, records[0][0], records[0][1]


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
    visible_token = uuid = gpu_name = None
    if args.require_cuda:
        try:
            visible_token, uuid, gpu_name = selected_gpu()
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    record = {"rank": rank, "local_rank": local_rank_value(),
              "hostname": socket.gethostname(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "visible_device_token": visible_token,
              "selected_uuid": uuid, "gpu_name": gpu_name,
              "binding_verified": bool(args.require_cuda)}
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
