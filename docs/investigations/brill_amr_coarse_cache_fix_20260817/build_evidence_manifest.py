#!/usr/bin/env python3
"""Build the strict, finite evidence manifest for this investigation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
MANIFEST = HERE / "EVIDENCE_MANIFEST.json"
SUMS = HERE / "SHA256SUMS"
DETACHED = HERE / "SHA256SUMS.sha256"
EXCLUDED = {MANIFEST, SUMS, DETACHED}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def artifact_paths() -> list[Path]:
    paths = [path for path in HERE.rglob("*") if path.is_file() and path not in EXCLUDED]
    paths.append(WORKTREE / "REMOTE_REVIEW_PROMPT.md")
    return sorted(paths, key=lambda path: path.relative_to(WORKTREE).as_posix())


def main() -> None:
    paths = artifact_paths()
    records = [
        {
            "path": path.relative_to(WORKTREE).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    payload = {
        "schema": "athenak_brill_amr_coarse_cache_evidence_manifest_v1",
        "source": {
            "branch": "codex/brill-amr-coarse-cache-coherence-20260817",
            "repair_commit": "ab651f0ebd113f8718fefbf6d802976e6b3e8738",
            "tree": "fae0f46e52717ab0e9a3f6c3ffc2dbbc0261b96f",
            "kokkos_commit": "6739bc623081648af9e752b616d9671527922cbf",
        },
        "remote_jobs": {
            "zero_pde": {"job_id": 57168348, "status": "completed", "exit_code": "0:0"},
            "short_pde": {
                "job_id": 57168637,
                "allocation_status": "timeout",
                "allocation_exit_code": "0:0",
                "allocation_elapsed_seconds": 7229,
                "science_step_status": "timeout",
                "science_step_exit_code": "0:15",
                "science_step_elapsed_seconds": 5401,
                "raw_run_manifest_sha256":
                    "d2dce5a4fccdfb1a97507256de5dcb3765aa60d771d988d53a683854ace012e4",
            },
        },
        "qualification_claim": False,
        "artifacts": records,
    }
    atomic_text(MANIFEST, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")

    checksummed = sorted(paths + [MANIFEST], key=lambda path: path.relative_to(WORKTREE).as_posix())
    lines = [f"{sha256(path)}  {path.relative_to(WORKTREE).as_posix()}" for path in checksummed]
    atomic_text(SUMS, "\n".join(lines) + "\n")
    atomic_text(DETACHED, f"{sha256(SUMS)}  {SUMS.relative_to(WORKTREE).as_posix()}\n")


if __name__ == "__main__":
    main()
