#!/usr/bin/env python3
"""Build the strict local/remote evidence manifest for this campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "EVIDENCE_MANIFEST.json"

    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == output or "__pycache__" in path.parts:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": sha256(path)})

    manifest = {
        "schema": "z4c_vc_native_authority_evidence_manifest_v1",
        "verdict": "NATIVE_AMR_UNSTABLE",
        "repository": "https://github.com/HengruiZhu99/athenak",
        "branch": "codex/z4c-vc-figure3-native-authority-20260824",
        "reviewed_numerical_commit": "6dd20656a305f2543bbbd7001550c6ac67019180",
        "diagnostic_determinism_commit": "d63519328214a6315a9cc1f7d5e4a1aa4bca21b0",
        "production_source_tree": "9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4",
        "perlmutter_root": "/pscratch/sd/h/hzhu/z4c-vc-native-authority-20260824",
        "build": {
            "cuda_executable_sha256":
                "3a395bfdaf217d617fee43d2cbcd38e7a13c2a0f4207e3a764c3513eb8c0405f",
            "cmake_cache_sha256":
                "4079cdbf0ee8bfc12be32cd43a65d8a97978a21dd6c8198e5d3343c4280f1c1e",
            "host_enabled_tests": (
                "140/141 passed in concurrent aggregate; sole AMR-history timeout "
                "passed isolated in 177.96 s; 2 CUDA-required tests disabled"
            ),
            "focused_a100_tests": "20/20 passed in job 57524292",
        },
        "science_runs": [
            {"job": "57524377", "case": "N256 record", "target": "t=2.5",
             "status": "completed"},
            {"job": "57524427", "case": "N256 replay", "target": "t=2.5",
             "status": "completed"},
            {"job": "57524489", "case": "N128 replay", "target": "t=2.5",
             "status": "completed"},
            {"job": "57524522", "case": "N512 replay", "target": "t=2.5",
             "status": "completed"},
            {"job": "57525084", "case": "N256 record continuation",
             "target": "tau approximately 4", "status": "completed"},
            {"job": "57525355", "case": "N128 replay", "target": "tau approximately 4",
             "status": "completed"},
            {"job": "57525422", "case": "N256 replay", "target": "tau approximately 4",
             "status": "completed"},
            {"job": "57525474", "case": "N512 replay", "target": "tau approximately 4",
             "status": "completed"},
            {"job": "57525753", "case": "N256 record continuation",
             "target": "tau approximately 7", "status": "cancelled at fail gate",
             "terminal_time": 11.192887945084333,
             "terminal_tau": 6.847212341657571},
        ],
        "key_artifacts": {
            "fresh_early_authority_sha256":
                "fd08e6b32b094ef9e8e928a7ad8e061edcc6e67617609952f0aefa47e4b0f694",
            "historical_authority_sha256":
                "ce3cdea1a8d0465a7c19e4ac1134ce474d8908b5f5cb12be6f20110d12e9c851",
            "failed_extension_authority_sha256":
                "6c1fc27308cc85ad5b69d9cc7ed2aa060ca85ab41717a407a4868ea5af14ac67",
            "failed_extension_history_sha256":
                "f00d88ef138104e156adc3d5db9136d4ae0b23ce217a8956babe3e5da1a661b7",
            "failed_extension_sha256_manifest_sha256":
                "1d67c0867f04fe26a17ad57068ae7090b73918b7133f313372f7ba4d15575a8d",
            "compact_remote_evidence_tar_sha256":
                "f0c82c74fa64a883bb33c530b6da66f6821595e160e239fa401d4e52104c45b2",
        },
        "failed_non_science_allocations": [
            {"job": "57522958", "disposition": "shell quoting failed before Athena"},
            {"job": "57523029", "disposition": "cancelled pending; bad constraint quoting"},
            {"job": "57523096", "disposition": "runner environment assertion; no science"},
            {"job": "57524210", "disposition": "regression used system Python; no science"},
        ],
        "limitations": [
            "Raw binary/restart container hashes differ across record/replay headers; numerical payloads are exact.",
            "No cross-resolution replay was run past the failed N256 tau approximately 7 gate.",
            "No tau approximately 10.5 or full Figure-3 run was attempted.",
            "The failure evidence does not isolate parent under-resolution from transfer/interface feedback.",
        ],
        "files": files,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} with {len(files)} files")


if __name__ == "__main__":
    main()
