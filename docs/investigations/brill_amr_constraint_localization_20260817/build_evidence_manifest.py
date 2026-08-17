#!/usr/bin/env python3
"""Build the strict, self-verifying manifest for this investigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "athenak_brill_amr_constraint_localization_evidence_v1"
EXCLUDED = {"evidence_manifest.json", "SHA256SUMS", "SHA256SUMS.sha256"}


class ManifestError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ManifestError(f"non-finite JSON token {token} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def artifact_paths(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ManifestError(f"symlink forbidden: {path}")
        if path.is_dir():
            if path.name == "__pycache__":
                raise ManifestError(f"Python cache forbidden: {path}")
            continue
        relative = path.relative_to(root)
        if relative.name in EXCLUDED:
            continue
        if relative.suffix in {".pyc", ".pyo"}:
            raise ManifestError(f"Python bytecode forbidden: {path}")
        result.append(relative)
    return result


def build(root: Path) -> None:
    verdict = strict_load(root / "verdict.json")
    if verdict.get("primary_disposition") != "same_level_seam_derivative_dominates":
        raise ManifestError("unexpected primary disposition")
    if verdict.get("qualification_claim") is not False:
        raise ManifestError("qualification claim must be false")
    files = artifact_paths(root)
    manifest = {
        "schema": SCHEMA,
        "qualification_claim": False,
        "primary_disposition": "same_level_seam_derivative_dominates",
        "source": {
            "repository": "https://github.com/HengruiZhu99/athenak",
            "branch": "codex/brill-amr-coarse-cache-coherence-20260817",
            "diagnostic_commit": "55f9147bc80d574636c47bcd1dac86178d921988",
            "diagnostic_tree": "cb2ad270f0675230b77023877dc0fdf93b52cd59",
            "coarse_cache_fix_commit": "ab651f0ebd113f8718fefbf6d802976e6b3e8738",
        },
        "restart_sha256": "83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea",
        "files": [
            {"path": str(path), "size_bytes": (root / path).stat().st_size,
             "sha256": sha256(root / path)}
            for path in files
        ],
    }
    strict_dump(root / "evidence_manifest.json", manifest)
    checksum_files = artifact_paths(root)
    checksum_lines = [f"{sha256(root / path)}  {path}\n" for path in checksum_files]
    checksum_lines.append(
        f"{sha256(root / 'evidence_manifest.json')}  evidence_manifest.json\n")
    checksum_lines.sort(key=lambda line: line.split("  ", 1)[1])
    temporary = root / "SHA256SUMS.tmp"
    temporary.write_text("".join(checksum_lines), encoding="utf-8")
    os.replace(temporary, root / "SHA256SUMS")
    detached = f"{sha256(root / 'SHA256SUMS')}  SHA256SUMS\n"
    temporary = root / "SHA256SUMS.sha256.tmp"
    temporary.write_text(detached, encoding="utf-8")
    os.replace(temporary, root / "SHA256SUMS.sha256")


def verify(root: Path) -> None:
    manifest = strict_load(root / "evidence_manifest.json")
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {str(path) for path in artifact_paths(root)}
    actual.discard("evidence_manifest.json")
    if set(expected) != actual:
        raise ManifestError("manifest inventory mismatch")
    for relative, item in expected.items():
        path = root / relative
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ManifestError(f"manifest mismatch: {relative}")
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        if sha256(root / relative) != digest:
            raise ManifestError(f"checksum mismatch: {relative}")
    detached = (root / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()[0]
    if sha256(root / "SHA256SUMS") != detached:
        raise ManifestError("detached checksum mismatch")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    resolved = arguments.root.resolve()
    if not arguments.verify_only:
        build(resolved)
    verify(resolved)
