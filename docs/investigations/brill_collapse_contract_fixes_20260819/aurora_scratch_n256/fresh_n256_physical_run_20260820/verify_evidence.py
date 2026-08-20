#!/usr/bin/env python3
"""Fail-closed verifier for the compact fresh-N256 evidence handoff."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INVESTIGATION = ROOT.parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def strict_load(path: Path) -> dict:
    def reject(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON token {value} in {path}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def verify_sha_file(directory: Path) -> None:
    with (directory / "SHA256SUMS").open(encoding="utf-8") as stream:
        rows = [line.rstrip("\n") for line in stream if line.strip()]
    if not rows:
        raise RuntimeError(f"empty checksum manifest in {directory}")
    names = set()
    for row in rows:
        expected, name = row.split("  ", 1)
        if name in names or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe or duplicate checksum entry {name}")
        names.add(name)
        if digest(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")


def main() -> None:
    summary = strict_load(ROOT / "SUMMARY.json")
    manifest = strict_load(INVESTIGATION / "EVIDENCE_MANIFEST.json")
    fresh = manifest["fresh_n256_physical_run"]

    if summary["qualification_claim"] is not False:
        raise RuntimeError("summary promoted a qualification claim")
    if summary["source_commit"] != manifest["source"]["compiled_physical_run_commit"]:
        raise RuntimeError("source identity mismatch")
    if summary["terminal"]["cycle"] != 3329 or not math.isclose(
        summary["terminal"]["time"], 10.0, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError("terminal bound mismatch")
    target = summary["target_event"]
    if target["cycle"] != 2833 or target["explicit_refined_parent_gids"] != [28, 45]:
        raise RuntimeError("target-event identity mismatch")
    if abs(target["coordinate_ring_volume_relative_change"]) > 1.0e-13:
        raise RuntimeError("coordinate ring volume was not conserved")

    compact = fresh["local_compact_artifacts"]
    expected = {
        "report_sha256": ROOT / "REPORT.md",
        "summary_sha256": ROOT / "SUMMARY.json",
        "combined_history_plot_sha256": ROOT / "fresh_n256_history.png",
    }
    for key, path in expected.items():
        if digest(path) != compact[key]:
            raise RuntimeError(f"compact artifact mismatch: {key}")

    target_dir = ROOT / "analysis/target_event"
    parent_dir = ROOT / "analysis/parent_state"
    verify_sha_file(target_dir)
    verify_sha_file(parent_dir)
    if digest(target_dir / "SHA256SUMS") != fresh["target_event"][
        "analysis_manifest_sha256"
    ]:
        raise RuntimeError("target-event manifest identity mismatch")
    if digest(parent_dir / "SHA256SUMS") != fresh["target_event"][
        "parent_state_manifest_sha256"
    ]:
        raise RuntimeError("parent-state manifest identity mismatch")

    core = {
        "segment0/SHA256SUMS": (
            "640ba2e3a383a7f0b70abbeea7d5d0b4756365f27a73b146457b554b6771991a"
        ),
        "continuation/SHA256SUMS": (
            "04f49fde0703034a9df6f61697837890339a928c18932f8db1ef3ba6b21066e5"
        ),
        "continuation/stderr.log": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "continuation/chi_parent_provenance/phase1_stage_minima.csv": (
            "fe10f37081d251a36a9abeb74df12f2fe52967f08c6db84e5b337b0ad1cf3216"
        ),
        "continuation/chi_parent_provenance/preupdate_candidate_minima.csv": (
            "045eeec3fdd2fb7d5593744866555442d508c69c1d4b5db9174b1167819c7d6a"
        ),
    }
    for name, expected_digest in core.items():
        if digest(ROOT / name) != expected_digest:
            raise RuntimeError(f"root evidence mismatch: {name}")

    verify_sha_file(ROOT)
    detached = (ROOT / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()
    if len(detached) != 2 or detached[1] != "SHA256SUMS" or detached[0] != digest(
        ROOT / "SHA256SUMS"
    ):
        raise RuntimeError("detached root-manifest mismatch")

    print("FRESH_N256_STRICT_EVIDENCE_PASS")


if __name__ == "__main__":
    main()
