#!/usr/bin/env python3
"""Freeze and verify the common-tree Brill convergence evidence package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
GENERATED = {"EVIDENCE_MANIFEST.json", "SHA256SUMS", "SHA256SUMS.sha256"}
EXPECTED = {
    "authority/n256_amr_history.jsonl":
        "874551cc68e7dab4d40b854b31ab6b42aff9d2eae0ca9faf5985c41ef14a589f",
    "evidence/perlmutter/sacct-primary.txt":
        "51750baf7188e7548694a685ae3c875b78e95bb0a19be2863ec4ba49f84ac3d9",
    "data/field_convergence.csv":
        "192599c9755267d97d1f14a81e35420cf37477264e2ec3c938e2f09b176ab1fd",
}


class EvidenceError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject(token: str) -> None:
        raise EvidenceError(f"non-finite JSON token {token} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def inventory(*, include_manifest: bool) -> list[Path]:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    if not include_manifest:
        excluded.add("EVIDENCE_MANIFEST.json")
    result = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"symlink forbidden: {path}")
        if path.is_dir():
            if path.name == "__pycache__":
                raise EvidenceError(f"Python cache forbidden: {path}")
            continue
        if path.name in excluded:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            raise EvidenceError(f"Python bytecode forbidden: {path}")
        result.append(path.relative_to(ROOT))
    return result


def semantic_checks() -> dict[str, Any]:
    summary = strict_load(ROOT / "comparison_summary.json")
    if summary.get("schema") != "brill_o4_common_tree_final_comparison_v1":
        raise EvidenceError("unexpected comparison schema")
    verdicts = summary.get("formal_verdicts", {})
    if verdicts.get("replay_verdict") != "EXACT_CROSS_RESOLUTION_REPLAY":
        raise EvidenceError("replay disposition changed")
    if verdicts.get("overall") != "O4_NONCONVERGENT":
        raise EvidenceError("overall disposition changed")
    claims = summary.get("qualification_claims", {})
    if any(claims.values()):
        raise EvidenceError("a qualification claim was unexpectedly promoted")
    replay = summary.get("replay", {})
    if replay.get("authority_sha256") != EXPECTED["authority/n256_amr_history.jsonl"]:
        raise EvidenceError("authority binding changed")
    if replay.get("n128", {}).get("max_abs_ulp") != 0:
        raise EvidenceError("N128 replay is not exact")
    if replay.get("n512", {}).get("max_abs_ulp") != 0:
        raise EvidenceError("N512 replay is not exact")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    for token in ("O4_NONCONVERGENT", "no fictitious collapsed-y width",
                  "This is neither a convergence claim nor a Figure-3 reproduction"):
        if token not in report:
            raise EvidenceError(f"report boundary missing: {token}")
    for relative, digest in EXPECTED.items():
        if sha256(ROOT / relative) != digest:
            raise EvidenceError(f"frozen identity changed: {relative}")
    return summary


def build() -> None:
    summary = semantic_checks()
    files = inventory(include_manifest=False)
    manifest = {
        "schema": "athenak_brill_o4_common_tree_evidence_v1",
        "qualification_claim": False,
        "repository": "https://github.com/HengruiZhu99/athenak",
        "branch": "codex/brill-o4-dchi001-replay-convergence-20260821",
        "built_source_commit": summary["campaign"]["built_source_commit"],
        "analysis_source_commit": summary["campaign"]["analysis_source_commit"],
        "formal_verdicts": summary["formal_verdicts"],
        "jobs": {"n128": 57346928, "n256": 57342668, "n512": 57347956},
        "authority_sha256": EXPECTED["authority/n256_amr_history.jsonl"],
        "files": [
            {"path": str(relative), "bytes": (ROOT / relative).stat().st_size,
             "sha256": sha256(ROOT / relative)}
            for relative in files
        ],
        "limitations": summary["limitations"],
    }
    strict_dump(ROOT / "EVIDENCE_MANIFEST.json", manifest)
    lines = [
        f"{sha256(ROOT / relative)}  {relative}\n"
        for relative in inventory(include_manifest=True)
    ]
    temporary = ROOT / "SHA256SUMS.tmp"
    temporary.write_text("".join(lines), encoding="utf-8")
    os.replace(temporary, ROOT / "SHA256SUMS")
    detached = ROOT / "SHA256SUMS.sha256.tmp"
    detached.write_text(f"{sha256(ROOT / 'SHA256SUMS')}  SHA256SUMS\n",
                        encoding="utf-8")
    os.replace(detached, ROOT / "SHA256SUMS.sha256")


def verify() -> None:
    semantic_checks()
    manifest = strict_load(ROOT / "EVIDENCE_MANIFEST.json")
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {str(path) for path in inventory(include_manifest=False)}
    if set(expected) != actual:
        raise EvidenceError("evidence manifest inventory mismatch")
    for relative, item in expected.items():
        path = ROOT / relative
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise EvidenceError(f"evidence manifest mismatch: {relative}")
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        if sha256(ROOT / relative) != digest:
            raise EvidenceError(f"checksum mismatch: {relative}")
    detached = (ROOT / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()[0]
    if sha256(ROOT / "SHA256SUMS") != detached:
        raise EvidenceError("detached checksum mismatch")


if __name__ == "__main__":
    build()
    verify()
    print("STRICT_COMMON_TREE_EVIDENCE_PASS")
    print(f"EVIDENCE_MANIFEST_SHA256={sha256(ROOT / 'EVIDENCE_MANIFEST.json')}")
    print(f"SHA256SUMS_SHA256={sha256(ROOT / 'SHA256SUMS')}")
    print(f"DETACHED_FILE_SHA256={sha256(ROOT / 'SHA256SUMS.sha256')}")
