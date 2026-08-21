#!/usr/bin/env python3
"""Create and verify the final O4 campaign evidence/checksum closure."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GENERATED = {"EVIDENCE_MANIFEST.json", "SHA256SUMS", "SHA256SUMS.sha256"}
EXPECTED_SOURCE = "26d4c371ea57b8db2cde47e56ac0a8de8fb89dc9"
EXPECTED_TREE = "ffa46e2b51d4bf46929565deee92df52ad7d6add"
EXPECTED_EXE = "645de4273ab1509407cba6a5df93a153a2510c4bb4d335c0f10d892f41eeb4a0"
EXPECTED_INPUT = "02b627dca6ad6ddf1802137882c543e0aa56c79db6d8d8efcd07c4c4c495769b"
EXPECTED_COEFFICIENTS = "ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b"
EXPECTED_BUILD_MANIFEST = "41e3e09a2a2a80827b0ad86d61b541a6a947850dd93ac724550144c1beb3923a"
JOBS = (8769918, 8769947, 8769961)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relevant_files(*, include_manifest: bool) -> list[Path]:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    if not include_manifest:
        excluded.add("EVIDENCE_MANIFEST.json")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in excluded:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(ROOT)))


def parse_qstat(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    def value(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)} = (.+)$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"missing {name} in {path}")
        return match.group(1).strip()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "job_state": value("job_state"),
        "exit_status": int(value("Exit_status")),
        "walltime": value("resources_used.walltime"),
        "memory": value("resources_used.mem"),
        "execution_host": value("exec_host"),
        "requested_walltime": value("Resource_List.walltime"),
    }


def verify_remote_manifest(segment_dir: Path) -> dict[str, object]:
    manifest_path = segment_dir / "SHA256SUMS"
    records = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, remote = line.split(maxsplit=1)
        records[remote.strip()] = digest
    verified = []
    for path in sorted(segment_dir.iterdir()):
        if not path.is_file() or path.name.startswith("qstat-") or path == manifest_path:
            continue
        matches = [digest for remote, digest in records.items() if remote.endswith("/" + path.name)]
        if len(matches) != 1 or matches[0] != sha256(path):
            raise RuntimeError(f"remote manifest mismatch: {path}")
        verified.append(path.name)
    return {
        "path": str(manifest_path.relative_to(ROOT)),
        "sha256": sha256(manifest_path),
        "selected_files_verified": verified,
    }


def main() -> None:
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    if "<!-- TERMINAL_EVIDENCE -->" in report:
        raise RuntimeError("REPORT.md still contains terminal-evidence placeholder")
    summary = json.loads((ROOT / "data/comparison_summary.json").read_text())
    if summary.get("qualification_claim") is not False:
        raise RuntimeError("comparison summary must not promote qualification")

    build_manifest = ROOT / "evidence/build/SHA256SUMS"
    if sha256(build_manifest) != EXPECTED_BUILD_MANIFEST:
        raise RuntimeError("build manifest identity mismatch")

    segments = []
    for segment, job in enumerate(JOBS):
        directory = ROOT / f"evidence/segment{segment}"
        provenance = (directory / "provenance.log").read_text(encoding="utf-8")
        for token in (EXPECTED_SOURCE, EXPECTED_TREE, EXPECTED_EXE, EXPECTED_INPUT, EXPECTED_COEFFICIENTS):
            if token not in provenance:
                raise RuntimeError(f"missing provenance token {token} in segment {segment}")
        qstat_path = directory / f"qstat-{job}.txt"
        segments.append({
            "segment": segment,
            "job_id": job,
            "qstat": parse_qstat(qstat_path),
            "remote_manifest": verify_remote_manifest(directory),
            "provenance_sha256": sha256(directory / "provenance.log"),
            "run_status": (directory / "run-status").read_text().strip(),
            "disposition": (directory / "disposition").read_text().strip(),
        })

    inventory = {
        str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in relevant_files(include_manifest=False)
    }
    manifest = {
        "schema": "athenak_brill_o4_n256_evidence_manifest_v1",
        "qualification_claim": False,
        "formal_disposition": summary["formal_disposition"],
        "source": {
            "implementation_commit": "3453b65a6b13c8f72cc1da6f05c565d245ce0f45",
            "build_commit": EXPECTED_SOURCE,
            "build_tree": EXPECTED_TREE,
            "executable_sha256": EXPECTED_EXE,
        },
        "physical_contract": {
            "input_sha256": EXPECTED_INPUT,
            "coefficients_sha256": EXPECTED_COEFFICIENTS,
            "spatial_order": 4,
            "dchi_max": 0.02,
            "derefine_factor": 0.25,
            "z4c_constraint_damping": 0.0,
        },
        "build_manifest_sha256": EXPECTED_BUILD_MANIFEST,
        "segments": segments,
        "comparison_summary_sha256": sha256(ROOT / "data/comparison_summary.json"),
        "report_sha256": sha256(ROOT / "REPORT.md"),
        "inventory": inventory,
        "limitations": summary["limitations"],
    }
    manifest_path = ROOT / "EVIDENCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_path = ROOT / "SHA256SUMS"
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT)}"
        for path in relevant_files(include_manifest=True)
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    detached = ROOT / "SHA256SUMS.sha256"
    detached.write_text(f"{sha256(checksum_path)}  SHA256SUMS\n", encoding="utf-8")

    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if sha256(ROOT / relative.strip()) != digest:
            raise RuntimeError(f"checksum verification failed: {relative}")
    detached_digest = detached.read_text(encoding="utf-8").split()[0]
    if sha256(checksum_path) != detached_digest:
        raise RuntimeError("detached checksum verification failed")
    print(f"EVIDENCE_MANIFEST_SHA256={sha256(manifest_path)}")
    print(f"SHA256SUMS_SHA256={sha256(checksum_path)}")
    print(f"DETACHED_FILE_SHA256={sha256(detached)}")


if __name__ == "__main__":
    main()
