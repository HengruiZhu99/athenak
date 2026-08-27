#!/usr/bin/env python3
"""Build the strict, self-hash-excluding artifact manifest for this handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "EVIDENCE_MANIFEST.json"
EXCLUDED_SUFFIXES = {".aux", ".log", ".out"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


artifacts = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == OUTPUT or "__pycache__" in path.parts:
        continue
    if path.suffix in EXCLUDED_SUFFIXES:
        continue
    artifacts.append({
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    })

manifest = {
    "schema": "z4c_vc_ko05_bjorhus_figure3_evidence_v1",
    "repository": "https://github.com/HengruiZhu99/athenak",
    "branch": "codex/z4c-vc-ko05-bjorhus-figure3-20260826",
    "source_fix_commit": "d39822c6522688749fe5ead8025907bc055f02f8",
    "cancelled_job": {
        "job_id": 57636959,
        "state": "CANCELLED by user",
        "elapsed": "00:04:02",
        "new_submissions_after_cancel": 0,
    },
    "main_campaign": {
        "common_terminal_time": 14.405106171032207,
        "authority_sha256": "704352de1d33c5e57e7d597589d0956d8b258d66519308646ad8317db02d4f82",
        "n128": "REACHED_FORCED_COMMON_TLIM_NOT_STABILITY",
        "n256": "FAIL_CLOSED_NONPOSITIVE_METRIC_PIVOT",
        "n512": "REACHED_FORCED_COMMON_TLIM_WITH_SEVERE_RUNAWAY",
        "figure3_reproduced": False,
        "convergence_qualified": False,
    },
    "cpbc_discriminator": {
        "complete": False,
        "A": "REACHED_TLIM_6.5",
        "B": "FAIL_CLOSED_3.244461166925543",
        "C": "REACHED_TLIM_6.5_BEFORE_CANCELLATION",
        "D": "INTENTIONALLY_CANCELLED_1.5087910750714824_EXCLUDED",
        "boundary_ruled_out": False,
    },
    "claim_limits": [
        "No t=30 stability claim.",
        "No Figure-3 reproduction claim.",
        "No convergence or critical-phenomena claim.",
        "The physical boundary is deprioritized, not mathematically excluded.",
        "The CPBC comparison is incomplete and not qualified.",
    ],
    "artifacts": artifacts,
}
OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print(f"wrote {OUTPUT} with {len(artifacts)} artifacts")
