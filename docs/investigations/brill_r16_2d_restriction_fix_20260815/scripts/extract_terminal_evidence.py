#!/usr/bin/env python3
"""Verify selected terminal evidence and extract compact plotting histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
FIELDS = (
    "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
    "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "cycle",
    "axisLapse", "axisTau", "axisKret",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_entries(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative.lstrip("*")] = digest
    return entries


def extract_history(source: Path, destination: Path) -> dict[str, object]:
    labels: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for line in source.read_text(errors="strict").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if all(math.isfinite(value) for value in values):
            rows.append({field: values[labels[field]] for field in FIELDS})
    if not rows or not set(FIELDS) <= labels.keys():
        raise RuntimeError(f"invalid history: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "csv": destination.name,
        "csv_sha256": sha256(destination),
        "finite_rows": len(rows),
        "terminal_time": rows[-1]["time"],
        "terminal_cycle": int(rows[-1]["cycle"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_root", type=Path)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    selected = args.selected_root.resolve()
    data = args.data_dir.resolve()
    root_manifest = selected / "SHA256SUMS"
    detached = selected / "SHA256SUMS.sha256"
    entries = manifest_entries(root_manifest)

    checked = []
    for path in sorted(selected.rglob("*")):
        if not path.is_file() or path in {root_manifest, detached}:
            continue
        relative = path.relative_to(selected).as_posix()
        expected = entries.get(relative)
        actual = sha256(path)
        if expected is None or actual != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")
        checked.append({"path": relative, "sha256": actual})

    detached_line = detached.read_text().strip().split(maxsplit=1)
    manifest_sha = sha256(root_manifest)
    if len(detached_line) != 2 or detached_line[0] != manifest_sha:
        raise RuntimeError("detached root-manifest verification failed")

    histories = {}
    for case in ("n128", "n256"):
        case_root = selected / "run/cases" / case
        sources = list(case_root.glob("*.z4c.user.hst"))
        if len(sources) != 1:
            raise RuntimeError(f"expected one history for {case}")
        histories[case] = extract_history(
            sources[0], data / f"post_fix_{case}_history.csv"
        )
        shutil.copyfile(case_root / "result.json", data / f"post_fix_{case}_result.json")
        log_lines = (case_root / "run.log").read_text(errors="strict").splitlines()
        (data / f"post_fix_{case}_run_tail.txt").write_text(
            "\n".join(log_lines[-120:]) + "\n"
        )

    shutil.copyfile(selected / "allocation/sacct-settled.psv",
                    data / "post_fix_sacct_settled.psv")

    summary = {
        "schema": "athenak_selected_terminal_evidence_v1",
        "qualification_claim": False,
        "root_manifest_sha256": manifest_sha,
        "detached_manifest_file_sha256": sha256(detached),
        "detached_manifest_target": detached_line[1],
        "selected_files_verified": len(checked),
        "selected_verification_pass": True,
        "selected_files": checked,
        "histories": histories,
    }
    (data / "terminal_evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
