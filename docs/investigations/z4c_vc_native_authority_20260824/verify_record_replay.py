#!/usr/bin/env python3
"""Verify N256 record/replay state identity under AthenaK's payload contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MARKER = b"<par_end>\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(path: Path) -> bytes:
    data = path.read_bytes()
    if MARKER not in data:
        raise RuntimeError(f"missing parameter marker in {path}")
    return data.split(MARKER, 1)[1]


def canonical_key(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("n256_native_record", "RUN").replace(
        "n256_native_replay", "RUN"
    )


def compare_payloads(record: Path, replay: Path, suffix: str) -> dict:
    left = {canonical_key(path, record): path for path in record.rglob(f"*{suffix}")}
    right = {canonical_key(path, replay): path for path in replay.rglob(f"*{suffix}")}
    rows = []
    for key in sorted(left.keys() & right.keys()):
        left_raw = left[key].read_bytes()
        right_raw = right[key].read_bytes()
        left_payload = payload(left[key])
        right_payload = payload(right[key])
        rows.append({
            "key": key,
            "raw_exact": left_raw == right_raw,
            "raw_record_sha256": sha(left_raw),
            "raw_replay_sha256": sha(right_raw),
            "payload_exact": left_payload == right_payload,
            "payload_sha256": sha(left_payload) if left_payload == right_payload else None,
        })
    return {
        "keys_exact": left.keys() == right.keys(),
        "count": len(rows),
        "raw_exact_count": sum(row["raw_exact"] for row in rows),
        "payload_exact_count": sum(row["payload_exact"] for row in rows),
        "all_payload_exact": all(row["payload_exact"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exact_files = (
        ("history", "n256_native_record.z4c.user.hst",
         "n256_native_replay.z4c.user.hst"),
        ("constraint_summary", "n256_native_record-constraints.dat",
         "n256_native_replay-constraints.dat"),
        ("timestep_contract", "z4c_timestep_contract.csv", "z4c_timestep_contract.csv"),
        ("axis_regularity", "z4c_vertex_axis_regularity.csv",
         "z4c_vertex_axis_regularity.csv"),
        ("authority", "n256_native_authority.jsonl", "n256_native_authority.jsonl"),
    )
    exact = {}
    for label, left_name, right_name in exact_files:
        left = (args.record / left_name).read_bytes()
        right = (args.replay / right_name).read_bytes()
        exact[label] = {"exact": left == right,
                        "record_sha256": sha(left), "replay_sha256": sha(right)}

    binary = compare_payloads(args.record, args.replay, ".bin")
    restart = compare_payloads(args.record, args.replay, ".rst")
    result = {
        "schema": "z4c_record_replay_verification_v1",
        "exact_files": exact,
        "binary_outputs": binary,
        "restart_outputs": restart,
        "verdict": "EXACT_NUMERICAL_PAYLOAD" if (
            all(item["exact"] for item in exact.values()) and
            binary["all_payload_exact"] and restart["all_payload_exact"]
        ) else "MISMATCH",
        "qualification_note": (
            "Whole binary/restart containers include record/replay parameter metadata. "
            "Numerical identity uses the repository test contract: bytes after <par_end>."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("schema", "verdict", "qualification_note")}, indent=2))


if __name__ == "__main__":
    main()
