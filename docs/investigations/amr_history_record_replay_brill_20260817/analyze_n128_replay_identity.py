#!/usr/bin/env python3
"""Fail-closed same-resolution Brill AMR record/replay identity check.

The recording authority is expected to terminate at the strict chi gate after
the last accepted hierarchy event.  A replay intentionally stops at that last
event time and therefore writes one final time-limit output that the authority
does not have.  Compare every authority snapshot against the replay snapshot
with the same output index, time, and cycle; report replay-only snapshots
separately instead of rejecting that valid terminal-output asymmetry.
"""

import argparse
import hashlib
import json
import pathlib
import re


KINDS = ("z4c", "con", "adm", "weyl", "z4c_diag", "telegraph_mu")
PARAM_END = b"<par_end>\n"
TIME_RE = re.compile(rb"^  time=([^\n]+)$", re.MULTILINE)
CYCLE_RE = re.compile(rb"^  cycle=([0-9]+)$", re.MULTILINE)
INDEX_RE = re.compile(r"\.([0-9]+)\.bin$")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_snapshot(path):
    raw = path.read_bytes()
    if PARAM_END not in raw:
        raise SystemExit("binary output lacks parameter marker: {}".format(path))
    header, payload = raw.split(PARAM_END, 1)
    time_match = TIME_RE.search(header)
    cycle_match = CYCLE_RE.search(header)
    index_match = INDEX_RE.search(path.name)
    if time_match is None or cycle_match is None or index_match is None:
        raise SystemExit("binary output metadata is malformed: {}".format(path))
    return {
        "path": path,
        "index": int(index_match.group(1)),
        "time_text": time_match.group(1).decode("ascii"),
        "cycle": int(cycle_match.group(1)),
        "payload": payload,
    }


def inventory(case_root, kind):
    paths = sorted((case_root / "bin/rank_00000000").glob("*.{}.*.bin".format(kind)))
    rows = [read_snapshot(path) for path in paths]
    indices = [row["index"] for row in rows]
    if indices != list(range(len(rows))):
        raise SystemExit("{} output indices are not contiguous: {}".format(kind, indices))
    return rows


def digest_payloads(rows):
    digest = hashlib.sha256()
    for row in rows:
        payload = row["payload"]
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=pathlib.Path, required=True)
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    parser.add_argument("--authority", type=pathlib.Path, required=True)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.history.read_text().splitlines()]
    events = [row for row in events if row.get("type") == "event"][1:]
    ledger = [json.loads(line) for line in args.ledger.read_text().splitlines()]
    expected = [(row["event"], row["time_hex"], row["leaf_count"],
                 row["max_level"], row["tree_checksum"]) for row in events]
    observed = [(row["event"], row["time_hex"], row["leaves"],
                 row["max_level"], row["tree_checksum"]) for row in ledger]
    if expected != observed:
        raise SystemExit("same-resolution replay ledger differs from authority")

    comparisons = {}
    for kind in KINDS:
        authority_rows = inventory(args.authority, kind)
        replay_rows = inventory(args.replay, kind)
        if not authority_rows or len(replay_rows) < len(authority_rows):
            raise SystemExit("invalid {} output inventory".format(kind))

        common = replay_rows[:len(authority_rows)]
        metadata_equal = all(
            lhs["index"] == rhs["index"] and
            lhs["cycle"] == rhs["cycle"] and
            lhs["time_text"] == rhs["time_text"]
            for lhs, rhs in zip(authority_rows, common)
        )
        first_mismatch = None
        for lhs, rhs in zip(authority_rows, common):
            if lhs["payload"] != rhs["payload"]:
                first_mismatch = lhs["index"]
                break
        replay_only = replay_rows[len(authority_rows):]
        comparisons[kind] = {
            "authority_count": len(authority_rows),
            "replay_count": len(replay_rows),
            "common_metadata_equal": metadata_equal,
            "common_payloads_bitwise_equal": first_mismatch is None,
            "first_mismatch": first_mismatch,
            "authority_common_payload_sha256": digest_payloads(authority_rows),
            "replay_common_payload_sha256": digest_payloads(common),
            "replay_only": [
                {"index": row["index"], "cycle": row["cycle"],
                 "time_text": row["time_text"],
                 "payload_sha256": sha256(row["payload"])}
                for row in replay_only
            ],
        }

    passed = all(
        row["common_metadata_equal"] and row["common_payloads_bitwise_equal"]
        for row in comparisons.values()
    )
    result = {
        "schema": "brill_amr_history_n128_identity_v2",
        "disposition": "N128_REPLAY_IDENTITY_PASS" if passed
                       else "N128_NUMERICAL_IDENTITY_FAILED",
        "authority_history_sha256": sha256(args.history.read_bytes()),
        "events_replayed": len(ledger),
        "endpoint_time_hex": events[-1]["time_hex"],
        "terminal_output_asymmetry":
            "replay writes its tlim snapshot; strict-chi-failed authority does not",
        "outputs": comparisons,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("same-resolution replay numerical payload differs")
    print("N128_REPLAY_IDENTITY_PASS")


if __name__ == "__main__":
    main()
