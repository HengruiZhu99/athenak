#!/usr/bin/env python3
"""Build the single target-refinement replay authorized by the chi audit."""

import argparse
import json
from pathlib import Path


TARGET_GIDS = (35, 60)


def checksum(text):
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return "{:016x}".format(value)


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def canonical(leaves):
    ordered = sorted(leaves, key=lambda item: (item[0], item[3], item[2], item[1]))
    return compact(ordered), ordered


def refine(leaves, location):
    if location not in leaves:
        raise SystemExit("requested target/balance leaf is absent: {}".format(location))
    leaves.remove(location)
    level, lx1, lx2, lx3 = location
    if lx3 != 0:
        raise SystemExit("the bounded Cartoon control requires collapsed lx3=0")
    for ox1 in range(2):
        for ox2 in range(2):
            leaves.add((level + 1, 2 * lx1 + ox1, 2 * lx2 + ox2, 0))


def neighbor(left, right, finest):
    def interval(level, index):
        scale = 1 << (finest - level)
        return index * scale, (index + 1) * scale

    l1, lx, ly, _ = left
    r1, rx, ry, _ = right
    ax0, ax1 = interval(l1, lx)
    ay0, ay1 = interval(l1, ly)
    bx0, bx1 = interval(r1, rx)
    by0, by1 = interval(r1, ry)
    return (ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1)


def balance(leaves):
    induced = []
    while True:
        finest = max(item[0] for item in leaves)
        ordered = list(leaves)
        coarse = set()
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if abs(left[0] - right[0]) <= 1:
                    continue
                if neighbor(left, right, finest):
                    coarse.add(left if left[0] < right[0] else right)
        if not coarse:
            return induced
        for location in sorted(coarse):
            if location in leaves:
                induced.append(location)
                refine(leaves, location)


def event_line(index, time_hex, cycle, leaves, requested, induced):
    tree, ordered = canonical(leaves)
    time_decimal = format(float.fromhex(time_hex), ".17g")
    created = 3 * (requested + induced)
    base = (
        '{{"type":"event","event":{},"time":"{}","time_hex":"{}",'
        '"cycle":{},"leaves":{},"leaf_count":{},"max_level":{},'
        '"requested_refine":{},"requested_derefine":0,"created":{},'
        '"deleted":0,"balance_induced":{},"tree_checksum":"{}"'.format(
            index, time_decimal, time_hex, cycle, tree, len(ordered),
            max(item[0] for item in ordered), requested, created, 3 * induced,
            checksum(tree)))
    return '{},"checksum":"{}"}}'.format(base, checksum(base))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shadow-requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    authority_lines = [line.rstrip("\n") for line in args.authority.open() if line.strip()]
    header = json.loads(authority_lines[0])
    events = [json.loads(line) for line in authority_lines[1:]]
    requests = [json.loads(line) for line in args.shadow_requests.open() if line.strip()]
    first = {}
    for row in requests:
        gid = row.get("gid")
        if gid in TARGET_GIDS and row.get("requested_flag") == 1 and gid not in first:
            first[gid] = row
    if set(first) != set(TARGET_GIDS):
        raise SystemExit("missing first persistent target request")
    time_hex = first[TARGET_GIDS[0]]["actual_time_hex"]
    cycle = first[TARGET_GIDS[0]]["cycle"]
    if any(first[gid]["actual_time_hex"] != time_hex or first[gid]["cycle"] != cycle
           for gid in TARGET_GIDS):
        raise SystemExit("paired targets did not first request refinement together")
    request_time = float.fromhex(time_hex)
    prior = [event for event in events if float.fromhex(event["time_hex"]) < request_time]
    if not prior:
        raise SystemExit("no authority tree precedes the target request")
    base_event = prior[-1]
    leaves = set(tuple(item) for item in base_event["leaves"])
    targets = []
    for gid in TARGET_GIDS:
        row = first[gid]
        target = (row["level"] + header["root_level"], row["lx1"],
                  row["lx2"], row["lx3"])
        targets.append(target)
        refine(leaves, target)
    induced = balance(leaves)
    if induced != [(6, 9, 31, 0), (6, 9, 32, 0)]:
        raise SystemExit("unexpected 2:1 balance closure: {}".format(induced))

    output_lines = authority_lines[:base_event["event"] + 2]
    output_lines.append(event_line(base_event["event"] + 1, time_hex, cycle,
                                   leaves, len(targets), len(induced)))
    args.output.write_text("\n".join(output_lines) + "\n")
    summary = {
        "schema": "athenak_targeted_refinement_control_v1",
        "authority_base_event": base_event["event"],
        "authority_base_time_hex": base_event["time_hex"],
        "target_event": base_event["event"] + 1,
        "target_time_hex": time_hex,
        "target_cycle_provenance": cycle,
        "target_gids_at_request": list(TARGET_GIDS),
        "target_locations": [list(item) for item in targets],
        "balance_refined_locations": [list(item) for item in induced],
        "leaf_count": len(leaves),
        "max_level": max(item[0] for item in leaves),
        "tree_checksum": checksum(canonical(leaves)[0]),
        "production_adoption_claim": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("TARGETED_REFINEMENT_CONTROL_SCHEDULE_PASS")


if __name__ == "__main__":
    main()
