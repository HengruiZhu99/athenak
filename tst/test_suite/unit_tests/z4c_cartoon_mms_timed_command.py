#!/usr/bin/env python3
"""Run one configure/build command and emit an outer peak-RSS timing record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("a command argv is required after --")
    if args.record.exists() or args.log.exists():
        raise RuntimeError("refusing to overwrite timing evidence")
    time_path = args.record.with_suffix(args.record.suffix + ".gnu-time")
    if time_path.exists():
        raise RuntimeError("refusing to overwrite GNU time evidence")
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with args.log.open("wb") as log:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_path), "--", *command],
            stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    finished_utc = datetime.now(timezone.utc).isoformat()
    timing_text = time_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", timing_text)
    if match is None:
        raise RuntimeError("GNU time output lacks maximum resident set size")
    record = {"schema": "athenak_z4c_cartoon_mms_timed_command_v1",
              "command": command, "started_at_utc": started_utc,
              "finished_at_utc": finished_utc, "elapsed_seconds": elapsed,
              "peak_rss_kib": int(match.group(1)),
              "exit_code": completed.returncode,
              "log_path": str(args.log.resolve()),
              "gnu_time_path": str(time_path.resolve())}
    temporary = args.record.with_suffix(args.record.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.record)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"FAIL: {error}")
