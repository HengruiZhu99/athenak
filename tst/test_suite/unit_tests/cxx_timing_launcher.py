#!/usr/bin/env python3
"""Transparent compiler launcher recording per-TU wall time and peak RSS."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import resource
import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        os.execvp(sys.argv[1], sys.argv[1:])
    _, status, usage = os.wait4(pid, 0)
    argv = sys.argv[1:]
    source_suffixes = (".c", ".cc", ".cpp", ".cxx", ".C", ".cu")
    sources = [value for value in argv if value.endswith(source_suffixes)]
    record = json.dumps({"schema": "athenak_z4c_cartoon_mms_tu_timing_v1",
                         "argv": argv, "source": sources[0] if len(sources) == 1 else None,
                         "wall_seconds": time.monotonic() - started,
                         "max_rss_kib": usage.ru_maxrss,
                         "object": next((argv[index + 1]
                                         for index, value in enumerate(argv[:-1])
                                         if value == "-o"), None),
                         "exit_code": os.waitstatus_to_exitcode(status)},
                        separators=(",", ":")) + "\n"
    log = os.environ.get("ATHENA_MMS_TIMING_LOG")
    if log:
        descriptor = os.open(Path(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, record.encode())
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    raise SystemExit(main())
