#!/usr/bin/env python3
"""Verify copied Perlmutter evidence whose SHA256SUMS uses absolute paths."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(directory: Path) -> tuple[int, list[str]]:
    manifest = directory / "SHA256SUMS"
    failures = 0
    messages: list[str] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        expected, recorded_path = line.split(maxsplit=1)
        recorded_path = recorded_path.lstrip("*")
        marker = "/evidence/"
        if marker not in recorded_path:
            failures += 1
            messages.append(f"FAIL line={line_number} unsupported_path={recorded_path}")
            continue
        relative = Path(recorded_path.split(marker, 1)[1])
        parts = relative.parts
        if len(parts) < 2:
            failures += 1
            messages.append(f"FAIL line={line_number} malformed_path={recorded_path}")
            continue
        local_path = directory.joinpath(*parts[1:])
        if not local_path.is_file():
            failures += 1
            messages.append(f"FAIL line={line_number} missing={local_path}")
            continue
        actual = digest(local_path)
        status = "OK" if actual == expected else "FAIL"
        failures += actual != expected
        messages.append(
            f"{status} expected={expected} actual={actual} path={local_path}"
        )
    return failures, messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    total_failures = 0
    for directory in args.directories:
        failures, messages = verify(directory)
        total_failures += failures
        print(f"DIRECTORY {directory}")
        print("\n".join(messages))
        print(f"RESULT failures={failures}")
    return int(total_failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
