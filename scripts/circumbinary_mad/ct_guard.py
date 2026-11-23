"""Check every completed magnetic history row since the last successful check."""

import math
from pathlib import Path
import re
import sys


def check(path, cursor=None):
    with path.open("rb") as stream:
        stream.readline()
        names = re.findall(r"\[\d+\]=(\S+)", stream.readline().decode())
        offset = int(cursor.read_text()) if cursor and cursor.exists() else 0
        if offset > path.stat().st_size:
            offset = 0
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b"\n"):
                break
            offset += len(line)
            if line.startswith(b"#") or not line.strip():
                continue
            row = line.split()
            if len(row) != len(names):
                raise ValueError("Malformed completed history row")
            div, bad = (float(row[names.index(k)]) for k in ("ct_divB_relative", "ct_nonfinite"))
            if bad or not math.isfinite(div) or div > 1e-7:
                raise ValueError(
                    f"Magnetic constraint at t={row[0].decode()}: "
                    f"relative divB={div}, nonfinite={bad}"
                )
    if cursor:
        cursor.write_text(str(offset))


if __name__ == "__main__":
    check(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
