"""Check completed magnetic-history samples and record abrupt post-return field growth."""

import json
import math
from pathlib import Path
import re
import sys


def check(path, cursor=None):
    state = {"offset": 0}
    if cursor and cursor.exists():
        saved = json.loads(cursor.read_text())
        state = {"offset": saved} if isinstance(saved, int) else saved
    stat = path.stat()
    if state.get("inode", stat.st_ino) != stat.st_ino or state["offset"] > stat.st_size:
        state = {"offset": 0}
    with path.open("rb") as stream:
        stream.readline()
        names = re.findall(r"\[\d+\]=(\S+)", stream.readline().decode())
        indices = [
            names.index(k)
            for k in (
                "time",
                "zoom_zone",
                "zoom_event",
                "ct_face_B_max",
                "ct_divB_relative",
                "ct_nonfinite",
            )
        ]
        stream.seek(state["offset"])
        for line in stream:
            if not line.endswith(b"\n"):
                break
            state["offset"] += len(line)
            if line.startswith(b"#") or not line.strip():
                continue
            row = line.split()
            if len(row) != len(names):
                raise ValueError("Malformed completed history row")
            t, zone, event, field, div, bad = (float(row[i]) for i in indices)
            if bad or not math.isfinite(div) or not math.isfinite(field) or div > 1e-7:
                raise ValueError(
                    f"Magnetic constraint at t={t}: relative divB={div}, "
                    f"peak face field={field}, nonfinite={bad}"
                )
            previous = state.get("previous")
            if (
                previous
                and t > previous[0]
                and event > previous[2]
                and zone == 0
                and previous[1] != 0
                and 0 < previous[3] < field / 10
            ):
                message = {
                    "time": t,
                    "event": int(event),
                    "prior_peak": previous[3],
                    "returned_peak": field,
                    "relative_divB": div,
                }
                print(
                    "Inspect post-return field amplification", json.dumps(message), file=sys.stderr
                )
                if cursor:
                    with path.with_suffix(".magnetic-events.jsonl").open("a") as f:
                        print(json.dumps(message), file=f)
            state["previous"] = [t, zone, event, field]
    if cursor:
        state["inode"] = stat.st_ino
        tmp = cursor.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(cursor)


if __name__ == "__main__":
    history = Path(sys.argv[1])
    check(history, Path(sys.argv[2]) if len(sys.argv) > 2 else history.with_suffix(".ct-cursor"))
