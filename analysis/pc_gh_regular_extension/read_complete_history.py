"""Read all restart segments without silently discarding earlier history headers."""
import re
import numpy as np


def read_history(path):
    names, segments, current = None, [], []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            found = re.findall(r'\[\d+\]=(\S+)', line)
            if found:
                if names is not None and names != found:
                    raise ValueError(f'History columns changed between segments: {path}')
                if current:
                    segments.append(np.asarray(current))
                    current = []
                names = found
            continue
        row = list(map(float, line.split()))
        if names is None or len(row) != len(names) or not np.isfinite(row).all():
            raise ValueError(f'Incomplete/nonfinite history record: {path}')
        current.append(row)
    if current:
        segments.append(np.asarray(current))
    if not segments or names[0] != 'time':
        raise ValueError(f'No complete time history: {path}')
    merged = np.empty((0, len(names)))
    replaced, repeated = 0, 0
    audit = []
    for segment in segments:
        if np.any(np.diff(segment[:, 0]) < 0):
            raise ValueError(f'Non-increasing time within a history segment: {path}')
        raw_count = len(segment)
        # Scheduled and forced wall-stop output can write the same timestamp.
        # Retain the last record and count this explicitly, after checking both
        # records for finiteness and column completeness above.
        segment = segment[np.r_[np.diff(segment[:, 0]) > 0, True]]
        repeated += raw_count-len(segment)
        # If a restart replays an earlier interval, its new branch supersedes
        # that interval. The untouched raw file retains both branches.
        cut = int(np.searchsorted(merged[:, 0], segment[0, 0]))
        replaced += len(merged)-cut
        merged = np.vstack([merged[:cut], segment])
        audit.append(dict(rows=len(segment), raw_rows=raw_count, start=float(segment[0, 0]),
                          end=float(segment[-1, 0])))
    return ({name: merged[:, i] for i, name in enumerate(names)},
            dict(segments=audit, superseded_rows=replaced, repeated_timestamp_rows=repeated,
                 retained_rows=len(merged)))
