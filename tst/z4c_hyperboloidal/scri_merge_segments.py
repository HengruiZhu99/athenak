#!/usr/bin/env python3
"""Merge inspected native run segments, retaining restart/window provenance.

Requires start/final checkpoint receipts and successful native exit receipts.
Never interpolate histories or combine independently reduced extrema. Rows
beyond a restored checkpoint are superseded, not retained as a second branch.
The first post-restart extrema window is flagged if the checkpoint was between
history dumps. Such a window must be excluded from matched-window comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scri_run_summary import history


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(arrays, starts, windowed=False):
    """Return rows, row provenance, and rollback/duplicate counts."""
    rows, origins, discarded = [], [], []
    for segment, (array, start) in enumerate(zip(arrays, starts)):
        time = start['time_native']
        assert np.all(array[:, 0] >= time-1e-12), 'history predates restored state'
        removed = 0
        if segment:
            while rows and rows[-1][0] > time+1e-12:
                rows.pop(); origins.pop(); removed += 1
            assert rows and rows[-1][0] <= time+1e-12
        first_after_restart = bool(segment)
        previous = time
        for row_number, row in enumerate(array):
            if segment and abs(row[0]-time) < 1e-12:
                # A startup sample has no new stage window. Instantaneous
                # diagnostics at the same state must agree if both were saved.
                if rows and abs(rows[-1][0]-time) < 1e-12 and not windowed:
                    np.testing.assert_allclose(row, rows[-1], rtol=2e-13, atol=1e-25)
                if windowed or (rows and abs(rows[-1][0]-time) < 1e-12):
                    continue
            assert not rows or row[0] > rows[-1][0], 'overlapping/nonmonotone history'
            complete = not (windowed and first_after_restart and
                            abs(rows[-1][0]-time) > 1e-12)
            origins.append(dict(segment=segment, source_row=row_number,
                                time_native=float(row[0]),
                                window_start_native=float(previous) if windowed else None,
                                complete_window=complete))
            rows.append(row)
            previous = row[0]
            first_after_restart = False
        discarded.append(removed)
    return np.array(rows), origins, discarded


def merge(segments, output):
    segments = [Path(s).resolve() for s in segments]
    output = Path(output)
    assert len(segments) >= 2 and not output.exists()
    records, starts, finals = [], [], []
    for i, root in enumerate(segments):
        assert int((root/'exit-code.txt').read_text()) == 0, root
        start = json.loads((root/'start-checkpoint.json').read_text())
        final = json.loads((root/'final-checkpoint.json').read_text())
        assert start['identity_sha256'] == final['identity_sha256']
        assert final['cycle'] > start['cycle'] and final['time_native'] > start['time_native']
        if i:
            assert start['sha256'] == finals[-1]['sha256'], 'restart chain mismatch'
            assert start['identity_sha256'] == starts[0]['identity_sha256']
            assert start['dt_native'] == starts[0]['dt_native']
            assert (root/'executable.sha256').read_text().split()[0] == records[0]['executable_sha256']
        else:
            assert start['cycle'] == 0 and start['time_native'] == 0
        starts.append(start); finals.append(final)
        records.append(dict(directory=str(root), start=start, final=final,
                            executable_sha256=(root/'executable.sha256').read_text().split()[0]))
    names = sorted(p.name for p in segments[0].glob('*.hst')
                   if '.scri' in p.name or '.z4c_hyperboloidal.hst' in p.name)
    assert len(names) == 10, ('expected main + old scri + four investigation + four shells', names)
    output.mkdir(parents=True)
    manifest = dict(segments=records, streams={},
                    note='Raw inputs remain in each source directory; extrema are never interpolated.')
    for name in names:
        arrays, columns, hashes = [], [], []
        for root in segments:
            a, c = history(root/name)
            arrays.append(a); columns.append(list(c)); hashes.append(digest(root/name))
        assert all(c == columns[0] for c in columns)
        windowed = '.scri.' in name or '.scri-investigation-' in name
        a, origins, discarded = combine(arrays, starts, windowed)
        assert a[-1, 0] <= finals[-1]['time_native']+1e-12
        header = 'Merged native segments; see merge-manifest.json.\n' + ' '.join(
            '[{}]={}'.format(i+1, label) for i, label in enumerate(columns[0]))
        np.savetxt(output/name, a, fmt='%.17e', header=header)
        manifest['streams'][name] = dict(windowed=windowed, rows=origins,
                                        discarded_after_restart=discarded,
                                        source_sha256=hashes, sha256=digest(output/name))
    # Horizon records use an integer cycle because their printed timestamps are
    # low precision. Preserve the paired shape row from the same finder call.
    summary_names = sorted(p.name for p in segments[0].glob('*.horizon_summary_*.txt'))
    for name in summary_names:
        shape_name = name.replace('horizon_summary_', 'horizon_shape_')
        result, shapes = [], []
        for i, root in enumerate(segments):
            a = np.loadtxt(root/name, ndmin=2)
            b = np.loadtxt(root/shape_name, ndmin=2)
            assert len(a) == len(b) and np.isfinite(a).all() and np.isfinite(b).all()
            if i:
                while result and result[-1][0] > starts[i]['cycle']:
                    result.pop(); shapes.pop()
            for row, shape in zip(a, b):
                assert row[0] >= starts[i]['cycle']
                if result and row[0] == result[-1][0]:
                    np.testing.assert_allclose(row, result[-1], rtol=2e-13, atol=1e-25)
                    np.testing.assert_allclose(shape, shapes[-1], rtol=2e-13, atol=1e-25)
                    continue
                assert not result or row[0] > result[-1][0]
                result.append(row); shapes.append(shape)
        np.savetxt(output/name, result, fmt='%.17e', header=(segments[0]/name).read_text().splitlines()[0].lstrip('# '))
        np.savetxt(output/shape_name, shapes, fmt='%.17e')
    (output/'merge-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Endpoint and field-analysis consumers use the original initial state and
    # the last restored branch's final state, not a fabricated merged checkpoint.
    (output/'start-checkpoint.json').write_text(json.dumps(starts[0], indent=2)+'\n')
    (output/'final-checkpoint.json').write_text(json.dumps(finals[-1], indent=2)+'\n')
    (output/'executable.sha256').write_text((segments[0]/'executable.sha256').read_text())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.segment, args.output)
    print(json.dumps({k: dict(rows=len(v['rows']), discarded=v['discarded_after_restart'],
                             partial_windows=sum(not row['complete_window'] for row in v['rows']))
                      for k, v in result['streams'].items()}, indent=2))
