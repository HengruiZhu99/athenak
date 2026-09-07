#!/usr/bin/env python3
"""Check same-cell invariance and correction timing in the one-step fixture."""
import argparse
import json
from pathlib import Path
import numpy as np
from read_state_budget import records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', type=Path)
parser.add_argument('--physical-boundary', action='store_true')
args = parser.parse_args()
rows = []
for header, geometry, data in records(args.file):
    assert np.isfinite(data).all()
    assert np.array_equal(data[..., 1]-data[..., 0], data[..., 2])
    if header['operation'] not in [11, 12, 13]:
        continue
    active = data[:, header['ks']:header['ke']+1, header['js']:header['je']+1,
                  header['is']:header['ie']+1, 2]
    assert not np.count_nonzero(active)
    if header['operation'] != 13:
        assert not np.count_nonzero(data[:22, ..., 2])
    rows.append(dict(operation=header['operation'], stage=header['stage'],
                     cycle=header['cycle'],
                     max_primary_delta=float(abs(data[:22, ..., 2]).max()), max_delta=float(abs(data[..., 2]).max())))
assert [r['stage'] for r in rows if r['operation'] == 11] == [0, 1, 2, 3]
assert [r['stage'] for r in rows if r['operation'] == 12] == [3]
assert [r['stage'] for r in rows if r['operation'] == 13] == (
    [0, 1, 2, 3, 3] if args.physical_boundary else [])
print(json.dumps(dict(status='PASS', records=rows), indent=2))
