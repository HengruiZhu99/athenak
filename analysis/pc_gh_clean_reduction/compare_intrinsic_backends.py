#!/usr/bin/env python3
"""Export or compare float64 seeded decomposition snapshots, including all ghosts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
from intrinsic_restart import read_restart

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--export', type=Path)
p.add_argument('--compare', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert (a.export is None) != (a.compare is None)
arrays = {}; manifest = {}; records = []
for dim in [2, 3]:
    for order in [2, 4, 6]:
        case = f'fd{order}-{dim}d'
        paths = [a.root / (case + '-seed-multi.rst')]
        paths += sorted((a.root / (case + '-seeded-multi') / 'rst').glob('*.rst'))
        assert len(paths) == 4
        for cycle, path in enumerate(paths):
            data = read_restart(path)
            assert data['cycle'] == cycle
            key = f'{case}-cycle{cycle}'
            arrays[key] = data['state']
            arrays[key + '-locations'] = data['locations']
            manifest[key] = dict(path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                time=data['time'], shape=list(data['state'].shape))
if a.export:
    np.savez_compressed(a.export, **arrays)
    result = dict(status='EXPORTED', snapshots=manifest,
        archive_sha256=hashlib.sha256(a.export.read_bytes()).hexdigest())
else:
    with np.load(a.compare) as other:
        for key, info in manifest.items():
            assert np.array_equal(arrays[key + '-locations'], other[key + '-locations'])
            expected = arrays[key]; actual = other[key]
            assert expected.shape == actual.shape and np.isfinite(actual).all()
            delta = actual - expected
            error = float(np.max(abs(delta) / (1 + abs(expected))))
            records.append(dict(snapshot=key, normalized_max=error,
                absolute_max=float(np.max(abs(delta))),
                status='PASS' if error <= 2e-12 else 'FAIL'))
    result = dict(status='PASS' if all(r['status']=='PASS' for r in records) else 'FAIL',
        tolerance=2e-12, records=records, local_snapshots=manifest,
        archive_sha256=hashlib.sha256(a.compare.read_bytes()).hexdigest(),
        scope='All 50 float64 fields, all blocks and stored ghost cells, four synchronized cycles; uniform periodic seeded fixtures')
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','snapshots','local_snapshots']}))
assert result['status'] != 'FAIL'
