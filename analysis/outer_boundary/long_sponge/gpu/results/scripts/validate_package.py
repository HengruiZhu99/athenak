#!/usr/bin/env python3
"""Verify the compact evidence package without raw dumps or cluster access."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'manifest.json').read_text())
for name, record in manifest.items():
    path = root / name
    assert path.is_file(), name
    assert path.stat().st_size == record['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name

last_times = {}
for path in sorted((root / 'histories').glob('*.npz')):
    with np.load(path, allow_pickle=False) as arrays:
        for key in ('user', 'z4c', 'mhd'):
            if key not in arrays:
                continue
            data = arrays[key]
            columns = [str(value) for value in arrays[key + '_columns']]
            assert data.ndim == 2 and data.shape[1] == len(columns), path
            assert np.isfinite(data).all(), path
            times = data[:, columns.index('time')]
            assert np.all(np.diff(times) >= 0), path
            if key == 'user':
                last_times[path.stem] = float(times[-1])

for directory in sorted((root / 'jobs').iterdir()):
    for name, expected in json.loads((directory / 'input-manifest.json').read_text()).items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, (directory, name)

assert last_times['8842171-radial_k0'] == 20000.0
assert abs(last_times['8842248-theta_primary'] - 30800.0) < 1e-7
assert '8842248-theta_lapse01' not in last_times
assert '8842283-theta_amplitude' not in last_times
for name in ('8842171-radial_k0-checkpoint-validation.json',
             '8842171-zero-checkpoint-validation.json',
             '8842172-zero-checkpoint-validation.json',
             '8842248-zero_primary-checkpoint-validation.json'):
    record = json.loads((root / 'validation' / name).read_text())
    assert record['passed'] and record['all_payload_finite']
    assert record['invalid_metric_cells_including_ghosts'] == 0
    assert record['ranks'] == 8
access = json.loads((root / 'collection-access.json').read_text())
assert not access['fresh_aurora_collection_succeeded']
assert not access['C_final_checkpoint_or_stop_known']
assert not access['D_final_checkpoint_or_stop_known']
print(json.dumps({'passed': True, 'manifest_files': len(manifest),
                  'cached_history_last_times': last_times,
                  'C_and_D_final_results_available': False}, indent=2))
