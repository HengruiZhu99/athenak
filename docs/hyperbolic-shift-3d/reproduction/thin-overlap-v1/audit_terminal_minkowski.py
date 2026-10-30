#!/usr/bin/env python3
"""Audit native completion when PBS interrupted the enclosing result collector.

Never changes the scheduler/worker outcome or generates an admission document.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tst/hyperbolic_shift_3d/aurora'))
import control
import preflight

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args(); base = a.root/'campaigns/stability-thin-minkowski-v1'
read = lambda path: json.loads(path.read_text())
binding = read(base/'binding.json'); plan = read(base/'plan.json')
terminal = read(base/'collected.json'); worker = read(base/'worker-status.json')
assert terminal['scheduler']['job_state'] == 'F' and terminal['scheduler']['Exit_status'] == 143
assert 'terminated by' in terminal['scheduler']['comment']
assert worker['status'] == 'running' and len(worker['cases']) == 1
assert worker['job_id'] == terminal['job_id']
assert worker['plan_sha256'] == control.digest(base/'plan.json') == binding['plan_sha256']
assert worker['source_manifest_sha256'] == binding['source_manifest_sha256'] == control.digest(a.root/binding['source_manifest'])
build = read(a.root/'build-logs/thin-v1/build-status.json')
assert build['status'] == 'passed'
for name, digest in binding['binaries'].items():
    assert build['binaries'][name] == worker['binaries'][name] == digest
affinity = worker['affinity']['records']
assert worker['affinity']['passed'] and len(affinity) == 12
assert {r['rank'] for r in affinity} == set(range(12))
assert len({r['host'] for r in affinity}) == 1
assert all(r['mask'] == f"{r['rank']//2}.{r['rank']%2}" for r in affinity)
manifest = read(base/'terminal-output-manifest.json')
for name, item in manifest.items():
    path = a.root/name
    assert path.stat().st_size == item['bytes'] and control.digest(path) == item['sha256'], name
command = plan['commands'][0]; prefix = a.root/command['args'][1]
report = read(prefix.with_suffix('.report.json'))
checks = [control.evaluate_check(report, check) for check in command['checks']]
assert all(c['passed'] for c in checks)
checkpoint = prefix.with_suffix('.checkpoint'); header = preflight.read_checkpoint(checkpoint)
assert header['magic'] == 'HYP3D_MPI_V12' and header['time'] == report['t'] == 1
assert header['steps'] == report['steps'] == 2048 and header['dt'] == report['dt'] == 1/2048
for key in ['core_evolved_layers', 'shifted_overlap_donors', 'trace_kappa', 'trace_normal_ratio']:
    assert header[key] == report[key]
payload = np.fromfile(checkpoint, dtype='<f8', offset=header['header_bytes'])
assert len(payload) == 9000416 and np.isfinite(payload).all()
history = np.genfromtxt(prefix.with_suffix('.history.csv'), delimiter=',', names=True)
np.testing.assert_array_equal(history['t'], np.arange(33)/32)
np.testing.assert_array_equal(history['steps'], np.arange(33)*64)
maxima = {}
for key, bound in [('state_deviation', 1e-10), ('H_L2', 1e-10), ('M_L2', 1e-10),
                   ('null_surface', 1e-12), ('null_ghost', 1e-12),
                   ('determinant_residual', 1e-12), ('tracefree_residual', 1e-12)]:
    assert np.isfinite(history[key]).all() and np.all(abs(history[key]) <= bound), key
    maxima[key] = float(np.max(abs(history[key])))
assert np.all(history['min_chi'] > 0) and np.all(history['min_lapse'] > 0)
assert report['lapse_floor_node_interventions'] == 0
result = dict(status='verified_native_completion_with_interrupted_worker',
              job_id=terminal['job_id'], scheduler_exit=143, worker_status=worker['status'],
              native_status=report['status'], t=report['t'], steps=report['steps'], seconds=report['seconds'],
              checks=checks, history_samples=len(history), history_maxima=maxima,
              checkpoint_sha256=control.digest(checkpoint), finite_payload_doubles=len(payload),
              terminal_output_manifest_sha256=control.digest(base/'terminal-output-manifest.json'),
              source_manifest_sha256=binding['source_manifest_sha256'], plan_sha256=binding['plan_sha256'],
              scope='Native L1 paired Minkowski completion through coordinate time 1 verified from terminal outputs. PBS exit143 and unfinished worker are retained. No whole-batch pass, admission, perturbed stability, puncture or convergence claim.')
with a.out.open('x') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps({key: result[key] for key in ['status', 't', 'steps', 'seconds', 'history_maxima']}, indent=2))
