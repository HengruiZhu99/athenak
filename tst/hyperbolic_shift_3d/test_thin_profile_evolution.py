#!/usr/bin/env python3
"""Prove optional cached profiles preserve evolution and match snapshot sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ('old-full', 'full', 'sparse', 'profile', 'mpiexec', 'out'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
deadline = time.monotonic()+400
records = []
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def run(name, binary, ranks, profiles, kind='oblique', steps=10):
    prefix = (a.out/name).resolve(); dt = 1/4096
    cmd = [str(a.mpiexec.resolve()), '-n', str(ranks), str(binary.resolve()), '12', str(prefix), kind,
           str(steps*dt), str(dt), '-', '-', '120', str(steps//2*dt), '.001', '6', '12', '2']
    env = dict(os.environ, HYPERBOLIC_TRACE_LAYERS='1', HYPERBOLIC_TRACE_SHIFTED='1',
               HYPERBOLIC_TRACE_KAPPA='1', HYPERBOLIC_TRACE_NORMAL_RATIO='.25',
               HYPERBOLIC_TRACE_PROFILES=str(int(profiles)), HYPERBOLIC_SPARSE_POISON='1')
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=min(150, deadline-time.monotonic()))
    prefix.with_suffix('.log').write_text(result.stdout+result.stderr)
    records.append(dict(name=name, command=cmd, returncode=result.returncode,
                        environment={k: v for k, v in env.items() if k.startswith('HYPERBOLIC_')}, binary_sha256=sha(binary)))
    (a.out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    assert result.returncode == 0, (name, result.stderr)
    report = json.loads(prefix.with_suffix('.report.json').read_text())
    assert report['status'] == 'completed' and report['t'] == steps*dt
    assert report['lapse_floor_node_interventions'] == 0 and report['checkpoint_admitted']
    if binary != a.old_full:
        assert report['profiles_enabled'] == profiles and report['profile_samples'] == (3 if profiles else 0)
    if profiles:
        assert report['profile_rows_per_sample'] == 1462
        data = np.genfromtxt(prefix.with_suffix('.profiles.csv'), delimiter=',', names=True)
        assert len(data) == 3*1462 and np.isfinite(data.view(float)).all()
        np.testing.assert_array_equal(np.unique(data['t']), [0, steps//2*dt, steps*dt])
        snapshot = prefix.with_suffix('.snapshot.csv')
        subprocess.run([str(a.profile.resolve()), str(prefix.with_suffix('.checkpoint')), str(snapshot)], check=True,
                       capture_output=True, timeout=min(100, deadline-time.monotonic()))
        lines = prefix.with_suffix('.profiles.csv').read_bytes().splitlines(keepends=True)
        assert snapshot.read_bytes() == lines[0]+b''.join(lines[-1462:]), 'Online and snapshot profiles differ'
        if kind == 'minkowski':
            assert np.max(abs(data['v0']-1)) < 1e-11
            assert max(np.max(abs(data[f'v{f}'])) for f in range(1, 31)) < 1e-11
            assert max(np.max(abs(data[f'dr_v{f}'])) for f in range(31)) < 1e-9
    else:
        assert not prefix.with_suffix('.profiles.csv').exists()
    print(name, 'passed', flush=True)
    return prefix.with_suffix('.checkpoint')


old = run('old-full', a.old_full, 1, False)
disabled = run('new-disabled', a.full, 1, False)
enabled = run('new-enabled-sparse', a.sparse, 2, True)
assert old.read_bytes() == disabled.read_bytes() == enabled.read_bytes(), 'Observation changed checkpoint bytes'
run('minkowski-enabled-sparse', a.sparse, 2, True, kind='minkowski', steps=4)
result = dict(status='passed', comparisons={'old_vs_new_disabled': True, 'old_vs_sparse_profiles': True,
                                          'online_vs_snapshot_profiles': True, 'minkowski_profiles': True},
              checkpoint_sha256=sha(old), profile_binary_sha256=sha(a.profile),
              scope='Bounded CPU diagnostic invariance across one full and two sparse ranks; no GPU or long-time stability claim.')
(a.out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
