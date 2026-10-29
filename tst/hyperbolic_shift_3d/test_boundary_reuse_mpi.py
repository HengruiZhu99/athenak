#!/usr/bin/env python3
"""Bounded bitwise trajectory, partition, and restart audit of boundary reuse."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'control', 'mpiexec', 'initial', 'output-dir'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), 'Use a fresh evidence directory'
deadline = time.monotonic() + 410
records = []

def run(name, kind='pulse', fast=True, ranks=1, end=.00075, restart='-'):
    prefix = out / name
    n = 16 if kind == 'puncture' else 12
    cmd = [str(a.mpiexec.resolve()), '-n', str(ranks),
           str((a.binary if fast else a.control).resolve()), str(n), str(prefix),
           kind, str(end), '.00025',
           str(a.initial.resolve()) if kind == 'puncture' else '-', str(restart),
           '120', '.001', '.001', '6', '12', '2']
    start = time.monotonic()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=min(150, deadline-start))
    prefix.with_suffix('.log').write_text(result.stdout)
    assert result.returncode == 0, (name, result.stdout)
    report = json.loads(prefix.with_suffix('.report.json').read_text())
    assert report['status'] == 'completed' and report['checkpoint_admitted']
    assert report['diagnostics_current'] and report['t'] == end
    assert report['lapse_floor_node_interventions'] == 0
    assert report['max_null_surface'] < 5e-13 and report['max_null_ghost'] < 5e-13
    if fast:
        assert report['boundary_execution'] == 'cached_global_projection_angular'
        assert report['boundary_cache_hits'] > 0
        assert all(row[2] == 0 for row in report['collective_calls_by_rank'])
    cp = prefix.with_suffix('.checkpoint')
    records.append(dict(name=name, command=cmd, seconds=time.monotonic()-start,
                        sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
    return cp

control = run('control', fast=False)
one = run('one')
two = run('two', ranks=2)
part = run('part', end=.00025)
restart = run('restart', ranks=2, restart=part)
assert control.read_bytes() == one.read_bytes() == two.read_bytes() == restart.read_bytes(), 'Pulse trajectory differs'
for kind in ('minkowski', 'puncture'):
    control = run(kind+'-control', kind=kind, fast=False, end=.0005)
    candidate = run(kind+'-fast', kind=kind, end=.0005, ranks=2)
    assert control.read_bytes() == candidate.read_bytes(), kind+' trajectory differs'
(out/'verification.json').write_text(json.dumps(dict(status='passed',
    scope='short bitwise trajectories only; no performance or long-time claim',
    commands=records), indent=2)+'\n')
print('PASS: pulse partition/restart and Minkowski/puncture checkpoints bitwise equal')
