#!/usr/bin/env python3
"""Poisoned-dependency bitwise audit of sparse MPI stage exchange."""
import argparse
import array
import struct
import sys
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'control', 'mpiexec', 'initial', 'output-dir'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--measured-partition', action='store_true', help='Test the balanced target with deliberately unequal synthetic cuts, not performance weights')
a = p.parse_args()
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), 'Use a fresh evidence directory'
deadline = time.monotonic() + 410
records = []

def partition(n, ranks):
    if not a.measured_partition:
        return []
    cells=12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2
    assert ranks in (1, 2)
    cuts=[0,cells] if ranks==1 else [0,int(.61*cells),cells]
    path=out/f'partition-test-n{n}-r{ranks}.txt'
    text=f'HYP3D_COST_PARTITION_V1\n{n} {cells} {ranks}\n'+('0'*64)+'\n'+' '.join(map(str,cuts))+'\n'
    if path.exists():
        assert path.read_text()==text
    else:
        path.write_text(text)
    return [str(path)]


def run(name, kind='pulse', fast=True, ranks=1, end=.00075, restart='-'):
    prefix = out / name
    n = 16 if kind == 'puncture' else 12
    cmd = [str(a.mpiexec.resolve()), '-n', str(ranks),
           str((a.binary if fast else a.control).resolve()), str(n), str(prefix),
           kind, str(end), '.00025',
           str(a.initial.resolve()) if kind == 'puncture' else '-', str(restart),
           '120', '.001', '.001', '6', '12', '2']
    if fast:cmd += partition(n, ranks)
    start = time.monotonic()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=min(150, deadline-start),
                            env=dict(os.environ,HYPERBOLIC_SPARSE_POISON='1'))
    prefix.with_suffix('.log').write_text(result.stdout)
    assert result.returncode == 0, (name, result.stdout)
    report = json.loads(prefix.with_suffix('.report.json').read_text())
    assert report['status'] == 'completed' and report['checkpoint_admitted']
    assert report['diagnostics_current'] and report['t'] == end
    assert report['lapse_floor_node_interventions'] == 0
    assert report['max_null_surface'] < 5e-13 and report['max_null_ghost'] < 5e-13
    if fast:
        if a.measured_partition:
            assert report['ownership_partition']=='measured_fixed_contiguous'
            assert report['partition_source_report_sha256']=='0'*64
            if ranks==2:assert report['minimum_owned_cells'] != report['maximum_owned_cells']
        assert report['boundary_execution'] == 'cached_global_projection_angular'
        assert report['boundary_cache_hits'] > 0
        assert report['volume_exchange'] == 'cached_sparse_alltoallv'
        assert report['sparse_poison_unused'] is True
        assert all(row[5]>0 and row[6]==1 for row in report['sparse_exchange_by_rank'])
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
# Deliberately nonphysical but finite interior extrinsic curvature: initial
# RHS evaluation succeeds, then an RK stage rejects the metric. This exercises
# distributed failed-state assembly and rollback without a production backdoor.
b = one.read_bytes()
cells = struct.unpack_from('<Q', b, 32)[0]
values = array.array('d')
values.frombytes(b[152:152+31*cells*8])
if sys.byteorder != 'little':
    values.byteswap()
values[31*5000+7], values[31*5000+10] = 1e5, -1e5
stress = out/'stress.txt'
with stress.open('w') as stream:
    stream.write(str(len(values))+'\n')
    for point in range(cells):
        stream.write(' '.join(format(v, '.17g') for v in values[31*point:31*(point+1)])+'\n')
failed = []
for fast, ranks in ((False, 1), (True, 2)):
    name = 'failure-sparse' if fast else 'failure-control'
    prefix = out/name
    cmd = [str(a.mpiexec.resolve()), '-n', str(ranks),
           str((a.binary if fast else a.control).resolve()), '12', str(prefix),
           'pulse', '.00025', '.00025', str(stress), '-', '120', '.001', '.001', '6', '12', '2']
    if fast:cmd += partition(12, ranks)
    start = time.monotonic()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=min(150, deadline-start),
                            env=dict(os.environ, HYPERBOLIC_SPARSE_POISON='1'))
    prefix.with_suffix('.log').write_text(result.stdout)
    report = json.loads(prefix.with_suffix('.report.json').read_text())
    assert result.returncode == 2 and report['status'] == 'failed'
    assert report['initial_rhs'] is not None and report['steps'] == 0
    assert report['error'] == 'device RHS failure at active cell 5000 reason 2'
    assert report['checkpoint_written'] and not report['checkpoint_admitted']
    failed.append(prefix)
    records.append(dict(name=name, command=cmd, seconds=time.monotonic()-start,
                        checkpoint_sha256=hashlib.sha256(prefix.with_suffix('.checkpoint').read_bytes()).hexdigest(),
                        failed_stage_sha256=hashlib.sha256(prefix.with_suffix('.failed-stage.bin').read_bytes()).hexdigest()))
for suffix in ('.checkpoint', '.failed-stage.bin'):
    assert failed[0].with_suffix(suffix).read_bytes() == failed[1].with_suffix(suffix).read_bytes(), suffix+' failure evidence differs'
(out/'verification.json').write_text(json.dumps(dict(status='passed',
    scope='short bitwise trajectories, restart, rollback and failed-stage capture; no performance or long-time claim',
    commands=records), indent=2)+'\n')
print('PASS: poisoned sparse pulse partition/restart and Minkowski/puncture checkpoints bitwise equal')
