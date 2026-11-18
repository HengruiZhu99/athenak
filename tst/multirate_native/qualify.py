#!/usr/bin/env python3
"""Native Scheme A execution/restart gates; CPU measurements are not GPU claims.

Every invocation creates an isolated result directory. Reads actual shared
checkpoints in stable-GID order, including persistent diagnostic records.
The small-step comparison is a smoke gate, not scientific convergence evidence.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time

HEADER = struct.calcsize('=ii9d38i2di')
MAGIC = 0x4859504552525354


def checkpoint(path):
    with path.open('rb') as stream:
        parameters = bytearray()
        while not parameters.endswith(b'<par_end>\n'):
            byte = stream.read(1)
            if not byte:
                raise ValueError('missing native parameter terminator')
            parameters.extend(byte)
        fixed = stream.read(HEADER)
        if len(fixed) != HEADER:
            raise ValueError('truncated native header')
        nmb, = struct.unpack_from('=i', fixed)
        cycle, = struct.unpack_from('=i', fixed, HEADER-4)
        elapsed, dt = struct.unpack_from('=2d', fixed, HEADER-20)
        atlas_size, = struct.unpack('=Q', stream.read(8))
        atlas = stream.read(atlas_size)
        topology = stream.read(nmb*20)
        stream.read(nmb*4)  # Load costs can change without changing accepted state.
        meta = struct.unpack('=8Q', stream.read(64))
        if meta[:2] != (MAGIC, 1) or meta[7] != 0:
            raise ValueError('expected shared native Scheme A restart version 1')
        identity = stream.read(meta[2])
        stride, = struct.unpack('=Q', stream.read(8))
        data = stream.read()
    if stride != meta[3]*8 or stride % (39*8) or len(data) != nmb*stride:
        raise ValueError('inconsistent native payload extent')
    values = array('d')
    values.frombytes(data)
    if any(not math.isfinite(value) for value in values):
        raise AssertionError('nonfinite accepted field or persistent diagnostic record')
    return dict(path=str(path), blocks=nmb, cycle=cycle, time=elapsed, dt=dt,
                atlas=atlas, topology=topology, identity=identity, stride=stride,
                payload=data, values=values, parameters=parameters.decode())


def identical(a, b):
    for key in ('blocks', 'cycle', 'time', 'dt', 'atlas', 'topology', 'identity', 'stride'):
        if a[key] != b[key]:
            raise AssertionError(f'restarted {key} differs')
    if a['payload'] != b['payload']:
        offset = next(i for i, (x, y) in enumerate(zip(a['payload'], b['payload'])) if x != y)
        raise AssertionError(f'restarted accepted payload differs at byte {offset}')
    return dict(bitwise_equal=True, scalars=len(a['values']),
                payload_sha256=hashlib.sha256(a['payload']).hexdigest())


def numerical_difference(a, b):
    if (a['blocks'], a['stride'], a['cycle'], a['time']) != (
            b['blocks'], b['stride'], b['cycle'], b['time']):
        raise AssertionError('different endpoint or checkpoint layout')
    maximum = 0.0
    worst = 0
    # The first 31 scalars are evolved fields; the remaining 8 are cumulative
    # floor diagnostics whose intermediate-history values need not agree.
    for i, (x, y) in enumerate(zip(a['values'], b['values'])):
        if i % 39 < 31:
            error = abs(x-y)/max(1.0, abs(x), abs(y))
            if error > maximum:
                maximum, worst = error, i
    return dict(max_scaled_field_difference=maximum, worst_scalar=worst)


def write_replay(initial_log, replay_file, interval, owners):
    """Copy actual graph identity, replace placement at one common endpoint."""
    with initial_log.open('rb') as stream:
        size = int(stream.readline())
        first = stream.read(size).decode()
    header = first.splitlines()[0].split()
    if len(header) != 5 or header[0] != 'multirate-partition-v1':
        raise ValueError('invalid actual partition log')
    if int(header[4]) != len(owners):
        raise ValueError('partition log block count changed')
    header[1] = str(interval)
    event = ' '.join(header)+'\n'+''.join(f'{gid} {rank}\n' for gid, rank in enumerate(owners))
    with replay_file.open('wb') as stream:
        for record in (first, event):
            data = record.encode()
            stream.write(str(len(data)).encode()+b'\n'+data)


def run(binary, output, name, launcher, ranks, options, failure=None):
    directory = output/name
    directory.mkdir()
    command = [part.format(ranks=ranks) for part in shlex.split(launcher)]
    command += [str(binary), *map(str, options)]
    started = time.monotonic()
    with (directory/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1200,
                                env=dict(os.environ, OMP_NUM_THREADS='1'))
    text = (directory/'run.log').read_text()
    receipt = dict(command=command, wall_seconds=time.monotonic()-started,
                   returncode=result.returncode, log=str(directory/'run.log'))
    (directory/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if failure:
        if result.returncode == 0 or not re.search(failure, text, re.IGNORECASE):
            raise AssertionError(f'{name}: expected rejection /{failure}/; {text[-5000:]}')
        return dict(**receipt, rejected=True), text
    if result.returncode:
        raise RuntimeError(f'{name}: exit {result.returncode}: {text[-6000:]}')
    paths = sorted((directory/'rst').glob('**/*.rst'))
    if not paths:
        raise AssertionError(f'{name}: no actual checkpoint written')
    data = checkpoint(paths[-1])
    return data, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--cases', nargs='+', choices=('smoke', 'invalid', 'restart', 'ownership', 'replay'),
                        default=('smoke', 'invalid', 'restart', 'ownership', 'replay'))
    args = parser.parse_args()
    if args.ranks < 1 or (args.ranks > 1 and not args.launcher):
        parser.error('positive ranks and an MPI launcher for ranks > 1 required')
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=args.output.resolve()))
    immutable_binary = output/'athena'
    shutil.copy2(binary,immutable_binary)
    binary = immutable_binary
    fixture = output/'small.athinput'
    fixture.write_text((Path(__file__).parent/'small.athinput').read_text())
    provenance = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                      ranks=args.ranks, scope='native execution/restart smoke gates; not scientific or GPU qualification')
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    results = {}
    base = ['-i', str(fixture)]
    mixed = None
    if any(case in args.cases for case in ('smoke', 'restart', 'ownership', 'replay')):
        mixed, text = run(binary, output, 'mixed', args.launcher, args.ranks,
                          base+['multirate/partition_log='+str(output/'initial.partition')])
        if mixed['cycle'] != 2 or not re.search(r'MULTIRATE_INTERVAL\b', text):
            raise AssertionError('native multirate interval receipt missing')
        evaluations = [int(n) for n in re.findall(r'MULTIRATE_INTERVAL[^\n]*stage_evaluations=(\d+)',text)]
        if evaluations != [124,124]:
            raise AssertionError(f'expected 4*(19+2*6) RHS evaluations per interval, got {evaluations}')
        results['mixed'] = dict(blocks=mixed['blocks'], cycle=mixed['cycle'], time=mixed['time'])
    if 'smoke' in args.cases:
        grouped, _ = run(binary, output, 'complete_local_batches', args.launcher, args.ranks,
                         base+['multirate/complete_local_batches=true'])
        results['complete_local_batches'] = identical(mixed, grouped)
        uniform, text = run(binary, output, 'uniform', args.launcher, args.ranks,
                            base+['multirate/spherical_levels=0,0,0,0'])
        difference = numerical_difference(mixed, uniform)
        if difference['max_scaled_field_difference'] > 1e-7:
            raise AssertionError(f'small-step uniform/mixed disagreement: {difference}')
        results['uniform_vs_mixed_smoke'] = difference
    if 'invalid' in args.cases:
        results['reject_explicit_gap'], _ = run(binary, output, 'invalid_gap', args.launcher,
            args.ranks, base+['multirate/spherical_levels=0,0,0,3'],
            failure=r'2:1|adjacent|coupling.*level|level.*coupling')
        results['reject_stability_bound'], _ = run(binary, output, 'invalid_stability', args.launcher,
            args.ranks, base+['multirate/max_characteristic_cfl=1e-30'],
            failure=r'CFL|stability|characteristic')
    if 'restart' in args.cases:
        first, _ = run(binary, output, 'first_interval', args.launcher, args.ranks,
                       base+['time/nlim=1'])
        resumed, _ = run(binary, output, 'resumed', args.launcher, args.ranks,
                         ['-r', first['path'], 'time/nlim=2', 'output1/last_time=-1'])
        results['restart_same_schedule'] = identical(mixed, resumed)
    if 'ownership' in args.cases and args.ranks > 1:
        owners = ','.join(str(i % args.ranks) for i in range(mixed['blocks']))
        alternate, _ = run(binary, output, 'noncontiguous', args.launcher, args.ranks,
                           base+['multirate/block_owners='+owners])
        difference = numerical_difference(mixed, alternate)
        if difference['max_scaled_field_difference'] > 1e-11:
            raise AssertionError(f'ownership changes numerical trajectory: {difference}')
        first, _ = run(binary, output, 'noncontiguous_first', args.launcher, args.ranks,
                       base+['multirate/block_owners='+owners, 'time/nlim=1'])
        resumed, _ = run(binary, output, 'noncontiguous_resumed', args.launcher, args.ranks,
                         ['-r', first['path'], 'time/nlim=2', 'output1/last_time=-1'])
        results['noncontiguous_same_schedule_restart'] = identical(alternate, resumed)
        results['different_decomposition'] = difference
    if 'replay' in args.cases and args.ranks > 1:
        replay = output/'replay.partition'
        owners = [i % args.ranks for i in range(mixed['blocks'])]
        write_replay(output/'initial.partition', replay, 2, owners)
        options = base+['time/nlim=3', 'multirate/replay_file='+str(replay)]
        continuous, text = run(binary, output, 'replay_continuous', args.launcher, args.ranks, options)
        saved_owners = re.search(r'^\s*block_owners\s*=\s*([0-9,]+)', continuous['parameters'], re.MULTILINE)
        if not saved_owners or saved_owners.group(1) != ','.join(map(str, owners)):
            raise AssertionError('replayed migration did not persist its actual stable-GID ownership')
        first, _ = run(binary, output, 'replay_first', args.launcher, args.ranks,
                       options+['time/nlim=1'])
        resumed, _ = run(binary, output, 'replay_resumed', args.launcher, args.ranks,
                         ['-r', first['path'], 'time/nlim=3', 'output1/last_time=-1'])
        results['replayed_redistribution_restart'] = identical(continuous, resumed)
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), results=results), indent=2))


if __name__ == '__main__':
    main()
