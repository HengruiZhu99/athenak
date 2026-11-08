#!/usr/bin/env python3
"""Structural/cadence admission for the bounded native N12 common-time segments.

This does not replace the complete-field, chi and diagnostic reference gates.
It never evolves, projects or alters a checkpoint. Only stdlib is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

DT = 0.00046944203735617445
PERIOD = 1.0 / 6.0
CYCLES = [0, 356, 711, 1066, 1421, 1776, 2131, 2486, 2841, 3196, 3551]
HEADER = struct.calcsize('<ii9d38i2di')
INITIAL_SHA = 'cfa18d09e96a52f1cf22fb724c907a097a0ca42925c623cdbbba22af2ebeae77'
EXE_SHA = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parameters(text):
    result, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line:
            key, value = line.split('=', 1)
            result[block + '/' + key.strip()] = value.strip()
    return result


def time_at(cycle):
    value = 0.0
    for _ in range(cycle):
        value += DT
    return value


def clock_at(segment):
    value = 0.0
    for _ in range(segment):
        value += PERIOD
    return value


def checkpoint(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        text = bytearray()
        while not text.endswith(b'<par_end>\n'):
            line = stream.readline(262144)
            require(line and len(text) + len(line) < 8 * 1024 * 1024, 'invalid parameter header')
            text.extend(line)
        def take(n):
            require(0 <= n <= 1024 * 1024, 'invalid checkpoint metadata size')
            data = stream.read(n)
            require(len(data) == n, 'truncated checkpoint metadata')
            return data
        fixed = take(HEADER)
        require(struct.unpack_from('<ii', fixed) == (106, 4), 'not fixed N12 topology')
        time, dt, cycle = struct.unpack_from('<ddi', fixed, HEADER - 20)
        n, = struct.unpack('<Q', take(8))
        atlas = take(n)
        topology = take(106 * 20)
        costs = struct.unpack('<106f', take(106 * 4))
        require(all(x == 1.0 for x in costs), 'block costs changed')
        metadata = struct.unpack('<8Q', take(64))
        require(metadata[:2] == (0x4859504552525354, 1), 'unknown hyperboloidal schema')
        identity = take(metadata[2])
        stride, = struct.unpack('<Q', take(8))
        require(metadata[3] == (18**3 + 12**2) * 39 and stride == metadata[3] * 8,
                'N12 payload layout differs')
        require(metadata[4:] == (24, 0, 4, 0), 'expected shared checkpoint written by24ranks')
        offset = stream.tell()
    require(path.stat().st_size == offset + 106 * stride, 'incomplete checkpoint payload')
    return dict(path=str(path), sha256=sha(path), size=path.stat().st_size,
                fixed=fixed, atlas=atlas, topology=topology, identity=identity,
                parameters=parameters(text.decode()), time=time, dt=dt, cycle=cycle)


def validate(info, reference, segment, cycle, tlim=None):
    require(info['cycle'] == cycle and info['dt'] == DT and info['time'] == time_at(cycle),
            'checkpoint time/cycle/dt changed')
    # Nine unused native coarse-index integers are uninitialized, as documented
    # in n12_checkpoint.py. All meaningful mesh metadata remains exact.
    require(info['fixed'][:120] == reference['fixed'][:120] and
            info['fixed'][156:-20] == reference['fixed'][156:-20], 'mesh header changed')
    for key in ('atlas', 'topology', 'identity'):
        require(info[key] == reference[key], key + ' differs from admitted native initial state')
    p, rp = info['parameters'], reference['parameters']
    mutable = {'time/nlim', 'time/tlim', 'output1/last_time', 'output2/last_time',
               'output2/file_number', 'fastflow/last_a0_0', 'fastflow/ah_found_a0_0',
               'fastflow/time_first_found_0'}
    require(set(p) == set(rp), 'checkpoint parameter set changed')
    for key in p:
        require(key in mutable or p[key] == rp[key], 'parameter changed: ' + key)
    require(float(p['time/dt']) == DT and int(p['time/nlim']) == CYCLES[segment],
            'segment limits changed')
    require(float(p['time/tlim']) == (clock_at(segment) if tlim is None else tlim),
            'nominal segment tlim changed')
    output_segment = 0 if cycle == 0 else segment
    for block in ('output1', 'output2'):
        require(float(p[block + '/last_time']) == clock_at(output_segment),
                'extra/missing output clock advance: ' + block)
        require(float(p[block + '/dt']) == PERIOD, 'diagnostic cadence changed')
    require(int(p['output2/file_number']) == output_segment + 1,
            'restart output counter changed')
    require(float(p['fastflow/last_a0_0']) > 0 and math.isfinite(float(p['fastflow/last_a0_0'])) and
            p['fastflow/ah_found_a0_0'] == '1', 'missing successful retained horizon seed')
    require(float(p['fastflow/time_first_found_0']) == 0, 'horizon first-found history changed')
    return {key: info[key] for key in ('path', 'sha256', 'size', 'time', 'dt', 'cycle')} | {
        'output_clock': clock_at(output_segment), 'next_restart_number': output_segment + 1,
        'last_a0': float(p['fastflow/last_a0_0']), 'ah_found': True,
        'atlas_sha256': hashlib.sha256(info['atlas']).hexdigest(),
        'identity_sha256': hashlib.sha256(info['identity']).hexdigest()}


def audit(case, reference_path, segment, predecessor=None):
    case = Path(case).resolve()
    reference = checkpoint(reference_path)
    require(reference['sha256'] == INITIAL_SHA, 'unadmitted initial reference checkpoint')
    start, end = CYCLES[segment-1:segment+1]
    expected_files = [f'native_n12_5M.{segment:05d}.rst']
    if segment == 1:
        expected_files.insert(0, 'native_n12_5M.00000.rst')
    files = sorted((case / 'rst').glob('*.rst'))
    require([p.name for p in files] == expected_files, 'missing/extra segment restart outputs')
    reports = []
    if segment == 1:
        initial = checkpoint(files[0])
        reports.append(validate(initial, reference, segment, 0))
    else:
        require(predecessor is not None, 'restart segment lacks predecessor')
        previous = checkpoint(predecessor)
        reports.append(validate(previous, reference, segment - 1, start))
    endpoint = checkpoint(files[-1])
    reports.append(validate(endpoint, reference, segment, end))
    log = (case / 'run.log').read_text()
    require('Terminating on time limit' in log or 'Terminating on cycle limit' in log,
            'missing normal limit termination')
    require(re.search(r'^time=\S+ cycle=' + str(end) + r'$', log, re.M), 'wrong final log cycle')
    rows = re.findall(r'^elapsed=\S+ cycle=(\d+) time=(\S+) dt=(\S+)$', log, re.M)
    require([int(c) for c, _, _ in rows] == list(range(start, end + 1)), 'cycle log coverage differs')
    require(all(float(dt) == DT for _, _, dt in rows), 'logged dt changed')
    require(all(float(t) == time_at(int(c)) for c, t, _ in rows), 'logged cycle time changed')
    hst = case / 'native_n12_5M.z4c_hyperboloidal.hst'
    history = [[float(x) for x in line.split()] for line in hst.read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    required_cycles = [0, end] if segment == 1 else [end]
    require(len(history) == len(required_cycles), 'extra/missing history diagnostics')
    for row, cycle in zip(history, required_cycles):
        require(len(row) == 12 and all(math.isfinite(v) for v in row), 'invalid history row')
        require(row[0] == time_at(cycle) and row[1] == DT, 'off-cadence history row')
    summary = case / 'native_n12_5M.horizon_summary_0.txt'
    horizon = [line for line in summary.read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    require(len(horizon) == len(required_cycles), 'extra/missing horizon searches in summary')
    require([int(line.split()[0]) for line in horizon] == required_cycles,
            'off-cadence horizon search in summary')
    require(all(math.isfinite(float(x)) for line in horizon for x in line.split()),
            'nonfinite horizon summary')
    return dict(schema='native-n12-common-segment-v1', structural_cadence_passed=True,
                segment=segment, start_cycle=start, end_cycle=end, checkpoints=reports,
                history_sha256=sha(hst), horizon_summary_sha256=sha(summary),
                complete_field_chi_and_reference_diagnostics_admission='pending separate comparison',
                continuation_admitted=False)


def verify_runtime(root):
    root = Path(root)
    build = root / 'build-sycl'
    manifest = json.loads((build / 'runtime-object-sha256.json').read_text())
    require(manifest['executable_sha256'] == EXE_SHA, 'unadmitted native executable')
    require(sha(manifest['executable']) == EXE_SHA, 'native executable changed')
    for rel, want in manifest['object_sha256'].items():
        require(sha(build / rel) == want, 'native object changed: ' + rel)
    for rel, want in json.loads((build / 'source-sha256.json').read_text()).items():
        require(sha(root / 'source' / rel) == want, 'native source changed: ' + rel)
    print('Frozen native executable, source and objects verified')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--case', type=Path, required=True)
    a.add_argument('--reference', type=Path, required=True)
    a.add_argument('--segment', type=int, choices=range(1, 11), required=True)
    a.add_argument('--predecessor', type=Path)
    a.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('verify-runtime');v.add_argument('root', type=Path)
    c = sub.add_parser('limits');c.add_argument('segment', type=int, choices=range(1, 11))
    args = p.parse_args()
    if args.command == 'limits':
        print(CYCLES[args.segment], repr(clock_at(args.segment)))
    elif args.command == 'verify-runtime':
        verify_runtime(args.root)
    else:
        result = audit(args.case, args.reference, args.segment, args.predecessor)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
