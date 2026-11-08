#!/usr/bin/env python3
"""Audit native observer coverage and exact clean/poison or cross-rank parity.

Reads explicit per-point dumps only; does not run an evolution or submit jobs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

LABELS = ['geometry', 'accepted0', 'accepted1', 'accepted8'] + [
    'stage'+str(s)+'-'+phase for s in range(1, 5) for phase in ('input', 'output')
] + ['first-rhs']
COUNTS = {'bulk': 229392, 'scri': 864}
MAGIC = b'NATIVE_TRACE_V1\0'
assert len(MAGIC) == 16

def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata(folder):
    report = json.loads((folder/'results.json').read_text())
    require(report['schema'] == 'native-stage-trace-v1' and report['passed'] is True,
            'missing completed native trace')
    require(report['mode'] in ('clean', 'poison'), 'unknown observer mode')
    require(report['ranks'] > 0 and report['steps'] == 8, 'invalid rank or step count')
    require(report['bulk_points'] == 229392 and report['scri_points'] == 864,
            'changed owned workload')
    require(report['unowned_cells_per_stage'] > 0 and
            0 <= report['zero_surface_ranks'] < report['ranks'], 'invalid local ownership counts')
    dt = report['dt']
    require(dt == 0.00046944203735617445 and len(report['stages']) == 32,
            'changed dt or missing stage records')
    time = 0.
    for cycle in range(8):
        for stage in range(1, 5):
            row = report['stages'][4*cycle+stage-1]
            require((row['cycle'], row['stage']) == (cycle, stage), 'stage order differs')
            require(row['rhs_time'] == (time if stage == 1 else time+(1 if stage == 4 else .5)*dt)
                    and row['state_time'] == time+(.5 if stage < 3 else 1)*dt,
                    'stage clocks differ')
            require(all(math.isfinite(row[key]) for key in
                        ('compatibility', 'clock', 'minimum_chi', 'maximum_rhs')), 'nonfinite stage metric')
            require(0 <= row['compatibility'] <= 1e-7 and row['clock'] >= 0
                    and row['minimum_chi'] >= 1e-10 and row['maximum_rhs'] >= 0,
                    'failed stage admissibility')
        time += dt
    require(report['time'] == time, 'accepted time differs')
    expected = {f'{label}-{region}-rank{rank}.bin' for label in LABELS
                for region in COUNTS for rank in range(report['ranks'])}
    require({p.name for p in folder.glob('*.bin')} == expected,
            'missing or unexpected observer dump files')
    return report


def load(folder, label, region, ranks):
    values = 3 if label == 'geometry' else 31 if label == 'first-rhs' else 39
    stride = 24+8*values
    result = {}
    hashes = {}
    for rank in range(ranks):
        path = folder/f'{label}-{region}-rank{rank}.bin'
        raw = path.read_bytes()
        require(len(raw) >= 40 and raw[:16] == MAGIC, 'invalid trace magic/header: '+str(path))
        count, keys, nv, kind, version = struct.unpack_from('<QIIII', raw, 16)
        require((keys, nv, kind, version) == (6, values, int(region == 'scri'), 1),
                'changed trace schema: '+str(path))
        require(len(raw) == 40+count*stride, 'truncated or extra trace bytes: '+str(path))
        for offset in range(40, len(raw), stride):
            key = struct.unpack_from('<6i', raw, offset)
            require(key[0] not in result, 'duplicate canonical point')
            require(0 <= key[1] < 106 and -1 <= key[5] < 6, 'invalid native leaf/chart')
            if region == 'scri':
                require(key[4] == 12 and 0 <= key[2] < 12 and 0 <= key[3] < 12
                        and key[0] == (key[5]*12+key[2])*12+key[3], 'invalid canonical scri point')
            value = raw[offset+24:offset+stride]
            require(all(math.isfinite(v) for v in struct.unpack('<'+str(values)+'d', value)),
                    'nonfinite owned dump value')
            result[key[0]] = (key, value)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
    require(set(result) == set(range(COUNTS[region])), 'incomplete canonical coverage')
    digest = hashlib.sha256()
    for index in range(COUNTS[region]):
        key, value = result[index]
        digest.update(struct.pack('<6i', *key));digest.update(value)
    return result, digest.hexdigest(), hashes


def compare(a, b, values):
    differences = [0]*values
    maxima = [0.]*values
    key_differences = 0
    first = None
    for index, (key, data) in a.items():
        other_key, reference = b[index]
        key_differences += key != other_key
        if data == reference:
            continue
        av, bv = struct.unpack('<'+str(values)+'d', data), struct.unpack('<'+str(values)+'d', reference)
        for field in range(values):
            if data[8*field:8*(field+1)] != reference[8*field:8*(field+1)]:
                differences[field] += 1
                maxima[field] = max(maxima[field], abs(av[field]-bv[field]))
                if first is None:
                    first = dict(canonical_index=index, key=key, field=field,
                                 actual=av[field], reference=bv[field])
    return dict(exact=not key_differences and not any(differences), key_differences=key_differences,
                field_bit_differences=differences, field_max_absolute=maxima, first_difference=first)


def audit(folder, other=None):
    current = metadata(folder)
    reference = metadata(other) if other else None
    if reference:
        require(current['backend'] == reference['backend'], 'exact trace comparison requires same backend')
    summary = dict(schema='native-stage-trace-audit-v1', output=str(folder),
                   mode=current['mode'], ranks=current['ranks'], backend=current['backend'],
                   passed=True, labels={}, dump_files_sha256={},
                   result_sha256=sha(folder/'results.json'), transition_checks={})
    if reference:
        summary['reference'] = str(other)
        summary['reference_result_sha256'] = sha(other/'results.json')
        summary['stage_records_exact'] = current['stages'] == reference['stages']
        summary['passed'] &= summary['stage_records_exact']
    for region in COUNTS:
        expected_keys = None
        fingerprints = {}
        for label in LABELS:
            data, digest, hashes = load(folder, label, region, current['ranks'])
            keys = tuple(data[i][0] for i in range(COUNTS[region]))
            if expected_keys is None:
                expected_keys = keys
            require(keys == expected_keys, 'point identities changed between observer labels')
            item = dict(points=len(data), canonical_sha256=digest)
            fingerprints[label] = digest
            summary['dump_files_sha256'].update(hashes)
            if reference:
                other_data, other_digest, other_hashes = load(other, label, region, reference['ranks'])
                values = 3 if label == 'geometry' else 31 if label == 'first-rhs' else 39
                item['comparison'] = compare(data, other_data, values)
                item['reference_canonical_sha256'] = other_digest
                summary['passed'] &= item['comparison']['exact']
                del other_data
            summary['labels'][label+'-'+region] = item
            del data, keys
        pairs = [('accepted0', 'stage1-input'), ('stage4-output', 'accepted1')]
        pairs += [(f'stage{s}-output', f'stage{s+1}-input') for s in range(1, 4)]
        for left, right in pairs:
            exact = fingerprints[left] == fingerprints[right]
            summary['transition_checks'][region+':'+left+'='+right] = exact
            summary['passed'] &= exact
    (folder/'trace-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve(), args.compare.resolve() if args.compare else None)
    print(json.dumps({k: result[k] for k in ('passed', 'ranks', 'mode', 'backend', 'transition_checks')}, indent=2))
    if args.compare:
        print('stage_records_exact', result['stage_records_exact'])
        for label, item in result['labels'].items():
            print(label, 'exact', item['comparison']['exact'])
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
