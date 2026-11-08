#!/usr/bin/env python3
"""Audit cross-platform N12 checkpoints without admitting metadata differences.

The strict decoder's numerical-identity and coordinate gates stay unchanged.
This separate diagnostic reports their failures, then measures fields/records
under the original short-step tolerances. It cannot turn a strict failure into
an admission pass.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from n12_checkpoint import (NBULK, NSURFACE, NF, NR, DT, compare_fields,
    decode, expected_time, load_mapping, parse_native, require, sha256,
    validate_header, chi_summary)


def identity_difference(expected, actual):
    def entries(text):
        result = {}
        for term in text.split(';'):
            if not term:
                continue
            key, value = term.split('=', 1) if '=' in term else ('schema', term)
            require(key not in result, 'duplicate numerical identity key')
            result[key] = value
        return result
    a, b = entries(expected), entries(actual)
    result = []
    for key in sorted(a.keys() | b.keys()):
        if a.get(key) == b.get(key):
            continue
        row = dict(key=key, reference=a.get(key), candidate=b.get(key))
        try:
            x, y = float.fromhex(a[key]), float.fromhex(b[key])
            row.update(reference_value=x, candidate_value=y, absolute_difference=abs(x-y))
            if x >= 0 and y >= 0:
                row['ulp_distance'] = abs(struct.unpack('<Q', struct.pack('<d', x))[0] -
                                          struct.unpack('<Q', struct.pack('<d', y))[0])
        except (ValueError, KeyError):
            pass
        result.append(row)
    return result


def coordinate_audit(paths, mapping):
    """Keep complete key validation, report exact-coordinate failures separately."""
    seen = np.zeros(NBULK, dtype=bool)
    xyz = np.empty((NBULK, 3)); masks = np.empty((NBULK, 2), dtype=np.int32)
    weight = np.empty(NBULK)
    dtype = np.dtype([('keys', '<i4', (8,)), ('values', '<f8', (26,))])
    for path in paths:
        with path.open('rb') as stream:
            require(stream.read(16) == b'NATIVE_DIAG_V1\0\0', 'unknown coordinate dump')
            n, nk, nv = struct.unpack('<QII', stream.read(16))
            require((nk, nv) == (8, 26), 'wrong coordinate dump shape')
            rows = np.fromfile(stream, dtype=dtype)
        require(len(rows) == n, 'coordinate dump length mismatch')
        ids = rows['keys'][:, 0]
        require(((ids >= 0) & (ids < NBULK)).all() and np.unique(ids).size == n,
                'coordinate ordinal out of range or duplicated')
        require(not seen[ids].any(), 'coordinate ordinal repeated across ranks')
        keys = mapping['bulk_keys'][ids]
        require(np.array_equal(rows['keys'][:, 1], keys[:, 0]) and
                np.array_equal(rows['keys'][:, 2:5], keys[:, [4, 3, 2]]),
                'coordinate native/legacy key mismatch')
        require(np.isfinite(rows['values']).all(), 'nonfinite coordinate audit input')
        xyz[ids] = rows['values'][:, :3]; masks[ids] = rows['keys'][:, 5:7]
        weight[ids] = rows['values'][:, 22]; seen[ids] = True
    require(seen.all(), 'coordinate audit omits evolved points')
    reference = mapping['bulk_xyz']; different = xyz.view('<u8') != reference.view('<u8')
    delta = abs(xyz-reference); worst = np.unravel_index(delta.argmax(), delta.shape)
    # Scalar expression has two additions, matching the no-contraction CPU map
    # diagnostic. Actual GPU and Aurora-host masks remain those in the dump.
    reference_radius = np.sqrt(reference[:, 0]*reference[:, 0] +
                               reference[:, 1]*reference[:, 1] + reference[:, 2]*reference[:, 2])
    reference_mask = reference_radius > .94
    comparisons = {}
    for column, name in [(0, 'aurora_device'), (1, 'aurora_host')]:
        mismatch = masks[:, column].astype(bool) != reference_mask
        comparisons[name] = dict(different_points=int(mismatch.sum()),
            positive_weight=int(np.count_nonzero(mismatch & (weight > 0))),
            zero_weight=int(np.count_nonzero(mismatch & (weight == 0))),
            candidate_only=int(np.count_nonzero(mismatch & masks[:, column].astype(bool))),
            cpu_map_only=int(np.count_nonzero(mismatch & reference_mask)))
    return dict(complete=True, bulk_points=NBULK, coordinate_scalars=3*NBULK,
        exact_coordinate_gate_passed=not different.any(),
        different_points=int(different.any(axis=1).sum()),
        different_components=int(different.sum()), max_absolute=float(delta[worst]),
        worst=dict(legacy_ordinal=int(worst[0]), component=int(worst[1]),
                   actual=xyz[worst[0]].tolist(), expected=reference[worst[0]].tolist(),
                   native_key=mapping['bulk_keys'][worst[0]].tolist()),
        fixed_cut_mask_comparison_to_cpu_map=comparisons,
        geometry_generation='native host caches mirrored after device upload, not GPU map generation',
        scope='saved3551 replay uses same frozen atlas/build; not a time0/eight runtime dump',
        source_sha256={str(p): sha256(p) for p in paths})


def record_comparison(actual, expected, keys):
    require(actual.shape == expected.shape and actual.shape[1] == NR, 'record shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'nonfinite record')
    # At steps0/8 all references have zero events, so counts/corrections/first
    # event sentinel must agree exactly. Only two raw minima use the chi gate.
    require((expected[:, 2:7] == 0).all() and (expected[:, 7] == 1e300).all(),
            'this short-run record gate requires a zero-floor reference')
    names = ['minimum_raw_candidate', 'minimum_raw_acceptance_candidate',
             'floor_events', 'nonpositive_candidates', 'nonfinite_candidates',
             'max_correction', 'sum_corrections', 'first_floor_time']
    result = dict(comparisons=int(actual.size), failed_values=0, fields=[])
    for field, name in enumerate(names):
        delta = abs(actual[:, field]-expected[:, field])
        tolerance = 1e-14+5e-10*abs(expected[:, field]) if field < 2 else np.zeros(len(actual))
        failures = int(np.count_nonzero(delta > tolerance)); worst = int(delta.argmax())
        result['failed_values'] += failures
        result['fields'].append(dict(name=name, comparisons=len(actual), failed_values=failures,
            bitwise_equal=bool(np.array_equal(actual[:, field].view('<u8'), expected[:, field].view('<u8'))),
            max_absolute=float(delta[worst]), rms=float(np.sqrt(np.mean(delta*delta))),
            gate='1e-14+5e-10*abs(reference)' if field < 2 else 'exact zero-floor record',
            worst=dict(legacy_ordinal=worst, native_key=keys[worst].tolist(),
                       actual=float(actual[worst, field]), expected=float(expected[worst, field]))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--eight', type=Path, required=True)
    parser.add_argument('--eight-reference', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--coordinate-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    manifest, mapping = load_mapping(args.mapping)
    template = Path(manifest['template_checkpoint'])
    require(sha256(template) == manifest['template_checkpoint_sha256'], 'changed initial record fixture')
    initial_info = parse_native(template); validate_header(initial_info, manifest, 0)
    initial = decode(initial_info, mapping)
    oracle_path = args.trajectory/'results.json'; oracle = json.loads(oracle_path.read_text())
    require(oracle['passed'] and oracle['chi_records']['8']['failed_values'] == 0,
            'CPU trajectory did not validate complete ChiRecords against legacy')
    report = dict(passed=False, strict_admission_passed=False,
        scope='separate cross-platform diagnostic; strict decoder gates unchanged',
        mapping_sha256=manifest['mapping_sha256'],
        source_sha256={str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name('n12_checkpoint.py').resolve()):
                sha256(Path(__file__).with_name('n12_checkpoint.py'))},
        cpu_trajectory_result_sha256=sha256(oracle_path), steps={})
    for step, candidate, reference in [(0, args.initial, args.mapping/'initial-reference'),
                                      (8, args.eight, args.eight_reference)]:
        info = parse_native(candidate)
        differences = identity_difference(manifest['physics_identity'], info['identity'].decode())
        # This copy is only for checking all OTHER header conditions. The
        # identity failure is retained above/below and prevents strict admission.
        validate_header(info, dict(manifest, physics_identity=info['identity'].decode()), step)
        data = decode(info, mapping)
        ref_meta = json.loads((reference/'reference-manifest.json').read_text())
        require(ref_meta['mapping_sha256'] == manifest['mapping_sha256'], 'reference mapping differs')
        clock = json.loads((reference/'time.json').read_text())
        require(clock['steps'] == step and clock['t'] == expected_time(step), 'reference clock differs')
        entry = dict(candidate=str(candidate.resolve()), candidate_sha256=sha256(candidate),
            writer_mpi_ranks=info['metadata'][4], time=info['time'], dt=info['dt'],
            exact_physics_identity_passed=not differences, physics_identity_differences=differences,
            all_other_header_checks_passed=True, complete_finite_padding_checks_passed=True,
            fields={}, chi_records={}, chi_summary=chi_summary(data, mapping), exports={})
        folder = args.output/f'step{step}'; folder.mkdir()
        for region, filename, count in [('bulk', 'fields.bin', NBULK),
                                         ('surface', 'boundary.bin', NSURFACE)]:
            require(sha256(reference/filename) == ref_meta['sha256'][filename], 'changed field reference')
            expected = np.fromfile(reference/filename, dtype='<f8').reshape(count, NF)
            entry['fields'][region] = compare_fields(data[region][:, :NF], expected,
                mapping[region+'_keys'], mapping[region+'_xyz'], mapping[region+'_weight'], 'short')
            if step == 0:
                records = initial[region][:, NF:]
            else:
                name = 'native-step8-'+('chi' if region == 'bulk' else 'surface-chi')+'.bin'
                require(sha256(args.trajectory/name) == oracle['snapshot_sha256'][name], 'changed chi fixture')
                records = np.fromfile(args.trajectory/name, dtype='<f8').reshape(count, NR)
                records = records[mapping[region+'_keys'][:, 5]]
            entry['chi_records'][region] = record_comparison(data[region][:, NF:], records,
                                                            mapping[region+'_keys'])
            recordfile = 'chi-records.bin' if region == 'bulk' else 'surface-chi-records.bin'
            data[region][:, :NF].tofile(folder/filename); data[region][:, NF:].tofile(folder/recordfile)
            entry['exports'].update({n: sha256(folder/n) for n in (filename, recordfile)})
        entry['short_field_gate_passed'] = all(v['failed_values'] == 0 for v in entry['fields'].values())
        entry['chi_record_gate_passed'] = all(v['failed_values'] == 0 for v in entry['chi_records'].values())
        entry['chi_record_reference_scope'] = ('complete CPU initial reference records' if step == 0 else
            'complete CPU native records previously bytewise validated against independent legacy Step3D')
        report['steps'][str(step)] = entry
    paths = sorted(args.coordinate_directory.glob('points-rank*.bin'))
    require(paths, 'no actual cached-coordinate dumps')
    report['coordinates'] = coordinate_audit(paths, mapping)
    report['strict_admission_passed'] = (all(v['exact_physics_identity_passed'] for v in report['steps'].values())
        and report['coordinates']['exact_coordinate_gate_passed'])
    report['short_field_gates_passed'] = all(v['short_field_gate_passed'] for v in report['steps'].values())
    report['chi_record_gates_passed'] = all(v['chi_record_gate_passed'] for v in report['steps'].values())
    report['passed'] = report['strict_admission_passed'] and report['short_field_gates_passed'] and report['chi_record_gates_passed']
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'strict_admission_passed',
        'short_field_gates_passed', 'chi_record_gates_passed')}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
