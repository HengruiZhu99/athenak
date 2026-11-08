#!/usr/bin/env python3
"""Admit the pinned GPU accessor, then compare its actual rates/state to native.

Reads completed artifacts only. CPU mapping supplies integer identities and
diagnostic weights, never exact GPU coordinates or a reference evolution.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from n12_checkpoint import (DT, NF, NBULK, NSURFACE, compare_fields, expected_time,
                            load_mapping, parse_parameters, require, sha256)
from n12_checkpoint_cross_platform import record_comparison
from n12_checkpoint_gpu_control import ORIGINAL_CONFIG, ORIGINAL_EXE, old_checkpoint
from native_stage_trace_compare import LABELS as NATIVE_LABELS, metadata as native_metadata

MAGIC = {'legacy': b'LEGACY_TRACE_V1\0', 'native': b'NATIVE_TRACE_V1\0'}
COUNTS = {'bulk': NBULK, 'scri': NSURFACE}
REGIONS = {'bulk': 'bulk', 'scri': 'surface'}
assert all(len(value) == 16 for value in MAGIC.values())


def legacy_metadata(folder):
    report = json.loads((folder/'results.json').read_text())
    require(report['schema'] == 'legacy-stage-trace-v1' and report['passed'] is True,
            'legacy accessor did not complete')
    require(report['backend'] == 'SYCL' and report['ranks'] in (24, 96),
            'scientific admission requires actual Aurora GPU runs')
    require(report['steps'] in (1, 8) and report['dt'] == DT and
            report['time'] == expected_time(report['steps']), 'legacy trace clocks differ')
    require((report['bulk_points'], report['scri_points']) == (NBULK, NSURFACE),
            'legacy trace workload differs')
    labels = ['geometry', 'accepted0', 'accepted1', 'first-rhs', 'final-snapshot']
    if report['steps'] == 8:
        labels.append('accepted8')
    expected = {f'{label}-{region}-rank{rank}.bin' for label in labels
                for region in COUNTS for rank in range(report['ranks'])}
    require({p.name for p in folder.glob('*.bin')} == expected,
            'missing or extra legacy trace files')
    return report, labels


def load_trace(folder, flavor, label, region, ranks, mapping, expected_keys=None):
    """Return actual keys/values in legacy ordinal order with full coverage."""
    count = COUNTS[region]
    width = 3 if label == 'geometry' else 31 if label == 'first-rhs' else 39
    dtype = np.dtype([('keys', '<i4', (6,)), ('values', '<f8', (width,))])
    mapped = mapping[REGIONS[region]+'_keys']
    inverse = np.argsort(mapped[:, 5])
    values = np.empty((count, width), dtype='<f8')
    keys = np.empty((count, 6), dtype='<i4')
    seen = np.zeros(count, dtype=bool)
    hashes = {}
    for rank in range(ranks):
        path = folder/f'{label}-{region}-rank{rank}.bin'
        with path.open('rb') as stream:
            header = stream.read(40)
            require(len(header) == 40 and header[:16] == MAGIC[flavor],
                    'invalid trace header: '+str(path))
            local, nk, nv, kind, version = struct.unpack_from('<QIIII', header, 16)
            require((nk, nv, kind, version) == (6, width, int(region == 'scri'), 1),
                    'trace schema differs: '+str(path))
            require(path.stat().st_size == 40+local*dtype.itemsize,
                    'truncated or extra trace bytes: '+str(path))
            rows = np.fromfile(stream, dtype=dtype)
        ordinal = rows['keys'][:, 0]
        require(len(rows) == local and ((ordinal >= 0) & (ordinal < count)).all() and
                np.unique(ordinal).size == len(ordinal), 'invalid/duplicate trace ordinal')
        ids = inverse[ordinal] if flavor == 'native' else ordinal
        require(not seen[ids].any(), 'duplicate trace point across ranks')
        require(np.isfinite(rows['values']).all(), 'nonfinite owned trace value')
        require(np.array_equal(rows['keys'][:, 2:5], mapped[ids][:, [4, 3, 2]]),
                'trace k/j/i differs from independent integer map')
        require(((rows['keys'][:, 1] >= 0) & (rows['keys'][:, 1] < 106)).all() and
                ((rows['keys'][:, 5] >= -1) & (rows['keys'][:, 5] < 6)).all(),
                'invalid trace block or chart')
        if flavor == 'native':
            require(np.array_equal(rows['keys'][:, 1], mapped[ids, 0]),
                    'native GID differs from independent integer map')
        if region == 'scri':
            require(np.array_equal(mapped[ids, 5],
                    144*rows['keys'][:, 5]+12*rows['keys'][:, 2]+rows['keys'][:, 3]) and
                    (rows['keys'][:, 4] == 12).all(), 'scri face/order differs')
        values[ids] = rows['values']; keys[ids] = rows['keys']; seen[ids] = True
        hashes[str(path.resolve())] = sha256(path)
    require(seen.all(), 'incomplete trace point coverage')
    if expected_keys is not None:
        require(np.array_equal(keys, expected_keys), 'trace identity changes between labels')
    canonical = hashlib.sha256(keys.tobytes()+values.tobytes()).hexdigest()
    return keys, values, dict(points=count, values=width, canonical_sha256=canonical,
                              source_sha256=hashes)


def exact(actual, expected):
    require(actual.shape == expected.shape and np.isfinite(actual).all() and
            np.isfinite(expected).all(), 'invalid exact-comparison arrays')
    changed = actual.view('<u8') != expected.view('<u8')
    delta = np.abs(actual-expected)
    worst = np.unravel_index(int(delta.argmax()), delta.shape)
    return dict(bitwise_equal=not bool(changed.any()), different_values=int(changed.sum()),
                different_points=int(changed.any(axis=1).sum()), max_absolute=float(delta[worst]),
                worst=dict(legacy_ordinal=int(worst[0]), component=int(worst[1]),
                           actual=float(actual[worst]), reference=float(expected[worst])))


def build_admission(path):
    report = json.loads(path.read_text())
    require(report['schema'] == 'original-stage-accessor-build-v1' and
            report['status'] == 'built' and report['build_exit_code'] == 0 and
            report['original_runtime_inputs_unchanged'] is True and
            report['original_executable_sha256'] == ORIGINAL_EXE,
            'legacy accessor build provenance failed')
    source = Path(__file__).with_name('legacy_stage_trace_gpu.cpp')
    matches = [value for name, value in report['sources'].items()
               if Path(name).name == source.name]
    require(matches == [sha256(source)], 'legacy accessor C++ differs from compiled source')
    flags = report['compile_command']
    require('-ffp-contract=off' in flags and '-fp-model=precise' in flags and '-O2' in flags,
            'legacy accessor FP policy differs')
    require(report.get('dependencies') and report.get('objects') and
            len(report['executable_sha256']) == 64, 'incomplete accessor build provenance')
    manifest_path = Path(__file__).resolve().parents[2]/'docs/z4c_hyperboloidal/evidence/n12-build-static-checkpoint-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(report['original_compile_command'] == manifest['commands'][0]['compile'] and
            report['original_link_command'] == manifest['commands'][0]['link'],
            'accessor original command attribution differs from pinned manifest')
    for relative, expected in manifest['sources'].items():
        matches = [value for name, value in report['sources'].items() if name.endswith('/'+relative)]
        require(matches == [expected], 'accessor pinned source changed: '+relative)
    return dict(path=str(path.resolve()), sha256=sha256(path),
                executable_sha256=report['executable_sha256'],
                original_executable_sha256=ORIGINAL_EXE, compile_command=flags,
                pinned_manifest_sha256=sha256(manifest_path), pinned_sources_checked=len(manifest['sources']),
                scope='archived successful build and source hash; remote runtime guards are retained by the launcher')


def control(folder, staging_path, step, ranks, mapping, one_step=False):
    """Admit unchanged executable snapshots and retained compensation checkpoint."""
    staging = json.loads(staging_path.read_text())
    require(staging['binary_sha256'] == ORIGINAL_EXE and
            staging['configuration_sha256'] == ORIGINAL_CONFIG,
            'control is not the pinned original executable/configuration')
    raw_input = (folder/'run.in').read_bytes()
    if one_step:
        # This launcher changes just nlim in the original eight-step bytes. Its
        # original staging record remains useful; do not invent a new provenance.
        require(step == 1 and raw_input.count(b'\nnlim=1\n') == 1,
                'one-step control input framing differs')
        restored = raw_input.replace(b'\nnlim=1\n', b'\nnlim=8\n')
        if 'source_input_sha256' in staging:
            require(hashlib.sha256(raw_input).hexdigest() == staging['input_sha256'],
                    'one-step bytes differ from one-step staging record')
        require(hashlib.sha256(restored).hexdigest() ==
                staging.get('source_input_sha256', staging['input_sha256']),
                'one-step control changed more than the documented nlim')
    else:
        require(hashlib.sha256(raw_input).hexdigest() == staging['input_sha256'],
                'control input differs from staging')
    require(sha256(folder/'configuration.cfg') == ORIGINAL_CONFIG, 'control configuration changed')
    params = parse_parameters(raw_input.decode())
    for key, value in [('time/dt', DT), ('time/nlim', step), ('numerics/ko', .1),
                       ('numerics/chi_floor', 1e-10)]:
        require(float(params[key]) == value, 'control parameter differs: '+key)
    require(float(params['time/tlim']) > expected_time(step) and
            params.get('tracker/enabled', 'false').lower() in ('false', '0'),
            'control endpoint clipping or tracker differs')
    plan = json.loads((folder/'plan.json').read_text())
    result = json.loads((folder/'result.json').read_text())
    require(plan['backend'] == 'SYCL' and plan['ranks'] == ranks and
            plan.get('cells_per_axis', 12) == 12 and plan['nlim'] == step and
            (plan['bulk_points'], plan['surface_points']) == (NBULK, NSURFACE) and
            plan['dt_M'] == 3*DT and not plan['host_evolution'], 'control GPU plan differs')
    require(result['status'] == 'step_limit' and result['steps'] == step and
            result['time_M'] == 3*expected_time(step) and not result['failure'],
            'control did not complete the requested steps')
    with (folder/'steps.csv').open() as stream:
        history = list(csv.DictReader(stream))
    require(len(history) == step, 'control step history incomplete')
    for number, row in enumerate(history, 1):
        require(int(row['step']) == number and float(row['t']) == expected_time(number)
                and float(row['dt']) == DT, 'control step history differs')
    coordinates = np.fromfile(folder/'coordinates.bin', dtype='<f8').reshape(NBULK, 11)
    require(np.isfinite(coordinates).all() and
            np.array_equal(coordinates[:, 8:11], mapping['bulk_keys'][:, 2:5]),
            'control coordinate order differs')
    snapshots = {}; reports = {}
    paths = [staging_path] + [folder/name for name in
        ('run.in', 'configuration.cfg', 'plan.json', 'result.json', 'steps.csv', 'coordinates.bin')]
    for label, cycle in [('initial', 0), ('final', step)]:
        target = folder/label
        clock = json.loads((target/'time.json').read_text())
        require(clock['steps'] == cycle and clock['t'] == expected_time(cycle),
                'control snapshot time differs')
        bulk = np.fromfile(target/'fields.bin', dtype='<f8').reshape(NBULK, NF)
        scri = np.fromfile(target/'boundary.bin', dtype='<f8').reshape(NSURFACE, NF)
        require(np.isfinite(bulk).all() and np.isfinite(scri).all(), 'nonfinite control fields')
        reports[label] = old_checkpoint(target/'restart.bin', cycle, bulk, scri)
        snapshots[label] = {'bulk': bulk, 'scri': scri}
        paths += [target/name for name in ('time.json', 'fields.bin', 'boundary.bin', 'restart.bin')]
    identity_path = Path(__file__).resolve().parents[2]/'docs/z4c_hyperboloidal/evidence/step-3551-checkpoint-layout.json'
    pinned_identity = json.loads(identity_path.read_text())
    for snapshot in reports.values():
        require(snapshot['physics_identity'] == pinned_identity['configuration_identity'] and
                snapshot['topology_identity'] == pinned_identity['static_identity'],
                'control physics/topology identity differs from pinned checkpoint')
    paths.append(identity_path)
    return snapshots, coordinates, dict(path=str(folder.resolve()), steps=step, ranks=ranks,
        stage_clocks=[float(row['stage_clock']) for row in history],
        reported_cfl=[float(row['CFL']) for row in history],
        plan=plan, snapshots=reports, source_sha256={str(p.resolve()): sha256(p) for p in paths})


def fields(actual, expected, region, mapping, xyz):
    return compare_fields(actual, expected, mapping[REGIONS[region]+'_keys'], xyz,
                          mapping[REGIONS[region]+'_weight'], 'short')


def audit(args):
    manifest, mapping = load_mapping(args.mapping)
    legacy, labels = legacy_metadata(args.legacy_trace)
    build = build_admission(args.legacy_build_provenance)
    originals, original_xyz, control_report = control(args.legacy_control,
        args.legacy_staging, legacy['steps'], legacy['ranks'], mapping)
    one = None; one_report = None
    if args.one_control:
        require(legacy['steps'] == 8, 'optional one-step control is redundant for accessor1')
        one, one_xyz, one_report = control(args.one_control,
            args.one_staging or args.legacy_staging, 1, legacy['ranks'], mapping, True)
        require(exact(one_xyz, original_xyz)['bitwise_equal'], 'one/eight control geometry differs')
    native = [native_metadata(path) for path in args.native_trace]
    require(all(r['backend'] == legacy['backend'] and r['ranks'] in (24, 96) for r in native),
            'native/legacy GPU backend or admitted rank count differs')
    report = dict(schema='legacy-native-stage-comparison-v1', passed=False,
        mapping_sha256=manifest['mapping_sha256'], legacy=legacy, legacy_build=build,
        control=control_report, one_control=one_report,
        gates=dict(accessor_final_snapshot='unchanged short field/chi gates against immutable original at same step/ranks; bitwise status reported separately',
                   geometry='bitwise actual GPU-build geometry; CPU XYZ excluded',
                   native_fields_and_rhs='1e-12+5e-10*abs(reference); chi 1e-14+5e-10*abs(reference)',
                   histories='same chi gate for two minima; exact zero-event/correction/first-floor records'),
        phases=dict(legacy_initial='after original initial diagnostic RHS',
                    native_initial='after Driver initialization and boundary fill',
                    accepted='after the complete step; no added diagnostic RHS',
                    final_snapshot='after original final diagnostic RHS',
                    first_rhs='legacy persistent first RK rate; native after-stage just-consumed RHS'),
        weighted_rms_scope='CPU independent map weights for diagnostics only; acceptance is pointwise',
        input_sha256={str((args.legacy_trace/'results.json').resolve()): sha256(args.legacy_trace/'results.json')},
        accessor_admission={}, comparisons=[], native_trace_admission=[])
    geometry = {}; legacy_keys = {}; accessor_ok = True
    for region in COUNTS:
        keys, xyz, provenance = load_trace(args.legacy_trace, 'legacy', 'geometry', region,
                                         legacy['ranks'], mapping)
        geometry[region] = xyz; legacy_keys[region] = keys
        report['input_sha256'].update(provenance['source_sha256'])
        if region == 'bulk':
            require(np.array_equal(keys[:, 1], original_xyz[:, 5]) and
                    np.array_equal(keys[:, 5], original_xyz[:, 6]), 'legacy block/face identity differs from unchanged control')
            pairs = np.unique(np.column_stack((keys[:, 1], mapping['bulk_keys'][:, 0])), axis=0)
            require(len(pairs) == 106 and len(np.unique(pairs[:, 0])) == 106 and
                    len(np.unique(pairs[:, 1])) == 106, 'legacy/native leaf mapping is not bijective')
            leaf_map = dict(pairs.tolist())
            report['leaf_bijection'] = pairs.tolist()
            comparison = exact(xyz, original_xyz[:, :3].copy())
            report['accessor_admission']['bulk_geometry'] = comparison
            accessor_ok &= comparison['bitwise_equal']
        else:
            require(np.array_equal([leaf_map[int(gid)] for gid in keys[:, 1]],
                                   mapping['surface_keys'][:, 0]), 'scri legacy/native leaf relation differs')
        ledger = None
        for label in labels:
            if label == 'geometry':
                continue
            _, data, item = load_trace(args.legacy_trace, 'legacy', label, region,
                                      legacy['ranks'], mapping, keys)
            report['input_sha256'].update(item['source_sha256'])
            if label == 'accepted'+str(legacy['steps']):
                ledger = data[:, NF:].copy()
            if label in ('accepted0', 'final-snapshot'):
                reference = originals['initial' if label == 'accepted0' else 'final'][region]
                result = dict(exact=exact(data[:, :NF].copy(), reference),
                              short=fields(data[:, :NF], reference, region, mapping, xyz))
                report['accessor_admission'][label+'-'+region] = result
                accessor_ok &= result['short']['failed_values'] == 0
            if label == 'accepted1' and one:
                # Different snapshot phases are deliberate; require the unchanged
                # scientific gate and report exactness without assuming it.
                result = fields(data[:, :NF], one['final'][region], region, mapping, xyz)
                report['accessor_admission']['accepted1-vs-control-final-'+region] = result
                accessor_ok &= result['failed_values'] == 0
        # Labels are not ordered chronologically; read final ledger explicitly.
        _, final, _ = load_trace(args.legacy_trace, 'legacy', 'final-snapshot', region,
                                legacy['ranks'], mapping, keys)
        require(ledger is not None, 'missing accepted-step ledger')
        result = exact(final[:, NF:].copy(), ledger)
        report['accessor_admission']['final_rhs_history_readonly-'+region] = result
        accessor_ok &= result['bitwise_equal']
    report['accessor_admission']['passed'] = bool(accessor_ok)
    all_ok = accessor_ok
    selected = {'accepted0', 'accepted1', 'first-rhs'}
    if legacy['steps'] == 8:
        selected.add('accepted8')
    for folder, info in zip(args.native_trace, native):
        result = dict(path=str(folder.resolve()), ranks=info['ranks'], mode=info['mode'],
                      fields={}, geometry={}, transitions={}, step_clocks=[], passed=True,
                      cfl_scope='not recorded by this frozen observer; maximum_rhs is not CFL')
        for cycle, expected in enumerate(control_report['stage_clocks']):
            actual = max(row['clock'] for row in info['stages'][4*cycle:4*cycle+4])
            require(np.isfinite(expected), 'nonfinite immutable control stage clock')
            delta = abs(actual-expected); tolerance = 1e-12+5e-10*abs(expected)
            clock = dict(step=cycle+1, actual=actual, reference=expected,
                         absolute_difference=delta, tolerance=tolerance,
                         bitwise_equal=struct.pack('<d', actual) == struct.pack('<d', expected),
                         passed=delta <= tolerance)
            result['step_clocks'].append(clock); result['passed'] &= clock['passed']
        report['input_sha256'][str((folder/'results.json').resolve())] = sha256(folder/'results.json')
        for region in COUNTS:
            keys = None; fingerprints = {}
            for label in NATIVE_LABELS:
                key, data, item = load_trace(folder, 'native', label, region, info['ranks'], mapping, keys)
                if keys is None:
                    keys = key
                    require(np.array_equal(keys[:, 5], legacy_keys[region][:, 5]), 'native/legacy chart identity differs')
                fingerprints[label] = item['canonical_sha256']
                report['input_sha256'].update(item['source_sha256'])
                if label == 'geometry':
                    result['geometry'][region] = exact(data, geometry[region])
                    result['passed'] &= result['geometry'][region]['bitwise_equal']
                elif label in selected:
                    _, reference, _ = load_trace(args.legacy_trace, 'legacy', label, region,
                        legacy['ranks'], mapping, legacy_keys[region])
                    comparison = dict(fields=fields(data[:, :NF], reference[:, :NF], region, mapping, geometry[region]))
                    okay = comparison['fields']['failed_values'] == 0
                    if label == 'accepted1' and one:
                        comparison['immutable_one_step_fields'] = fields(data[:, :NF],
                            one['final'][region], region, mapping, geometry[region])
                        comparison['immutable_one_step_phase'] = (
                            'native accepted1 versus immutable original final diagnostic snapshot')
                        okay &= comparison['immutable_one_step_fields']['failed_values'] == 0
                    if label != 'first-rhs':
                        comparison['records'] = record_comparison(data[:, NF:], reference[:, NF:],
                                                                 mapping[REGIONS[region]+'_keys'])
                        okay &= comparison['records']['failed_values'] == 0
                    comparison['passed'] = bool(okay)
                    result['fields'][label+'-'+region] = comparison
                    result['passed'] &= okay
            pairs = [('accepted0', 'stage1-input'), ('stage4-output', 'accepted1')]
            pairs += [(f'stage{s}-output', f'stage{s+1}-input') for s in range(1, 4)]
            for left, right in pairs:
                same = fingerprints[left] == fingerprints[right]
                result['transitions'][region+':'+left+'='+right] = same
                result['passed'] &= same
        all_ok &= result['passed']; report['comparisons'].append(result)
        report['native_trace_admission'].append(dict(path=str(folder.resolve()),
            complete_labels=len(NATIVE_LABELS)*2, stages=len(info['stages']),
            all_owned_values_finite=True, complete_unique_integer_keys=True))
    report['source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
        sha256(Path(__file__).with_name(name)) for name in
        ('legacy_stage_trace_compare.py', 'legacy_stage_trace_gpu.cpp',
         'legacy_stage_trace_gpu_contract.md', 'native_stage_trace_compare.py',
         'n12_checkpoint.py', 'n12_checkpoint_cross_platform.py', 'n12_checkpoint_gpu_control.py')}
    report['passed'] = bool(all_ok)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mapping', 'legacy-trace', 'legacy-control', 'legacy-staging',
                 'legacy-build-provenance', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--native-trace', type=Path, action='append', required=True)
    parser.add_argument('--one-control', '--legacy-one-control', type=Path, dest='one_control')
    parser.add_argument('--one-staging', '--legacy-one-staging', type=Path, dest='one_staging')
    args = parser.parse_args()
    require(not args.one_staging or args.one_control, '--one-staging requires --one-control')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = audit(args)
    except Exception as error:
        report = dict(schema='legacy-native-stage-comparison-v1', passed=False,
                      validation_error=f'{type(error).__name__}: {error}')
    (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('passed', 'validation_error') if key in report}))
    if 'accessor_admission' in report:
        print('accessor_admission', report['accessor_admission']['passed'])
        for item in report['comparisons']:
            print('native', item['ranks'], item['mode'], item['passed'])
            for label, result in item['fields'].items():
                print(label, 'failed', result['fields']['failed_values'],
                      'max_absolute', result['fields']['max_absolute'],
                      'records_failed', result.get('records', {}).get('failed_values', 0))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
