#!/usr/bin/env python3
"""Compare actual native N12 GPU checkpoints with an unchanged original GPU run."""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from n12_checkpoint import (NBULK, NSURFACE, NF, DT, compare_fields, decode,
    expected_time, load_mapping, parse_native, parse_parameters, require, sha256,
    validate_header, chi_summary)
from n12_checkpoint_cross_platform import identity_difference

ORIGINAL_EXE = '494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753'
ORIGINAL_CONFIG = '9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363'


def old_checkpoint(path, step, fields, surface):
    """Read fixed SP1 framing and identities, retaining no invented chi records."""
    size = path.stat().st_size
    payload = 2*(NBULK+NSURFACE)*NF*8
    offset = size-payload
    require(72 < offset < 65536, 'unexpected old SP1 header/payload extent')
    with path.open('rb') as stream:
        prefix = stream.read(offset)
    magic, n, nf, kind, admitted, cells, edges, steps, time, dt = struct.unpack_from('<16s4i3Q2d', prefix)
    require(magic.rstrip(b'\0') == b'HYP3D_MPI_SP1' and (n, nf, kind, admitted) == (12, 31, 2, 1),
            'old checkpoint schema/admission differs')
    require((cells, edges, steps, time, dt) == (NBULK, NSURFACE*NF, step, expected_time(step), DT),
            'old checkpoint coverage/clock differs')
    marker = b'RF2;flat-inner-Cartesian-Z4c-v1;'
    start = prefix.find(marker)
    require(start >= 80 and prefix.find(marker, start+1) == -1, 'missing/duplicate old physics identity')
    length, = struct.unpack_from('<Q', prefix, start-8)
    require(start+length+8 < len(prefix), 'old physics identity length invalid')
    identity = prefix[start:start+length].decode()
    length2, = struct.unpack_from('<Q', prefix, start+length)
    topology = prefix[start+length+8:].decode()
    require(len(topology.encode()) == length2 and topology.startswith('endpoint-static-mesh-v1;'),
            'old topology identity length/schema differs')
    data = np.memmap(path, mode='r', dtype='<f8', offset=offset)
    nbulk, nsurface = NBULK*NF, NSURFACE*NF
    require(np.isfinite(data).all(), 'old checkpoint nonfinite field/compensation')
    require(np.array_equal(data[:nbulk].view('<u8'), fields.reshape(-1).view('<u8')) and
            np.array_equal(data[2*nbulk:2*nbulk+nsurface].view('<u8'), surface.reshape(-1).view('<u8')),
            'old snapshot fields disagree with original checkpoint state')
    return dict(physics_identity=identity, topology_identity=topology,
                field_snapshot_matches_checkpoint_bitwise=True,
                compensation_scalars=nbulk+nsurface, chi_records_stored=False,
                sha256=sha256(path))


def resolved_comparison(native, legacy):
    new = dict(term.split('=', 1) for term in native.split(';') if '=' in term)
    old = dict(term.split('=', 1) for term in legacy.split(';') if '=' in term)
    groups = [('omega', ['g.profile.flat_radius', 'g.profile.outer_radius', 'g.profile.plateau']),
              ('gauge', ['g.inner_begin', 'g.inner_end', 'g.inner_gamma']),
              ('coefficient-window', ['g.coefficient_begin', 'g.coefficient_end']),
              ('restoration-window', ['g.restoration_begin', 'g.restoration_end'])]
    result = []
    for old_key, keys in groups:
        values = [float(x) for x in old[old_key].split(',')]
        require(len(values) == len(keys), 'old resolved field shape differs')
        for key, expected in zip(keys, values):
            actual = float.fromhex(new[key])
            result.append(dict(native_key=key, legacy_key=old_key,
                actual=actual, expected=expected, bitwise_equal=struct.pack('<d', actual) == struct.pack('<d', expected)))
    # The native identity is more comprehensive than historical RF2. Preserve
    # both strings; this comparison only claims the serialized common fields.
    return dict(compared_common_scalars=len(result), all_common_exact=all(x['bitwise_equal'] for x in result),
        fields=result, scope='common resolved doubles serialized by both identities; remaining physics bound by pinned input/source audits')


def native_coordinates(directory, mapping):
    xyz = np.empty((NBULK, 3)); seen = np.zeros(NBULK, dtype=bool)
    dtype = np.dtype([('keys', '<i4', (8,)), ('values', '<f8', (26,))])
    paths = sorted(directory.glob('points-rank*.bin')); require(paths, 'missing native coordinate dumps')
    for path in paths:
        with path.open('rb') as stream:
            require(stream.read(16) == b'NATIVE_DIAG_V1\0\0', 'unknown native coordinate format')
            count, nk, nv = struct.unpack('<QII', stream.read(16))
            rows = np.fromfile(stream, dtype=dtype)
        require((nk, nv) == (8, 26) and len(rows) == count, 'coordinate format/length differs')
        ids = rows['keys'][:, 0]
        require(((ids >= 0) & (ids < NBULK)).all() and np.unique(ids).size == len(ids), 'invalid coordinate ordinal')
        require(not seen[ids].any(), 'coordinate ordinal repeated across ranks')
        keys = mapping['bulk_keys'][ids]
        require(np.array_equal(rows['keys'][:, 1], keys[:, 0]) and
                np.array_equal(rows['keys'][:, 2:5], keys[:, [4, 3, 2]]), 'native coordinate keys differ')
        xyz[ids] = rows['values'][:, :3]; seen[ids] = True
    require(seen.all() and np.isfinite(xyz).all(), 'native coordinate coverage incomplete/nonfinite')
    return xyz, {str(p): sha256(p) for p in paths}


def compare_chi(summary, legacy, step):
    require(legacy['ko'] == .1 and legacy['chi_floor'] == 1e-10 and legacy['time_M'] == 3*expected_time(step),
            'old chi policy/time differs')
    result = dict(passed=True, per_cell_reference_available=False, per_cell_records_compared=False,
        scope='all available aggregate chi diagnostics only; historical SP1 has no per-cell records', fields=[])
    for key, actual in summary.items():
        require(key in legacy, 'old chi aggregate missing '+key)
        expected = legacy[key]
        if actual == 1e300 and key.startswith('minimum_raw'):
            actual = None
        numeric_minimum = key in ('minimum_chi', 'minimum_raw_candidate', 'minimum_raw_acceptance_candidate')
        tolerance = (1e-14+5e-10*abs(expected)) if numeric_minimum and expected is not None else 0.
        equal = (actual is None and expected is None) if actual is None or expected is None else abs(actual-expected) <= tolerance
        result['passed'] &= equal
        result['fields'].append(dict(key=key, actual=actual, expected=expected,
            absolute_difference=None if actual is None or expected is None else abs(actual-expected),
            tolerance=tolerance, passed=equal))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--eight', type=Path, required=True)
    parser.add_argument('--legacy-control', type=Path, required=True)
    parser.add_argument('--legacy-staging', type=Path, required=True)
    parser.add_argument('--native-coordinate-directory', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=(24, 96), default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    manifest, mapping = load_mapping(args.mapping)
    staging = json.loads(args.legacy_staging.read_text())
    require(staging['binary_sha256'] == ORIGINAL_EXE and staging['configuration_sha256'] == ORIGINAL_CONFIG,
            'old control does not use pinned executable/configuration')
    oldroot = args.legacy_control; params = parse_parameters((oldroot/'run.in').read_text())
    require(sha256(oldroot/'run.in') == staging['input_sha256'], 'old staged input bytes changed')
    require(sha256(oldroot/'configuration.cfg') == ORIGINAL_CONFIG, 'old configuration bytes changed')
    for key, value in [('time/dt', DT), ('time/nlim', 8), ('numerics/ko', .1),
                       ('numerics/chi_floor', 1e-10)]:
        actual = params[key]
        require(float(actual) == value, 'old control parameter differs '+key)
    require(float(params['time/tlim']) > expected_time(8), 'old control clips endpoint')
    require(params.get('tracker/enabled', 'false').lower() in ('false', '0'), 'old tracker enabled')
    plan = json.loads((oldroot/'plan.json').read_text())
    finished = json.loads((oldroot/'result.json').read_text())
    require(finished['status'] == 'step_limit' and finished['steps'] == 8 and
            finished['time_M'] == 3*expected_time(8) and not finished['failure'],
            'old GPU run did not finish the requested eight steps successfully')
    # Historical fixed-N12 wrapper predates the later optional N input/plan
    # field. The actual SP1 header below must independently contain n=12.
    require(plan['ranks'] == args.ranks and plan.get('cells_per_axis', 12) == 12 and
            plan['bulk_points'] == NBULK and plan['surface_points'] == NSURFACE and
            plan['dt_M'] == 3*DT and not plan['host_evolution'], 'old GPU plan mismatch')
    with (oldroot/'steps.csv').open() as stream:
        steps = list(csv.DictReader(stream))
    require(len(steps) == 8, 'old control did not take eight steps')
    for number, row in enumerate(steps, 1):
        require(int(row['step']) == number and float(row['t']) == expected_time(number) and
                float(row['dt']) == DT, 'old control step history differs')
    oldxyz = np.fromfile(oldroot/'coordinates.bin', dtype='<f8').reshape(NBULK, 11)
    require(np.isfinite(oldxyz).all() and np.array_equal(oldxyz[:, 8:11], mapping['bulk_keys'][:, 2:5]),
            'old sample ordering/indices differ')
    pairs = np.unique(np.column_stack((oldxyz[:, 5], mapping['bulk_keys'][:, 0])), axis=0)
    require(len(pairs) == 106 and len(np.unique(pairs[:, 0])) == 106 and len(np.unique(pairs[:, 1])) == 106,
            'old/native leaves are not bijective')
    xyz, coordinate_hashes = native_coordinates(args.native_coordinate_directory, mapping)
    changed = xyz.view('<u8') != oldxyz[:, :3].copy().view('<u8')
    report = dict(passed=False, scope='matched Aurora GPU initial/eight-step all-field comparison',
        gates=dict(fields='1e-12+5e-10*abs(reference)', chi='1e-14+5e-10*abs(reference)',
                   counts_and_zero_corrections='exact', coordinates='bitwise', common_resolved_policy='bitwise'),
        mapping_sha256=manifest['mapping_sha256'], original_binary_sha256=ORIGINAL_EXE,
        original_staging=staging, original_plan=plan,
        coordinates=dict(bulk_points=NBULK, exact_equal=not changed.any(),
            different_points=int(changed.any(axis=1).sum()), different_components=int(changed.sum()),
            max_absolute=float(abs(xyz-oldxyz[:, :3]).max()),
            scope='native frozen saved3551 replay cached geometry versus fresh original control geometry; fixed atlas/build',
            surface_scope='same fixed topology/legacy surface order from independent mapping; no new surface dump'),
        source_sha256={str(Path(__file__).with_name(name).resolve()):
            sha256(Path(__file__).with_name(name)) for name in
            ('n12_checkpoint_gpu_control.py', 'n12_checkpoint.py', 'n12_checkpoint_cross_platform.py')},
        weighted_rms_scope='independent CPU map evolved-point volumes/scri areas; pointwise acceptance is unweighted',
        coordinate_source_sha256=coordinate_hashes, steps={})
    for step, candidate, name in [(0, args.initial, 'initial'), (8, args.eight, 'final')]:
        info = parse_native(candidate)
        validate_header(info, dict(manifest, physics_identity=info['identity'].decode()), step)
        require(info['metadata'][4] == args.ranks, 'native writer/control rank count differs')
        state = decode(info, mapping); folder = oldroot/name
        clock = json.loads((folder/'time.json').read_text())
        require(clock['steps'] == step and clock['t'] == expected_time(step), 'old snapshot clock differs')
        oldfields = np.fromfile(folder/'fields.bin', dtype='<f8').reshape(NBULK, NF)
        oldsurface = np.fromfile(folder/'boundary.bin', dtype='<f8').reshape(NSURFACE, NF)
        checkpoint = old_checkpoint(folder/'restart.bin', step, oldfields, oldsurface)
        identity = resolved_comparison(info['identity'].decode(), checkpoint['physics_identity'])
        entry = dict(candidate=str(candidate.resolve()), candidate_sha256=sha256(candidate),
            time=info['time'], dt=info['dt'], writer_mpi_ranks=args.ranks,
            cpu_sidecar_identity_differences_preserved=identity_difference(manifest['physics_identity'], info['identity'].decode()),
            original_checkpoint=checkpoint, common_resolved_policy=identity, fields={})
        for region, expected in [('bulk', oldfields), ('surface', oldsurface)]:
            entry['fields'][region] = compare_fields(state[region][:, :NF], expected,
                mapping[region+'_keys'], xyz if region == 'bulk' else mapping['surface_xyz'],
                mapping[region+'_weight'], 'short')
        summary = chi_summary(state, mapping)
        # Original minimum-chi aggregate is defined over bulk only.
        summary['minimum_chi'] = float(state['bulk'][:, 0].min())
        entry['chi_aggregates'] = compare_chi(summary, json.loads((folder/'chi-policy.json').read_text()), step)
        entry['passed'] = identity['all_common_exact'] and entry['chi_aggregates']['passed'] and all(
            value['failed_values'] == 0 for value in entry['fields'].values())
        entry['reference_sha256'] = {n: sha256(folder/n) for n in
            ('fields.bin', 'boundary.bin', 'time.json', 'chi-policy.json', 'restart.bin')}
        report['steps'][str(step)] = entry
    report['passed'] = report['coordinates']['exact_equal'] and all(v['passed'] for v in report['steps'].values())
    report['legacy_control_sha256'] = {name: sha256(oldroot/name) for name in
        ('run.in', 'configuration.cfg', 'coordinates.bin', 'steps.csv', 'plan.json', 'result.json')}
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], coordinates=report['coordinates'],
        steps={s:dict(passed=v['passed'], fields={r:x['failed_values'] for r,x in v['fields'].items()},
                     chi_passed=v['chi_aggregates']['passed']) for s,v in report['steps'].items()}), indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
