#!/usr/bin/env python3
"""Decode/export/compare fixed-N12 native shared checkpoints, independent of MPI size."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

DT = 0.00046944203735617445
NBULK, NSURFACE, NBLOCK, NSLOT, NF, NR = 229392, 864, 106, 18**3+12**2, 31, 8
HEADER = struct.calcsize('<ii9d38i2di')
MAGIC = 0x4859504552525354
FIELDS = ['chi', 'gxx-1', 'gxy', 'gxz', 'gyy-1', 'gyz', 'gzz-1',
          'Axx', 'Axy', 'Axz', 'Ayy', 'Ayz', 'Azz', 'P+3', 'Theta',
          'Lambda_x', 'Lambda_y', 'Lambda_z', 'alpha-(1+r2)/2',
          'B_x', 'B_y', 'B_z', 'beta_x+x', 'beta_y+y', 'beta_z+z',
          'D_x', 'D_y', 'D_z', 'J_x', 'J_y', 'J_z']
PINNED = {'fields.bin': '7d37c940eea84a04dc471bb90178c3e2facf153f14382cd2a47cf1fba6a320be',
          'boundary.bin': '343d4b04c6dc6456eb8bdf84bf1ddaa97fc1aee9531bdb1299c4d8573394e6ef',
          'time.json': '1608393d11c0020a9e3ffc25af19bf9ea92c0a70a5e57b62cb8b10829cf0e469',
          'chi-policy.json': '399d499a5e67dcfdf34c6642806876293cd33a74ef2e3a26af1f6f8d2019ed7b'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_parameters(text):
    values, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line:
            key, value = line.split('=', 1)
            values[block+'/'+key.strip()] = value.strip()
    return values


def parse_native(path):
    path = Path(path).resolve()
    size = path.stat().st_size
    with path.open('rb') as stream:
        text = bytearray()
        while not text.endswith(b'<par_end>\n'):
            line = stream.readline(262144)
            require(line and len(text)+len(line) <= 8*1024*1024,
                    'missing or oversized native parameter header')
            text.extend(line)
        start = stream.tell()

        def take(count):
            require(0 <= count <= size-stream.tell(), 'truncated or oversized checkpoint record')
            value = stream.read(count)
            require(len(value) == count, 'truncated checkpoint record')
            return value

        fixed = take(HEADER)
        blocks, root_level = struct.unpack_from('<ii', fixed)
        require(blocks == NBLOCK and root_level == 4, 'checkpoint is not the fixed N12 tree')
        time, dt, cycle = struct.unpack_from('<ddi', fixed, HEADER-20)
        atlas_length, = struct.unpack('<Q', take(8))
        require(atlas_length <= 65536, 'invalid atlas identity size')
        atlas = take(atlas_length)
        topology = take(20*blocks)
        costs = take(4*blocks)
        metadata = struct.unpack('<8Q', take(64))
        require(metadata[:2] == (MAGIC, 1), 'unknown hyperboloidal checkpoint schema')
        require(metadata[2] <= 65536, 'invalid numerical identity size')
        identity = take(metadata[2])
        stride, = struct.unpack('<Q', take(8))
        require(metadata[3] == NSLOT*(NF+NR) and stride == metadata[3]*8,
                'checkpoint field/record stride differs from N12')
        require(metadata[7] == 0, 'only native shared checkpoints are supported')
        require(1 <= metadata[4] <= NBLOCK and metadata[5] == 0 and
                metadata[6] == NBLOCK//metadata[4], 'invalid shared-checkpoint ownership metadata')
        offset = stream.tell()
        require(size == offset+NBLOCK*stride, 'checkpoint payload truncated or has trailing bytes')
    return dict(path=path, size=size, start=start, fixed=fixed, atlas=atlas,
                topology=topology, costs=costs, identity=identity, metadata=metadata,
                offset=offset, stride=stride, time=time, dt=dt, cycle=cycle,
                parameters=parse_parameters(text.decode('utf-8')))


def load_mapping(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    require(manifest['schema'] == 'native-n12-legacy-map-v1', 'unknown N12 map schema')
    require(sha256(directory/'mapping.npz') == manifest['mapping_sha256'], 'mapping checksum differs')
    with np.load(directory/'mapping.npz', allow_pickle=False) as source:
        mapping = {key: source[key] for key in source.files}
    require(manifest['coordinate_mismatches'] == 0 and manifest['coordinate_comparison'] == 'bitwise-float64',
            'mapping lacks exact coordinate proof')
    all_slots = []
    for region, count in [('bulk', NBULK), ('surface', NSURFACE)]:
        keys, xyz = mapping[region+'_keys'], mapping[region+'_xyz']
        require(keys.shape == (count, 6) and xyz.shape == (count, 3), 'mapping coverage shape differs')
        require(np.isfinite(xyz).all(), 'nonfinite mapping coordinates')
        require(np.array_equal(np.sort(keys[:, 5]), np.arange(count)), 'canonical map is not bijective')
        require(((keys[:, 0] >= 0) & (keys[:, 0] < NBLOCK)).all(), 'mapping block out of range')
        require(((keys[:, 1] >= 0) & (keys[:, 1] < NSLOT)).all(), 'mapping slot out of range')
        weight = mapping[region+'_weight']
        require(weight.shape == (count,) and np.isfinite(weight).all() and (weight > 0).all(),
                'invalid error norm weights')
        all_slots.append(keys[:, 0]*NSLOT+keys[:, 1])
    require(np.unique(np.concatenate(all_slots)).size == NBULK+NSURFACE, 'duplicate mapped restart slot')
    return manifest, mapping


def expected_time(step):
    time = 0.
    for _ in range(step):
        time += DT
    return time


def validate_header(info, manifest, step):
    require(info['cycle'] == step and info['dt'] == DT and info['time'] == expected_time(step),
            'step/time/dt differs from full nominal N12 steps')
    # Original native Mesh initializes the ten global fine indices, but not its
    # nine unused global coarse indices. RestartOutput writes the whole struct.
    # Compare every meaningful field, retaining those unused bytes for audit.
    reference_header = bytes.fromhex(manifest['fixed_mesh_header_hex'])
    require(info['fixed'][:120] == reference_header[:120] and
            info['fixed'][156:-20] == reference_header[156:], 'native mesh header differs')
    require(info['atlas'].decode() == manifest['atlas_identity'], 'atlas identity differs')
    require(info['identity'].decode() == manifest['physics_identity'], 'numerical policy identity differs')
    require(info['topology'].hex() == manifest['topology_hex'], 'native leaf order/topology differs')
    require(np.array_equal(np.frombuffer(info['costs'], dtype='<f4'), np.ones(NBLOCK, dtype='<f4')),
            'native block costs are not all one')
    for key, expected in manifest['required_parameters'].items():
        require(key in info['parameters'], 'missing required parameter '+key)
        actual = info['parameters'][key]
        require((float(actual) == expected) if isinstance(expected, (int, float)) else actual == expected,
                'changed required parameter '+key)
    for key, value in [('fastflow/use_puncture_0', -1), ('fastflow/npunct', 0),
                       ('fastflow/center_x_0', 0), ('fastflow/center_y_0', 0),
                       ('fastflow/center_z_0', 0)]:
        require(float(info['parameters'].get(key, value)) == value, 'N12 fixed-origin policy differs')


def decode(info, mapping):
    data = np.memmap(info['path'], mode='r', dtype='<f8', offset=info['offset'],
                     shape=(NBLOCK, NSLOT, NF+NR))
    active = np.zeros((NBLOCK, NSLOT), dtype=bool)
    for region in ('bulk', 'surface'):
        keys = mapping[region+'_keys'];active[keys[:, 0], keys[:, 1]] = True
    for block in range(NBLOCK):
        require(np.isfinite(data[block]).all(), 'nonfinite field/history/padding in checkpoint')
        require((data[block, ~active[block], :] == 0).all(), 'nonzero unused native checkpoint padding')
    result = {}
    for region in ('bulk', 'surface'):
        keys = mapping[region+'_keys']
        result[region] = np.array(data[keys[:, 0], keys[:, 1], :], copy=True)
        state, records = result[region][:, :NF], result[region][:, NF:]
        require((state[:, 0] > 0).all(), 'nonpositive current chi')
        require((records[:, 2:7] >= 0).all(), 'negative chi event/correction record')
        require(np.equal(records[:, 2:5], np.floor(records[:, 2:5])).all(), 'nonintegral chi event count')
        require((records[:, 2:5] <= 2**53).all(), 'chi count exceeds exact float64 integer range')
    return result


def audit_coordinates(paths, mapping):
    """Validate actual native cached geometry dumps, generated on host then uploaded."""
    seen = np.zeros(NBULK, dtype=bool);count = 0
    dtype = np.dtype([('keys', '<i4', (8,)), ('values', '<f8', (26,))])
    for path in paths:
        with Path(path).open('rb') as stream:
            require(stream.read(16).rstrip(b'\0') == b'NATIVE_DIAG_V1', 'unknown diagnostic dump')
            n, nk, nv = struct.unpack('<QII', stream.read(16))
            require((nk, nv) == (8, 26), 'diagnostic dump shape changed')
            rows = np.fromfile(stream, dtype=dtype)
        require(len(rows) == n, 'truncated diagnostic coordinate dump')
        ids = rows['keys'][:, 0]
        require(((ids >= 0) & (ids < NBULK)).all() and np.unique(ids).size == n,
                'duplicate/out-of-range coordinate ordinal')
        require(not seen[ids].any(), 'coordinate ordinal appears on multiple ranks')
        keys = mapping['bulk_keys'][ids]
        require(np.array_equal(rows['keys'][:, 1], keys[:, 0]) and
                np.array_equal(rows['keys'][:, 2:5], keys[:, [4, 3, 2]]), 'coordinate key mismatch')
        require(np.array_equal(rows['values'][:, :3].view('<u8'),
                               mapping['bulk_xyz'][ids].view('<u8')), 'cached native coordinates differ')
        seen[ids] = True;count += n
    require(seen.all(), 'actual coordinate proof omits evolved bulk points')
    return dict(bulk_points=count, bitwise_equal=True, surface_scope='host atlas proof only',
                geometry_generation='native host cache mirrored after device upload; not GPU map generation')


def compare_fields(actual, expected, keys, xyz, weight, gate):
    require(actual.shape == expected.shape and actual.shape[1] == NF, 'field comparison shape differs')
    require(np.isfinite(expected).all(), 'nonfinite reference field')
    absolute, relative, chi = (1e-12, 5e-10, 1e-14) if gate == 'short' else (1e-10, 1e-8, 1e-13)
    report = dict(comparisons=int(actual.size), failed_values=0, bitwise_equal=True,
                  max_absolute=0., max_scaled=0., max_tolerance_ratio=0., fields=[])
    weighted_square, total_weight = 0., float(weight.sum())

    def site(index, field):
        return dict(legacy_ordinal=int(index), native_gid=int(keys[index, 0]),
                    native_restart_slot=int(keys[index, 1]), ijk=keys[index, 2:5].tolist(),
                    xyz=xyz[index].tolist(), field=field, field_name=FIELDS[field],
                    actual=float(actual[index, field]), expected=float(expected[index, field]))

    for field in range(NF):
        delta = np.abs(actual[:, field]-expected[:, field])
        scaled = delta/(1+np.abs(expected[:, field]))
        ratio = delta/((chi if field == 0 else absolute)+relative*np.abs(expected[:, field]))
        square = float(np.sum(weight*delta*delta));weighted_square += square
        a, s, t = int(delta.argmax()), int(scaled.argmax()), int(ratio.argmax())
        failures = int(np.count_nonzero(ratio > 1));report['failed_values'] += failures
        same = np.array_equal(actual[:, field].view('<u8'), expected[:, field].view('<u8'))
        report['bitwise_equal'] &= same
        per_field = dict(field=field, name=FIELDS[field], comparisons=len(delta),
            failed_values=failures, max_absolute=float(delta[a]), max_scaled=float(scaled[s]),
            max_tolerance_ratio=float(ratio[t]), weighted_rms=math.sqrt(square/total_weight),
            worst_absolute=site(a, field), worst_scaled=site(s, field), worst_ratio=site(t, field))
        report['fields'].append(per_field)
        for metric, index in [('absolute', a), ('scaled', s), ('tolerance_ratio', t)]:
            value = float({'absolute': delta, 'scaled': scaled, 'tolerance_ratio': ratio}[metric][index])
            if field == 0 or value > report['max_'+metric]:
                report['max_'+metric] = value;report['worst_'+metric] = site(index, field)
    report['weighted_rms'] = math.sqrt(weighted_square/(NF*total_weight))
    report['weight_sum_per_field'] = total_weight
    return report


def chi_summary(decoded, mapping):
    records = np.concatenate([decoded[r][:, NF:] for r in ('bulk', 'surface')])
    state_chi = np.concatenate([decoded[r][:, 0] for r in ('bulk', 'surface')])
    xyz = np.concatenate([mapping[r+'_xyz'] for r in ('bulk', 'surface')])
    floors = records[:, 2] > 0
    return dict(minimum_chi=float(state_chi.min()), minimum_raw_candidate=float(records[:, 0].min()),
        minimum_raw_acceptance_candidate=float(records[:, 1].min()),
        floor_events=int(records[:, 2].sum()), nonpositive_candidates=int(records[:, 3].sum()),
        nonfinite_candidates=int(records[:, 4].sum()), distinct_floored_cells=int(floors.sum()),
        sum_corrections_over_attempts=float(records[:, 6].sum()),
        surface_floor_events=int(decoded['surface'][:, NF+2].sum()),
        max_correction=float(records[:, 5].max()),
        max_floor_radius=float(np.sqrt((xyz[floors]**2).sum(axis=1)).max()) if floors.any() else 0.,
        first_floor_M=float(3*records[floors, 7].min()) if floors.any() else None)


def execute(candidate, mapping_directory, output, step, reference=None, coordinate_dumps=()):
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=False)
    manifest, mapping = load_mapping(mapping_directory)
    info = parse_native(candidate);validate_header(info, manifest, step)
    decoded = decode(info, mapping)
    report = dict(passed=True, step=step, time=info['time'], dt=info['dt'],
        writer_mpi_ranks=info['metadata'][4], candidate=str(info['path']), candidate_sha256=sha256(info['path']),
        mapping_sha256=manifest['mapping_sha256'], all31_fields_decoded=True,
        bulk_points=NBULK, scri_points=NSURFACE, chi_record_scalars=NR*(NBULK+NSURFACE),
        coordinate_scope='bitwise host native-atlas/independent-legacy proof bound by exact checkpoint metadata',
        unused_global_coarse_indices=list(struct.unpack_from('<9i', info['fixed'], 120)),
        mesh_header_scope='all initialized fields exact; nine unused native global coarse indices reported only',
        stage_diagnostics_scope='not stored in native restart; require separate stage/histories evidence',
        chi_history_reference_scope='none', chi_summary=chi_summary(decoded, mapping))
    for region, field_name, record_name in [('bulk', 'fields.bin', 'chi-records.bin'),
                                           ('surface', 'boundary.bin', 'surface-chi-records.bin')]:
        decoded[region][:, :NF].astype('<f8').tofile(output/field_name)
        decoded[region][:, NF:].astype('<f8').tofile(output/record_name)
    if coordinate_dumps:
        report['actual_coordinate_audit'] = audit_coordinates(coordinate_dumps, mapping)
    if reference:
        reference = Path(reference).resolve();clock = json.loads((reference/'time.json').read_text())
        require(clock['steps'] == step and clock['t'] == info['time'], 'reference step/time differs')
        if step == 3551:
            for name, digest in PINNED.items():
                require(sha256(reference/name) == digest, 'reference differs from pinned N12 '+name)
        else:
            metadata = json.loads((reference/'reference-manifest.json').read_text())
            require(metadata['mapping_sha256'] == manifest['mapping_sha256'], 'reference map proof differs')
            for name in ('fields.bin', 'boundary.bin'):
                require(sha256(reference/name) == metadata['sha256'][name], 'reference checksum differs')
        gate = 'short' if step <= 8 else 'long'
        report['gate'] = gate;report['reference'] = str(reference);report['fields'] = {}
        for region, name, count in [('bulk', 'fields.bin', NBULK), ('surface', 'boundary.bin', NSURFACE)]:
            require((reference/name).stat().st_size == count*NF*8, 'reference field length differs')
            values = np.memmap(reference/name, mode='r', dtype='<f8', shape=(count, NF))
            metric = compare_fields(decoded[region][:, :NF], values, mapping[region+'_keys'],
                                    mapping[region+'_xyz'], mapping[region+'_weight'], gate)
            report['fields'][region] = metric;report['passed'] &= metric['failed_values'] == 0
        if (reference/'chi-policy.json').exists():
            policy = json.loads((reference/'chi-policy.json').read_text())
            require(policy['ko'] == .1 and policy['chi_floor'] == 1e-10, 'reference chi policy differs')
            report['chi_history_reference_scope'] = 'aggregate only; no per-cell reference histories'
            report['chi_reference_aggregate'] = policy
            # Before the first attempted step the original writer records no
            # raw-candidate minimum. Keep that absence explicit; it is not zero
            # and cannot be subtracted from a native initialization sentinel.
            report['chi_aggregate_differences'] = {
                key: (None if value is None or policy[key] is None else value-policy[key])
                for key, value in report['chi_summary'].items() if key in policy}
            report['chi_aggregate_difference_unavailable'] = [
                key for key, value in report['chi_summary'].items()
                if key in policy and (value is None or policy[key] is None)]
            report['chi_aggregate_exact_equal'] = {
                key: value == policy[key] for key, value in report['chi_summary'].items() if key in policy}
            report['per_cell_chi_history_compared'] = False
    (output/'time.json').write_text(json.dumps(dict(steps=step, t=info['time'], t_M=3*info['time'], dt=DT))+'\n')
    report['export_sha256'] = {name: sha256(output/name) for name in
        ('fields.bin', 'boundary.bin', 'chi-records.bin', 'surface-chi-records.bin')}
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--coordinate-dumps', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    require(sys.byteorder == 'little' and args.step >= 0, 'requires little-endian float64 native checkpoint')
    result = execute(args.candidate, args.mapping, args.output, args.step, args.reference, args.coordinate_dumps)
    print(json.dumps({key: result[key] for key in ('passed', 'step', 'writer_mpi_ranks',
        'bulk_points', 'scri_points', 'chi_record_scalars')}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
