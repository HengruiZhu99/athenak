#!/usr/bin/env python3
"""Create a separate strict Aurora map only after independent full geometry proof.

Never edits the CPU sidecar. Requires the passed matched-GPU control, an
Aurora-built independent map helper, and actual native trace scri coordinates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from n12_checkpoint import (NBULK, NSURFACE, NF, compare_fields, decode, load_mapping,
    parse_native, require, sha256, validate_header)
from n12_checkpoint_cross_platform import identity_difference
from n12_checkpoint_gpu_control import native_coordinates


def read_geometry(directory):
    path = directory/'mapping.bin'
    require(path.stat().st_size == 48+(NBULK+NSURFACE)*64,
            'independent map byte length differs')
    with path.open('rb') as stream:
        require(stream.read(16) == b'N12_MAP_V1'+b'\0'*6, 'invalid independent map magic')
        require(struct.unpack('<4Q', stream.read(32)) == (NBULK, NSURFACE, 106, 64),
                'independent map dimensions differ')
        data = np.fromfile(stream, dtype=np.dtype([('keys', '<i4', (6,)), ('values', '<f8', (5,))]))
    require(len(data) == NBULK+NSURFACE, 'independent map incomplete')
    arrays = {}
    for region, rows in [('bulk', data[:NBULK]), ('surface', data[NBULK:])]:
        arrays[region+'_keys'] = rows['keys'].copy()
        arrays[region+'_xyz'] = rows['values'][:, :3].copy()
        arrays[region+'_weight'] = rows['values'][:, 3].copy()
        arrays[region+'_diagnostic_weight'] = rows['values'][:, 4].copy()
    return arrays


def surface_geometry(directory, mapping):
    seen = np.zeros(NSURFACE, dtype=bool); xyz = np.empty((NSURFACE, 3))
    by_canonical = np.argsort(mapping['surface_keys'][:, 5])
    paths = sorted(directory.glob('geometry-scri-rank*.bin'))
    require(paths, 'missing actual native scri geometry trace')
    for path in paths:
        with path.open('rb') as stream:
            require(stream.read(16) == b'NATIVE_TRACE_V1\0', 'unknown scri trace schema')
            count, = struct.unpack('<Q', stream.read(8))
            require(struct.unpack('<4I', stream.read(16)) == (6, 3, 1, 1), 'wrong scri trace dimensions')
            require(path.stat().st_size == 40+count*48, 'scri geometry trace byte length differs')
            rows = np.fromfile(stream, dtype=np.dtype([('keys', '<i4', (6,)), ('values', '<f8', (3,))]))
        require(len(rows) == count, 'scri geometry trace truncated')
        ids = rows['keys'][:, 0]
        require(((ids >= 0) & (ids < NSURFACE)).all() and np.unique(ids).size == count and
                not seen[ids].any(), 'scri canonical membership invalid')
        keys = mapping['surface_keys'][by_canonical[ids]]
        require(np.array_equal(rows['keys'][:, 1], keys[:, 0]) and
                np.array_equal(rows['keys'][:, 2:5], keys[:, [4, 3, 2]]) and
                np.array_equal(rows['keys'][:, 5], ids//144), 'actual scri trace keys differ')
        xyz[ids] = rows['values']; seen[ids] = True
    require(seen.all() and np.isfinite(xyz).all(), 'actual scri coordinate coverage incomplete/nonfinite')
    return xyz[mapping['surface_keys'][:, 5]], {str(p): sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-mapping', type=Path, required=True)
    parser.add_argument('--template-checkpoint', type=Path, required=True)
    parser.add_argument('--matched-gpu-result', type=Path, required=True)
    parser.add_argument('--native-run-evidence', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--geometry-build', type=Path, required=True)
    parser.add_argument('--geometry-execution-evidence', type=Path, required=True)
    parser.add_argument('--native-bulk-coordinates', type=Path, required=True)
    parser.add_argument('--native-scri-trace', type=Path, required=True)
    parser.add_argument('--preserved-extra-check-failure', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'new Aurora mapping directory is required')
    manifest, cpu = load_mapping(args.cpu_mapping)
    control = json.loads(args.matched_gpu_result.read_text())
    require(control['passed'] and control['coordinates']['exact_equal'] and
            all(v['passed'] for v in control['steps'].values()), 'matched original GPU control did not pass')
    template = parse_native(args.template_checkpoint)
    digest = sha256(args.template_checkpoint)
    require(control['steps']['0']['candidate_sha256'] == digest, 'template was not admitted by matched GPU control')
    validate_header(template, dict(manifest, physics_identity=template['identity'].decode()), 0)
    proof = json.loads((args.geometry_build/'provenance.json').read_text())
    native_run = json.loads(args.native_run_evidence.read_text())
    require(native_run['exit_status'] == 0 and native_run['source_runtime_unchanged'] and
            proof['native_executable_sha256'] == native_run['native_executable_sha256'],
            'independent geometry build does not bind the validated native GPU executable')
    require(proof['schema'] == 'native-n12-map-build-v1' and proof['status'] == 'built' and
            proof['runtime_inputs_unchanged'], 'independent Aurora geometry helper build not verified')
    execution = json.loads(args.geometry_execution_evidence.read_text())
    require(execution['schema'] == 'native-n12-map-execution-v1' and
            execution['exit_status'] == 0 and execution['ranks'] == 1 and
            execution['executable_sha256'] == proof['executable_sha256'],
            'independent Aurora geometry helper execution failed or changed')
    for name in ('mapping.bin', 'topology.bin', 'atlas.txt', 'initial-fields.bin',
                 'initial-boundary.bin', 'run.log'):
        require(execution['artifacts'][name] == sha256(args.geometry/name),
                'independent geometry execution artifact changed: '+name)
    require('PASS N12 mapping leaves=106 bulk=229392 scri=864 coordinate_mismatches=0'
            in (args.geometry/'run.log').read_text(), 'geometry helper did not complete teardown')
    helper_source = Path(__file__).with_name('n12_checkpoint_map.cpp')
    recorded = [v for p,v in proof['sources'].items() if p.endswith('/n12_checkpoint_map.cpp')]
    require(recorded == [sha256(helper_source)], 'independent geometry source differs from reviewed helper')
    require(any('sycl' in arg.lower() for arg in proof['compile_command']), 'geometry helper did not use actual SYCL build flags')
    require((args.geometry/'topology.bin').read_bytes() == template['topology'] and
            (args.geometry/'atlas.txt').read_bytes() == template['atlas'], 'Aurora geometry topology/atlas differs')
    arrays = read_geometry(args.geometry)
    for region in ('bulk', 'surface'):
        require(np.array_equal(arrays[region+'_keys'], cpu[region+'_keys']), 'structural mapping changed across platforms')
    bulk, bulk_hashes = native_coordinates(args.native_bulk_coordinates, arrays)
    scri, scri_hashes = surface_geometry(args.native_scri_trace, arrays)
    for name, actual in [('bulk', bulk), ('surface', scri)]:
        require(np.array_equal(actual.view('<u8'), arrays[name+'_xyz'].view('<u8')),
                'Aurora independent/actual '+name+' coordinates differ')
    state = decode(template, arrays)
    initial_checks = {}
    for region, name, count in [('bulk', 'initial-fields.bin', NBULK),
                                ('surface', 'initial-boundary.bin', NSURFACE)]:
        require((args.geometry/name).stat().st_size == count*NF*8,
                'independent initial '+region+' field byte length differs')
        expected = np.fromfile(args.geometry/name, dtype='<f8').reshape(count, NF)
        actual = state[region][:, :NF].copy()
        checked = compare_fields(actual, expected, arrays[region+'_keys'],
                                 arrays[region+'_xyz'], arrays[region+'_weight'], 'short')
        different = actual.view('<u8') != expected.view('<u8')
        zero_sign = different & (actual == 0) & (expected == 0)
        checked['bitwise_differences'] = int(different.sum())
        checked['signed_zero_differences'] = int(zero_sign.sum())
        checked['numerically_different_values'] = int((actual != expected).sum())
        for field in checked['fields']:
            f=field['field']
            field['bitwise_differences']=int(different[:,f].sum())
            field['signed_zero_differences']=int(zero_sign[:,f].sum())
        initial_checks[region] = checked
        require(checked['failed_values'] == 0,
                'independent Aurora initial '+region+' fields fail adopted short gates')
    phase_scope = ('independent host SphericalTransfer3D raw constructor fields versus '
                   'native accepted initial fields after algebraic constraints and scri projection; '
                   'this extra check uses unchanged adopted short/chi gates, while geometry and '
                   'the matched original-GPU initial-state proof remain bitwise exact')
    prior_failure = None
    if args.preserved_extra_check_failure:
        prior_failure = dict(path=str(args.preserved_extra_check_failure.resolve()),
                             sha256=sha256(args.preserved_extra_check_failure))
        require(json.loads(args.preserved_extra_check_failure.read_text())['passed'] is False,
                'prior extra-check failure was not preserved')
    args.output.mkdir(parents=True)
    initial_report = dict(phase_scope=phase_scope, gate='adopted short',
                          regions=initial_checks, preserved_extra_bitwise_failure=prior_failure)
    (args.output/'initial-constructor-comparison.json').write_text(json.dumps(initial_report,indent=2)+'\n')
    np.savez_compressed(args.output/'mapping.npz', **arrays)
    bound = dict(manifest, mapping_sha256=sha256(args.output/'mapping.npz'),
        physics_identity=template['identity'].decode(), fixed_mesh_header_hex=template['fixed'][:-20].hex(),
        template_checkpoint=str(args.template_checkpoint.resolve()), template_checkpoint_sha256=digest,
        coordinate_mismatches=0, coordinate_comparison='bitwise-float64',
        platform='Aurora frozen SYCL build',
        coordinate_proof_scope='independent native/legacy Aurora helper, actual native bulk replay and scri trace all bitwise equal',
        cpu_parent_mapping_sha256=manifest['mapping_sha256'],
        preserved_cpu_physics_identity_differences=identity_difference(manifest['physics_identity'], template['identity'].decode()),
        initial_constructor_comparison_sha256=sha256(args.output/'initial-constructor-comparison.json'),
        initial_constructor_phase_scope=phase_scope,
        provenance=dict(geometry_build=proof, matched_gpu_result_sha256=sha256(args.matched_gpu_result),
            geometry_execution=execution,
            geometry_execution_evidence_sha256=sha256(args.geometry_execution_evidence),
            native_run_evidence_sha256=sha256(args.native_run_evidence),
            cpu_parent_manifest_sha256=sha256(args.cpu_mapping/'manifest.json'),
            geometry_artifacts={name:sha256(args.geometry/name) for name in
                ('mapping.bin','topology.bin','atlas.txt','initial-fields.bin','initial-boundary.bin')},
            actual_bulk_coordinate_sha256=bulk_hashes, actual_scri_coordinate_sha256=scri_hashes,
            binding_source_sha256={str(Path(__file__).with_name(name).resolve()):sha256(Path(__file__).with_name(name))
                for name in ('n12_checkpoint_bind_aurora.py','n12_checkpoint.py',
                             'n12_checkpoint_cross_platform.py','n12_checkpoint_gpu_control.py')}))
    (args.output/'manifest.json').write_text(json.dumps(bound, indent=2)+'\n')
    initial = args.output/'initial-reference'; initial.mkdir()
    for source, target in [('initial-fields.bin','fields.bin'),('initial-boundary.bin','boundary.bin')]:
        shutil.copyfile(args.geometry/source, initial/target)
    (initial/'time.json').write_text(json.dumps(dict(steps=0,t=0.,t_M=0.,dt=template['dt']))+'\n')
    (initial/'reference-manifest.json').write_text(json.dumps(dict(
        mapping_sha256=bound['mapping_sha256'], scope='independent raw host legacy constructor fields from actual Aurora geometry helper; preprojection phase',
        sha256={name:sha256(initial/name) for name in ('fields.bin','boundary.bin')}),indent=2)+'\n')
    validate_header(template, bound, 0); load_mapping(args.output)
    result = dict(schema='native-n12-aurora-map-binding-v1', passed=True,
        platform=bound['platform'], mapping_sha256=bound['mapping_sha256'],
        manifest_sha256=sha256(args.output/'manifest.json'),
        template_checkpoint_sha256=digest, template_cycle=0, template_time=0.,
        physics_identity=bound['physics_identity'],
        physics_identity_sha256=hashlib.sha256(template['identity']).hexdigest(),
        native_executable_sha256=proof['native_executable_sha256'],
        helper_executable_sha256=proof['executable_sha256'],
        helper_exit_status=execution['exit_status'], helper_ranks=execution['ranks'],
        bulk_points=NBULK, scri_points=NSURFACE,
        bulk_coordinate_comparisons=3*NBULK, scri_coordinate_comparisons=3*NSURFACE,
        bulk_coordinate_mismatches=0, scri_coordinate_mismatches=0,
        coordinate_comparison='bitwise-float64 actual cached native geometry versus independent Aurora legacy helper',
        independent_initial_fields_bitwise_equal=all(v['bitwise_equal'] for v in initial_checks.values()),
        independent_initial_fields_passed_adopted_short_gates=True,
        initial_constructor_phase_scope=phase_scope,
        initial_constructor_comparison_sha256=bound['initial_constructor_comparison_sha256'],
        initial_constructor_comparison={region:{key:checked[key] for key in
            ('comparisons','failed_values','bitwise_equal','bitwise_differences',
             'signed_zero_differences','numerically_different_values','max_absolute',
             'max_tolerance_ratio','weighted_rms')} for region,checked in initial_checks.items()},
        preserved_extra_bitwise_failure=prior_failure,
        provenance=bound['provenance'], cpu_parent_sidecar_preserved=True)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('passed','mapping_sha256',
        'bulk_coordinate_comparisons','scri_coordinate_comparisons')}))


if __name__ == '__main__':
    main()
