#!/usr/bin/env python3
"""Targeted negative controls for the matched-GPU comparison helpers."""
import argparse
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from n12_checkpoint import NF, NSURFACE, NBULK, compare_fields, load_mapping, require, sha256
from n12_checkpoint_gpu_control import compare_chi, native_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--legacy-control', type=Path, required=True)
    parser.add_argument('--native-coordinate-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    _, mapping = load_mapping(args.mapping); passed = {}
    boundary_path = args.legacy_control/'final/boundary.bin'
    reference = np.fromfile(boundary_path, dtype='<f8').reshape(NSURFACE, NF)[-1:].copy()
    actual = reference.copy(); actual[0, 30] += 1e-8
    metric = compare_fields(actual, reference, mapping['surface_keys'][-1:],
        mapping['surface_xyz'][-1:], mapping['surface_weight'][-1:], 'short')
    require(metric['failed_values'] == 1 and metric['worst_absolute']['field'] == 30,
            'last scri field corruption was not detected')
    passed['last_scri_Jz'] = dict(failed_values=1, injected_absolute=1e-8)
    policy_path = args.legacy_control/'final/chi-policy.json'
    policy = json.loads(policy_path.read_text())
    summary = {k:v for k,v in policy.items() if k not in ('ko', 'chi_floor', 'time_M')}
    require(compare_chi(summary, policy, 8)['passed'], 'unaltered aggregate helper baseline failed')
    for key in ('floor_events', 'surface_floor_events', 'nonpositive_candidates', 'max_correction'):
        changed = dict(summary); changed[key] += 1
        result = compare_chi(changed, policy, 8)
        require(not result['passed'] and sum(not x['passed'] for x in result['fields']) == 1,
                'changed chi aggregate was not detected')
        passed['chi_'+key] = 'one changed aggregate rejected'
    source = sorted(args.native_coordinate_directory.glob('points-rank*.bin'))[0]
    for name, byte_offset, replacement, expected in [
        ('wrong_native_gid', 36, -1, 'native coordinate keys differ'),
        ('out_of_range_legacy_ordinal', 32, NBULK, 'invalid coordinate ordinal')]:
        folder = args.output/name; folder.mkdir()
        target = folder/source.name; shutil.copyfile(source, target)
        with target.open('r+b') as stream:
            stream.seek(byte_offset); stream.write(struct.pack('<i', replacement))
        try:
            native_coordinates(folder, mapping)
        except ValueError as error:
            require(expected in str(error), 'wrong coordinate rejection: '+str(error))
            passed[name] = str(error)
        else:
            raise AssertionError('malformed coordinate membership accepted')
        target.unlink()
    report = dict(passed=True, scope='helper negative tests; no repeat evolution or valid matched-GPU baseline',
        controls=passed, source_sha256={str(Path(__file__).with_name(name).resolve()):
            sha256(Path(__file__).with_name(name)) for name in
            ('n12_checkpoint_gpu_controls.py', 'n12_checkpoint_gpu_control.py',
             'n12_checkpoint.py', 'n12_checkpoint_cross_platform.py')},
        reference_sha256={str(path): sha256(path) for path in (boundary_path, policy_path, source)})
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, negative_controls=len(passed))))


if __name__ == '__main__':
    main()
