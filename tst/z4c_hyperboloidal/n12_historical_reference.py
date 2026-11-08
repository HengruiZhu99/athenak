#!/usr/bin/env python3
"""Bind preserved historical N12 snapshots to the exact Aurora comparison map.

This reads copied historical data only. Every field snapshot is checked against
its SP1 checkpoint; the final snapshot additionally has independently pinned
hashes. No evolution, projection, interpolation or tolerance change is involved.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from n12_checkpoint import (DT, NBULK, NSURFACE, NF, PINNED, expected_time,
                            load_mapping, require, sha256)
from n12_checkpoint_gpu_control import old_checkpoint, resolved_comparison
from n12_long_history import STEPS


def bind(root, mapping_directory, source_provenance, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'new historical reference output required')
    manifest, _ = load_mapping(mapping_directory)
    source_provenance = Path(source_provenance).resolve()
    # The remote collection record accompanies the copies, rather than silently
    # converting newly computed local checksums into historical provenance.
    provenance = json.loads(source_provenance.read_text())
    require(provenance['schema'] == 'original-n12-common-reference-copy-v1' and
            provenance['source_read_only'] is True and
            provenance['cycles'] == list(STEPS) and provenance['writer_ranks'] == 48,
            'wrong historical source/copy collection provenance')
    require(provenance['source_directory'] ==
            '/lus/flare/projects/CompactBinaryMerger/hzhu/scri-ko01-20261106/'
            'campaign-constant20m/segment-01', 'historical collection source differs')
    output.mkdir(parents=True)
    result = dict(schema='native-n12-historical-reference-v1', passed=False,
                  mapping_sha256=manifest['mapping_sha256'], snapshots={},
                  source_provenance=str(source_provenance),
                  source_provenance_sha256=sha256(source_provenance),
                  source_provenance_scope='every used snapshot matches the reviewed remote before/after source and private-copy hash record',
                  per_cell_historical_chi_records_available=False)
    first_identity = None
    try:
        for step in STEPS:
            source = root / ('initial' if step == 0 else f'step-{step}')
            names = ('fields.bin', 'boundary.bin', 'restart.bin', 'time.json',
                     'chi-policy.json')
            require(all((source/name).is_file() for name in names),
                    'historical snapshot incomplete: '+str(source))
            hashes = {name:sha256(source/name) for name in names}
            for name in names:
                relative = str((source/name).relative_to(root))
                copied = provenance['files'][relative]
                require(copied['sha256'] == hashes[name] and
                        copied['source_size'] == (source/name).stat().st_size and
                        copied['source'] == provenance['source_directory']+'/'+relative,
                        'historical snapshot differs from verified collection: '+relative)
            clock = json.loads((source/'time.json').read_text())
            require(clock['steps'] == step and clock['t'] == expected_time(step),
                    'historical snapshot clock differs')
            policy = json.loads((source/'chi-policy.json').read_text())
            require(policy['ko'] == .1 and policy['chi_floor'] == 1e-10 and
                    policy['time_M'] == 3*expected_time(step),
                    'historical chi policy/time differs')
            require((source/'fields.bin').stat().st_size == NBULK*NF*8 and
                    (source/'boundary.bin').stat().st_size == NSURFACE*NF*8,
                    'historical field coverage differs')
            bulk = np.memmap(source/'fields.bin', dtype='<f8', mode='r', shape=(NBULK,NF))
            surface = np.memmap(source/'boundary.bin', dtype='<f8', mode='r',
                                shape=(NSURFACE,NF))
            checkpoint = old_checkpoint(source/'restart.bin', step, bulk, surface)
            identity = (checkpoint['physics_identity'], checkpoint['topology_identity'])
            if first_identity is None:
                first_identity = identity
            require(identity == first_identity,
                    'historical physics/topology changed between snapshots')
            resolved = resolved_comparison(manifest['physics_identity'], identity[0])
            require(resolved['all_common_exact'],
                    'historical/native common resolved doubles differ')
            if step == 3551:
                for name, digest in PINNED.items():
                    require(sha256(source/name) == digest,
                            'historical endpoint differs from independently pinned '+name)
            target = output/f'step-{step}'
            target.mkdir()
            # Small metadata is copied verbatim; immutable field inputs remain
            # linked to the verified collection so another gigabyte is not made.
            for name in ('fields.bin', 'boundary.bin'):
                (target/name).symlink_to(source/name)
            for name in ('time.json', 'chi-policy.json'):
                shutil.copyfile(source/name, target/name)
            entry = dict(step=step, time=expected_time(step), dt=DT,
                         source=str(source), source_sha256=hashes, checkpoint=checkpoint,
                         common_resolved_policy=resolved,
                         mapping_sha256=manifest['mapping_sha256'],
                         source_provenance_sha256=result['source_provenance_sha256'],
                         sha256={name:sha256(target/name) for name in
                             ('fields.bin', 'boundary.bin', 'time.json', 'chi-policy.json')})
            (target/'reference-manifest.json').write_text(json.dumps(entry,indent=2)+'\n')
            result['snapshots'][str(step)] = entry
        result.update(passed=True, common_steps=list(STEPS),
                      fixed_physics_and_topology_exact=True,
                      all_field_snapshots_match_sp1_bitwise=True,
                      endpoint_matches_independently_pinned_hashes=True)
    except Exception as error:
        result['validation_error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
            sha256(Path(__file__).with_name(name)) for name in
            ('n12_historical_reference.py', 'n12_checkpoint.py',
             'n12_checkpoint_gpu_control.py', 'n12_long_history.py')}
        (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--source-provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = bind(args.root,args.mapping,args.source_provenance,args.output)
    print(json.dumps(dict(passed=result['passed'],common_steps=result['common_steps']),indent=2))


if __name__ == '__main__':
    main()
