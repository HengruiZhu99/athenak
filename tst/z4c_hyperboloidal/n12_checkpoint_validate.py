#!/usr/bin/env python3
"""Validate offline N12 decoding against real native writer and known CPU fixtures."""
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess

import numpy as np
from n12_checkpoint import (DT, HEADER, NBULK, NSURFACE, NBLOCK, NSLOT, NF, NR,
    compare_fields, decode, execute, expected_time, load_mapping, parse_native, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 8])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build, mapping_dir, trajectory, output = [p.resolve() for p in
        (args.build, args.mapping, args.trajectory, args.output)]
    output.mkdir(parents=True, exist_ok=False)
    manifest, mapping = load_mapping(mapping_dir)
    oracle = json.loads((trajectory/'results.json').read_text())
    assert oracle['passed'] and oracle['chi_records']['8']['failed_values'] == 0
    names = ['native-step8-bulk.bin', 'native-step8-surface.bin', 'native-step8-chi.bin',
             'native-step8-surface-chi.bin', 'reference-step8-bulk.bin', 'reference-step8-surface.bin']
    for name in names:
        assert sha256(trajectory/name) == oracle['snapshot_sha256'][name]
    tracked = [p for p in (root/'src').rglob('*') if p.suffix in ('.cpp', '.hpp')]
    tracked += [root/'tst/z4c_hyperboloidal'/name for name in
        ('n12_checkpoint.py', 'n12_checkpoint_validate.py', 'n12_checkpoint_fixture.cpp',
         'n12_checkpoint_contract.md', 'n12_initialize.athinput')]
    tracked += [trajectory/name for name in names]+[trajectory/'results.json',
                mapping_dir/'manifest.json', mapping_dir/'mapping.npz']
    sources = {str(path): sha256(path) for path in tracked}
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    obj, exe = output/'fixture.o', output/'fixture'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
        '-c', str(root/'tst/z4c_hyperboloidal/n12_checkpoint_fixture.cpp'), '-o', str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if token == 'CMakeFiles/athena.dir/main.cpp.o' else token for token in link]
    assert str(obj) in link
    link[link.index('-o')+1] = str(exe)
    objects = {str((build/'src'/token).resolve()): sha256(build/'src'/token)
               for token in link if token.endswith('.o') and token != str(obj)}
    (output/'provenance.json').write_text(json.dumps(dict(sources=sources, objects=objects,
        fixture_compile=compile_command, fixture_link=link), indent=2)+'\n')
    makefile = output/'fixture.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src',
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    reference = output/'eight-reference';reference.mkdir()
    for region, count, source_name, target_name in [
            ('bulk', NBULK, 'reference-step8-bulk.bin', 'fields.bin'),
            ('surface', NSURFACE, 'reference-step8-surface.bin', 'boundary.bin')]:
        values = np.memmap(trajectory/source_name, mode='r', dtype='<f8', shape=(count, NF))
        values[mapping[region+'_keys'][:, 5]].tofile(reference/target_name)
    (reference/'time.json').write_text(json.dumps(dict(steps=8, t=expected_time(8), dt=DT))+'\n')
    (reference/'reference-manifest.json').write_text(json.dumps(dict(
        mapping_sha256=manifest['mapping_sha256'],
        scope='independent legacy CPU eight-step trajectory values, reordered by the proven map',
        original_oracle_result_sha256=sha256(trajectory/'results.json'),
        sha256={name: sha256(reference/name) for name in ('fields.bin', 'boundary.bin')}), indent=2)+'\n')
    result = dict(passed=False, scope='format/mapping validation using imported known trajectory; no new evolution')
    reports, negatives, shared_rank_controls = {}, {}, {}
    try:
        checkpoints = {}
        for ranks in args.ranks:
            folder = output/f'writer-ranks{ranks}';folder.mkdir()
            command = ['mpiexec', '-n', str(ranks), str(exe),
                       str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'), str(trajectory)]
            with (folder/'run.log').open('w') as log:
                subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoint = folder/'rst/trajectory_fixture.00000.rst';checkpoints[ranks] = checkpoint
            report = execute(checkpoint, mapping_dir, output/f'compare-ranks{ranks}', 8, reference)
            assert report['passed']
            decoded = decode(parse_native(checkpoint), mapping)
            for region, count, state_name, chi_name in [
                    ('bulk', NBULK, 'native-step8-bulk.bin', 'native-step8-chi.bin'),
                    ('surface', NSURFACE, 'native-step8-surface.bin', 'native-step8-surface-chi.bin')]:
                order = mapping[region+'_keys'][:, 5]
                expected_fields = np.memmap(trajectory/state_name, mode='r', dtype='<f8', shape=(count, NF))
                expected_records = np.memmap(trajectory/chi_name, mode='r', dtype='<f8', shape=(count, NR))
                assert np.array_equal(decoded[region][:, :NF].view('<u8'), expected_fields[order].view('<u8'))
                assert np.array_equal(decoded[region][:, NF:].view('<u8'), expected_records[order].view('<u8'))
            reports[str(ranks)] = dict(command=command, complete_fields_and_records_roundtrip_exact=True,
                bulk_failed_values=report['fields']['bulk']['failed_values'],
                surface_failed_values=report['fields']['surface']['failed_values'],
                exported_sha256=report['export_sha256'])
        first = checkpoints[args.ranks[0]]
        scratch = output/'mutated.rst';shutil.copyfile(first, scratch)
        info = parse_native(scratch)

        def bytes_at(offset, value):
            with scratch.open('r+b') as stream:
                stream.seek(offset);old = stream.read(len(value));stream.seek(offset);stream.write(value)
            return old

        def field_offset(region, ordinal, field):
            gid, slot = mapping[region+'_keys'][ordinal, :2]
            return info['offset']+int(gid)*info['stride']+(int(slot)*(NF+NR)+field)*8

        def rejection(name, offset, value, message):
            original = bytes_at(offset, value)
            try:
                try:
                    execute(scratch, mapping_dir, output/name, 8, reference)
                except ValueError as error:
                    assert message in str(error);negatives[name] = str(error)
                else:
                    raise AssertionError(name+' was not rejected')
            finally:
                bytes_at(offset, original)

        indices = mapping['bulk_keys'][:, 2:5]
        extra = int(np.flatnonzero(((indices < 0) | (indices >= 12)).any(axis=1))[0])
        assert mapping['bulk_diagnostic_weight'][extra] == 0
        for name, region, ordinal, field in [
                ('last-scri-field', 'surface', NSURFACE-1, 30),
                ('zero-diagnostic-weight', 'bulk', extra, 29)]:
            offset = field_offset(region, ordinal, field)
            original = bytes_at(offset, struct.pack('<d', 1e-8))
            try:
                failed = execute(scratch, mapping_dir, output/name, 8, reference)
                assert not failed['passed'] and failed['fields'][region]['failed_values'] == 1
                assert failed['fields'][region]['worst_absolute']['field'] == field
                negatives[name] = dict(detected=True, failed_values=1)
            finally:
                bytes_at(offset, original)
        expected = np.memmap(reference/'fields.bin', mode='r', dtype='<f8', shape=(NBULK, NF))
        ordinal = int(expected[:, 0].argmin());chi = float(expected[ordinal, 0])
        tolerance = 1e-14+5e-10*abs(chi);offset = field_offset('bulk', ordinal, 0)
        for name, factor, should_pass in [('chi-inside-gate', .25, True), ('chi-outside-gate', 2., False)]:
            original = bytes_at(offset, struct.pack('<d', chi+factor*tolerance))
            try:
                checked = execute(scratch, mapping_dir, output/name, 8, reference)
                assert checked['passed'] == should_pass
                negatives[name] = dict(correct=True, candidate_passed=checked['passed'])
            finally:
                bytes_at(offset, original)
        rejection('nonfinite-evolved-extra', field_offset('bulk', extra, 30),
                  struct.pack('<d', float('nan')), 'nonfinite')
        rejection('wrong-dt', info['start']+HEADER-12, struct.pack('<d', DT/2), 'step/time/dt')
        metadata_offset = info['start']+HEADER+8+len(info['atlas'])+NBLOCK*24
        rejection('wrong-identity', metadata_offset+64, b'X', 'numerical policy identity')
        topology_offset = info['start']+HEADER+8+len(info['atlas'])
        rejection('wrong-topology', topology_offset+16, struct.pack('<i', 6), 'leaf order/topology')
        with scratch.open('r+b') as stream:
            stream.seek(-1, 2);last = stream.read(1);stream.truncate(info['size']-1)
        try:
            try:
                parse_native(scratch)
            except ValueError as error:
                negatives['truncated-payload'] = str(error)
            else:
                raise AssertionError('truncated checkpoint accepted')
        finally:
            with scratch.open('ab') as stream:
                stream.write(last)
        for ranks in (24, 96):
            old_ranks = bytes_at(metadata_offset+4*8, struct.pack('<Q', ranks))
            old_count = bytes_at(metadata_offset+6*8, struct.pack('<Q', NBLOCK//ranks))
            try:
                checked = execute(scratch, mapping_dir, output/f'writer-metadata-{ranks}', 8, reference)
                assert checked['passed'] and checked['writer_mpi_ranks'] == ranks
                shared_rank_controls[str(ranks)] = dict(passed=True, scope='synthetic shared-file metadata only; not a real MPI run')
            finally:
                bytes_at(metadata_offset+4*8, old_ranks);bytes_at(metadata_offset+6*8, old_count)
        # Direct long-gate guard: uses known values only to test comparator math,
        # not an evolved native3551 state or invented per-cell reference records.
        a = np.ones((2, NF));a[:, 0] = 1e-10
        b = a.copy();b[1, 0] += 2*(1e-13+1e-8*1e-10)
        long_metric = compare_fields(b, a, np.zeros((2, 6), dtype=int), np.zeros((2, 3)), np.ones(2), 'long')
        assert long_metric['failed_values'] == 1 and long_metric['fields'][0]['failed_values'] == 1
        negatives['long-chi-gate-arithmetic'] = dict(detected=True, native3551_evolution=False)
        assert sha256(scratch) == sha256(first), 'negative tests did not restore the fixture'
        scratch.unlink()
        result.update(passed=True, eight_step_roundtrip=reports, negative_controls=negatives,
                      shared_writer_metadata_controls=shared_rank_controls,
                      reference_per_cell_chi_history_3551_available=False,
                      fixture_executable_sha256=sha256(exe))
    finally:
        result['sources_objects_unchanged'] = all(sha256(Path(name)) == digest
                                                 for group in (sources, objects) for name, digest in group.items())
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['sources_objects_unchanged']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
