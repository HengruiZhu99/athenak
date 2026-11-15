#!/usr/bin/env python3
"""Exercise native AthenaK restart I/O, repartitioning and corruption rejection.

This uses the actual athena executable and its native tree/parameter/MPI-IO
paths. Payload comparisons include all 31 fields and all eight persistent chi
record scalars, with no tolerance. Only local CPU MPI runs are requested.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

HEADER = struct.calcsize('=ii9d38i2di')
MAGIC = 0x4859504552525354


def parse(path):
    with path.open('rb') as file:
        text = bytearray()
        while not text.endswith(b'<par_end>\n'):
            byte = file.read(1)
            if not byte:
                raise ValueError('missing native parameter terminator')
            text += byte
        start = file.tell()
        fixed = file.read(HEADER)
        nmb, = struct.unpack_from('=i', fixed)
        cycle, = struct.unpack_from('=i', fixed, HEADER-4)
        time, dt = struct.unpack_from('=2d', fixed, HEADER-20)
        atlas_size, = struct.unpack('=Q', file.read(8))
        atlas = file.read(atlas_size)
        topology = file.read(nmb*20)
        costs = file.read(nmb*4)
        meta_at = file.tell()
        meta = struct.unpack('=8Q', file.read(64))
        assert meta[0] == MAGIC and meta[1] == 1
        identity = file.read(meta[2])
        stride_at = file.tell()
        stride, = struct.unpack('=Q', file.read(8))
        payload_at = file.tell()
    assert stride == meta[3]*8 and stride % (39*8) == 0
    record_blocks = meta[6] if meta[7] else nmb
    assert path.stat().st_size == payload_at+record_blocks*stride
    return dict(path=str(path), blocks=nmb, record_blocks=record_blocks,
                cycle=cycle, time=time, dt=dt,
                atlas=atlas, topology=topology, costs=costs, metadata=meta,
                identity=identity, meta_at=meta_at, stride_at=stride_at,
                payload_at=payload_at, stride=stride, start=start)


def payload(info, mode='r'):
    return np.memmap(info['path'], dtype='=f8', mode=mode, offset=info['payload_at'],
                     shape=(info['record_blocks'], info['stride']//(39*8), 39))


def compare(a, b):
    for name in ['blocks', 'record_blocks', 'cycle', 'time', 'dt', 'atlas', 'topology', 'identity', 'stride']:
        assert a[name] == b[name], f'restart {name} differs'
    x, y = payload(a), payload(b)
    different = x.view('=u8') != y.view('=u8')
    if different.any():
        index = tuple(np.argwhere(different)[0])
        raise AssertionError(f'payload differs at block/slot/field {index}: {x[index]} vs {y[index]}')
    return dict(bitwise_equal=True, scalar_values=int(x.size),
                payload_sha256=hashlib.sha256(x.tobytes()).hexdigest())


def run(binary, output, name, ranks, options, fail=None):
    directory = output/name
    directory.mkdir()
    command = ['mpiexec', '-n', str(ranks), str(binary), *options]
    with (directory/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1'), timeout=1200)
    text = (directory/'run.log').read_text()
    if fail is not None:
        assert result.returncode != 0 and fail in text, (name, result.returncode, text[-3000:])
        return dict(rejected=True, expected_message=fail, ranks=ranks, command=command)
    if result.returncode:
        raise RuntimeError(f'{name} failed: {text[-5000:]}')
    paths = sorted((directory/'rst').glob('**/*.rst'))
    assert paths, f'{name} did not write a restart'
    return parse(paths[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-ranks', type=int, default=8)
    parser.add_argument('--cases', nargs='+', choices=['n12', 'outer_smr'],
                        default=['n12', 'outer_smr'])
    parser.add_argument('--trajectory', action='store_true',
                        help='also compare 1+restart+8 steps with a continuous 9-step run')
    parser.add_argument('--only-trajectory', action='store_true',
                        help='run only the 1+restart+8 versus continuous9 comparison')
    parser.add_argument('--only-per-rank', action='store_true',
                        help='test coherent per-rank files and collective failure cases')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    source_names = [
        'src/z4c_hyperboloidal/restart.cpp', 'src/z4c_hyperboloidal/restart.hpp',
        'src/outputs/restart.cpp', 'src/outputs/io_wrapper.cpp',
        'src/outputs/io_wrapper.hpp', 'src/outputs/outputs.hpp',
        'src/pgen/pgen.cpp', 'src/pgen/z4c/hyperboloidal_puncture.cpp',
        'src/z4c_hyperboloidal/z4c_hyperboloidal_lifecycle.cpp',
        'src/z4c_hyperboloidal/z4c_hyperboloidal.cpp',
        'src/z4c_hyperboloidal/z4c_hyperboloidal.hpp',
        'tst/z4c_hyperboloidal/restart_roundtrip.py']
    provenance = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      binary_mtime_ns=binary.stat().st_mtime_ns, sources={})
    for name in source_names:
        path = root/name
        entry = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     mtime_ns=path.stat().st_mtime_ns)
        obj = binary.parent/'CMakeFiles/athena.dir'/str(Path(name).relative_to('src')) if name.startswith('src/') else None
        if obj is not None and name.endswith('.cpp'):
            obj = Path(str(obj)+'.o')
            if obj.exists():
                entry['object_mtime_ns'] = obj.stat().st_mtime_ns
                assert obj.stat().st_mtime_ns >= path.stat().st_mtime_ns, f'stale object: {obj}'
        provenance['sources'][name] = entry
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    shutil.copyfile(Path(__file__).resolve(), output/'restart_roundtrip.py')
    results = {}
    for case in ([] if args.only_trajectory or args.only_per_rank else args.cases):
        fixture = 'n12_initialize.athinput' if case == 'n12' else 'surface_smr.athinput'
        input_path = output/(case+'.athinput')
        input_path.write_text((source/fixture).read_text()+
                             '\n<time>\nintegrator = rk4_classical\ntlim = 1\nnlim = 0\nstart_time = 0\n'
                             '<output1>\nfile_type = rst\ndt = 100\nsingle_file_per_rank = false\n')
        original = run(binary, output, case+'_write2', 2, ['-i', str(input_path)])
        enriched = output/(case+'_nonzero_records.rst')
        shutil.copyfile(original['path'], enriched)
        injected = parse(enriched)
        data = payload(injected, 'r+')
        active = data[:, :, 31] != 0  # default record minimum=1e300, padding=0
        # Distinct, exactly representable values check record order and all eight
        # scalar members, including both minima and cumulative correction fields.
        unique = np.arange(active.size, dtype=np.float64).reshape(active.shape)+1
        for field in range(8):
            data[:, :, 31+field][active] = unique[active]+(field+1)/16
        data.flush()
        del data
        restored = run(binary, output, case+'_read'+str(args.target_ranks), args.target_ranks,
                       ['-r', str(enriched), 'output1/last_time=-1'])
        results[case+'/2_to_'+str(args.target_ranks)] = {
            **compare(injected, restored), 'blocks': original['blocks'],
            'active_points': int(active.sum()), 'nonzero_chi_values': int(8*active.sum()),
            'source': str(enriched), 'restored': restored['path']}
        again = run(binary, output, case+'_readback2', 2,
                    ['-r', restored['path'], 'output1/last_time=-1'])
        results[case+'/back_to_2'] = compare(injected, again)
        if case == args.cases[0]:
            restart_options = ['-r', str(enriched), 'output1/last_time=-1']
            results['reject/ko'] = run(binary, output, 'wrong_ko', 2,
                restart_options+['z4c_hyperboloidal/ko=.2'], 'mathematical parameters')
            results['reject/dt'] = run(binary, output, 'wrong_dt', 2,
                restart_options+['time/dt=.0004'], 'mathematical parameters')
            for name, extent, message in [
                    ('metadata', injected['meta_at']+20, 'truncated metadata'),
                    ('identity', injected['stride_at']-5, 'truncated metadata'),
                    ('stride', injected['payload_at']-4, 'truncated metadata'),
                    ('payload', enriched.stat().st_size-13, 'truncated block payload')]:
                bad = output/('truncated_'+name+'.rst')
                shutil.copyfile(enriched, bad)
                with bad.open('r+b') as file:
                    file.truncate(extent)
                results['reject/'+name] = run(binary, output, 'bad_'+name, 2,
                                             ['-r', str(bad)], message)
            bad = output/'overflow_stride.rst'
            shutil.copyfile(enriched, bad)
            with bad.open('r+b') as file:
                file.seek(injected['stride_at'])
                file.write(struct.pack('=Q', 2**64-1))
            results['reject/overflow_stride'] = run(binary, output, 'bad_overflow', 2,
                ['-r', str(bad)], 'block record byte count mismatch')
            bad = output/'nonfinite.rst'
            shutil.copyfile(enriched, bad)
            bad_info = parse(bad)
            data = payload(bad_info, 'r+')
            m, slot = np.argwhere(active)[0]
            data[m, slot, 0] = np.nan
            data.flush()
            del data
            results['reject/nonfinite'] = run(binary, output, 'bad_nonfinite', 2,
                ['-r', str(bad)], 'nonfinite checkpoint state')
        (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    if args.only_per_rank:
        input_path = output/'n12.athinput'
        input_path.write_text((source/'n12_initialize.athinput').read_text()+
                              '\n<time>\nintegrator = rk4_classical\ntlim = 1\nnlim = 0\nstart_time = 0\n'
                             '<output1>\nfile_type = rst\ndt = 100\nsingle_file_per_rank = false\n')
        options = ['-i', str(input_path), 'output1/single_file_per_rank=true']
        original = run(binary, output, 'per_rank_write2', 2, options)
        restored = run(binary, output, 'per_rank_read2', 2,
                       ['-r', original['path'], 'output1/last_time=-1'])
        def parts(info):
            path = Path(info['path'])
            return {p.parent.name: p for p in path.parent.parent.glob('rank_*/'+path.name)}
        originals, restorations = parts(original), parts(restored)
        assert len(originals) == len(restorations) == 2
        results['per_rank/coherent'] = {
            rank: compare(parse(path), parse(restorations[rank]))
            for rank, path in originals.items()}
        results['per_rank/changed_ownership'] = run(binary, output, 'per_rank_changed', 1,
            ['-r', original['path']], 'per-rank restart ownership changed')
        later = run(binary, output, 'per_rank_later', 2, options+['time/start_time=.001'])
        later_parts = parts(later)
        mixed = output/'mixed_files'/'rst'
        for rank, original_path in originals.items():
            target = mixed/rank/Path(original['path']).name
            target.parent.mkdir(parents=True)
            shutil.copyfile(original_path if rank == 'rank_00000000' else later_parts[rank],
                            target)
        results['per_rank/mixed_generations'] = run(binary, output, 'per_rank_mixed', 2,
            ['-r', str(mixed/'rank_00000000'/Path(original['path']).name)],
            'mixed per-rank checkpoint metadata')
        truncated = output/'truncated_rank_files'/'rst'
        for rank, original_path in originals.items():
            target = truncated/rank/original_path.name
            target.parent.mkdir(parents=True)
            shutil.copyfile(original_path, target)
            if rank == 'rank_00000001':
                with target.open('r+b') as file:
                    file.truncate(target.stat().st_size-13)
        results['per_rank/one_truncated_rank'] = run(binary, output, 'per_rank_truncated', 2,
            ['-r', str(truncated/'rank_00000000'/Path(original['path']).name)],
            'truncated block payload')
    if args.trajectory or args.only_trajectory:
        input_path = output/'n12.athinput'
        if not input_path.exists():
            input_path.write_text((source/'n12_initialize.athinput').read_text()+
                                  '\n<time>\nintegrator = rk4_classical\ntlim = 1\nnlim = 0\nstart_time = 0\n'
                             '<output1>\nfile_type = rst\ndt = 100\nsingle_file_per_rank = false\n')
        options = ['-i', str(input_path), 'time/evolution=dynamic',
                   'time/integrator=rk4_classical', 'time/tlim=1']
        continuous = run(binary, output, 'continuous9', 2, options+['time/nlim=9'])
        first = run(binary, output, 'first1', 2, options+['time/nlim=1'])
        continuation = run(binary, output, 'restart_plus8', args.target_ranks,
                           ['-r', first['path'], 'time/nlim=9'])
        results['trajectory/1_plus_8'] = {
            **compare(continuous, continuation), 'continuous': continuous['path'],
            'restart': first['path'], 'continuation': continuation['path']}
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
