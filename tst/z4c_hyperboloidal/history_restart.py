#!/usr/bin/env python3
"""Restart native diagnostics at the same accepted time on another decomposition."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from restart_roundtrip import compare, parse, run

LABELS = ['time', 'dt', 'Volume', 'H-rms', 'M-rms', 'Theta-rms', 'Z-rms',
          'eVolume', 'eH-rms', 'eM-rms', 'eTheta-rms', 'eZ-rms']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    values = [[float(x) for x in line.split()] for line in path.read_text().splitlines()
              if line and not line.startswith('#')]
    assert values and all(math.isfinite(x) for row in values for x in row)
    return values


def build_cache_test(root, build, output):
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    obj, exe = output/'history_restart_cache.o', output/'history_restart_cache'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                       '-c', str(root/'tst/z4c_hyperboloidal/history_restart_cache.cpp'),
                       '-o', str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if token == 'CMakeFiles/athena.dir/main.cpp.o' else token for token in link]
    assert str(obj) in link
    link[link.index('-o')+1] = str(exe)
    makefile = output/'history_restart_cache.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restart-ranks', type=int, default=8)
    parser.add_argument('--cache-ranks', type=int, nargs='+', default=[2, 8])
    args = parser.parse_args()
    binary, source_run, output = args.binary.resolve(), args.source_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    build = binary.parent.parent
    source_paths = sorted((source_run/'rst').glob('*.rst'))
    source_info = parse(source_paths[-1])
    assert source_info['cycle'] == 1 and source_info['metadata'][4] == 2
    source_history = source_run/'history_lifecycle.z4c_hyperboloidal.hst'
    source_horizon = source_run/'history_lifecycle.horizon_summary_0.txt'
    reference_history = rows(source_history)[-1]
    reference_horizon = rows(source_horizon)[-1]
    assert reference_history[:2] == [source_info['time'], source_info['dt']]
    assert reference_horizon[0] == source_info['cycle']
    source_files = [Path(source_info['path']), source_history, source_horizon]
    input_hashes = {str(path): sha256(path) for path in source_files}
    names = [name for name in subprocess.check_output(['rg', '--files', 'src'], cwd=root,
             text=True).splitlines() if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
    names += ['CMakeLists.txt', 'tst/z4c_hyperboloidal/history_restart.py',
              'tst/z4c_hyperboloidal/history_restart_cache.cpp',
              'tst/z4c_hyperboloidal/history_restart_contract.md',
              'tst/z4c_hyperboloidal/restart_roundtrip.py',
              'tst/z4c_hyperboloidal/n12_initialize.athinput']
    sources = {name: sha256(root/name) for name in sorted(names)}
    link_path = binary.parent/'CMakeFiles/athena.dir/link.txt'
    objects = {str(binary.parent/token): sha256(binary.parent/token)
               for token in shlex.split(link_path.read_text()) if token.endswith('.o')}
    binary_hash = sha256(binary)
    source_result = json.loads((source_run.parent/'results.json').read_text())
    assert source_result['passed'] and source_result['executable_sha256'] == binary_hash
    (output/'provenance.json').write_text(json.dumps(dict(binary=str(binary),
        binary_sha256=binary_hash, sources=sources, objects=objects,
        source_inputs=input_hashes), indent=2)+'\n')
    for name in ('history_restart.py', 'history_restart_cache.cpp',
                 'history_restart_contract.md', 'restart_roundtrip.py'):
        shutil.copyfile(root/'tst/z4c_hyperboloidal'/name, output/name)
    result = dict(passed=False, scope='native process restart and same-object cache, local CPU MPI')
    try:
        options = ['-r', source_info['path'], 'job/basename=history_restart',
                   'time/evolution=dynamic', 'time/integrator=rk4_classical', 'time/nlim=1',
                   'output1/file_type=hst', 'output1/dcycle=1', 'output1/file_number=0',
                   'output1/data_format=%.17e', 'output2/file_type=rst',
                   'output2/dcycle=1', 'output2/file_number=0',
                   'output2/single_file_per_rank=false']
        candidate = run(binary, output, 'restart', args.restart_ranks, options)
        equality = compare(source_info, candidate)
        assert source_info['costs'] == candidate['costs']
        assert candidate['metadata'][4] == args.restart_ranks
        assert len(list((output/'restart/rst').glob('*.rst'))) == 1
        hst = output/'restart/history_restart.z4c_hyperboloidal.hst'
        headers = [line for line in hst.read_text().splitlines() if line.startswith('#')]
        assert len(headers) == 2 and re.findall(r'\[\d+\]=(\S+)', headers[1]) == LABELS
        history = rows(hst)
        assert len(history) == 1 and history[0][:2] == reference_history[:2]
        errors = [abs(a-b) for a, b in zip(history[0][2:], reference_history[2:])]
        assert all(error <= 1e-12+5e-10*abs(ref)
                   for error, ref in zip(errors, reference_history[2:]))
        horizon = rows(output/'restart/history_restart.horizon_summary_0.txt')
        assert len(horizon) == 1 and len(horizon[0]) == len(reference_horizon) == 12
        assert horizon[0][:2] == reference_horizon[:2]
        horizon_report = {}
        for label, index, scale, relative in [
                ('mass_M', 2, 3, 1e-7), ('spin_x_M2', 3, 9, 1e-7),
                ('spin_y_M2', 4, 9, 1e-7), ('spin_z_M2', 5, 9, 1e-7),
                ('spin_M2', 6, 9, 1e-7), ('area_M2', 7, 9, 2e-7),
                ('mean_radius_M', 10, 3, 1e-7), ('minimum_radius_M', 11, 3, 1e-7)]:
            ref, value = reference_horizon[index]*scale, horizon[0][index]*scale
            error, gate = abs(value-ref), 1e-10+relative*abs(ref)
            assert error <= gate, (label, error, gate)
            horizon_report[label] = dict(reference=ref, candidate=value,
                                         absolute_difference=error, gate=gate)
        shapes = (output/'restart/history_restart.horizon_shape_0.txt').read_text().splitlines()
        assert len([line for line in shapes if line.startswith('# iter')]) == 1
        coefficients = rows(output/'restart/history_restart.horizon_shape_0.txt')
        assert len(coefficients) == 1 and len(coefficients[0]) == 169
        result['process_restart'] = dict(ranks=args.restart_ranks,
            command=['mpiexec', '-n', str(args.restart_ranks), str(binary), *options],
            state=equality, history_reference=reference_history, history_candidate=history[0],
            history_max_absolute=max(errors), horizon=horizon_report,
            expansion_rms_reference=math.sqrt(reference_horizon[8]),
            expansion_rms_candidate=math.sqrt(horizon[0][8]),
            integrated_expansion_reference=reference_horizon[9],
            integrated_expansion_candidate=horizon[0][9])
        cache_exe = build_cache_test(root, build, output)
        cache_runs = {}
        for ranks in args.cache_ranks:
            directory = output/f'cache-ranks{ranks}';directory.mkdir()
            command = ['mpiexec', '-n', str(ranks), str(cache_exe),
                       str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
            with (directory/'run.log').open('w') as log:
                completed = subprocess.run(command, cwd=directory, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=1200)
            completed.check_returncode()
            assert 'PASS same-object restart diagnostic cache' in (directory/'run.log').read_text()
            cache_runs[str(ranks)] = dict(passed=True, command=command)
        result.update(passed=True, same_object_cache=cache_runs,
                      native_executable_sha256=binary_hash, cache_executable_sha256=sha256(cache_exe))
    finally:
        stable = dict(binary_unchanged=sha256(binary) == binary_hash,
                      sources_unchanged=all(sha256(root/name) == digest
                                            for name, digest in sources.items()),
                      objects_unchanged=all(sha256(Path(name)) == digest
                                            for name, digest in objects.items()),
                      source_checkpoint_and_outputs_unchanged=all(sha256(Path(name)) == digest
                                            for name, digest in input_hashes.items()))
        result['provenance'] = stable
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(stable.values()), stable
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
