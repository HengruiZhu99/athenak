#!/usr/bin/env python3
"""Characterize the preserved compact-radius restart failure and test Unpack caching.

This does not replace or relax history_restart_contract.md. The same-rank
control separates a new FastFlow search from a change of MPI decomposition.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

from history_restart import build_cache_test, rows, sha256
from restart_roundtrip import compare, parse, run


def parameters(path, start):
    text = path.read_bytes()[:start].decode()
    values, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line and block == 'fastflow':
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--restart-eight-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, source_run = args.binary.resolve(), args.source_run.resolve()
    eight_run, output = args.restart_eight_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root, build = Path(__file__).resolve().parents[2], binary.parent.parent
    files = [root/name for name in subprocess.check_output(['rg', '--files', 'src'],
             cwd=root, text=True).splitlines() if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
    files += [Path(__file__).resolve(), root/'tst/z4c_hyperboloidal/history_restart.py',
              root/'tst/z4c_hyperboloidal/history_restart_cache.cpp',
              root/'tst/z4c_hyperboloidal/history_restart_contract.md']
    files += [binary]+[binary.parent/token for token in shlex.split(
        (binary.parent/'CMakeFiles/athena.dir/link.txt').read_text()) if token.endswith('.o')]
    source_info = parse(sorted((source_run/'rst').glob('*.rst'))[-1])
    source_summary = source_run/'history_lifecycle.horizon_summary_0.txt'
    eight_summary = eight_run/'history_restart.horizon_summary_0.txt'
    files += [Path(source_info['path']), source_summary, eight_summary]
    manifest = {str(path): sha256(path) for path in files}
    (output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'history_restart_controls.py').write_text(Path(__file__).read_text())
    result = dict(completed=False, original_compact_radius_gate_relaxed=False)
    try:
        options = ['-r', source_info['path'], 'job/basename=history_restart',
                   'time/nlim=1', 'fastflow/verbose=true',
                   'output1/dcycle=1', 'output1/file_number=0',
                   'output1/data_format=%.17e', 'output2/dcycle=1', 'output2/file_number=0']
        candidate = run(binary, output, 'restart-two', 2, options)
        reference = rows(source_summary)[-1]
        two = rows(output/'restart-two/history_restart.horizon_summary_0.txt')[0]
        eight = rows(eight_summary)[0]
        rank_errors = [abs(a-b) for a, b in zip(two, eight)]
        assert all(error <= 1e-12+5e-10*abs(reference)
                   for error, reference in zip(rank_errors, eight))
        native_state = compare(source_info, candidate)
        areal_reference = 3*math.sqrt(reference[7]/(4*math.pi))
        areal_candidate = 3*math.sqrt(two[7]/(4*math.pi))
        areal_error = abs(areal_candidate-areal_reference)
        areal_gate = 1e-10+1e-7*abs(areal_reference)
        assert areal_error <= areal_gate
        verbose = (output/'restart-two/history_restart.horizon_verbose_0.txt').read_text()
        iterations = [line.split() for line in verbose.splitlines()
                      if re.match(r'^\s*\d+\s', line)]
        assert iterations
        result['restart_control'] = dict(command=['mpiexec', '-n', '2', str(binary), *options],
            complete_state=native_state, fresh_two_vs_fresh_eight_summary_exact=two == eight,
            fresh_two_vs_fresh_eight_max_absolute=max(rank_errors),
            source_summary=reference, fresh_summary=two,
            adopted_areal_radius_M=dict(reference=areal_reference, candidate=areal_candidate,
                absolute_difference=areal_error, gate=areal_gate, passed=True),
            actual_expansion_rms_reference=math.sqrt(reference[8]),
            actual_expansion_rms_candidate=math.sqrt(two[8]),
            raw_hrms_is_mean_square=True, fresh_iteration_count=len(iterations),
            fresh_last_iteration=int(iterations[-1][0]),
            source_fastflow_parameters=parameters(Path(source_info['path']), source_info['start']),
            candidate_fastflow_parameters=parameters(Path(candidate['path']), candidate['start']))
        result['additional_compact_radius_gates'] = {}
        for name, index in [('mean', 10), ('minimum', 11)]:
            reference_radius, candidate_radius = 3*reference[index], 3*two[index]
            error, gate = abs(candidate_radius-reference_radius), 1e-10+1e-7*abs(reference_radius)
            result['additional_compact_radius_gates'][name] = dict(
                absolute_difference=error, gate=gate, passed=error <= gate)
        exe = build_cache_test(root, build, output)
        cache = {}
        for ranks in (2, 8):
            folder = output/f'cache-ranks{ranks}';folder.mkdir()
            command = ['mpiexec', '-n', str(ranks), str(exe),
                       str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
            with (folder/'run.log').open('w') as log:
                subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
            assert 'PASS same-object restart diagnostic cache' in (folder/'run.log').read_text()
            cache[str(ranks)] = dict(passed=True, command=command)
        result.update(completed=True, same_object_cache=cache,
                      cache_executable_sha256=sha256(exe))
    finally:
        result['all_provenance_unchanged'] = all(sha256(Path(name)) == digest
                                                for name, digest in manifest.items())
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['all_provenance_unchanged']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
