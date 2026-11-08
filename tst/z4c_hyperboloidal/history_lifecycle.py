#!/usr/bin/env python3
"""Check native executable history cadence and complete accepted-state preservation."""
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

DT = 0.00046944203735617445
LABELS = ['time', 'dt', 'Volume', 'H-rms', 'M-rms', 'Theta-rms', 'Z-rms',
          'eVolume', 'eH-rms', 'eM-rms', 'eTheta-rms', 'eZ-rms']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_history(path):
    lines = path.read_text().splitlines()
    headers = [line for line in lines if line.startswith('#')]
    assert len(headers) == 2
    assert re.findall(r'\[\d+\]=(\S+)', headers[1]) == LABELS
    rows = [[float(value) for value in line.split()] for line in lines
            if line and not line.startswith('#')]
    assert len(rows) == 3, 'expected initial, accepted-cycle and final output'
    assert all(len(row) == len(LABELS) and all(math.isfinite(x) for x in row)
               for row in rows)
    assert [row[:2] for row in rows] == [[0., DT], [DT, DT], [DT, DT]]
    assert rows[1] == rows[2], 'final duplicate changed accepted-state diagnostics'
    assert all(row[2] > 0 and row[7] > 0 for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 8])
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    names = [name for name in subprocess.check_output(['rg', '--files', 'src'], cwd=root,
             text=True).splitlines() if name.endswith(('.cpp', '.hpp', 'CMakeLists.txt'))]
    names += ['CMakeLists.txt', 'tst/z4c_hyperboloidal/history_lifecycle.py',
              'tst/z4c_hyperboloidal/history_lifecycle_contract.md',
              'tst/z4c_hyperboloidal/restart_roundtrip.py',
              'tst/z4c_hyperboloidal/n12_initialize.athinput']
    sources = {name: sha256(root/name) for name in sorted(names)}
    link_path = binary.parent/'CMakeFiles/athena.dir/link.txt'
    objects = {str(binary.parent/token): sha256(binary.parent/token)
               for token in shlex.split(link_path.read_text()) if token.endswith('.o')}
    binary_hash = sha256(binary)
    (output/'provenance.json').write_text(json.dumps(dict(binary=str(binary),
        binary_sha256=binary_hash, link_sha256=sha256(link_path), sources=sources,
        objects=objects), indent=2)+'\n')
    for name in ('history_lifecycle.py', 'history_lifecycle_contract.md',
                 'restart_roundtrip.py'):
        shutil.copyfile(source/name, output/name)
    input_path = output/'history_lifecycle.athinput'
    input_path.write_text((source/'n12_initialize.athinput').read_text()+'''
<job>
basename = history_lifecycle
<time>
evolution = dynamic
integrator = rk4_classical
tlim = 1
nlim = 1
ndiag = 1
<output1>
file_type = hst
dcycle = 1
data_format = %.17e
<output2>
file_type = rst
dcycle = 1
single_file_per_rank = false
''')
    runs, checkpoints, commands = {}, {}, {}
    result = dict(passed=False, scope='actual executable native history lifecycle, local CPU MPI')
    try:
        for ranks in args.ranks:
            reports = {}
            for enabled in (False, True):
                name = f'{"history" if enabled else "control"}-ranks{ranks}'
                options = ['-i', str(input_path)]
                if not enabled:
                    options += ['output1/dcycle=0']
                commands[name] = ['mpiexec', '-n', str(ranks), str(binary), *options]
                last = run(binary, output, name, ranks, options)
                assert last['cycle'] == 1 and last['time'] == DT and last['dt'] == DT
                paths = sorted((output/name/'rst').glob('*.rst'))
                assert len(paths) == 3
                by_cycle = {}
                for path in paths:
                    info = parse(path)
                    assert info['blocks'] == 106
                    if info['cycle'] in by_cycle:
                        compare(by_cycle[info['cycle']], info)
                    by_cycle[info['cycle']] = info
                assert set(by_cycle) == {0, 1}
                checkpoints[name] = by_cycle
                if enabled:
                    hst = output/name/'history_lifecycle.z4c_hyperboloidal.hst'
                    reports['rows'] = read_history(hst)
                    reports['history_sha256'] = sha256(hst)
                    summary = output/name/'history_lifecycle.horizon_summary_0.txt'
                    # FastFlow summary/shape are written once per accepted state,
                    # although Driver::Finalize deliberately repeats the hst row.
                    summary_rows = [line for line in summary.read_text().splitlines()
                                    if line and not line.startswith('#')]
                    assert len(summary_rows) == 2
                    assert [int(line.split()[0]) for line in summary_rows] == [0, 1]
                else:
                    assert not list((output/name).glob('*.z4c_hyperboloidal.hst'))
            reports['history_preserves_all_state'] = {
                str(cycle): compare(checkpoints[f'control-ranks{ranks}'][cycle],
                                    checkpoints[f'history-ranks{ranks}'][cycle])
                for cycle in (0, 1)}
            runs[str(ranks)] = reports
        first = str(args.ranks[0])
        for ranks, report in runs.items():
            errors = [abs(a-b) for row, ref in zip(report['rows'], runs[first]['rows'])
                      for a, b in zip(row, ref)]
            refs = [value for row in runs[first]['rows'] for value in row]
            assert all(error <= 1e-12+5e-10*abs(reference)
                       for error, reference in zip(errors, refs))
            report['cross_rank_max_absolute'] = max(errors)
            report['cross_rank_state'] = compare(checkpoints[f'history-ranks{first}'][1],
                                                checkpoints[f'history-ranks{ranks}'][1])
        result.update(passed=True, runs=runs, commands=commands,
                      test_admission_override=False, executable_sha256=binary_hash)
    finally:
        stable = dict(binary_unchanged=sha256(binary) == binary_hash,
                      sources_unchanged=all(sha256(root/name) == digest
                                            for name, digest in sources.items()),
                      objects_unchanged=all(sha256(Path(name)) == digest
                                            for name, digest in objects.items()))
        result.update(runs=runs, commands=commands, provenance=stable)
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(stable.values()), stable
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
