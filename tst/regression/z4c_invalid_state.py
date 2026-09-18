#!/usr/bin/env python3
"""Invalid TOV/KS state must fail on any MPI rank, without relying on NaN maxima.

Inject faults only into disposable copies of a valid joined checkpoint. Tests
finite-but-indefinite metric, negative chi, and NaN in a non-metric Z4c field.
"""
import argparse
import json
import re
from pathlib import Path
import shutil
import struct
import subprocess
from z4c_background_restart import checkpoint, validate_run


def poison(source, target, kind):
    record = checkpoint(source)
    assert record['total'] == 8 and len(record['state']) == 8
    cells = record['cells']
    n = round(cells**(1/3))
    assert n == 16 and n**3 == cells  # This test input has mb=8 and ng=4.
    faces = 3*(n+1)*n*n
    z4c_offset = 8*(5*cells+faces)
    stride = z4c_offset + 8*25*cells
    shutil.copy2(source, target)
    payload_start = target.stat().st_size - record['total']*stride
    # Last block, active outer corner: exterior, and owned by rank 3 with 4 ranks.
    cell = (11*n+11)*n+11
    faults = {'nan_gamma': {14: float('nan')}, 'negative_chi': {0: -10.0},
              'indefinite_metric': {1: -10.0, 4: -10.0}}[kind]
    with target.open('r+b') as stream:
        for field, value in faults.items():
            pos = payload_start + 7*stride + z4c_offset + 8*(field*cells+cell)
            stream.seek(pos)
            assert struct.unpack('<d', stream.read(8))[0] == 0.0
            stream.seek(pos)
            stream.write(struct.pack('<d', value))


def run(args, directory, text, ranks, restart=None):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'input.athinput').write_text(text)
    cmd = [args.launcher, '-n', str(ranks), str(args.exe), '-i', 'input.athinput']
    if restart:
        cmd += ['-r', str(restart)]
    with (directory/'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=60)
    return result.returncode, (directory/'run.log').read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec')
    args = parser.parse_args()
    args.exe = args.exe.resolve(); args.output = args.output.resolve()
    source = Path(__file__).resolve().parents[1]/'inputs/z4c_ks_background.athinput'
    base = source.read_text().replace('nlim = 3', 'nlim = 1')
    base = base.replace('debug_balance = true', 'debug_balance = false')
    base += '\n<output2>\nfile_type = rst\ndt = 100\n'
    rc, log = run(args, args.output/'valid', base, 1)
    assert rc == 0; validate_run(args.output/'valid')
    saved = max((args.output/'valid/rst').glob('*.rst'),
                key=lambda p: checkpoint(p)['cycle'])
    assert checkpoint(saved)['cycle'] == 1
    results = {'valid': {'exit_code': rc}}
    for kind in ('nan_gamma', 'negative_chi', 'indefinite_metric'):
        fault = args.output/(kind+'.rst'); poison(saved, fault, kind)
        for ranks in (1, 4):
            name = kind+'_r'+str(ranks)
            rc, log = run(args, args.output/name, base,
                          ranks, fault)
            assert rc != 0 and 'Z4C_INVALID_STATE ' in log, (name, rc, log[-3000:])
            if ranks == 4:
                assert 'rank=3 gid=7' in log, 'Non-root fault was not reported'
            assert 'Terminating on cycle limit' not in log
            assert 'Terminating on time limit' not in log
            results[name] = {'exit_code': rc, 'invalid_state_reported': True}
    # Diagnostic opt-out remains explicit, for examining an invalid checkpoint.
    text = base.replace(
        '<problem>', '<problem>\nmetric_diag_abort_on_invalid = false')
    rc, log = run(args, args.output/'diagnostic_opt_out', text, 1,
                  args.output/'nan_gamma.rst')
    assert rc == 0 and 'Z4C_INVALID_STATE ' not in log
    history = (args.output/'diagnostic_opt_out/ks_background.user.hst').read_text()
    labels = re.findall(r'\[\d+\]=([^\s]+)', history.splitlines()[1])
    rows = [line.split() for line in history.splitlines() if line and not line.startswith('#')]
    assert float(rows[-1][labels.index('bad-metric')]) == 1.0
    results['diagnostic_opt_out'] = {'exit_code': rc, 'bad_cells': 1}
    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
