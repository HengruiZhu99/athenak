#!/usr/bin/env python3
"""Stage tests for physical constraint radiation, defaults, threads and MPI.

Uses a supplied immutable baseline executable and sixth-order trumpet input.
This checks discrete zero preservation/repeatability, not long-term stability.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def set_value(text, block, key, value):
    match = re.search(rf'(<{block}>\n)(.*?)(?=\n<|\Z)', text, re.S)
    assert match, block
    body = match[2]
    pattern = rf'(?m)^{key}\s*=.*$'
    replacement = f'{key} = {value}'
    body = re.sub(pattern, replacement, body) if re.search(pattern, body) else body+'\n'+replacement+'\n'
    return text[:match.start()]+match[1]+body+text[match.end():]


def snapshots(folder):
    values = {}
    for path in folder.glob('z4c_snapshot_*.json'):
        if path.name.endswith('.background.json'):
            continue
        meta = json.loads(path.read_text())
        state = np.fromfile(path.with_suffix('.bin'), dtype='<f8').reshape(meta['shape'])
        assert np.isfinite(state).all(), path
        operation = path.name.split('_rank')[0].removeprefix('z4c_snapshot_')
        for block, data in zip(meta['blocks'], state):
            key = (operation, meta['cycle'], meta['stage'], block['gid'])
            assert key not in values
            values[key] = data
    assert values, folder
    return values


def equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].tobytes() == b[key].tobytes(), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default=None)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = args.input.read_text()
    for block, key, value in [
        ('time','nlim',3), ('z4c','debug_balance','true'),
        ('z4c','debug_reduction_stride',1),
        ('z4c','debug_snapshot_operations','pre_boundary_rhs,post_boundary_rhs,post_recast'),
        ('z4c','characteristic_bc_diagnostics','true'),
        ('z4c','characteristic_bc_diagnostic_interval',1),
        ('mhd','debug_metric_before_c2p','true'),
    ]:
        base = set_value(base, block, key, value)
    for number in [2,3,4]:
        base = set_value(base, f'output{number}', 'single_file_per_rank', 'true')
    results = {}

    def run(label, exe, source, amplitude, ranks=1, threads=1):
        folder = root/label
        folder.mkdir()
        text = set_value(base,'z4c','characteristic_bc_source',source)
        text = set_value(text,'problem','vacuum_gauge_pulse_amplitude',amplitude)
        (folder/'input.athinput').write_text(text)
        command = ([args.launcher,'-n',str(ranks)] if args.launcher else [])
        assert ranks == 1 or args.launcher
        command += [str(exe.resolve()),'-i','input.athinput']
        with (folder/'run.log').open('w') as log:
            subprocess.run(command, cwd=folder, check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=1200,
                           env={**os.environ,'OMP_NUM_THREADS':str(threads),
                                'OMP_PROC_BIND':'false'})
        log = (folder/'run.log').read_text()
        assert 'Terminating on cycle limit' in log
        assert not any(word in log for word in ['C2P_INVALID_ADM_INPUT','Z4C_INVALID_STATE','FATAL ERROR'])
        arrays = snapshots(folder)
        if amplitude == 0:
            assert all(np.count_nonzero(v) == 0 for v in arrays.values())
            records = []
            for path in folder.glob('z4c_balance_rank*.csv'):
                with path.open() as stream:
                    records += list(csv.DictReader(stream))
            assert records
            for row in records:
                assert float(row['max_abs']) == 0 and int(row['nonfinite']) == 0
        results[label] = {'ranks':ranks,'threads':threads,'source':source,
                          'snapshot_blocks':len(arrays),'finite':True,
                          'max_abs':max(float(np.max(abs(v))) for v in arrays.values()),
                          'hashes':{str(k):hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
        (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,'PASS',flush=True)
        return arrays

    reference = run('baseline_default',args.baseline,'zero_rate',1e-8)
    equal(reference,run('candidate_default',args.exe,'zero_rate',1e-8))
    reference_pulse = None
    for ranks in args.ranks:
        for threads in args.threads:
            run(f'zero_r{ranks}_t{threads}',args.exe,'physical_constraint_radiation',0,ranks,threads)
            pulse = run(f'pulse_r{ranks}_t{threads}',args.exe,'physical_constraint_radiation',1e-8,ranks,threads)
            if reference_pulse is None:
                reference_pulse = pulse
                assert any(not np.array_equal(reference[k],v) for k,v in pulse.items())
            else:
                equal(reference_pulse,pulse)
    results['checks'] = {'default_bitwise_unchanged':True,'zero_exact_including_ghosts':True,
                         'pulse_repeatability_bitwise':True,
                         'scope':'Three steps only; long evolution and refinement separate.'}
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
