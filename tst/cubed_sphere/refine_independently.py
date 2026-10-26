#!/usr/bin/env python3
"""Reproduce the regular pulse, then vary bulk and shell resolution separately.

Runs independent cases on the two tiles of one Aurora GPU. Each executable is
frozen before launch; numerical acceptance and boundary causality are separate
from successful completion. The baseline values are the original job 8923658.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

EXPECTED = {
    'bulk': {'l2': .74635894949858805, 'linf': 11.689358183446709,
             'H_l2': .48370772634319603, 'H_linf': 16.490344199392581,
             'M_l2': .17442831672236678, 'Z_l2': .021976626229127993,
             'weyl_l2': .11188642111355453},
    'shell': {'l2': .027682422824907908, 'linf': .063060091579957983,
              'H_l2': .080473430700475454, 'H_linf': .66028025113520661,
              'M_l2': .047795495270285107, 'Z_l2': .0066481977056522179,
              'weyl_l2': .070341954306218354}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = output/'athena_frozen'
    shutil.copy2(args.binary, binary)
    source = args.source.resolve()
    files = [*source.glob('src/**/*.cpp'), *source.glob('src/**/*.hpp'),
             source/'CMakeLists.txt', source/'src/CMakeLists.txt', source/'config.hpp.in',
             source/'inputs/z4c/cubed_sphere.athinput', Path(__file__).resolve()]
    report = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(set(files))}, 'cases': {}}
    summary = output/'summary.json'

    def run(bulk, shell, tile):
        directory = output/('B%d_S%d' % (bulk, shell))
        directory.mkdir()
        shutil.copy2(source/'inputs/z4c/cubed_sphere.athinput',directory/'base.athinput')
        command = [str(binary), '-i', str(directory/'base.athinput'), '-d', str(directory),
                   'mesh/nx1=%d' % bulk, 'mesh/nx2=%d' % bulk, 'mesh/nx3=%d' % bulk,
                   'cubed_sphere/nr=%d' % shell, 'cubed_sphere/nangular=%d' % shell,
                   'problem/initial_data=angular_pulse', 'problem/amplitude=0.2',
                   'cubed_sphere/outer_boundary=sommerfeld', 'time/tlim=2',
                   'time/cfl_number=0.15']
        env = dict(os.environ, ZE_AFFINITY_MASK=tile, ONEAPI_DEVICE_SELECTOR='level_zero:gpu')
        start = time.monotonic()
        with (directory/'run.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, timeout=450)
        result = {'command': command, 'device_tile': tile, 'exit_code': process.returncode,
                  'elapsed_seconds': time.monotonic()-start}
        if process.returncode == 0:
            result['norms'] = {r['region']: {k:float(v) for k,v in r.items() if k!='region'}
                              for r in csv.DictReader((directory/'cubed_sphere_errors.csv').open())}
            assert all(math.isfinite(v) for r in result['norms'].values() for v in r.values())
        (directory/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        assert process.returncode == 0, result
        return result

    baseline = run(32,16,'0.0')
    report['cases']['B32_S16'] = baseline
    for region, row in EXPECTED.items():
        for key, value in row.items():
            actual = baseline['norms'][region][key]
            assert abs(actual-value) <= 1e-10*abs(value), (region,key,actual,value)
    report['baseline_reproduced'] = True
    summary.write_text(json.dumps(report,indent=2)+'\n')
    print('Baseline reproduced before independent refinements.', flush=True)

    def sequence(cases,tile):
        return [('B%d_S%d' % (b,s),run(b,s,tile)) for b,s in cases]

    with ThreadPoolExecutor(max_workers=2) as pool:
        bulk = pool.submit(sequence,[(48,16),(64,16)],'0.0')
        shell = pool.submit(sequence,[(32,24),(32,32)],'0.1')
        for future in [bulk,shell]:
            for name,result in future.result():
                report['cases'][name] = result
            summary.write_text(json.dumps(report,indent=2)+'\n')
    print('Independent refinement runs completed; constraints require analysis.', flush=True)


if __name__ == '__main__':
    main()
