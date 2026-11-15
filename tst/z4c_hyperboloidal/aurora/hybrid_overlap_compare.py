#!/usr/bin/env python3
"""Paired fresh-initial-data N16 windows on a single two-node allocation.

No previous campaign is resumed. All output remains in the new allocation's
run directory. The selected dt, precision and science output cadence are shared.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from hybrid_profile import offset
from hybrid_science_segment import header


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_value(text, block, key, value):
    pattern = rf'(<{re.escape(block)}>\n)(.*?)(?=\n<|\Z)'
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(block)
    body = match[2]
    entry = rf'(?m)^{re.escape(key)}\s*=.*$'
    line = f'{key} = {value}'
    body = re.sub(entry, lambda _: line, body) if re.search(entry, body) else line+'\n'+body
    return text[:match.start(2)]+body+text[match.end(2):]


def measure(log, wall, dt, steps):
    rows = [(float(a), int(b), float(c)) for a, b, c in re.findall(
        r'elapsed=([\deE.+-]+) cycle=(\d+) time=([\deE.+-]+)', log)]
    if not rows or rows[-1][1] != steps or abs(rows[-1][2]-steps*dt) > 1e-12:
        raise RuntimeError('window did not reach matched physical endpoint')
    intervals = [b[0]-a[0] for a, b in zip(rows, rows[1:]) if b[1] == a[1]+1 and a[1] >= 20]
    result = {'wall_seconds': wall, 'steps': steps, 'dt': dt, 'evolved_M0': steps*dt*3,
              'end_to_end_seconds_per_M0': wall/(steps*dt*3),
              'startup_to_cycle0_seconds': rows[0][0],
              'evolution_seconds': rows[-1][0]-rows[0][0],
              'evolution_seconds_per_M0': (rows[-1][0]-rows[0][0])/(steps*dt*3)}
    if intervals:
        result.update(typical_step_seconds=statistics.median(intervals),
                      average_step_seconds=statistics.mean(intervals),
                      step_stdev_seconds=statistics.stdev(intervals) if len(intervals)>1 else 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--wrapper', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--long-windows', action='store_true',
                        help='four paired one-M0 windows ending on the normal restart cadence; requires prior GPU admissions')
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    env = [line.split() for line in (run/'rank-environment.txt').read_text().splitlines()]
    assert len(env) == 24 and len({r[0] for r in env}) == 2
    for host in {r[0] for r in env}:
        rows = [r for r in env if r[0] == host]
        assert len(rows) == 12 and len({r[2] for r in rows}) == 12
        assert {int(r[1]) for r in rows} == set(range(12))
    assert all(r[3:8] == ['1', '1', 'specialized', 'limited', '128'] for r in env), env
    cpu_sets = [set(range(int(part.split('-')[0]),int(part.split('-')[1])+1))
                for part in args.binding.split(':')]
    assert all(set(map(int,r[8].split(','))) == cpu_sets[int(r[1])] for r in env), env
    builds = {'legacy': root/'build-baseline', 'masked': root/'qualified-candidate-v4',
              'agreement': root/'qualified-candidate-v4'}
    for build in set(builds.values()):
        subprocess.run(['sha256sum', '-c', str(build/'executable.sha256')], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    original = (root/'input-N16.athinput').read_text()
    data_dir = Path(re.search(r'(?m)^binary_directory\s*=\s*(.+)$',original)[1].strip())
    initial_manifest = json.loads((data_dir/'manifest.json').read_text())
    assert initial_manifest['identity'] == re.search(r'(?m)^binary_identity\s*=\s*(.+)$',original)[1].strip()
    assert (root/'export-check.txt').read_text().startswith('PASS ')
    assert sha(data_dir/'manifest.json') == sha(root/'initial-data-manifest.json')
    (run/'initial-data-manifest.json').write_bytes((root/'initial-data-manifest.json').read_bytes())
    (run/'export-check.txt').write_bytes((root/'export-check.txt').read_bytes())
    dt = float(re.search(r'(?m)^dt\s*=\s*([\deE.+-]+)', original)[1])
    assert dt == 0.00035208152801713085
    launch = {'protocol': ('fresh matched N16; 20 warmup steps; four one-M0 windows on normal restart cadence'
                          if args.long_windows else 'fresh matched N16; 20 warmup steps; alternating 300-step windows'),
              'expected_windows':4 if args.long_windows else 10,
              'files':{name:sha(run/name) for name in ['hybrid-overlap-two.pbs',
                  'hybrid_overlap_compare.py','hybrid_rank_environment.py','rank-environment.txt',
                  'modules.txt','nodes.txt','input-N16.athinput','initial-data-manifest.json']},
              'build_scripts':{name:sha(root/name) for name in ['build-hybrid-overlap.sh','qualify-v4.sh']},
              'validation_helpers':{name:sha(root/name) for name in ['hybrid_profile.py','hybrid_science_segment.py']}}
    (run/'launch-provenance.json').write_text(json.dumps(launch,indent=2)+'\n')
    measurements = []
    # Short real-binary/GPU admissions precede sustained timing. Repeat opposite
    # ordering after warmup so cache and thermal drift are visible in the table.
    sequence = [(m, 8) for m in ['legacy', 'masked', 'agreement']]
    sequence += [(m, 300) for m in ['legacy', 'agreement', 'masked', 'agreement', 'legacy', 'agreement', 'legacy']]
    if args.long_windows:
        for index, mode in enumerate(['legacy', 'masked', 'agreement']):
            admission = root/'runs/paired-9076396'/f'{index:02d}-{mode}-8'
            assert json.loads((admission/'equivalence.json').read_text())['accepted_payload_bitwise_equal']
            assert json.loads((admission/'provenance.json').read_text())['executable_sha256'] == sha(builds[mode]/'src/athena')
        steps = math.ceil((1/3)/dt)
        assert steps == 947
        sequence = [(m, steps) for m in ['legacy', 'agreement', 'agreement', 'legacy']]
    for index, (mode, steps) in enumerate(sequence):
        folder = run/f'{index:02d}-{mode}-{steps}'
        folder.mkdir()
        text = original
        for block, key, value in [('time', 'nlim', steps), ('time', 'tlim', format(steps*dt,'.17g')),
                                  ('time', 'ndiag', 1), ('z4c_hyperboloidal','overlap_mode',mode),
                                  ('z4c_hyperboloidal','cartesian_cutoff',.25),
                                  ('z4c_hyperboloidal','agreement_kappa',1),
                                  ('z4c_hyperboloidal','agreement_inner',.18),
                                  ('z4c_hyperboloidal','agreement_outer',.195)]:
            text = set_value(text, block, key, value)
        inp = folder/'input.athinput'; inp.write_text(text)
        exe = builds[mode]/'src/athena'
        command = ['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+args.binding,
                   args.wrapper,str(exe),'-i',str(inp),'-d',str(folder)]
        provenance = {'command':command,'executable_sha256':sha(exe),'input_sha256':sha(inp),
                      'build_source_manifest_sha256':sha(builds[mode]/'source-sha256.json'),
                      'initial_data_manifest_sha256':sha(data_dir/'manifest.json'),
                      'mode':mode,'steps':steps,'PBS_JOBID':os.environ.get('PBS_JOBID')}
        (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        start = time.perf_counter()
        with (folder/'run.log').open('w') as log:
            process = subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=1100 if args.long_windows else 900)
        wall = time.perf_counter()-start
        (folder/'exit-code.txt').write_text(str(process.returncode)+'\n')
        process.check_returncode()
        if steps == 8:
            reference = root/'runs/paired-9076258'/f'{index:02d}-{mode}-8'
            old = max((reference/'rst').glob('*.rst'), key=lambda f:header(f)['cycle'])
            new = max((folder/'rst').glob('*.rst'), key=lambda f:header(f)['cycle'])
            assert header(old) == header(new) and header(new)['cycle'] == 8
            digest = hashlib.sha256(); size = 0
            with old.open('rb') as x, new.open('rb') as y:
                x.seek(offset(old)); y.seek(offset(new))
                while True:
                    b = x.read(8*1024*1024); c = y.read(8*1024*1024)
                    assert b == c, 'optimized build changed accepted checkpoint payload'
                    if not b: break
                    digest.update(b); size += len(b)
            (folder/'equivalence.json').write_text(json.dumps({
                'reference':str(reference), 'accepted_payload_bitwise_equal':True,
                'payload_sha256':digest.hexdigest(), 'payload_bytes':size},indent=2)+'\n')
        measurement = measure((folder/'run.log').read_text(),wall,dt,steps)
        measurement.update(mode=mode,index=index,provenance=provenance)
        measurements.append(measurement)
        (run/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
        print(json.dumps(measurement),flush=True)
    for build in set(builds.values()):
        subprocess.run(['sha256sum','-c',str(build/'executable.sha256')],check=True)


if __name__ == '__main__':
    main()
