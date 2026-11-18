"""Advance one restartable segment of the isolated 10 M0 N12 comparison."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from binding import launcher
from activity import snapshot, utilization

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'z4c_hyperboloidal'))
from restart_roundtrip import parse, payload


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(part)
        return h.hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def placement_for_binary(root, run, binary_sha):
    # Preserve earlier timings. Each new evolution binary has its own receipts.
    report_path = root/'throughput-results.json'
    def measurements(path):
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {r['mode']:r for r in data['results'] if r['binary_sha256']==binary_sha}
    measured = measurements(report_path)
    candidates = ('cohort','locality','spread_fixed')
    if not set(candidates)<=measured.keys():
        cases = root/f'kernel-{binary_sha[:12]}.json'
        report_path = root/(cases.stem+'-results.json')
        measured = measurements(report_path)
        if not set((*candidates,'classical'))<=measured.keys():
            save(cases,[['current754',24,mode,6] for mode in (*candidates,'classical')])
            subprocess.run(['/usr/bin/python3.11',str(Path(__file__).with_name('measure.py')),
                            str(root),'--cases',str(cases)],check=True)
            measured = measurements(report_path)
    checks = json.loads(report_path.with_name(report_path.name.replace('results','comparison')).read_text())['checks']
    for mode,control in [('cohort','spread'),('locality','cohort'),('spread_fixed','spread')]:
        check = next(c for c in checks if c['mesh']=='current754' and c['gpus']==12 and
                     c['method']==mode and c['comparison']==control)
        if not check['identical_checkpoint_payload']:
            raise ValueError(f'{mode} failed to reproduce {control} on the N12 atlas')
    choices = [measured[m] for m in candidates]
    selected = min(choices,key=lambda r:r['steady_seconds_per_M0'])
    save(run/'placement-selection.json',dict(selected=selected['mode'],binary_sha256=binary_sha,
         report=str(report_path),steady_seconds_per_M0={r['mode']:r['steady_seconds_per_M0'] for r in choices}))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('mode', choices=['multirate','uniform'])
    parser.add_argument('--wall-seconds', type=int, required=True)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wall_seconds
    root = args.root.resolve()
    campaign = root.parent/'n12-radial-blocks-20261117'
    original = json.loads((campaign/'control/plan.json').read_text())
    directory = root/'n12-10M'/args.mode
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'segment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = Path(original['initial_manifest'])
    if digest(manifest) != original['initial_manifest_sha256']:
        raise ValueError('N12 initial-data manifest changed')
    original_input = campaign/'input.athinput'
    if digest(original_input) != original['immutable_files'][str(original_input)]:
        raise ValueError('N12 spatial input changed')
    # All methods land on the same final common endpoint with a fixed dt.
    nominal_base = 16*original['dt']
    coarse_steps = math.ceil(10/(3*nominal_base))
    ratio = 16 if args.mode == 'uniform' else 1
    target = coarse_steps*ratio
    dt = (10/3)/target
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state['cycle'] == target:
        return
    if state:
        executable = Path(state['executable'])
        if digest(executable) != state['binary_sha256'] or digest(Path(state['checkpoint'])) != state['checkpoint_sha256']:
            raise ValueError('pinned restart or executable digest changed')
    else:
        built = root/'build/src/athena'
        binary_sha = digest(built)
        executable = root/'artifacts'/binary_sha/'athena'
        executable.parent.mkdir(parents=True,exist_ok=True)
        if not executable.exists():
            temp = executable.with_suffix(f'.{os.getpid()}.copy')
            shutil.copy2(built,temp)
            temp.replace(executable)
        if digest(executable) != binary_sha:
            raise ValueError('isolated executable copy differs')
    job = os.environ['PBS_JOBID'].split('.')[0]
    run = directory/f'segment-{job}'
    run.mkdir()
    start_cycle = state['cycle'] if state else 0
    if state:
        # Native restart input retains the integrator, phase placement and output counters.
        input_args = ['-r',state['checkpoint']]
    else:
        extra = ''
        if args.mode == 'multirate':
            selected = placement_for_binary(root,run,binary_sha)
            # Hold the profiled owner map fixed throughout the trajectory and on restart.
            with Path(selected['log']).with_name('partition.log').open('rb') as stream:
                lines = stream.read(int(stream.readline())).decode().splitlines()[1:]
            owners = ','.join(row.split()[1] for row in lines)
            extra = (f'\n<multirate>\nbalance = none\nblock_owners = {owners}\n'
                     f'complete_local_batches = {str(selected["mode"] in ("cohort","locality")).lower()}\n'
                     'spherical_levels = 2,2,2,1,1,1,1,1,2,3,4\nprepost_exchanges = true\n')
        integrator = 'rk4_classical' if ratio==16 else 'mrgark4'
        inputs = (original_input.read_text()+extra+f'\n<time>\nintegrator = {integrator}\ndt = {dt:.17g}\n'
                  f'nlim = {target}\ntlim = {(target+2)*dt:.17g}\nndiag = {ratio}\n'
                  f'<output1>\ndcycle = {37*ratio}\n<output2>\ndcycle = {111*ratio}\n'
                  f'<output20>\nfile_type = scri\nid = scri\ndcycle = {6*ratio}\nnodal = true\n'
                  'modes = true\nlmax = 8\nboundary_diagnostics = true\nunit_description = geometric code units, M0=1/3\n')
        (run/'input.athinput').write_text(inputs)
        input_args = ['-i',str(run/'input.athinput')]
    remaining = int(deadline-time.monotonic())
    if remaining<120:
        raise ValueError('segment has insufficient remaining allocation time')
    m,s = divmod(remaining,60)
    h,m = divmod(m,60)
    cmd = [*launcher(24),str(executable),*input_args,'-d',str(run),'-t',f'{h:02d}:{m:02d}:{s:02d}']
    receipt = dict(job=job, mode=args.mode, executable=str(executable), binary_sha256=digest(executable),
                   manifest_sha256=original['initial_manifest_sha256'], input_source_sha256=digest(original_input),
                   start_cycle=start_cycle, target_cycles=target, dt=dt, command=cmd)
    save(run/'launch.json',receipt)
    counters = snapshot(root,run,"start",2)
    begin = time.monotonic()
    with (run/'run.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=run,check=True)
    wall = time.monotonic()-begin
    receipt["engine_activity"] = utilization(counters,snapshot(root,run,"end",2))
    final = max((parse(path) for path in (run/'rst').glob('*.rst')),key=lambda item:item['cycle'])
    cycle = final['cycle']
    if not start_cycle<cycle<=target or final['blocks']!=754 or final['dt']!=dt:
        raise ValueError('unexpected N12 checkpoint or lack of forward progress')
    if abs(final['time']-cycle*dt)>max(1,cycle)*math.ulp(cycle*dt):
        raise ValueError('N12 endpoint time differs from fixed-step schedule')
    values = payload(final)
    for block in values:
        if not np.isfinite(block).all():
            raise ValueError('nonfinite N12 accepted fields or diagnostics')
    receipt.update(cycle=cycle, time_M0=3*final['time'], blocks=754, wall_seconds=wall,
                   segment_seconds_per_M0=wall/(3*dt*(cycle-start_cycle)),
                   checkpoint=final['path'], checkpoint_sha256=digest(Path(final['path'])),
                   target_reached=cycle==target)
    save(run/'receipt.json',receipt)
    save(state_path,receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
