"""Advance one restartable segment of the isolated 10 M0 N12 comparison."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from n12_paths import trajectory_directory
import re
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


def input_parameters(text):
    # AthenaK accepts repeated blocks; the last assignment wins.
    parameters = {}
    block = None
    for raw in text.splitlines():
        line = raw.partition('#')[0].strip()
        if line.startswith('<') and line.endswith('>'):
            block = line[1:-1]
        elif '=' in line:
            key, value = line.split('=', 1)
            parameters[block, key.strip()] = value.strip()
    return parameters


def verify_export(manifest, parameters):
    data = json.loads(manifest.read_text())
    directory = Path(parameters['problem','binary_directory'])
    identity = parameters['problem','binary_identity']
    if data.get('identity') != identity:
        raise ValueError('N12 export identity differs from the input')
    blocks = data.get('blocks')
    if not isinstance(blocks,dict) or len(blocks) != 754:
        raise ValueError('N12 export must contain the 754 recorded block digests')
    if data.get('status') != 'COMPLETE':
        raise ValueError('incomplete N12 export')
    if set(blocks) != {p.name for p in directory.glob('*.hbi')}:
        raise ValueError('N12 export block inventory differs from the manifest')
    for name, entry in blocks.items():
        if digest(directory/name) != entry['sha256']:
            raise ValueError(f'N12 initial data changed: {name}')
    return len(blocks)


def verify_header(info, cycle, provenance):
    dt = provenance['dt']
    if (info['blocks'] != 754 or info['record_blocks'] != 754 or info['cycle'] != cycle or
            info['dt'] != dt or not math.isfinite(info['time']) or
            abs(info['time']-cycle*dt) > max(1,cycle)*math.ulp(cycle*dt)):
        raise ValueError('N12 native restart disagrees with the fixed hierarchy or schedule')
    with Path(info['path']).open('rb') as stream:
        parameters = input_parameters(stream.read(info['start']).decode())
    integrator = 'rk4_classical' if provenance['mode']=='uniform' else 'mrgark4'
    if parameters['time','integrator'] != integrator or int(parameters['time','nlim']) != provenance['target_cycles']:
        raise ValueError('native restart integrator or cycle limit differs from the N12 run')


def final_checkpoint(run):
    # Finalize writes after periodic output even when both fall on the same cycle.
    paths = sorted((run/'rst').glob('*.rst'),key=lambda p:int(p.suffixes[-2][1:]))
    if not paths:
        raise ValueError(f'no native checkpoint in {run}')
    final = parse(paths[-1])
    with (run/'run.log').open() as stream:
        endings = [m.groups() for line in stream
                   if (m := re.fullmatch(r'time=(\S+) cycle=(\d+)\s*',line))]
    if len(endings)!=1 or (float(endings[0][0]),int(endings[0][1])) != (final['time'],final['cycle']):
        raise ValueError('last native checkpoint differs from the driver termination record')
    return final


def restore(directory, provenance):
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    receipts = sorted((json.loads(p.read_text()) for p in directory.glob('segment-*/receipt.json')),
                      key=lambda item:item['start_cycle'])
    if not receipts:
        if state is not None:
            raise ValueError('missing N12 segment receipt')
        return None
    if state is not None and state not in receipts:
        raise ValueError('N12 state does not identify a committed segment')
    initial = Path(receipts[0]['checkpoint']).parent.parent/'input.athinput'
    input_sha = digest(initial)
    previous = None
    for item in receipts:
        if (any(item[k] != v for k,v in provenance.items()) or
                item['start_cycle'] != (previous['cycle'] if previous else 0) or
                not item['start_cycle'] < item['cycle'] <= provenance['target_cycles'] or
                item['binary_sha256'] != receipts[0]['binary_sha256'] or
                item.get('input_sha256',input_sha) != input_sha or
                ('parent_checkpoint_sha256' in item and item['parent_checkpoint_sha256'] !=
                 (previous['checkpoint_sha256'] if previous else None))):
            raise ValueError('N12 segment provenance or progress differs from the committed chain')
        previous = item
    last = receipts[-1]
    checkpoint = Path(last['checkpoint'])
    if (digest(Path(last['executable'])) != last['binary_sha256'] or
            digest(checkpoint) != last['checkpoint_sha256']):
        raise ValueError('pinned restart or executable digest changed')
    final = final_checkpoint(checkpoint.parent.parent)
    if Path(final['path']) != checkpoint:
        # Older receipts selected by maximum cycle alone when Finalize duplicated a
        # periodic dump. Retain that explicit restart only if the geometric state,
        # accepted fields and persistent diagnostics match the final sequence.
        old = parse(checkpoint)
        keys = ('cycle','time','dt','blocks','record_blocks','stride','atlas','topology','identity')
        if ('input_sha256' in last or any(old[k] != final[k] for k in keys) or
                any(not np.array_equal(a.view(np.uint64),b.view(np.uint64))
                    for a,b in zip(payload(old),payload(final)))):
            raise ValueError('receipt does not point to an equivalent final native checkpoint')
        final = old
    verify_header(final,last['cycle'],provenance)
    if last['time_M0'] != 3*final['time'] or last['target_reached'] != (last['cycle']==provenance['target_cycles']):
        raise ValueError('N12 state disagrees with the accepted endpoint')
    # A job can terminate after committing receipt.json but before replacing state.json.
    if last != state:
        save(state_path,last)
    return dict(last,input_sha256=input_sha,input_path=str(initial))


def placement_for_binary(root, run, binary_sha):
    # Preserve earlier timings. Each new evolution binary has its own receipts.
    report_path = root/'throughput-results.json'
    def measurements(path):
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {r['mode']:r for r in data['results'] if r['binary_sha256']==binary_sha}
    # Reuse same-executable six-interval controls when the full accepted
    # payload was checked against the archived spread run for this atlas.
    modes = ['cohort','cohort_cached']
    cache_path = root/'send-cache-trial-results.json'
    trial = root/'n12-10M/multirate-trial.json'
    if trial.is_file() and json.loads(trial.read_text()).get('compare_stage_packets'):
        modes.append('cohort_packets')
        cases = root/f'stage-packets-{binary_sha[:12]}.json'
        cache_path = root/(cases.stem+'-results.json')
        if not set(modes+['classical'])<=measurements(cache_path).keys():
            save(cases,[['current754',24,mode,6] for mode in (*modes,'classical')])
            subprocess.run(['/usr/bin/python3.11',str(Path(__file__).with_name('measure.py')),
                            str(root),'--cases',str(cases)],check=True)
    cache = measurements(cache_path)
    if all(m in cache and cache[m]['mesh']=='current754' and cache[m]['tiles']==24
           and cache[m]['intervals']==6 for m in (*modes,'classical')):
        checks_path=cache_path.with_name(cache_path.name.replace('results','comparison'))
        checks=json.loads(checks_path.read_text())['checks']
        required=set(zip(modes,['spread',*modes[:-1]]))
        verified={(c['method'],c['comparison']) for c in checks
                  if c['mesh']=='current754' and c['gpus']==12 and c['identical_checkpoint_payload']}
        if not required<=verified:
            raise ValueError('missing verified N12 payload comparison: '+str(required-verified))
        choices=[cache[m] for m in modes]
        selected=min(choices,key=lambda r:r['steady_seconds_per_M0'])
        save(run/'placement-selection.json',dict(selected=selected['mode'],binary_sha256=binary_sha,
             report=str(cache_path),steady_seconds_per_M0={r['mode']:r['steady_seconds_per_M0'] for r in choices}))
        return selected
    if len(modes)>2:
        raise ValueError('incomplete stage-packet N12 GPU controls')
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
    directory = trajectory_directory(root,args.mode)
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
    provenance = dict(mode=args.mode, manifest_sha256=original['initial_manifest_sha256'],
                      input_source_sha256=digest(original_input), target_cycles=target, dt=dt)
    state = restore(directory,provenance)
    job = os.environ['PBS_JOBID'].split('.')[0]
    if state and (state['cycle'] == target or state['job'] == job):
        return
    if state:
        executable = Path(state['executable'])
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
    run = directory/f'segment-{job}'
    attempt = 0
    while run.exists():
        attempt += 1
        run = directory/f'segment-{job}-attempt-{attempt}'
    run.mkdir()
    start_cycle = state['cycle'] if state else 0
    if state:
        # Native restart input retains the integrator, phase placement and output counters.
        input_args = ['-r',state['checkpoint']]
        initial_input = Path(state['input_path'])
        verified_blocks = state.get('verified_export_blocks')
    else:
        verified_blocks = verify_export(manifest,input_parameters(original_input.read_text()))
        extra = ''
        if args.mode == 'multirate':
            selected = placement_for_binary(root,run,binary_sha)
            # Hold the profiled owner map fixed throughout the trajectory and on restart.
            with Path(selected['log']).with_name('partition.log').open('rb') as stream:
                lines = stream.read(int(stream.readline())).decode().splitlines()[1:]
            owners = ','.join(row.split()[1] for row in lines)
            extra = (f'\n<multirate>\nbalance = none\nblock_owners = {owners}\n'
                     f'complete_local_batches = {str(selected["mode"] in ("cohort","locality","cohort_cached","cohort_packets")).lower()}\n'
                     f'send_cache_mib = {96 if selected["mode"] in ("cohort_cached","cohort_packets") else 0}\n'
                     'spherical_levels = 2,2,2,1,1,1,1,1,2,3,4\nprepost_exchanges = true\n')
            if selected['mode']=='cohort_packets':
                extra += 'coalesce_stage_packets = true\n'
        integrator = 'rk4_classical' if ratio==16 else 'mrgark4'
        inputs = (original_input.read_text()+extra+f'\n<time>\nintegrator = {integrator}\ndt = {dt:.17g}\n'
                  f'nlim = {target}\ntlim = {(target+2)*dt:.17g}\nndiag = {ratio}\n'
                  f'<output1>\ndcycle = {37*ratio}\n<output2>\ndcycle = {111*ratio}\n'
                  f'<output20>\nfile_type = scri\nid = scri\ndcycle = {6*ratio}\nnodal = true\n'
                  'modes = true\nlmax = 8\nboundary_diagnostics = true\nunit_description = geometric code units, M0=1/3\n')
        initial_input = run/'input.athinput'
        initial_input.write_text(inputs)
        input_args = ['-i',str(initial_input)]
    remaining = int(deadline-time.monotonic())
    if remaining<120:
        raise ValueError('segment has insufficient remaining allocation time')
    m,s = divmod(remaining,60)
    h,m = divmod(m,60)
    cmd = [*launcher(24),str(executable),*input_args,'-d',str(run),'-t',f'{h:02d}:{m:02d}:{s:02d}']
    receipt = dict(provenance, job=job, executable=str(executable), binary_sha256=digest(executable),
                   input_path=str(initial_input), input_sha256=digest(initial_input), verified_export_blocks=verified_blocks,
                   launcher_sha256=digest(Path(__file__)), start_cycle=start_cycle, command=cmd,
                   parent_checkpoint_sha256=state['checkpoint_sha256'] if state else None)
    save(run/'launch.json',receipt)
    counters = snapshot(root,run,"start",2)
    begin = time.monotonic()
    with (run/'run.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=run,check=True)
    wall = time.monotonic()-begin
    receipt["engine_activity"] = utilization(counters,snapshot(root,run,"end",2))
    final = final_checkpoint(run)
    cycle = final['cycle']
    verify_header(final,cycle,provenance)
    if not start_cycle<cycle<=target:
        raise ValueError('unexpected N12 checkpoint or lack of forward progress')
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
