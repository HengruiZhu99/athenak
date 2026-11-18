"""Measure matched endpoint evolution on explicitly bound Aurora GPU tiles."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive import adaptive_input

root = Path(sys.argv[1]).resolve()
source = root / 'source/tst/multirate_native'
campaign = root.parent / 'n12-radial-blocks-20261117'
exe = root / 'build/src/athena'
with exe.open('rb') as binary:
    binary_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
results = []
failures = []


def launcher(ranks):
    ppn = min(ranks, 12)
    cores = '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'.split(':')
    return ['mpiexec', '-n', str(ranks), '-ppn', str(ppn), '--cpu-bind=list:'+':'.join(cores[:ppn]),
            shutil.which('gpu_tile_compact.sh')]


def case(mesh, ranks, mode, intervals):
    production = mesh == 'current754'
    adaptive = mesh == 'adaptive88'
    uniform = mode in ('uniform', 'classical')
    base = 4*4.6944203735617445e-4 if production else 2**-14
    factor = (16 if production else 2) if uniform else 1
    run = root/'runs'/f'{mesh}-{ranks}tiles-{mode}'
    complete = run/'completed.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['binary_sha256']==binary_sha256:
            return receipt
    run.mkdir(parents=True, exist_ok=True)
    if (run/'run.log').exists():
        run = run/('retry-'+str(time.time_ns()))
        run.mkdir()
    if adaptive:
        inp = adaptive_input(ranks, uniform=uniform, intervals=intervals, base=base)
        # This scaling control refines a core block offset from the puncture.
        # The 25-block restart control separately refines the puncture parent.
        inp = inp.replace('location_x1 = .015','location_x1 = .065')
        start, end = inp.index('<mesh>'), inp.index('<meshblock>')
        inp = inp[:start]+inp[start:end].replace('= 12', '= 48')+inp[end:]
        inp += f'\n<mesh_refinement>\nmax_nmb_per_rank = {max(12, (130+ranks-1)//ranks)}\n'
    else:
        inp = (campaign/'input.athinput' if production else source/'small.athinput').read_text()
    inp += '\n<time>\nintegrator = '+('rk4_classical' if mode=='classical' else 'mrgark4')
    inp += f'\ndt = {base/factor:.17g}\nnlim = {intervals*factor}\ntlim = 1\nndiag = 1\n'
    if mode != 'classical':
        regions = ('2,2,2,1,1,1,1,1,2,3,4' if production else
                   '1,1,0,1' if adaptive else '0,0,0,1')
        balance = mode if mode in ('scalar','metis') else 'none'
        inp += (f'<multirate>\nspherical_levels = {regions}\nuniform_reference = {str(uniform).lower()}\n'
                f'balance = {balance}\nwarmup_intervals = 2\nreview_interval = 32\n'
                f'partition_log = {run}/partition.log\n')
    (run/'input.athinput').write_text(inp)
    launch = launcher(ranks)
    with (run/'binding.log').open('w') as log:
        subprocess.run([*launch, '/bin/bash', '-c', 'echo "$(hostname) tile=$ZE_AFFINITY_MASK"'],
                       stdout=log, stderr=log, check=True)
    pairs = re.findall(r'^([\w.-]+) tile=(\d+\.\d+)$', (run/'binding.log').read_text(), re.M)
    if len(set(pairs)) != ranks or len({(host,tile.split('.')[0]) for host,tile in pairs}) != ranks//2:
        raise RuntimeError('unexpected Aurora tile binding: '+str(run/'binding.log'))
    command = [*launch, str(exe), '-i', str(run/'input.athinput')]
    start = time.monotonic()
    with (run/'run.log').open('w') as log:
        code = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic()-start
    text = (run/'run.log').read_text()
    final = re.search(r'^time=([^ ]+) cycle=(\d+)$', text, re.M)
    if code or not final or int(final[2]) != intervals*factor or abs(float(final[1])-intervals*base)>1.e-13:
        raise RuntimeError(f'GPU evolution failed in {run} (exit {code})')
    if not re.search(r'MULTIRATE_MEMORY space=SYCLDeviceUSM peak_allocated_bytes=[1-9]\d*', text):
        raise AssertionError('missing Kokkos GPU allocation receipt in '+str(run))
    regrids = re.findall(r'MULTIRATE_REGRID cycle=(\d+) created=(\d+) deleted=(\d+) blocks=(\d+)', text)
    if adaptive and not any(int(created) and int(cycle)==factor for cycle,created,_,_ in regrids):
        raise AssertionError('missing synchronized first regrid: '+str(run))
    # Use the driver clock for every method. Differences include AMR, migration
    # and diagnostics between cuts. Ignore warmup and report individual samples.
    cuts = {int(n):float(t) for t,n in re.findall(r'^elapsed=(\S+) cycle=(\d+) ',text,re.M)}
    samples = [cuts[n+factor]-cuts[n] for n in range(2*factor,intervals*factor,factor)
               if n in cuts and n+factor in cuts]
    if not samples:
        raise AssertionError('missing post-warmup driver timing samples in '+str(run))
    endpoints = sorted((run/'rst').glob('*.rst'))
    if not endpoints:
        raise AssertionError('missing accepted checkpoint in '+str(run))
    receipt = dict(binary_sha256=binary_sha256, input_sha256=hashlib.sha256(inp.encode()).hexdigest(),
                   job_id=os.environ.get('PBS_JOBID'), mesh=mesh, mode=mode, tiles=ranks, physical_gpus=ranks//2,
                   nodes=len({host for host,_ in pairs}), simulated_M0=3*intervals*base,
                   seconds=wall, seconds_per_M0=wall/(3*intervals*base),
                   steady_base_seconds=samples, steady_seconds_per_M0=median(samples)/(3*base),
                   command=command, regrids=regrids, log=str(run/'run.log'),
                   work=[line for line in text.splitlines() if line.startswith('MULTIRATE_')],
                   checkpoint=str(endpoints[-1]))
    encoded = json.dumps(receipt,indent=2)+'\n'
    (run/'receipt.json').write_text(encoded)
    temporary = complete.with_suffix('.tmp')
    temporary.write_text(encoded)
    temporary.replace(complete)
    return receipt


def measure(*args):
    try:
        results.append(case(*args))
    except (RuntimeError, ValueError, AssertionError, OSError, subprocess.SubprocessError) as exc:
        failures.append({'case':args,'error':str(exc)})
        print('FAIL', args, exc, flush=True)
    else:
        print('PASS', args, results[-1]['steady_seconds_per_M0'], flush=True)
    (root/'results.json').write_text(json.dumps(dict(results=results, failures=failures),indent=2)+'\n')


memory_tool = root/'memory-tool.so'
subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC',
                '-I'+str(root/'source/kokkos/core/src'), str(source/'aurora/memory.cpp'),
                '-o', str(memory_tool)], check=True)
os.environ['KOKKOS_TOOLS_LIBS'] = str(memory_tool)

# Cover both ends of the strong-scaling curve before the intermediate sizes.
for ranks in (2, 24):
    for mode in ('metis', 'uniform'):
        measure('adaptive88', ranks, mode, 8)
    restart_dir = root/'runs'/f'adaptive-restart-{ranks}tiles'
    if not (restart_dir/'results.json').exists():
        with (root/f'adaptive-restart-{ranks}tiles.log').open('w') as log:
            status = subprocess.run([sys.executable, str(source/'adaptive.py'), '--binary', str(exe),
                                     '--output', str(restart_dir), '--ranks', str(ranks),
                                     '--launcher', shlex.join(launcher(ranks))],
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            failures.append(dict(case=['adaptive-restart',ranks],error=f'exit {status}'))
for mode in ('metis', 'uniform', 'scalar', 'classical'):
    measure('current754', 24, mode, 6)
for ranks in (2, 4, 8, 12, 24):
    for mesh, count in [('adaptive88',8), ('current754',6)]:
        for mode in ('metis','uniform','scalar'):
            if not any((item['mesh'], item['tiles'], item['mode'])==(mesh,ranks,mode) for item in results):
                measure(mesh,ranks,mode,count)
analysis_python = root.parent/'n12-full-scri-20261115/analysis-env/bin/python'
subprocess.run([str(analysis_python), str(source/'aurora/compare.py'), str(root)], check=True)
sys.exit(bool(failures))
