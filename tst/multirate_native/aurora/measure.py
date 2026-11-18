"""Measure matched endpoint evolution on explicitly bound Aurora GPU tiles."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive import adaptive_input
from transport_modes import TRANSPORT, transport_parameters
from binding import launcher, rank_devices
from activity import snapshot, utilization
from node_placement import reorder_nodes

root = Path(sys.argv[1]).resolve()
source = root / 'source/tst/multirate_native'
campaign = root.parent / 'n12-radial-blocks-20261117'
large_suite = sys.argv[2:] == ['--larger-adaptive']
throughput = sys.argv[2:] == ['--throughput']
case_file = None
if len(sys.argv)==4 and sys.argv[2]=='--cases':
    case_file = Path(sys.argv[3]).resolve()
elif sys.argv[2:] and not (large_suite or throughput):
    raise ValueError('expected --larger-adaptive, --throughput or --cases FILE')
exe = Path(os.environ.get('MR_BENCHMARK_EXECUTABLE',root/'build/src/athena')).resolve()
pinned_sha = None
if case_file:
    specification = json.loads(case_file.read_text())
    cases = specification.get('cases',[]) if isinstance(specification,dict) else specification
    selected_suite = (any(row[2].startswith('selected') for row in cases) or
                      isinstance(specification,dict) and ('accuracy' in specification or 'mpi_comparison' in specification))
    if selected_suite:
        # Every scaling/accuracy control follows the binary that actually evolved
        # the N12 state. Reconfiguration can relink a new output-provenance object.
        from n12_paths import trajectory_directory
        state=json.loads((trajectory_directory(root,'multirate')/'state.json').read_text())
        pinned_sha=state['binary_sha256']
        exe=Path(os.environ.get('MR_BENCHMARK_EXECUTABLE',state['executable'])).resolve()
with exe.open('rb') as binary:
    binary_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
if pinned_sha is not None and binary_sha256!=pinned_sha:
    raise ValueError('benchmark executable differs from the accepted N12 trajectory')
campaign_report = (case_file.stem+'-results.json' if case_file else
                   'throughput-results.json' if throughput else
                   'larger-results.json' if large_suite else 'results.json')
results = []
failures = []


def case(mesh, ranks, mode, intervals):
    # Instrumented runs have separate receipts and never enter elapsed-time comparisons.
    device_profile = mode.endswith('_trace')
    whole_gpu = mode.endswith('_whole_gpu')
    processes = ranks//2 if whole_gpu else ranks
    method = mode.removesuffix('_trace').removesuffix('_sibling_batch').removesuffix('_whole_gpu')
    deep_adaptive = mesh == 'adaptive754'
    production = mesh == 'current754' or deep_adaptive
    adaptive = mesh in ('adaptive88','adaptive1024')
    uniform = method in ('uniform', 'classical')
    base = 4*4.6944203735617445e-4 if production else 2**-14
    factor = (16 if production else 2) if uniform else 1
    runs = root/'runs'
    series = os.environ.get('MR_BENCHMARK_SERIES')
    if series:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',series):
            raise ValueError('invalid benchmark series name')
        runs = runs/series
    run = runs/f'{mesh}-{ranks}tiles-{mode}'
    complete = run/'completed.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['binary_sha256']==binary_sha256 and receipt.get('intervals')==intervals:
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
        mesh_cells = 120 if mesh == 'adaptive1024' else 48
        inp = inp[:start]+inp[start:end].replace('= 12', f'= {mesh_cells}')+inp[end:]
        capacity = 1400 if mesh == 'adaptive1024' else 130
        inp += f'\n<mesh_refinement>\nmax_nmb_per_rank = {max(12, (capacity+ranks-1)//ranks)}\n'
    else:
        inp = (campaign/'input.athinput' if production else source/'small.athinput').read_text()
    if deep_adaptive:
        # Start from the deployed initial atlas, then refine an off-puncture
        # Cartesian leaf at the first common physical cut in both controls.
        inp += f"""
<mesh_refinement>
refinement = adaptive
num_levels = 5
max_nmb_per_rank = {max(64,(1000+ranks-1)//ranks)}
ncycle_check = {factor}
refinement_interval = 0
<amr_criterion0>
method = location
location_x1 = .053
location_x2 = .041
location_x3 = .037
location_rad = .0001
"""
    inp += '\n<time>\nintegrator = '+('rk4_classical' if method=='classical' else 'mrgark4')
    inp += f'\ndt = {base/factor:.17g}\nnlim = {intervals*factor}\ntlim = 1\nndiag = 1\n'
    if method != 'classical':
        regions = ('2,2,2,1,1,1,1,1,2,3,4' if production else
                   '1,1,0,1' if adaptive else '0,0,0,1')
        balance = method if method in ('scalar','metis','spread') else 'none'
        if method in TRANSPORT:
            balance = 'spread'
        if method in ('posted_fixed','polled_fixed','spread_fixed') or (method in TRANSPORT and mesh == 'current754' and ranks == 24):
            baseline = 'spread' if method in TRANSPORT or method=='spread_fixed' else 'metis'
            balance = 'none'
            with (root/'runs'/f'current754-24tiles-{baseline}'/'partition.log').open('rb') as stream:
                record=stream.read(int(stream.readline())).decode().splitlines()
            owners=[int(line.split()[1]) for line in record[1:]]
            if method == 'locality':
                measured=next(x for x in results if x['mode']=='cohort' and x['tiles']==ranks)
                # JSON receipts persist rank IDs as strings.
                measured=dict(measured,rank_binding={str(k):v for k,v in measured['rank_binding'].items()})
                owners,traffic=reorder_nodes(owners,measured)
                (run/'node-placement.json').write_text(json.dumps(traffic,indent=2)+'\n')
            if mode.endswith('_sibling_batch'):
                control=next((r for r in results if r['mode']==method and r['tiles']==ranks),None)
                if control is None:
                    raise ValueError('missing same-allocation sibling control')
                python=root.parent/'n12-full-scri-20261115/analysis-env/bin/python'
                profile=json.loads(subprocess.check_output([str(python),str(source/'aurora/node_placement.py'),
                                   str(Path(control['log']).with_name('receipt.json'))],text=True))
                if owners!=profile['original']:
                    raise ValueError('sibling batching control differs from the spread layout')
                (run/'sibling-placement.json').write_text(json.dumps(profile,indent=2)+'\n')
                owners=profile['owners']
            if whole_gpu:
                owners=[tile_rank//2 for tile_rank in owners]
            ownership=','.join(map(str,owners))
            inp += f'\n<multirate>\nblock_owners = {ownership}\n'
        inp += (f'<multirate>\nspherical_levels = {regions}\nuniform_reference = {str(uniform).lower()}\n'
                f'{transport_parameters(method)}profile_peer_bytes = true\nbalance = {balance}\nranks_per_gpu = {1 if whole_gpu else 2}\nprepost_exchanges = {str(mode!="polled_fixed").lower()}\nwarmup_intervals = 2\nreview_interval = 32\n'
                f'partition_log = {run}/partition.log\n')
    (run/'input.athinput').write_text(inp)
    launch = launcher(processes,whole_gpu)
    with (run/'binding.log').open('w') as log:
        binding = ('echo "$(hostname) tile=$ZE_AFFINITY_MASK '
                   'rank=${PMI_RANK:-${PALS_RANKID:-${PMIX_RANK:-unknown}}}"')
        subprocess.run([*launch, '/bin/bash', '-c', binding],
                       stdout=log, stderr=log, check=True)
    rank_binding = rank_devices((run/'binding.log').read_text(),processes,whole_gpu)
    command = [*launch]
    if device_profile:
        tools = sorted(Path('/opt/aurora/default/spack/unified').glob(
            '*/install/linux-x86_64/pti-gpu-0.17.0-*/bin/unitrace'))
        if not tools:
            raise RuntimeError('Aurora unitrace was unavailable')
        command += ['/usr/bin/python3.11',str(source/'aurora/device_profile.py'),
                    str(tools[0]),str(run),str(2*factor)]
    command += [str(exe), '-i', str(run/'input.athinput')]
    nodes = len({host for host,_ in rank_binding.values()})
    reserved_nodes = len(set(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    if reserved_nodes<nodes:
        raise ValueError('active ranks span more nodes than the PBS reservation')
    counters = snapshot(root,run,"start",nodes)
    start = time.monotonic()
    with (run/'run.log').open('w') as log:
        code = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic()-start
    activity = utilization(counters,snapshot(root,run,"end",nodes))
    text = (run/'run.log').read_text()
    final = re.search(r'^time=([^ ]+) cycle=(\d+)$', text, re.M)
    if code or not final or int(final[2]) != intervals*factor or abs(float(final[1])-intervals*base)>1.e-13:
        raise RuntimeError(f'GPU evolution failed in {run} (exit {code})')
    if not re.search(r'MULTIRATE_MEMORY space=SYCLDeviceUSM peak_allocated_bytes=[1-9]\d*', text):
        raise AssertionError('missing Kokkos GPU allocation receipt in '+str(run))
    regrids = re.findall(r'(?:MULTIRATE|SCHEME_A)_REGRID cycle=(\d+) created=(\d+) deleted=(\d+) blocks=(\d+)', text)
    if (adaptive or deep_adaptive) and not any(int(created) and int(cycle)==factor for cycle,created,_,_ in regrids):
        raise AssertionError('missing synchronized first regrid: '+str(run))
    # Use the driver clock for every method. Differences include AMR, migration
    # and diagnostics between cuts. Ignore warmup and report individual samples.
    cuts = {int(n):float(t) for t,n in re.findall(r'^elapsed=(\S+) cycle=(\d+) ',text,re.M)}
    samples = [cuts[n+factor]-cuts[n] for n in range(2*factor,intervals*factor,factor)
               if n in cuts and n+factor in cuts]
    if len(samples)!=intervals-2:
        raise AssertionError('missing post-warmup driver timing samples in '+str(run))
    endpoints = sorted((run/'rst').glob('*.rst'))
    if not endpoints:
        raise AssertionError('missing accepted checkpoint in '+str(run))
    receipt = dict(intervals=intervals, binary_sha256=binary_sha256, input_sha256=hashlib.sha256(inp.encode()).hexdigest(),
                   job_id=os.environ.get('PBS_JOBID'), mesh=mesh, mode=mode, tiles=ranks, mpi_ranks=processes, whole_gpu=whole_gpu, physical_gpus=ranks//2,
                   nodes=nodes, reserved_nodes=reserved_nodes, reserved_node_hours=reserved_nodes*wall/3600,
                   engine_activity=activity, rank_binding=rank_binding, simulated_M0=3*intervals*base,
                   seconds=wall, seconds_per_M0=wall/(3*intervals*base),
                   steady_base_seconds=samples, steady_seconds_per_M0=median(samples)/(3*base),
                   # Driver::Finalize writes all outputs before printing its final clock.
                   evolution_seconds_per_M0=median(samples[:-1])/(3*base) if len(samples)>1 else None,
                   final_base_interval_seconds=samples[-1],
                   command=command, regrids=regrids, log=str(run/'run.log'),
                   work=[line for line in text.splitlines() if line.startswith('MULTIRATE_')],
                   checkpoint=str(endpoints[-1]))
    if device_profile:
        receipt['device_profile_coarse_intervals'] = intervals-2
        from device_profile import device_totals
        receipt['device_profile'] = device_totals(run,ranks)
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
    (root/campaign_report).write_text(json.dumps(dict(results=results, failures=failures),indent=2)+'\n')


memory_tool = root/'memory-tool.so'
subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC',
                '-I'+str(root/'source/kokkos/core/src'), str(source/'aurora/memory.cpp'),
                '-o', str(memory_tool), '-ldl'], check=True)
os.environ['KOKKOS_TOOLS_LIBS'] = str(memory_tool)

if case_file:
    specification=json.loads(case_file.read_text())
    if isinstance(specification,dict) and 'mpi_comparison' in specification:
        analysis_python = root.parent/'n12-full-scri-20261115/analysis-env/bin/python'
        subprocess.run([str(analysis_python),str(source/'aurora/mpi_runtime.py'),str(root),
                        str(exe),str(case_file)],check=True)
        sys.exit(0)
    if isinstance(specification,dict) and 'accuracy' in specification:
        from accuracy import qualify
        checks=qualify(root,exe,specification['accuracy'])
        (root/campaign_report).write_text(json.dumps(dict(accuracy=checks),indent=2)+'\n')
        sys.exit(0)
    if isinstance(specification,dict) and 'series' in specification:
        os.environ['MR_BENCHMARK_SERIES']=specification['series']+'-'+os.environ['PBS_JOBID'].split('.')[0]
        specification=specification['cases']
    for mesh,ranks,mode,intervals in specification:
        if mode in ('selected','selected_sibling_batch','selected_whole_gpu'):
            from n12_paths import trajectory_directory
            directory = trajectory_directory(root,'multirate')
            selections = list(directory.glob('segment-*/placement-selection.json'))
            choices = {row['selected'] for path in selections
                       if (row:=json.loads(path.read_text()))['binary_sha256']==binary_sha256}
            if len(choices)!=1:
                raise ValueError('scaling requires one qualified placement for this executable')
            mode = choices.pop()+mode[len('selected'):]
        if mesh not in ('current754','adaptive754','adaptive1024','adaptive88') or ranks not in (2,4,8,12,24):
            raise ValueError('unsupported Aurora workload or tile count')
        untraced = mode.removesuffix('_trace').removesuffix('_sibling_batch').removesuffix('_whole_gpu')
        if mode.endswith(('_sibling_batch','_whole_gpu')) and (mesh,ranks)!=('current754',24):
            raise ValueError('sibling batching experiment requires the fixed two-node N12 mesh')
        if (untraced not in TRANSPORT and untraced not in ('spread_fixed','spread','metis','scalar','zorder',
                'uniform','classical')) or intervals<3:
            raise ValueError('unsupported benchmark method or insufficient timing intervals')
        measure(mesh,ranks,mode,intervals)
elif throughput:
    measure('current754',24,'cohort',6)
    if results and results[-1]['mode']=='cohort':
        measure('current754',24,'locality',6)
    measure('current754',24,'classical',6)
    measure('current754',24,'zorder',6)
elif large_suite:
    # Isolate transport and placement on the deployed deep hierarchy first.
    for mode in ('posted_fixed','spread','polled_fixed'):
        measure('current754',24,mode,6)
    for ranks,mode in [(24,'spread'),(2,'spread'),(24,'uniform'),(12,'spread')]:
        measure('adaptive754',ranks,mode,5)
    for mode in ('zorder','metis'):
        measure('current754',24,mode,6)
    for ranks,mode in [(24,'metis'),(2,'metis'),(24,'uniform'),(12,'metis')]:
        measure('adaptive1024',ranks,mode,6)
else:
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
subprocess.run([str(analysis_python), str(source/'aurora/compare.py'), str(root), campaign_report], check=True)
sys.exit(bool(failures))
