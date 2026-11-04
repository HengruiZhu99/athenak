"""Bounded equal/weighted static-ownership comparison in one PBS allocation.

Both cases restart the same admitted state. No checkpoint from this benchmark
is automatically used by a physics campaign. Uses frozen RF2 verification.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time

root = Path(sys.argv[1]).resolve()
pf = (root / sys.argv[2]).resolve()
sys.path.insert(0, str(root / 'launcher-v7'))
from control import read, write, digest, bindings, verify
from worker import check_affinity
from storage import capacity

plan = read(pf)
bindings(root, plan)
assert plan['purpose'] == 'static_ownership_pair'
assert os.environ.get('PBS_JOBID')
submission = read(pf.parent / 'submission.json')
assert submission['job_id'] == os.environ['PBS_JOBID']
assert submission['plan_sha256'] == digest(pf)
nodes = set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines())
assert len(nodes) == plan['nodes'] == 4 and plan['ranks'] == 48
out = root / plan['output']
out.mkdir(exist_ok=False)
state = dict(status='running', job_id=os.environ['PBS_JOBID'],
             plan_sha256=digest(pf), started=time.time(), cases=[], artifacts={})
write(out / 'worker.json', state)
cpu = '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpi = ['mpiexec', '-n', '48', '-ppn', '12', '--cpu-bind=list:' + cpu,
       'gpu_tile_compact.sh']
started = time.monotonic()
stop = threading.Event()
guard_result = {}

def guard():
    while not stop.is_set():
        try:
            guard_result['latest'] = capacity(root,
                planned=plan['planned_output_bytes'],
                reserve=plan['atomic_reserve_bytes'])
        except Exception as e:
            guard_result['error'] = repr(e)
            return
        stop.wait(60)

code = 2
try:
    state['storage_before'] = capacity(root, planned=plan['planned_output_bytes'],
                                       reserve=plan['atomic_reserve_bytes'])
    threading.Thread(target=guard, daemon=True).start()
    affinity = subprocess.run(mpi + ['bash', '-c',
        'printf "RF2AFF %s %s\\n" "$(hostname)" "$ZE_AFFINITY_MASK"'],
        capture_output=True, text=True, timeout=60)
    (out / 'affinity.log').write_text(affinity.stdout + affinity.stderr)
    assert affinity.returncode == 0
    placements = check_affinity(affinity.stdout, 4, 48)
    assert set(placements) == {n.split('.')[0] for n in nodes}
    state['verified_placements'] = placements
    assert len(plan['cases']) == 2
    for case in plan['cases']:
        # Exact scientific/time/diagnostic parameters must match between cases.
        for key in ('n', 'dt', 'identity', 'config', 'restart', 'start_time',
                    'start_steps', 'target_time', 'diagnostic_interval', 'binary'):
            assert case[key] == plan['cases'][0][key], key
        bindings(root, case)
        case_out = root / case['output']
        case_out.mkdir(exist_ok=False)
        env = dict(os.environ, HYP3D_ANALYSIS_PARTITION=str(root / case['partition']))
        command = mpi + [str(root / case['binary']), str(case['n']), 'boost', '0',
            str(case['dt']), str(case_out / 'state'), str(root / case['restart']),
            'smr', str(root / case['config']), '900',
            str(case['diagnostic_interval']), str(case['target_time'])]
        entry = dict(name=case['name'], command=command, started=time.time(),
                     partition_sha256=digest(root / case['partition']))
        state['cases'].append(entry)
        write(out / 'worker.json', state)
        begin = time.monotonic()
        with (case_out / 'native.log').open('w') as f:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=f,
                                       stderr=subprocess.STDOUT)
            while process.poll() is None:
                used = sum(max(p.stat().st_size, p.stat().st_blocks * 512)
                           for p in out.rglob('*') if p.is_file())
                error = guard_result.get('error')
                if used > plan['planned_output_bytes'] + plan['atomic_reserve_bytes']:
                    error = 'output and atomic reserve exceeded'
                if time.monotonic() - started > 2900 or time.monotonic() - begin > 1050:
                    error = 'bounded benchmark time exceeded'
                if error:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError(error)
                time.sleep(2)
        entry.update(native_returncode=process.returncode,
                     native_seconds=time.monotonic() - begin)
        assert process.returncode == 0, 'Benchmark must complete; no automatic restart'
        verification = verify(root, case, case_out)
        assert verification['status'] == 'completed'
        assert verification['steps'] == case['start_steps'] + plan['measured_steps']
        write(case_out / 'checkpoint-verification.json', verification)
        entry['verification'] = verification
        write(out / 'worker.json', state)
    state['checkpoint_byte_identical'] = (
        state['cases'][0]['verification']['sha256'] ==
        state['cases'][1]['verification']['sha256'])
    state['status'] = 'completed'
    state['qualification'] = ('Bounded ownership experiment only. Compare stepping '
        'timers separately from setup/verification/I/O. Adoption requires physical '
        'checkpoint parity and measured improvement; no strong-scaling claim.')
    code = 0
except Exception as e:
    state.update(status='terminal_failure', error=repr(e))
finally:
    stop.set()
    state.update(finished=time.time(), storage_guard=guard_result)
    for p in sorted(out.rglob('*')):
        if p.is_file() and p != out / 'worker.json':
            state['artifacts'][str(p.relative_to(out))] = dict(
                bytes=p.stat().st_size, sha256=digest(p))
    write(out / 'worker.json', state)
sys.exit(code)
