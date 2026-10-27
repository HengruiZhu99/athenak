"""Serial, bounded continuation of verified completed puncture checkpoints.

No retries, numerical overrides, or duplicate output roots. Each child uses a
frozen copy of the existing continuation harness. Completion here means only
that the requested trajectory finished, never that convergence is established.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from continue_puncture import atomic, digest


def verify(case):
    m = json.loads((case/'manifest.json').read_text())
    if m['status'] != 'completed' or not m['arguments'].get('puncture'):
        raise ValueError('Need a completed puncture parent')
    if m.get('stage_stats', {}).get('min_stage_shift_lapse_gap_times_g', 1.) <= 0:
        raise ValueError('Parent has a nonpositive stage speed margin')
    for name in ['profiles.npz', 'history.json']:
        if m.get('checkpoint_hashes', {}).get(name) != digest(case/name):
            raise ValueError('Checkpoint hash mismatch: '+name)
    for name, sha in m['sources'].items():
        if digest(case/'source'/name) != sha:
            raise ValueError('Source hash mismatch: '+name)
    lib = case/'source'/Path(m['arguments']['library']).name
    if digest(lib) != m['library_sha256']:
        raise ValueError('Library hash mismatch')
    if 'continuation_harness_sha256' in m:
        if digest(case/'continuation_harness.py') != m['continuation_harness_sha256']:
            raise ValueError('Continuation harness hash mismatch')
    if 'parent' in m:
        for name, sha in m['parent']['hashes'].items():
            if digest(Path(m['parent']['path'])/name) != sha:
                raise ValueError('Ancestry hash mismatch: '+name)
    with np.load(case/'profiles.npz') as z:
        if float(z['last_time']) != m['t'] or abs(m['last_accepted_step']*m['dt']-m['t']) > 1e-12:
            raise ValueError('Parent checkpoint time mismatch')
        if not np.isfinite(z['last']).all() or not np.isfinite(z['compensation']).all():
            raise ValueError('Nonfinite checkpoint')
    return m


def invariant(m):
    # These excluded inputs affect duration and observation, not the equations.
    args = {k:v for k,v in m['arguments'].items()
            if k not in ['output', 't', 'wall_seconds', 'report_dt']}
    return dict(arguments=args, dt=m['dt'], sources=m['sources'],
                library_sha256=m['library_sha256'])


def mass_drift(case):
    h = json.loads((case/'history.json').read_text())
    masses = np.array([r['apparent_horizon_mass'] for r in h])
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('Invalid horizon mass history')
    return float(np.max(abs(masses/masses[0]-1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--t', type=float, required=True)
    p.add_argument('--segment-duration', type=float, required=True)
    p.add_argument('--wall-seconds', type=float, default=420.)
    p.add_argument('--mass-drift-limit', type=float)
    a = p.parse_args()
    parent = a.parent.resolve()
    m = verify(parent)
    values = [a.t, a.segment_duration, a.wall_seconds]
    if not all(math.isfinite(v) and v > 0 for v in values) or a.wall_seconds > 420:
        raise ValueError('Invalid duration or per-segment wall limit (maximum420 seconds)')
    if a.mass_drift_limit is not None and not (math.isfinite(a.mass_drift_limit) and a.mass_drift_limit > 0):
        raise ValueError('Invalid mass-drift limit')
    dt = m['dt']
    target_step = int(round(a.t/dt))
    segment_steps = int(round(a.segment_duration/dt))
    if a.t <= m['t'] or abs(target_step*dt-a.t) > 1e-12:
        raise ValueError('Final target must lie on the inherited time lattice and exceed the parent time')
    if segment_steps < 1 or abs(segment_steps*dt-a.segment_duration) > 1e-12:
        raise ValueError('Segment duration must lie on the inherited time lattice')
    initial_drift = mass_drift(parent)
    if a.mass_drift_limit is not None and initial_drift > a.mass_drift_limit:
        raise ValueError('Parent already exceeds the requested mass-drift limit')
    anchor = invariant(m)
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    harness = root/'frozen_continuation.py'
    shutil.copy2(Path(__file__).with_name('continue_puncture.py'), harness)
    shutil.copy2(__file__, root/'frozen_series.py')
    record = dict(status='running', pid=os.getpid(), command=sys.argv,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  initial_parent=str(parent), initial_parent_sha256=digest(parent/'manifest.json'),
                  last_completed_parent=str(parent), target_t=a.t, target_step=target_step,
                  dt=dt, segment_steps=segment_steps, wall_seconds_limit=a.wall_seconds,
                  mass_drift_limit=a.mass_drift_limit,
                  invariant_sha256=hashlib.sha256(json.dumps(anchor,sort_keys=True).encode()).hexdigest(),
                  continuation_sha256=digest(harness), runner_sha256=digest(root/'frozen_series.py'),
                  active_child=None, segments=[],
                  scope='Serial execution status only; no convergence certificate. Failed, bounded, or interrupted children are never retried.')
    state = root/'series.json'
    atomic(state, record)
    process = None
    try:
        while m['last_accepted_step'] < target_step:
            if (root/'STOP').exists():
                record['status'] = 'stopped_after_completed_segment'
                break
            end_step = min(target_step, m['last_accepted_step']+segment_steps)
            target = end_step*dt
            child = root/f'segment-{len(record["segments"])+1:04d}'
            if digest(harness) != record['continuation_sha256']:
                raise ValueError('Frozen continuation harness changed')
            cmd = [sys.executable, str(harness), '--parent', str(parent),
                   '--output', str(child), '--t', repr(target),
                   '--wall-seconds', str(a.wall_seconds)]
            with child.with_suffix('.log').open('w') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                record['active_child'] = dict(path=str(child), pid=process.pid, command=cmd, target_t=target)
                atomic(state, record)
                print(json.dumps(dict(event='started', **record['active_child'])), flush=True)
                code = process.wait()
            record['active_child'] = None
            process = None
            cm = json.loads((child/'manifest.json').read_text()) if (child/'manifest.json').exists() else {}
            event = dict(path=str(child), target_t=target, exit_code=code,
                         status=cm.get('status', 'missing_manifest'), t=cm.get('t'), error=cm.get('error'))
            record['segments'].append(event)
            print(json.dumps(dict(event='terminal', **event)), flush=True)
            if code or cm.get('status') != 'completed':
                record['status'] = 'stopped_on_child_outcome'
                break
            checked = verify(child)
            if invariant(checked) != anchor or checked['last_accepted_step'] != end_step:
                raise ValueError('Child changed frozen invariants or missed the target')
            parent, m = child, checked
            record['last_completed_parent'] = str(parent)
            event['peak_relative_AH_mass_drift'] = mass_drift(child)
            if a.mass_drift_limit is not None and event['peak_relative_AH_mass_drift'] > a.mass_drift_limit:
                record['status'] = 'qualification_stop'
                record['error'] = 'Completed child exceeds requested horizon-mass drift limit; child outcome preserved'
                break
            atomic(state, record)
        else:
            record['status'] = 'completed'
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            record['interrupted_child'] = record['active_child']
            process.terminate()
            process.wait()
            record['interrupted_child']['exit_code'] = process.returncode
        record['status'] = 'interrupted'
        record['error'] = 'Interrupted; no child restart is allowed'
    except Exception as exc:
        if process is not None and process.poll() is None:
            record['interrupted_child'] = record['active_child']
            process.terminate()
            process.wait()
            record['interrupted_child']['exit_code'] = process.returncode
        record['status'] = 'harness_error'
        record['error'] = str(exc)
    finally:
        record['active_child'] = None
        record['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(state, record)
    print(json.dumps(dict(status=record['status'], last_completed_parent=record['last_completed_parent'],
                          segments=len(record['segments']), error=record.get('error'))), flush=True)
    if record['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
