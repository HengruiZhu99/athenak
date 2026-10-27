"""Test serial orchestration against an existing verified short RK oracle."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--fixture', type=Path, required=True,
               help='Completed output of test_continuation.py, containing half and full12')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
parent = a.fixture/'half'
oracle = a.fixture/'full12'
start = json.loads((parent/'manifest.json').read_text())['t']
target = json.loads((oracle/'manifest.json').read_text())['t']
duration = (target-start)/2
runner = Path(__file__).with_name('continue_series.py')

def run(parent, out, wall='30', mass_limit=None):
    cmd = [sys.executable, str(runner), '--parent', str(parent),
        '--output', str(out), '--t', repr(target), '--segment-duration', repr(duration),
        '--wall-seconds', wall]
    if mass_limit is not None:
        cmd += ['--mass-drift-limit', repr(mass_limit)]
    return subprocess.run(cmd, capture_output=True, text=True)

good = a.output/'good'
r = run(parent, good)
assert r.returncode == 0, r.stdout+r.stderr
s = json.loads((good/'series.json').read_text())
assert s['status'] == 'completed' and len(s['segments']) == 2
z = np.load(Path(s['last_completed_parent'])/'profiles.npz')
reference = np.load(oracle/'profiles.npz')
assert np.array_equal(z['last'], reference['last'])
assert np.array_equal(z['compensation'], reference['compensation'])

h = json.loads((parent/'history.json').read_text())
parent_drift = max(abs(x['apparent_horizon_mass']/h[0]['apparent_horizon_mass']-1) for x in h)
child_drift = s['segments'][0]['peak_relative_AH_mass_drift']
assert child_drift > parent_drift
qualification = a.output/'qualification'
r = run(parent, qualification, mass_limit=(parent_drift+child_drift)/2)
qs = json.loads((qualification/'series.json').read_text())
assert r.returncode and qs['status'] == 'qualification_stop' and len(qs['segments']) == 1
assert qs['segments'][0]['status'] == 'completed'
assert not (qualification/'segment-0002').exists()
assert json.loads((Path(qs['last_completed_parent'])/'manifest.json').read_text())['status'] == 'completed'

bounded = a.output/'bounded'
r = run(parent, bounded, wall='0.000001')
s = json.loads((bounded/'series.json').read_text())
assert r.returncode != 0 and s['status'] == 'stopped_on_child_outcome'
assert len(s['segments']) == 1 and s['segments'][0]['status'] == 'bounded_stop'
assert not (bounded/'segment-0002').exists()
assert json.loads((Path(s['segments'][0]['path'])/'manifest.json').read_text())['status'] == 'bounded_stop'

failed = a.output/'failed-parent'
failed.mkdir()
m = json.loads((parent/'manifest.json').read_text())
m['status'] = 'failed'
(failed/'manifest.json').write_text(json.dumps(m))
r = run(failed, a.output/'rejected-failed')
assert r.returncode and 'completed puncture parent' in r.stderr
assert not (a.output/'rejected-failed').exists()

corrupt = a.output/'corrupt-parent'
shutil.copytree(parent, corrupt)
with (corrupt/'history.json').open('a') as f:
    f.write(' ')
r = run(corrupt, a.output/'rejected-corrupt')
assert r.returncode and 'Checkpoint hash mismatch' in r.stderr
assert not (a.output/'rejected-corrupt').exists()

before = (good/'series.json').read_bytes()
r = run(parent, good)
assert r.returncode and 'FileExistsError' in r.stderr
assert (good/'series.json').read_bytes() == before
result = dict(two_segment_state_and_compensation_bitwise_equal=True,
              bounded_child_stops_without_retry=True, bounded_status_preserved=True,
              failed_parent_rejected=True, corrupt_checkpoint_rejected=True,
              existing_series_not_relaunched=True,
              mass_budget_stop_preserves_completed_child=True,
              scope='Serial orchestration checks on the existing short RK oracle; not long-time stability or convergence evidence.')
(a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
