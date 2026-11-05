"""Verify a downloaded terminal radial18 segment and retain compact evidence.

This neither admits new scientific states nor changes a checkpoint. Admission
comes from the frozen remote controller; all worker artifacts are rehashed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
R = next(p for p in D.parents if (p / 'CMakeLists.txt').exists())
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, choices=[12, 14, 16], required=True)
p.add_argument('--segment', type=int, required=True)
a = p.parse_args()
name = f'stationary20-radial18-resolution-n{a.n}-segment{a.segment:02d}'
raw = R / 'output/refined-convergence/stationary-reference-v2-aurora/campaigns' / name
out = D / f'radial18-n{a.n}-segment{a.segment:02d}'
out.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

plan = read(raw / 'plan.json')
terminal = read(raw / 'terminal-verification.json')
release = read(raw / 'allocation-released.json')
worker = read(raw / 'output/worker.json')
assert plan['variant'] == 'radial_resolution' and plan['n'] == a.n
assert plan['segment'] == a.segment and plan['dt'] == 1 / (2000 * a.n)
assert release['job_id'] == terminal['job_id'] == worker['job_id']
assert release['plan_sha256'] == worker['plan_sha256'] == sha(raw / 'plan.json')
assert release['terminal_verification_sha256'] == sha(raw / 'terminal-verification.json')
verified = {}
for name, v in worker['artifacts'].items():
    f = (raw / 'output' / name).resolve()
    assert f.is_relative_to((raw / 'output').resolve())
    assert f.stat().st_size == v['bytes'] and sha(f) == v['sha256'], name
    verified[name] = v
if terminal['admitted']:
    cp = terminal['checkpoint']
    assert cp['identity'] == plan['identity'] and cp['n'] == a.n
    assert cp['plan_sha256'] == sha(raw / 'plan.json')
    assert cp['sha256'] == verified['state.checkpoint']['sha256']
    assert terminal['scheduler_exit'] == terminal['native_returncode'] == (
        0 if cp['status'] == 'completed' else 3)
    assert worker['status'] == cp['status']
for name in ['plan.json', 'submission.json', 'binding-storage.json',
             'terminal-verification.json', 'allocation-released.json',
             'terminal-qstat.txt', 'pbs.log']:
    if (raw / name).exists():
        shutil.copy2(raw / name, out / name)
(out / 'output').mkdir(exist_ok=True)
for name in ['state.report.json', 'state.history.jsonl', 'worker.json',
             'checkpoint-verification.json', 'native.log']:
    if (raw / 'output' / name).exists():
        shutil.copy2(raw / 'output' / name, out / 'output' / name)
result = dict(all_worker_artifacts_verified=True, count=len(verified),
              artifacts=verified, admitted=terminal['admitted'],
              job_id=terminal['job_id'], matching_terminal_lock_released=True,
              terminal_verification_sha256=sha(raw / 'terminal-verification.json'),
              qualification='RF2 floor/sign counters are cumulative across restart; do not sum segment totals.')
(out / 'local-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}))
