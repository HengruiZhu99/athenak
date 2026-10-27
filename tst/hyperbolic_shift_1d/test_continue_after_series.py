"""Exercise prerequisite and serial failure gates without evolving a PDE."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from continue_after_series import check_gate, run_jobs
from continue_puncture import atomic, digest
from continue_series import verify

p = argparse.ArgumentParser()
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
parent = a.fixture.resolve()
m = verify(parent)
a.output.mkdir(parents=True, exist_ok=False)
root = a.output.resolve()
sleeper = root/'sleeper.py'
sleeper.write_text('import time\ntime.sleep(30)\n')
process = subprocess.Popen([sys.executable, str(sleeper)])
anchor = dict(pid=process.pid, command=[str(sleeper)], started_utc='test',
              initial_parent_sha256=digest(parent/'manifest.json'), invariant_sha256='test',
              target_t=m['t'], runner_sha256='test', continuation_sha256='test')
gate = root/'gate.json'
state = dict(anchor, status='running', active_child=None, last_completed_parent=str(parent))
atomic(gate, state)
assert check_gate(gate, anchor) is False
process.terminate()
process.wait()

def rejected(call, message):
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError('Expected rejection: '+message)

rejected(lambda:check_gate(gate, anchor), 'process identity is missing')
state['status'] = 'stopped_on_child_outcome'
atomic(gate, state)
rejected(lambda:check_gate(gate, anchor), 'without completing')
state['status'] = 'completed'
atomic(gate, state)
assert check_gate(gate, anchor) is True
changed = dict(anchor, invariant_sha256='changed')
rejected(lambda:check_gate(gate, changed), 'identity changed')
state['active_child'] = {'pid':123}
atomic(gate, state)
rejected(lambda:check_gate(gate, anchor), 'active child')

# An actual subprocess stub records order and outcomes but performs no evolution.
runner = root/'stub_runner.py'
runner.write_text('''import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir()
parent = sys.argv[sys.argv.index('--parent')+1]
failed = out.name == 'bad'
(out/'series.json').write_text(json.dumps(dict(status='stopped_on_child_outcome' if failed else 'completed',last_completed_parent=parent)))
sys.exit(1 if failed else 0)
''')
hashes = {runner.name:digest(runner)}
job = dict(parent=str(parent), parent_manifest_sha256=digest(parent/'manifest.json'),
           t=m['t'], segment_duration=m['dt'])
good = root/'serial-good';good.mkdir()
record = dict(finished=[], active_series=None)
run_jobs(good, [dict(job,name='first'),dict(job,name='second')], record, runner, hashes)
assert [r['name'] for r in record['finished']] == ['first','second']
bad = root/'serial-failure';bad.mkdir()
record = dict(finished=[], active_series=None)
rejected(lambda:run_jobs(bad,[dict(job,name='first'),dict(job,name='bad'),dict(job,name='never')],record,runner,hashes), 'did not complete')
assert [r['name'] for r in record['finished']] == ['first','bad']
assert not (bad/'never').exists()
changed = root/'changed-parent';changed.mkdir()
rejected(lambda:run_jobs(changed,[dict(job,name='never',parent_manifest_sha256='wrong')],dict(finished=[]),runner,hashes), 'parent changed')
assert not (changed/'never').exists()
result = dict(live_prerequisite_waits=True, missing_process_never_releases=True,
              failed_prerequisite_never_releases=True, completed_target_verified=True,
              changed_identity_rejected=True, active_child_rejected=True,
              actual_subprocess_jobs_serial=True, failure_stops_later_jobs=True,
              changed_temporal_parent_rejected=True,
              scope='Scheduling tests use real process stubs and verified existing checkpoints; no PDE evolution or long-time accuracy claim.')
atomic(root/'result.json', result)
print(json.dumps(result))
