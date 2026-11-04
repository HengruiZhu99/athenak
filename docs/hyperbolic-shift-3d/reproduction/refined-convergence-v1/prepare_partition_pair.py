"""Bind a short ownership-only GPU pair after the radial control is admitted.

This prepares files only. The existing reviewed submit helper performs the
single qsub, queue checks, storage guard, and shared allocation lock.
"""
from pathlib import Path
import bisect
import json
import sys

root = Path(sys.argv[1]).resolve()
reference_file = root / sys.argv[2]
package = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'launcher-v7'))
from control import read, write, digest, bindings, verify
from storage import capacity

reference = read(reference_file)
terminal = read(reference_file.parent / 'terminal-verification.json')
assert reference['variant'] == 'radial'
assert terminal['admitted'] and terminal['checkpoint']['status'] == 'completed'
release = read(reference_file.parent / 'allocation-released.json')
assert release['terminal_verification_sha256'] == digest(reference_file.parent / 'terminal-verification.json')
endpoint = verify(root, reference, root / reference['output'])
assert abs(endpoint['time'] - .2) < 1e-12
build = read(package / 'build-result.json')
assert build['status'] == 'completed' and build['production_unchanged']
assert digest(package / 'probe') == build['binary_sha256']
assert not (root / 'allocation-lock.json').exists()
rel = 'campaigns/stationary20-radial18-ownership-pair-v1'
directory = root / rel
directory.mkdir(exist_ok=False)

old = [0]
cumulative = [0.]
inputs = {}
for rank in range(48):
    p = root / reference['output'] / f'state.rank{rank}.generations.jsonl'
    inputs[str(p.relative_to(root))] = digest(p)
    row = json.loads(p.read_text().splitlines()[-1])
    old.append(old[-1] + row['active'])
    seconds = row['nested_seconds']['rhs'] + row['nested_seconds']['halo']
    assert seconds > 0
    cumulative.append(cumulative[-1] + seconds)
cuts = [0]
for i in range(1, 48):
    target = cumulative[-1] * i / 48
    j = bisect.bisect_right(cumulative, target) - 1
    fraction = (target - cumulative[j]) / (cumulative[j + 1] - cumulative[j])
    cuts.append(round(old[j] + fraction * (old[j + 1] - old[j])))
cuts.append(old[-1])
assert all(b > a for a, b in zip(cuts, cuts[1:]))
proposal = dict(inputs=inputs, original_cuts=old, weighted_cuts=cuts,
    fit='Piecewise constant RHS+halo cost per original owned point',
    qualification='No predicted speedup claim. New communication/occupancy can invalidate this proxy.')
write(directory / 'proposal.json', proposal)
for name, values in [('equal', old), ('weighted', cuts)]:
    (directory / f'{name}.partition').write_text(
        f'HYP3D_COST_PARTITION_V1 16 {old[-1]} 48 {digest(directory / "proposal.json")}\n' +
        ' '.join(map(str, values)) + '\n')
package_rel = str(package.relative_to(root))
pbs = (reference_file.parent / 'job.pbs').read_text()
pbs = pbs.replace('#PBS -N rf2-dt4-radial', '#PBS -N rf2-ownership-pair')
pbs = pbs.replace(str(reference_file.parent.relative_to(root)), rel)
pbs = pbs.replace('launcher-v7/worker.py', package_rel + '/partition_pair_worker.py')
(directory / 'job.pbs').write_text(pbs)
files = set(reference['bindings'])
files.update(str(p.relative_to(root)) for p in package.iterdir() if p.is_file())
files.update([reference['output'] + '/state.checkpoint',
              str(reference_file.relative_to(root)),
              str((reference_file.parent / 'terminal-verification.json').relative_to(root)),
              rel + '/job.pbs', rel + '/proposal.json',
              rel + '/equal.partition', rel + '/weighted.partition'])
files.update(inputs)
bound = {name: digest(root / name) for name in sorted(files)}
plan = dict(plan=rel + '/plan.json', purpose='static_ownership_pair',
            nodes=4, ranks=48, output=rel + '/output', measured_steps=256,
            queue='debug-scaling', planned_output_bytes=2_800_000_000,
            atomic_reserve_bytes=1_200_000_000, bindings=bound,
            source_commit=reference['source_commit'], cases=[])
for name in ('equal', 'weighted'):
    case = dict(reference)
    case.update(name=name, plan=plan['plan'], start_time=endpoint['time'],
                start_steps=endpoint['steps'],
                restart=reference['output'] + '/state.checkpoint',
                target_time=endpoint['time'] + 256 * reference['dt'],
                diagnostic_interval=reference['diagnostic_interval'],
                binary=package_rel + '/probe', bindings=bound,
                partition=rel + f'/{name}.partition',
                output=rel + f'/output/{name}',
                analysis_modification='Static constructor ownership probe, not original production binary')
    plan['cases'].append(case)
write(directory / 'binding-storage.json', capacity(root,
      planned=plan['planned_output_bytes'], reserve=plan['atomic_reserve_bytes']))
write(directory / 'plan.json', plan)
print(json.dumps(dict(plan=plan['plan'], sha256=digest(directory / 'plan.json'),
                     prepared_only=True, submitted=False)))
