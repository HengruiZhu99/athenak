#!/usr/bin/env python3
"""Recheck the frozen V3 preflight evidence without submitting or evolving."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = '86b602e7e68fc93e6a76e16c9bd5c4aea9fcef64363670c5f95c6b0adf30fc13'
BINARY = 'a557b2090b8c41c6999969a008f1731e743eea21caa3fa257af386ab15b1847d'
HEADER = struct.Struct('<16s4i3Q2d')


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_value(check, report):
    value = report
    for part in check['path']:
        value = value[part]
    op = check['op']
    if op == 'eq':
        passed = value == check['value']
    else:
        passed = isinstance(value, (int, float)) and math.isfinite(value)
        if passed and op == 'le':
            passed = value <= check['value']
        elif passed and op == 'ge':
            passed = value >= check['value']
        elif op not in ['finite', 'le', 'ge']:
            raise ValueError('Unknown acceptance operator')
    return value, bool(passed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, default=ROOT/'output/hyperbolic-core-shell/aurora/preflight-angular-v1')
    p.add_argument('--build-dir', type=Path, default=ROOT/'output/hyperbolic-core-shell/aurora/build-v10')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Output must be new')
    raw = args.raw_dir.resolve()
    worker = read(raw/'worker-status.json')
    binding = read(raw/'binding.json')
    plan = read(raw/'plan.json')
    build = read(args.build_dir/'v10/build-status.json')
    terminal = read(raw/'collected.json')
    release = read(raw/'allocation-released.json')
    require(terminal['scheduler']['job_state'] == 'F' and terminal['scheduler']['Exit_status'] == 0,
            'Preflight lacks successful terminal scheduler evidence')
    require(terminal['job_id'] == worker['job_id'] == release['job_id'], 'Job identity mismatch')
    affinity = worker['affinity']
    require(affinity['passed'] and affinity['returncode'] == 0 and not affinity['timed_out'],
            'Affinity probe failed')
    records = affinity['records']
    require(len(records) == 12 and {r['rank'] for r in records} == set(range(12)), 'Rank placement mismatch')
    require({r['mask'] for r in records} == {f'{i}.{j}' for i in range(6) for j in range(2)},
            'GPU tiles are not distinct')
    require(len({r['host'] for r in records}) == 1 and all(r['hierarchy'] == 'COMPOSITE' for r in records),
            'Node/device hierarchy mismatch')
    require(worker['storage_at_finish']['bytes'] < 5_000_000_000, 'Campaign storage cap exceeded')
    require(worker['source_manifest_sha256'] == binding['source_manifest_sha256'] == SOURCE,
            'Unexpected frozen source')
    require(worker['plan_sha256'] == binding['plan_sha256'] == sha(raw/'plan.json'), 'Plan mismatch')
    require(binding['pbs_sha256'] == sha(raw/'run_debug.pbs'), 'PBS mismatch')
    require(worker['binaries'] == binding['binaries'], 'Worker/binding binaries differ')
    require(BINARY in binding['binaries'].values(), 'Missing frozen V3 binary')
    require(build['status'] == 'passed' and build['returncode'] == 0, 'Build did not pass')
    require(not build['timed_out'] and build['storage_error'] is None, 'Build resource failure')
    require(build['source_manifest_sha256'] == sha(args.build_dir/'source-v10-manifest.json') == SOURCE,
            'Build/source manifest mismatch')
    require(build['cache_sha256'] == sha(args.build_dir/'CMakeCache.txt'), 'Build cache changed')
    require(binding['inputs']['build-sycl-mpi-v10/CMakeCache.txt'] == build['cache_sha256'],
            'Preflight used a different build cache')
    for name in set(binding['binaries']) | set(binding['inputs']):
        if name.startswith('build-sycl-mpi-v10/tst/'):
            require(name in build['binaries'], 'Bound executable absent from build receipt')
    for name, digest in build['binaries'].items():
        if name in binding['binaries']:
            require(binding['binaries'][name] == digest, 'Built executable hash mismatch')
        if name in binding['inputs']:
            recorded = binding['inputs'][name]
            require((recorded.get('sha256') if isinstance(recorded, dict) else recorded) == digest,
                    'Component executable hash mismatch')
    commands = {c['name']: c for c in plan['commands']}
    require(len(commands) == len(plan['commands']) and
            [c['name'] for c in worker['cases']] == [c['name'] for c in plan['commands']],
            'Missing, duplicate or reordered cases')
    artifacts = []
    checkpoints = []
    checks_count = 0
    reports = {}
    prior_artifacts = {}
    for case in worker['cases']:
        command = commands[case['name']]
        report = read(raw/'output'/Path(command['report']).name)
        reports[case['name']] = report
        for remote, expected in case['generated_inputs'].items():
            require(remote in prior_artifacts and prior_artifacts[remote]['sha256'] == expected,
                    'Generated restart input binding mismatch')
        require([x['check'] for x in case['checks']] == command['checks'], 'Changed acceptance checks')
        for record in case['checks']:
            observed, passed = check_value(record['check'], report)
            require(observed == record['observed'] and passed == record['passed'], 'Check replay mismatch')
            checks_count += 1
        for remote, expected in case['artifacts'].items():
            path = raw/'output'/Path(remote).name
            actual = dict(sha256=sha(path), bytes=path.stat().st_size)
            require(actual == expected, 'Artifact hash/size mismatch: '+remote)
            artifacts.append(dict(remote_path=remote, **actual))
            prior_artifacts[remote] = actual
        if command['kind'] != 'check':
            path = raw/'output'/(Path(command['args'][1]).name+'.checkpoint')
            with path.open('rb') as f:
                magic,n,nf,kind,admitted,cells,surface_size,steps,t,dt = HEADER.unpack(f.read(72))
            require((magic.rstrip(b'\0'),n,nf,cells,surface_size) ==
                    (b'HYP3D_MPI_V3',16,31,19*16**3,31*6*16**2), 'Unexpected checkpoint header')
            require(path.stat().st_size == 72+16*(31*cells+surface_size), 'Checkpoint length mismatch')
            require(steps == report['steps'] and t == report['t'], 'Checkpoint/report clock mismatch')
            require(kind == {'minkowski':0, 'pulse':1, 'puncture':2}[command['kind']] and
                    dt == float(command['args'][4]), 'Checkpoint kind/timestep mismatch')
            require(bool(admitted) == report['checkpoint_admitted'], 'Checkpoint admission mismatch')
            checkpoints.append(dict(case=case['name'], kind=kind, admitted=bool(admitted), steps=steps, t=t, dt=dt,
                                    sha256=sha(path)))
    direct = raw/'output/pulse-direct.checkpoint'
    resumed = raw/'output/pulse-resumed.checkpoint'
    bitwise = direct.read_bytes() == resumed.read_bytes() if direct.exists() and resumed.exists() else False
    passed = (worker['status'] == 'passed' and len(worker['cases']) == len(commands) == 12
              and all(c['status'] == 'passed' and c['returncode'] == 0 and not c['timed_out']
                      and c['storage_error'] is None and all(x['passed'] for x in c['checks'])
                      for c in worker['cases']) and bitwise)
    result = dict(passed=passed, source_manifest_sha256=SOURCE, binary_sha256=BINARY,
                  worker_sha256=sha(raw/'worker-status.json'), binding_sha256=sha(raw/'binding.json'),
                  plan_sha256=sha(raw/'plan.json'), artifacts_verified=len(artifacts), checks_recomputed=checks_count,
                  all_recorded_checks_reproduced=True, full_checkpoint_byte_equality=bitwise,
                  terminal_exit_status=0, allocation_released=True, one_node_twelve_tiles_verified=True,
                  storage_at_finish=worker['storage_at_finish'], verifier_sha256=sha(Path(__file__)),
                  checkpoints=checkpoints, artifacts=artifacts,
                  scope='Component/short evolution/restart preflight only; no duration or convergence claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','checkpoints']}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
