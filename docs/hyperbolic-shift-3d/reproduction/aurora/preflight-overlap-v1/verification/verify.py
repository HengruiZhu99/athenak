#!/usr/bin/env python3
"""Read-only verification of the pinned V9 preflight; never submits or evolves."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = 'a6dfbd4474e653b7a15a8df39d3c79f5e66e0a13'
EXPECTED_SOURCE = 'd959a7cb8a587314d80ba6af7af9d692ddbef1815925f09bb46ef062429d3641'
EXPECTED_PLAN = '1c97cb408ae9ccdee8b2ea556e1e037962631afa2b4b35a6ae6be33d93b928c8'
ARCHIVE = '5c81329c967957cc7d62b915da871b2ac9f99ab06cf1492177d78674cd77c84b'
SOURCE = None
PLAN = None
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'
INITIAL = '46ae2bd2b2931f48fbf9b84af64b89cb0792dfceabb98c93cdb7060dd2a1896d'
BINARIES = {}
REMOTE = '/home/hzhu/hyperbolic-core-shell-20261028/'
CAMPAIGN = 'campaigns/preflight-overlap-v1/output/'
CASE_ORDER = ['parity-overlap', 'minkowski-short', 'pulse-half',
              'pulse-resumed', 'pulse-direct', 'restart-audit', 'puncture-short']
HEADER = struct.Struct('<16s4i3Q2d')
CONFIG = struct.Struct('<dii')
NORMAL = struct.Struct('<4i')
TARGETS = ['mpi_overlap','overlap_device']
NORMAL_FIELDS = dict(scri_normal_derivative='sampled_native_numerator7',
                     scri_normal_radial_nodes=7,scri_normal_bulk_derivative_order=4,
                     scri_radial_closure='donor_seven_last3',scri_radial_bulk_rows=3,
                     interpatch_interpolation='logical_inner_transition',
                     evolved_overlap=3)
ARRAY_NAMES = ['volume_state', 'volume_compensation', 'scri_state', 'scri_compensation']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def git(repo, *args, data=None):
    return subprocess.run(['git', '-C', str(repo), *args], input=data,
                          check=True, capture_output=True).stdout


def git_hashes(repo, commit, paths):
    """Hash immutable Git blobs in one batch, avoiding mutable checkout files."""
    tree = {}
    for item in git(repo, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
        if item:
            meta, name = item.split(b'\t', 1)
            mode, kind, oid = meta.split()
            if kind == b'blob':
                tree[name.decode()] = oid
    require(set(paths) <= set(tree), 'Manifest contains paths absent from frozen Git tree')
    names = sorted(paths)
    blob = git(repo, 'cat-file', '--batch', data=b'\n'.join(tree[p] for p in names)+b'\n')
    offset, result = 0, {}
    for name in names:
        end = blob.index(b'\n', offset)
        oid, kind, size = blob[offset:end].split()
        size = int(size)
        require(oid == tree[name] and kind == b'blob', 'Unexpected Git object')
        offset = end+1
        result[name] = hashlib.sha256(blob[offset:offset+size]).hexdigest()
        offset += size
        require(blob[offset:offset+1] == b'\n', 'Malformed Git batch output')
        offset += 1
    require(offset == len(blob), 'Trailing Git batch data')
    return result


def load_pins(path):
    """Pins are coordinator-reviewed before results, not inferred from a run."""
    global SOURCE, PLAN, BINARIES
    pins=read(path)
    require(pins.get('reviewed') is True and pins.get('source_commit')==FROZEN,
            'Pins must be explicitly reviewed for the frozen V9 source')
    targets={f'build-sycl-mpi-v17/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_{t}' for t in TARGETS}
    binaries=pins.get('binaries',{})
    require(isinstance(binaries,dict) and set(binaries)==targets, 'Pins must cover both named build targets')
    hashes=[pins.get('source_manifest_sha256'),pins.get('plan_sha256'),*binaries.values()]
    require(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in hashes),
            'Missing/invalid independently pinned SHA256')
    require(pins['source_manifest_sha256']==EXPECTED_SOURCE and pins['plan_sha256']==EXPECTED_PLAN,
            'Pins differ from the independently reviewed source/plan identities')
    SOURCE,PLAN,BINARIES=pins['source_manifest_sha256'],pins['plan_sha256'],binaries
    return pins


def verify_build_targets(binaries):
    require(binaries==BINARIES and len(binaries)==2,
            'Pinned two-target build identity differs')


def verify_release(release):
    stamp = release.get('released')
    require(type(stamp) in [int,float] and math.isfinite(stamp) and stamp > 0,
            'Allocation release lacks a valid timestamp')


def verify_build(build_dir, repo):
    build = read(build_dir/'v17/build-status.json')
    manifest = read(build_dir/'source-v17-manifest.json')
    non = read(build_dir/'source-v17-non-kokkos-manifest.json')
    prov = read(build_dir/'source-v17-provenance.json')
    require(sha(build_dir/'source-v17-manifest.json') == SOURCE == build['source_manifest_sha256'],
            'Frozen source manifest differs')
    require(prov['source_commit'] == FROZEN and prov['kokkos_pin'] == KOKKOS and prov['files'] == 1156
            and prov['archive_sha256']==ARCHIVE,
            'Source provenance differs')
    require(len(manifest) == 2642 and len(non) == 1156 and all(manifest.get(k) == v for k,v in non.items()),
            'Source manifest counts/subset differ')
    main, kokkos = {}, {}
    for path, digest in manifest.items():
        require(path.startswith('source-v17/') and '..' not in PurePosixPath(path).parts,
                'Invalid manifest path')
        if path.startswith('source-v17/kokkos/'):
            kokkos[path[len('source-v17/kokkos/'):]] = digest
        else:
            main[path[len('source-v17/'):]] = digest
    require(len(kokkos) == 1486 and {f'source-v17/{k}':v for k,v in main.items()} == non,
            'Manifest source partition differs')
    require(git_hashes(repo, FROZEN, main) == main, 'Main source differs from frozen Git blobs')
    require(git_hashes(repo/'kokkos', KOKKOS, kokkos) == kokkos, 'Kokkos differs from pinned Git blobs')
    require(build['status'] == 'passed' and build['returncode'] == 0 and not build['timed_out']
            and build['storage_error'] is None, 'Native build failed or incomplete')
    for key in ['storage_before','storage_after']:
        require(build[key]['bytes'] < 5_000_000_000, 'Build campaign storage cap exceeded')
    require(build['cache_sha256'] == sha(build_dir/'CMakeCache.txt'), 'Build cache hash differs')
    require(build['build_script_sha256'] == sha(build_dir/'build-v17.sh'), 'Build script hash differs')
    cache = (build_dir/'CMakeCache.txt').read_text()
    for setting in ['Athena_ENABLE_MPI:BOOL=ON', 'Athena_SINGLE_PRECISION:BOOL=OFF',
                    'Kokkos_ENABLE_SYCL:BOOL=ON', 'Kokkos_ARCH_INTEL_PVC:BOOL=ON']:
        require(setting in cache.splitlines(), 'Missing build setting '+setting)
    verify_build_targets(build['binaries'])
    proof = dict(source_commit=FROZEN, kokkos_pin=KOKKOS, source_manifest_sha256=SOURCE,
                 git_blob_hashes_verified=len(manifest), build_receipt_sha256=sha(build_dir/'v17/build-status.json'),
                 cache_sha256=build['cache_sha256'], binaries=build['binaries'],
                 records={str(p.relative_to(build_dir)):sha(p) for p in sorted(build_dir.rglob('*')) if p.is_file()})
    return build, manifest, proof


def frozen_parser(repo, manifest):
    """Replay the pinned parser, with its dependencies, without executing main()."""
    saved = {name:sys.modules.get(name) for name in ['storage','control','preflight']}
    modules = {}
    try:
        for name in saved:
            path = f'tst/hyperbolic_shift_3d/aurora/{name}.py'
            source = git(repo, 'show', f'{FROZEN}:{path}')
            require(hashlib.sha256(source).hexdigest() == manifest['source-v17/'+path], 'Parser source differs')
            module = types.ModuleType(name)
            module.__file__ = str(repo/path)
            sys.modules[name] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            modules[name] = module
        return modules['preflight'], modules['control']
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def checkpoint(path):
    """Independent strict decoder; report hashes of each native double array."""
    with Path(path).open('rb') as f:
        base = f.read(72)
        require(len(base) == 72, 'Truncated base header')
        magic,n,nf,kind,admitted,cells,ns,steps,t,dt = HEADER.unpack(base)
        names = {b'HYP3D_MPI_V9'.ljust(16,b'\0'):9}
        require(magic in names, 'Unknown checkpoint magic/padding')
        require(n >= 10 and n % 2 == 0 and nf == 31 and cells == 12*n**3+7*(n+6)**3 and ns == 31*6*n*n,
                'Invalid checkpoint dimensions')
        require(kind in [0,1,2] and admitted in [0,1], 'Invalid kind/admission')
        require(math.isfinite(t) and t >= 0 and math.isfinite(dt) and dt > 0, 'Invalid checkpoint clock')
        epsilon, degree, header_bytes = 0., 0, 72
        if names[magic] == 9:
            config = f.read(16)
            require(len(config) == 16, 'Truncated V9 angular configuration')
            epsilon, degree, reserved = CONFIG.unpack(config)
            require(math.isfinite(epsilon) and epsilon >= 0 and degree == 7 and reserved == 0,
                    'Invalid V9 angular configuration')
            normal=f.read(16)
            require(len(normal)==16 and NORMAL.unpack(normal)==(3,7,4,3), 'Invalid/truncated V9 normal configuration')
            mesh=f.read(16)
            require(len(mesh)==16 and NORMAL.unpack(mesh)==(2,1,3,3), 'Invalid/truncated V9 mesh configuration')
            header_bytes = 120
        require(Path(path).stat().st_size == header_bytes+16*(nf*cells+ns), 'Checkpoint length differs')
        arrays = []
        for name, count in zip(ARRAY_NAMES, [nf*cells,nf*cells,ns,ns]):
            digest, remaining = hashlib.sha256(), count*8
            while remaining:
                block = f.read(min(1 << 20, remaining))
                require(block and len(block)%8 == 0, 'Truncated double array')
                require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<d',block)), 'Nonfinite checkpoint data')
                digest.update(block)
                remaining -= len(block)
            arrays.append(dict(name=name, doubles=count, sha256=digest.hexdigest()))
        require(not f.read(1), 'Trailing checkpoint bytes')
    return dict(magic=magic.rstrip(b'\0').decode(), n=n, nf=nf, kind=kind, admitted=bool(admitted),
                cells=cells, nominal_cells=19*n**3, surface_size=ns, steps=steps, time=t, dt=dt, header_bytes=header_bytes,
                scri_dissipation_epsilon=epsilon, scri_dissipation_degree=degree,
                base_except_magic_sha256=hashlib.sha256(base[16:]).hexdigest(),
                arrays=arrays, bytes=Path(path).stat().st_size, sha256=sha(path), **NORMAL_FIELDS)


def files_equal(a, b, offset_a=0, offset_b=0):
    with Path(a).open("rb") as left, Path(b).open("rb") as right:
        left.seek(offset_a)
        right.seek(offset_b)
        while True:
            x, y = left.read(1 << 20), right.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def compare(a, b):
    require(a['base_except_magic_sha256'] == b['base_except_magic_sha256'], 'Checkpoint core headers differ')
    for key in ['magic','header_bytes','scri_dissipation_epsilon','scri_dissipation_degree',*NORMAL_FIELDS]:
        require(a[key]==b[key], 'Restart schema/configuration differs: '+key)
    require(a['sha256'] == b['sha256'] and a['bytes'] == b['bytes'], 'Restart whole files differ')
    require(a['arrays'] == b['arrays'], 'One or more checkpoint arrays differ')
    return dict(passed=True, whole_file_equal=True, four_arrays_equal=True,
                core_header_equal=True, arrays=a['arrays'])


def check_value(check, report):
    value = report
    for key in check['path']:
        value = value[key]
    op = check['op']
    require(op in ['eq','finite','le','ge'], 'Unknown acceptance operator')
    if op == 'eq':
        passed = value == check['value'] and (type(value) is bool) == (type(check['value']) is bool)
    else:
        passed = type(value) in [int,float] and math.isfinite(value)
        if passed and op == 'le':
            passed = value <= check['value']
        if passed and op == 'ge':
            passed = value >= check['value']
    return value, bool(passed)


def placement(records):
    require(len(records) == 12 and {r['rank'] for r in records} == set(range(12)), 'Missing/duplicate rank')
    require(len({r['host'] for r in records}) == 1, 'More than one node')
    require(all(r['mask'] == f"{r['rank']//2}.{r['rank']%2}" and r['hierarchy'] == 'COMPOSITE'
                for r in records), 'GPU tile binding differs')


def artifact_path(raw, remote):
    require(remote.startswith(CAMPAIGN) and PurePosixPath(remote).parent.as_posix() == CAMPAIGN.rstrip('/'),
            'Artifact outside declared output directory')
    return raw/'output'/PurePosixPath(remote).name


def component_report(report, command, raw, binding, build, parser, host):
    require(report['mode'] == 'component' and report['mpi_size'] == 12 and report['backend'] == 'SYCL'
            and report['placement_passed'] and report['source_manifest_sha256'] == SOURCE
            and report['plan_sha256'] == PLAN, 'Component report identity differs')
    args = command['args']
    kind = args[args.index('--component')+1]
    binary = args[args.index('--test-binary')+1]
    require(report['component'] == kind, 'Component kind differs')
    placement(report['ranks'])
    require(report['ranks'][0]['host'] == host, 'Component launched on a different node')
    maxima = {}
    for row in report['ranks']:
        result = row['result']
        require(result['log'] == command['report'].removesuffix('.report.json')+f".rank-{row['rank']:02d}.log",
                'Rank log path differs')
        log = artifact_path(raw, result['log'])
        require(sha(log) == result['log_sha256'], 'Rank log hash differs')
        require(result['binary'] == binary and result['binary_sha256'] == build['binaries'][binary]
                == binding['inputs'][binary] and result['cache_sha256'] == build['cache_sha256']
                and result['binary_args'] == [], 'Component executable/cache binding differs')
        parsed = parser.parse_component(log.read_text(), kind, result['returncode'])
        require(all(result.get(k) == v for k,v in parsed.items()), 'Rank log parse differs from report')
        require(parsed['passed'] and result['returncode'] == 0, 'Component rank failed')
        for key, value in parsed['maxima'].items():
            maxima[key] = max(maxima.get(key, value), value)
    require(report['maxima'] == maxima and report['gpu_cpu_parity'] is True, 'Component aggregation differs')
    return dict(component=kind, ranks_verified=12, maxima=maxima)


def restart_report(report, checkpoints, host):
    require(report['mode'] == 'restart' and report['source_manifest_sha256'] == SOURCE
            and report['plan_sha256'] == PLAN and report['placement_passed'], 'Restart adapter identity differs')
    placement(report['ranks'])
    require(report['ranks'][0]['host'] == host and all(r['result']['passed'] for r in report['ranks']),
            'Restart rank placement/result differs')
    rank_zero = next(r['result'] for r in report['ranks'] if r['rank'] == 0)
    require(all(report.get(k) == value for k,value in rank_zero.items() if k != 'passed'),
            'Restart rank-zero aggregation differs')
    for key in ['atol','rtol','maximum_absolute_error','maximum_scaled_error']:
        require(report[key] == 0, 'Restart comparison is not exact')
    reference, candidate = checkpoints['pulse-direct'], checkpoints['pulse-resumed']
    require(report['reference_sha256'] == reference['sha256'] and report['candidate_sha256'] == candidate['sha256'],
            'Restart adapter compared different files')
    require(report['compared_doubles'] == sum(a['doubles'] for a in reference['arrays']), 'Restart comparison size differs')
    header = report['header']
    for key in ['magic','n','nf','kind','admitted','cells','steps','time','dt','header_bytes',
                'scri_dissipation_epsilon','scri_dissipation_degree',*NORMAL_FIELDS]:
        require(header[key] == reference[key], 'Restart adapter header differs: '+key)
    require(header['surface'] == reference['surface_size'] and header['payload_doubles'] == report['compared_doubles'],
            'Restart adapter array dimensions differ')


def verify(args, result):
    raw, build_dir = args.raw_dir.resolve(), args.build_dir.resolve()
    build, manifest, result['build'] = verify_build(build_dir, args.repo)
    parser, control = frozen_parser(args.repo, manifest)
    plan, binding, worker = [read(raw/p) for p in ['plan.json','binding.json','worker-status.json']]
    require(sha(raw/'plan.json') == PLAN == binding['plan_sha256'] == worker['plan_sha256'], 'Plan hash differs')
    control.validate_plan(args.repo.resolve(), plan)
    require(plan['reviewed'] is True and plan['phase'] == 'preflight' and plan['n'] == 16, 'Wrong plan phase/resolution')
    require([c['name'] for c in plan['commands']] == CASE_ORDER and sum(len(c['checks']) for c in plan['commands']) == 194,
            'Plan case/check identities differ')
    require(binding['source_manifest_sha256'] == worker['source_manifest_sha256'] == SOURCE, 'Worker source differs')
    require(binding['source_manifest']=='source-v17-manifest.json' and binding.get('admission') is None,
            'Preflight must bind the new manifest without a duration admission')
    require(binding['pbs_sha256'] == sha(raw/'run_debug.pbs'), 'PBS script changed')
    for name, digest in binding['launcher'].items():
        require(digest == manifest['source-v17/tst/hyperbolic_shift_3d/aurora/'+name], 'Launcher differs from frozen source')
    require(binding['binaries'] == worker['binaries'], 'Worker executable binding differs')
    expected_binaries = {c['executable'] for c in plan['commands']}
    expected_inputs = set().union(*(set(c.get('inputs', [])) for c in plan['commands']))
    require(set(binding['binaries']) == expected_binaries and set(binding['inputs']) == expected_inputs,
            'Bound input/executable coverage differs')
    bound = dict(binding['inputs'], **binding['binaries'])
    require({k for k in bound if k.startswith('build-sycl-mpi-v17/tst/')} ==
            {f'build-sycl-mpi-v17/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_{t}'
             for t in ['mpi_overlap','overlap_device']}, 'Used native build target coverage differs')
    for name, digest in bound.items():
        if name in build['binaries']:
            require(digest == build['binaries'][name], 'Native executable hash differs')
        elif name.startswith('source-v17/'):
            require(digest == manifest[name], 'Adapter source hash differs')
    require(bound['build-sycl-mpi-v17/CMakeCache.txt'] == build['cache_sha256'], 'Bound build cache differs')
    require(bound['inputs/puncture-overlap-n16.txt'] == INITIAL == sha(args.initial_data)
            and args.initial_data.stat().st_size == 23571224, 'Puncture input differs')
    result['initial_data'] = dict(sha256=INITIAL, bytes=23571224)
    for key in ['storage_at_start','storage_at_finish']:
        require(worker[key]['bytes'] < 5_000_000_000, 'Campaign home storage cap exceeded')
    result['storage_at_finish'] = worker['storage_at_finish']
    require(worker['resources'] == binding['resources'] == plan['resources'], 'Resources differ')
    require(worker['phase'] == 'preflight' and worker['n'] == 16, 'Worker phase/resolution differs')
    affinity = worker['affinity']
    require(affinity['passed'] and affinity['returncode'] == 0 and not affinity['timed_out']
            and affinity['storage_error'] is None, 'Affinity probe failed')
    placement(affinity['records'])
    records = [json.loads(line.split('AFFINITY_JSON=',1)[1]) for line in (raw/'output/affinity.log').read_text().splitlines()
               if line.startswith('AFFINITY_JSON=')]
    require(sorted(records,key=lambda r:r['rank']) == sorted(affinity['records'],key=lambda r:r['rank']), 'Affinity log differs')
    result['affinity_log_sha256'] = sha(raw/'output/affinity.log')
    result['one_node_twelve_tiles_verified'] = True
    terminal, release, submission = [read(raw/p) for p in ['collected.json','allocation-released.json','submission.json']]
    require(terminal['job_id'] == worker['job_id'] == release['job_id'] == submission['job_id'], 'Job identity differs')
    verify_release(release)
    require(submission['plan_sha256'] == PLAN and submission['source_manifest_sha256'] == SOURCE
            and submission['resources'] == plan['resources'], 'Submission binding differs')
    result['terminal'] = dict(job_id=worker['job_id'], state=terminal['scheduler']['job_state'],
                              exit_status=terminal['scheduler'].get('Exit_status'), allocation_released=True)
    if result['terminal']['state'] != 'F' or result['terminal']['exit_status'] != 0:
        result['issues'].append('Scheduler is not terminal F with exit status zero')
    started = [c['name'] for c in worker['cases']]
    require(started == CASE_ORDER[:len(started)], 'Duplicate, unknown or reordered worker cases')
    checkpoints, prior = {}, {}
    for index, command in enumerate(plan['commands']):
        case_result = dict(name=command['name'], status='not_started', checks_recomputed=0, issues=[])
        result['cases'].append(case_result)
        if index >= len(worker['cases']):
            continue
        case = worker['cases'][index]
        case_result['status'] = case['status']
        case_result['returncode'] = case.get('returncode')
        try:
            require(case['kind'] == command['kind'], 'Worker case kind differs')
            argv = case['argv']
            require(argv[:7] == ['mpiexec','-n','12','-ppn','12','--cpu-bind=list:'+control.CPU_BIND,'gpu_tile_compact.sh']
                    and argv[7] == REMOTE+command['executable'] and len(argv[8:]) == len(command['args']), 'Launch command differs')
            for actual, expected in zip(argv[8:], command['args']):
                if expected == '@WALL_SECONDS@':
                    require(math.isfinite(float(actual)) and 0 < float(actual) <= command['timeout_seconds']-30,
                            'Invalid runtime deadline')
                else:
                    require(actual == expected, 'Launch argument differs')
            require(set(case['generated_inputs']) == set(command.get('generated_inputs', [])), 'Generated input coverage differs')
            for remote, digest in case['generated_inputs'].items():
                require(remote in prior and prior[remote]['sha256'] == digest, 'Generated input hash chain differs')
            require(set(case['artifacts']) <= set(command['artifacts']), 'Undeclared artifact')
            for remote, expected in case['artifacts'].items():
                path = artifact_path(raw, remote)
                actual = dict(sha256=sha(path), bytes=path.stat().st_size)
                require(actual == expected, 'Artifact hash/size differs: '+remote)
                result['artifacts'].append(dict(remote_path=remote, **actual))
                require(remote not in prior, 'Artifact path reused by a later case')
                prior[remote] = actual
            if case['status'] == 'passed':
                required = {p for p in command['artifacts'] if not p.endswith('.failed-stage.bin')}
                require(required <= set(case['artifacts']), 'Passed case lacks required artifact')
            require(command['report'] in case['artifacts'], 'Report is not worker-hash-bound')
            report = read(artifact_path(raw, command['report']))
            require([r['check'] for r in case['checks']] == command['checks'], 'Acceptance checks differ')
            failed = []
            for record in case['checks']:
                observed, passed = check_value(record['check'], report)
                require(observed == record['observed'] and passed == record['passed'], 'Recorded check replay differs')
                case_result['checks_recomputed'] += 1
                if not passed:
                    failed.append(record)
            case_result['failed_checks'] = failed
            require(report['status'] == case['report_status'], 'Worker report status differs')
            if command['kind'] != 'check':
                cp = checkpoint(artifact_path(raw, command['args'][1]+'.checkpoint'))
                require(cp['n'] == 16 and cp['kind'] == {'minkowski':0,'pulse':1,'puncture':2}[command['kind']], 'Checkpoint grid/kind differs')
                for cp_key, report_key in [('time','t'),('steps','steps'),('dt','dt'),('admitted','checkpoint_admitted'),
                                           ('magic','checkpoint_schema'),('header_bytes','checkpoint_header_bytes'),
                                           ('scri_dissipation_epsilon','scri_dissipation_epsilon'),('scri_dissipation_degree','scri_dissipation_degree'),
                                           *[(k,k) for k in NORMAL_FIELDS]]:
                    require(cp[cp_key] == report[report_key], 'Checkpoint/report differs: '+cp_key)
                eps = float(command['args'][12]) if len(command['args']) == 13 else 2.
                schema = 'HYP3D_MPI_V9'
                require(eps==2 and cp['header_bytes']==120, 'Preflight covers V9 epsilon2 only')
                require(cp['dt'] == float(command['args'][4]) and cp['magic'] == schema and cp['scri_dissipation_epsilon'] == eps,
                        'Checkpoint runtime configuration differs')
                case_result['checkpoint'] = cp
                checkpoints[command['name']] = cp
            elif command['name'].startswith('parity-') and report['status'] == 'completed':
                case_result['component'] = component_report(report, command, raw, binding, build, parser, records[0]['host'])
            elif command['name'] == 'restart-audit' and report['status'] == 'completed':
                restart_report(report, checkpoints, records[0]['host'])
            if not (case['status'] == 'passed' and case['returncode'] == 0 and not case['timed_out']
                    and case['storage_error'] is None and not failed):
                case_result['issues'].append('Case did not pass all execution/scientific gates')
        except (ValueError, KeyError, OSError, TypeError) as error:
            case_result['issues'].append(f'{type(error).__name__}: {error}')
    try:
        names=['pulse-direct','pulse-resumed']
        result['epsilon2_restart'] = compare(*(checkpoints[n] for n in names))
        paths=[raw/'output'/(n+'.checkpoint') for n in names]
        require(files_equal(*paths), 'Streaming whole-file byte comparison differs')
        result['epsilon2_restart']['streaming_byte_equality'] = True
    except (KeyError, ValueError, OSError) as error:
        result['epsilon2_restart'] = dict(passed=False, error=str(error))
    if worker['status'] != 'passed':
        result['issues'].append('Worker status is '+str(worker['status']))
    result['metadata_sha256'] = {p:sha(raw/p) for p in ['plan.json','binding.json','worker-status.json','run_debug.pbs',
                                                       'submission.json','collected.json','allocation-released.json']}
    result['checks_recomputed'] = sum(c['checks_recomputed'] for c in result['cases'])
    result['artifacts_verified'] = len(result['artifacts'])
    result['passed'] = (not result['issues'] and len(result['cases']) == 7 and result['checks_recomputed'] == 194
                        and all(c['status'] == 'passed' and not c['issues'] for c in result['cases'])
                        and result['epsilon2_restart']['passed'])
    result['verified_preflight_configurations'] = [dict(checkpoint_schema='HYP3D_MPI_V9',
        checkpoint_header_bytes=120,scri_dissipation_epsilon=2.,scri_dissipation_degree=7,**NORMAL_FIELDS)] if result['passed'] else []
    result['admission_written'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'output/hyperbolic-core-shell/aurora/preflight-overlap-v1')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'output/hyperbolic-core-shell/aurora/preflight-overlap-v1/build-v17')
    parser.add_argument('--initial-data', type=Path, default=ROOT/'output/hyperbolic-core-shell/overlap-native-local-v1/initial-n16/initial.txt')
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pins',type=Path,default=HERE/'pins.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be new')
    result = dict(passed=False, admission_written=False, issues=[], cases=[], artifacts=[], verifier_sha256=sha(__file__),
                  source_manifest_sha256=SOURCE, plan_sha256=PLAN,
                  scope='Implementation/restart preflight only; no duration, convergence, AMR or nonspherical stability claim.',
                  component_replay='Frozen source-v17 parser replays hash-bound logs; checkpoint decoder is independent.')
    try:
        load_pins(args.pins)
        result.update(source_manifest_sha256=SOURCE,plan_sha256=PLAN,pins_sha256=sha(args.pins))
        verify(args, result)
    except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        result['issues'].append(f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ['passed','issues','artifacts_verified','checks_recomputed','verified_preflight_configurations']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
