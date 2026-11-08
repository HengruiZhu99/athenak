#!/usr/bin/env python3
"""Offline complete-state checks for the bounded native GPU AMR test bundle.

This script neither builds nor launches anything. GPU execution provenance must
be supplied separately; --cpu-fixture-evidence validates parsing against an
existing passing local fixture without making a GPU claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import shlex

import numpy as np

from amr_restart import DT, parameters
from native_amr_admission_candidate import SAFE_DT, REPLACED, check_package
from native_amr_gpu_input import helper_text, restart_text
from restart_roundtrip import HEADER, compare, parse, payload

ROW = np.dtype([('keys', '<i4', (9,)), ('values', '<f8', (39,))])
ROOT, N, ORIGIN, OVERLAP = 4, 12, 7, 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def topology(refined, outer_only=False):
    # Logical leaf order here is chart, level, lx1, lx2, lx3.
    leaves = {(0, ROOT, i, j, k) for i in range(3) for j in range(3) for k in range(3)}
    leaves.remove((0, ROOT, 1, 1, 1))
    leaves.update((0, ROOT+1, 2+i, 2+j, 2+k) for i in range(2) for j in range(2) for k in range(2))
    leaves.update((chart, ROOT, layer, 0, 0) for chart in range(1, 7) for layer in range(12))
    if refined:
        selected = [(1, ROOT, 11, 0, 0)]
        if not outer_only:
            selected += [(0, ROOT, 0, 0, 0), (1, ROOT, 0, 0, 0)]
        for chart, level, x, y, z in selected:
            leaves.remove((chart, level, x, y, z))
            leaves.update((chart, level+1, 2*x+i, 2*y+j, 2*z+k)
                          for i in range(2) for j in range(2) for k in range(2))
    return leaves


def bounds(leaf, origin=ORIGIN):
    chart, level, x, y, z = leaf
    require(level in (ROOT, ROOT+1), 'unexpected fixture refinement level')
    factor = 1 << (level-ROOT)
    lower, upper = [origin]*3, [origin+N]*3
    if chart == 0:
        for axis, index in enumerate((x, y, z)):
            lower[axis] -= OVERLAP*(index == 0)
            upper[axis] += OVERLAP*(index == 3*factor-1)
    elif x//factor == 0:
        # Spherical inner radius .08: only outer radial overlap in first layer.
        upper[0] += OVERLAP*(x == factor-1)
        for axis, index in ((1, y), (2, z)):
            lower[axis] -= OVERLAP*(index == 0)
            upper[axis] += OVERLAP*(index == factor-1)
    return lower, upper, int(chart > 0 and x == 12*factor-1)


def expected_keys(leaves):
    pieces = []
    for leaf in sorted(leaves):
        lower, upper, surface = bounds(leaf)
        k, j, i = np.meshgrid(np.arange(lower[2], upper[2]),
                             np.arange(lower[1], upper[1]),
                             np.arange(lower[0], upper[0]), indexing='ij')
        keys = np.empty((i.size, 9), dtype='<i4')
        keys[:, :5] = leaf
        keys[:, 5] = 0
        keys[:, 6:] = np.column_stack((k.ravel(), j.ravel(), i.ravel()))
        pieces.append(keys)
        if surface:
            k, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            keys = np.empty((N*N, 9), dtype='<i4')
            keys[:, :5] = leaf
            keys[:, 5] = 1
            keys[:, 6:] = np.column_stack((k.ravel(), j.ravel(), np.full(N*N, N)))
            pieces.append(keys)
    return np.concatenate(pieces)


def summary(rows):
    values = rows['values']
    records = values[:, 31:]
    totals = [float(records[:, 0].min()), float(records[:, 1].min())]
    totals += [float(records[:, f].max() if f == 5 else records[:, f].sum())
               for f in range(2, 7)]
    totals.append(float(records[:, 7].min()))
    return dict(points=len(rows), scalar_values=39*len(rows),
                sha256=hashlib.sha256(rows.tobytes()).hexdigest(), chi_totals=totals)


def read_snapshot(folder, stem, ranks, refined, evolution, hashes):
    leaves = topology(refined)
    expected = expected_keys(leaves)
    owners, chunks = {}, []
    for rank in range(ranks):
        meta = folder/f'{stem}-rank{rank}.txt'
        binary = folder/f'{stem}-rank{rank}.bin'
        hashes[str(meta)] = sha(meta)
        hashes[str(binary)] = sha(binary)
        local = set()
        for line in meta.read_text().splitlines():
            fields = tuple(map(int, line.split()))
            require(len(fields) == (12 if evolution else 5), 'invalid AMR leaf metadata')
            leaf = fields[:5]
            require(leaf in leaves and leaf not in owners, 'duplicate/unexpected AMR leaf')
            owners[leaf] = rank
            local.add(leaf)
            if evolution:
                lo, hi, face = bounds(leaf)
                require(fields[5:] == tuple(q for pair in zip(lo, [x-1 for x in hi])
                                             for q in pair)+(face,), 'evolved bounds differ from fixture')
        require(binary.stat().st_size % ROW.itemsize == 0, 'truncated AMR snapshot')
        data = np.fromfile(binary, dtype=ROW)
        require(np.isfinite(data['values']).all(), 'nonfinite AMR owned field/history')
        if len(data):
            require(all(tuple(row) in local for row in np.unique(data['keys'][:, :5], axis=0)),
                    'snapshot point outside rank ownership')
        chunks.append(data)
    require(owners.keys() == leaves, 'missing AMR leaf ownership')
    rows = np.concatenate(chunks)
    keys = rows['keys']
    order = np.lexsort(tuple(keys[:, q] for q in reversed(range(9))))
    rows = rows[order]
    require(np.array_equal(rows['keys'], expected), 'missing/duplicate/extra AMR owned point')
    slices = {}
    start = 0
    for leaf in sorted(leaves):
        lo, hi, face = bounds(leaf)
        size = int(np.prod(np.array(hi)-lo))+face*N*N
        slices[leaf] = rows[start:start+size].tobytes()
        start += size
    require(start == len(rows), 'incomplete leaf slicing')
    region = {name: summary(rows[rows['keys'][:, 5] == kind])
              for name, kind in [('bulk', 0), ('surface', 1)]}
    return dict(summary=summary(rows), regions=region), owners, slices


def records(log, prefix):
    return [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]


def run_folders(values):
    result = {}
    for value in values:
        rank, folder = value.split('=', 1)
        rank = int(rank)
        require(rank > 0 and rank not in result, 'duplicate/invalid MPI rank count')
        result[rank] = Path(folder).resolve()
    require(len(result) >= 2, 'need at least two decompositions')
    return result


def snapshot_case(args, output, hashes):
    evolution = args.case == 'evolution'
    results = {}
    for ranks, folder in run_folders(args.run).items():
        path = folder/'run.log'
        hashes[str(path)] = sha(path)
        log = path.read_text()
        require(('PASS four native Driver cycles with two AMR events' if evolution else
                 'PASS native field AMR lifecycle') in log, 'AMR harness did not reach PASS')
        reports = records(log, 'AMR_CYCLE ' if evolution else 'AMR_REPORT ')
        indices = list(range(5 if evolution else 3))
        require([p['cycle' if evolution else 'phase'] for p in reports] == indices,
                'missing/duplicate AMR snapshots')
        refined = [False, True, True, False, False] if evolution else [False, True, False]
        previous = owners = None
        initial_totals = None
        for index, (report, refined_here) in enumerate(zip(reports, refined)):
            bulk, surface = (277236, 1296) if refined_here else (229392, 864)
            require((report['blocks'], report['bulk_points'], report['scri_points']) ==
                    (127 if refined_here else 106, bulk, surface), 'wrong AMR snapshot shape')
            require(ranks < 8 or report['zero_scri_ranks'] > 0, 'missing zero-scri rank case')
            stem = ('cycle' if evolution else 'phase')+str(index)
            canonical, current_owners, current = read_snapshot(folder, stem, ranks,
                                                               refined_here, evolution, hashes)
            report['canonical'] = canonical
            if evolution:
                expected_time = 0.
                for _ in range(index):
                    expected_time += args.dt
                require(report['time'] == expected_time and report['dt'] == args.dt, 'AMR exact time changed')
                require(report['sequence'] == reports[0]['sequence']+(index >= 1)+(index >= 3),
                        'AMR mesh sequence did not advance exactly twice')
                for region in canonical['regions'].values():
                    totals = region['chi_totals']
                    require(totals[0] > 0 and totals[1] > 0 and totals[2:7] == [0.]*5
                            and totals[7] == 1e300, 'unexpected evolution chi history event')
            else:
                require(np.isfinite(report['compatibility']) and report['compatibility'] <= 1e-7,
                        'AMR compatibility gate failed')
                require(report['max_polynomial_error'] <= 256*np.finfo(float).eps*32768*.02,
                        'AMR polynomial gate failed')
                require(index == 0 or report['polynomial_checks'] > 1000, 'missing polynomial coverage')
                totals = {key: region['chi_totals'] for key, region in canonical['regions'].items()}
                if initial_totals is None:
                    initial_totals = totals
                    require(all(sum(t[2:7]) > 0 for t in totals.values()), 'history transfer was not seeded')
                require(totals == initial_totals, 'AMR transfer changed history totals/extrema')
                if previous is not None:
                    common = previous.keys() & current.keys()
                    migrated = [leaf for leaf in common if owners[leaf] != current_owners[leaf]]
                    require(common and migrated, 'missing unchanged migrated leaf coverage')
                    require(all(previous[leaf] == current[leaf] for leaf in common),
                            'unchanged AMR leaf state/history changed')
                    report['unchanged_leaves'] = len(common)
                    report['unchanged_migrated_leaves'] = len(migrated)
            previous, owners = current, current_owners
        diagnostics = records(log, 'DIAGNOSTICS_REPORT ')
        stages = records(log, 'AMR_STAGE ')
        if evolution:
            require(not diagnostics, 'unexpected diagnostic report in evolution harness')
            require([(s['cycle'], s['stage']) for s in stages] ==
                    [(c, s) for c in range(1, 5) for s in range(1, 5)], 'missing/duplicate poisoned stages')
            for s in stages:
                cycle, stage = s['cycle'], s['stage']
                prior = reports[cycle-1]
                require(s['bulk_values'] == 31*prior['bulk_points'] and
                        s['scri_values'] == 31*prior['scri_points'], 'incomplete stage RHS coverage')
                rhs_fraction = (0., .5, .5, 1.)[stage-1]
                state_fraction = (.5, .5, 1., 1.)[stage-1]
                require(s['rhs_time'] == prior['time']+rhs_fraction*args.dt and
                        s['state_time'] == prior['time']+state_fraction*args.dt, 'wrong RK stage clock')
                require(all(np.isfinite(s[key]) for key in ('compatibility', 'clock', 'cfl'))
                        and s['compatibility'] <= 1e-7, 'nonfinite or invalid stage diagnostic')
        else:
            require(not stages, 'unexpected evolution stages in transfer harness')
            require([d['phase'] for d in diagnostics] == [0, 1, 2] and
                    [d['refreshes'] for d in diagnostics] == [1, 2, 3], 'missing AMR diagnostic refreshes')
            for report, diagnostic in zip(reports, diagnostics):
                require(diagnostic['constraint_weight_scalars'] == 23*report['bulk_points'] and
                        diagnostic['owned_scalars'] == 39*(report['bulk_points']+report['scri_points'])
                        and diagnostic['rk_status_values'] > 0, 'incomplete diagnostic coverage')
                require(all(diagnostic[key] is True for key in ('fresh_exact', 'repeat_cached', 'readonly')),
                        'AMR diagnostics cache/read-only gate failed')
        results[str(ranks)] = dict(folder=str(folder), snapshots=reports, stages=stages, diagnostics=diagnostics)
    first = next(iter(results.values()))
    for run in results.values():
        require([p['canonical'] for p in run['snapshots']] == [p['canonical'] for p in first['snapshots']],
                'cross-rank complete31+8 AMR state differs')
        require(run['stages'] == first['stages'], 'cross-rank stage diagnostics differ')
    return dict(passed=True, case=args.case, cross_rank_complete_state_bitwise=True, runs=results,
                real_driver_cycles=4 if evolution else 0, legacy_solution_comparison=False,
                admission_scope='test harness constructs static then uses actual dynamic Driver; ordinary restart case gates input admission')


def checkpoint_audit(path, ranks, capacity, expected_dt=DT):
    info = parse(path)
    with path.open('rb') as stream:
        stream.seek(info['start'])
        fixed = stream.read(HEADER)
    require(struct.unpack_from('<i', fixed, 4)[0] == ROOT, 'wrong restart root level')
    locations = list(struct.iter_unpack('<5i', info['topology']))
    leaves = {(c, l, x, y, z) for x, y, z, l, c in locations}
    refined = info['cycle'] >= 2
    require(len(leaves) == len(locations) and leaves == topology(refined, True), 'wrong ordinary AMR topology')
    require(info['blocks'] == info['record_blocks'] == (113 if refined else 106), 'wrong restart block count')
    require(np.array_equal(np.frombuffer(info['costs'], dtype='<f4'), np.ones(info['blocks'], dtype='<f4')),
            'native load balancing costs differ from one')
    clock = 0.
    for _ in range(info['cycle']):
        clock += expected_dt
    require(info['dt'] == expected_dt and info['time'] == clock, 'ordinary AMR exact time changed')
    meta = info['metadata']
    require(meta[2:4] == (len(info['identity']), (18**3+144)*39) and
            meta[4:] == (ranks, 0, info['blocks']//ranks, 0), 'wrong shared restart layout/rank metadata')
    values = parameters(info)
    required = {('time','evolution'):'dynamic', ('time','integrator'):'rk4_classical',
                ('mesh_refinement','refinement'):'adaptive', ('amr_criterion0','method'):'location',
                ('output1','file_type'):'rst', ('output1','single_file_per_rank'):'false'}
    for key, expected in required.items():
        require(values.get(key) == expected, 'wrong ordinary AMR parameter '+str(key))
    numerical = {('time','dt'):expected_dt, ('time','cfl_number'):.2, ('mesh_refinement','num_levels'):2,
        ('mesh_refinement','refinement_interval'):0, ('mesh_refinement','ncycle_check'):1,
        ('mesh_refinement','max_nmb_per_rank'):capacity, ('amr_criterion0','location_x1'):.99,
        ('amr_criterion0','location_x2'):0, ('amr_criterion0','location_x3'):0,
        ('amr_criterion0','location_rad'):.005, ('z4c_hyperboloidal','ko'):.1,
        ('z4c_hyperboloidal','chi_floor'):1e-10, ('output1','dcycle'):1,
        ('mesh','logical_location_version'):2}
    for key, expected in numerical.items():
        require(float(values[key]) == expected, 'wrong ordinary AMR parameter '+str(key))
    require(info['atlas'].startswith(b'athenak-chart-atlas-v2;') and
            info['identity'].startswith(b'native-hyperboloidal-v1;fields=31;chi-record=8;'),
            'wrong restart atlas/physics identity version')
    data = payload(info)
    require(data.shape == (info['blocks'], 18**3+144, 39) and np.isfinite(data).all(),
            'wrong/nonfinite complete restart payload')
    active = np.zeros(data.shape[:2], dtype=bool)
    faces = 0
    for m, (x,y,z,level,chart) in enumerate(locations):
        lo, hi, face = bounds((chart,level,x,y,z), OVERLAP)
        active[m,:18**3].reshape(18,18,18)[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]] = True
        if face:
            active[m,18**3:] = True
            faces += 1
    require(int(active[:,:18**3].sum()) == (241488 if refined else 229392) and
            faces == (9 if refined else 6), 'wrong ordinary evolved coverage')
    require(np.array_equal(data[:,:,31] != 0, active) and (data[~active] == 0).all(),
            'missing/extra evolved slot or nonzero padding')
    owned = data[active]
    require((owned[:,0] > 0).all() and (owned[:,31:33] > 0).all() and
            (owned[:,33:38] == 0).all() and (owned[:,38] == 1e300).all() and
            (owned[:,28:31] == 0).all(), 'ordinary AMR physical/chi/dormant-field check failed')
    # Match every initialized fixed-header member. The nine global coarse-index
    # integers at120:156 are unused/unspecified in the existing native writer.
    invariant = (fixed[4:120]+fixed[156:-20], info['atlas'], info['identity'])
    report = dict(path=str(path), cycle=info['cycle'], time=info['time'], dt=info['dt'],
        blocks=info['blocks'], faces=faces, all_costs_one=True,
        initialized_header_sha256=hashlib.sha256(invariant[0]).hexdigest(),
        unused_global_coarse_indices=list(struct.unpack('<9i',fixed[120:156])),
        scalar_values=int(data.size), owned_scalar_values=int(owned.size),
        checkpoint_sha256=sha(path), payload_sha256=hashlib.sha256(data.tobytes()).hexdigest())
    return info, invariant, report


def restart_case(args, output, hashes):
    datasets, reports, invariant = {}, {}, None
    expected = {'prefix':set(range(3)), 'continuous':set(range(5)), 'resumed':{3,4}}
    for name in expected:
        directory = getattr(args,name).resolve()
        ranks = (args.prefix_ranks if name == 'prefix' else
                 (args.continuous_ranks or args.target_ranks) if name == 'continuous' else args.target_ranks)
        paths = sorted((directory/'rst').glob('*.rst'))
        require(paths, 'missing ordinary AMR checkpoints '+name)
        datasets[name], reports[name] = {}, []
        for path in paths:
            info, current, report = checkpoint_audit(path, ranks, args.capacity, args.dt)
            hashes[str(path)] = report['checkpoint_sha256']
            if invariant is None:
                invariant = current
            require(current == invariant, 'ordinary immutable initialized metadata changed')
            cycle = info['cycle']
            if cycle in datasets[name]:
                compare(datasets[name][cycle], info)
            datasets[name][cycle] = info
            reports[name].append(report)
        require(datasets[name].keys() == expected[name], 'missing/extra ordinary checkpoint cycle '+name)
    comparisons = {'prefix2':compare(datasets['prefix'][2],datasets['continuous'][2])}
    for cycle in (3,4):
        comparisons['resumed'+str(cycle)] = compare(datasets['resumed'][cycle], datasets['continuous'][cycle])
    return dict(passed=True, case='restart', ordinary_native_executable=True,
        test_admission_override=False, comparisons=comparisons, checkpoints=reports,
        all31_fields_all8_records_bitwise=True, writer_ranks=[args.prefix_ranks,args.target_ranks])


def candidate_binding(candidate, package_dir, baseline_sha, executable_sha, gpu=True):
    """Require coherent CMake evidence; mixed-object prototypes are not admitted."""
    package = check_package(package_dir)
    require(candidate['schema'] == 'native-amr-coherent-candidate-build-v1'
            and candidate['status'] == 'built' and candidate['candidate'] is True
            and candidate['baseline_inputs_unchanged'] is True
            and candidate['baseline_native_executable_sha256'] == baseline_sha
            and candidate['executable_sha256'] == executable_sha and executable_sha != baseline_sha
            and candidate['fixed_dt'] == SAFE_DT and candidate['original_main_source_unchanged'] is True
            and candidate['all_native_compile_recipes_match'] is True
            and candidate['normalized_link_policy_matches'] is True
            and candidate['baseline_objects_linked'] is False
            and candidate['production_patch_sha256'] == package['production_patch_sha256']
            and candidate['package_manifest_sha256'] == sha(package_dir/'manifest.json')
            and set(candidate['changed_sources']) == set(package['baseline_sources']),
            'GPU candidate requires coherent reviewed source/build and frozen baseline')
    source_root = Path(candidate['root'])
    for name in package['baseline_sources']:
        require(candidate['candidate_files'][str(source_root/name)] == package['files']['overlay/'+name],
                'unbound coherent candidate source '+name)
    require(candidate['candidate_files'][str(source_root/'src/main.cpp')] == candidate['original_main_sha256'],
            'candidate main source mismatch')
    entries = []
    for entry in candidate['compile_commands']:
        command = entry.get('arguments') or shlex.split(entry['command'])
        if '-o' in command and 'athena.dir/' in command[command.index('-o')+1]:
            entries.append(entry)
    require(len(entries) == candidate['native_translation_units'] and len(entries) > 100,
            'incomplete coherent translation-unit coverage')
    if gpu:
        require(all(any('sycl' in arg.lower() for arg in
                        (entry.get('arguments') or shlex.split(entry['command']))) for entry in entries),
                'candidate objects lack actual SYCL recipes')
    require(sum(token.endswith('athena.dir/main.cpp.o') for token in candidate['link_command']) == 1,
            'candidate did not retain original main')
    return package


def provenance(args, hashes):
    if args.cpu_fixture_evidence:
        path = args.cpu_fixture_evidence.resolve()
        record = json.loads(path.read_text())
        if record.get('schema') == 'native-amr-clean-admission-restart-v1':
            require(args.case == 'restart' and record['passed'] and record['dt'] == args.dt == SAFE_DT
                    and record['frozen_runtime_unchanged'] and record['genuine_checkpoint_deserialization']
                    and record['restart_without_input_or_dt_override'], 'clean CPU restart proof failed')
        else:
            require(record.get('passed') is True and all(record['provenance'].values()), 'CPU fixture evidence failed')
        hashes[str(path)] = sha(path)
        return dict(backend='existing CPU fixture parsing validation only', gpu_evidence=False,
                    evidence=str(path), evidence_sha256=hashes[str(path)])
    path = args.execution_evidence.resolve()
    record = json.loads(path.read_text())
    require(record['schema'] == 'native-amr-gpu-execution-v2' and record['exit_status'] == 0
            and record['baseline_inputs_unchanged'] is True and record['candidate'] is True,
            'GPU execution must identify the isolated admission candidate')
    require(record['case'] == args.case, 'wrong GPU execution case')
    require(args.dt == record['fixed_dt'] == SAFE_DT and args.candidate_package,
            'candidate GPU protocol requires reviewed package and dt=.0002')
    native_sha = record['baseline_native_executable_sha256']
    require(len(native_sha) == 64 and all(c in '0123456789abcdef' for c in native_sha),
            'invalid native executable hash')
    # Evidence lists every downloaded artifact relative to its own directory.
    for name, digest in record['artifacts'].items():
        artifact = (path.parent/name).resolve()
        require(sha(artifact) == digest, 'GPU execution artifact hash mismatch '+name)
    artifact_paths = {str((path.parent/name).resolve()) for name in record['artifacts']}
    candidate_path = (path.parent/record['candidate_build_provenance']).resolve()
    require(str(candidate_path) in artifact_paths, 'unbound candidate build provenance')
    candidate = json.loads(candidate_path.read_text())
    package = candidate_binding(candidate, args.candidate_package, native_sha,
                                record['candidate_executable_sha256'])
    for name in ('compile_database','link_file'):
        artifact = (path.parent/record['candidate_'+name]).resolve()
        require(str(artifact) in artifact_paths and sha(artifact) == candidate[name+'_sha256'],
                'unbound candidate compilation/link recipe '+name)
    if args.case == 'restart':
        require(args.prefix_ranks == 24 and args.target_ranks == 96 and
                args.continuous_ranks in (None,96) and args.capacity == 16,
                'GPU restart protocol requires24-prefix/96-continuous+resume and capacity16')
        expected_runs = {name: (getattr(args,name).resolve(), 24 if name=='prefix' else 96)
                         for name in ('prefix','continuous','resumed')}
        executable_sha = candidate['executable_sha256']
    else:
        folders = run_folders(args.run)
        require(set(folders) == {24,96}, 'GPU snapshot protocol requires24/96 decompositions')
        expected_runs = {str(rank):(folder,rank) for rank,folder in folders.items()}
        build_path = (path.parent/record['build_provenance']).resolve()
        require(str(build_path) in {str((path.parent/name).resolve()) for name in record['artifacts']},
                'unbound AMR build provenance')
        build = json.loads(build_path.read_text())
        require(build['schema'] == 'native-amr-gpu-build-v3' and build['status'] == 'built'
                and build['case'] == args.case and build['runtime_inputs_unchanged'] is True
                and build['baseline_native_executable_sha256'] == native_sha and build['candidate'] is True
                and build['coherent_candidate'] is True
                and build['native_executable_sha256'] == candidate['executable_sha256']
                and build['compile_database_sha256'] == candidate['compile_database_sha256']
                and build['link_file_sha256'] == candidate['link_file_sha256']
                and build['candidate_build_sha256'] == sha(candidate_path)
                and build['candidate_executable_sha256'] == candidate['executable_sha256']
                and build['production_patch_sha256'] == package['production_patch_sha256']
                and build['fixed_dt'] == SAFE_DT,
                'AMR helper build does not bind reviewed admission candidate')
        source = Path(__file__).with_name('amr_'+args.case+'.cpp')
        require([digest for name,digest in build['sources'].items() if name.endswith('/'+source.name)]
                == [sha(source)], 'AMR helper source differs from reviewed fixture')
        require(any('sycl' in arg.lower() for arg in build['compile_command']),
                'AMR helper did not use actual SYCL flags')
        executable_sha = build['executable_sha256']
    executions = {entry['label']:entry for entry in record['executions']}
    require(len(executions) == len(record['executions']) and executions.keys() == expected_runs.keys(),
            'missing/duplicate GPU execution records')
    for name,(folder,ranks) in expected_runs.items():
        entry=executions[name]
        require((path.parent/entry['directory']).resolve() == folder and entry['ranks'] == ranks
                and entry['exit_status'] == 0 and entry['executable_sha256'] == executable_sha,
                'wrong/failed GPU AMR execution '+name)
        if args.case != 'restart':
            require(entry['input_sha256'] == hashlib.sha256(helper_text().encode()).hexdigest(),
                    'GPU AMR harness input differs from reviewed safe fixture')
        elif name != 'resumed':
            require(entry['input_sha256'] == hashlib.sha256(restart_text().encode()).hexdigest(),
                    'GPU ordinary AMR input differs from reviewed safe fixture')
        else:
            require(entry['input_override'] is False and entry['dt_override'] is False,
                    'ordinary AMR restart must recover input/dt from checkpoint')
    hashes[str(path)] = sha(path)
    return dict(backend='Aurora complete coherent admission candidate; frozen baseline preserved', gpu_evidence=True,
                baseline_native_executable_sha256=native_sha,
                candidate_executable_sha256=candidate['executable_sha256'],
                production_patch_sha256=package['production_patch_sha256'], fixed_dt=SAFE_DT,
                evidence=str(path), evidence_sha256=hashes[str(path)],
                artifact_paths={str((path.parent/name).resolve()) for name in record['artifacts']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('lifecycle','evolution','restart'),required=True)
    parser.add_argument('--run',action='append',default=[],metavar='RANKS=DIRECTORY')
    parser.add_argument('--prefix',type=Path)
    parser.add_argument('--continuous',type=Path)
    parser.add_argument('--resumed',type=Path)
    parser.add_argument('--prefix-ranks',type=int,default=24)
    parser.add_argument('--target-ranks',type=int,default=96)
    parser.add_argument('--continuous-ranks',type=int,
                        help='defaults to target ranks; CPU fixture validation may use the prior layout')
    parser.add_argument('--capacity',type=int,default=16)
    parser.add_argument('--dt',type=float,default=SAFE_DT,
                        help='exact expected timestep; GPU candidate requires .0002')
    parser.add_argument('--candidate-package',type=Path,
                        help='locally reviewed package, required for GPU evidence')
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--execution-evidence',type=Path)
    evidence.add_argument('--cpu-fixture-evidence',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'new evidence directory required')
    require(args.case != 'restart' or all((args.prefix,args.continuous,args.resumed)),
            'restart requires prefix, continuous and resumed directories')
    args.output.mkdir(parents=True)
    hashes={}
    result=dict(passed=False,case=args.case)
    try:
        proof=provenance(args,hashes)
        result=(restart_case if args.case=='restart' else snapshot_case)(args,args.output,hashes)
        if proof['gpu_evidence']:
            checked=proof.pop('artifact_paths')
            require(set(hashes)-{proof['evidence']} <= checked, 'unbound GPU input artifact')
        result['provenance']=proof
    except Exception as error:
        result['passed']=False
        result['failure']=str(error)
        raise
    finally:
        result['input_sha256']=hashes
        result['source_sha256']={name:sha(Path(__file__).with_name(name)) for name in
            ('native_amr_gpu_compare.py','native_amr_admission_candidate.py','native_amr_candidate_cmake.py','native_amr_gpu_input.py',
             'amr_restart.py','restart_roundtrip.py')}
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('passed','case','provenance')},indent=2))


if __name__=='__main__':
    main()
