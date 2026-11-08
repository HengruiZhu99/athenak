#!/usr/bin/env python3
"""Read matched native normal/timed/observed24-rank continuations from2486.

No execution occurs here. Timings remain withheld unless every scientific,
state/output, build, launch and window check passes. Legacy parity is separate.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import struct

import numpy as np

import native_cycle_timing_gpu_validate as qualified
from native_step_observer_gpu_compare import checkpoint_set, diagnostics_exact
from native_step_observer_validate import stages
from n12_checkpoint import DT, expected_time, parse_native, require, sha256
from restart_roundtrip import compare, parse, payload

SIDE = qualified.SIDE
FIRST, LAST, CONFIG = 2486, 2556, (10, 20, 3)
SOURCE = 'dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec'
ADMISSION = 'babb035acdcb505d2f8407815981731168ae2c75c3217dcc43ba2ce56c363bd9'
REVIEW = '6322dcb077c8ac51bc105791d024cd93686994cbc89ffaf90b5819a9f3b99d51'
COMBINED = '222893425aae3419e55d72b7db78d2fdc0253ecd7676182fd5a52db8d9a7db45'
COMBINED_PROOF = 'b3076d063321f8f939f654cfe0ffae736832f9eba1743fde4934b7a1047bbcfd'
OBSERVER_PROOF = '642507b2529419d290f00eea609d8ebc0ba6ce5dc52ed41d6cfe2fe995eb1a8a'
SOURCE_REMOTE = str(SIDE/'runs/native-n12-common-07/rst/native_n12_5M.00007.rst')
CASE = Path('runs/native-n12-timing-two-v1')
EXECUTABLES = {
    'normal': (SIDE/'build-sycl/src/athena', qualified.NORMAL),
    'timed-normal': (qualified.BUILD/'athena-cycle-timing', qualified.TIMED),
    'timed-observer': (SIDE/'validation/perf-two-node-v1/native-timed-observer/build/athena-timed-observer', COMBINED)}
read_json = qualified.read_json


def check_admission(admission, review, source_sha, admission_sha, review_sha):
    require((source_sha, admission_sha, review_sha) == (SOURCE, ADMISSION, REVIEW),
            'not the reviewed2486 checkpoint/admission/review')
    require(admission['passed'] is True and admission['end_cycle'] == FIRST and
            admission['checkpoint_sha256'] == SOURCE and
            admission['comparisons'][str(FIRST)]['passed'] is True,
            'bounded2486 science admission missing')
    require(review['passed'] is True and review['end_cycle'] == FIRST and
            review['science_admission_sha256'] == ADMISSION and review['checkpoint_sha256'] == SOURCE and
            review['stage_rows'] == 4*FIRST and review['exact_stage_clocks'] is True and
            review['history_completed_prefix_passed'] is True and
            review['history_times'] == [0,356,711,1066,1421,1776,2131,2486] and
            review['harmonic_coefficients_each_time'] == 169, 'bounded independent review incomplete')


def check_source(info):
    require((info['cycle'], info['time'], info['dt']) == (FIRST, expected_time(FIRST), DT),
            'source checkpoint clock differs')
    require(info['metadata'][4:] == (24,0,4,0), 'source ownership differs')
    require(np.array_equal(np.frombuffer(info['costs'], dtype='<f4'), np.ones(106,dtype='<f4')),
            'source equal weights differ')
    p = info['parameters']; clock = sum([1/6]*7)
    for block in ('output1','output2'):
        require(float(p[block+'/dt']) == 1/6 and float(p[block+'/last_time']) == clock and
                int(p.get(block+'/dcycle','0')) <= 0, 'source output schedule differs')
        for cycle in range(FIRST+1,LAST+1):
            require(np.float32(expected_time(cycle)) < np.float32(clock+1/6),
                    'scheduled output falls inside continuation')
    require(p['output1/file_type'] == 'hst' and p['output2/file_type'] == 'rst' and
            p['output2/single_file_per_rank'] == 'false' and int(p['output2/file_number']) == 8,
            'source output mode/counter differs')
    require(p['time/integrator'] == 'rk4_classical' and p['time/evolution'] == 'dynamic' and
            p['mesh_refinement/refinement'] == 'static' and float(p['time/dt']) == DT and
            float(p['z4c_hyperboloidal/ko']) == .1 and float(p['z4c_hyperboloidal/chi_floor']) == 1e-10 and
            expected_time(LAST) < 2, 'source numerical policy differs')


def check_final(info, source, cycle=LAST):
    require((info['cycle'],info['time'],info['dt']) == (cycle,expected_time(cycle),DT),
            'final checkpoint clock differs')
    require(info['fixed'][:120] == source['fixed'][:120] and
            info['fixed'][156:-20] == source['fixed'][156:-20], 'initialized fixed header differs')
    for key in ('atlas','topology','costs','identity','metadata'):
        require(info[key] == source[key], 'final metadata differs: '+key)
    p, original = info['parameters'], source['parameters']
    mutable = {'time/nlim','time/tlim','time/ndiag','output1/last_time','output2/last_time',
               'output2/file_number','fastflow/last_a0_0','fastflow/ah_found_a0_0'}
    require(p.keys() == original.keys() and all(p[k] == original[k] for k in p if k not in mutable),
            'final numerical/output parameter changed')
    require((int(p['time/nlim']),float(p['time/tlim']),int(p['time/ndiag'])) == (cycle,2.,1000000000),
            'final stopping controls differ')
    require(int(p['output2/file_number']) == 9 and
            all(float(p[b+'/last_time']) == sum([1/6]*8) for b in ('output1','output2')),
            'final output counter/clock differs')
    require(p['fastflow/ah_found_a0_0'] == '1' and math.isfinite(float(p['fastflow/last_a0_0'])) and
            float(p['fastflow/last_a0_0']) > 0, 'final horizon state invalid')


def check_combined_build(folder, hashes):
    m = read_json(folder/'artifact-manifest.json')
    require(m['schema'] == 'native-timed-observer-build-collection-v1' and m['passed'] is True,
            'missing combined build collection')
    qualified.check_files(folder,m['files'],hashes)
    p = read_json(folder/'build/provenance.json')
    require(sha256(folder/'build/provenance.json') == COMBINED_PROOF and
            p['status'] == 'built' and p['build_exit_code'] == 0 and
            p['executable_sha256'] == COMBINED and p['baseline_executable_sha256'] == qualified.NORMAL and
            p['timing_proof_sha256'] == qualified.PROVENANCE and p['observer_proof_sha256'] == OBSERVER_PROOF and
            p['compilation_performed'] is False and p['runtime_inputs_unchanged'] is True and
            p['only_main_driver_output_replaced'] is True, 'combined build provenance differs')
    require((folder/'baseline-before.json').read_bytes() == (folder/'baseline-after.json').read_bytes(),
            'combined build changed baseline')
    for name in ('native_timed_observer.py','native_cycle_timing.py','native_step_observer.py',
                 'native_n12_two_node_timing_contract.md'):
        require((folder/'package'/name).read_bytes() == Path(__file__).with_name(name).read_bytes(),
                'combined package source differs: '+name)
    hashes[str((folder/'artifact-manifest.json').resolve())] = sha256(folder/'artifact-manifest.json')
    return dict(provenance_sha256=COMBINED_PROOF,executable_sha256=COMBINED,
                composition='qualified observer main + qualified timed driver + unchanged numerical objects')


def check_guard(folder, suffix):
    g = read_json(folder/f'guard-{suffix}.json')
    expected = dict(schema='native-n12-two-node-timing-guard-v1',passed=True,
        source_checkpoint_remote=SOURCE_REMOTE,source_checkpoint_sha256=SOURCE,
        science_admission_sha256=ADMISSION,independent_review_sha256=REVIEW,
        baseline_executable_sha256=qualified.NORMAL,timing_executable_sha256=qualified.TIMED,
        timed_observer_executable_sha256=COMBINED,timing_build_provenance_sha256=qualified.PROVENANCE,
        timed_observer_build_provenance_sha256=COMBINED_PROOF,
        source_manifest_sha256=qualified.SOURCE_MANIFEST,object_manifest_sha256=qualified.OBJECT_MANIFEST,
        verified_source_files=1972,verified_objects=228,gpu_wrapper_sha256=qualified.WRAPPER_SHA,
        timing_verify_exit_code=0,timed_observer_verify_exit_code=0)
    require(all(g.get(k) == v for k,v in expected.items()), 'runtime guard differs: '+suffix)
    for name,text in [('timing',qualified.VERIFY_TEXT),
                      ('timed-observer',qualified.VERIFY_TEXT+
                       'Verified invertible original-main observer and unchanged native source/object/executable inputs\n'+
                       'Verified unchanged native timed observer composition\n')]:
        log = folder/f'{name}-verify-{suffix}.txt'
        require(log.read_text() == text and sha256(log) == g[name.replace('-','_')+'_verify_log_sha256'],
                'actual instrumentation verification missing: '+name)


def check_environment(environment, remote):
    expected = dict(OMP_NUM_THREADS='1',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
        ZE_ENABLE_PCI_ID_DEVICE_ORDER='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
        MPIR_CVAR_ENABLE_GPU='1',ZE_AFFINITY_MASK=None,
        NEO_CACHE_DIR=str(remote/'cache/neo'),SYCL_CACHE_DIR=str(remote/'cache/sycl'),
        NATIVE_TIMING_WARMUP='10',NATIVE_TIMING_STEPS='20',NATIVE_TIMING_WINDOWS='3')
    expected.update({key:None for key in qualified.PROFILE})
    require(environment == expected,'environment/profiling/cache differs')


def check_command(command, kind, remote):
    cycle = FIRST if kind == 'init-only' else LAST
    executable = EXECUTABLES['normal' if kind == 'init-only' else kind][0]
    require(command == ['mpiexec','-n','24','-ppn','12',qualified.BINDING,qualified.WRAPPER,
        str(executable),'-r',SOURCE_REMOTE,'-d',str(remote),
        f'time/nlim={cycle}','time/tlim=2','time/ndiag=1000000000'], 'actual native command differs')


def collection(folder, hashes):
    p = read_json(folder/'package.json'); root = folder/'run-artifacts'
    require(p['schema'] == 'native-n12-two-node-timing-collection-v1' and
            (p['nodes'],p['ranks'],p['job_state'],p['exit_status']) == (2,24,'F',0) and
            (p['source_checkpoint_remote'],p['source_checkpoint_sha256'],p['science_admission_sha256'],
             p['independent_review_sha256']) == (SOURCE_REMOTE,SOURCE,ADMISSION,REVIEW),
            'collection execution/admission differs')
    files = read_json(root/'files.json'); require(len(files) == p['files'],'file count differs')
    qualified.check_files(root,files,hashes); case = root/CASE
    terminal = root/'terminal-qstat.txt'; text = terminal.read_text()
    job = re.search(r'^Job Id:\s*(\S+)',text,re.M)
    require(job and job[1].split('.')[0] == str(p['job_id']).split('.')[0] ==
            (case/'job-id.txt').read_text().strip().split('.')[0], 'actual/package/terminal job differs')
    for key,value in [('job_state','F'),('Exit_status','0'),('Job_Name','native-n12-timing-two'),
                      ('Resource_List.nodect','2')]:
        found = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',text,re.M)
        require(found and found[1] == value,'terminal field differs: '+key)
    for path in [terminal,*case.rglob('*')]:
        if path.is_file() and 'cache' not in path.relative_to(root).parts:
            require(str(path.resolve()) in hashes,'unbound consumed artifact: '+str(path))
    for suffix in ('before','after'):check_guard(case,suffix)
    owners = set()
    for line in (case/'binding.txt').read_text().splitlines():
        m = re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
        require(m is not None,'invalid binding line')
        host,rank,tile = m.groups(); rank = int(rank)
        require((host,rank) not in owners and 0 <= rank < 12 and tile == f'{rank//2}.{rank%2}',
                'duplicate/wrong tile binding');owners.add((host,rank))
    require(len(owners) == 24 and len({h for h,r in owners}) == 2,'wrong node/rank coverage')
    for path in (folder/'package.json',root/'files.json'):hashes[str(path.resolve())] = sha256(path)
    return case


def check_initial_capture(folder, source_path, source, hashes):
    check_command(shlex.split((folder/'command.txt').read_text()),'init-only',SIDE/CASE/'init-only')
    check_environment(read_json(folder/'environment.json'),SIDE/CASE/'init-only')
    require((folder/'exit-code.txt').read_text().strip() == '0' and
            'Terminating on cycle limit' in (folder/'run.log').read_text(),'initial capture did not complete')
    require([p.name for p in (folder/'rst').glob('*.rst')] == ['native_n12_5M.00008.rst'],
            'initial capture output differs')
    require(not (folder/'native-cycle-timing.json').exists() and not (folder/'native-stages.csv').exists(),
            'initial capture unexpectedly instrumented')
    found,records = checkpoint_set(folder,{FIRST},24,hashes)
    info = parse_native(Path(found[FIRST]['path']));check_final(info,source,FIRST)
    diagnostics = qualified.check_diagnostics(folder,[FIRST])
    fields, records_exact = initial_payloads(payload(found[FIRST]),payload(parse(source_path)))
    return dict(checkpoints=records,diagnostics=diagnostics,fields=fields,chi_records_bitwise_exact=records_exact,
        complete_padded_scalars=106*(18**3+144)*39,
        scope='Actual ordinary zero-step post-initialization state; diagnostic preparation changes only halo/workspace, then Finalize writes. FastFlow seed/output metadata may change. This capture is never used as the timed source.')


def initial_payloads(actual, expected):
    require(actual.shape == expected.shape and actual.ndim == 3 and actual.shape[2] == 39,
            'initial payload shape differs')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),'initial payload is nonfinite')
    # Same adopted short gates as n12_checkpoint.compare_fields; complete padded
    # fields are included here, so no owned point can be silently omitted.
    fields=[]
    for field in range(31):
        a,b=actual[:,:,field],expected[:,:,field];delta=np.abs(a-b)
        ratio=delta/((1e-14 if field==0 else 1e-12)+5e-10*np.abs(b))
        worst=np.unravel_index(np.argmax(ratio),ratio.shape)
        fields.append(dict(field=field,comparisons=int(a.size),failed_values=int(np.count_nonzero(ratio>1)),
            bitwise_equal=bool(np.array_equal(a.view('<u8'),b.view('<u8'))),
            max_absolute=float(delta.max()),max_tolerance_ratio=float(ratio[worst]),
            rms=float(np.sqrt(np.mean(delta*delta))),worst_gid=int(worst[0]),worst_slot=int(worst[1])))
    records_exact=np.array_equal(actual[:,:,31:].view('<u8'),expected[:,:,31:].view('<u8'))
    require(records_exact,'initial capture changed chi records')
    require(all(x['failed_values']==0 for x in fields),'post-init fields exceed existing short gates')
    return fields, bool(records_exact)


def audit(args, result):
    hashes = result['input_sha256']
    for path in (args.source_checkpoint,args.admission,args.independent_review):
        hashes[str(path.resolve())] = sha256(path)
    check_admission(read_json(args.admission),read_json(args.independent_review),sha256(args.source_checkpoint),
                    sha256(args.admission),sha256(args.independent_review))
    source = parse_native(args.source_checkpoint);check_source(source)
    result['unused_global_coarse_indices']['source'] = list(struct.unpack('<9i',source['fixed'][120:156]))
    result['timing_build'] = qualified.check_build(args.timing_build_artifacts,hashes)
    result['combined_build'] = check_combined_build(args.combined_build_artifacts,hashes)
    case = collection(args.collection,hashes); states,params,timings = {},{},{}
    result['initial_capture'] = check_initial_capture(case/'init-only',args.source_checkpoint,source,hashes)
    for kind in EXECUTABLES:
        folder = case/kind;remote = SIDE/CASE/kind
        check_command(shlex.split((folder/'command.txt').read_text()),kind,remote)
        check_environment(read_json(folder/'environment.json'),remote)
        require((folder/'exit-code.txt').read_text().strip() == '0' and
                'Terminating on cycle limit' in (folder/'run.log').read_text(),'native run did not complete')
        require([p.name for p in (folder/'rst').glob('*.rst')] == ['native_n12_5M.00008.rst'],
                'missing/extra final checkpoint')
        found,records = checkpoint_set(folder,{LAST},24,hashes);states[kind] = found[LAST]
        final = parse_native(Path(states[kind]['path']));check_final(final,source);params[kind] = final['parameters']
        result['checkpoints'][kind] = records
        result['unused_global_coarse_indices'][kind] = list(struct.unpack('<9i',final['fixed'][120:156]))
        result['diagnostic_coverage'][kind] = qualified.check_diagnostics(folder,[LAST])
        if kind != 'normal':
            timings[kind] = qualified.timing_windows(read_json(folder/'native-cycle-timing.json'),FIRST,24,CONFIG,'SYCL',True)
        else:require(not (folder/'native-cycle-timing.json').exists(),'normal unexpectedly instrumented')
        if kind == 'timed-observer':
            rows,summary = stages(folder,FIRST,LAST,24)
            result['observer'] = dict(completion=summary,stage_rows=len(rows),
                maximum_compatibility=max(r['compatibility'] for r in rows),
                maximum_clock=max(r['clock'] for r in rows),maximum_CFL=max(r['CFL'] for r in rows))
        else:
            require(not (folder/'native-stages.csv').exists() and not (folder/'native-step-observer.json').exists(),
                    'unobserved control contains observer')
    for kind in ('timed-normal','timed-observer'):
        require(params[kind] == params['normal'],'final parameter maps differ: '+kind)
        item = dict(state=compare(states['normal'],states[kind]),final_parameters_exact=True,
                    diagnostics=diagnostics_exact(case/'normal',case/kind,hashes))
        result['equivalence'][kind] = item
        require(item['diagnostics']['passed'],'ordinary diagnostics differ: '+kind)
    summary = {}
    for kind,r in timings.items():
        values = [w['seconds_per_complete_step'] for w in r['windows']]
        summary[kind] = dict(windows_seconds_per_step=values,median=statistics.median(values),
                            minimum=min(values),maximum=max(values))
    result['timing_summary'] = dict(cases=summary,
        observer_median_overhead_seconds=summary['timed-observer']['median']-summary['timed-normal']['median'],
        observer_median_ratio=summary['timed-observer']['median']/summary['timed-normal']['median'],
        reference_parity_evaluated=False,
        scope='Three repeated max-rank complete-cycle windows; additional repetitions needed if spread prevents a clear comparison')
    result.update(accepted_timings=timings,timings_accepted=True,passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection','source-checkpoint','admission','independent-review','timing-build-artifacts',
                 'combined-build-artifacts','output'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result = dict(schema='native-n12-two-node-timing-validation-v1',passed=False,timings_accepted=False,
        accepted_timings={},timing_summary=None,input_sha256={},checkpoints={},equivalence={},diagnostic_coverage={},
        unused_global_coarse_indices={},
        mesh_header_scope='All initialized fields exact; nine unused global coarse-index integers excluded and reported',
        initial_state_scope='Same exact source in three continuations; separate ordinary zero-step capture checks actual post-init fields at pre-existing short gates and chi records exactly',
        scope='Native solver/observer overhead controls only; no legacy parity claim, new resolution or eight-node run')
    try:audit(args,result)
    except Exception as exc:
        result.update(passed=False,timings_accepted=False,accepted_timings={},timing_summary=None,
                      failure=f'{type(exc).__name__}: {exc}')
    result['source_sha256'] = {name:sha256(Path(__file__).with_name(name)) for name in
        ('native_n12_two_node_timing_validate.py','native_timed_observer.py','native_n12_two_node_timing_contract.md','native_n12_two_node_initial_capture.md',
         'native_cycle_timing_gpu_validate.py','native_cycle_timing.py','native_cycle_timing.hpp',
         'native_step_observer_gpu_compare.py','native_step_observer_validate.py','native_step_observer.py',
         'n12_checkpoint.py','n12_long_history.py','restart_roundtrip.py')}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','timings_accepted','failure') if k in result}))
    return 0 if result['passed'] else 2

if __name__ == '__main__':raise SystemExit(main())
