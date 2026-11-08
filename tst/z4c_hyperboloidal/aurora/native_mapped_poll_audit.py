#!/usr/bin/env python3
"""Operational ABBA checks, reusing the admitted native timing/state readers."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import native_two_node_timing_audit as base
import native_mapped_poll_candidate as candidate
from native_n12_profile_audit import checkpoint, compare_checkpoints

ROOT=base.ROOT
WORK=ROOT/'validation/perf-two-node-v1/mapped-test-poll-v1'
LAUNCH=WORK/'launch'
CASE=ROOT/'runs/native-n12-mapped-poll-abba-v1'
ORDER=('timed-C0','timed-K0','timed-K1','timed-C1')
NORMAL=ROOT/'runs/native-n12-timing-two-v1/normal'
DIAGNOSTICS=('native_n12_5M.z4c_hyperboloidal.hst','native_n12_5M.horizon_summary_0.txt',
             'native_n12_5M.horizon_shape_0.txt')
require,sha,read_json,write_json=base.require,base.sha,base.read_json,base.write_json
base.PROFILE=base.PROFILE+('LD_PRELOAD','APS_STAT_LEVEL','APS_COLLECTION_MODE',
                         'APS_RESULT_DIR','APS_MPI_IMBALANCE','MPS_STAT_LEVEL',
                         'APS_IMBALANCE_TYPE','APS_STAT_BASE_DIR')


def unprofiled():
    require('LD_PRELOAD' not in os.environ and
            not any(k.startswith(('APS_','MPS_')) for k in os.environ),
            'preload or APS/MPS instrumentation environment present')


def binding():
    b=read_json(LAUNCH/'build-binding.json')
    require(b['schema']=='native-mapped-poll-abba-binding-v1','wrong build binding')
    for group in ('build_proofs','executables','reference_files'):
        for name,digest in b[group].items():require(sha(name)==digest,'changed '+group+': '+name)
    p=read_json(WORK/'candidate-build/provenance.json')
    require(p['candidate_source_sha256']==candidate.CANDIDATE and
            p['same_timing_driver_control_candidate'] is True and
            p['changed_numerical_sources']==['src/bvals/mapped_exchange.cpp'],'candidate scope differs')
    control=p['control_executable'];experiment=p['executables']['candidate-timed']
    require(b['case_executables']=={name:(control if '-C' in name else experiment) for name in ORDER},
            'ABBA executable selection differs')
    return b


def guard(suffix):
    unprofiled()
    base.guard(SimpleNamespace(admission=LAUNCH/'admission.json',directory=CASE,suffix=suffix))
    b=binding()
    log=CASE/f'candidate-verify-{suffix}.txt'
    with log.open('w') as f:
        code=subprocess.run(['/usr/bin/python3.11',str(LAUNCH/'native_mapped_poll_candidate.py'),
            'verify','--output',str(WORK/'candidate-build')],stdout=f,stderr=subprocess.STDOUT).returncode
    require(code==0,'candidate/timer provenance verification failed')
    write_json(CASE/f'candidate-guard-{suffix}.json',dict(passed=True,
        build_binding_sha256=sha(LAUNCH/'build-binding.json'),
        base_guard_sha256=sha(CASE/f'guard-{suffix}.json'),verify_log_sha256=sha(log)))


def completed(name):
    require(name in ORDER,'unknown ABBA case');b=binding();folder=CASE/name
    base.CASE=CASE;base.EXECUTABLES={k:(Path(v),b['executables'][v]) for k,v in b['case_executables'].items()}
    operation=base.completed_case(folder,name)
    comparison=compare_checkpoints(checkpoint(NORMAL/'rst/native_n12_5M.00008.rst',2556),
                                   checkpoint(folder/'rst/native_n12_5M.00008.rst',2556))
    require(all((folder/n).read_bytes()==(NORMAL/n).read_bytes() for n in DIAGNOSTICS),
            'ordinary final diagnostic bytes differ from admitted normal control')
    require({p.name for pattern in ('*.hst','*.horizon*.txt') for p in folder.glob(pattern)}
            ==set(DIAGNOSTICS),'missing/extra ordinary diagnostic output')
    timer=read_json(folder/'native-cycle-timing.json')
    require(timer['backend']=='SYCL' and timer['restart'] is True and
            (timer['initial_time'],timer['final_time'],timer['dt'])==
            (base.expected_time(2486),base.expected_time(2556),base.DT),
            'timer backend/restart/clock differs')
    require((timer['bulk_points'],timer['scri_points'],timer['blocks'])==(229392,864,106)
            and timer['console_ndiag']==1000000000 and
            timer['profiling_library_loaded'] is False and timer['profiling_environment']==[]
            and timer['timed_output_count']==timer['timed_console_count']==0,
            'timing workload or profiling/output differs')
    for index,w in enumerate(timer['windows']):
        first=2496+20*index
        require((w['index'],w['first_cycle'],w['last_cycle'],w['complete_steps'])==
                (index,first,first+20,20) and (w['first_time'],w['last_time'])==
                (base.expected_time(first),base.expected_time(first+20)),
                'partial/repeated or wrongly clocked complete-step window')
        elapsed=w['max_rank_seconds']
        require(math.isfinite(elapsed) and elapsed>0 and
                w['seconds_per_complete_step']==elapsed/20 and
                w['bulk_cell_updates']==229392*20 and
                w['bulk_cell_updates_per_second']==229392*20/elapsed,
                'invalid timer complete-update counts/rates')
        values=w['rank_order_seconds']
        require(len(values)==24 and all(math.isfinite(x) and x>0 for x in values),
                'missing/nonfinite rank timing array')
        require(max(values)==w['max_rank_seconds'] and
                w['max_rank_ids']==[i for i,v in enumerate(values) if v==max(values)],
                'rank maximum does not reproduce timing window')
    for key in ('startup_max_seconds','warmup_max_seconds','execute_max_seconds','finalize_max_seconds'):
        require(math.isfinite(timer[key]) and timer[key]>0,'invalid separated timing: '+key)
    return dict(schema='native-mapped-poll-case-v1',passed=True,case=name,
                operational=operation,state_equivalence=comparison,
                ordinary_diagnostics_bitwise_equal=True,rank_arrays_checked=True,
                performance_admission=False)


def screen():
    for name in ORDER[:2]:require(read_json(CASE/name/'operational.json')==completed(name),
                                  'first-pair state proof changed')
    c,k=[read_json(CASE/name/'native-cycle-timing.json')['windows'] for name in ORDER[:2]]
    cv,kv=[[w['max_rank_seconds']/w['complete_steps'] for w in windows] for windows in (c,k)]
    reject=min(kv)>=1.05*max(cv)
    report=dict(schema='native-mapped-poll-screen-v1',clearly_slower=reject,
        prespecified_rule='min(K0 seconds/step) >= 1.05 * max(C0 seconds/step)',
        control_seconds_per_step=cv,candidate_seconds_per_step=kv,
        full_state_and_diagnostics_passed=True,performance_improvement_claim=False)
    write_json(CASE/'first-pair-screen.json',report)
    print('reject' if reject else 'continue')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('guard');p.add_argument('suffix',choices=('before','after'))
    p=sub.add_parser('completed');p.add_argument('name',choices=ORDER)
    p=sub.add_parser('environment');p.add_argument('directory',type=Path)
    p=sub.add_parser('binding');p.add_argument('path',type=Path)
    sub.add_parser('screen')
    a=parser.parse_args()
    if a.mode=='guard':guard(a.suffix)
    elif a.mode=='completed':write_json(CASE/a.name/'operational.json',completed(a.name))
    elif a.mode=='environment':unprofiled();write_json(a.directory/'environment.json',base.environment(a.directory))
    elif a.mode=='screen':screen()
    else:base.binding(a.path)
