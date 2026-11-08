#!/usr/bin/env python3
"""Operational cache-capacity ABBA checks on the unchanged native timed executable."""
import argparse
import json
import math
import os
import socket
import shlex
from pathlib import Path
import subprocess
from types import SimpleNamespace

import native_two_node_timing_audit as base
import native_cycle_timing as timing
from native_n12_profile_audit import checkpoint, compare_checkpoints

ROOT=base.ROOT
WORK=ROOT/'validation/perf-two-node-v1/ipc-cache-v1'
TIMING_WORK=ROOT/'validation/perf-two-node-v1/mapped-test-poll-v1'
TIMED='0ef1d6921b2184baa13446e267067ce183adc6341545580678822f7670daebb8'
TIMING_PROOF='4112e39f65c14d6c9ed15444f1666dee364acdc808c70a6ff89bda51a0886654'
LAUNCH=WORK/'launch'
CASE=ROOT/'runs/native-n12-ipc-cache-abba-v1'
ORDER=('timed-C0','timed-K0','timed-K1','timed-C1')
NORMAL=ROOT/'runs/native-n12-timing-two-v1/normal'
DIAGNOSTICS=('native_n12_5M.z4c_hyperboloidal.hst','native_n12_5M.horizon_summary_0.txt',
             'native_n12_5M.horizon_shape_0.txt')
require,sha,read_json,write_json=base.require,base.sha,base.read_json,base.write_json
base.PROFILE=base.PROFILE+('LD_PRELOAD','APS_STAT_LEVEL','APS_COLLECTION_MODE',
                         'APS_RESULT_DIR','APS_MPI_IMBALANCE','MPS_STAT_LEVEL',
                         'APS_IMBALANCE_TYPE','APS_STAT_BASE_DIR')


CACHE_COMMON={'MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE':'specialized',
              'MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE':'limited'}
CAPACITY='MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES'
ALIASES=[key.replace('MPIR_CVAR_',prefix) for key in (*CACHE_COMMON,CAPACITY)
         for prefix in ('MPICH_','MPIR_PARAM_')]
BASE_ENVIRONMENT=dict(base.ENVIRONMENT)
CACHE_EVIDENCE='3c86b9aa9650ffa55692d9a5a81d9f0d0b47ded875235346ab7fe058e46b3288'


def configure(name):
    require(name in ORDER,'unknown cache case')
    base.ENVIRONMENT=dict(BASE_ENVIRONMENT,**CACHE_COMMON,**{CAPACITY:'16' if '-C' in name else '128'},
                          **{key:None for key in ALIASES})


def unprofiled():
    require('LD_PRELOAD' not in os.environ and
            not any(k.startswith(('APS_','MPS_')) for k in os.environ),
            'preload or APS/MPS instrumentation environment present')


def rank_environment():
    return dict(host=socket.gethostname(),
        local_rank=int(os.environ.get('MPI_LOCALRANKID',os.environ.get('PALS_LOCAL_RANKID','-1'))),
        tile=os.environ.get('ZE_AFFINITY_MASK'),implicit_scaling=os.environ.get('EnableImplicitScaling'),
        cache={key:os.environ.get(key) for key in (*CACHE_COMMON,CAPACITY,*ALIASES)})


def rank_cache(path,name):
    configure(name);seen=set()
    expected={key:base.ENVIRONMENT[key] for key in (*CACHE_COMMON,CAPACITY,*ALIASES)}
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);r=row['local_rank'];key=(row['host'],r)
        require(set(row)=={'host','local_rank','tile','implicit_scaling','cache'} and
                0<=r<12 and row['tile']==f'{r//2}.{r%2}' and row['implicit_scaling']=='0'
                and row['cache']==expected and key not in seen,'rank cache environment/binding differs')
        seen.add(key)
    require(len(seen)==24 and len({h for h,r in seen})==2,'incomplete24-rank cache environment probe')


def binding():
    b=read_json(LAUNCH/'build-binding.json')
    require(b['schema']=='native-ipc-cache-abba-binding-v1','wrong build binding')
    require(b['cache_source_evidence_sha256']==CACHE_EVIDENCE and
            sha(LAUNCH/'cache-source-evidence.json')==CACHE_EVIDENCE,'cache source evidence changed')
    for group in ('build_proofs','executables','reference_files'):
        for name,digest in b[group].items():require(sha(name)==digest,'changed '+group+': '+name)
    p=read_json(TIMING_WORK/'timing-build/provenance.json')
    require(sha(TIMING_WORK/'timing-build/provenance.json')==TIMING_PROOF and
            p['executable_sha256']==TIMED and p['native_executable_sha256']==base.NORMAL,
            'not the qualified unchanged native rank-export timer')
    require(b['case_executables']=={name:p['executable'] for name in ORDER},
            'cache test must use the same unchanged executable in all cases')
    require(b['cache_capacity']=={name:16 if '-C' in name else 128 for name in ORDER}
            and b['common_cache_environment']==CACHE_COMMON and b['cleared_aliases']==ALIASES,
            'cache factor/common configuration differs')
    return b


def guard(suffix):
    unprofiled()
    base.guard(SimpleNamespace(admission=LAUNCH/'admission.json',directory=CASE,suffix=suffix))
    b=binding()
    log=CASE/f'rank-timing-verify-{suffix}.txt'
    with log.open('w') as f:
        code=subprocess.run(['/usr/bin/python3.11',str(TIMING_WORK/'package/native_cycle_timing.py'),
            'verify','--output',str(TIMING_WORK/'timing-build')],stdout=f,stderr=subprocess.STDOUT).returncode
    require(code==0,'unchanged rank-export timer provenance verification failed')
    write_json(CASE/f'cache-guard-{suffix}.json',dict(passed=True,
        build_binding_sha256=sha(LAUNCH/'build-binding.json'),
        base_guard_sha256=sha(CASE/f'guard-{suffix}.json'),verify_log_sha256=sha(log)))



def completed(name):
    require(name in ORDER,'unknown ABBA case');configure(name);b=binding();folder=CASE/name
    base.CASE=CASE;base.EXECUTABLES={k:(Path(v),b['executables'][v]) for k,v in b['case_executables'].items()}
    operation=base.completed_case(folder,name)
    require(shlex.split((folder/'rank-cache-command.txt').read_text())==
            ['mpiexec','-n','24','-ppn','12',base.BINDING,base.WRAPPER,
             '/usr/bin/python3.11',str(LAUNCH/'native_ipc_cache_audit.py'),'rank-environment'],
            'rank cache probe command differs')
    rank_cache(folder/'rank-cache.txt',name)
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
    return dict(schema='native-ipc-cache-case-v1',passed=True,case=name,
                operational=operation,state_equivalence=comparison,
                ordinary_diagnostics_bitwise_equal=True,rank_arrays_checked=True,
                rank_cache_environment_verified=True,
                cache_observation_scope='24 launcher/wrapper rank probes before solver; no in-solver MPI CVAR readback',
                performance_admission=False)


def screen():
    for name in ORDER[:2]:require(read_json(CASE/name/'operational.json')==completed(name),
                                  'first-pair state proof changed')
    c,k=[read_json(CASE/name/'native-cycle-timing.json')['windows'] for name in ORDER[:2]]
    cv,kv=[[w['max_rank_seconds']/w['complete_steps'] for w in windows] for windows in (c,k)]
    reject=min(kv)>=1.05*max(cv)
    report=dict(schema='native-ipc-cache-screen-v1',clearly_slower=reject,
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
    sub.add_parser('rank-environment')
    p=sub.add_parser('rank-cache');p.add_argument('path',type=Path);p.add_argument('name',choices=ORDER)
    a=parser.parse_args()
    if a.mode=='guard':guard(a.suffix)
    elif a.mode=='completed':write_json(CASE/a.name/'operational.json',completed(a.name))
    elif a.mode=='environment':configure(a.directory.name);unprofiled();write_json(a.directory/'environment.json',base.environment(a.directory))
    elif a.mode=='screen':screen()
    elif a.mode=='rank-environment':print(json.dumps(rank_environment(),separators=(',',':')))
    elif a.mode=='rank-cache':rank_cache(a.path,a.name)
    else:base.binding(a.path)
