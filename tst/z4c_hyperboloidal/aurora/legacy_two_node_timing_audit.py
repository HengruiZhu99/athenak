#!/usr/bin/env python3
"""Frozen predecessor suite bracketed by unchanged native runs; no qsub."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from native_two_node_timing_audit import (ROOT,PROFILE,DT,WRAPPER,WRAPPER_SHA,BINDING,
    binding,expected_time,read_json,require,sha,write_json)

WORK=ROOT/'validation/perf-two-node-v1/legacy-run-v2'
BUILD=ROOT/'validation/perf-two-node-v1/legacy'
ENDPOINT=BUILD/'endpoint-diagnostics-v1'
SEED=ROOT/'validation/perf-two-node-v1/legacy-seed'
CASE=ROOT/'runs/legacy-n12-timing-two-v2'
SEED_SHA='27245d19010dee629d12c6e59872564bd70c19af8c06aa41c46cbe6b58143f18'
ORIGINAL=ROOT/'validation/original-eight/evolve-static'
ORIGINAL_SHA='494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753'
KINDS=('immutable','control','v3','v6','v8')
FLAGS=('HYPERBOLIC_PROFILE_SMALL_GROUPS','HYPERBOLIC_BULK_STREAM','HYPERBOLIC_HOST_SURFACE',
       'HYPERBOLIC_SPARSE_POISON')
NATIVE=('timed-native-before','timed-native-after')
PROFILE=(*PROFILE,'LD_PRELOAD','APS_STAT_LEVEL','APS_COLLECTION_MODE','APS_RESULT_DIR',
         'APS_MPI_IMBALANCE','MPS_STAT_LEVEL','APS_IMBALANCE_TYPE','APS_STAT_BASE_DIR')
CACHE_COMMON={'MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE':'specialized',
              'MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE':'limited',
              'MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES':'128'}
CACHE_ALIASES=tuple(key.replace('MPIR_CVAR_',prefix) for key in CACHE_COMMON
                    for prefix in ('MPICH_','MPIR_PARAM_'))
CACHE_ENVIRONMENT=dict(CACHE_COMMON,**{key:None for key in CACHE_ALIASES})


def native_adapter(guard=False):
    # Fixed composition, confined to native-only CLI invocations. The imported
    # reader keeps its exact existing endpoint/window/rank checks and binary pins.
    import native_ipc_cache_audit as native
    native.ORDER=NATIVE
    native.CASE=CASE/'native-guards' if guard else CASE
    native.LAUNCH=WORK
    native.base.PROFILE=tuple(dict.fromkeys((*native.base.PROFILE,*FLAGS)))
    return native


def build_directory(kind): return BUILD/('control-v3' if kind=='control' else kind)


def builder_path(kind):
    return BUILD/('control-v3-package' if kind=='control' else 'package')/'legacy_n12_benchmark.py'


def environment(kind):
    out=CASE/kind
    expected=dict(OMP_NUM_THREADS='1',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
        ZE_ENABLE_PCI_ID_DEVICE_ORDER='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',MPIR_CVAR_ENABLE_GPU='1',
        ZE_AFFINITY_MASK=None,NEO_CACHE_DIR=str(out/'cache/neo'),SYCL_CACHE_DIR=str(out/'cache/sycl'))
    expected.update({key:None for key in (*PROFILE,*FLAGS)})
    expected.update(CACHE_ENVIRONMENT)
    if kind in ('v3','v6','v8'):
        expected.update(HYPERBOLIC_PROFILE_SMALL_GROUPS='1',HYPERBOLIC_BULK_STREAM='1')
    if kind=='v8':expected['HYPERBOLIC_HOST_SURFACE']='1'
    return expected


def command(kind):
    launch=['mpiexec','-n','24','-ppn','12',BINDING,WRAPPER]
    if kind=='immutable':return launch+[str(ORIGINAL),str(SEED/'run.in'),str(CASE/kind/'output'),str(SEED/'restart.bin')]
    return launch+[str(build_directory(kind)/'legacy_n12_benchmark'),str(SEED),str(CASE/kind/'output')]


def endpoint_command():
    return ['mpiexec','-n','24','-ppn','12',BINDING,WRAPPER,
        str(ENDPOINT/'build/legacy_n12_endpoint_diagnostics'),str(SEED),str(CASE/'control/output'),
        str(CASE/'control/diagnostics')]


def endpoint_environment():
    result=environment('control')
    result.update(NEO_CACHE_DIR=str(CASE/'control/postprocess-cache/neo'),
                  SYCL_CACHE_DIR=str(CASE/'control/postprocess-cache/sycl'))
    return result


def seed():
    require(sha(SEED/'seed.json')==SEED_SHA,'seed conversion proof changed')
    proof=read_json(SEED/'seed.json')
    require(proof['passed'] is True and proof['initial_fields_bitwise_native'] is True and
            proof['initial_records_bitwise_native'] is True and
            (proof['start_cycle'],proof['end_cycle'],proof['warmup'],proof['window_steps'],proof['windows'])==
            (2486,2556,10,20,3),'wrong converted seed/workload')
    for name,digest in proof['output_sha256'].items():require(sha(SEED/name)==digest,'seed file changed: '+name)
    return proof


def guard(suffix):
    require(all(k not in os.environ for k in PROFILE),'profiling environment requested')
    require(not any(k.startswith(('APS_','MPS_')) for k in os.environ),
            'APS/MPS instrumentation environment present')
    require({k:os.environ.get(k) for k in CACHE_ENVIRONMENT}==CACHE_ENVIRONMENT,
            'common native/legacy cache configuration differs')
    require(sha(WRAPPER)==WRAPPER_SHA and sha(ORIGINAL)==ORIGINAL_SHA,'immutable binary/wrapper changed')
    s=seed();descriptor=read_json(WORK/'build-bindings.json')
    require(descriptor['schema']=='legacy-n12-two-node-build-bindings-v1' and
            set(descriptor['variants'])==set(KINDS[1:]),'incomplete build binding')
    builds={}
    for kind,item in descriptor['variants'].items():
        directory=build_directory(kind);proof=directory/'provenance.json';p=read_json(proof)
        require(sha(proof)==item['provenance_sha256'] and p['variant']==kind and p['status']=='built' and
                p['build_exit_code']==0 and p['original_runtime_inputs_unchanged'] is True and
                item['build_directory']==str(directory) and item['builder']==str(builder_path(kind)) and
                sha(directory/'legacy_n12_benchmark')==item['executable_sha256']==p['executable_sha256'],
                'benchmark build changed: '+kind)
        if kind=='control':
            generation=BUILD/'control-v3-package/generation.json';g=read_json(generation)
            require(sha(generation)==item['generation_sha256']==
                    '0654827cad627f017b4a0f152b2c798bce3652cc3ad1d0ec83f58fb5a8062244' and
                    g['exact_inverse_to_qualified_measurement_entry'] is True and
                    g['timed_step_statements_unchanged'] is True and
                    g['archive_numerical_sources_modified'] is False,'control entry generation differs')
            for name,digest in g['output_sha256'].items():
                require(sha(generation.parent/name)==digest,'control generated package changed: '+name)
        log=CASE/f'{kind}-verify-{suffix}.txt'
        with log.open('w') as output:
            code=subprocess.run(['/usr/bin/python3.11',str(builder_path(kind)),
                'verify','--output',str(directory)],stdout=output,stderr=subprocess.STDOUT).returncode
        require(code==0 and log.read_text()=='Verified benchmark and unchanged archived predecessor inputs\n',
                'benchmark verifier failed: '+kind)
        builds[kind]=dict(**item,verify_exit_code=code,verify_log_sha256=sha(log))
    item=descriptor['endpoint_diagnostics'];proof=ENDPOINT/'build/provenance.json';p=read_json(proof)
    require(item['build_directory']==str(ENDPOINT/'build') and
            item['builder']==str(ENDPOINT/'package/legacy_n12_endpoint_diagnostics.py') and
            sha(proof)==item['provenance_sha256'] and p['schema']=='legacy-n12-endpoint-diagnostics-build-v1'
            and p['status']=='built' and p['build_exit_code']==0 and p['evolution_steps']==0 and
            p['original_runtime_inputs_unchanged'] is True and
            sha(ENDPOINT/'build/legacy_n12_endpoint_diagnostics')==item['executable_sha256']==p['executable_sha256'],
            'endpoint-only build differs')
    log=CASE/f'endpoint-verify-{suffix}.txt'
    with log.open('w') as output:
        code=subprocess.run(['/usr/bin/python3.11',item['builder'],'verify','--output',item['build_directory']],
            stdout=output,stderr=subprocess.STDOUT).returncode
    require(code==0 and log.read_text()=='Verified unchanged diagnostic-only v8 endpoint build\n',
            'endpoint-only verifier failed')
    result=dict(schema='legacy-n12-two-node-runtime-guard-v1',passed=True,seed_sha256=SEED_SHA,
        seed_files=s['output_sha256'],immutable_executable_sha256=ORIGINAL_SHA,gpu_wrapper_sha256=WRAPPER_SHA,
        build_bindings_sha256=sha(WORK/'build-bindings.json'),variants=builds,
        common_cache_environment=CACHE_ENVIRONMENT,
        endpoint_diagnostics=dict(**item,verify_exit_code=code,verify_log_sha256=sha(log)))
    write_json(CASE/f'guard-{suffix}.json',result)


def completed(kind):
    directory=CASE/kind;out=directory/'output';s=seed()
    require(shlex.split((directory/'command.txt').read_text())==command(kind),'actual command differs')
    require(read_json(directory/'environment.json')==environment(kind),'actual environment differs')
    require((directory/'exit-code.txt').read_text().strip()=='0','case failed')
    if kind=='immutable':
        plan=read_json(out/'plan.json');final=read_json(out/'result.json')
        require(plan['backend']=='SYCL' and plan['ranks']==24 and plan['nlim']==2556 and plan['tlim']==2
                and (plan['bulk_points'],plan['surface_points'])==(229392,864),'immutable workload differs')
        require(final['status']=='step_limit' and not final['failure'] and final['steps']==2556 and
                final['time_M']==3*expected_time(2556),'immutable endpoint differs')
        require(len((out/'steps.csv').read_text().splitlines())==71,'incomplete immutable steps')
        for label,cycle in [('initial',2486),('final',2556)]:
            clock=read_json(out/label/'time.json')
            require(clock['steps']==cycle and clock['t']==expected_time(cycle),'immutable snapshot clock differs')
            for name,count in [('fields.bin',229392*31*8),('boundary.bin',864*31*8)]:
                require((out/label/name).stat().st_size==count,'incomplete immutable snapshot')
            require((out/label/'restart.bin').is_file(),'missing immutable SP1 snapshot')
        proof=dict(schema='legacy-n12-immutable-control-run-v1',executable_sha256=ORIGINAL_SHA,
            seed_sha256=s['output_sha256']['restart.bin'],input_sha256=s['output_sha256']['run.in'],
            configuration_sha256=s['output_sha256']['configuration.cfg'],ranks=24,
            command_sha256=sha(directory/'command.txt'),exit_code_sha256=sha(directory/'exit-code.txt'))
        write_json(directory/'immutable-control-provenance.json',proof)
    else:
        r=read_json(out/'results.json')
        schema='legacy-n12-original-control-benchmark-v1' if kind=='control' else 'legacy-n12-matched-benchmark-v1'
        require(r['schema']==schema and r['passed'] is True and
                r['backend']=='SYCL' and (r['ranks'],r['start_cycle'],r['end_cycle'],r['warmup'],
                r['window_steps'],r['windows'])==(24,2486,2556,10,20,3) and r['dt']==DT and
                r['time']==expected_time(2556),'wrapper endpoint/workload differs')
        require(r['initial_fields_bitwise_seed'] and r['initial_records_bitwise_seed'] and
                not r['profiling_library_loaded'] and not r['detailed_timers_enabled'],'wrong initial/profiling state')
        expected={f'accepted-final-{region}-rank{rank}.bin' for region in ('bulk','scri') for rank in range(24)}
        if kind=='control':expected.add('endpoint-restart.bin')
        require({p.name for p in out.glob('*.bin')}==expected,'incomplete final31+8 rank files')
        if kind!='control':require((out/'horizon.json').is_file(),'missing final horizon')
        require(len(r['maximum_rank_window_seconds'])==3 and len(r['rank_window_seconds'])==72 and
                all(math.isfinite(x) and x>0 for x in r['rank_window_seconds']),'invalid timing records')
        require((out/'exit-code.txt').read_text().strip()=='0','missing wrapper exit record')
    result=dict(schema='legacy-n12-two-node-case-operational-v1',passed=True,kind=kind,
                scientific_comparison_pending=True,performance_admission=False)
    write_json(directory/'operational.json',result)


def completed_endpoint():
    directory=CASE/'control';out=directory/'diagnostics'
    require(shlex.split((directory/'endpoint-command.txt').read_text())==endpoint_command(),
            'postprocessor command differs')
    require(read_json(directory/'endpoint-environment.json')==endpoint_environment() and
            (directory/'endpoint-exit-code.txt').read_text().strip()=='0','postprocessor environment/exit differs')
    for suffix in ('before','after'):
        require((directory/f'endpoint-inputs-verify-{suffix}.txt').read_text()==
                'Verified unchanged original-control endpoint and2486 horizon seed\n','endpoint inputs unverified')
    bound=read_json(directory/'endpoint-inputs.json')
    require(bound['schema']=='legacy-n12-endpoint-inputs-v1' and bound['passed'] is True and
            bound['accepted_directory']==str(directory/'output'),'wrong endpoint input binding')
    for name,digest in bound['files'].items():require(sha(name)==digest,'endpoint input changed: '+name)
    r=read_json(out/'results.json')
    require(r['schema']=='legacy-n12-endpoint-diagnostics-v1' and r['passed'] is True and
            r['backend']=='SYCL' and r['ranks']==24 and r['evolution_steps']==0 and
            r['timing_measurement'] is False and (r['cycle'],r['time'],r['dt'])==(2556,expected_time(2556),DT)
            and (r['bulk_points'],r['scri_points'])==(229392,864) and
            r['post_load_fields_bitwise_control_endpoint'] is True and
            r['post_load_records_bitwise_control_endpoint'] is True and
            r['diagnostics_preserved_chi_records_bitwise'] is True and
            (out/'exit-code.txt').read_text().strip()=='0','postprocessor did not complete without evolution')
    require({p.name for p in out.glob('*.bin')}==
        {f'accepted-final-{region}-rank{rank}.bin' for region in ('bulk','scri') for rank in range(24)},
        'incomplete post-load diagnostic snapshots')
    require((out/'horizon.json').is_file(),'missing postprocessor horizon')
    write_json(directory/'endpoint-operational.json',dict(passed=True,evolution_steps=0,
        scientific_comparison_pending=True,performance_admission=False))


def collect(job):
    status=subprocess.check_output(['qstat','-xf',str(job)],text=True)
    m=re.search(r'^Job Id:\s*(\S+)\s*$',status,re.M)
    fields=dict(re.findall(r'^\s+([A-Za-z_.]+)\s*=\s*(\S+)\s*$',status,re.M))
    require(m and m[1].split('.')[0]==str(job) and all(fields.get(k)==v for k,v in
        [('job_state','F'),('Exit_status','0'),('Job_Name','legacy-n12-timing-two'),
         ('Resource_List.nodect','2')]),'not expected terminalF/0 job')
    require((CASE/'job-id.txt').read_text().strip().split('.')[0]==str(job) and
            (CASE/'pipeline-complete.txt').is_file(),'incomplete/wrong allocation')
    require(read_json(CASE/'guard-before.json')==read_json(CASE/'guard-after.json'), 'runtime guards differ')
    for stem in ('guard','cache-guard'):
        directory=CASE/'native-guards'
        require(read_json(directory/f'{stem}-before.json')==
                read_json(directory/f'{stem}-after.json'),'native bracket guards differ')
    binding(CASE/'binding.txt')
    for kind in KINDS:completed(kind)
    completed_endpoint()
    # Keep the native reader's module globals out of legacy validation/collection.
    for kind in NATIVE:
        subprocess.run(['/usr/bin/python3.11',str(WORK/'legacy_two_node_timing_audit.py'),
                        'native-completed',kind],check=True)
    out=ROOT/f'validation/legacy-two-node-timing-artifacts-{job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file() and
           not {'cache','postprocess-cache'}.intersection(p.relative_to(CASE).parts)}
    for p in WORK.iterdir():
        if p.is_file() and p.suffix in ('.py','.sh','.pbs','.json','.sha256'):files[str(p.relative_to(ROOT))]=p
    # Seed payload remains remote and already has an exact verified local copy.
    files[str((SEED/'seed.json').relative_to(ROOT))]=SEED/'seed.json'
    native_proof=ROOT/'validation/perf-two-node-v1/mapped-test-poll-v1/timing-build/provenance.json'
    files[str(native_proof.relative_to(ROOT))]=native_proof
    for kind in KINDS[1:]:
        for name in ('provenance.json','build.log','verify.log','benchmark.mk'):
            p=build_directory(kind)/name;files[str(p.relative_to(ROOT))]=p
    for package in ('package','control-v3-package','endpoint-diagnostics-v1/package'):
        for p in (BUILD/package).iterdir():
            if p.is_file():files[str(p.relative_to(ROOT))]=p
    for name in ('provenance.json','build.log','verify.log','endpoint.mk'):
        p=ENDPOINT/'build'/name;files[str(p.relative_to(ROOT))]=p
    files['terminal-qstat.txt']=out/'terminal-qstat.txt'
    manifest={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    write_json(out/'files.json',manifest);archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
        t.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='legacy-n12-two-node-timing-collection-v2',job_id=str(job),job_state='F',exit_status=0,
        nodes=2,ranks=24,files=len(files),seed_sha256=SEED_SHA,archive=str(archive),sha256=sha(archive),
        native_brackets=list(NATIVE),common_cache_environment=CACHE_ENVIRONMENT,
        archive_bytes=archive.stat().st_size,total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_comparison_pending=True,
        performance_admission=False)
    write_json(out/'package.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('guard');a.add_argument('suffix',choices=('before','after'))
    a=s.add_parser('environment');a.add_argument('kind',choices=KINDS)
    a=s.add_parser('binding');a.add_argument('path',type=Path)
    a=s.add_parser('completed');a.add_argument('kind',choices=KINDS)
    s.add_parser('endpoint-environment');s.add_parser('completed-endpoint')
    a=s.add_parser('collect');a.add_argument('--job',type=int,required=True)
    a=s.add_parser('native-guard');a.add_argument('suffix',choices=('before','after'))
    a=s.add_parser('native-environment');a.add_argument('kind',choices=NATIVE)
    a=s.add_parser('native-completed');a.add_argument('kind',choices=NATIVE)
    a=s.add_parser('native-rank-cache');a.add_argument('kind',choices=NATIVE)
    a=p.parse_args()
    if a.mode=='guard':guard(a.suffix)
    elif a.mode=='environment':
        e=environment(a.kind);require({k:os.environ.get(k) for k in e}==e,'execution environment differs')
        write_json(CASE/a.kind/'environment.json',e)
    elif a.mode=='binding':binding(a.path)
    elif a.mode=='completed':completed(a.kind)
    elif a.mode=='endpoint-environment':
        e=endpoint_environment();require({k:os.environ.get(k) for k in e}==e,'endpoint environment differs')
        write_json(CASE/'control/endpoint-environment.json',e)
    elif a.mode=='completed-endpoint':completed_endpoint()
    elif a.mode=='native-guard':native_adapter(True).guard(a.suffix)
    elif a.mode=='native-environment':
        native=native_adapter();native.configure(a.kind);native.unprofiled()
        write_json(CASE/a.kind/'environment.json',native.base.environment(CASE/a.kind))
    elif a.mode=='native-completed':
        native=native_adapter();write_json(CASE/a.kind/'operational.json',native.completed(a.kind))
    elif a.mode=='native-rank-cache':
        native=native_adapter();native.rank_cache(CASE/a.kind/'rank-cache.txt',a.kind)
    else:collect(a.job)
