#!/usr/bin/env python3
"""Offline reader controls using real2486/build/CPU fixtures; no GPU evidence."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import numpy as np

import native_n12_two_node_timing_validate as reader
from n12_checkpoint import DT, expected_time, parse_native, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);checks=[]
    def ok(name,call):call();checks.append(dict(name=name,passed=True,expected='accept'))
    def no(name,call):
        try:call()
        except (ValueError,KeyError,AssertionError) as error:
            checks.append(dict(name=name,passed=True,expected='reject',message=str(error)));return
        raise AssertionError('negative control accepted: '+name)
    source_path=Path('build-native/aurora-n12-common-07/run-artifacts/runs/native-n12-common-07/rst/native_n12_5M.00007.rst')
    admission_path=Path('build-native/n12-common-segment-07-science-v1/continuation-admission.json')
    review_path=Path('docs/z4c_hyperboloidal/evidence/native-n12-common-07-independent-review.json')
    source=parse_native(source_path);admission=reader.read_json(admission_path);review=reader.read_json(review_path)
    ok('actual reviewed2486 source header',lambda:reader.check_source(source))
    ok('actual bounded2486 science and independent review',lambda:reader.check_admission(admission,review,
        sha256(source_path),sha256(admission_path),sha256(review_path)))
    no('wrong source bytes',lambda:reader.check_admission(admission,review,'0'*64,reader.ADMISSION,reader.REVIEW))
    for key,value in [('cycle',2485),('time',source['time']+DT),('dt',DT*2),('metadata',source['metadata'][:4]+(96,0,1,0))]:
        changed=copy.deepcopy(source);changed[key]=value
        no('source '+key,lambda changed=changed:reader.check_source(changed))
    for key,value in [('output1/last_time','1'),('output2/file_number','7'),('output1/dcycle','1'),
                      ('z4c_hyperboloidal/ko','0'),('z4c_hyperboloidal/chi_floor','1e-9')]:
        changed=copy.deepcopy(source);changed['parameters'][key]=value
        no('source '+key,lambda changed=changed:reader.check_source(changed))
    hashes={}
    ok('actual qualified timing build',lambda:reader.qualified.check_build(
        Path('build-native/aurora-native-cycle-timing-build-v1/artifacts'),hashes))
    ok('actual two-object Aurora link proof',lambda:reader.check_combined_build(
        Path('build-native/native-two-node-timing-build-v1/artifacts'),hashes))
    # Header-only final fixture. No fabricated payload or solver result.
    final=copy.deepcopy(source);final.update(cycle=reader.LAST,time=expected_time(reader.LAST))
    final['parameters'].update({'time/nlim':str(reader.LAST),'time/tlim':'2','time/ndiag':'1000000000',
        'output1/last_time':str(sum([1/6]*8)),'output2/last_time':str(sum([1/6]*8)),'output2/file_number':'9'})
    ok('synthetic final header only',lambda:reader.check_final(final,source))
    for key,value in [('time/nlim','2555'),('output2/file_number','10'),('fastflow/ah_found_a0_0','0'),
                      ('fastflow/last_a0_0','nan'),('z4c_hyperboloidal/ko','0')]:
        changed=copy.deepcopy(final);changed['parameters'][key]=value
        no('final '+key,lambda changed=changed:reader.check_final(changed,source))
    changed=copy.deepcopy(final);raw=bytearray(changed['fixed']);raw[0]^=1;changed['fixed']=bytes(raw)
    no('initialized mesh header mutation',lambda:reader.check_final(changed,source))
    changed=copy.deepcopy(final);raw=bytearray(changed['fixed']);raw[120]^=1;changed['fixed']=bytes(raw)
    ok('documented unused coarse-header byte excluded',lambda:reader.check_final(changed,source))
    cpu_path=Path('build-native/native-cycle-timing-local-validation-v1/native-cycle-timing.json')
    cpu=reader.read_json(cpu_path)
    ok('actual CPU timing fixture with its own scope',lambda:reader.qualified.timing_windows(cpu,0,2,(2,3,2),'Serial',False))
    # Adapt clocks/counts only to exercise future reader branches; never GPU evidence.
    report=copy.deepcopy(cpu)
    report.update(backend='SYCL',restart=True,initial_cycle=reader.FIRST,final_cycle=reader.LAST,
        complete_steps=70,ranks=24,initial_time=expected_time(reader.FIRST),final_time=expected_time(reader.LAST),
        warmup_steps=10,steps_per_window=20,window_count=3,console_ndiag=1000000000)
    report['windows']=[]
    for index in range(3):
        first=reader.FIRST+10+20*index;elapsed=float(100+index)
        report['windows'].append(dict(index=index,first_cycle=first,last_cycle=first+20,complete_steps=20,
            first_time=expected_time(first),last_time=expected_time(first+20),max_rank_seconds=elapsed,
            seconds_per_complete_step=elapsed/20,bulk_cell_updates=229392*20,bulk_cell_updates_per_second=229392*20/elapsed))
    check=lambda r:reader.qualified.timing_windows(r,reader.FIRST,24,reader.CONFIG,'SYCL',True)
    ok('synthetic future window parser only',lambda:check(report))
    for key,value in [('ranks',96),('complete_steps',280),('timed_output_count',1),('timed_console_count',1),
                      ('profiling_library_loaded',True),('warmup_steps',0)]:
        changed=copy.deepcopy(report);changed[key]=value
        no('window '+key,lambda changed=changed:check(changed))
    for key,value in [('max_rank_seconds',float('nan')),('complete_steps',80),('last_cycle',2555),
                      ('bulk_cell_updates',229392*80)]:
        changed=copy.deepcopy(report);changed['windows'][0][key]=value
        no('window row '+key,lambda changed=changed:check(changed))
    remote=reader.SIDE/reader.CASE/'normal'
    env=dict(OMP_NUM_THREADS='1',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',ZE_ENABLE_PCI_ID_DEVICE_ORDER='1',
        ONEAPI_DEVICE_SELECTOR='level_zero:gpu',MPIR_CVAR_ENABLE_GPU='1',ZE_AFFINITY_MASK=None,
        NEO_CACHE_DIR=str(remote/'cache/neo'),SYCL_CACHE_DIR=str(remote/'cache/sycl'),
        NATIVE_TIMING_WARMUP='10',NATIVE_TIMING_STEPS='20',NATIVE_TIMING_WINDOWS='3')
    env.update({k:None for k in reader.qualified.PROFILE})
    ok('synthetic expected environment',lambda:reader.check_environment(env,remote))
    for key,value in [('NATIVE_TIMING_STEPS','40'),('KOKKOS_TOOLS_LIBS','profile.so'),('ZE_AFFINITY_MASK','0.0')]:
        changed=dict(env);changed[key]=value
        no('environment '+key,lambda changed=changed:reader.check_environment(changed,remote))
    command=['mpiexec','-n','24','-ppn','12',reader.qualified.BINDING,reader.qualified.WRAPPER,
        str(reader.EXECUTABLES['normal'][0]),'-r',reader.SOURCE_REMOTE,'-d',str(remote),
        'time/nlim=2556','time/tlim=2','time/ndiag=1000000000']
    ok('synthetic exact command',lambda:reader.check_command(command,'normal',remote))
    for i,value in [(2,'96'),(7,str(reader.EXECUTABLES['timed-normal'][0])),(9,'other.rst'),(-3,'time/nlim=2555')]:
        changed=list(command);changed[i]=value
        no('command token '+str(i),lambda changed=changed:reader.check_command(changed,'normal',remote))
    init=copy.deepcopy(final);init.update(cycle=reader.FIRST,time=expected_time(reader.FIRST))
    init['parameters']['time/nlim']=str(reader.FIRST)
    ok('synthetic zero-step output header',lambda:reader.check_final(init,source,reader.FIRST))
    command_init=list(command);command_init[-3]='time/nlim=2486';command_init[11]=str(reader.SIDE/reader.CASE/'init-only')
    ok('synthetic zero-step exact command',lambda:reader.check_command(command_init,'init-only',reader.SIDE/reader.CASE/'init-only'))
    tiny=np.zeros((2,3,39));tiny[:,:,0]=1
    ok('synthetic full39-slot initial identity',lambda:reader.initial_payloads(tiny,tiny))
    for field,value in [(30,1e-2),(38,1e-100),(0,float('nan'))]:
        changed=tiny.copy();changed[-1,-1,field]=value
        no('late initial payload field '+str(field),lambda changed=changed:reader.initial_payloads(changed,tiny))
    no('initial truncated records',lambda:reader.initial_payloads(tiny[:,:,:38],tiny[:,:,:38]))
    # End-to-end failure path on absent collection, after real source/build checks:
    # an operational failure cannot accidentally retain accepted timing values.
    missing=args.output/'absent-collection';failure=args.output/'withheld-on-missing-collection'
    completed=subprocess.run([sys.executable,str(Path(reader.__file__)),
        '--collection',str(missing),'--source-checkpoint',str(source_path),'--admission',str(admission_path),
        '--independent-review',str(review_path),'--timing-build-artifacts','build-native/aurora-native-cycle-timing-build-v1/artifacts',
        '--combined-build-artifacts','build-native/native-two-node-timing-build-v1/artifacts','--output',str(failure)],
        capture_output=True,text=True)
    report=reader.read_json(failure/'results.json')
    assert completed.returncode==2 and report['passed'] is False and report['timings_accepted'] is False
    assert report['accepted_timings']=={} and report['timing_summary'] is None
    checks.append(dict(name='end-to-end missing collection withholds all timing values',passed=True,expected='reject'))
    result=dict(schema='native-n12-two-node-timing-reader-controls-v1',passed=True,checks=checks,
        scope='Offline actual2486 and qualified build proofs; existing CPU timing fixture; explicitly synthetic metadata/window/env/command negative controls. No GPU run or timing admission.',
        real_fixture_sha256={str(p):sha256(p) for p in (source_path,admission_path,review_path,cpu_path)},
        source_sha256={str(p):sha256(p) for p in (Path(__file__),Path(reader.__file__))})
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks))))

if __name__=='__main__':main()
