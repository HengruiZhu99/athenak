#!/usr/bin/env python3
"""Operational guards for the isolated2486 native controls; never submits jobs.

Full state/diagnostic equivalence and timing acceptance belong to the separate
native_n12_two_node_timing_validate.py reader. This producer uses only stdlib.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
WORK = ROOT/'validation/perf-two-node-v1/native-run'
CASE = ROOT/'runs/native-n12-timing-two-v1'
SOURCE_PATH = ROOT/'runs/native-n12-common-07/rst/native_n12_5M.00007.rst'
SOURCE = 'dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec'
ADMISSION = 'babb035acdcb505d2f8407815981731168ae2c75c3217dcc43ba2ce56c363bd9'
REVIEW = '6322dcb077c8ac51bc105791d024cd93686994cbc89ffaf90b5819a9f3b99d51'
NORMAL = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
TIMED = 'e47888006e1221d5e14bce734a97d77a45126b271f7ef39c2ceb9e9b31c0f6cf'
COMBINED = '222893425aae3419e55d72b7db78d2fdc0253ecd7676182fd5a52db8d9a7db45'
TIMING_PROOF = '89265de52cfec416139ddf4188618b10b91b4e130c3fa844ffeed473d0d492ea'
COMBINED_PROOF = 'b3076d063321f8f939f654cfe0ffae736832f9eba1743fde4934b7a1047bbcfd'
SOURCES = '4706aae696919253201da4cac9dd14bff52743ee8826dc93660a382f255ec9a5'
OBJECTS = '462da52c2e152e9c94818a534314aec72f2a827f30decdb2d23a3559e6954547'
WRAPPER = '/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh'
WRAPPER_SHA = '4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd'
BINDING = '--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
DT = .00046944203735617445
PROFILE = ('KOKKOS_TOOLS_LIBS','KOKKOS_PROFILE_LIBRARY','KOKKOS_TOOLS_ARGS',
           'SYCL_PI_TRACE','SYCL_UR_TRACE','ZE_ENABLE_TRACING_LAYER',
           'UNITRACE_ENABLE','PTI_ENABLE','VTUNE_PROFILER_ENABLE')
ENVIRONMENT = dict(OMP_NUM_THREADS='1',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
    ZE_ENABLE_PCI_ID_DEVICE_ORDER='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
    MPIR_CVAR_ENABLE_GPU='1',NATIVE_TIMING_WARMUP='10',NATIVE_TIMING_STEPS='20',
    NATIVE_TIMING_WINDOWS='3')
EXECUTABLES = {
    'init-only':(ROOT/'build-sycl/src/athena',NORMAL),
    'normal':(ROOT/'build-sycl/src/athena',NORMAL),
    'timed-normal':(ROOT/'validation/cycle-timing/build/athena-cycle-timing',TIMED),
    'timed-observer':(ROOT/'validation/perf-two-node-v1/native-timed-observer/build/athena-timed-observer',COMBINED)}
VERIFY = {
    'timing':(ROOT/'validation/cycle-timing', 'native_cycle_timing.py', TIMING_PROOF,
              'Verified invertible original-driver timing and unchanged native source/object/executable inputs\n'),
    'timed-observer':(ROOT/'validation/perf-two-node-v1/native-timed-observer',
              'native_timed_observer.py', COMBINED_PROOF,
              'Verified invertible original-driver timing and unchanged native source/object/executable inputs\n'
              'Verified invertible original-main observer and unchanged native source/object/executable inputs\n'
              'Verified unchanged native timed observer composition\n')}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def expected_time(cycle):
    time=0.
    for _ in range(cycle):time+=DT
    return time


def checkpoint_metadata(path, cycle=2486):
    """Check shared-file structure/clock; no field data are interpreted here."""
    with Path(path).open('rb') as f:
        text=bytearray()
        while True:
            line=f.readline(262144);text.extend(line)
            require(line and len(text)<8*1024*1024,'invalid parameter header')
            if line==b'<par_end>\n':break
        size=struct.calcsize('<ii9d38i2di');fixed=f.read(size)
        require(len(fixed)==size,'truncated fixed header')
        blocks,level=struct.unpack_from('<ii',fixed)
        time,dt,actual=struct.unpack_from('<ddi',fixed,size-20)
        require((blocks,level,actual,time,dt)==(106,4,cycle,expected_time(cycle),DT),
                'wrong N12 checkpoint clock/shape')
        count,=struct.unpack('<Q',f.read(8));require(0<count<1024*1024,'bad atlas length')
        require(len(f.read(count))==count and len(f.read(106*20))==106*20,'truncated geometry')
        require(all(x==1. for x in struct.unpack('<106f',f.read(424))),'unequal block weights')
        meta=struct.unpack('<8Q',f.read(64))
        require(meta[:2]==(0x4859504552525354,1) and 0<meta[2]<16384 and
                meta[3:]==((18**3+144)*39,24,0,4,0),'unexpected shared restart metadata')
        require(len(f.read(meta[2]))==meta[2],'truncated physics identity')
        stride,=struct.unpack('<Q',f.read(8))
        require(stride==meta[3]*8 and Path(path).stat().st_size==f.tell()+106*stride,
                'truncated/extended payload')
    params={};block=None
    for line in text.decode().splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('<'):block=line[1:line.index('>')]
        elif '=' in line:
            key,value=line.split('=',1);params[block+'/'+key.strip()]=value.strip()
    require(params['job/basename']=='native_n12_5M' and params['time/integrator']=='rk4_classical'
            and params['time/evolution']=='dynamic' and params['mesh_refinement/refinement']=='static'
            and float(params['time/dt'])==DT and float(params['z4c_hyperboloidal/ko'])==.1
            and float(params['z4c_hyperboloidal/chi_floor'])==1e-10,'numerical policy differs')
    return dict(cycle=cycle,time=time,dt=dt,blocks=blocks,writer_ranks=24,parameters=params)


def admission(path):
    path=Path(path).resolve();data=read_json(path)
    require(data['schema']=='native-n12-two-node-timing-admission-v1','wrong descriptor')
    files={}
    for key,digest in [('checkpoint',SOURCE),('science_admission',ADMISSION),('independent_review',REVIEW)]:
        p=Path(data[key]['path']);p=p if p.is_absolute() else path.parent/p
        require(data[key]['sha256']==digest and sha(p)==digest,'changed admission: '+key)
        files[key]=p.resolve()
    require(files['checkpoint']==SOURCE_PATH,'not the admitted remote checkpoint')
    science=read_json(files['science_admission']);review=read_json(files['independent_review'])
    require(science['passed'] is True and science['end_cycle']==2486 and
            science['checkpoint_sha256']==SOURCE and science['comparisons']['2486']['passed'] is True,
            'missing completed2486 field admission')
    require(review['passed'] is True and review['end_cycle']==2486 and
            review['science_admission_sha256']==ADMISSION and review['checkpoint_sha256']==SOURCE and
            review['stage_rows']==9944 and review['exact_stage_clocks'] is True and
            review['history_completed_prefix_passed'] is True and
            review['history_times']==[0,356,711,1066,1421,1776,2131,2486] and
            review['harmonic_coefficients_each_time']==169,'incomplete independent2486 review')
    source=checkpoint_metadata(files['checkpoint']);p=source['parameters'];clock=sum([1/6]*7)
    require(p['output1/file_type']=='hst' and p['output2/file_type']=='rst' and
            p['output2/single_file_per_rank']=='false' and int(p['output2/file_number'])==8,
            'source output mode/counter differs')
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    for block in ('output1','output2'):
        require(float(p[block+'/dt'])==1/6 and float(p[block+'/last_time'])==clock and
                int(p.get(block+'/dcycle','0'))<=0,'source output cadence differs')
        require(all(f32(expected_time(c))<f32(clock+1/6) for c in range(2487,2557)),
                'scheduled output inside timed continuation')
    require(math.isfinite(float(p['fastflow/last_a0_0'])) and float(p['fastflow/last_a0_0'])>0
            and p['fastflow/ah_found_a0_0'] in ('1','true'),'missing horizon seed')
    return dict(source_checkpoint_remote=str(SOURCE_PATH),source_checkpoint_sha256=SOURCE,
                science_admission_sha256=ADMISSION,independent_review_sha256=REVIEW)


def environment(directory):
    directory=Path(directory)
    expected=dict(ENVIRONMENT,ZE_AFFINITY_MASK=None,NEO_CACHE_DIR=str(directory/'cache/neo'),
                  SYCL_CACHE_DIR=str(directory/'cache/sycl'))
    expected.update({k:None for k in PROFILE})
    actual={key:os.environ.get(key) for key in expected}
    require(actual==expected,'profiling/affinity/execution environment differs')
    return actual


def binding(path):
    seen=set()
    for line in Path(path).read_text().splitlines():
        m=re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
        require(m is not None,'invalid binding record')
        host,rank,tile=m.groups();rank=int(rank)
        require(0<=rank<12 and tile==f'{rank//2}.{rank%2}' and (host,rank) not in seen,'wrong tile')
        seen.add((host,rank))
    require(len(seen)==24 and len({h for h,r in seen})==2,'incomplete two-node binding')


def terminal(text, job):
    m=re.search(r'^Job Id:\s*(\S+)\s*$',text,re.M)
    require(m and m[1].split('.')[0]==str(job).split('.')[0],'terminal job mismatch')
    fields=dict(re.findall(r'^\s+([A-Za-z_.]+)\s*=\s*(\S+)\s*$',text,re.M))
    require(all(fields.get(k)==v for k,v in [('job_state','F'),('Exit_status','0'),
        ('Job_Name','native-n12-timing-two'),('Resource_List.nodect','2')]),'not successful expected job')


def guard(args):
    report=admission(args.admission)
    require(all(k not in os.environ for k in PROFILE),'profiling environment requested')
    require(sha(WRAPPER)==WRAPPER_SHA,'GPU wrapper changed')
    source_file=ROOT/'build-sycl/source-sha256.json';object_file=ROOT/'build-sycl/runtime-object-sha256.json'
    require(sha(source_file)==SOURCES and sha(object_file)==OBJECTS,'frozen manifests changed')
    sources=read_json(source_file);objects=read_json(object_file)['object_sha256']
    require(len(sources)==1972 and len(objects)==228,'incomplete manifests')
    for rel,digest in sources.items():require(sha(ROOT/'source'/rel)==digest,'source changed: '+rel)
    for rel,digest in objects.items():require(sha(ROOT/'build-sycl'/rel)==digest,'object changed: '+rel)
    for executable,digest in EXECUTABLES.values():require(sha(executable)==digest,'executable changed')
    for name,(work,builder,digest,text) in VERIFY.items():
        require(sha(work/'build/provenance.json')==digest,'instrument build proof changed')
        log=args.directory/f'{name}-verify-{args.suffix}.txt'
        with log.open('w') as output:
            code=subprocess.run(['/usr/bin/python3.11',str(work/'package'/builder),'verify','--output',
                str(work/'build')],stdout=output,stderr=subprocess.STDOUT).returncode
        require(code==0 and log.read_text()==text,'instrument verifier failed: '+name)
        report[name.replace('-','_')+'_verify_exit_code']=code
        report[name.replace('-','_')+'_verify_log_sha256']=sha(log)
    report.update(schema='native-n12-two-node-timing-guard-v1',passed=True,
        baseline_executable_sha256=NORMAL,timing_executable_sha256=TIMED,
        timed_observer_executable_sha256=COMBINED,timing_build_provenance_sha256=TIMING_PROOF,
        timed_observer_build_provenance_sha256=COMBINED_PROOF,source_manifest_sha256=SOURCES,
        object_manifest_sha256=OBJECTS,verified_source_files=1972,verified_objects=228,
        gpu_wrapper_sha256=WRAPPER_SHA)
    write_json(args.directory/f'guard-{args.suffix}.json',report)


def completed_case(directory, kind):
    directory=Path(directory);cycle=2486 if kind=='init-only' else 2556
    command=['mpiexec','-n','24','-ppn','12',BINDING,WRAPPER,str(EXECUTABLES[kind][0]),
             '-r',str(SOURCE_PATH),'-d',str(CASE/kind),f'time/nlim={cycle}','time/tlim=2','time/ndiag=1000000000']
    require(shlex.split((directory/'command.txt').read_text())==command,'actual command differs')
    require((directory/'exit-code.txt').read_text().strip()=='0','case returned failure')
    log=(directory/'run.log').read_text()
    require('Terminating on cycle limit' in log and re.search(r'cycle='+str(cycle)+r'(?:\s|$)',log),
            'case did not reach endpoint')
    require(sorted(p.name for p in (directory/'rst').glob('*.rst'))==['native_n12_5M.00008.rst'],
            'missing/extra checkpoint')
    info=checkpoint_metadata(directory/'rst/native_n12_5M.00008.rst',cycle);p=info['parameters']
    require(int(p['output2/file_number'])==9 and all(float(p[b+'/last_time'])==sum([1/6]*8)
            for b in ('output1','output2')),'final output counters differ')
    expected=dict(ENVIRONMENT,ZE_AFFINITY_MASK=None,NEO_CACHE_DIR=str(CASE/kind/'cache/neo'),
                  SYCL_CACHE_DIR=str(CASE/kind/'cache/sycl'));expected.update({k:None for k in PROFILE})
    require(read_json(directory/'environment.json')==expected,'case environment differs')
    for name in ('native_n12_5M.z4c_hyperboloidal.hst','native_n12_5M.horizon_summary_0.txt',
                 'native_n12_5M.horizon_shape_0.txt'):
        require((directory/name).is_file() and (directory/name).stat().st_size>0,'missing diagnostic: '+name)
    if kind.startswith('timed-'):
        timing=read_json(directory/'native-cycle-timing.json')
        require(timing['schema']=='native-cycle-timing-v1' and timing['passed'] is True and
                (timing['ranks'],timing['initial_cycle'],timing['final_cycle'],timing['complete_steps'],
                 timing['warmup_steps'],timing['steps_per_window'],timing['window_count'])==
                (24,2486,2556,70,10,20,3) and len(timing['windows'])==3,'incomplete timing windows')
    else:require(not (directory/'native-cycle-timing.json').exists(),'unrequested timing file')
    if kind=='timed-observer':
        observed=read_json(directory/'native-step-observer.json')
        require(observed['passed'] is True and (observed['ranks'],observed['initial_cycle'],
                observed['final_cycle'],observed['stage_rows'])==(24,2486,2556,280),
                'incomplete observer rows')
        require(len((directory/'native-stages.csv').read_text().splitlines())==281,'incomplete stage CSV')
    else:require(not (directory/'native-stages.csv').exists() and
                 not (directory/'native-step-observer.json').exists(),'unrequested observer files')
    return dict(schema='native-n12-two-node-case-operational-v1',passed=True,kind=kind,
                final_cycle=cycle,final_checkpoint_sha256=sha(directory/'rst/native_n12_5M.00008.rst'),
                scientific_comparison_pending=True,performance_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('guard');p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--suffix',choices=('before','after'),required=True)
    p=sub.add_parser('environment');p.add_argument('directory',type=Path)
    p=sub.add_parser('binding');p.add_argument('path',type=Path)
    p=sub.add_parser('completed');p.add_argument('directory',type=Path);p.add_argument('kind',choices=tuple(EXECUTABLES))
    args=parser.parse_args()
    if args.mode=='guard':guard(args)
    elif args.mode=='environment':write_json(args.directory/'environment.json',environment(args.directory))
    elif args.mode=='binding':binding(args.path)
    else:write_json(args.directory/'operational.json',completed_case(args.directory,args.kind))


if __name__=='__main__':main()
