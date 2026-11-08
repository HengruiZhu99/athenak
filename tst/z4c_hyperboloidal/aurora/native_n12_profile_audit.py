#!/usr/bin/env python3
"""Remote stdlib complete-state/profile qualification; raw traces remain remote."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from native_two_node_timing_audit import (ROOT,SOURCE,DT,NORMAL,BINDING,WRAPPER,VERIFY,
    checkpoint_metadata,expected_time,read_json,require,sha,write_json)
from native_n12_profile_rank import SELECTED,TOOL,TOOL_SHA,OPTIONS,CACHE_ENV

WORK=ROOT/'validation/perf-two-node-v1/profile-cache128-v1'
CASE=ROOT/'runs/native-n12-profile-cache128-v1'
CORE={
 'libmpi.so':'aae85d14c5687653512009291dc5683da1c023495d803e2f798c5d1e0d78a220',
 'libsycl.so':'0336997fdfed9b2e6385e9f1cea2395eb5e130d3e5e9c943df5b0c10c1b5e57f',
 'libstdc++.so':'afb5f1afa1a3e004cca1a23b89271f12c643ae763bd258a9aab4013f7518451f',
 'libgcc_s.so':'992a135c8b0328549dbf25ebc3d16859ca11a587f59254e53ca42aa932c362c7'}
TOOL_LIBRARY='ca4ff7fcbb622c99d44084a1cf2e9366206936162ef9a93273fcef5b01c2462d'


def checkpoint(path,cycle=2490):
    info=checkpoint_metadata(path,cycle)
    with Path(path).open('rb') as f:
        while f.readline()!=b'<par_end>\n':pass  # bounded parser above already validated it
        fixed=f.read(struct.calcsize('<ii9d38i2di'))
        n,=struct.unpack('<Q',f.read(8));atlas=f.read(n)
        topology=f.read(2120);costs=f.read(424);meta=f.read(64)
        n=struct.unpack('<8Q',meta)[2];identity=f.read(n);stride=f.read(8)
        offset=f.tell()
    return dict(**info,path=str(path),fixed=fixed,atlas=atlas,topology=topology,costs=costs,
                metadata=meta,identity=identity,stride=stride,offset=offset)


def compare_checkpoints(a,b):
    for key in ('parameters','atlas','topology','costs','metadata','identity','stride'):
        require(a[key]==b[key],'profile checkpoint differs: '+key)
    require(a['fixed'][:120]==b['fixed'][:120] and a['fixed'][156:]==b['fixed'][156:],
            'profile initialized mesh header differs')
    count=0;digest=hashlib.sha256()
    with open(a['path'],'rb') as x,open(b['path'],'rb') as y:
        x.seek(a['offset']);y.seek(b['offset'])
        while True:
            left,right=x.read(1024*1024),y.read(1024*1024)
            require(left==right,'full31+8 profile payload differs at byte chunk '+str(count*8))
            if not left:break
            require(len(left)%8==0 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<d',left)),
                    'nonfinite/truncated checkpoint payload')
            count+=len(left)//8;digest.update(left)
    require(count==106*(18**3+144)*39,'payload coverage differs')
    return dict(bitwise_equal=True,padded31_plus8_scalar_values=count,payload_sha256=digest.hexdigest(),
                excluded_unused_coarse_indices=[list(struct.unpack('<9i',v['fixed'][120:156])) for v in (a,b)])


def diagnostics(directory):
    stem='native_n12_5M';names=[stem+'.z4c_hyperboloidal.hst',stem+'.horizon_summary_0.txt',stem+'.horizon_shape_0.txt']
    actual={p.name for pattern in ('*.hst','*.horizon*.txt') for p in directory.glob(pattern)}
    require(actual==set(names),'missing/extra ordinary diagnostic outputs')
    tables=[]
    for name in names[:2]:
        rows=[[float(x) for x in line.split()] for line in (directory/name).read_text().splitlines()
              if line.strip() and not line.startswith('#')]
        require(len(rows)==1 and len(rows[0])==12 and all(math.isfinite(x) for x in rows[0]),
                'incomplete/nonfinite final diagnostics');tables.append(rows[0])
    h,a=tables
    require(h[:2]==[expected_time(2490),DT] and 0<h[7]<=h[2] and
            all(h[i]>=0 for i in (3,4,5,6,8,9,10,11)),'history clock/norms invalid')
    require(a[:2]==[2490,float(format(expected_time(2490),'.6g'))] and
            all(a[i]>0 for i in (2,7,10,11)) and a[8]>=0,'horizon clock/properties invalid')
    shape=[x.strip() for x in (directory/names[2]).read_text().splitlines() if x.strip()]
    require(len(shape)==2 and re.fullmatch(r'# iter = 2490, Time = \S+',shape[0]),'shape frame missing')
    require(float(shape[0].split('=')[-1])==float(format(expected_time(2490),'.6g')),'shape clock differs')
    values=[float(x) for x in shape[1].split()]
    require(len(values)==169 and all(math.isfinite(x) for x in values),'incomplete shape coefficients')
    return {name:sha(directory/name) for name in names}


def runtime(directory,mode,rank):
    folder=directory/mode/'ranks'/f'rank{rank}'
    launch=read_json(folder/'launch.json');r=read_json(folder/'runtime.json')
    require((r['schema'],r['passed'],r['rank'],r['mode'],r['exit_code'])==
            ('native-n12-profile-runtime-v1',True,rank,mode,0),'runtime capture failed')
    require(len(r['solver_processes'])==1 and r['samples']>0,'missing/multiple observed solver process')
    process=next(iter(r['solver_processes'].values()))
    require(process['executable']==str(ROOT/'build-sycl/src/athena'),'observed another executable')
    require(process['environment']['ZE_AFFINITY_MASK']==launch['environment']['ZE_AFFINITY_MASK'] and
            process['environment']['EnableImplicitScaling']=='0','observed solver binding differs')
    require({k:process['environment'].get(k) for k in CACHE_ENV}==CACHE_ENV,
            'observed solver cache128 environment differs')
    files=process['mapped_files'];selected={}
    for prefix,digest in CORE.items():
        matches=[(name,data) for name,data in files.items() if Path(name).name.startswith(prefix)]
        require(len(matches)==1 and matches[0][1]['sha256']==digest,'actual loaded baseline library differs: '+prefix)
        selected[prefix]=dict(path=matches[0][0],**matches[0][1])
    require(files[str(ROOT/'build-sycl/src/athena')]['sha256']==NORMAL,'actual loaded f3 differs')
    tool=[x for name,x in files.items() if Path(name).name.startswith('libunitrace_tool.so')]
    require((len(tool)==1 and tool[0]['sha256']==TOOL_LIBRARY) if mode=='profiled' else not tool,
            'profiling tool injection/identity differs')
    # The monitor used a filename substring, which also captured the transient
    # dynamic-loader lookup cache. It is data, not a loaded executable image.
    # Retain the compute-node observation: its bytes/magic were not captured,
    # so a later login-node cache cannot establish its byte identity or format.
    cache={name:data for name,data in files.items() if name=='/etc/ld.so.cache'}
    return dict(pid=int(next(iter(r['solver_processes']))),core_libraries=selected,
                loaded_images={name:d['sha256'] for name,d in files.items() if name not in cache},
                non_executable_loader_cache_mappings=cache,
                cache_classification_scope='Only canonical /etc/ld.so.cache excluded by loader-data role; captured inode/size/hash retained, ELF magic not captured. Every other observed image is still compared.',
                samples=r['samples'],observed_cache_environment=CACHE_ENV)


def trace_summary(folder,shared_log=None,text_report=None):
    """Conservative schema inventory. Raw timestamps from different ranks are never merged."""
    paths=[p for p in folder.glob('*.json') if p.name not in ('launch.json','runtime.json')]
    reports=[]
    for path in sorted(paths):
        require(path.stat().st_size<=512*1024*1024,'trace exceeds bounded512MiB parser; raw file retained')
        raw=read_json(path);events=raw.get('traceEvents',[]) if isinstance(raw,dict) else raw
        require(isinstance(events,list),'unrecognized trace container')
        categories=collections.Counter();types=collections.Counter();by_name={};examples=[]
        for event in events:
            require(isinstance(event,dict),'malformed trace event')
            categories[str(event.get('cat',''))]+=1;types[str(event.get('ph',''))]+=1
            if event.get('ph')!='X':continue
            duration=event.get('dur');timestamp=event.get('ts')
            require(isinstance(duration,(float,int)) and isinstance(timestamp,(float,int)) and
                    math.isfinite(duration) and math.isfinite(timestamp) and duration>=0,'invalid trace interval')
            key=(str(event.get('cat','')),str(event.get('name','')))
            row=by_name.setdefault(key,dict(category=key[0],name=key[1],calls=0,total_duration=0.,maximum_duration=0.,
                first_timestamp=timestamp,last_timestamp=timestamp,args_examples=[]))
            row['calls']+=1;row['total_duration']+=duration;row['maximum_duration']=max(row['maximum_duration'],duration)
            row['first_timestamp']=min(row['first_timestamp'],timestamp);row['last_timestamp']=max(row['last_timestamp'],timestamp+duration)
            args=event.get('args',{})
            if args not in row['args_examples'] and len(row['args_examples'])<2:row['args_examples'].append(args)
        ordered=sorted(by_name.values(),key=lambda x:x['total_duration'],reverse=True)
        reports.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,event_count=len(events),
            category_counts=dict(categories),phase_counts=dict(types),complete_event_groups=ordered[:120],
            omitted_groups=max(0,len(ordered)-120),
            interval_scope='Raw Chrome timestamp/duration units; same-file intervals only. Categories and kernel identities require actual-source review. Overlapping totals are not wall time.'))
    text=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()
          if p.is_file() and p.suffix!='.json' and p.name not in ('launch.json','runtime.json')]
    locator=None
    if text_report is not None:
        require(shared_log is not None and
                f'[INFO] Log is stored in {text_report.name}' in shared_log.read_text(),
                'shared log does not identify selected-rank text report')
        body=text_report.read_text()
        require('=== API Timing Summary ===' in body and 'Total Execution Time (ns)' in body,
                'missing actual unitrace API timing summary')
        text.append(dict(path=str(text_report),bytes=text_report.stat().st_size,sha256=sha(text_report),
                         attribution='Actual solver PID and rank in filename; unitrace launch-working-directory report'))
        locator=dict(path=str(shared_log),bytes=shared_log.stat().st_size,sha256=sha(shared_log),
                     attribution='Shared stdout/stderr locates reports; not itself a rank-specific timing table')
    require(reports and any(r['complete_event_groups'] for r in reports),'no actual complete Chrome trace intervals')
    require(text,'no profiler host/device summary output')
    return dict(traces=reports,other_profile_artifacts=text,shared_output_locator=locator,
        coverage_scope='Raw event/schema inventory only; does not itself prove MPI progress, cross-rank clocks, critical rank, per-stage boundaries or Kokkos runtime-label mapping')


def audit(args,result):
    d=args.directory;require(sha(args.source)==SOURCE,'source changed')
    source=checkpoint(args.source,2486)
    before,after=[read_json(d/f'guard-{s}.json') for s in ('before','after')]
    require(before==after and before['passed'] is True and before['baseline_executable_sha256']==NORMAL,
            'baseline guards differ/fail')
    for suffix in ('before','after'):
        for name,(work,builder,digest,text) in VERIFY.items():
            p=d/f'{name}-verify-{suffix}.txt'
            require(p.read_text()==text and sha(p)==before[name.replace('-','_')+'_verify_log_sha256'],'guard log mismatch')
    require(sha(TOOL)==TOOL_SHA,'unitrace executable changed')
    states={};outputs={};runtimes={}
    for mode in ('normal','profiled'):
        folder=d/mode
        import shlex
        expected_outer=['mpiexec','-n','24','-ppn','12',BINDING,WRAPPER,'/usr/bin/python3.11',
            str(WORK/'native_n12_profile_rank.py'),
            '--mode',mode,'--directory',str(CASE/mode)]
        require(shlex.split((folder/'command.txt').read_text())==expected_outer,'MPI launch command differs')
        require((folder/'exit-code.txt').read_text().strip()=='0' and
                'Terminating on cycle limit' in (folder/'run.log').read_text(),'solver/profiler did not complete')
        require([p.name for p in (folder/'rst').glob('*.rst')]==['native_n12_5M.00008.rst'],'checkpoint set differs')
        states[mode]=checkpoint(folder/'rst/native_n12_5M.00008.rst');outputs[mode]=diagnostics(folder)
        p=states[mode]['parameters'];original=source['parameters']
        mutable={'time/nlim','time/tlim','time/ndiag','output1/last_time','output2/last_time',
                 'output2/file_number','fastflow/last_a0_0','fastflow/ah_found_a0_0'}
        require(p.keys()==original.keys() and all(p[k]==original[k] for k in p if k not in mutable),
                'numerical/output policy differs from source')
        for key in ('atlas','topology','costs','metadata','identity','stride'):
            require(states[mode][key]==source[key],'source mesh identity differs: '+key)
        require(states[mode]['fixed'][:120]==source['fixed'][:120] and
                states[mode]['fixed'][156:-20]==source['fixed'][156:-20],'source mesh header differs')
        require((int(p['time/nlim']),float(p['time/tlim']),int(p['time/ndiag']),int(p['output2/file_number']))
                ==(2490,2.,1000000000,9),'endpoint controls/counters differ')
        require(all(float(p[b+'/last_time'])==sum([1/6]*8) for b in ('output1','output2')),'output clocks differ')
        owners=set()
        for rank in range(24):
            launch=read_json(folder/'ranks'/f'rank{rank}'/'launch.json')
            local=launch['local_rank'];env=launch['environment'];host=launch['host']
            require(launch['rank']==rank and launch['mode']==mode and launch['selected']==(rank in SELECTED) and
                    launch['profiled']==(mode=='profiled' and rank in SELECTED),'rank selection differs')
            require(0<=local<12 and env['ZE_AFFINITY_MASK']==f'{local//2}.{local%2}' and
                    env['EnableImplicitScaling']=='0' and (host,local) not in owners,'actual tile binding differs')
            require({k:env.get(k) for k in CACHE_ENV}==CACHE_ENV,'rank cache128 environment differs')
            owners.add((host,local))
            expected=[str(ROOT/'build-sycl/src/athena'),'-r',str(args.source),'-d',str(CASE/mode),
                      'time/nlim=2490','time/tlim=2','time/ndiag=1000000000']
            if launch['profiled']:expected=[str(TOOL),*OPTIONS,'--output-dir-path',str(CASE/mode/'ranks'/f'rank{rank}'),
                                          '--output','unitrace.txt',*expected]
            require(launch['command']==expected and launch['baseline_executable_sha256']==NORMAL,'rank command differs')
        require(len(owners)==24 and len({h for h,r in owners})==2,'wrong actual nodes/ranks')
        runtimes[mode]={str(r):runtime(d,mode,r) for r in SELECTED}
    result['state_equivalence']=compare_checkpoints(states['normal'],states['profiled'])
    require(outputs['normal']==outputs['profiled'],'ordinary diagnostics differ')
    result['diagnostics_exact']=outputs['normal'];result['runtime_identities']=runtimes
    for rank in map(str,SELECTED):
        for name,digest in runtimes['normal'][rank]['loaded_images'].items():
            require(runtimes['profiled'][rank]['loaded_images'].get(name)==digest,
                    'profile replaces/removes observed normal image: '+name)
    result['scientific_qualification_passed']=True
    work=WORK
    texts={str(r):work/f'unitrace.{runtimes["profiled"][str(r)]["pid"]}.{r}.txt' for r in SELECTED}
    result['selected_rank_trace_inventory']={str(r):trace_summary(d/'profiled'/'ranks'/f'rank{r}',
        d/'profiled'/'run.log',texts[str(r)]) for r in SELECTED}
    result['external_profile_artifacts']={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in texts.values()}
    result['passed']=True
    result['artifact_manifest']={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in d.rglob('*') if p.is_file() and 'cache' not in p.relative_to(d).parts and p!=args.output}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-failure',type=Path)
    p.add_argument('--terminal',type=Path)
    args=p.parse_args();require(not args.output.exists(),'qualification output already exists; preserve prior evidence')
    result=dict(schema='native-n12-selected-profile-qualification-v1',passed=False,
        scientific_qualification_passed=False,performance_measurement=False,trace_interpretation_pending=True,selected_ranks=list(SELECTED),
        cycles=[2486,2490],cache_environment=CACHE_ENV,
        cache_observation_scope='All24 launch environments and actual selected-rank8/23 process environments in both cases; no MPI CVAR readback.',scope='Four complete cycles, selected-rank tracing; no throughput or whole-job critical-path claim')
    try:
        if args.previous_failure is not None or args.terminal is not None:
            require(args.previous_failure is not None and args.terminal is not None,
                    'reanalysis requires original failure and actual terminal record')
            require(not args.output.exists() and args.output.name=='qualification-v2.json' and
                    args.previous_failure.resolve()==(args.directory/'qualification.json').resolve(),
                    'reanalysis must preserve original qualification and use new v2 output')
            old=read_json(args.previous_failure)
            require(old['passed'] is False and old.get('failure')==
                    'ValueError: profile replaces/removes observed normal image: /etc/ld.so.cache',
                    'unexpected original audit failure')
            terminal=args.terminal.read_text();job=re.search(r'^Job Id:\s*(\S+)',terminal,re.M)
            require(job and job[1].split('.')[0]==(args.directory/'job-id.txt').read_text().strip().split('.')[0],
                    'reanalysis terminal job differs')
            for key,value in [('job_state','F'),('Exit_status','2'),('Job_Name','native-n12-profile-two'),('Resource_List.nodect','2')]:
                found=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',terminal,re.M)
                require(found and found[1]==value,'reanalysis terminal field differs: '+key)
            result['reanalysis']=dict(previous_report=str(args.previous_failure),previous_report_sha256=sha(args.previous_failure),
                previous_failure=old['failure'],terminal_record=str(args.terminal),terminal_sha256=sha(args.terminal),
                job_id=job[1].split('.')[0],job_state='F',batch_exit_status=2,solver_exit_statuses={'normal':0,'profiled':0},
                scope='Offline parser correction only; original F/2 audit failure and all raw observations retained. Both solver exits independently checked below. No solver/profiler rerun.')
        audit(args,result)
    except Exception as e:result.update(passed=False,failure=type(e).__name__+': '+str(e))
    result['source_sha256']={name:sha(Path(__file__).with_name(name)) for name in
        ('native_n12_profile_audit.py','native_n12_profile_rank.py','native_two_node_timing_audit.py')}
    write_json(args.output,result);print(json.dumps({k:result[k] for k in ('passed','scientific_qualification_passed','failure') if k in result}))
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
