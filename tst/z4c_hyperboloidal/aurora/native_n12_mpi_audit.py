#!/usr/bin/env python3
"""Bounded APS MPI-only control audit; no solver execution or throughput claim."""
import argparse
import csv
import io
import math
from pathlib import Path
import re
import shlex

from native_n12_profile_audit import checkpoint,compare_checkpoints,diagnostics,CORE
from native_n12_mpi_rank import TOOL,TOOL_SHA,OPTIONS,SELECTED,BASELINE_PRELOAD
from native_two_node_timing_audit import (ROOT,SOURCE,SOURCE_PATH,NORMAL,BINDING,WRAPPER,
    VERIFY,read_json,require,sha,write_json)

CASE=ROOT/'runs/native-n12-mpi-two-v2'
WORK=ROOT/'validation/perf-two-node-v1/mpi-profile-v2'
REPORT=Path('/opt/aurora/26.26.0/oneapi/vtune/latest/bin64/aps-report')
REPORT_SHA='f1ed1b1ea6e45150a2b91b24fe662f05e10c90caace356e19b5f2a127e217b3a'
MPS_SHA='b955adb9e0c665834b80ca676f8d96ec9d4c17b95fc424fb2708cb9484c235ba'
COLLECTOR_LIBS={'libmps.so':MPS_SHA,
    'libamplxe_libarchive_3.8.so':'a23854a9f6572e6131e255beb71fecde3fb3a0515e45630fe0614f9243d02358',
    'libssl.so.3.5.0':'6b0bd0ed1c1b8c275cd8b590c6106821a7cd99fae109da4b90f7b9187703563c',
    'libcrypto.so.3.5.0':'e412eb49711f57086eb9ded394804e7ba05d4acbbdccd90a099733d08d0484da'}
REPORT_OPTIONS={'summary':['-s'],'functions':['-f','-D','-C','-T','0'],
    'per-rank':['-t','-D','-T','0'],'communicators':['-l'],
    'messages':['-m','-D','-C','-T','0'],'topology':['--node-topology']}


def runtime(directory,mode,rank):
    folder=directory/mode/'ranks'/f'rank{rank}';r=read_json(folder/'runtime.json')
    launch=read_json(folder/'launch.json')
    require((r['schema'],r['passed'],r['rank'],r['mode'],r['exit_code'])==
        ('native-n12-profile-runtime-v1',True,rank,mode,0),'runtime capture failed')
    require(len(r['solver_processes'])==1 and r['samples']>0,'missing/multiple f3 processes')
    pid,process=next(iter(r['solver_processes'].items()));files=process['mapped_files']
    require(process['executable']==str(ROOT/'build-sycl/src/athena') and
        files[process['executable']]['sha256']==NORMAL,'actual f3 differs')
    require(process['environment']['ZE_AFFINITY_MASK']==launch['environment']['ZE_AFFINITY_MASK'] and
        process['environment']['EnableImplicitScaling']=='0','actual solver binding differs')
    require(process['environment']['LD_LIBRARY_PATH']==launch['environment']['LD_LIBRARY_PATH'],
            'APS changed baseline library search path')
    for prefix,digest in CORE.items():
        found=[v for name,v in files.items() if Path(name).name.startswith(prefix)]
        require(len(found)==1 and found[0]['sha256']==digest,'actual core library differs: '+prefix)
    injected=[v for name,v in files.items() if Path(name).name=='libmps.so']
    require((len(injected)==1 and injected[0]['sha256']==MPS_SHA) if mode=='profiled' else not injected,
            'actual MPI collector injection differs')
    if mode=='profiled':
        for name,digest in COLLECTOR_LIBS.items():
            found=[v for path,v in files.items() if Path(path).name==name]
            require(len(found)==1 and found[0]['sha256']==digest,'actual collector library differs: '+name)
        require(all(not (Path(path).name.startswith(('libssl.so','libcrypto.so'))) or path in BASELINE_PRELOAD
                    for path in files),'bundled/replacement SSL library loaded')
        preload=[Path(x).resolve() for x in re.split(r'[:\s]+',process['environment']['LD_PRELOAD'] or '') if x]
        expected=[Path(x).resolve() for x in BASELINE_PRELOAD]+[(TOOL.parent.parent/'lib64/libmps.so').resolve()]
        require(len(preload)==len(expected) and set(preload)==set(expected),'actual APS child preload differs')
    else:require(not process['environment']['LD_PRELOAD'],'normal solver unexpectedly preloaded')
    cache={name:v for name,v in files.items() if name=='/etc/ld.so.cache'}
    return dict(pid=int(pid),loaded_images={name:v['sha256'] for name,v in files.items() if name not in cache},
        loader_cache_observations=cache,samples=r['samples'],
        cache_scope='Canonical loader data excluded from executable-image comparison; captured metadata retained, magic not captured.')


def mpi_reports(directory):
    reports=directory/'reports';data=directory/'profiled'/'aps';outputs={}
    for name,options in REPORT_OPTIONS.items():
        fmt='csv' if name in ('functions','per-rank') else 'txt'
        require(shlex.split((reports/(name+'.command.txt')).read_text())==
            [str(REPORT),*options,'--format='+fmt,str(CASE/'profiled'/'aps')],
            'APS report command differs: '+name)
        require((reports/(name+'.exit-code.txt')).read_text().strip()=='0','APS report failed: '+name)
        log=reports/(name+'.log');require(log.stat().st_size>0,'empty APS report: '+name)
        outputs[name]=dict(path=str(log),sha256=sha(log))
    # Actual APS compact layout: 0-23/<host>/stat-<rank>-<PID>._bin.
    raw={}
    for path in data.rglob('*._bin'):
        match=re.fullmatch(r'stat-(\d+)-(\d+)\._bin',path.name)
        if match:
            rank=int(match[1]);require(rank not in raw and path.stat().st_size>0,'duplicate/empty APS rank record')
            launch=read_json(directory/'profiled'/'ranks'/f'rank{rank}'/'launch.json')
            require(path.parent.name==launch['host'],'APS rank host differs from actual launch')
            if rank in SELECTED:
                observed=read_json(directory/'profiled'/'ranks'/f'rank{rank}'/'runtime.json')
                require(set(observed['solver_processes'])=={match[2]},'APS PID differs from observed solver')
            raw[rank]=dict(path=str(path),pid=int(match[2]),host=path.parent.name,
                           bytes=path.stat().st_size,sha256=sha(path))
    require(set(raw)==set(range(24)),'APS raw statistics do not cover exactly24 ranks')
    rank_rows={}
    for line in (reports/'per-rank.log').read_text().splitlines():
        cells=[x.strip() for x in line.split(';')]
        if len(cells)!=6 or not cells[0].isdigit():continue
        rank=int(cells[0]);values=[float(x) for x in cells[1:]]
        require(rank not in rank_rows and all(math.isfinite(x) and x>=0 for x in values),
                'duplicate/nonfinite APS rank summary')
        rank_rows[rank]=dict(zip(('elapsed_seconds','mpi_seconds','mpi_percent','disabled_imbalance_seconds',
                                 'disabled_imbalance_percent'),values))
    require(set(rank_rows)==set(range(24)),'APS report does not cover exactly24 actual ranks')
    tables=[p for p in reports.rglob('*') if p.is_file() and p.suffix in ('.csv','.log')]
    calls={name:[] for name in ('MPI_Testall','MPI_Isend','MPI_Irecv')}
    for path in tables:
        call_column=None;column_name=None
        for line in path.read_text().splitlines():
            if not line.strip():continue
            cells=next(csv.reader(io.StringIO(line),delimiter=';'))
            columns=[i for i,x in enumerate(cells) if x.strip().lower() in
                     ('calls','number of calls','call count')]
            if len(columns)==1:
                call_column=columns[0];column_name=cells[call_column].strip();continue
            for name in calls:
                if name not in line:continue
                if call_column is None or len(cells)<=call_column:continue
                try:value=float(cells[call_column].strip())
                except ValueError:continue
                if math.isfinite(value) and value>0:
                    calls[name].append(dict(path=str(path),line=line,call_column=column_name,positive_calls=value))
    require(all(calls.values()),'APS reports lack actual positive Testall/Isend/Irecv data rows')
    return dict(rank_records=raw,rank_summary=rank_rows,required_function_data=calls,report_logs=outputs,
        scope='Whole-process aggregate MPI statistics, including initialization/final outputs. Positive rows establish interception; column meanings and critical-rank attribution require actual report review. Imbalance0 inserts no requested imbalance barriers. No chronological MPI/GPU overlay or throughput claim.')


def audit(args,result):
    d=args.directory;require(sha(args.source)==SOURCE,'source changed');source=checkpoint(args.source,2486)
    require(sha(TOOL)==TOOL_SHA and sha(REPORT)==REPORT_SHA,'APS tools changed')
    before,after=[read_json(d/f'guard-{s}.json') for s in ('before','after')]
    require(before==after and before['passed'] and before['baseline_executable_sha256']==NORMAL,'baseline guards differ')
    for suffix in ('before','after'):
        for name,(_,_,_,text) in VERIFY.items():
            p=d/f'{name}-verify-{suffix}.txt'
            require(p.read_text()==text and sha(p)==before[name.replace('-','_')+'_verify_log_sha256'],'guard log differs')
    states={};out={};observed={}
    for mode in ('normal','profiled'):
        folder=d/mode
        require(shlex.split((folder/'command.txt').read_text())==['mpiexec','-n','24','-ppn','12',BINDING,WRAPPER,
            '/usr/bin/python3.11',str(WORK/'native_n12_mpi_rank.py'),'--mode',mode,'--directory',str(CASE/mode)],
            'outer launch differs')
        require((folder/'exit-code.txt').read_text().strip()=='0' and 'Terminating on cycle limit' in
                (folder/'run.log').read_text(),'solver/APS did not complete')
        require([p.name for p in (folder/'rst').glob('*.rst')]==['native_n12_5M.00008.rst'],'checkpoint set differs')
        state=checkpoint(folder/'rst/native_n12_5M.00008.rst');states[mode]=state;out[mode]=diagnostics(folder)
        p,original=state['parameters'],source['parameters']
        mutable={'time/nlim','time/tlim','time/ndiag','output1/last_time','output2/last_time',
            'output2/file_number','fastflow/last_a0_0','fastflow/ah_found_a0_0'}
        require(p.keys()==original.keys() and all(p[k]==original[k] for k in p if k not in mutable),'physics policy differs')
        for key in ('atlas','topology','costs','metadata','identity','stride'):require(state[key]==source[key],'source mesh differs: '+key)
        require(state['fixed'][:120]==source['fixed'][:120] and state['fixed'][156:-20]==source['fixed'][156:-20],
                'initialized source header differs')
        require((int(p['time/nlim']),float(p['time/tlim']),int(p['time/ndiag']),int(p['output2/file_number']))==
                (2490,2.,1000000000,9),'endpoint controls differ')
        require(all(float(p[b+'/last_time'])==sum([1/6]*8) for b in ('output1','output2')),'output cadence differs')
        owners=set()
        for rank in range(24):
            launch=read_json(folder/'ranks'/f'rank{rank}'/'launch.json');local=launch['local_rank'];env=launch['environment']
            require(launch['schema']=='native-n12-mpi-rank-v1' and launch['rank']==rank and launch['mode']==mode and
                launch['selected']==(rank in SELECTED) and launch['profiled']==(mode=='profiled'),'rank/profile scope differs')
            require(0<=local<12 and env['ZE_AFFINITY_MASK']==f'{local//2}.{local%2}' and env['EnableImplicitScaling']=='0' and
                (launch['host'],local) not in owners,'rank/tile differs');owners.add((launch['host'],local))
            expected=[str(ROOT/'build-sycl/src/athena'),'-r',str(SOURCE_PATH),'-d',str(CASE/mode),
                'time/nlim=2490','time/tlim=2','time/ndiag=1000000000']
            if mode=='profiled':expected=[str(TOOL),*OPTIONS,'--result-dir='+str(CASE/mode/'aps'),*expected]
            require(launch['command']==expected and launch['baseline_executable_sha256']==NORMAL,'rank command differs')
            overrides={'LD_PRELOAD':':'.join(BASELINE_PRELOAD)} if mode=='profiled' else {}
            require(launch['child_environment_overrides']==overrides,'child preload override differs')
        require(len(owners)==24 and len({host for host,rank in owners})==2,'not24 ranks on2 nodes')
        observed[mode]={str(rank):runtime(d,mode,rank) for rank in SELECTED}
    result['state_equivalence']=compare_checkpoints(states['normal'],states['profiled'])
    require(out['normal']==out['profiled'],'ordinary diagnostics differ');result['diagnostics_exact']=out['normal']
    result['runtime_identities']=observed
    for rank in map(str,SELECTED):
        for name,h in observed['normal'][rank]['loaded_images'].items():
            require(observed['profiled'][rank]['loaded_images'].get(name)==h,'profile replaces observed image: '+name)
    result['scientific_qualification_passed']=True
    result['mpi_statistics']=mpi_reports(d);result['mpi_interception_verified']=True;result['passed']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-failure',type=Path)
    p.add_argument('--terminal',type=Path)
    args=p.parse_args();require(not args.output.exists(),'preserve previous qualification')
    result=dict(schema='native-n12-mpi-profile-qualification-v1',passed=False,scientific_qualification_passed=False,
        mpi_interception_verified=False,performance_measurement=False,cycles=[2486,2490],ranks=24,
        scope='Four-cycle APS aggregate MPI probe; numerical identity and actual interception required. No throughput/causal critical-path claim.')
    try:
        if args.previous_failure is not None or args.terminal is not None:
            require(args.previous_failure is not None and args.terminal is not None and
                args.output.name=='qualification-v2.json' and
                args.previous_failure.resolve()==(args.directory/'qualification.json').resolve(),
                'reanalysis must retain original qualification and use new v2 output')
            old=read_json(args.previous_failure);terminal=args.terminal.read_text()
            require(old['passed'] is False and old['scientific_qualification_passed'] is True and
                old.get('failure')=='ValueError: APS raw statistics do not cover exactly24 ranks',
                'unexpected original audit failure')
            job=re.search(r'^Job Id:\s*(\S+)',terminal,re.M)
            require(job and job[1].split('.')[0]==(args.directory/'job-id.txt').read_text().strip().split('.')[0],
                    'reanalysis terminal job differs')
            for key,value in [('job_state','F'),('Exit_status','2'),('Job_Name','native-n12-mpi-two'),('Resource_List.nodect','2')]:
                found=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',terminal,re.M)
                require(found and found[1]==value,'reanalysis terminal field differs: '+key)
            result['reanalysis']=dict(previous_report=str(args.previous_failure),previous_report_sha256=sha(args.previous_failure),
                previous_failure=old['failure'],terminal_record=str(args.terminal),terminal_sha256=sha(args.terminal),
                job_id=job[1].split('.')[0],job_state='F',batch_exit_status=2,solver_exit_statuses={'normal':0,'profiled':0},
                scope='Offline actual APS layout/CSV parsing correction. Original F/2 audit report and all raw evidence retained; both solver exits checked below. No GPU rerun.')
        audit(args,result)
    except Exception as e:result.update(passed=False,failure=type(e).__name__+': '+str(e))
    result['artifact_manifest']={str(p.relative_to(args.directory)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in args.directory.rglob('*') if p.is_file() and 'cache' not in p.relative_to(args.directory).parts and p!=args.output}
    result['source_sha256']={name:sha(Path(__file__).with_name(name)) for name in
        ('native_n12_mpi_audit.py','native_n12_mpi_rank.py','native_n12_profile_audit.py',
         'native_n12_profile_rank.py','native_two_node_timing_audit.py')}
    write_json(args.output,result);print({k:result[k] for k in ('passed','scientific_qualification_passed','mpi_interception_verified','failure') if k in result})
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
