#!/usr/bin/env python3
"""Launch unchanged f3; profile only ranks8/23 and sample their actual library maps."""
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time

from native_two_node_timing_audit import ROOT,SOURCE_PATH,NORMAL,PROFILE,require,sha,write_json

TOOL=Path('/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-hi6zrvx/bin/unitrace')
TOOL_SHA='af42167b3842eaaffacd99a7be56aaf6cbcb5f8b6df259f9e3a7bb4eb1f9e17c'
OPTIONS=['--host-timing','--device-timing','--kernel-submission','--chrome-call-logging',
         '--chrome-kernel-logging','--chrome-sycl-logging','--verbose','--pid','--tid',
         '--chrome-event-buffer-size','4096']
SELECTED=(8,23)
CACHE_ENV={'MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE':'specialized',
           'MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE':'limited',
           'MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES':'128'}
CACHE_ENV.update({key.replace('MPIR_CVAR_',prefix):None for key in tuple(CACHE_ENV)
                  for prefix in ('MPICH_','MPIR_PARAM_')})
ENV_KEYS=['OMP_NUM_THREADS','ZE_FLAT_DEVICE_HIERARCHY','ZE_ENABLE_PCI_ID_DEVICE_ORDER',
    'ONEAPI_DEVICE_SELECTOR','MPIR_CVAR_ENABLE_GPU','ZE_AFFINITY_MASK','EnableImplicitScaling',
    'NEO_CACHE_DIR','SYCL_CACHE_DIR','LD_LIBRARY_PATH','LD_PRELOAD',*PROFILE,*CACHE_ENV]


def descendants(pid):
    result=[pid]
    try:
        for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
            result.extend(descendants(int(child)))
    except (FileNotFoundError,ProcessLookupError):pass
    return result


def snapshot(pid, executable):
    """Only actual f3 processes; no unrelated /proc process scan."""
    rows=[]
    for child in descendants(pid):
        try:
            proc=Path(f'/proc/{child}')
            expected=executable.resolve(strict=True)
            if (proc/'exe').resolve(strict=True)!=expected:continue
            maps=(proc/'maps').read_text()
            environment=(proc/'environ').read_bytes()
            # proc files are separate observations, not an atomic snapshot.
            # Exec/exit can clear maps/environ between these reads. Accept only
            # a nonempty sample bracketed by the same live native executable.
            if not maps.strip() or not environment:continue
            if (proc/'exe').resolve(strict=True)!=expected:continue
            loaded={}
            for line in maps.splitlines():
                parts=line.split(maxsplit=5)
                if len(parts)!=6 or not parts[5].startswith('/'):continue
                path=parts[5]
                if '.so' not in Path(path).name and path!=str(executable):continue
                require(not path.endswith(' (deleted)'),'deleted mapped runtime image')
                loaded[path]=dict(inode=int(parts[4]),device=parts[3])
            environ=dict(x.split('=',1) for x in environment.decode().split('\0') if '=' in x)
            rows.append(dict(pid=child,executable=str(expected),
                mapped_files=loaded,environment={k:environ.get(k) for k in ENV_KEYS}))
        except (FileNotFoundError,ProcessLookupError):pass
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['normal','profiled'],required=True)
    p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    rank_values={int(os.environ[k]) for k in ('PMI_RANK','PALS_RANKID') if k in os.environ}
    require(len(rank_values)==1,'missing/conflicting global rank');rank=rank_values.pop()
    local=int(os.environ.get('MPI_LOCALRANKID',os.environ.get('PALS_LOCAL_RANKID','-1')))
    require(0<=rank<24 and 0<=local<12 and os.environ.get('ZE_AFFINITY_MASK')==f'{local//2}.{local%2}'
            and os.environ.get('EnableImplicitScaling')=='0','wrong actual rank/tile')
    require(all(not os.environ.get(k) for k in (*PROFILE,'LD_PRELOAD')) and
            not any(k.startswith(('APS_','MPS_')) for k in os.environ),
            'unexpected inherited profiling environment')
    require({k:os.environ.get(k) for k in CACHE_ENV}==CACHE_ENV,'cache128 environment differs')
    executable=ROOT/'build-sycl/src/athena'
    # The outer collective guard already hashes f3. Avoid24 simultaneous executable hashes.
    command=[str(executable),'-r',str(SOURCE_PATH),'-d',str(args.directory),
             'time/nlim=2490','time/tlim=2','time/ndiag=1000000000']
    rankdir=args.directory/'ranks'/f'rank{rank}';rankdir.mkdir(parents=True,exist_ok=False)
    profile=args.mode=='profiled' and rank in SELECTED
    if profile:
        require(sha(TOOL)==TOOL_SHA,'unitrace executable changed')
        command=[str(TOOL),*OPTIONS,'--output-dir-path',str(rankdir),'--output','unitrace.txt',*command]
    initial=dict(schema='native-n12-profile-rank-v1',rank=rank,local_rank=local,host=socket.gethostname(),
        mode=args.mode,selected=rank in SELECTED,profiled=profile,command=command,
        environment={k:os.environ.get(k) for k in ENV_KEYS},baseline_executable_sha256=NORMAL,
        tool_sha256=TOOL_SHA if profile else None,map_sample_period_seconds=.25,
        scope='Selected-rank process/library observations only; profiled elapsed time is never throughput')
    write_json(rankdir/'launch.json',initial)
    if rank not in SELECTED:
        os.execv(command[0],command)
    # Same external monitor in both controls; no solver instrumentation or numerical changes.
    # Preserve PMI_FD/PALS descriptors inherited from mpiexec. No extra files
    # remain open here; all JSON/proc reads use scoped opens.
    child=subprocess.Popen(command,close_fds=False)
    observations={};samples=0;begin=time.monotonic()
    try:
        while child.poll() is None:
            require(time.monotonic()-begin<600,'profile rank monitor exceeded600s')
            for row in snapshot(child.pid,executable):
                key=row['pid'];entry=observations.setdefault(key,dict(first_seconds=time.monotonic()-begin,
                    last_seconds=0.,executable=row['executable'],mapped_files={},environment=row['environment']))
                if entry['environment']!=row['environment']:
                    changed={k:dict(previous=entry['environment'].get(k),current=row['environment'].get(k))
                        for k in ENV_KEYS if entry['environment'].get(k)!=row['environment'].get(k)}
                    write_json(rankdir/'runtime-environment-mismatch.json',dict(rank=rank,mode=args.mode,
                        pid=key,elapsed_seconds=time.monotonic()-begin,changed=changed,
                        previous=entry['environment'],current=row['environment']))
                    require(False,'solver environment changed during observation')
                entry['last_seconds']=time.monotonic()-begin;entry['mapped_files'].update(row['mapped_files'])
            samples+=1;time.sleep(.25)
        require(child.returncode==0,'selected solver/profiler failed')
        require(len(observations)==1,'expected exactly one observed f3 process')
        for entry in observations.values():
            for name,record in entry['mapped_files'].items():
                path=Path(name);stat=path.stat()
                require(stat.st_ino==record['inode'],'mapped file inode changed')
                record.update(sha256=sha(path),bytes=stat.st_size)
        write_json(rankdir/'runtime.json',dict(schema='native-n12-profile-runtime-v1',passed=True,
            rank=rank,mode=args.mode,launcher_pid=child.pid,solver_processes=observations,
            samples=samples,exit_code=child.returncode,observed_seconds=time.monotonic()-begin,
            interpretation='Sampling records loaded file identities; it is not GPU utilization or throughput'))
    except BaseException as error:
        # Save descendants before parent termination can reparent them.
        victims=set(descendants(child.pid)) if child.poll() is None else set()
        for pid in reversed(sorted(victims)):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        for pid in victims:
            if pid==child.pid and child.poll() is not None:continue
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        child.wait()
        write_json(rankdir/'runtime-failure.json',dict(passed=False,rank=rank,mode=args.mode,
            failure=type(error).__name__+': '+str(error),terminated_pids=sorted(victims),
            observations=observations,samples=samples))
        raise

if __name__=='__main__':main()
