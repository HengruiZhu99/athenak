#!/usr/bin/env python3
"""Launch unchanged f3; aggregate MPI on all24 ranks, monitor actual images on0/8/23."""
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time

from native_two_node_timing_audit import ROOT,SOURCE_PATH,NORMAL,PROFILE,require,sha,write_json

TOOL=Path('/opt/aurora/26.26.0/oneapi/vtune/latest/bin64/aps')
TOOL_SHA='8c53f8342daef6067f4191eea0f222170a7e05ff6e282c1f9dd7a96e166dba39'
OPTIONS=['--collection-mode=mpi','--stat-level=3','--mpi-imbalance=0']
SELECTED=(0,8,23)
BASELINE_PRELOAD={
    '/usr/lib64/libssl.so.3.5.0':'6b0bd0ed1c1b8c275cd8b590c6106821a7cd99fae109da4b90f7b9187703563c',
    '/usr/lib64/libcrypto.so.3.5.0':'e412eb49711f57086eb9ded394804e7ba05d4acbbdccd90a099733d08d0484da'}
ENV_KEYS=['OMP_NUM_THREADS','ZE_FLAT_DEVICE_HIERARCHY','ZE_ENABLE_PCI_ID_DEVICE_ORDER',
    'ONEAPI_DEVICE_SELECTOR','MPIR_CVAR_ENABLE_GPU','ZE_AFFINITY_MASK','EnableImplicitScaling',
    'NEO_CACHE_DIR','SYCL_CACHE_DIR','LD_LIBRARY_PATH','LD_PRELOAD',
    'APS_STAT_LEVEL','MPS_STAT_LEVEL','APS_IMBALANCE_TYPE','APS_STAT_BASE_DIR',*PROFILE]


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
            if (proc/'exe').resolve()!=executable.resolve():continue
            loaded={}
            for line in (proc/'maps').read_text().splitlines():
                parts=line.split(maxsplit=5)
                if len(parts)!=6 or not parts[5].startswith('/'):continue
                path=parts[5]
                if '.so' not in Path(path).name and path!=str(executable):continue
                require(not path.endswith(' (deleted)'),'deleted mapped runtime image')
                loaded[path]=dict(inode=int(parts[4]),device=parts[3])
            environ=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x)
            rows.append(dict(pid=child,executable=str((proc/'exe').resolve()),
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
    require(all(not os.environ.get(k) for k in (*PROFILE,'LD_PRELOAD','APS_STAT_LEVEL','MPS_STAT_LEVEL','APS_IMBALANCE_TYPE','APS_STAT_BASE_DIR')),'unexpected inherited profiling environment')
    executable=ROOT/'build-sycl/src/athena'
    # The outer collective guard already hashes f3. Avoid24 simultaneous executable hashes.
    command=[str(executable),'-r',str(SOURCE_PATH),'-d',str(args.directory),
             'time/nlim=2490','time/tlim=2','time/ndiag=1000000000']
    rankdir=args.directory/'ranks'/f'rank{rank}';rankdir.mkdir(parents=True,exist_ok=False)
    profile=args.mode=='profiled'
    child_environment=os.environ.copy();overrides={}
    if profile:
        require(sha(TOOL)==TOOL_SHA,'APS executable changed')
        # libmps otherwise resolves VTune's older bundled OpenSSL, which cannot
        # satisfy the native libcurl dependency. Pin the same original native
        # images for the APS child; preserve the ordinary environment and paths.
        if local==0:
            for name,digest in BASELINE_PRELOAD.items():
                require(sha(Path(name))==digest,'original native OpenSSL changed')
        overrides={'LD_PRELOAD':':'.join(BASELINE_PRELOAD)}
        child_environment.update(overrides)
        command=[str(TOOL),*OPTIONS,'--result-dir='+str(args.directory/'aps'),*command]
    initial=dict(schema='native-n12-mpi-rank-v1',rank=rank,local_rank=local,host=socket.gethostname(),
        mode=args.mode,selected=rank in SELECTED,profiled=profile,command=command,
        environment={k:os.environ.get(k) for k in ENV_KEYS},baseline_executable_sha256=NORMAL,
        child_environment_overrides=overrides,
        tool_sha256=TOOL_SHA if profile else None,map_sample_period_seconds=.25,
        scope='All24-rank aggregate MPI statistics with imbalance0; selected-rank actual runtime observations. Profiled elapsed time is never throughput')
    write_json(rankdir/'launch.json',initial)
    if rank not in SELECTED:
        os.execve(command[0],command,child_environment)
    # Same external monitor in both controls; no solver instrumentation or numerical changes.
    # Preserve PMI_FD/PALS descriptors inherited from mpiexec. No extra files
    # remain open here; all JSON/proc reads use scoped opens.
    child=subprocess.Popen(command,close_fds=False,env=child_environment)
    observations={};samples=0;begin=time.monotonic()
    try:
        while child.poll() is None:
            require(time.monotonic()-begin<600,'profile rank monitor exceeded600s')
            for row in snapshot(child.pid,executable):
                key=row['pid'];entry=observations.setdefault(key,dict(first_seconds=time.monotonic()-begin,
                    last_seconds=0.,executable=row['executable'],mapped_files={},environment=row['environment']))
                require(entry['environment']==row['environment'],'solver environment changed during observation')
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
            failure=type(error).__name__+': '+str(error),terminated_pids=sorted(victims)))
        raise

if __name__=='__main__':main()
