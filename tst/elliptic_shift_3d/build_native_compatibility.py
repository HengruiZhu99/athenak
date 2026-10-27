#!/usr/bin/env python3
"""Bound only the production athena target build; no scientific execution."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();build=root/'build-shift-3d';out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[f for f in (root/'src').rglob('*') if f.is_file() and f.suffix in ['.hpp','.h','.cpp']]+[root/'CMakeLists.txt',root/'src/CMakeLists.txt',build/'config.hpp']
    source_hashes={str(f.relative_to(root)):sha(f) for f in sorted(files)}
    command=['cmake','--build',str(build),'--target','athena','--parallel','2']
    manifest=dict(scope='Production AthenaK target compilation/link compatibility only; no composite-shift integration or evolution claim',
        command=command,workers=2,maximum_wall_seconds=180,chunk_seconds=90,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        head=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        dirty_status=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout,
        source_hashes_before=source_hashes,status='starting',checkpoints=[])
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save();start=time.monotonic()
    with (out/'build.stdout').open('w') as stdout,(out/'build.stderr').open('w') as stderr:
        process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
        manifest.update(pid=process.pid,process_group=process.pid,status='building');save()
        for chunk in range(2):
            try:
                code=process.wait(timeout=90);manifest.update(exit_code=code,status='build_succeeded' if code==0 else 'compiler_or_build_failure');break
            except subprocess.TimeoutExpired:
                manifest['checkpoints'].append(dict(chunk=chunk+1,elapsed_seconds=time.monotonic()-start,status='still_running'));save()
                print(json.dumps(manifest['checkpoints'][-1]),flush=True)
        else:
            manifest.update(exit_code=124,status='wall_timeout',owned_group_termination='pending');save()
            try:
                group=os.getpgid(process.pid)
                if group!=process.pid:raise RuntimeError('refusing to signal group other than newly created build group')
                os.killpg(group,signal.SIGTERM);manifest['owned_group_termination']='SIGTERM_sent';save()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(group,signal.SIGKILL);manifest['owned_group_termination']='SIGKILL_sent_after_SIGTERM';process.wait(timeout=5)
                manifest['owned_group_termination']='completed';manifest['build_process_exit_code']=process.returncode
            except (PermissionError,ProcessLookupError,RuntimeError,subprocess.TimeoutExpired) as error:
                manifest.update(owned_group_termination='denied_or_uncertain',termination_error=f'{type(error).__name__}: {error}',build_process_exit_code=process.poll())
            save()
    manifest['build_wall_seconds']=time.monotonic()-start
    changes=[]
    for name,before in source_hashes.items():
        path=root/name;after=sha(path) if path.exists() else None
        if before!=after:changes.append(dict(path=name,before=before,after=after))
    manifest['source_changes_during_build']=changes
    binary=build/'src/athena'
    if manifest.get('exit_code')==0:
        manifest['binary_sha256']=sha(binary);manifest['binary_bytes']=binary.stat().st_size
        try:
            help_run=subprocess.run([str(binary),'-h'],cwd=root,capture_output=True,text=True,timeout=15)
            (out/'help.stdout').write_text(help_run.stdout);(out/'help.stderr').write_text(help_run.stderr)
            manifest['help']=dict(command=[str(binary),'-h'],exit_code=help_run.returncode,scope='Read-only usage output; no input file/evolution')
        except subprocess.TimeoutExpired as error:
            manifest['help']=dict(command=[str(binary),'-h'],exit_code=124,error='usage-output timeout')
            for suffix,value in [('stdout',error.stdout),('stderr',error.stderr)]:
                (out/('help.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
    save();print(json.dumps({key:manifest.get(key) for key in ['status','exit_code','build_wall_seconds','owned_group_termination','termination_error','help']}),flush=True)


if __name__=='__main__':main()
