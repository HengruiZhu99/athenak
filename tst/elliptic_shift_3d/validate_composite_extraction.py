#!/usr/bin/env python3
"""Bounded before/after qualification for the mechanical stage-engine extraction."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['baseline','after']);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();build=root/'build-shift-3d';out=args.output.resolve()
    if args.phase=='baseline':out.mkdir(parents=True,exist_ok=False)
    else:
        if not (out/'baseline.json').exists():raise FileNotFoundError('baseline.json')
    manifest=dict(phase=args.phase,scope='Mechanical production-header extraction compatibility only; no native Task wiring or new scientific success',commands=[],source_hashes={str(p.relative_to(root)):sha(p) for p in sorted((root/'src').rglob('*.hpp'))},coupled_sha256=sha(root/'tst/elliptic_shift_3d/coupled.cpp'))
    def save():(out/(args.phase+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    def run(command,name,timeout=180):
        row=dict(name=name,command=list(map(str,command)),timeout=timeout);manifest['commands'].append(row);save();start=time.monotonic()
        with (out/(args.phase+'-'+name+'.stdout')).open('w') as stdout,(out/(args.phase+'-'+name+'.stderr')).open('w') as stderr:
            process=subprocess.Popen(row['command'],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['pid']=process.pid;save()
            try:row['exit_code']=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row.update(exit_code=124,status='timeout');save()
                try:
                    group=os.getpgid(process.pid)
                    if group!=process.pid:raise RuntimeError('unexpected group; no signal sent')
                    os.killpg(group,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(group,signal.SIGKILL);process.wait(timeout=5)
                    row['owned_group_cleanup']='completed'
                except Exception as error:row['owned_group_cleanup_error']=f'{type(error).__name__}: {error}'
            row['seconds']=time.monotonic()-start
        save();print(name,row['exit_code'],row['seconds'],flush=True);return row
    if args.phase=='baseline':shutil.copy2(root/'tst/elliptic_shift_3d/coupled.cpp',out/'baseline-coupled.cpp')
    buildrow=run(['cmake','--build',str(build),'--target','elliptic_shift_3d_coupled_pilot','--parallel','2'],'build')
    if buildrow['exit_code']:raise SystemExit(buildrow['exit_code'])
    binary=build/'elliptic_shift_3d_coupled_pilot';manifest['binary_sha256']=sha(binary)
    if args.phase=='baseline':shutil.copy2(binary,build/'composite-extraction-baseline-pilot')
    initial=out/'wormhole-initial.dat'
    if args.phase=='baseline':
        origin=root/'docs/elliptic-shift-3d/evidence/enriched-buffered-pilot-20261027/initial.dat';path=origin if origin.exists() else origin.with_name(origin.name+'.gz');data=path.read_bytes();initial.write_bytes(gzip.decompress(data) if path.suffix=='.gz' else data);manifest['initial_source']=dict(path=str(path),sha256=sha(path),decoded_sha256=sha(initial))
    tag='e'*64
    cases=[]
    for shells in [1,2]:
        name=f'minkowski-{shells}-shell';profile=out/(args.phase+'-'+name+'.csv');checkpoint=out/(args.phase+'-'+name+'.bin')
        cases.append((name,[str(binary),'6','0','.0005','2','0',str(shells),'1e-11',str(profile),str(checkpoint),tag,'4','1']))
    name='wormhole-zero';profile=out/(args.phase+'-'+name+'.csv')
    cases.append((name,[str(binary),'8','0','.0001','0','0','7','1e-10',str(profile),'','','4','1','4','8','1',str(initial),'2','0','.45,.55,.60,.65,.70,.75,1','8','1','2','.24','1','1']))
    for name,command in cases:
        result=run(command,name)
        stdout=out/(args.phase+'-'+name+'.stdout')
        if stdout.read_text():
            try:result['result']=json.loads(stdout.read_text())
            except ValueError:result['unparsed_stdout']=True
        result['output_hashes']={p.name:sha(p) for p in out.glob(args.phase+'-'+name+'.*') if p.suffix not in ['.stdout','.stderr']};save()
        if result['exit_code']:raise SystemExit(result['exit_code'])
    manifest['source_changes_during_phase']=[name for name,value in manifest['source_hashes'].items() if not (root/name).exists() or sha(root/name)!=value]
    if args.phase=='after':
        baseline=json.loads((out/'baseline.json').read_text());checks=[]
        for name,_ in cases:
            before=next(c for c in baseline['commands'] if c['name']==name);after=next(c for c in manifest['commands'] if c['name']==name)
            b={k:v for k,v in before['result'].items() if k!='seconds'};a={k:v for k,v in after['result'].items() if k!='seconds'}
            different={k:dict(before=b.get(k),after=a.get(k)) for k in set(b)|set(a) if b.get(k)!=a.get(k)};files={}
            for suffix in ['.csv','.csv.spheres.csv','.bin']:
                bp=out/('baseline-'+name+suffix);ap=out/('after-'+name+suffix)
                if bp.exists():files[suffix]=dict(before=sha(bp),after=sha(ap),identical=bp.read_bytes()==ap.read_bytes())
            checks.append(dict(name=name,diagnostics_equal=not different,diagnostic_differences=different,files=files))
        manifest['before_after_checks']=checks
        if not all(c['diagnostics_equal'] and all(v['identical'] for v in c['files'].values()) for c in checks):save();raise SystemExit('Mechanical extraction changed diagnostic or state bits')
        # Syntax-check every existing renamed-main harness using the pilot flags.
        flags=(build/'CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make').read_text();options=[]
        for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:options+=shlex.split(next(line.split('=',1)[1] for line in flags.splitlines() if line.startswith(key+' =')))
        for name in ['diagnose_wormhole.cpp','instantaneous_radial_snapshot.cpp']:
            result=run(['/usr/bin/c++',*options,'-fsyntax-only',str(root/'tst/elliptic_shift_3d'/name)],'harness-'+name)
            if result['exit_code']:raise SystemExit(result['exit_code'])
        result=run(['ctest','--test-dir',str(build),'-R','^elliptic_shift_3d_','--output-on-failure','--parallel','1'],'ctest')
        if result['exit_code']:raise SystemExit(result['exit_code'])
    manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256'];save();print(json.dumps(dict(phase=args.phase,status='passed',commands=len(manifest['commands']))),flush=True)


if __name__=='__main__':main()
