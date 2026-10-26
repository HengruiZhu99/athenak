#!/usr/bin/env python3
"""Fresh disturbed-Minkowski cubed-sphere controls in bounded one-node jobs.

Scientific failures are recorded as outcomes. Timeout is inconclusive. The
boundary is a spherical mapped face; extrapolation is not a derived scri BC.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run_case(binary,source,out,name,tile,bulk,nr,na,amp,kappa,cfl,tlim,degree,timeout,pole_cfl=.03,extra=None):
    d=out/name;d.mkdir(exist_ok=False)
    shutil.copy2(source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',d/'input.athinput')
    command=[str(binary),'-i',str(d/'input.athinput'),'-d',str(d),
        *['mesh/nx%d=%d'%(a,bulk) for a in [1,2,3]],
        'cubed_sphere/nr=%d'%nr,'cubed_sphere/nangular=%d'%na,
        'cubed_sphere/conformal_amplitude=%.17g'%amp,
        'cubed_sphere/conformal_kappa1=%.17g'%kappa,
        'cubed_sphere/conformal_cfl=%.17g'%cfl,
        'cubed_sphere/conformal_pole_cfl=%.17g'%pole_cfl,'cubed_sphere/conformal_diss=0.1',
        'cubed_sphere/conformal_probe_tlim=%.17g'%tlim,
        'cubed_sphere/conformal_outer_degree=%d'%degree,
        'cubed_sphere/conformal_max_cycles=40000','cubed_sphere/conformal_binary_reference=true']
    if extra:
        command.extend(f'{key}={str(value).lower() if isinstance(value,bool) else value}' for key,value in extra.items())
    receipt={'command':command,'device_tile':tile,'input_sha256':sha(d/'input.athinput'),
        'bulk':bulk,'nr':nr,'nangular':na,'amplitude':amp,'kappa1':kappa,'cfl':cfl,
        'target_time':tlim,'outer_degree':degree,'pole_cfl':pole_cfl,
        'dissipation':(extra or {}).get('cubed_sphere/conformal_diss',.1)}
    if extra:receipt['extra_parameters']=extra
    restart=(extra or {}).get('cubed_sphere/conformal_restart','none')
    if restart!='none':
        rp=Path(restart);receipt['restart_payload']={'path':str(rp),'sha256':sha(rp),'bytes':rp.stat().st_size}
    (d/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    env=dict(os.environ,ZE_AFFINITY_MASK=tile,ONEAPI_DEVICE_SELECTOR='level_zero:gpu')
    start=time.monotonic()
    with (d/'run.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:receipt['exit_code']=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            receipt['timeout']=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    receipt['elapsed_seconds']=time.monotonic()-start
    path=d/'cubed_sphere_conformal.csv';rows=list(csv.DictReader(path.open())) if path.exists() else []
    receipt['last_rows']=rows[-2:]
    receipt['all_rows_finite']=bool(rows) and all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k!='region')
    refs=[d/('cubed_sphere_reference_'+s+'.bin') for s in ['before','after']]
    receipt['reference_payloads']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in refs if p.exists()}
    receipt['reference_unchanged']=all(p.exists() for p in refs) and sha(refs[0])==sha(refs[1])
    receipt['reached_time_target']=receipt.get('exit_code')==0 and len(rows)>=2 and all(
        abs(float(r['time'])-tlim)<1e-13 for r in rows[-2:])
    requested_steps=(extra or {}).get('cubed_sphere/conformal_probe_steps',0)
    receipt['reached_step_target']=receipt.get('exit_code')==0 and requested_steps>0 and len(rows)>=2 and all(
        int(r['cycle'])==requested_steps for r in rows[-2:])
    receipt['reached_target']=receipt['reached_time_target'] or receipt['reached_step_target']
    profiles=[d/('cubed_sphere_initial_profile_'+s+'.bin') for s in ['before','after']]
    if any(p.exists() for p in profiles):
        receipt['initial_profile_payloads']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in profiles if p.exists()}
        receipt['initial_profile_unchanged']=all(p.exists() for p in profiles) and sha(profiles[0])==sha(profiles[1])
    text=(d/'run.log').read_text(errors='replace')
    receipt['terminal_messages']=[l for l in text.splitlines() if 'what()' in l or 'uncaught exception' in l or 'FATAL ERROR' in l]
    receipt['failure_files']=[p.name for p in d.glob('*bad_stage*')]+[p.name for p in d.glob('*rhs_failure*')]
    cp=(extra or {}).get('cubed_sphere/conformal_checkpoint','none')
    if cp!='none' and Path(cp).exists():
        path=Path(cp);receipt['checkpoint_payload']={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    receipt['bounded_wall_stop']='bounded_wall_stop=true' in text
    receipt['outcome']='complete' if receipt['reached_target'] else ('timeout' if receipt.get('timeout') else 'failed')
    if receipt.get('exit_code')==0 and receipt['bounded_wall_stop'] and 'checkpoint_payload' in receipt and receipt['reference_unchanged'] and receipt['all_rows_finite']:
        receipt['outcome']='checkpointed'
    (d/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':name,'outcome':receipt['outcome'],'elapsed_seconds':receipt['elapsed_seconds'],
                      'last_rows':receipt['last_rows'],'terminal_messages':receipt['terminal_messages']}),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary',type=Path);ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--mode',choices=['smoke','comparison'],required=True)
    ap.add_argument('--timeout',type=int,default=450)
    ap.add_argument('--nr',type=int,default=272);ap.add_argument('--na',type=int,default=12)
    ap.add_argument('--bulk',type=int,default=16);ap.add_argument('--tlim',type=float,default=2)
    ap.add_argument('--cfl',type=float,default=.1);ap.add_argument('--pole-cfl',type=float,default=.03)
    ap.add_argument('--cpu-golden',type=Path)
    args=ap.parse_args();assert 0<args.timeout<=480
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    binary=out/'athena_frozen';assert not binary.exists();shutil.copy2(args.binary,binary)
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',
           source/'src/CMakeLists.txt',source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',Path(__file__).resolve()]
    launch={'binary_sha256':sha(binary),'mode':args.mode,
            'source_sha256':{str(p.relative_to(source)):sha(p) for p in sorted(set(files))},
            'scope':'One rank, uniform bulk, immutable Minkowski reference, original wave-map equations and pulse; mapped derivatives/interfaces/KO and outer ghosts differ from old cut-sphere.'}
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    if args.mode=='smoke':
        cases={}
        for name,amp,tlim in [('stationary',0,.001),('angular',.2,.005)]:
            cases[name]=run_case(binary,source,out,name,'0.0',16,12,12,amp,0,.025,tlim,2,args.timeout)
            assert cases[name]['reached_target'] and cases[name]['reference_unchanged'] and cases[name]['all_rows_finite']
        assert all(float(r['max_deviation'])<1e-10 and float(r['H_linf'])<1e-8 for r in cases['stationary']['last_rows'])
        if args.cpu_golden:
            gold=json.loads(args.cpu_golden.read_text())['last_rows']
            diffs={}
            keys=['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2','min_alpha','min_chi','max_deviation']
            for a,b in zip(gold,cases['angular']['last_rows']):
                assert a['region']==b['region']
                diffs[a['region']]={k:abs(float(a[k])-float(b[k]))/max(abs(float(a[k])),1e-20) for k in keys}
                assert max(diffs[a['region']].values())<1e-7,diffs
            launch['CPU_GPU_relative_difference']=diffs
        launch['device_smoke_acceptance']=True
        # A longer old-parameter run at the wider collar is a separate science
        # control. Its completion is not a requirement for admitting the port.
        cases['original-pulse-wide-collar']=run_case(binary,source,out,
            'original-pulse-wide-collar','0.0',16,12,12,.2,10,.1,2,2,args.timeout)
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs={str(degree):pool.submit(run_case,binary,source,out,'degree%d'%degree,'0.%d'%tile,
                args.bulk,args.nr,args.na,.2,10,args.cfl,args.tlim,degree,args.timeout,args.pole_cfl)
                for tile,degree in enumerate([2,4])}
            # An independent physical GPU repeats the wider-collar baseline
            # with the same bounded case budget, avoiding contention with the
            # two matched-collar controls on the first GPU's two tiles.
            jobs['wide-collar']=pool.submit(run_case,binary,source,out,'wide-collar','1.0',
                16,12,12,.2,10,.1,2,2,args.timeout,.03)
            cases={name:job.result() for name,job in jobs.items()}
    launch['cases']=cases
    (out/'summary.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
