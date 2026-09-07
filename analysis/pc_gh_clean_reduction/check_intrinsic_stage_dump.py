#!/usr/bin/env python3
"""Audit actual RK and exchange stage dumps on independently seeded periodic data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart
from intrinsic_diagnostics import diagnostics, norms, reduction_injection

def read_dump(path):
    with path.open('rb') as f:
        header=json.loads(f.readline());raw=np.frombuffer(f.read(),dtype=np.float64)
    assert header['version']==1 and header['dtype']=='native_float64'
    shape=header['shape'];ks,ke,js,je,iss,ie=header['active_kji']
    active=(shape[0],50,ke-ks+1,je-js+1,ie-iss+1)
    count=int(np.prod(shape));acount=int(np.prod(active))
    assert len(raw)==count+(2*acount if header['rhs_active_only'] else 0)
    state=raw[:count].reshape(shape)
    result=dict(header=header,state=state,active=state[:,:,ks:ke+1,js:je+1,iss:ie+1])
    if header['rhs_active_only']:
        result.update(rhs=raw[count:count+acount].reshape(active),register=raw[count+acount:].reshape(active))
    assert np.isfinite(raw).all()
    return result

def global_fields(data,values):
    blocks=data['header']['blocks'];spacing=np.array(blocks[0]['spacing'])
    origins=np.array([b['origin'] for b in blocks]);minimum=origins.min(axis=0)
    offsets=np.rint((origins-minimum)/spacing).astype(int)
    local=np.array(values.shape[2:][::-1]);shape=offsets.max(axis=0)+local
    result=np.empty((50,*shape[::-1]));owners=np.zeros(tuple(shape[::-1]),dtype=int)
    for m,(i,j,k) in enumerate(offsets):
        nx,ny,nz=local;result[:,k:k+nz,j:j+ny,i:i+nx]=values[m]
        owners[k:k+nz,j:j+ny,i:i+nx]+=1
    assert np.all(owners==1)
    return result,spacing,offsets

def ghost_error(data):
    g,_,offsets=global_fields(data,data['active']);nz,ny,nx=g.shape[1:]
    ks,ke,js,je,iss,ie=data['header']['active_kji'];error=0.
    for m,(i,j,k) in enumerate(offsets):
        z=(k+np.arange(data['state'].shape[2])-ks)%nz
        y=(j+np.arange(data['state'].shape[3])-js)%ny
        x=(i+np.arange(data['state'].shape[4])-iss)%nx
        expected=g[:,z[:,None,None],y[None,:,None],x[None,None,:]]
        error=max(error,float(np.max(abs(data['state'][m]-expected))))
    return error

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[];runs=[]
for dim in [2,3]:
    for order in [2,4,6]:
        case=f'fd{order}-{dim}d';folders=[]
        source=a.fixtures/(case+'-seed-multi.rst')
        text=(a.fixtures/(case+'-seeded-multi')/'used.athinput').read_text().replace('nlim = 3','nlim = 1').replace('tlim = 0.0003','tlim = 0.0001')
        for enabled in [False,True]:
            d=a.output/(case+('-dump' if enabled else '-control'));d.mkdir();folders.append(d)
            inp=d/'used.athinput';inp.write_text(text.replace('formulation = intrinsic_clean',f'formulation = intrinsic_clean\nintrinsic_stage_dump = {str(enabled).lower()}'))
            command=[str(a.binary.resolve()),'-i',str(inp.resolve()),'-r',str(source.resolve())]
            with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            runs.append(dict(command=command,returncode=r.returncode,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),restart_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            assert r.returncode==0,(d,(d/'run.log').read_text())
        baseline=read_restart(sorted((folders[0]/'rst').glob('*.rst'))[-1])['state']
        observed=read_restart(sorted((folders[1]/'rst').glob('*.rst'))[-1])['state']
        assert np.array_equal(baseline,observed)
        paths=sorted(folders[1].glob('intrinsic-stage-*.dat'));assert len(paths)==9
        snapshots={(v['header']['stage'],v['header']['operation']):v for v in map(read_dump,paths)}
        stages=[]
        for stage in [1,2,3]:
            pre=snapshots[stage,'pre-rk'];post=snapshots[stage,'post-rk'];exchange=snapshots[stage,'post-exchange'];h=pre['header']
            assert h['ghosts_valid'] and not post['header']['ghosts_valid'] and exchange['header']['ghosts_valid']
            predicted=h['gam0']*pre['active']+h['gam1']*pre['register']+h['beta_dt']*pre['rhs']
            rk_error=float(np.max(abs(predicted-post['active'])/(1+abs(post['active']))))
            transfer_error=float(np.max(abs(post['active']-exchange['active'])))
            ghosts=max(ghost_error(pre),ghost_error(exchange))
            assert rk_error<=2e-12 and transfer_error==0 and ghosts<=2e-12
            before,spacing,_=global_fields(pre,pre['active']);after,_,_=global_fields(post,post['active'])
            rhs,_,_=global_fields(pre,pre['rhs'])
            rate=h['reduction_rate']*(before[0]*before[1] if h['reduction_profile']=='lapse_scaled' else np.ones_like(before[0]))
            injection,curl_injection=reduction_injection(before,rhs,spacing,order,rate,h['dissipation'])
            assert np.isfinite(injection).all() and np.isfinite(curl_injection).all()
            db=diagnostics(before,spacing,order);da=diagnostics(after,spacing,order)
            delta={n:da[n]-db[n] for n in da};volume=float(np.prod(spacing)*np.prod(before.shape[1:]))
            stages.append(dict(stage=stage,rk_error=rk_error,active_exchange_increment=transfer_error,valid_ghost_error=ghosts,diagnostic_norm_of_RK_difference=norms(delta,volume),semidiscrete_defect=norms(dict(reduction=injection,curl=curl_injection),volume)))
            np.savez(folders[1]/f'signed-stage{stage}.npz',state_increment=after-before,reduction_injection=injection,curl_injection=curl_injection,**delta)
        original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with (folders[1]/'duplicate.log').open('w') as log:
            duplicate=subprocess.run(command,cwd=folders[1],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert duplicate.returncode!=0 and 'cannot exclusively create stage dump' in (folders[1]/'duplicate.log').read_text()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original_hashes.items())
        record=dict(case=case,status='PASS',dump_neutral_bitwise=True,duplicate_rejected=True,stages=stages)
        records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        print(case,'PASS',flush=True)
(a.output/'summary.json').write_text(json.dumps(dict(status='PASS',cases=6,stages=18,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),scope='actual serial multi-block RK and periodic exchange; physical diagnostics assembled offline from global active cells; no stale-ghost derivatives'),indent=2)+'\n')
