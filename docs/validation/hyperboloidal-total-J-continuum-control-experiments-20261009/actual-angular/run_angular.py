"""Staged local continuum angular extraction; no radial operator/evolution."""
from pathlib import Path
import hashlib
import io
import json
import math
import os
import subprocess
import time
import warnings
import numpy as np
warnings.filterwarnings('error',category=RuntimeWarning)

P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=json.loads((P/'tolerances.json').read_text());plan=json.loads((P/'angular-plan.json').read_text())
source=sha(P/'bridge.cpp');exe=P/'bridge-release'
for mode in ('release','debug'):
    local=json.loads((P/('local-'+mode+'-latest.json')).read_text());assert local['passed_all_local_gates']
    receipt=Path(local['attempt'])/'receipt.json';assert sha(receipt)==local['receipt_sha256']
    assert json.loads(receipt.read_text())['bridge_source_sha256']==source
build=json.loads((P/'build-release-latest.json').read_text());assert sha(exe)==build['executable_sha256']
groups=[];lines=[];cursor=0
def append(J,r,family,m,phase,rot,angles):
    global cursor
    nch=(8,16,20)[J];start=cursor
    for direction in angles:
        x=[r*a for a in direction]
        for c in range(nch):
            for w in plan['radial_W_jets']:
                values=[J,m,c,phase,rot,*x,*w]
                lines.append(' '.join(format(a,'.17g') if isinstance(a,float) else str(a) for a in values)+'\n')
                cursor+=1
    groups.append({'J':J,'r':r,'family':family,'m':m,'phase':phase,'rotation':rot,'directions':angles,'start':start,'stop':cursor,'shape':[len(angles),nch,3,49]})
for r in plan['radii']:
    for J in range(3):
        append(J,r,'fit',0,0,0,plan['fit_directions'])
        append(J,r,'holdout',0,0,0,plan['heldout_directions'])
        append(J,r,'allm0',0,2,0,plan['heldout_directions'])
        for m in plan['independent_m_by_J'][str(J)]:
            for phase in (0,1):append(J,r,'independent_m',m,phase,0,plan['heldout_directions'])
        for rot in (0,1):append(J,r,'rotation',0,0,rot,plan['rotation_test_directions'])
payload=''.join(lines);input_bytes=payload.encode();del lines
query_hash=hashlib.sha256(input_bytes).hexdigest()
query_meta={'groups':groups,'rows':cursor,'columns':49,'query_sha256':query_hash,'bridge_source_sha256':source,'plan_sha256':sha(P/'angular-plan.json')}
metadata=P/'angular-queries.json'
if metadata.exists():assert json.loads(metadata.read_text())==query_meta
else:metadata.write_text(json.dumps(query_meta,indent=2)+'\n')
out=P/'angular-actions.txt';runtime=P/'angular-runtime.json'
if out.exists():
    saved=json.loads(runtime.read_text());assert saved['query_sha256']==query_hash and saved['source_sha256']==source and saved['output_sha256']==sha(out)
else:
    started=time.monotonic();run=subprocess.run([str(exe),'--batch'],input=payload,text=True,capture_output=True)
    out.write_text(run.stdout);(P/'angular-stderr').write_text(run.stderr)
    runtime.write_text(json.dumps({'command':[str(exe),'--batch'],'exit_code':run.returncode,'seconds':time.monotonic()-started,'rows':cursor,'query_sha256':query_hash,'output_sha256':sha(out),'source_sha256':source,'executable_sha256':sha(exe)},indent=2)+'\n')
    assert run.returncode==0,run.stderr
del payload,input_bytes
raw=np.fromstring(out.read_text(),sep=' ');assert raw.size==cursor*49,(raw.size,cursor)
raw=raw.reshape(cursor,49);assert np.isfinite(raw).all()
np.savez_compressed(P/'angular-actions.npz',actions=raw)
def data(g):return raw[g['start']:g['stop']].reshape(g['shape'])
def matrices(g):
    a=data(g);n=a.shape[1]
    B=a[:,:,0,:22].transpose(0,2,1).reshape(-1,n)
    F=a[:,:,:,22:44].transpose(0,3,1,2).reshape(-1,3*n)
    return B,F
def errors(actual,predicted):
    d=actual-predicted
    return {'aggregate':float(np.linalg.norm(d)/max(1,np.linalg.norm(actual))),
            'per_action_max':float(np.max(np.linalg.norm(d,axis=0)/np.maximum(1,np.linalg.norm(actual,axis=0)))),
            'absolute_L2':float(np.linalg.norm(d)),'actual_L2':float(np.linalg.norm(actual)),
            'absolute_max':float(np.max(np.abs(d)))}
def lookup(J,r,family,m=0,phase=0,rotation=0):
    return next(g for g in groups if (g['J'],g['r'],g['family'],g['m'],g['phase'],g['rotation'])==(J,r,family,m,phase,rotation))
axis=np.array(plan['rotation_axis']);c=math.cos(plan['rotation_angle']);s=math.sin(plan['rotation_angle']);x,y,z=axis
R=c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
assert np.linalg.norm(np.einsum('ki,kj->ij',R,R,optimize=False)-np.eye(3))<1e-14 and abs(np.linalg.det(R)-1)<1e-14
ti=np.array([0,0,0,1,1,2]);tj=np.array([0,1,2,1,2,2])
def rotate_raw(a):
    b=a.copy()
    for start in (1,8):
        tensor=np.zeros(a.shape[:-1]+(3,3));tensor[...,ti,tj]=a[...,start:start+6];tensor[...,tj,ti]=a[...,start:start+6]
        tensor=np.einsum('ik,...kl,jl->...ij',R,tensor,R)
        b[...,start:start+6]=tensor[...,ti,tj]
    for start in (14,19):b[...,start:start+3]=np.einsum('ij,...j->...i',R,a[...,start:start+3])
    return b
results=[];coefficients={};worst={'fit':0.,'holdout':0.,'independent_m':0.,'rotation':0.,'m0_allm':0.}
for r in plan['radii']:
    for J in range(3):
        g=lookup(J,r,'fit');B,F=matrices(g);column=np.linalg.norm(B,axis=0);assert (column>0).all()
        Bs=B/column;u,singular,vh=np.linalg.svd(Bs,full_matrices=False)
        cond=float(singular[0]/singular[-1]);rank=int(np.sum(singular>singular[0]/T['scaled_basis_fit_condition_max']))
        projected=np.einsum('ki,kj->ij',u,F,optimize=False)/singular[:,None]
        A=np.einsum('ki,kj->ij',vh,projected,optimize=False)/column[:,None]
        fit_error=errors(F,np.einsum('ik,kj->ij',B,A,optimize=False));worst['fit']=max(worst['fit'],fit_error['aggregate'],fit_error['per_action_max'])
        raw_s=np.linalg.svd(B,compute_uv=False)
        row={'J':J,'r':r,'Omega':float(data(g)[0,0,0,48]),'amplitudes':B.shape[1],
             'raw_condition':float(raw_s[0]/raw_s[-1]),'scaled_condition':cond,'scaled_rank':rank,
             'column_norms':column.tolist(),'scaled_singular_values':singular.tolist(),'fit_error':fit_error}
        coefficients['J%d_r%.17g_B0'%(J,r)]=A[:,0::3]
        coefficients['J%d_r%.17g_B1'%(J,r)]=A[:,1::3]
        coefficients['J%d_r%.17g_B2'%(J,r)]=A[:,2::3]
        bh,fh=matrices(lookup(J,r,'holdout'));e=errors(fh,np.einsum('ik,kj->ij',bh,A,optimize=False));row['heldout_angles_error']=e;worst['holdout']=max(worst['holdout'],e['aggregate'],e['per_action_max'])
        ba,fa=matrices(lookup(J,r,'allm0',phase=2));e=errors(np.column_stack((bh,fh)),np.column_stack((ba,fa)));row['primary_vs_allm_m0_error']=e;worst['m0_allm']=max(worst['m0_allm'],e['aggregate'],e['per_action_max'])
        row['independent_m_errors']=[]
        for m in plan['independent_m_by_J'][str(J)]:
            br,fr=matrices(lookup(J,r,'independent_m',m,0));bi,fi=matrices(lookup(J,r,'independent_m',m,1))
            bm,fm=br+1j*bi,fr+1j*fi;e=errors(fm,np.einsum('ik,kj->ij',bm,A,optimize=False));row['independent_m_errors'].append({'m':m,**e});worst['independent_m']=max(worst['independent_m'],e['aggregate'],e['per_action_max'])
        base=data(lookup(J,r,'rotation',rotation=0));rot=data(lookup(J,r,'rotation',rotation=1))
        er=errors(rot[:,:,:,22:44].reshape(-1,22).T,rotate_raw(base[:,:,:,22:44]).reshape(-1,22).T)
        eu=errors(rot[:,:,:,:22].reshape(-1,22).T,rotate_raw(base[:,:,:,:22]).reshape(-1,22).T)
        row['rotation_rhs_error']=er;row['rotation_input_error']=eu;worst['rotation']=max(worst['rotation'],er['aggregate'],er['per_action_max'],eu['aggregate'],eu['per_action_max'])
        row['passed']=bool(cond<=T['scaled_basis_fit_condition_max'] and rank==B.shape[1] and fit_error['aggregate']<=T['angular_fit_relative_residual_max'] and fit_error['per_action_max']<=T['angular_fit_relative_residual_max'] and max(e['aggregate'] for e in [row['heldout_angles_error'],er,eu]+row['independent_m_errors'])<=T['heldout_angles_m_rotation_relative_error_max'] and max(e['per_action_max'] for e in [row['heldout_angles_error'],er,eu]+row['independent_m_errors'])<=T['heldout_angles_m_rotation_relative_error_max'])
        results.append(row)
norm_u=np.maximum(1,np.linalg.norm(raw[:,:22],axis=1));norm_f=np.maximum(1,np.linalg.norm(raw[:,22:44],axis=1))
input_normal=float(np.max(np.abs(raw[:,44:46])/norm_u[:,None]));output_normal=float(np.max(np.abs(raw[:,46:48])/norm_f[:,None]))
np.savez_compressed(P/'angular-coefficients.npz',**coefficients)
report={'scope':'Local continuum C0/spatial-norm angular coefficient action only; no radial operator/boundary/evolution/stability result.',
        'query_rows':cursor,'radial_points':len(plan['radii']),'J_values':[0,1,2],'input_order':'frozen basis-data channel_layouts','raw_output_order':'native22 in API-CONTRACT',
        'radial_derivatives':'B0 W+B1 W_rho+B2 W_rhorho; each action contains all actual analytic coefficient jets',
        'runtime':json.loads(runtime.read_text()),'worst_scaled_errors':worst,'input_normal_scaled':input_normal,'output_normal_scaled':output_normal,
        'maximum_scaled_condition':max(r['scaled_condition'] for r in results),'maximum_raw_condition':max(r['raw_condition'] for r in results),
        'results':results,'passed_all_angular_gates':bool(all(r['passed'] for r in results) and input_normal<=T['input_algebraic_normal_scaled_max'] and output_normal<=T['output_algebraic_normal_scaled_max'] and worst['m0_allm']<=T['primary_vs_all_m_m0_basis_scaled_max']),
        'source_sha256':source,'script_sha256':sha(__file__),'plan_sha256':sha(P/'angular-plan.json'),'coefficients_sha256':sha(P/'angular-coefficients.npz'),'actions_npz_sha256':sha(P/'angular-actions.npz')}
k=1
while (P/('angular-analysis-%03d.json'%k)).exists():k+=1
dest=P/('angular-analysis-%03d.json'%k);dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('results',)},indent=2))
assert report['passed_all_angular_gates'], str(dest)
