#!/usr/bin/env python3
"""Rank/reflection only for the actual finite-radius native characteristic system."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
assert (out/'protocol-v1.json').exists()
raw=Path('docs/validation/goal1-minkowski-20261018/oblique-symbol.txt')
meta=json.loads(Path('docs/validation/goal1-minkowski-20261018/oblique-review-classified.json').read_text())['records']
def mm(a,b):return np.einsum('ij,jk->ik',a,b)
def rank(a):
    s=np.linalg.svd(a,compute_uv=False)
    return int(sum(s>1e-10*s[0])),s
def scaled_rows(a):
    norm=np.linalg.norm(a,axis=1);assert min(norm)>0
    return a/norm[:,None]
def gram_tt(g,inv,n):
    unit=inv.dot(n)/np.sqrt(n.dot(inv.dot(n)))
    candidates=[]
    for axis in np.eye(3):
        t=axis-unit*(unit.dot(g.dot(axis)))
        candidates.append((t.dot(g.dot(t)),t))
    t=max(candidates,key=lambda x:x[0])[1];t/=np.sqrt(t.dot(g.dot(t)))
    candidates=[]
    for axis in np.eye(3):
        s=axis-unit*(unit.dot(g.dot(axis)))-t*t.dot(g.dot(axis))
        candidates.append((s.dot(g.dot(s)),s))
    s=max(candidates,key=lambda x:x[0])[1];s/=np.sqrt(s.dot(g.dot(s)))
    return [(np.outer(t,t)-np.outer(s,s))/np.sqrt(2),(np.outer(t,s)+np.outer(s,t))/np.sqrt(2)]
records=[];index=-1;lines=iter(raw.read_text().splitlines())
for line in lines:
    if not line.startswith('case '):continue
    index+=1;_,family,radius,label,omega,lo,hi=line.split();lo,hi=map(float,[lo,hi]);beta=-(lo+hi)/2;c=(hi-lo)/2
    inv=np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3);g=np.linalg.inv(inv)
    n=np.array(list(map(float,next(lines).split()[1:])));assert next(lines)=='constraints'
    C=np.array([list(map(float,next(lines).split())) for _ in range(8)])
    assert next(lines)=='matrix 1';M=np.array([list(map(float,next(lines).split())) for _ in range(22)])
    assert next(lines)=='matrix 0.5';other=np.array([list(map(float,next(lines).split())) for _ in range(22)])
    assert np.linalg.norm(M-other)/max(1,np.linalg.norm(M))<1e-11
    trace=np.array([inv[0,0],2*inv[0,1],2*inv[0,2],inv[1,1],2*inv[1,2],inv[2,2]])
    cq=np.zeros((1,11));cq[0,1:7]=trace;cp=np.zeros((1,11));cp[0,:6]=trace
    qb=np.linalg.svd(cq,full_matrices=True)[2][1:].T;pb=np.linalg.svd(cp,full_matrices=True)[2][1:].T
    basis=np.zeros((22,20));basis[:11,:10]=qb;basis[11:,10:]=pb
    M=mm(mm(basis.T,M),basis);C=mm(C,basis);A=M[:10,:10];B=M[:10,10:]
    P_in=np.linalg.solve(B,(beta+c)*np.eye(10)-A);P_out=np.linalg.solve(B,(beta-c)*np.eye(10)-A)
    J=C[4:];Ji=J[:,:10]+mm(J[:,10:],P_in);Jo=J[:,:10]+mm(J[:,10:],P_out)
    Fraw=np.zeros((6,11));Fraw[0,7]=1;Fraw[1:4,8:11]=np.eye(3)
    for row,T in enumerate(gram_tt(g,inv,n),4):Fraw[row,1:7]=[T[0,0],2*T[0,1],2*T[0,2],T[1,1],2*T[1,2],T[2,2]]
    F=mm(Fraw,qb);K=scaled_rows(np.vstack([Ji,F]));target=np.vstack([-Jo,np.zeros((6,10))])/np.linalg.norm(np.vstack([Ji,F]),axis=1)[:,None]
    kr,ks=rank(K);rec={'case':index,'pilot_case':meta[index]['pilot_case'],'Omega':float(omega),'speed_in':lo,'speed_out':hi,'condition_rank':kr,'condition_number':float(ks[0]/ks[-1])}
    # Compare physically incorrect alternatives without concealing rank loss.
    G=np.eye(10)[:6];rec['six_spatial_metric_condition_rank']=rank(scaled_rows(np.vstack([Ji,G])))[0]
    rec['momentum_only_lift_rank']=rank(scaled_rows(np.vstack([J[:,10:],mm(F,B)])))[0]
    if kr==10:
        R=np.linalg.solve(K,target);error=float(np.linalg.norm(mm(K,R)-target)/max(1,np.linalg.norm(target)))
        physical=float(np.linalg.norm(mm(Ji,R)+Jo)/max(1,np.linalg.norm(Jo)))
        free=float(np.linalg.norm(mm(F,R))/max(1,np.linalg.norm(F)*np.linalg.norm(R)))
        # Use Pi=+/-c q_n: canonical branch energy has common 2c^2 factor.
        flux=abs(lo)*mm(R.T,R)-hi*np.eye(10)
        rec.update(reflection_identity_error=error,constraint_residual=physical,free_condition_residual=free,
          reflection_singular_values=np.linalg.svd(R,compute_uv=False).tolist(),reflection_eigenvalue_abs=np.sort(np.abs(np.linalg.eigvals(R))).tolist(),
          canonical_flux_maximum=float(np.linalg.eigvalsh((flux+flux.T)/2)[-1]),
          canonical_flux_relative_maximum=float(np.linalg.eigvalsh((flux+flux.T)/2)[-1]/max(1,abs(lo),abs(hi))),
          incoming_flux_applicable=bool(lo<0<hi),rank_identity_pass=bool(max(error,physical,free)<=1e-10))
    else:rec['rank_identity_pass']=False
    records.append(rec)
report={'scope':__doc__,'passed_rank_and_identities':all(r['rank_identity_pass'] for r in records),'records':records,
    'states':len(records),'evolved_states':sum(r['pilot_case']['kind']=='evolved_native_t0.2' for r in records),
    'positive_canonical_flux_cases':[r['case'] for r in records if r.get('incoming_flux_applicable') and r['canonical_flux_relative_maximum']>1e-10],
    'native_boundary_adopted':False,'exact_scri_prescription':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in report.items() if k!='records'})
print('condition ranks',sorted(set(r['condition_rank'] for r in records)), 'spatial metric ranks',sorted(set(r['six_spatial_metric_condition_rank'] for r in records)), 'momentum-only ranks',sorted(set(r['momentum_only_lift_rank'] for r in records)))
