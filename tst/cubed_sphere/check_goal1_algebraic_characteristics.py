#!/usr/bin/env python3
"""Audit actual22-variable principal algebra, constrained20 and RK projection."""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
assert (out/'protocol-v1.json').exists()
raw=Path('docs/validation/goal1-minkowski-20261018/oblique-symbol.txt')
meta=json.loads(Path('docs/validation/goal1-minkowski-20261018/oblique-review-classified.json').read_text())['records']
def mm(x,y):return np.einsum('ij,jk->ik',x,y)
def RK(M,nu):
    Z=1j*nu*M;Z2=mm(Z,Z);return np.eye(len(M))+Z+Z2/2+mm(Z2,Z)/6
records=[];index=-1
lines=iter(raw.read_text().splitlines())
for line in lines:
    if not line.startswith('case '):continue
    index+=1;_,family,radius,label,omega,lo,hi=line.split();lo,hi=map(float,[lo,hi]);beta=-(lo+hi)/2
    inv=np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3);metric=np.linalg.inv(inv)
    normal=next(lines);assert next(lines)=='constraints'
    for _ in range(8):next(lines)
    assert next(lines)=='matrix 1';M=np.array([list(map(float,next(lines).split())) for _ in range(22)])
    assert next(lines)=='matrix 0.5';other=np.array([list(map(float,next(lines).split())) for _ in range(22)])
    assert np.linalg.norm(M-other)/max(1,np.linalg.norm(M))<1e-11
    trace=np.array([inv[0,0],2*inv[0,1],2*inv[0,2],inv[1,1],2*inv[1,2],inv[2,2]])
    packed=np.array([metric[0,0],metric[0,1],metric[0,2],metric[1,1],metric[1,2],metric[2,2]])
    C=np.zeros((2,22));C[0,1:7]=trace;C[1,11:17]=trace
    _,_,V=np.linalg.svd(C,full_matrices=True);N=V[:2].T;B=V[2:].T
    P=np.eye(22);P[1:7,1:7]-=np.outer(packed,trace)/3;P[11:17,11:17]-=np.outer(packed,trace)/3
    projector=float(np.linalg.norm(mm(P,P)-P));leak=float(np.linalg.norm(mm(mm(C,M),B))/max(1,np.linalg.norm(M)))
    Q=mm(mm(N.T,M),N);J=Q-beta*np.eye(2);nil=float(np.linalg.norm(mm(J,J))/max(1,np.linalg.norm(Q)**2));rank=int(np.linalg.matrix_rank(J,tol=1e-9))
    rec={'case':index,'pilot_case':meta[index]['pilot_case'],'Omega':float(omega),'projector_identity_error':projector,
        'algebraic_kernel_leak':leak,'algebraic_normal_speed':-beta,'normal_quotient':Q.tolist(),
        'normal_nilpotent_norm':float(np.linalg.norm(J)),'normal_nilpotency_relative_error':nil,
        'normal_nilpotent_rank':rank,'unprojected_normal_sector_defective':bool(rank==1 and nil<=1e-10),'RK':[]}
    if meta[index]['pilot_case']['kind']=='evolved_native_t0.2':
        A=mm(mm(B.T,M),B);I=np.eye(20);S=-A;minus=(hi*I-S)/(hi-lo);plus=(S-lo*I)/(hi-lo)
        W=mm(minus.T,minus)+mm(plus.T,plus);L=np.linalg.cholesky(W);speed=max(abs(lo),abs(hi))
        for nu in [.1,.5,1.,math.sqrt(3),1.9]:
            full=mm(P,mm(RK(M,nu/speed),B));target=mm(B,RK(A,nu/speed));error=float(np.linalg.norm(full-target)/max(1,np.linalg.norm(target)))
            R=mm(mm(B.T,full),np.eye(20));E=mm(mm(R.conj().T,W),R)-W
            left=np.linalg.solve(L,E);Z=np.linalg.solve(L.conj(),left.T).T
            positive=float(np.linalg.eigvalsh((Z+Z.conj().T)/2)[-1])
            rec['RK'].append({'scaled_CFL':nu,'projected_reconstruction_error':error,
                'maximum_weighted_energy_change':positive,'deliberately_unsafe':nu>math.sqrt(3)+1e-12,
                'passed':bool(error<=1e-10 and (positive>1e-5 if nu>math.sqrt(3)+1e-12 else positive<=1e-10))})
    rec['passed']=bool(projector<=1e-10 and leak<=1e-10 and nil<=1e-10 and rank==1 and all(r['passed'] for r in rec['RK']))
    records.append(rec)
bridge=Path('src/z4c/hyperboloidal/athenak_bridge.hpp');probe=Path('src/mesh/cubed_sphere/conformal_probe.cpp')
frozen_aux='q.vB_d(m,a,k,j,i) = 0;' in bridge.read_text() and 'n<Z::I_Z4C_BX' in probe.read_text()
report={'passed':all(r['passed'] for r in records) and frozen_aux,'records':records,'raw22_cases':len(records),
    'projected_RK_cases':sum(bool(r['RK']) for r in records),'frozen_auxiliary_slots':3,'frozen_auxiliary_zero_rhs_verified':frozen_aux,
    'interpretation':'The unprojected algebraic normal quotient has a genuine2x2 Jordan block at the advective speed. Strong characteristic completeness is established only on the constrained20-field manifold, with three additional frozen auxiliary slots. Existing final-step principal projection removes these algebraic modes for invariant-manifold Fourier data. This does not establish stability of the complete nonlinear mapped RK/source/ghost method or a uniform scri chart.',
    'gauge_change_required':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [raw,bridge,probe]}}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['passed','raw22_cases','projected_RK_cases','frozen_auxiliary_slots','gauge_change_required']},indent=2));print('failures',[(r['case'],r['normal_nilpotent_rank'],r['normal_nilpotency_relative_error'],[t for t in r['RK'] if not t['passed']]) for r in records if not r['passed']])
