#!/usr/bin/env python3
"""Frozen derivative-level constraint lifting, not a native boundary SAT."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('--symbols',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def mm(x,y):return np.einsum("ij,jk->ik",x,y)

def symbols(path):
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if not line.startswith('case '):
            continue
        _, family, radius, label, omega, lo, hi = line.split()
        inverse = np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3)
        normal = next(lines)
        assert normal.startswith('normal ')
        assert next(lines) == 'constraints'
        constraints = np.array([list(map(float,next(lines).split())) for _ in range(8)])
        assert next(lines) == 'matrix 1'
        matrix = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert next(lines) == 'matrix 0.5'
        other = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert np.linalg.norm(matrix-other)/max(1,np.linalg.norm(matrix)) < 1e-11
        tr = np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
        cq = np.zeros((1,11));cq[0,1:7] = tr
        cp = np.zeros((1,11));cp[0,:6] = tr
        q = np.linalg.svd(cq,full_matrices=True)[2][1:].T
        v = np.linalg.svd(cp,full_matrices=True)[2][1:].T
        basis = np.zeros((22,20));basis[:11,:10] = q;basis[11:,10:] = v
        yield dict(source=str(path),family=int(family),radius=float(radius),sample=label,omega=float(omega),
                   lo=float(lo),hi=float(hi),matrix=mm(mm(basis.T,matrix),basis),constraints=mm(constraints,basis),basis=basis,inverse=inverse,normal=np.array(list(map(float,normal.split()[1:]))))
records=[]
for path in a.symbols:
 for s in symbols(path):
  M=s.pop('matrix');C=s.pop('constraints');basis=s.pop('basis');inverse=s.pop('inverse');normal=s.pop('normal');S=-M;I=np.eye(20);lo,hi=s['lo'],s['hi'];Pin=(hi*I-S)/(hi-lo);Pout=I-Pin
  ui,si,_=np.linalg.svd(Pin);uo,so,_=np.linalg.svd(Pout);Vin=ui[:,:10];Vout=uo[:,:10];assert si[9]>1e-8 and si[10]<1e-8 and so[9]>1e-8 and so[10]<1e-8
  J=C[4:];K=mm(J,M+hi*I);Knorm=K/np.linalg.norm(K,axis=1)[:,None];leak=float(np.linalg.norm(mm(Knorm,Vout))/max(1,np.linalg.norm(Knorm)));sing=np.linalg.svd(mm(Knorm,Vin),compute_uv=False)
  _,_,nv=np.linalg.svd(normal[None,:],full_matrices=True);E=nv[1:].T;metric=np.linalg.inv(inverse);gram=mm(mm(E.T,metric),E);E=mm(E,np.linalg.inv(np.linalg.cholesky(gram).T));e,f=E[:,0],E[:,1];TT=np.zeros((2,22))
  for i,t in enumerate([np.outer(e,e)-np.outer(f,f),np.outer(e,f)+np.outer(f,e)]):TT[i,1:7]=[t[0,0],2*t[0,1],2*t[0,2],t[1,1],2*t[1,2],t[2,2]]
  gauge=np.zeros((4,22));gauge[:,7:11]=np.eye(4);G=mm(np.vstack([gauge,TT]),basis);sensor=np.vstack([K,G]);raw=mm(sensor,Vin);scale=np.linalg.norm(raw,axis=1);equilibrated=raw/scale[:,None];sv=np.linalg.svd(equilibrated,compute_uv=False);rank=int(sum(sv>1e-10));tests=[]
  if rank==10:
   lift=mm(Vin,np.linalg.solve(equilibrated,np.diag(1/scale)));targets=[np.eye(4)[i] for i in range(4)]+[np.array([1.,-2.,.3,.7]),np.array([-.4,.2,1.,-1.])]
   for target in targets:
    request=np.r_[-target,np.zeros(6)];delta=np.einsum('ij,j->i',lift,request);res=float(np.linalg.norm(np.einsum('ij,j->i',K,delta)+target)/max(1,np.linalg.norm(target)));out=float(np.linalg.norm(np.einsum('ij,j->i',Pout,delta))/max(1,np.linalg.norm(delta)));free=float(np.linalg.norm(np.einsum('ij,j->i',G,delta))/max(1,np.linalg.norm(delta)));tests.append(dict(target=target.tolist(),constraint_residual=res,outgoing_change=out,selected_free_sensor_change=free,correction_norm=float(np.linalg.norm(delta)),passed=res<=1e-9 and out<=1e-9 and free<=1e-9))
   gain=float(np.linalg.norm(lift[:,:4],2))
  else:gain=None
  passed=leak<=1e-10 and sum(sing>1e-9)==4 and rank==10 and all(x['passed'] for x in tests)
  records.append(dict(**s,outgoing_constraint_operator_leakage=leak,incoming_constraint_rank=int(sum(sing>1e-9)),incoming_constraint_singular_values=sing.tolist(),incoming_full_sensor_rank=rank,raw_condition=float(np.linalg.cond(raw)),equilibrated_condition=float(np.linalg.cond(equilibrated)),unit_residual_lift_amplification=gain,tests=tests,passed=bool(passed)))
r=dict(passed=all(x['passed'] for x in records),records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.symbols},scope='Frozen derivative-level principal lifting only. It cancels four constraint residuals while holding six chosen coordinate-gauge/TT sensors, not full nonlinear gauge/radiation data. No actual SBP SAT or reflection/injection validation.')
a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(dict(passed=r['passed'],states=len(records),ranks=[x['incoming_full_sensor_rank'] for x in records],max_gain=max(x['unit_residual_lift_amplification'] or 0 for x in records)))
