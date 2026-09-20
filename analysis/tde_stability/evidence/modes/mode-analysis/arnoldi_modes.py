#!/usr/bin/env python3
"""Bounded Arnoldi proof of concept, central finite difference full RK map."""
import json,sys,time
import numpy as np
from mode_operator import *
steps=int(sys.argv[1]) if len(sys.argv)>1 else 40
m=int(sys.argv[2]) if len(sys.argv)>2 else 40
epsilon=float(sys.argv[3]) if len(sys.argv)>3 else 1e-6
label=f'arnoldi_s{steps}_m{m}_eps{epsilon:g}'
overrides=tuple(sys.argv[4:])
if overrides:label+='_'+ '_'.join(x.replace('/','_').replace('=','') for x in overrides)
dt=float(os.environ.get('MODE_DT',str(DT)))
if os.environ.get('MODE_TAG'):label+='_'+os.environ['MODE_TAG']
op=Operator(label,overrides=overrides,binary=os.environ.get('MODE_BINARY'),dt=dt,input_file=os.environ.get('MODE_INPUT'))
seed=np.fromfile(ROOT/'late_seed_500M.bin').reshape(SHAPE)
q=seed.ravel()/np.linalg.norm(seed)
Q=np.empty((q.size,m+1));Q[:,0]=q
H=np.zeros((m+1,m));history=[]
for k in range(m):
    start=time.monotonic()
    w=op.response(Q[:,k].reshape(SHAPE),steps=steps,epsilon=epsilon).ravel()
    for _ in range(2):
        hh=Q[:,:k+1].T@w
        H[:k+1,k]+=hh
        w-=Q[:,:k+1]@hh
    H[k+1,k]=np.linalg.norm(w)
    if H[k+1,k]==0:raise RuntimeError('Exact breakdown')
    Q[:,k+1]=w/H[k+1,k]
    eig,Y=np.linalg.eig(H[:k+1,:k+1])
    order=np.argsort(-abs(eig))[:min(6,k+1)]
    row={'k':k+1,'seconds':time.monotonic()-start,'ritz':[]}
    for j in order:
        ev=eig[j];bound=abs(H[k+1,k]*Y[-1,j])
        row['ritz'].append(dict(real=float(ev.real),imag=float(ev.imag),modulus=float(abs(ev)),gamma=float(np.log(abs(ev))/(steps*dt)),omega=float(np.angle(ev)/(steps*dt)),arnoldi_residual=float(bound),relative_residual=float(bound/max(abs(ev),1e-99))))
    history.append(row)
    print(json.dumps(row),flush=True)
    (ROOT/(label+'-progress.json')).write_text(json.dumps(history,indent=2)+'\n')
np.savez_compressed(ROOT/(label+'-krylov.npz'),Q=Q,H=H)
# Save a real eigenvector for every nearly real leading Ritz mode; complex pairs
# must be validated through their real two-dimensional invariant subspace.
report={'steps':steps,'interval_M':steps*dt,'krylov_dimension':m,'history':history,'modes':[]}
for j in np.argsort(-abs(eig))[:6]:
    ev=eig[j];vec=Q[:,:m]@Y[:,j]
    mode=dict(real=float(ev.real),imag=float(ev.imag),modulus=float(abs(ev)),gamma=float(np.log(abs(ev))/(steps*dt)))
    if abs(ev.imag)<1e-10:
        v=vec.real.reshape(SHAPE);v/=np.max(abs(v))
        out=op.response(v,steps=steps,epsilon=epsilon,label=f'mode{len(report["modes"])}_check')
        mode['direct_eigen_residual']=discrepancy(ev.real*v,out)
        mode['profile']=profile(v)
        path=ROOT/(label+f'-mode{len(report["modes"])}.bin');v.tofile(path)
        mode['file']=str(path)
    report['modes'].append(mode)
(ROOT/(label+'-results.json')).write_text(json.dumps(report,indent=2)+'\n')
