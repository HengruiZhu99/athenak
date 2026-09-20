"""Bounded alternate-boundary checks using independent production Schur model.
These are rank checks near a known root, not a Kreiss or nonlinear stability proof.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import production_halfspace as p

def boundary(lam,k,cfg,mode,U,D,T):
    B=p.boundary(U,D,cfg)
    if mode=='zero_rate':return B
    if mode=='dirichlet_q':return U[p.QP]
    H=np.array([[U[1],U[3],U[4]],[U[3],U[2],U[5]],[U[4],U[5],-U[1]-U[2]]])
    DH=np.array([[D[1],D[3],D[4]],[D[3],D[2],D[5]],[D[4],D[5],-D[1]-D[2]]])
    Q=U[13:16]-DH[:,0]-1j*k*H[:,1]
    DQ=D[13:16]-DH[:,0]@T-1j*k*DH[:,1]
    kap=cfg.kappa*cfg.alpha if mode=='physical_damped' else 0.
    velocity=cfg.alpha*np.sqrt(cfg.chi)-cfg.beta_n
    B[2]=(lam+kap)*U[7]+velocity*D[7]
    FQ=(lam+kap)*Q+velocity*DQ
    B[3]=FQ[0];B[5]=FQ[1];B[7]=FQ[2]
    return B

def assess(lam,k,cfg,mode):
    U,D,T,res=p.schur(lam,k,cfg);B=boundary(lam,k,cfg,mode,U,D,T)
    _,R=np.linalg.qr(U);BB=B@np.linalg.inv(R);BB/=np.linalg.norm(BB,axis=1)[:,None]
    sv=np.linalg.svd(BB,compute_uv=False)
    return float(sv[-1])

if __name__=='__main__':
    cfg=p.Config();k=.1;out=[]
    for mode in ['zero_rate','dirichlet_q','physical','physical_damped']:
      # Narrow known-root interval first, then a finite coarse real-axis check.
      m=minimize_scalar(lambda x:assess(complex(x),k,cfg,mode),bounds=(.015,.035),method='bounded',options={'xatol':1e-13})
      rows=[{'lambda':float(x),'sigma':assess(complex(x),k,cfg,mode)} for x in np.geomspace(.001,.2,80)]
      out.append(dict(mode=mode,narrow_root_search={'lambda':float(m.x),'sigma':float(m.fun)},coarse_min=min(rows,key=lambda d:d['sigma']),data=rows))
      print(out[-1]['mode'],out[-1]['narrow_root_search'],out[-1]['coarse_min'],flush=True)
    Path('continuum-controls.json').write_text(json.dumps(out,indent=2)+'\n')
