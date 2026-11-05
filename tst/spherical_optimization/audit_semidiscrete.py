#!/usr/bin/env python3
"""Frozen semidiscrete Jacobians: spectra and RK4 versus exact linear flow.

Endpoint alpha and beta are eliminated with the prescribed clock/null chart.
Coordinates are scaled by local initial magnitudes, with a unit scale for fields
that vanish initially. Singular norms are therefore disclosed diagnostic norms,
not a continuum energy estimate or nonlinear stability proof.
"""
import argparse,json,time,hashlib,tarfile,subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import eig,svdvals
from scipy.linalg.blas import dgemm

def product(a,b):
 return dgemm(1.,a,b)

def exponential(a):
 # Scaling/Taylor/squaring, avoiding this platform NumPy matmul FP flags.
 scale=max(0,int(np.ceil(np.log2(max(1.,np.linalg.norm(a,1)/.5)))))
 b=a/(2.**scale);out=np.eye(len(a));term=out.copy()
 for j in range(1,80):
  term=product(term,b)/j;out+=term
  if np.linalg.norm(term,1)<1e-17*np.linalg.norm(out,1):break
 else:raise ArithmeticError("matrix exponential Taylor did not converge")
 for _ in range(scale):out=product(out,out)
 if not np.isfinite(out).all():raise FloatingPointError("nonfinite exponential")
 return out
from kernel import Kernel
from evolution import ChartEvolution,DerivativeBoundaryEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n',type=int,default=32);ap.add_argument('--reference-step',type=float,default=5e-4);ap.add_argument('--penalty',type=float,default=.6510416666660156);ap.add_argument('--derivative-boundary',action='store_true');ap.add_argument('--run',type=Path,help='Use the complete configuration of an immutable run');ap.add_argument('--state',type=Path);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cfg={}
 if a.run:
  cfg=json.loads((a.run/'manifest.json').read_text())['config'];a.n=cfg['n'];a.penalty=cfg['penalty'];a.reference_step=cfg.get('reference_step',5e-4);a.derivative_boundary=cfg.get('derivative_boundary',False)
 root=Path(__file__).resolve().parents[2]
 paths=sorted([*root.glob('src/z4c/hyperboloidal/*.hpp'),*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cpp'),Path(__file__).parent/'CMakeLists.txt',root/'tst/elliptic_shift_1d/numerics.py'])
 identity=dict(input_state_sha256=hashlib.sha256(a.state.read_bytes()).hexdigest() if a.state else None,run_configuration=cfg,config={key:str(v) if isinstance(v,Path) else v for key,v in vars(a).items()},commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest())
 (a.output/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
 with tarfile.open(a.output/'source.tar.gz','w:gz') as archive:
  for p in paths:archive.add(p,arcname=str(p.relative_to(root)))
 k=Kernel(a.library,cfg.get('mode',1),cfg.get('radius',20.),cfg.get('slicing_ratio',2.),cfg.get('separated',True));e=(DerivativeBoundaryEvolution if a.derivative_boundary else ChartEvolution)(k,a.n,grading=cfg.get('grading',0.),penalty=a.penalty,ko=cfg.get('ko',.5),floors=cfg.get('floors',False),normal_policy=cfg.get('normal_policy','analytic'),reference_jets=cfg.get('reference_jets',True),reference_step=a.reference_step,reference_tail=cfg.get('reference_tail','numerical'),grid_map=cfg.get('grid_map','quadratic'))
 q=e.initial.copy() if not a.state else np.load(a.state)['independent_state'].copy()
 if a.state and not np.array_equal(np.load(a.state)['r'],e.r):raise ValueError('saved state grid differs from the selected configuration')
 active=np.ones_like(q,dtype=bool);active[-1,6:8]=False
 if not a.derivative_boundary:active[-1,9]=False
 scales=np.maximum(abs(q),1e-3);scales[:,[2,3,4,5,8,9]]=np.maximum(abs(q[:,[2,3,4,5,8,9]]),1.)
 base=e.rhs(q);speed=e.last['speed_over_dr'];matrices=[]
 for eps in (1e-5,3e-6):
  J=np.empty((active.sum(),active.sum()))
  for col,(i,j) in enumerate(np.argwhere(active)):
   qp=q.copy();qm=q.copy();qp[i,j]+=eps*scales[i,j];qm[i,j]-=eps*scales[i,j]
   J[:,col]=((e.rhs(qp)-e.rhs(qm))/(2*eps)/scales)[active]
  matrices.append(J)
 J=matrices[-1];delta=matrices[1]-matrices[0]
 values,vectors=eig(J);ordering=np.argsort(values.real)[::-1];values=values[ordering];vectors=vectors[:,ordering]
 v=vectors[:,0];residual=np.linalg.norm(np.einsum("ij,j->i",J,v)-values[0]*v)/max(1,np.linalg.norm(J)*np.linalg.norm(v))
 mode=np.zeros_like(q,dtype=complex);mode[active]=v*scales[active]
 I=np.eye(len(J));rows=[]
 for cfl in (.1,.2,.3,.4,.5):
  dt=cfl/speed;A=dt*J;A2=product(A,A);rk=I+A+A2/2+product(A2,A)/6+product(A2,A2)/24
  exact=exponential(A);z=dt*values;p=1+z+z*z/2+z**3/6+z**4/24
  rows.append(dict(C_char=cfl,dt=dt,dt_M=3*dt,rk_spectral_radius=float(max(abs(p))),exact_spectral_radius=float(np.exp(dt*max(values.real))),rk_scaled_norm=float(svdvals(rk)[0]),exact_scaled_norm=float(svdvals(exact)[0]),rk_minus_exact_scaled_norm=float(svdvals(rk-exact)[0])))
 np.savez_compressed(a.output/'linearization.npz',r=e.r,state=q,scales=scales,active=active,J=J,J_difference=delta,eigenvalues=values,leading_mode=mode,base_rhs=base)
 out=dict(run_configuration=cfg,n=a.n,reference_step=a.reference_step,penalty=a.penalty,derivative_boundary=a.derivative_boundary,state=str(a.state),finite_difference_epsilons=[1e-5,3e-6],relative_jacobian_difference=float(np.linalg.norm(delta)/np.linalg.norm(J)),max_abs_jacobian_difference=float(np.max(abs(delta))),max_real_eigenvalue=float(max(values.real)),min_real_eigenvalue=float(min(values.real)),max_imaginary_eigenvalue=float(max(abs(values.imag))),leading_eigenvalues=[[float(v.real),float(v.imag)] for v in values[:12]],leading_eigenpair_residual=float(residual),speed_over_dr=float(speed),rk4=rows,seconds=time.perf_counter()-started,stage_map_limitation='Dependent endpoint D is eliminated; each stage already materializes it before the auxiliary advection derivative.' if not a.derivative_boundary else 'D is evolved; endpoint alpha/beta redundant coordinates eliminated.',qualification='Frozen time-dependent semidiscrete operator, not continuum or nonlinear stability. Positive real eigenvalues prevent absolute contractivity; displayed norms use the saved local variable scaling.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
