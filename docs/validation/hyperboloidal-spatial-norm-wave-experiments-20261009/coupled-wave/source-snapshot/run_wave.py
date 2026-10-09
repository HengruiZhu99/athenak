"""Actual native conformal scalar wave: controlled spectra and exact outgoing dipole."""
from pathlib import Path
import argparse, hashlib,json,subprocess,time
import numpy as np
import scipy
from scipy.io import mmread
from scipy.sparse.linalg import eigs,ArpackNoConvergence
w=Path(__file__).resolve().parent;root=w.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=16);p.add_argument('--span',type=float,default=2.2);p.add_argument('--closure',default='ray');p.add_argument('--ko',type=float,default=.1);p.add_argument('--end',type=float,default=6.);p.add_argument('--dt-factor',type=float,default=.1);p.add_argument('--spectra',action='store_true');p.add_argument('--cmc',action='store_true');args=p.parse_args()
name=f'N{args.n}-span{args.span}-{args.closure}-ko{args.ko}'+('-cmc' if args.cmc else '')
prefix=w/name;command=[str(w/'export-wave'),str(args.n),str(args.span),args.closure,str(args.ko),str(prefix)]+(['cmc'] if args.cmc else [])
begin=time.monotonic();done=subprocess.run(command,capture_output=True,text=True);Path(str(prefix)+'.stdout').write_text(done.stdout);Path(str(prefix)+'.stderr').write_text(done.stderr)
if done.returncode:raise RuntimeError(done.stderr)
meta=json.loads(done.stdout);A=mmread(str(prefix)+'.mtx').tocsr();G=[mmread(str(prefix)+f'-G{d}.mtx').tocsr() for d in range(3)];q=np.genfromtxt(str(prefix)+'-points.csv',delimiter=',',names=True);m=len(q);h=meta['spacing'];xyz=np.column_stack([q[d] for d in ['x','y','z']]);n=xyz/q['r'][:,None];axis=np.array([.3,.4,np.sqrt(.75)]);angle=np.einsum('ij,j->i',n,axis)
# Smooth analytic Gaussian physical profile; both retarded and advanced pieces retained.
def profile(s):
 z=(s+.5)/.35;F=.001*np.exp(-z*z)
 return F,-2*z*F/.35,(4*z*z-2)*F/.35**2,(12*z-8*z**3)*F/.35**3
# Physical Minkowski wave (directional derivative of [F(T-R)-F(T+R)]/R), divided by Omega.
# U=t-I(r), I'=1/outgoing; V=U+2*r/Omega. All evaluated strictly inside the ball.
def exact(t):
 r=q['r'];om=q['omega'];u=t-q['I'];v=u+2*r/om;fu,pu,ppu,pppu=profile(u);fv,pv,ppv,pppv=profile(v);du=-1/q['outgoing'];dv=du+2*q['L']/om**2
 B=fu-fv;S=pu+pv;value=-S/r-om*B/r**2
 dt=-(ppu+ppv)/r-om*(pu-pv)/r**2
 dr=-(ppu*du+ppv*dv)/r+S/r**2-q['omegap']*B/r**2-om*(pu*du-pv*dv)/r**2+2*om*B/r**3
 dtt=-(pppu+pppv)/r-om*(ppu-ppv)/r**2
 drdt=-(pppu*du+pppv*dv)/r+(ppu+ppv)/r**2-q['omegap']*(pu-pv)/r**2-om*(ppu*du-ppv*dv)/r**2+2*om*(pu-pv)/r**3
 br=np.einsum('ij,ij->i',np.column_stack([q[d] for d in ['bx','by','bz']]),n)
 phi=angle*value;pi=angle*(dt-br*dr)/q['alpha'];phidt=angle*dt;pidt=angle*(dtt-br*drdt)/q['alpha'];grad=axis[None,:]*value[:,None]/r[:,None]+angle[:,None]*n*(dr-value/r)[:,None]
 return np.r_[phi,pi],np.r_[phidt,pidt],grad
beta=np.column_stack([q[d] for d in ['bx','by','bz']]);inv=np.zeros((m,3,3));inv[:,0,0]=q['g00'];inv[:,0,1]=inv[:,1,0]=q['g01'];inv[:,0,2]=inv[:,2,0]=q['g02'];inv[:,1,1]=q['g11'];inv[:,1,2]=inv[:,2,1]=q['g12'];inv[:,2,2]=q['g22']
def energies(v,grad):
 phi,pi=v[:m],v[m:];gradient2=np.einsum('ni,nij,nj->n',grad,inv,grad);density=q['sqrtg']*(q['alpha']*(pi*pi+gradient2+q['R']/6*phi*phi)/2+pi*np.einsum('ni,ni->n',beta,grad));normal=q['sqrtg']*(pi*pi+gradient2+phi*phi)/2
 return float(h**3*density.sum()),float(h**3*normal.sum())
receipt={'name':name,'source_commit_at_export':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'export_command':command,'export':meta,'source_sha256':sha(w/'export_wave.cpp'),'executable_sha256':sha(w/'export-wave'),'driver_sha256':sha(Path(__file__)),'build_command':json.loads((w/'build-command.json').read_text()),'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'numpy':np.__version__,'scipy':scipy.__version__,'matrix_sha256':sha(Path(str(prefix)+'.mtx')),'points_sha256':sha(Path(str(prefix)+'-points.csv')),'headers':{str(p.relative_to(root)):sha(p) for p in [root/'src/utils/finite_diff.hpp',root/'src/z4c/hyperboloidal/spherical_ghosts.hpp',root/'src/z4c/hyperboloidal/interior_dissipation.hpp',root/'src/z4c/hyperboloidal/layer_reference.hpp',root/'src/z4c/hyperboloidal/cmc_reference.hpp',root/'src/z4c/hyperboloidal/conformal_rhs.hpp',root/'src/z4c/hyperboloidal/conformal_constraints.hpp']}}
if args.spectra:
 b=time.monotonic()
 try:ev,vec=eigs(A,k=8,which='LR',tol=1e-9,ncv=96,maxiter=500);scope='converged ARPACK LR8, no full-spectrum proof'
 except ArpackNoConvergence as e:ev,vec=e.eigenvalues,e.eigenvectors;scope=f'incomplete ARPACK LR8: {len(ev)} pairs; no rightmost bound'
 order=np.argsort(ev.real)[::-1];ev=ev[order];vec=vec[:,order];modes=[]
 for i in range(len(ev)):
  v=vec[:,i];sq=abs(v[:m])**2+abs(v[m:])**2;modes.append({'real':float(ev[i].real),'imag':float(ev[i].imag),'residual_l2':float(np.linalg.norm(A@v-ev[i]*v)/np.linalg.norm(v)),'squared_mode_fraction_r_gt_.8':float(sq[q['r']>.8].sum()/sq.sum()),'radius_peak':float(q['r'][sq.argmax()])})
 receipt['spectral']={'scope':scope,'seconds':time.monotonic()-b,'modes':modes};np.savez_compressed(str(prefix)+'-eigenpairs.npz',values=ev,vectors=vec);print(name,'spectral',receipt['spectral'],flush=True)
v,exactdt,exactgrad=exact(0);initialdefect=A@v-exactdt;receipt['initial_exact_pulse_discrete_defect']={'phi_rms':float(np.sqrt(np.mean(initialdefect[:m]**2))),'pi_rms':float(np.sqrt(np.mean(initialdefect[m:]**2))),'linf':float(abs(initialdefect).max())}
hist=[];t=0.;step=0;dt=args.dt_factor*h/meta['max_outgoing'];next_record=.025;threshold=1e6;stop='requested time reached'
def record():
 ex,_,eg=exact(t);grad=np.column_stack([d@v[:m] for d in G]);E,En=energies(v,grad);Ee,Ene=energies(ex,eg);error=v-ex
 return {'time':t,'phi_linf':float(abs(v[:m]).max()),'pi_linf':float(abs(v[m:]).max()),'both_linf':float(abs(v).max()),'error_linf':float(abs(error).max()),'error_rms':float(np.sqrt(np.mean(error**2))),'phi_error_rms':float(np.sqrt(np.mean(error[:m]**2))),'pi_error_rms':float(np.sqrt(np.mean(error[m:]**2))),'killing_energy':E,'sampled_exact_killing_energy':Ee,'positive_normal_energy':En,'sampled_exact_normal_energy':Ene}
hist.append(record());print(name,'start',meta,'initial',receipt['initial_exact_pulse_discrete_defect'],flush=True)
while t<args.end:
 ds=min(dt,args.end-t);k1=A@v;k2=A@(v+ds*k1/2);k3=A@(v+ds*k2/2);k4=A@(v+ds*k3);v+=ds*(k1+2*k2+2*k3+k4)/6;t+=ds;step+=1
 if t>=next_record or t==args.end:hist.append(record());next_record=t+.025
 if not np.isfinite(v).all() or abs(v).max()>threshold:stop='nonfinite or amplitude threshold1e6';hist.append(record());break
hist_path=Path(str(prefix)+f'-dt{args.dt_factor}-pulse.json');hist_path.write_text(json.dumps(hist,indent=2)+'\n');receipt['pulse']={'method':'RK4','dt_nominal':dt,'dt_factor':args.dt_factor,'stop':stop,'steps':step,'last':hist[-1],'initial':hist[0],'peak_both_linf':max(s['both_linf'] for s in hist),'peak_positive_normal_energy':max(s['positive_normal_energy'] for s in hist),'peak_killing_energy':max(s['killing_energy'] for s in hist),'history_sha256':sha(hist_path),'history_path':str(hist_path),'wall_seconds':time.monotonic()-begin};Path(str(prefix)+f'-dt{args.dt_factor}-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(name,'completed',receipt['pulse'],flush=True)
