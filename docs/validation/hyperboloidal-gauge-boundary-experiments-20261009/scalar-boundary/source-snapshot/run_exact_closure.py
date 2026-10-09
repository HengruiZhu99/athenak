"""Small exact native scalar operators, spectra, constant-preserving baselines and analytic pulse."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
import scipy
from scipy.io import mmread
from scipy.linalg import eig
from scipy.sparse.linalg import eigs,eigsh,ArpackNoConvergence
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'compiler_identity':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'compile_command':json.loads((work/'build-command.json').read_text()),'numpy':np.__version__,'scipy':scipy.__version__,'export_source_sha256':sha(work/'export_operator.cpp'),'export_executable_sha256':sha(work/'export-operator'),'consumed_headers':{str(p.relative_to(root)):sha(p) for p in [root/'src/utils/finite_diff.hpp',root/'src/z4c/hyperboloidal/spherical_ghosts.hpp',root/'src/z4c/hyperboloidal/interior_dissipation.hpp',root/'src/athena.hpp']},'command_environment':{'PYTHONPATH':str(root/'build-layer-research/boundary/python-deps'),'OPENBLAS_NUM_THREADS':'1'},'continuum':{'equation':'q_t+x.dot(grad q)=0 on unit ball, all boundary speeds outward','unweighted_energy':'E\'=3E-boundary_integral(q^2), E=integral(q^2 d3x)','time_weighted_energy':'exp(-3t)E is nonincreasing; no regular positive time-independent spatial L2 weight contracts all smooth core data while admitting constants','pointwise':'Exact q(x,t)=q0(exp(-t)x), hence L-infinity never exceeds initial continuous sup. L2 changes alone do not imply instability.'},'cases':[],'rejections':[]}
# Keep explicit planner rejection, with no silent degradation.
command=[str(work/'export-operator'),'12','2.2','ray','upwind','.1',str(work/'rejected-N12')];done=subprocess.run(command,capture_output=True,text=True);receipt['rejections'].append({'command':command,'exit_status':done.returncode,'stdout':done.stdout,'stderr':done.stderr});(work/'N12-rejection.log').write_text(done.stdout+done.stderr)
configs=[(24,2.1,'nearest','centered',.1),(24,2.1,'fallback','centered',.1)]
for n,span,closure,derivative,ko in configs:
 name=f'N{n}-span{span}-{closure}-{derivative}-ko{ko}';prefix=work/name;command=[str(work/'export-operator'),str(n),str(span),closure,derivative,str(ko),str(prefix)];done=subprocess.run(command,capture_output=True,text=True);(Path(str(prefix)+'.stdout')).write_text(done.stdout);(Path(str(prefix)+'.stderr')).write_text(done.stderr)
 if done.returncode:receipt['rejections'].append({'command':command,'exit_status':done.returncode,'stdout':done.stdout,'stderr':done.stderr});continue
 case={'name':name,'export_command':command,'export':json.loads(done.stdout),'matrix_sha256':sha(Path(str(prefix)+'.mtx')),'points_sha256':sha(Path(str(prefix)+'-points.csv'))};A=mmread(str(prefix)+'.mtx').tocsr();points=np.genfromtxt(str(prefix)+'-points.csv',delimiter=',',names=True);r=points['r'];xyz=np.column_stack([points[d] for d in ['x','y','z']]);size=A.shape[0];begin=time.monotonic()
 if n==16:
  ev,vec=eig(A.toarray(),check_finite=False);scope='full dense scalar spectrum'
 else:
  try:ev,vec=eigs(A,k=8,which='LR',tol=1e-10,ncv=64,maxiter=1200);scope='sparse rightmost LR8 (ARPACK converged)'
  except ArpackNoConvergence as error:ev,vec=error.eigenvalues,error.eigenvectors;scope=f'ARPACK incomplete ({len(ev)} converged pairs); no claimed rightmost bound'
 order=np.argsort(ev.real)[::-1];ev=ev[order];vec=vec[:,order];modes=[]
 for i in range(min(8,len(ev))):
  v=vec[:,i];square=abs(v)**2;residual=np.linalg.norm(A@v-ev[i]*v)/np.linalg.norm(v);modes.append({'real':float(ev[i].real),'imag':float(ev[i].imag),'absolute_residual_l2':float(residual),'radius_peak':float(r[np.argmax(square)]),'squared_fraction_r_gt_.8':float(square[r>.8].sum()/square.sum()),'squared_fraction_r_lt_.3':float(square[r<.3].sum()/square.sum())})
 case['spectral']={'scope':scope,'seconds':time.monotonic()-begin,'rightmost_found':modes,'positive_real_pair_count_gt_1e-8':int((ev.real>1e-8).sum())};np.savez_compressed(str(prefix)+'-eigenpairs.npz',eigenvalues=ev,eigenvectors=vec[:,:min(8,len(ev))])
 symmetric=(A+A.T)*.5;mu=float(eigsh(symmetric,k=1,which='LA',tol=1e-10,return_eigenvectors=False)[0]);case['euclidean_log_norm_mu2']=mu;case['constant_residual_linf_reloaded']=float(abs(A@np.ones(size)).max())
 # Analytic off-axis pulse over repeated characteristic expansion periods.
 h=case['export']['spacing'];dt=.06*h;end=12.;q0=np.exp(-30*((xyz[:,0]-.3)**2+(xyz[:,1]+.15)**2+(xyz[:,2]-.2)**2));q=q0.copy();hist=[];time_now=0.;step=0;next_record=0.
 def record():
  exact=np.exp(-30*((np.exp(-time_now)*xyz[:,0]-.3)**2+(np.exp(-time_now)*xyz[:,1]+.15)**2+(np.exp(-time_now)*xyz[:,2]-.2)**2));E=float(h**3*np.dot(q,q));Ee=float(h**3*np.dot(exact,exact));return {'time':time_now,'energy':E,'analytic_sampled_energy':Ee,'time_weighted_energy':float(np.exp(-3*time_now)*E),'analytic_sampled_time_weighted_energy':float(np.exp(-3*time_now)*Ee),'linf':float(abs(q).max()),'analytic_sampled_linf':float(abs(exact).max()),'error_linf':float(abs(q-exact).max()),'error_rms':float(np.sqrt(np.mean((q-exact)**2)))}
 hist.append(record());stop='requested t12 reached'
 while time_now<end:
  ds=min(dt,end-time_now);q1=q+ds*(A@q);q2=.75*q+.25*(q1+ds*(A@q1));q=(q+2*(q2+ds*(A@q2)))/3;time_now+=ds;step+=1
  if time_now>=next_record+.1 or time_now==end:
   hist.append(record());next_record=time_now
  if not np.isfinite(q).all() or abs(q).max()>1e8:stop='nonfinite or amplitude threshold1e8';break
 hist_path=Path(str(prefix)+'-pulse.json');hist_path.write_text(json.dumps(hist,indent=2)+'\n');case['pulse']={'rk':'SSPRK3','dt_nominal':dt,'stop':stop,'steps':step,'last':hist[-1],'initial_sampled_linf':float(abs(q0).max()),'peak_sampled_linf':max(hh['linf'] for hh in hist),'peak_error_linf':max(hh['error_linf'] for hh in hist),'peak_energy':max(hh['energy'] for hh in hist),'initial_energy':hist[0]['energy'],'max_time_weighted_energy_over_initial':max(hh['time_weighted_energy'] for hh in hist)/hist[0]['time_weighted_energy'],'history_sha256':sha(hist_path)}
 receipt['cases'].append(case);(work/'exact-closure-results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(name,'eig',modes[:2],'mu2',mu,'pulse',case['pulse']['last']['error_linf'],case['pulse']['peak_sampled_linf'],flush=True)
(work/'exact-closure-results.json').write_text(json.dumps(receipt,indent=2)+'\n')
