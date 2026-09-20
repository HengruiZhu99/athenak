"""Bounded damping-source halfspace screen; no finite scan proves stability."""
import argparse,sys,json,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.model.resolve()))
import numpy as np
from scipy import optimize
from volume import Config,QP,asdict
from boundary import assess_mode,profile
from validate import mode
base=dict(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295,G=1)
specs=[('baseline',.1,2,.1),('k03',.03,2,.1),('k01',.01,2,.1),('k0',0,2,.1),('weak_gauge',.1,.2,.01),('weak_all',.01,.2,.01),('sources_off',0,0,0)]
ks=[np.pi/256,np.pi/1024]
xs=np.geomspace(1e-7,.08,260)
out={'scope':'Frozen G1 weak-field constant-coefficient halfspace, original zero_rate, real RHP samples only; absence of sampled root is not a stability result. Tangentially resolved k; normal resolution not asserted.','model':str(args.model.resolve()),'model_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in args.model.glob('*.py')},'real_search_M_inverse':[float(xs[0]),float(xs[-1])],'real_samples':len(xs),'cases':[]}
start=time.monotonic()
for name,kap,eta,ld in specs:
 cfg=Config(**base,kappa=kap,eta=eta,lapse_damping=ld)
 for k in ks:
  ys=[];errors=[]
  for x in xs:
   try:ys.append(assess_mode(x,k,cfg)['sigma_min'])
   except ValueError as e:ys.append(float('nan'));errors.append({'lambda':float(x),'error':str(e)})
  roots=[];other=[]
  def canonical(x):
   _,(U,D,T,B,R,vh,norms)=assess_mode(x,k,cfg,'zero_rate',True)
   return B@np.linalg.inv(U[QP])
  for i in range(1,len(xs)-1):
   if not(ys[i]<ys[i-1] and ys[i]<ys[i+1]):continue
   try:
    fit=optimize.minimize_scalar(lambda x:assess_mode(x,k,cfg)['sigma_min'],bracket=xs[i-1:i+2],method='brent',options={'xtol':1e-13,'maxiter':160})
    if fit.fun>1e-6:other.append({'lambda':float(fit.x),'sigma_min':float(fit.fun)});continue
    E=canonical(fit.x);norms=np.linalg.norm(E,axis=1);left,_,vh=np.linalg.svd(E/norms[:,None]);lv=left[:,-1];rv=vh[-1].conj()
    signed=lambda x:float(np.real(np.vdot(lv,canonical(x)@rv/norms)))
    root=optimize.brentq(signed,xs[i-1],xs[i+1],xtol=1e-16,rtol=1e-14)
    check=mode(root,k,cfg,'zero_rate');assert check['boundary_scaled']<1e-10 and check['bulk_relative']<1e-10
    check['e_fold_M']=1/root
    check['profile']=[{key:val for key,val in row.items()if key!='state'}for row in profile(root,k,cfg)['profile']]
    roots.append(check)
   except(Exception)as e:other.append({'lambda':float(xs[i]),'error':str(e)})
  c=cfg.alpha*np.sqrt(cfg.chi);sig=cfg.alpha*kap
  slow=-sig+np.lib.scimath.sqrt(sig*sig-c*c*k*k)
  # The demonstrated constraint-free surface root survives physical constraint BC variants.
  surf_lam=cfg.beta_n*k/np.sqrt(2)+1j*c*k/np.sqrt(2)
  surf=mode(surf_lam,k,cfg,'physical_radiation_coupled')
  case={'name':name,'config':asdict(cfg),'k':float(k),'wavelength_M':float(2*np.pi/k),'cells_per_wavelength_at_dx32':float(2*np.pi/k/32),'roots':roots,'nonroot_minima_or_refinement_errors':other,'classification_errors':errors,'sample_min_sigma':float(np.nanmin(ys)),'constraint_wave_slow_lambda_comoving':[float(slow.real),float(slow.imag)],'constraint_wave_slow_efold_M':float(-1/slow.real)if slow.real<0 else None,'physical_coupled_known_surface_mode':surf}
  out['cases'].append(case);out['elapsed_seconds']=time.monotonic()-start;args.output.write_text(json.dumps(out,indent=2)+'\n')
  print(name,'k',k,'roots',[round(q['lambda_value'][0],10)for q in roots],'min',np.nanmin(ys),'elapsed',round(out['elapsed_seconds'],1),flush=True)
