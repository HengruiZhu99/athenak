from rational_dtn import *
from volume import boundary
from exterior_buffer import buffer

def defects(z,k,cfg,N):
 sv,(U,D,T,B,R,vh,norms)=sigma(z,k,cfg,N,True);coef=np.linalg.solve(R,vh[-1].conj());u=U@coef;du=D@coef;ddu=U@T@T@coef
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0;rhs=V0@u+V1@du+V2@ddu
 return {'lambda':[z.real,z.imag],'sigma_min':sv,'boundary_scaled_max':float(abs(B@coef/norms).max()),'volume_relative':float(np.linalg.norm(z*u-rhs)/max(np.linalg.norm(rhs),abs(z)*np.linalg.norm(u),1e-300)),'Theta':float(abs(u[7]))}

cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295);k=np.pi/256;p=Path(__file__).parent
fitdata=np.load(p/'rational-kpi256.npz');N=lambda z:z*fitdata['A']+fitdata['B']+np.einsum('p,pij->ij',1/(z-fitdata['poles']),fitdata['coeff'])
fit=json.load(open(p/'rational-kpi256.json'));rows=[]
for rec in fit['refined']:
 if rec['sigma_min']<1e-9 and not rec['lower_bound']:rows.append(defects(complex(*rec['lambda']),k,cfg,N))
base=json.load(open(p/'reference-screen-results.json'));old=[]
for case in base['cases']:
 if case['name']!='baseline'or abs(case['k']-k)>1e-10:continue
 for r in case['roots']:
  z=complex(*r['lambda_value']);U,D,_,_=schur(z,k,cfg);B=boundary(U,D,cfg);_,R=np.linalg.qr(U);B=B@np.linalg.inv(R);B/=np.linalg.norm(B,axis=1)[:,None]
  relaxed=[]
  for nu in [0,2e-5,1e-3,.1]:
   C=(z+nu)*B;C/=np.linalg.norm(C,axis=1)[:,None];relaxed.append({'nu':nu,'sigma_min':float(np.linalg.svd(C,compute_uv=False)[-1])})
  old.append({'lambda':[z.real,z.imag],'relaxed':relaxed,'exact_DtN_sigma':sigma(z,k,cfg,lambda zz:exterior(zz,k,cfg)),'fit_DtN_sigma':sigma(z,k,cfg,N)})
buff=[]
for rec in json.load(open(p/'exterior-buffer.json'))['cases']:
 h,n=rec['h0'],rec['nodes'];hs=h*np.minimum(1.055**np.arange(n),1000);Nbuf=lambda z:buffer(z,k,cfg,hs,rec['flux'])
 for r in rec['refined']:
  if not r['at_lower_bound']and r['sigma_min']<1e-9:buff.append({'h0':h,'nodes':n,'flux':rec['flux'],**defects(complex(*r['lambda']),k,cfg,Nbuf)})
out={'scope':'Independent full-PDE/boundary defects for candidate growing modes; exact statement for uniform positive incoming-state relaxation.','original_roots':old,'rejected_fit_roots':rows,'buffer_roots':buff};(p/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
