from pathlib import Path
import json,numpy as np,sympy as sy
p=Path(__file__).resolve().parent
np.seterr(all='raise')
def cm(x):
 a=np.asarray(x);return a[:,:,0]+1j*a[:,:,1]
def eig(A,k):
 weights=np.array([1.]*12+[1/max(k,1.)]*8)
 B=np.einsum('i,ij,j->ij',weights,A,1/weights)
 ev,V=np.linalg.eig(B)
 error=max(float(abs(np.einsum('ij,j->i',B,V[:,j])-ev[j]*V[:,j]).max()/((np.max(np.sum(abs(B),axis=1))+abs(ev[j]))*abs(V[:,j]).max())) for j in range(20))
 assert error<1e-12
 return ev,error
report={'scope':'Actual C0 live value-only damping at finite positive Omega, stationary Einstein reference, finite-Theta dual perturbations and coefficient-aware linear eight-constraint chain. No global/native/energy/nonlinear-bound-preservation/scri-closure acceptance.'}
# Exact live algebra. b denotes beta^i Omega_i and w=-b/alpha.
o,b,kap,V=sy.symbols('Omega b kappa V',real=True)
k2=V*(2*b/kap+o-1);m=sy.expand(kap*k2)
sigma=sy.cancel((2*b-kap*(1+k2))/o)
assert sy.simplify(sigma-((1-V)*(2*b-kap)/o-V*kap))==0
assert sy.simplify(sigma.subs(V,1)+kap)==0
report['exact_live_identity']={'m':str(m),'sigma':str(sigma),'sigma_at_V1':'-kappa_input','gradient':'m_i=V_i[2 beta^j Omega_j+kappa(Omega-1)]+V[2 (partial_i beta^j)Omega_j+2 beta^j Omega_ij+kappa Omega_i]','ADM_delta':'DeltaP=DeltaTheta=-mTheta/Omega; DeltaH=-4K mTheta/Omega; DeltaM_i=2 partial_i(mTheta/Omega); DeltaZ_i=0','Einstein_sector':'Theta=0 makes additions vanish; derivatives homogeneous in constraints','live_beta_columns':'DeltaL[P/Theta,beta_j]=-2 V Theta Omega_j/Omega','principal_unchanged':True,'no_new_double_pole':True}
principal=json.loads((p/'principal.log').read_text());assert principal['passed_kernel_cases']==360 and principal['max_kernel_symbol_error']<1e-12;report['principal']=principal
tensor=json.loads((p/'tensor-gate.json').read_text());assert tensor==json.loads((p/'tensor-gate-debug.json').read_text())
assert tensor['rows']==4004 and tensor['Einstein_addition_max']==0 and tensor['inner_kappa2_exact_zero'] and tensor['no_new_double_pole']
assert tensor['nonlinear_dual_parts_identity_error']<1e-14 and tensor['nonlinear_sigma_numerator_identity_error']<1e-13 and tensor['reference_fixedpoint']<2e-11
report['tensor']=tensor
grad=json.loads((p/'gradient-gate.json').read_text());assert grad['rows']==104 and grad['full_gradient_relative_FD_error']<1e-9 and grad['inner_value_and_gradient_max']==0;report['gradient']=grad
# Exact statements are for rationally reconstructed reference pole matrices only.
lam=sy.Symbol('lambda');pole_rows=json.loads((p/'leading-pole.json').read_text());assert len(pole_rows)==16
report['poles']=[]
for x in pole_rows:
 A=cm(x['P']);assert np.isfinite(A).all() and abs(A.imag).max()==0
 ev=np.linalg.eigvals(A.real);nonzero=ev[abs(ev)>1e-7];assert len(nonzero)==15 and nonzero.real.max()<0
 s1=np.linalg.svd(A.real,compute_uv=False);A2=np.einsum('ij,jk->ik',A.real,A.real);s2=np.linalg.svd(A2,compute_uv=False)
 null=20-int(sum(s1>1e-9*s1[0]));null2=20-int(sum(s2>1e-9*s2[0]));assert null==null2==5
 out={k:x[k] for k in ['a','profile','perturb']}|{'numerical_zero_nullity':null,'numerical_square_nullity':null2,'max_nonzero_real':float(nonzero.real.max())}
 if x['perturb']==0:
  R=sy.Matrix([[sy.Rational(float(z)).limit_denominator(1000000) for z in row] for row in A.real]);err=float(abs(np.array(R,dtype=float)-A.real).max());assert err<1e-11
  assert 20-R.rank()==20-(R*R).rank()==5
  char=R.charpoly(lam).as_expr();factors=sy.factor_list(char)[1];hurwitz=[]
  for factor,mult in factors:
   if factor==lam:assert mult==5;continue
   poly=sy.Poly(factor,lam);c=poly.all_coeffs();n=poly.degree()
   if c[0]<0:c=[-v for v in c]
   H=sy.Matrix(n,n,lambda i,j:c[2*j-i+1] if 0<=2*j-i+1<=n else 0);minors=[sy.factor(H[:i,:i].det()) for i in range(1,n+1)];assert all(v>0 for v in minors)
   hurwitz.append({'factor':str(factor),'multiplicity':mult,'exact_positive_minors':[str(v) for v in minors]})
  out.update(rational_reconstruction_error=err,characteristic=str(sy.factor(char)),Hurwitz=hurwitz,exact_reconstructed_reference_zero_nullity=5,exact_reconstructed_reference_square_nullity=5)
 report['poles'].append(out)
full=json.loads((p/'full20.json').read_text());pairs={};roots=[]
for x in full:
 A=cm(x['L']);assert np.isfinite(A).all() and x['reference_fixedpoint']<2e-9
 key=tuple(x[k] for k in ['a','perturb','r','Omega','k','oblique']);pairs.setdefault(key,{})[x['profile']]=x
 ev,back=eig(A,x['k']);j=ev.real.argmax();roots.append({k:x[k] for k in ['a','profile','perturb','r','Omega','k','oblique']}|{'max_real':float(ev[j].real),'imag':float(ev[j].imag),'backward_error':back})
identity=0;beta_count=0
for pair in pairs.values():
 assert set(pair)=={0,1};x=pair[1];expected=np.zeros((20,20));expected[2,3]=expected[3,3]=-10*x['kappa2']/x['Omega']
 for j in range(3):expected[2,4+j]=expected[3,4+j]=-2*x['V']*x['Theta']*x['Omega_gradient'][j]/x['Omega']
 delta=cm(x['L'])-cm(pair[0]['L']);identity=max(identity,float(abs(delta-expected).max()/(1+abs(expected).max())));beta_count+=int(abs(expected[:,4:7]).max()>0)
assert identity<1e-11 and beta_count>0
report['full20']={'rows':len(full),'matched_pairs':len(pairs),'live_Theta_beta_pairs':beta_count,'relative_exact_delta_error':identity,'max_reference_fixedpoint':max(x['reference_fixedpoint'] for x in full),'worst_reference_by_a_and_mode':[max((x for x in roots if x['a']==a and x['profile']==mode and x['perturb']==0),key=lambda x:x['max_real']) for a in [.5,.75,1,2] for mode in [0,1]],'roots':roots,'positive_primitive_roots_retained':True,'subsidiary_classification_from_primitive_roots':False}
# C_h here is the analytic constraint differential, not a native grid stencil.
report['subsidiary']={};summaries={}
for mode,name in [(0,'constraint-base.json'),(1,'constraint-profile.json')]:
 rows=json.loads((p/name).read_text());assert len(rows)==2330
 finest=[];omissions=[];gauge=[];groots=[];conv=[]
 for x in rows:
  q,dv,pred,ng,g=map(cm,[x['Q'],x['D'],x['subsidiary'],x['without_dkappa2'],x['constraint_generator']]);assert abs(q[:,[0,4,5,6]]).max()==0
  rel=float(abs(dv-pred).max()/(1+abs(dv).max()+abs(pred).max()))
  if x['level']==4:
   assert rel<1e-7;finest.append(rel)
   gauge.append(float(abs(dv[:,[0,4,5,6]]).max()/(1+abs(dv).max())))
   omissions.append(float(abs(dv-ng).max()/(1+abs(dv).max()+abs(ng).max())))
   weights=np.array([1.]*4+[max(x['k'],1.)]*4);B=np.einsum('i,ij,j->ij',weights,g[:,:8],1/weights);ev=np.linalg.eigvals(B);j=ev.real.argmax()
   groots.append({k:x[k] for k in ['a','r','k','oblique']}|{'max_real':float(ev[j].real),'imag':float(ev[j].imag)})
  if x['r'] in [.2,.225,.65,.95,.9983726993838523] and x['k']==64 and not x['oblique']:conv.append({k:x[k] for k in ['a','r','level','h']}|{'relative_error':rel})
 summaries[mode]={'rows':len(rows),'local8_rows':len(groots),'max_finest_scaled_chain_error':max(finest),'gauge_Q_columns_exact_zero':True,'max_finest_scaled_gauge_Cdot':max(gauge),'worst_omitted_dkappa2_error':max(omissions),'worst_k0_by_a':[max((x for x in groots if x['a']==a and x['k']==0),key=lambda x:x['max_real']) for a in [.5,.75,1,2]],'roots':groots,'refinement':conv,'all_local8_negative_claim':False}
assert summaries[1]['worst_omitted_dkappa2_error']>.1
report['subsidiary']={'base':summaries[0],'live':summaries[1],'interpretation':'Coefficient-aware local generators retain bounded positive inner k0 roots already in baseline. No global subsidiary spectrum or energy estimate follows.'}
# Actual native Cartesian cell-centered span, reference Omega_min for every a.
input_path=p.parents[2]/'build-layer-research/spatial-norm-native-controls/N36-t0.2/finite-angular-long-N36/layer.athinput'
mesh={};scope=''
for line in input_path.read_text().splitlines():
 line=line.split('#')[0].strip()
 if line.startswith('<'):scope=line.strip('<>')
 elif scope=='mesh' and '=' in line:
  key,val=map(str.strip,line.split('=',1));mesh[key]=val
span=float(mesh['x1max'])-float(mesh['x1min']);assert span==2.1
rkrows=json.loads((p/'full20-RK-all-a.json').read_text());assert len(rkrows)==192
cases=[]
for x in rkrows:
 N=x['N'];h=span/N;axis=float(mesh['x1min'])+(np.arange(N)+.5)*h;rr=axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2;rmax2=rr[rr<1].max();omega=(1-rmax2)/(2*x['a'])
 assert abs(omega-x['Omega'])<2e-14 and abs(x['dt']-.03*omega)<1e-14 and abs(x['r']**2-rmax2)<2e-14
 assert abs(x['k']-(256. if x['frequency'] else np.pi/h))<2e-13
 ev,back=eig(cm(x['L']),x['k']);damped=ev[ev.real<=0];z=x['dt']*damped;excess=float(max(0.,max(abs(1+z+z*z/2+z*z*z/6))-1));assert excess<1e-12
 cases.append({k:x[k] for k in ['a','N','span','Omega','dt','k','frequency','profile','perturb','oblique']}|{'nonpositive_root_count':len(damped),'max_nonpositive_root_RK3_excess':excess,'max_primitive_real':float(ev.real.max()),'backward_error':back})
report['RK3']={'cases':cases,'count':len(cases),'max_nonpositive_root_excess':max(x['max_nonpositive_root_RK3_excess'] for x in cases),'scope':'Continuum frozen actual20 generators evaluated at pi/h and k256, not the native finite-difference symbols. Scalar eigenvalue check excludes positive roots and does not bound nonnormal propagators/global RK stability.'}
report['passed_finite_Omega_local_gate']=True
report['global_native_or_scri_stability_accepted']=False
(p/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('PASS live damping local actual-kernel gate:',len(full),'finite20,16 poles,4660 chain,192 RK cases')
print('chain errors',summaries[0]['max_finest_scaled_chain_error'],summaries[1]['max_finest_scaled_chain_error'],'omittedgradient',summaries[1]['worst_omitted_dkappa2_error'])
print('finite reference max',report['full20']['max_reference_fixedpoint'],'delta',identity)
