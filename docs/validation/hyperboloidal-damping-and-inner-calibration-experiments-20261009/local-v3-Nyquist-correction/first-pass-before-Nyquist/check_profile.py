from pathlib import Path
import hashlib,json,numpy as np,sympy as sy
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def cm(x):a=np.asarray(x);return a[:,:,0]+1j*a[:,:,1]
report={'scope':'Stationary Einstein-reference coefficient-aware actual C0 full20/eight-constraint and local finite-Omega damping profile. No global/native/scri stability or stronger Theta falloff assertion.'}
# Exact general S,a normalization and derivative cancellation.
S,a,o,kap=sy.symbols('S a Omega kappa',positive=True)
kcrit=2*S/a**2;d=kap-kcrit;k2=(kcrit-kap)/kap*(1-o);eff=sy.factor(kap*(1+k2))
alpha=S/a-o;alpha_w=-S/a**2+2*o/a
sigma0=sy.cancel((-2*alpha_w-eff)/o);sigma1=sy.cancel((2*alpha/a-eff)/o)
assert sy.simplify(eff-(kcrit+d*o))==0
assert sy.simplify(sigma0+4/a+d)==sy.simplify(sigma1+2/a+d)==0
assert sy.diff(sigma0,o)==sy.diff(sigma1,o)==0
report['exact_profile']={'effective_kappa':str(eff),'sigma0':str(sigma0),'sigma1':str(sigma1),'d_sigma_d_Omega':'0','kappa2_core':'0','finite_Theta_assumed_only':True}
# Printed Eq19, independent of broad prose below Eq23. Map rho=kappa2.
s,k,w,rho=sy.symbols('s k w rho',real=True)
flat=sy.expand((s*s+k*(2+rho)*s+w*w)*(s*s+k*s+w*w)-k*k*rho*w*w)
c=sy.Poly(flat,s).all_coeffs();n=4
H=sy.Matrix(n,n,lambda i,j:c[2*j-i+1] if 0<=2*j-i+1<=n else 0)
deltas=[sy.factor(H[:i,:i].det()) for i in range(1,5)]
assert sy.factor(deltas[2]-2*k**4*(rho+3)**2*w*w*(rho+1))==0
assert sy.factor(c[-1]-w*w*(w*w-k*k*rho))==0
for rv in [sy.Rational(-1,5),sy.Rational(0)]:
 for kv in [sy.Rational(1),sy.Rational(10)]:
  for wv in [sy.Rational(1,100),sy.Rational(1),sy.Rational(100)]:
   assert all(x.subs({rho:rv,k:kv,w:wv})>0 for x in deltas)
assert c[-1].subs({rho:sy.Rational(1,5),k:1,w:sy.Rational(1,10)})<0
report['flat_primary_check']={'source':'https://arxiv.org/pdf/gr-qc/0504114','equation':'19','quartic':str(flat),'Hurwitz_minors':[str(x) for x in deltas],'all_nonzero_frequency_interval':'-1 < rho <= 0 for kappa>0; constant flat/inertial background only','positive_rho_counterexample_constant':str(c[-1].subs({rho:sy.Rational(1,5),k:1,w:sy.Rational(1,10)})),'independent_frozen_index_sha256':'3a2b38c659820243b4d850afeff0d819e4c946f01e25cf22ae6d8fb5ababd5d6'}
# The direct actual zero-Omega pole is rational on these exact reference jets.
poles=json.loads((p/'leading-pole.json').read_text());report['leading_poles']=[]
lam=sy.Symbol('lambda')
for x in poles:
 A=cm(x['P']);assert np.isfinite(A).all() and abs(A.imag).max()==0
 R=sy.Matrix([[sy.Rational(float(z)).limit_denominator(1000000) for z in row] for row in A.real])
 reconstruction=float(abs(np.array(R,dtype=float)-A.real).max());assert reconstruction<1e-11
 nullity=20-R.rank();square_nullity=20-(R*R).rank();assert nullity==square_nullity==5
 factors=sy.factor_list(R.charpoly(lam).as_expr())[1];hurwitz=[]
 for factor,multiplicity in factors:
  if factor==lam:assert multiplicity==5;continue
  poly=sy.Poly(factor,lam);coeff=poly.all_coeffs();deg=poly.degree()
  if coeff[0]<0:coeff=[-x for x in coeff]
  H=sy.Matrix(deg,deg,lambda i,j:coeff[2*j-i+1] if 0<=2*j-i+1<=deg else 0)
  minors=[H[:i,:i].det() for i in range(1,deg+1)];assert all(v>0 for v in minors)
  hurwitz.append({'factor':str(factor),'multiplicity':multiplicity,'positive_exact_Hurwitz_minors':[str(v) for v in minors]})
 report['leading_poles'].append({'a':x['a'],'profile':bool(x['profile']),'rational_reconstruction_error':reconstruction,'zero_nullity':nullity,'square_nullity':square_nullity,'characteristic':str(sy.factor(R.charpoly(lam).as_expr())),'Hurwitz':hurwitz})
# Nonlinear tensor and fixed-reference identity is checked separately under ASan.
tensor=json.loads((p/'tensor-gate.json').read_text());assert tensor==json.loads((p/'tensor-gate-debug.json').read_text())
assert tensor['rows']==4004 and tensor['Einstein_addition_max']==0 and tensor['nonlinear_parts_identity_error']<1e-14
assert tensor['reference_fixedpoint']<2e-11 and tensor['exact_core_kappa2_zero'] and tensor['no_new_double_pole']
assert -1<tensor['kappa2_min']<=tensor['kappa2_max']==0
report['nonlinear_tensor_gate']=tensor
principal=json.loads((p/'principal.log').read_text());assert principal['passed_kernel_cases']==360 and principal['max_kernel_symbol_error']<1e-12
report['principal']=principal
# Actual full20 finite-Omega samples: verify exact lower-order difference,
# retain positive slow roots rather than turning them into an acceptance test.
full=json.loads((p/'full20.json').read_text());groups={};roots=[];rk=[];raw=[]
for x in full:
 A=cm(x['L']);assert np.isfinite(A).all();assert x['reference_fixedpoint']<2e-9
 key=tuple(x[k] for k in ['a','perturb','r','Omega','k','oblique']);groups.setdefault(key,{})[x['profile']]=x
 # Frequency/coordinate weights are for eigensolver conditioning only.
 weights=np.array([1.]*12+[1/max(x['k'],1.)]*8)
 B=np.einsum('i,ij,j->ij',weights,A,1/weights);ev,V=np.linalg.eig(B);j=ev.real.argmax()
 backward=max(float(abs(np.einsum('ij,j->i',B,V[:,j])-ev[j]*V[:,j]).max()/((np.max(np.sum(abs(B),axis=1))+abs(ev[j]))*abs(V[:,j]).max())) for j in range(20))
 assert backward<1e-12
 roots.append({k:x[k] for k in ['a','profile','perturb','r','Omega','k','oblique']}|{'max_real':float(ev[j].real),'imag':float(ev[j].imag),'backward_error':backward})
 if x['a']==.5 and x['perturb']==0:
  for omega,dt in [(.0142578125,.000427734375),(.0038368055555556,.0001151041666666667),(.003251953125,.00009755859375)]:
   if abs(x['Omega']-omega)>1e-10:continue
   damp=ev[ev.real < -1e-8];z=dt*damp;excess=float(max(0.,np.max(abs(1+z+z*z/2+z*z*z/6))-1));assert excess<1e-12
   rk.append({'profile':x['profile'],'Omega':omega,'dt':dt,'k':x['k'],'oblique':x['oblique'],'max_damped_RK3_excess':excess})
 if x['a']==.5 and x['perturb']==0 and x['k']==0 and not x['oblique']:
  dt=.03*x['Omega'];Z=dt*A;Z2=np.einsum('ij,jk->ik',Z,Z);Z3=np.einsum('ij,jk->ik',Z2,Z)
  R=np.eye(20)+Z+Z2/2+Z3/6
  raw.append({'profile':x['profile'],'Omega':x['Omega'],'dt':dt,'raw_RK3_norm2':float(np.linalg.svd(R,compute_uv=False)[0])})
identity=0
for key,pair in groups.items():
 assert set(pair)=={0,1};x=pair[1];delta=cm(x['L'])-cm(pair[0]['L']);expected=np.zeros((20,20));k2=float((2/x['a']**2-10)/10*(1-x['Omega']))
 expected[2,3]=expected[3,3]=-10*k2/x['Omega'];identity=max(identity,float(abs(delta-expected).max()/(1+abs(expected).max())))
assert identity<1e-12
report['full20']={'rows':len(full),'only_two_Theta_column_entries_change_relative_error':identity,'max_reference_fixedpoint':max(x['reference_fixedpoint'] for x in full),'roots':roots,'worst_target_a_half_reference':[max((x for x in roots if x['a']==.5 and x['profile']==mode and x['perturb']==0),key=lambda x:x['max_real']) for mode in [0,1]],'negative_root_RK3_cases':rk,'k0_raw_step_norms':raw,'all_primitive_roots_negative_claim':False}
# Coefficient-aware actual dual20 -> eight physical constraints, including dκ2.
rows=json.loads((p/'constraint-profile.json').read_text());assert len(rows)==1000
errors=[];omissions=[];gauge=[];g_roots=[];convergence=[]
for x in rows:
 q,dv,pred,ng,g=map(cm,[x['Q'],x['D'],x['subsidiary'],x['without_dkappa2'],x['constraint_generator']]);assert abs(q[:,[0,4,5,6]]).max()==0
 rel=float(abs(dv-pred).max()/(1+abs(dv).max()+abs(pred).max()))
 if x['level']==4:
  assert rel<1e-7;errors.append(rel)
  gauge.append({'r':x['r'],'k':x['k'],'oblique':x['oblique'],'absolute':float(abs(dv[:,[0,4,5,6]]).max()),'scaled':float(abs(dv[:,[0,4,5,6]]).max()/(1+abs(dv).max()))})
  omissions.append({k:x[k] for k in ['r','k','oblique']}|{'relative_closure_error_without_dkappa2':float(abs(dv-ng).max()/(1+abs(dv).max()+abs(ng).max()))})
  weights=np.array([1.]*4+[max(x['k'],1.)]*4);A=np.einsum('i,ij,j->ij',weights,g[:,:8],1/weights);ev,V=np.linalg.eig(A);j=ev.real.argmax();assert ev[j].real<0
  g_roots.append({k:x[k] for k in ['r','k','oblique']}|{'max_real':float(ev[j].real),'imag':float(ev[j].imag)})
 if x['r'] in [.5,.65,.95,.992845500317144,.9983726993838523] and x['k']==64 and not x['oblique']:
  convergence.append({'r':x['r'],'level':x['level'],'h':x['h'],'relative_error':rel,'gauge_Cdot_absolute':float(abs(dv[:,[0,4,5,6]]).max())})
assert max(x['relative_closure_error_without_dkappa2'] for x in omissions)>1e-3
report['subsidiary']={'rows':len(rows),'local8_rows':len(g_roots),'max_finest_relative_error':max(errors),'worst_local_root':max(g_roots,key=lambda x:x['max_real']),'roots':g_roots,'gauge_Q_columns_exact_zero':True,'gauge':gauge,'convergence':convergence,'worst_omitted_dkappa2':max(omissions,key=lambda x:x['relative_closure_error_without_dkappa2'])}
(p/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('PASS profile local gates; no global/native/scri acceptance')
print('actual20 finite rows',len(full),'chain rows',len(rows),'relative error',max(errors),'negative local8 max',report['subsidiary']['worst_local_root'])
print('full20 reference maxima',report['full20']['worst_target_a_half_reference'])
