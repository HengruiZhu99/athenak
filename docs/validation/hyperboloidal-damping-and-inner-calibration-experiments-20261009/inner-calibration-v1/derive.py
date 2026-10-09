from pathlib import Path
import hashlib,json,subprocess,sys,time
import sympy as sy,mpmath as mp
p=Path(__file__).resolve().parent;repo=p.parents[2]
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
inputs=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]+[p/'actual_core_gauge.cpp',p/'derive.py']
before={str(f.relative_to(repo)):sha(f) for f in inputs};start=time.monotonic()
R,A,M,C=sy.symbols('R alpha M C',positive=True);q=sy.Function('q')(R);ar=sy.Symbol('alpha_R',real=True)
G=A*A-1+2*M/R-C*C*(A+2)**2/R**4
# Implicit equation derivative gives the stationary modified-BM relation.
qp=C*ar/R**2-2*C*(A+2)/R**3;qq=C*(A+2)/R**2
assert sy.simplify(qq*ar-(A+2)*(qp+2*qq/R))==0
# ADM Hamiltonian/momentum identities use q^2=alpha^2-F and its derivative.
F=1-2*M/R;Q2=A*A-F;QQp=A*ar-M/R**2
H=2*(1-A*A-2*R*A*ar)/R**2+4*QQp/R+2*Q2/R**2
assert sy.simplify(H)==0
qfun=sy.Function('q')(R);momentum=2*(sy.diff(qfun,R)-qfun/R)/R-2*sy.diff(qfun/R,R);assert sy.simplify(momentum)==0
rt=(sy.sqrt(13)-1)/4;Rc=sy.simplify(1/rt);Ac=rt-sy.Rational(1,2);C2=sy.simplify(Ac*Rc**4/(Ac+2));assert sy.simplify(C2-sy.Rational(8,243)*(13*sy.sqrt(13)-35))==0
assert sy.simplify(G.subs({R:Rc,A:Ac,M:1,C:sy.sqrt(C2)}))==0
assert sy.simplify(sy.diff(G,A).subs({R:Rc,A:Ac,M:1,C:sy.sqrt(C2)}))==0
assert sy.simplify(sy.diff(G,R).subs({R:Rc,A:Ac,M:1,C:sy.sqrt(C2)}))==0
# Signed factored discriminant removes the spurious absolute-value cusp at Rc.
Q3=R**3+(2*Rc-2)*R**2+(3*Rc**2-4*Rc)*R+4*Rc**3-6*Rc**2
P5=R**5-2*R**4+3*C2*R+2*C2
assert sy.simplify(sy.expand(P5-(R-Rc)**2*Q3))==0
assert all(bool(sy.simplify(z)>0) for z in sy.Poly(Q3,R).all_coeffs())
# Endpoint formulas, exact modulo the correct quartic root equation.
x,c2=sy.symbols('x c2',positive=True);a1=(3-2*x)*x*x/(2*c2);q0=2*sy.sqrt(c2)/x**2
pp=x*a1;vv=q0/x;kk=(3-2*x)/(q0*x*x)
assert sy.simplify(pp*vv-2*kk)==0
etaR=sy.factor(vv*((x*a1*a1-5*a1)/2-3/x))
report={'scope':'Exact Schwarzschild stationary modified-BM foliation and conditional isotropic inner asymptotics; pointwise actual core gauge only. No dynamic formation, global hyperboloidal matching, native BH evolution or uniform puncture hyperbolicity claim.',
 'symbolic':{'implicit_equation':str(G),'critical_R_over_M':str(Rc),'critical_alpha':str(Ac),'C2_over_M4':str(C2),'correct_alpha_zero_quartic':'x^4-2x^3+4C2=0','signed_discriminant_Q3':str(sy.simplify(Q3)),'Hamiltonian_exact_zero':True,'momentum_exact_zero':True,'pv_equals_2K_exact':True,'eta_required':'(q/R)*(1-3alpha+alpha*R*alpha_R/(alpha+2))','eta_derivative_at_endpoint':str(etaR)},
 'primary':'https://arxiv.org/pdf/0905.0450','primary_scope':'Eqs28-31 match derived BM implicit/critical equations after extrinsic-curvature sign conversion; printed R0=1.3955M is inconsistent with those equations, and is not a target.'}
mp.mp.dps=80
rt=(mp.sqrt(13)-1)/4;rc=1/rt;ac=rt-mp.mpf('.5');c2=ac*rc**4/(ac+2);cc=mp.sqrt(c2)
roots=mp.polyroots([1,-2,0,0,4*c2],maxsteps=1000);positive=sorted([z.real for z in roots if abs(z.imag)<mp.mpf('1e-60') and z.real>0]);r0=positive[0]
assert 1<r0<mp.mpf('1.5')<rc<positive[1]<2
poly=lambda z:z**4-2*z**3+4*c2
assert abs(poly(r0))<mp.mpf('1e-75')
def cubic(z):return z**3+(2*rc-2)*z*z+(3*rc*rc-4*rc)*z+4*rc**3-6*rc*rc
def alpha(z):return (2*c2+(z-rc)*mp.sqrt(z**3*cubic(z)))/(z**4-c2)
def slope(z):return mp.diff(alpha,z)
def values(z):
 a=alpha(z);ap=slope(z);q=cc*(a+2)/z**2;v=q/z;K=q*ap/(a+2);eta=v*(1-3*a+a*z*ap/(a+2));return a,ap,q,v,K,eta
a1=slope(r0);a2=mp.diff(alpha,r0,2);pp=a1*r0;vv=2*cc/r0**3;kk=(3-2*r0)/(2*cc);etaR=vv*((r0*a1*a1-5*a1)/2-3/r0)
assert abs(alpha(r0))<mp.mpf('1e-75') and a1>0 and abs(pp*vv-2*kk)<mp.mpf('1e-75')
assert etaR<0
# Critical slope from implicit l'Hopital quadratic independently agrees.
z=rc*slope(rc);assert abs(4*z*z/(ac+2)+16*ac*z-6*rt)<mp.mpf('1e-70')
report['high_precision']={k:mp.nstr(v,75) for k,v in {'Rc_over_M':rc,'alpha_c':ac,'C_over_M2':cc,'C2_over_M4':c2,'R0_over_M':r0,'other_alpha_zero_root_over_M':positive[1],'q0':vv*r0,'M_alphaPrime0':a1,'p':pp,'M_v':vv,'M_K0':kk,'M2_etaRequired_R_at_R0':etaR,'M_alphaPrime_c':slope(rc),'eta_inner_for_M_half':2*vv}.items()}
# Normalize the free isotropic radial scale by r(R=2M)=.01, just for core tests.
reg0=-(a1+r0*a2/2)/(pp*pp)
def regular(z):
 if abs(z-r0)<mp.mpf('1e-35'):return reg0
 return 1/(alpha(z)*z)-1/(pp*(z-r0))
def isotropic(z):return mp.mpf('.01')*((z-r0)/(2-r0))**(1/pp)*mp.exp(mp.quad(regular,[2,rc,z]))
report['radial_scale']='For numerical checks only: arbitrary isotropic normalization r(R=2M)=.01. No outer hyperboloidal matching imposed.'
report['samples']=[]
for z in [r0+mp.mpf(10)**(-j) for j in [1,2,3,4,6,8,10,12]]+[rc,mp.mpf(2),mp.mpf(4),mp.mpf(10),mp.mpf(100)]:
 a,ap,q,v,K,eta=values(z);assert a>=0 and ap>0
 assert abs(a*a-(1-2/z)-q*q)<mp.mpf('1e-70')
 assert abs(q*ap-(a+2)*K)<mp.mpf('1e-70')
 rr=isotropic(z) if z<=2 else None
 report['samples'].append({key:mp.nstr(val,60) for key,val in {'R_over_M':z,'alpha':a,'M_alphaPrime':ap,'q':q,'M_K':K,'M_q_over_R':v,'M_eta_required':eta,'constant_eta_shift_residual_over_beta':eta-vv}.items()}|({'isotropic_r':mp.nstr(rr,60)} if rr is not None else {}))
# Compile and check the actual flat-core gauge, not a replacement spherical evolution.
base=json.loads((p.parent/'covariant-z4-candidate/receipt.json').read_text());flags=base['commands'][0]['command'][:-3]
command=flags+[str(p/'actual_core_gauge.cpp'),'-o',str(p/'actual_core_gauge')]
r=subprocess.run(command,cwd=repo,text=True,capture_output=True);compile_receipt={'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};r.check_returncode();assert not r.stderr
checks=[]
for sm in report['samples']:
 if 'isotropic_r' not in sm:continue
 z=mp.mpf(sm['R_over_M']);rr=mp.mpf(sm['isotropic_r']);a,ap,q,v,K,eta=values(z)
 mass=mp.mpf('.5');physicalR=mass*z;chi=(rr/physicalR)**2;beta=rr*v/mass;db=eta/mass;da=ap/mass*(a*physicalR/rr);dc=2*chi*(1-a)/rr
 for name,rate in [('zero',mp.mpf(0)),('leading',vv/mass),('pointwise_required',eta/mass)]:
  args=[rr,a,da,K/mass,beta,db,chi,dc,rate];cmd=[str(p/'actual_core_gauge')]+[format(float(x),'.17g') for x in args]
  r=subprocess.run(cmd,cwd=repo,text=True,capture_output=True);r.check_returncode();out=json.loads(r.stdout)
  assert out['weight']==0 and abs(out['alpha_rhs'])<1e-12
  assert abs(out['beta_rhs']-out['driver_expression'])<1e-12
  if name=='pointwise_required':assert abs(out['beta_rhs'])<1e-14
  checks.append({'R_over_M':float(z),'eta_mode':name,'eta':float(rate),'command':cmd,'returncode':r.returncode,'actual':out})
report['actual_core_gauge']={'rows':len(checks),'checks':checks,'compile':compile_receipt,'binary_sha256':sha(p/'actual_core_gauge')}
(p/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
after={str(f.relative_to(repo)):sha(f) for f in inputs};assert before==after
receipt={'passed_stationary_inner_calibration':True,'native_BH_or_global_hyperboloidal_stationarity_claim':False,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':after,'sources_unchanged':True,'compile':compile_receipt,'result_sha256':sha(p/'result.json'),'seconds':time.monotonic()-start,'python':sys.version,'sympy':sy.__version__,'mpmath':mp.__version__}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS exact stationary modifiedBM/isotropic calibration; no nativeBH',report['high_precision']);print('actual core rows',len(checks))
