"""Compiled frozen audit of spatial-norm feedback with xi=1/a at S=1.

The separate BH first-jet derivation supplies this pair. This script checks
Minkowski reference/poles/Fourier operators and cannot establish nonlinear
BH closure, a global spectrum, or native pulse stability.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import numpy as np

p=Path(__file__).resolve().parent;repo=p.parents[4]
spec=importlib.util.spec_from_file_location('physical',repo/'tst/hyperboloidal/check_physical_gauge.py')
physical=importlib.util.module_from_spec(spec);spec.loader.exec_module(physical)
t=np.eye(20);t[10,7]=.5;t[15,12]=.5;ti=np.linalg.inv(t)
report={'scope':__doc__,'poles':[],'fourier':[],
 'constraint_order':['H_phys','M_cov_x','M_cov_y','M_cov_z','Z_cov_x','Z_cov_y','Z_cov_z','Theta_phys','null_residue']}
maxpole=0.
for s in json.loads((p/'poles.json').read_text()):
 a=s['a'];sc=np.ones(20);sc[[0,2,3,4,5,6]]=1/a
 expected=physical.expected_matrix(s['kappa']*a*a,1+2*s['xi']*a,True)*sc[:,None]/sc[None,:]/(a*a)
 for i in (4,5,6):expected[i,i]-=s['eta']
 expected[4,1]-=s['eta']*s['Ccoeff'];expected[4,7]+=s['eta']*s['Ccoeff']
 expected=np.einsum('ij,jk,kl->il',ti,expected,t)
 m=np.asarray(s['B']);error=float(np.max(abs(m-expected)));maxpole=max(maxpole,error);assert error<2e-8
 ev=np.linalg.eigvals(m);nz=ev[abs(ev)>1e-6];rank=np.linalg.matrix_rank(m,1e-6)
 assert len(nz)==rank==15 and np.linalg.matrix_rank(np.einsum('ij,jk->ik',m,m),1e-5)==15
 if s['family']==3:assert max(nz.real)<0
 report['poles'].append({'family':s['family'],'a':a,'kappa':s['kappa'],'xi':s['xi'],'eta':s['eta'],'C':s['Ccoeff'],'rho':s['eta']*a*a,
  'max_nonzero_real':float(max(nz.real)),'semisimple_zero_count':5,'matrix_error':error})

def cutoff(r,lo,hi):
 if r<=lo:return 0.
 if r>=hi:return 1.
 x=(r-lo)/(hi-lo);v=-1/x+1/(1-x);e=math.exp(-abs(v));return e/(1+e) if v<=0 else 1/(1+e)

base={}
for name in ('fourier-matrices.json','fourier-mode-matrices.json'):
 for s in json.loads((p.parent/name).read_text()):
  if s['wide'] and not s['candidate'] and s['eps']==5e-7:base[(s['r'],s['kappa'],s['oblique'],s['k'])]=s
groups={}
for s in json.loads((p/'fourier.json').read_text()):groups.setdefault((s['family'],s['r'],s['kappa'],s['oblique'],s['k']),[]).append(s)
maxeps=0.;maxidentity=0.;maxconstraints=0.
for key,rows in groups.items():
 rows.sort(key=lambda s:s['eps'],reverse=True);old,s=rows
 m=np.asarray(s['B'])+1j*np.asarray(s['C']);mo=np.asarray(old['B'])+1j*np.asarray(old['C'])
 error=float(np.max(abs(m-mo)/(1+abs(m))));maxeps=max(maxeps,error);assert error<2e-6
 original=base[key[1:]];expected=np.asarray(original['B'])+1j*np.asarray(original['C'])
 W=cutoff(s['r'],.45,.85);w=cutoff(s['r'],.05,.95);alpha=math.hypot(s['omega'],2*s['r']*w)
 expected[0,0]-=2*alpha*W*(s['xi']-1.5)/s['omega']
 for i in (4,5,6):expected[i,i]-=s['eta']*W/s['omega']
 x=(s['r']-.05)/(.95-.05)
 dw=0. if x<=0 or x>=1 else w*(1-w)*(1/x**2+1/(1-x)**2)/(.95-.05)
 op=-dw*s['r']**2-2*w*s['r'];L=s['omega']-s['r']*op
 chi=(alpha/L)**(2/3);gxx=chi*L*L/(alpha*alpha)
 expected[4,1]-=s['eta']*W*s['Ccoeff']/(chi*s['omega'])
 expected[4,7]+=s['eta']*W*s['Ccoeff']/(gxx*s['omega'])
 error=float(np.max(abs(m-expected)/(1+abs(m))));maxidentity=max(maxidentity,error);assert error<2e-6,(key,error,np.unravel_index(np.argmax(abs(m-expected)/(1+abs(m))),m.shape))
 q=np.asarray(s['HB'])+1j*np.asarray(s['HC']);src=original if 'HB' in original else old
 qo=np.asarray(src['HB'])+1j*np.asarray(src['HC']);qe=float(np.max(abs(q-qo)/(1+abs(q))));maxconstraints=max(maxconstraints,qe);assert qe<2e-6
 ev,v=np.linalg.eig(m);i=int(np.argmax(ev.real));vector=v[:,i];vector/=max(abs(vector));residue=np.einsum('ij,j->i',q,vector)
 speeds=np.array([s['beta_n']-s['light_speed'],s['beta_n'],s['beta_n']+s['light_speed']])
 phase=float(ev[i].imag/s['k']) if s['k'] else None
 report['fourier'].append({k:s[k] for k in ('family','a','r','kappa','xi','eta','Ccoeff','k','oblique')}|
  {'max_real':float(ev[i].real),'root_imag':float(ev[i].imag),'phase_generator_speed':phase,
   'principal_phase_speeds':speeds.tolist(),
   'nearest_principal_branch':['fast_outgoing','advective','slow_incoming'][int(np.argmin(abs(speeds-phase)))] if phase is not None else None,
   'constraint_abs':abs(residue).tolist(),'mode_real':vector.real.tolist(),'mode_imag':vector.imag.tolist()})
report['max_full20_pole_error']=maxpole;report['max_epsilon_scaled_change']=maxeps
report['max_exact_lower_order_scaled_error']=maxidentity;report['max_constraint_map_change']=maxconstraints
import sympy as sp
k,rho,x,y=sp.symbols('k rho x y',real=True)
scalar=sp.Matrix([[-3,0,-1,0,1],[-2,0,sp.Rational(2,3),sp.Rational(4,3),-2],
 [0,-3,-2,k-4,0],[0,-3,-2,-1-2*k,0],[0,1-rho,0,0,-rho]])
poly=scalar.charpoly();lam=poly.gen;coeff=poly.all_coeffs()
assert sp.expand(poly.as_expr().subs(lam,0)+12*(k-1)*(rho-4))==0
H=sp.Matrix([[coeff[1],coeff[3],coeff[5],0,0],[1,coeff[2],coeff[4],0,0],
 [0,coeff[1],coeff[3],coeff[5],0],[0,1,coeff[2],coeff[4],0],[0,0,coeff[1],coeff[3],coeff[5]]])
deltas=[sp.factor(H[:i,:i].det()) for i in range(1,5)]
# Exact domain for 1<=rho<4,k>1: Delta1..3 and a5 are positive;
# Delta4 alone determines the remaining Hurwitz condition.
for expr in deltas[:3]:
 numerator=sp.cancel(expr.subs({k:1+y,rho:(1+4*x)/(1+x)})).as_numer_denom()[0]
 shifted=sp.Poly(numerator,x,y);assert all(v>0 for v in shifted.coeffs())
# Uniform sufficient interval 1<=rho<=5/2 for every k>1.
for expr in deltas:
 numerator=sp.cancel(expr.subs({k:1+y,rho:(1+sp.Rational(5,2)*x)/(1+x)})).as_numer_denom()[0]
 shifted=sp.Poly(numerator,x,y);assert all(v>0 for v in shifted.coeffs())
 endpoint=sp.Poly(sp.expand(expr.subs({k:1+y,rho:sp.Rational(5,2)})),y)
 assert all(v>0 for v in endpoint.coeffs())
assert sp.expand(deltas[3].subs(k,1)+16*(2*rho-5)*(17*rho**3+336*rho**2+1833*rho+1702))==0
for s in report['poles']:
 rr=s['rho'];kk=s['kappa']*s['a']**2
 leading=[float(v.subs({rho:rr,k:kk})) for v in coeff]
 actual=next(z for z in json.loads((p/'poles.json').read_text()) if z['family']==s['family'] and z['a']==s['a'] and z['kappa']==s['kappa'])
 roots=np.linalg.eigvals(np.asarray(actual['B']))
 for z in np.roots(leading)/s['a']**2:assert min(abs(roots-z))<2e-7
report['exact_scalar_quintic']=str(poly.as_expr())
report['exact_Delta4']=str(deltas[3])
report['exact_hurwitz_domain_in_positive_feedback_family']='k=kappa*a^2>1, 1<=rho<4, Delta4>0; uniform sufficient 1<=rho<=5/2. rho>4 has a positive real root; rho>5/2 loses uniform stability as k approaches 1.'
print('PASS exact scalar quintic/full20 root match and Routh domain; scaled rho1.5 gates')
(p/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS actual full20 norm-feedback poles and exact lower-order Fourier/principal update',maxpole,maxeps,maxidentity,maxconstraints)
for family in range(4):
 for kappa in (5,10):
  worst=max([s for s in report['fourier'] if s['family']==family and s['kappa']==kappa and s['r']>=.85],key=lambda s:s['max_real'])
  print('FROZEN norm-feedback family/kappa',family,kappa,'worst',worst['r'],worst['k'],worst['oblique'],worst['max_real'],worst['phase_generator_speed'],worst['nearest_principal_branch'],'constraints',worst['constraint_abs'])
receipt=json.loads((p/'receipt.json').read_text());receipt['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt['artifacts_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.json') if f.name!='receipt.json'}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
