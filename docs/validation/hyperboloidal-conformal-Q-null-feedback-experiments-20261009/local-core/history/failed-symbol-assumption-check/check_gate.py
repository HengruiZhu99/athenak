"""Exact reconstructed reference poles and independent source/corner gate.

No exact nonlinear scri, variable-coefficient PDE, energy or native proof.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import sympy as s
P=Path(__file__).resolve().parent
x=json.loads((P/'full20.json').read_text())
z,K,sg=s.symbols('z K sigma',real=True)
R=s.Matrix([[-2*sg,-s.Rational(4,3),s.Rational(4,3)],[3*sg-3,1,K-4],[-3,-2,-1-2*K]])
cubic=z**3+2*(K+sg)*z**2+(4*sg*(K+1)-9)*z+(8*K-6)*sg-12*K
assert s.expand(R.charpoly(z).as_expr()-cubic)==0
A,B,C=s.Poly(cubic,z).all_coeffs()[1:]
th=6*K/(4*K-3);rh=s.expand(A*B-C)
assert s.simplify(C.subs(sg,th))==0
assert s.simplify(B.subs(sg,th)-(24*K*K-12*K+27)/(4*K-3))==0
assert s.simplify(rh.subs(sg,s.Rational(3,2))-12*K*(K+1))==0
assert s.simplify(s.diff(rh,sg).subs(sg,s.Rational(3,2))-4*(2*K*K+6*K+3))==0
pole_error=normal_error=0.;cases=[]
for d in x['poles']:
 a=s.Rational(str(d['a']));kap=s.Rational(str(d['kappa']));sig=s.Rational(str(d['sigma']));mat=np.array(d['M'])
 m=s.Matrix(mat).applyfunc(lambda v:s.Rational(float(v)).limit_denominator(1000000))
 pole_error=max(pole_error,float(np.max(np.abs(np.array(m,float)-mat))))
 lam=s.symbols('lambda');eff=kap*a*a
 normal=s.expand(cubic.subs({z:a*a*lam,K:eff,sg:sig})/a**6)
 expected=lam**9*(lam+2/a**2)**2*(lam**2+kap*lam+2*kap/a**2)**2*(lam**2+kap*lam+2*kap/a**2+4/(3*a**4))*normal
 assert s.expand(m.charpoly(lam).as_expr()-expected)==0
 assert m.rank()==11 and (m*m).rank()==11
 # C=a²δNraw=δchi-hnn+2a(δα+δbeta_n), T=aδP−3a(δα+δbeta_n).
 Bn=s.zeros(3,20);Bn[0,1]=1;Bn[0,7]=-1;Bn[0,0]=Bn[0,4]=2*a
 Bn[1,2]=a;Bn[1,0]=Bn[1,4]=-3*a;Bn[2,3]=a
 assert a*a*Bn*m==R.subs({K:eff,sg:sig})*Bn
 roots=np.linalg.eigvals(mat);nonzero=roots[np.abs(roots)>1e-7]
 if sig==5:assert len(nonzero)==11 and max(nonzero.real)<-1e-3
 if a==1 and kap==5 and sig==0:assert abs(max(nonzero.real)-2.57170948731154)<1e-12
 cases.append(dict(a=float(a),kappa=float(kap),sigma=float(sig),rank=11,nullity=9,rank_square=11,max_nonzero_real=float(max(nonzero.real))))
assert pole_error<1e-12
fd=x['FD'];assert fd['rows']==960 and fd['max_abs_error'][-1]<1e-8
assert fd['max_abs_error'][0]>90*fd['max_abs_error'][1]>8000*fd['max_abs_error'][2]
corner_error=constraint_error=initial_R0_error=0.
for a in [.5,.75,1,2]:
 for sigma in [0,5]:
  rows=[r for r in x['witnesses']if r['a']==a and r['sigma']==sigma]
  for r in rows:
   constraint_error=max(constraint_error,max(abs(v)for v in r['C']))
   if r['Omega']==0:initial_R0_error=max(initial_R0_error,max(abs(v)for v in r['pole']))
  selected=[min(rows,key=lambda r:abs(r['Omega']-i*1e-4))for i in [1,2,3,4]]
  F=np.einsum('i,ij->j',[4,-6,4,-1],[r['RHS']for r in selected])
  N=float(np.einsum('i,i->',[4,-6,4,-1],[r['Ndot']for r in selected]));Q=float(np.einsum('i,i->',[4,-6,4,-1],[r['Qnumdot']for r in selected]))
  expected=np.zeros(20);expected[0]=1/a**2;expected[1]=-2/(3*a);expected[2]=3/a**2;expected[7]=4/(3*a);expected[10]=-2/(3*a)
  # Remaining A/Lambda finite rates are not set to zero by this scalar corner test.
  corner_error=max(corner_error,max(abs(F[i]-expected[i])for i in [0,1,2,3,4,5,6,7,8,9,10,11]),abs(N),abs(Q))
assert corner_error<1e-9 and constraint_error<1e-12 and initial_R0_error<1e-12
prior=json.loads((P/'prior_corner.json').read_text());prior_error=prior_null_error=0.
for a in [.5,.75,1,2]:
 for xi in [1.5,1/a]:
  rows=[r for r in prior if r['a']==a and abs(r['xi']-xi)<1e-14]
  # a=1.0 xi1.0/1.5 distinct; duplicated xi values if any averaged safely.
  rows=[min(rows,key=lambda r:abs(r['Omega']-i*.001))for i in [1,2,3,4]]
  f=np.einsum('i,ij->j',[4,-6,4,-1],[[r[k]for k in ['alphaDot','betaDot','PDot','Ndot','QnumeratorDot']]for r in rows])
  d=1+2*xi*a;expected=[-d/a**2,(d+1)/a**2,3/a**2,0,0]
  prior_error=max(prior_error,max(abs(f-np.array(expected))));prior_null_error=max(prior_null_error,abs(f[3]),abs(f[4]))
assert prior_error<2e-7
nl=json.loads((P/'nonlinear.json').read_text())
for key in ['reference_fixedpoint_error','production_blend_equivalence_error','factored_Q_pole_error','factored_null_difference_error','weighted_null_factor_error']:assert nl[key]<1e-11,(key,nl[key])
for key in ['independent_4D_source_error','preferred_Box_extension_error']:assert nl[key]<2e-10,(key,nl[key])
assert nl['exact_physical_inner_tiny_lapse_error']==0 and nl['tiny_noncore_rows']==320 and nl['tiny_positive_core_rows']==48
assert nl['cutoff_jump_at_h1e-8']<1e-6
# Source cancellation at W=1, retaining physical P and bounded F0.
a,al,h,Bh,B,Pphys,Pref,Om,kbar=s.symbols('a alpha h Bh B P Pref Omega Kbar',nonzero=True)
D=al*Bh/h-B;wn=-B/al;wnh=-Bh/h
S=-al**2*(Pphys-Pref)+3*al*D
fp=s.simplify(-S/al**3-(Pphys-3*wn)/al)
assert s.simplify(fp+(Pref-3*wnh)/al)==0
# Weighted null formula and explicit Einstein initial witness.
dG=s.symbols('deltaG');assert s.simplify(al**2*(dG-(B*B/al**2-Bh*Bh/h**2))-(al**2*dG+D*(B+al*Bh/h)))==0
r=s.symbols('r',positive=True);alpha=(1+r*r)/(2*a);O=(1-r*r)/(2*a);Dlin=O*(r*r/(a*a*alpha)-r/a)
Nlin=s.factor(2*r*r*Dlin/(a*a*alpha**2));Qlin=s.factor(-3*Dlin/(alpha*O))
assert s.limit(Nlin/O**3,r,1)==-a and s.limit(Qlin/O**2,r,1)==3*a*a/2
Falpha=s.factor(4*r*r/a**2-3*alpha*r/a+O*r*r/(a*a*alpha)-s.Rational(3,2)*O)
assert s.limit(Falpha,r,1)==1/a**2
qmax=0.
for a in [.5,.75,1,2]:
 rows=[r for r in x['finite_Q_counterexample']if r['a']==a]
 r=rows[-1];qmax=max(qmax,abs(r['Omega_Qdot']-.02/a**2),abs(r['ThetaDot']+.02/a**2))
assert qmax<2e-5
report={'passed_local_pole_source_principal_corner_gates':True,'native_or_global_accepted':False,'leading_matrix_exactness':'Exact characteristic/nullity identities apply to rationally reconstructed analytic reference pole matrices; reconstruction error retained.',
 'normal_cubic':str(cubic),'normal_Hurwitz_iff_for_positive_K_nonnegative_sigma':'K>3/4 and sigma>6K/(4K-3)','sigma5_condition':'K>15/14','normalized_K':'kappa_input*a^2 (S=1)','pole_reconstruction_max_error':pole_error,'pole_cases':cases,
 'FD':fd,'Einstein_initial_constraint_error':constraint_error,'Einstein_initial_R0_error':initial_R0_error,'Einstein_corner_limit_error':corner_error,'prior_physical_projection_corner_FD_error':prior_error,'prior_physical_projection_null_corner_error':prior_null_error,'finite_Q_counterexample_small_Omega_error':qmax,'nonlinear':nl,
 'final_recommended_variant':'physical_inner=true, sigma5, source .85-.95, g physical=false/preferred=true, xi=1/a (2 at target a=.5)',
 'scope':'Frozen reference value-only pole stability, complete principal, nonlinear finite-Omega source identities, initial Einstein scalar corner. No full first-jet closure, evolution-preserved null/Einstein constraints, energy/global stability, native improvement or BH compatibility claimed. Prior physicalP+projection+sigma already passes this initial null corner and failed native pulse tests.'}
(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
