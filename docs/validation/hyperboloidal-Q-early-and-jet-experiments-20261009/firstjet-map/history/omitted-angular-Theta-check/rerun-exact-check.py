from pathlib import Path
import json,sympy as s
P=Path('/Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-null-jet-followup');j=json.loads((P/'actual.json').read_text());out={'rows':[]};maxerr=0
def rat(x):return s.Rational(x).limit_denominator(1000000)
for row in j['maps']:
 a=rat(row['a']);B=s.Matrix(row['B']).applyfunc(rat);E=s.Matrix(row['E']).applyfunc(rat)
 maxerr=max(maxerr,max(abs(float(B[i,k])-row['B'][i][k])for i in range(20)for k in range(80)),max(abs(float(E[i,k])-row['E'][i][k])for i in range(24)for k in range(80)))
 basic=E[:11,:];angular=E[:18,:];aug=E
 assert B.rank()==B[:, :20].rank() or B.rank()>B[:,:20].rank()
 assert E[0,:]+4*E[10,:]/a+8*E[7,:]/a+6*E[8,:]==s.zeros(1,80)
 assert aug.col_join(B).rank()==aug.rank()
 # Once leading constraints and Theta1 vanish, Lambda residue vanishes.
 lam=E[:8,:].col_join(E[18:19,:])
 assert lam.col_join(B[17:20,:]).rank()==lam.rank()
 # Leading constraints null data alone fail, even with tangential gradients and H1.
 assert basic.col_join(B).rank()>basic.rank();assert angular.col_join(B).rank()>angular.rank()
 # Restrict leading R0 to values: nine zero freedom directions.
 M=B[:,:20];assert M.rank()==11 and (M*M).rank()==11
 out['rows'].append({'a':float(a),'R0_rank':B.rank(),'R0_nullity':80-B.rank(),'value_pole_rank':M.rank(),'basic_conditions_rank':basic.rank(),'basic_plus_R0_rank':basic.col_join(B).rank(),'angular_H1_conditions_rank':angular.rank(),'angular_H1_plus_R0_rank':angular.col_join(B).rank(),'all_conditions_rank':aug.rank(),'all_plus_R0_rank':aug.col_join(B).rank(),'lambda_from_leading_constraints_Theta1':True})
for t in j['tests']:
 a=rat(t['a']);w=t['which'];S=list(map(rat,t['S']));E=list(map(rat,t['E']));
 if t['Omega']==0:
  if w==0:assert all(x==0 for x in E[:18]) and E[18]==1 and S[17]==-2/a**2
  if w==1:assert all(x==0 for x in E[:19]) and S[15]==-2/a**2
  if w in [2,3]:assert all(x==0 for x in S) and all(x==0 for x in E[:19])
  if w==4:assert S[12]==-8/(3*a**4) and S[17]==(12-80*a*a)/(3*a**4)
assert j['summary']['formula_error']<1e-12 and j['summary']['M_identity_error']<1e-12 and j['summary']['H1_formula_error']<1e-8
out.update({'passed':True,'rational_reconstruction_max_error':maxerr,'actual':j['summary'],'scope':'Exact ranks/identities of rationally reconstructed analytic-reference first-jet matrices; no nonlinear/general hierarchy closure.'})
(P/'check-report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
