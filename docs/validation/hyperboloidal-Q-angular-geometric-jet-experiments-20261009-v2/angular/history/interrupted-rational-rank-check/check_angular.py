from pathlib import Path
import json,sympy as s
P=Path(__file__).resolve().parent;j=json.loads((P/'actual-release.json').read_text());rat=lambda x:s.Rational(x).limit_denominator(1000000)
ranks=[];err=0
for t in j['input_maps']:
 M=s.Matrix(t['M']).applyfunc(rat);assert M.rank()==31;ranks.append({'a':t['a'],'dir':t['dir'],'compatible_gauge_secondjet_basis_rank':31});err=max(err,max(abs(float(M[i,k])-t['M'][i][k])for i in range(40)for k in range(70)))
n1err=[0,0,0];R=C=0
for t in j['controls']:
 n=(.36,-.48,.8)if t['dir']else(1,0,0);ang=[1,n[1],n[2],n[1]**2,n[1]*n[2],n[2]**2];expected=4*(3-t['sigma'])*ang[t['col']-64]/t['a']**3 if t['col']>=64 else 0
 for z in range(3):n1err[z]=max(n1err[z],abs(t['N1_t'][z]-expected))
 R=max(R,t['next_R0_error']);C=max(C,t['next_leading_constraints_error'])
q=j['summary'];assert q['initial_constraints_error']==0 and q['initial_R0_error']<1e-12 and q['actual_ADM_first_RHS_geometry_field_error']<1e-9 and q['actual_arbitrary_angular_N_transport_identity_error']<1e-8
assert n1err[0]<1e-6 and n1err[-1]<5e-6 and R<1e-7 and C<1e-8
# Exact scalar transport/reaction identity, independently rederived by parent/literature.
r,a,sigma=s.symbols('r a sigma',positive=True);O=(1-r*r)/(2*a);T=-(r**4+6*r*r+1)/(4*a*r);Craw=(r**6+(16*sigma-29)*r**4+15*r*r-3)/(4*a*r*r*(r*r-1));weighted=s.factor(Craw+2*T*s.diff(O,r)/O)
assert s.simplify(weighted.subs(sigma,3)+(r*r+3)*(3*r*r-1)/(4*a*r*r))==0
report={'passed_gauge_only_reference_firstjet_checks':True,'full_Einstein_geometry_ideal_proved':False,'sigma3_admitted_for_evolution':False,'controls':len(j['controls']),'sampled_kernel_points':len(j['controls'])*12,'secondjet_basis_maps':ranks,'reconstruction_error':err,'N1_time_expected_map_max_error_by_h':[{'h':h,'error':e}for h,e in zip([.001,.0005,.00025],n1err)],'next_R0_max_error':R,'next_leading_constraint_max_error':C,'actual':q,'weighted_N_reaction':str(weighted),'sigma3_weighted_reaction':str(s.factor(weighted.subs(sigma,3))),'scope':'Gauge-only tangent data at fixed exact analytic-reference spatial geometry. 31-dimensional compatible secondjet space; exact transport plus finite compiled checks, with binary64 cancellation amplified at smallOmega. No nonlinear/geometrically perturbed ideal or stability/adoption claim.'}
(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
