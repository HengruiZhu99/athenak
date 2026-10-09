from pathlib import Path
import json,sympy as s
P=Path(__file__).resolve().parent;j=json.loads((P/'actual-release.json').read_text());a,r,O,sigma,m=s.symbols('a r O sigma m',positive=True);h=(1+r*r)/(2*a)
# Exact radial ADM/gauge equations, from the independently checked full20 field.
af=3*h*r*O**(m-1)/a;bf=(m-2*sigma)*r*r*O**(m-1)/a**2+(-6/a+2*r*r/(a*a*h))*O**m
wndot=r*bf/(a*h)+r*r*af/(a*a*h*h);Gdot=2*(m-1)*r**3*O**(m-1)/a**3;Ndot=Gdot+2*r*r*wndot/(a*a*h)
limits={};err=0;next_err=0
for mi in [2,3,4]:
 L=s.simplify(s.limit((Ndot/O**(m-1)).subs({m:mi,r:s.sqrt(1-2*a*O)}),O,0));assert s.simplify(L-4*(mi+1-sigma)/a**3)==0;limits[str(mi)]=str(L)
for t in j['rows']:
 if t['which']!=0:continue
 aa=t['a'];ss=t['sigma'];mm=t['m'];oo=t['Omega'];
 if oo:
  rr=(1-2*aa*oo)**.5;hh=(1+rr*rr)/(2*aa);A=3*hh*rr*oo**(mm-1)/aa;B=(mm-2*ss)*rr*rr*oo**(mm-1)/aa**2+(-6/aa+2*rr*rr/(aa*aa*hh))*oo**mm;wd=rr*B/(aa*hh)+rr*rr*A/(aa*aa*hh*hh);pred=2*(mm-1)*rr**3*oo**(mm-1)/aa**3+2*rr*rr*wd/(aa*aa*hh)
  err=max(err,abs(pred-t['Ndot']),abs(-3*wd-t['Qnumdot']))
 else:
  pred=4*(3-ss)/aa**3 if mm==2 else 0;next_err=max(next_err,abs(pred-t['next_N1']));assert max(map(abs,t['next_R0']))<1e-12
assert j['summary']['initial_constraints_error']==0 and j['summary']['initial_R0_error']<1e-12
assert j['summary']['exact_radial_RHS_field_error']<1e-12 and j['summary']['initial_4D_Box_identity_error']<1e-12
assert err<1e-12 and next_err<1e-12
report={'passed_initial_Einstein_and_finite_conformal_regularity_checks':True,'sigma5_quadratic_null_ideal_invariant':False,'sigma3_is_a_candidate_admission':False,'rows':len(j['rows']),'radial_rows':sum(t['which']==0 for t in j['rows']),'initial_boundary_rows':j['summary']['initial_boundary_controls'],'actual_summary':j['summary'],'exact_radial_normalized_limits':limits,'radial_Ndot_Qnumdot_formula_max_error':err,'actual_first_RHS_jet_next_N1_formula_error':next_err,'scope':'Linear analytic-reference actual Q/null kernel. Initial physical ADM/Z/Theta constraints zero, finite four-dimensional initial conformal shear/curvature. Failure of smooth-in-time quadratic-null Taylor tangency, not finite-Omega amplitude blowup or global Einstein evolution failure.'}
(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
