from pathlib import Path
import sympy as s,json,math
P=Path(__file__).resolve().parent;j=json.loads((P/'actual-release.json').read_text());basis=json.loads((P/'basis.json').read_text());x,y,z=s.symbols('x y z');vs=[x,y,z];mons=[1,x,y,z,x*x,x*y,x*z,y*y,y*z,z*z];err=[0,0,0];boundary={k:max(abs(t[k])for t in j['controls'])for k in ['initial_N0_boundary','initial_N1_boundary','initial_Q0_boundary','next_R0_error']}
expected=[]
for b in basis:
 Y=vs[b['axis']]*mons[b['col']%10];ell=s.Poly(Y,*vs).total_degree();L=sum(s.diff(Y,v,2)for v in vs);expected.append((s.lambdify(vs,Y,'math'),s.lambdify(vs,L,'math'),ell))
for t in j['controls']:
 b=basis[t['col']];n=[.36,-.48,.8]if t['dir']else[1,0,0];Y,L,ell=expected[t['col']];v=2*(L(*n)+(2*t['sigma']-4-ell*(ell+1))*Y(*n))/t['a']**3 if b['m']==1 else 0
 for k in range(3):err[k]=max(err[k],abs(t['N1_t_actual'][k]-v))
q=j['summary'];assert q['initial_constraints_max']<1e-12 and q['initial_R0_max']<1e-12
assert q['stationary_geometric_first_RHS_max']<1e-8 and q['initial_4D_Box_identity_max_error']<1e-5
assert q['initial_coordinate_shear_over_Omega_max']<100 and q['initial_coordinate_curvature_max']<1000
assert max(boundary[k]for k in ['initial_N0_boundary','initial_N1_boundary','initial_Q0_boundary'])<1e-12
assert boundary['next_R0_error']<1e-6 and max(err)<2e-5
# Independent stationary 4D radial metric/divergence calculation, with delta=Lie_xi.
r,a,sigma=s.symbols('r a sigma',positive=True);O=(1-r*r)/(2*a);h=(1+r*r)/(2*a);beta=-r/a;wref=-r*r/(a*a*h);G=1-beta*beta/h**2;volume=h*r*r
B=s.factor(s.diff(volume*G*s.diff(O,r),r)/volume);N=s.factor(G*s.diff(O,r)**2);xi=s.Function('xi')(r)
A=xi*s.diff(h,r)-h*xi*s.diff(O,r)/O;V=xi*s.diff(beta,r)-beta*s.diff(xi,r);rr=2*s.diff(xi,r)-2*xi*s.diff(O,r)/O;tt=2*xi/r-2*xi*s.diff(O,r)/O;dvol=A/h+rr/2+tt;dG=-rr-2*beta*V/h**2+2*beta**2*A/h**3
dB=s.factor(s.diff(volume*(dG+dvol*G)*s.diff(O,r),r)/volume-dvol*B);dN=s.factor(dG*s.diff(O,r)**2)
# Separate covariance identities for fixed prescribed Omega and the conformal factor.
w=xi*s.diff(O,r);f=w/O
Box=lambda F:s.diff(volume*G*s.diff(F,r),r)/volume
assert s.factor(dB-(xi*s.diff(B,r)-Box(w)+2*f*B-2*G*s.diff(f,r)*s.diff(O,r)))==0
assert s.factor(dN-(xi*s.diff(N,r)-2*G*s.diff(w,r)*s.diff(O,r)+2*f*N))==0
rates=[]
for X,label in [(O,'Omega*n'),(O*r,'Omega*x')]:
 sub={xi:X,s.diff(xi,r):s.diff(X,r),s.diff(xi,r,2):s.diff(X,r,2)};dn=s.factor(dN.subs(sub));db=s.factor(dB.subs(sub));ndot=s.factor(-2*wref*h*(db-sigma*dn/O));N2=s.limit(dn/O**2,r,1);Box1=s.limit(db/O,r,1);N1t=s.factor(s.limit(ndot/O,r,1));assert s.simplify(N2+2/a)==0 and s.simplify(Box1+4/a)==0 and s.simplify(N1t-4*(sigma-2)/a**3)==0;rates.append(dict(field=label,deltaN2=str(N2),deltaBox_stationary1=str(Box1),N1_time=str(N1t),exact_radial_Ndot=str(ndot)))
# Full angular m1 coefficient from the same covariance identity, without a radial ansatz.
o,ell,Y,L=s.symbols('o ell Y L');rrho=1-2*a*o;hh=1/a-o;nn=rrho*o**2/(a*a*hh*hh);cc=1/(a*hh)-4/(a*a*hh*hh)+rrho/(a**3*hh**3);bb=-o*(5*o**2*a**2-8*o*a+2)/(o*a-1)**3
boxY=L-ell*(ell-1)*Y/(a*a*hh*hh)+cc*ell*Y
boxw=-(o*boxY+Y*bb-2*ell*o**2*Y/(a*hh*hh))/a
dbb=-o*Y*s.diff(bb,o)/a-boxw-2*Y*bb/a-2*ell*o**2*Y/(a*a*hh*hh)
dnn=-o*Y*s.diff(nn,o)/a-2*ell*o**3*Y/(a*a*hh*hh)
angularN2=s.factor(s.limit(dnn/o**2,o,0));angularBox1=s.factor(s.limit(dbb/o,o,0));angularRate=s.factor(s.limit(2*rrho*(dbb-sigma*dnn/o)/(a*a*o),o,0))
assert s.simplify(angularN2+2*Y/a)==0
assert s.simplify(angularBox1-(L-(ell*(ell+1)+4)*Y)/a)==0
assert s.simplify(angularRate-2*(L+(2*sigma-4-ell*(ell+1))*Y)/a**3)==0
rad=[]
for aa in [.5,.75,1.,2.]:
 for sig in [3,5]:
  for direc in [0,1]:
   rows=[next(t for t in j['controls']if t['a']==aa and t['sigma']==sig and t['dir']==direc and t['col']==c)for c in [1,12,23]];values=[sum(t['N1_t_actual'][k]for t in rows)for k in range(3)];rad.append(dict(a=aa,sigma=sig,dir=direc,expected=4*(sig-2)/aa**3,actual=values))
report={'passed_actual_linear_Einstein_pullback_and_negative_tangency_gate':True,'sigma3_full_compatible_Taylor_ideal_invariant':False,'finite_Omega_amplitude_instability_inferred':False,'sigma3_evolution_admitted':False,'controls':len(j['controls']),'sampled_kernel_points':12*len(j['controls']),'input_fields':len(basis),'initial_boundary_checks':boundary,'actual':q,'N1_time_expected_angular_map_max_error_by_h':[dict(h=h,error=e)for h,e in zip([.001,.0005,.00025],err)],'independent_exact_radial_4D_divergence':rates,'actual_radial_Omega_x_controls':rad,'independent_exact_angular_m1_coefficient':{'N2':str(angularN2),'Box_stationary1':str(angularBox1),'N1_time':str(angularRate)},'m1_angular_formula':'N1_t=(2/a^3)[Delta_S Y+(2*sigma-4)Y], Y=n_j X(n)','constant_sigma_obstruction':'Gauge-only beta=Omega^2*n requires sigma3; Einstein spatial pullback xi=Omega*x requires sigma2 for smooth quadratic-null time tangency. No single constant sigma cancels both directions. This is not a finite-Omega amplitude blowup or full PDE stability statement.'}
(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
