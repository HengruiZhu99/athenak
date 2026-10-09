"""Exact linear outer-CMC constraint Taylor maps, matched to actual dual kernel."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,time
import sympy as s
import numpy as np
w=Path(__file__).resolve().parent;t0=time.monotonic();x,y,z,r,a,o=s.symbols('x y z r a Omega',positive=True);R=s.sqrt(x*x+y*y+z*z);om=(1-x*x-y*y-z*z)/(2*a);ny=y/R;nz=z/R;functions=[1,om,ny,nz,om**2,om*ny,om*nz,ny**2,ny*nz,nz**2];family_names=['u0','u1','D_y u0','D_z u0','u2','D_y u1','D_z u1','half_D_yy u0','D_yz u0','half_D_zz u0'];xyz=[x,y,z]
rows=['H0','H1','H2','M_x0','M_y0','M_z0','M_x1','M_y1','M_z1','Z_x0','Z_y0','Z_z0','Z_x1','Z_y1','Z_z1','Theta0','Theta1','Theta2'];field_names=['alpha','chi','P','Theta','beta_x','beta_y','beta_z','h_xx','h_xy','h_xz','h_yy','h_yz','A_xx','A_xy','A_xz','A_yy','A_yz','Lambda_x','Lambda_y','Lambda_z'];zero=[s.S.Zero]*8
@lru_cache(None)
def coeffs(expr):
 q=s.expand(expr.subs(r,s.sqrt(1-2*a*o)));q=s.series(q,o,0,3).removeO().expand();return [s.simplify(q.coeff(o,k)) for k in range(3)]
def at(expr):return s.sympify(expr).subs({x:r,y:0,z:0}).simplify()
expressions=[];Taylor=s.zeros(18,200);basis_jets=[]
for family,f in enumerate(functions):
 f=s.sympify(f);v=at(f);d=[at(s.diff(f,q)) for q in xyz];dd=[[at(s.diff(f,q,e)) for e in xyz] for q in xyz];basis_jets.append((v,d,dd))
 for col in range(20):
  C=zero.copy();divh=[s.S.Zero]*3
  if col==1:C[0]=o*o*2*sum(dd[i][i] for i in range(3))+4*o*(-3*v/a+r*d[0]/(2*a))-6*r*r*v/(a*a)
  if col==2:C[0]=-4*v/a;C[1:4]=[-s.Rational(2,3)*q for q in d]
  if col==3:C[0]=-8*v/a;C[1:4]=[-s.Rational(4,3)*q for q in d];C[7]=v
  if 7<=col<17:
   i,j=[(0,0),(0,1),(0,2),(1,1),(1,2)][(col-7)%5];E=s.zeros(3,3);E[i,j]=E[j,i]=1
   if i==j:E[2,2]=-1
   div=[sum(E[j,i]*d[j] for j in range(3)) for i in range(3)]
   if col<12:
    C[0]=o*o*sum(E[i,j]*dd[i][j] for i in range(3) for j in range(3))+4*o*r*div[0]/a+6*r*r*E[0,0]*v/(a*a);C[4:7]=[-q/2 for q in div]
   else:C[1:4]=[o*div[i]+2*r*E[0,i]*v/a for i in range(3)]
  if col>=17:C[4+col-17]=v/2
  expressions.append(C);cc=[coeffs(q) for q in C];flat=cc[0]+[cc[i][0] for i in range(1,4)]+[cc[i][1] for i in range(1,4)]+[cc[i][0] for i in range(4,7)]+[cc[i][1] for i in range(4,7)]+cc[7]
  for row,q in enumerate(flat):Taylor[row,family*20+col]=q
print('assembled exact analytic Taylor map',Taylor.shape,'seconds',time.monotonic()-t0,flush=True)
# Independent actual kernel samples at finite Omega on complete Cartesian angular/radial basis.
data=json.loads((w/'actual-kernel-release.json').read_text());formula_max=0;basis_max=0
fun=[s.lambdify((a,o,r),C,'numpy') for C in expressions]
for row in data['samples']:
 av=row['a'];ov=row['Omega'];rv=np.sqrt(1-2*av*ov);expected=np.asarray(fun[row['family']*20+row['col']](av,ov,rv),dtype=float);basis_max=max(basis_max,float(abs(expected-row['C']).max()))
# Exact formula for the R0-compatible scalar corner, no imposed Theta order beyond stated map constraints.
H1,L0,P1,T1,X1,nu,k,k2=s.symbols('H1 L0 P1 Theta1 X1 nu kappa kappa2')
h1=-4*(P1+2*T1)/a+4*L0-6*X1/a**2;theta_corner=2*L0/a-2*(P1-3*nu)/a**2-k*(2+k2)*T1-3*(X1/a**2+2*nu/a)/a
corner_identity=s.simplify(theta_corner-h1/(2*a)-(4/a**2-k*(2+k2))*T1);assert corner_identity==0
# Exact full-gradient momentum/connection residue identity, even before imposing R0 or tangential Theta0=0.
residue=[]
for av in [s.Rational(1,2),s.Rational(3,4),s.S.One,s.Integer(2)]:
 T=Taylor.subs(a,av);B=s.zeros(3,200);gradTheta=s.zeros(3,200)
 for fam,(v,d,dd) in enumerate(basis_jets):
  for i in range(3):gradTheta[i,fam*20+3]=s.simplify(d[i].subs(r,1).subs(a,av))
 # Formula R0Lambda=−2(kappa−2/a²)Z−2/(3a)grad(2P+Theta)+4A_n/a².
 for col in range(200):
  C=expressions[col];fam=col//20;f=col%20;v,d,dd=basis_jets[fam]
  for i in range(3):
   value=-2*(10-2/av**2)*T[9+i,col]
   if f in [2,3]:value-=s.Rational(2,3)/av*(2 if f==2 else 1)*d[i].subs(r,1).subs(a,av)
   if 12<=f<17:
    ii,jj=[(0,0),(0,1),(0,2),(1,1),(1,2)][f-12];E=s.zeros(3,3);E[ii,jj]=E[jj,ii]=1
    if ii==jj:E[2,2]=-1
    value+=4*E[0,i]*v.subs(r,1).subs(a,av)/av**2
   B[i,col]=value
 exact=T[3:6,:]-(av*10-2/av)*T[9:12,:]+gradTheta-av*B/2;assert exact==s.zeros(3,200)
 # Constant/first angular scalar values vanish on smooth R0-compatible boundary field.
 maps={str(rows[i]):[str(T[i,j]) for j in range(200)] for i in range(18)};residue.append({'a':str(av),'map':maps,'rank':int(T.rank()),'momentum_Z_Theta_gradient_R0Lambda_identity':True})
# Omega² P witness: H2=-4/a, M1=4n/(3a), all earlier H/M/Z/Theta vanish.
v=s.zeros(200,1);v[4*20+2]=1;witness=Taylor*v;assert witness[0]==witness[1]==0 and witness[2]==-4/a and witness[6]==4/(3*a);assert all(witness[i]==0 for i in list(range(3,6))+list(range(7,18)))
# Gauge-only alpha=Omega, beta=−Omega n has exactly zero ADM/Z4 constraints. Its initial N variation starts at Omega³.
alpha=(1+r*r)/(2*a);wn=-r*r/(a*a*alpha);delta_w=-o*r/(a*alpha)-wn*o/alpha;Ndelta=-2*wn*delta_w;Qdelta=-3*delta_w/o
Nseries=s.series(Ndelta.subs(r,s.sqrt(1-2*a*o)),o,0,4);Qseries=s.series(Qdelta.subs(r,s.sqrt(1-2*a*o)),o,0,3)
assert s.simplify(Nseries.removeO().coeff(o,0))==0 and s.simplify(Nseries.removeO().coeff(o,1))==0
corner_error=0;C0error=0;R0error=0
for row in data['Einstein_gauge_corners']:
 if row['which']!=0:continue
 av=row['a'];expected={'alpha0':-3/av**2,'beta_n0':1.5/av**2,'P0':3/av**2,'Theta0':0,'chi0':-2/(3*av),'h_nn0':4/(3*av),'R0alpha_t':7.5/av**4,'Nraw_t0':-5/av**3,'Qnumerator_t0':7.5/av**2};F=row['F0'];actual={'alpha0':F[0],'beta_n0':F[4],'P0':F[2],'Theta0':F[3],'chi0':F[1],'h_nn0':F[7],'R0alpha_t':(-F[2]-3*F[0]+F[4])/av**2,'Nraw_t0':row['Nraw_t0'],'Qnumerator_t0':row['Qnumerator_t0']};corner_error=max(corner_error,max(abs(actual[q]-expected[q]) for q in expected));C0error=max(C0error,max(abs(q) for q in row['C0']));R0error=max(R0error,row['R0_max'])
assert all(Taylor[row,fam*20+col]==0 for row in range(18) for fam in range(10) for col in [0,4,5,6])
output={'gauge_columns_identically_zero':True,'basis_families':family_names,'fields':field_names,'rows':rows,'column_order':'20 fields per scalar local angular/Taylor basis, family-major; u1/u2 are Taylor coefficients, not radial derivatives; quadratic angular columns use the stated monomials','scope':'Exact analytic linear physical constraint Taylor map on outer pureCMC Minkowski reference, matched to actual double dual kernel. Not arbitrary nonlinear scri closure or a proof of evolution preservation. H2/M1/Z1 require complete second jets; no M2/Z2 map claimed because higher jets would enter.','maps':residue,'full_finite_Omega_basis_checks':len(data['samples']),'finite_Omega_analytic_vs_actual_max_error':basis_max,'actual_summary':data['summary'],'exact_scalar_corner_identity':str(corner_identity),'Omega_squared_P_witness_map':[str(q) for q in witness],'gauge_witness_initial_Nraw_series':str(Nseries),'gauge_witness_initial_Q_series':str(Qseries),'gauge_witness_actual_corner_error':corner_error,'gauge_witness_initial_C0_error':C0error,'gauge_witness_initial_R0_error':R0error,'all_pass':basis_max<1e-11 and corner_error<1e-10 and C0error<1e-12 and R0error<1e-12 and data['summary']['R0_compatible_M0_Z0_Theta1_error']<1e-12 and data['summary']['Theta_t0_H1_Theta1_error']<1e-10,'seconds':time.monotonic()-t0}
(w/'check-report.json').write_text(json.dumps(output,indent=2)+'\n');print({k:v for k,v in output.items() if k not in ['maps']},flush=True);assert output['all_pass']
