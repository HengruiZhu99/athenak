#!/usr/bin/env python3
"""Exact commuting-differential-operator proof for the actual C0 flat core.

Symbols d_i denote Cartesian differential operators, not Fourier frequencies.
This proves a continuum polynomial identity on the det/trace tangent chart.
It does not assert that arbitrary numerical first/second stencils commute in
the identities needed by that continuum chart.
"""
from pathlib import Path
import hashlib
import json
import sympy as s

d=s.symbols('d_x d_y d_z');kap=s.symbols('kappa_input',real=True)
u=s.symbols('chi gxx gxy gxz gyy gyz gzz P Axx Axy Axz Ayy Ayz Azz Lambda_x Lambda_y Lambda_z Theta alpha beta_x beta_y beta_z')
chi,P,theta,alpha=u[0],u[7],u[17],u[18]
g=s.Matrix([[u[1],u[2],u[3]],[u[2],u[4],u[5]],[u[3],u[5],u[6]]])
A=s.Matrix([[u[8],u[9],u[10]],[u[9],u[11],u[12]],[u[10],u[12],u[13]]])
beta=s.Matrix(u[19:22]);lam=s.Matrix(u[14:17]);lap=sum(z*z for z in d)
divbeta=sum(d[i]*beta[i] for i in range(3));divlam=sum(d[i]*lam[i] for i in range(3))
Gamma=s.Matrix([sum(d[j]*g[i,j] for j in range(3)) for i in range(3)])
f=s.zeros(22,1);f[0]=s.Rational(2,3)*(P+2*theta-divbeta)
f[7]=-lap*alpha+kap*theta;f[17]=lap*chi+divlam/2-2*kap*theta;f[18]=-3*P
trace=-lap*alpha+2*lap*chi+divlam
slots=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
for q,(i,j) in enumerate(slots):
    delta=int(i==j)
    f[1+q]=-2*A[i,j]+d[i]*beta[j]+d[j]*beta[i]-s.Rational(2,3)*delta*divbeta
    f[8+q]=-d[i]*d[j]*alpha-lap*g[i,j]/2+d[i]*d[j]*chi/2+(d[i]*lam[j]+d[j]*lam[i])/2+delta*(lap*chi/2-trace/3)
for i in range(3):
    f[14+i]=lap*beta[i]+d[i]*divbeta/3-s.Rational(4,3)*d[i]*P-s.Rational(2,3)*d[i]*theta-kap*(lam[i]-Gamma[i])
    f[19+i]=s.Rational(3,8)*lam[i]
H=2*lap*chi+sum(d[i]*d[j]*g[i,j] for i in range(3) for j in range(3))
M=s.Matrix([sum(d[j]*A[i,j] for j in range(3))-s.Rational(2,3)*d[i]*(P+2*theta) for i in range(3)])
Z=(lam-Gamma)/2;q=s.Matrix([H,*M,*Z,theta])
divM=sum(d[i]*M[i] for i in range(3));divZ=sum(d[i]*Z[i] for i in range(3))
rate=s.Matrix([-2*divM,
    *[-d[i]*H/2+lap*Z[i]-d[i]*divZ+2*kap*d[i]*theta for i in range(3)],
    *[M[i]+d[i]*theta-kap*Z[i] for i in range(3)],H/2+divZ-2*kap*theta])
C=q.jacobian(u);L=f.jacobian(u)
# Independent raw22 -> complete20 algebraic tangent chart.  At this flat
# stationary point gzz=-gxx-gyy and Azz=-Axx-Ayy, with all spatial jets obeying
# the same identities.  No stronger asymptotic falloff is imposed.
keep=[i for i in range(22) if i not in (6,13)];T=s.zeros(22,20)
for col,row in enumerate(keep):T[row,col]=1
T[6,keep.index(1)]=-1;T[6,keep.index(4)]=-1
T[13,keep.index(8)]=-1;T[13,keep.index(11)]=-1
residual=(C*f-rate).applyfunc(s.expand)
raw_residual=[str(v) for v in residual]
constrained=residual.subs({u[6]:-u[1]-u[4],u[13]:-u[8]-u[11]},simultaneous=True).applyfunc(s.expand)
assert constrained==s.zeros(8,1)
v=s.symbols('H M_x M_y M_z Z_x Z_y Z_z Theta_phys')
Hv=v[0];Mv=v[1:4];Zv=v[4:7];th=v[7];dvM=sum(d[i]*Mv[i] for i in range(3));dvZ=sum(d[i]*Zv[i] for i in range(3))
subs=s.Matrix([-2*dvM,*[-d[i]*Hv/2+lap*Zv[i]-d[i]*dvZ+2*kap*d[i]*th for i in range(3)],
              *[Mv[i]+d[i]*th-kap*Zv[i] for i in range(3)],Hv/2+dvZ-2*kap*th])
B=subs.jacobian(v);matched=(C*L*T-B*C*T).applyfunc(s.expand)
assert matched==s.zeros(8,20)
result={'passed_exact_flat_core_constraint_rate_identity':True,
    'identity':'C(d) L(d) T = B(d) C(d) T',
    'raw22_order':[str(x) for x in u], 'physical8_order':[str(x) for x in v],
    'kappa_input':'arbitrary; numerical driver sets10', 'kappa2':0,
    'core_gauge':'alpha_t=-3P, beta_t=(3/8)Lambda, no inner restoring',
    'matrix_dimensions':{'C':[8,22],'L':[22,22],'T':[22,20],'B':[8,8]},
    'matched_entries':160,'exact_nonzero_matched_entries':0,
    'formal_raw22_extension_chain_residual':raw_residual,
    'raw_extension_caveat':'C shown in raw22 is the algebraic extension of the tracefree formula; actual off-normal constraints are not asserted equal',
    'constraint_rate_formulas':[str(s.expand(x)) for x in subs],
    'physical_lift':'chi=-tau/sqrt(3), g=h_STF, A=independent_STF_A; Kphysical=P+2Theta',
    'scope':'commuting Cartesian continuum differential identity at exact flat reference; no radial/FD/stability theorem',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
print(json.dumps(result,indent=2))
