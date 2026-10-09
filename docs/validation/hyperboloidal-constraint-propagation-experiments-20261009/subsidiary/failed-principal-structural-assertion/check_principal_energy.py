"""Exact frozen principal constraint-wave reduction, no lower-order claim."""
import sympy as s
A=s.symbols('A',positive=True);k=s.Matrix(s.symbols('kx ky kz',real=True));I=s.I;k2=(k.T*k)[0]
G=s.zeros(8)
for i in range(3):
 G[0,1+i]=-2*I*A*k[i]
 G[1+i,0]=-I*A*k[i]/2
 G[1+i,4+i]=-A*k2
 for j in range(3):G[1+i,4+j]+=A*k[i]*k[j]
 G[4+i,1+i]=A;G[4+i,7]=I*A*k[i];G[7,4+i]=I*A*k[i]
G[7,0]=A/2
T=s.eye(8)
for i in range(3):T[0,4+i]=2*I*k[i];T[1+i,7]=I*k[i]
R=s.simplify(T*G*T.inv());expected=s.zeros(8)
expected[0,7]=-2*A*k2;expected[7,0]=A/2
for i in range(3):expected[1+i,4+i]=-A*k2;expected[4+i,1+i]=A
assert R==expected
W=s.diag(s.Rational(1,4),1,1,1,k2,k2,k2,k2)
assert s.simplify(R.conjugate().T*W+W*R)==s.zeros(8)
assert T.det()==1
print('PASS exact four-wave principal reduction and conserved positive Fourier energy for |k|>0; no lower-order/scri/boundary estimate.')
