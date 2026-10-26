"""Finite native constraint differential closure: analysis only, not Cauchy fields.

All coefficient commutators are retained. The input source and constraints are
the independently tensor-verified native8 linearization, unchanged RWM/damping.
"""
import mpmath as mp
from functools import lru_cache
import goal1_native_radial_linear as native

def matrices(o):
 eye=mp.eye(8)
 def M(s,con=False):
  f=native.constraints if con else native.action;cols=[f(s,o,eye[:,j])for j in range(8)];return mp.matrix([[cols[j][i]for j in range(8)]for i in range(len(cols[0]))])
 def coefficients(con=False):
  a,b,c=M(0,con),M(1,con),M(-1,con);return a,(b-c)/2,(b+c)/2-a
 a,b,c=coefficients();F=[a/o,b+c,o*c];a,b,c=coefficients(True);C=[a,o*(b+c),o*o*c];return F,C

def closure(o,precision=100):
 with mp.workdps(precision):
  o=mp.mpf(o);F,C=matrices(o);Fj=[[mp.diff(lambda z:matrices(z)[0][j],o,k)for k in range(3)]for j in range(3)];Cj=[[mp.diff(lambda z:matrices(z)[1][j],o,k)for k in range(3)]for j in range(3)]
  L=[mp.matrix(4,8)for _ in range(5)]
  for i in range(3):
   for j in range(3):
    for k in range(i+1):L[j+k]+=mp.binomial(i,k)*C[i]*Fj[j][i-k]
  B=mp.matrix(40,12);Y=mp.matrix(40,4)
  for n in range(5):
   for col in range(8):
    for a in range(4):Y[8*n+col,a]=L[n][a,col]
    for i in range(3):
     for j in range(3):
      k=n-j
      if 0<=k<=i:
       for b in range(4):B[8*n+col,4*i+b]+=mp.binomial(i,k)*Cj[j][i-k][b,col]
  X=(B.T*B)**-1*B.T*Y;K=[mp.matrix([[X[4*i+b,a]for b in range(4)]for a in range(4)])for i in range(3)];res=mp.norm(B*X-Y,p=mp.inf)/max(1,mp.norm(Y,p=mp.inf));return K,res

def natural(K,o):
 o=mp.mpf(o);a,b,c=K;return [a+2*b/o+2*c/o**2,b+4*c/o,c]

def indicial(K,o,s):
 o=mp.mpf(o);s=mp.mpf(s);a,b,c=natural(K,o);return o*a+s*b+s*(s-1)*c/o
