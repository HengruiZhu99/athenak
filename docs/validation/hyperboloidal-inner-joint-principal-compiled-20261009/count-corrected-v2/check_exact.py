"""HELD standard-library exact algebra proposal. Not executed in preparation."""
from fractions import Fraction as F
import json


def mm(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def identity(n):return [[F(i==j) for j in range(n)] for i in range(n)]


def coefficients(mu):
    ec=2*mu*mu/(1+mu)**2
    q=(4*mu-2*ec)/3
    C=2*(1+mu)**2/(4*mu*mu+5*mu+3)
    return ec,q,C


def scalar(f,mu,ec):
    return [list(map(F,row)) for row in [
        [0,0,0,-f,0,0,0,0],
        [0,0,0,F(2,3),F(4,3),0,0,F(-2,3)],
        [0,0,0,0,0,-2,0,F(4,3)],
        [-1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,F(1,2),0],
        [F(-2,3),F(1,3),F(-1,2),0,0,0,F(2,3),0],
        [0,0,0,F(-4,3),F(-2,3),0,0,F(4,3)],
        [-1,ec,0,0,0,0,mu,0]]]


def transform(f,C):
    # Input ell,cchi,h,pi,vartheta,A,Lambda,beta.
    return [list(map(F,row)) for row in [
        [1,0,0,0,0,0,0,0],
        [0,0,0,-f,0,0,0,0],
        [0,1-2*C,0,0,0,0,-C,0],
        [0,0,0,F(2,3),F(4,3)-2*C,0,0,F(-2,3)],
        [0,2,1,0,0,0,0,0],
        [0,0,0,F(4,3),F(8,3),-2,0,0],
        [0,2,0,0,0,0,1,0],
        [0,0,0,0,2,0,0,0]]]


def inverse(f,C):
    # Input ell,ell_t,X,X_t,H,H_t,V,V_t; exact pencil inverse.
    return [list(map(F,row)) for row in [
        [1,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,C,0],
        [0,0,-2,0,1,0,-2*C,0],
        [0,-1/f,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,F(1,2)],
        [0,F(-2,3)/f,0,0,0,F(-1,2),0,F(2,3)],
        [0,0,-2,0,0,0,1-2*C,0],
        [0,-1/f,0,F(-3,2),0,0,0,1-F(3,2)*C]]]


def poly_mul(a,b):
    z=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):z[i+j]+=x*y
    return z


def main():
    # Numerator q-1 = (mu-1)(4mu²+5mu+3), denominator3(1+mu)².
    assert poly_mul([-1,1],[3,5,4])==[-3,-2,1,4]
    assert [1,-2,1]==poly_mul([-1,1],[-1,1])
    rows=[]
    for mu in map(F,['1/8','3/8','3/4','1','2','8']):
        ec,q,C=coefficients(mu)
        assert mu>0 and q>0 and C*(q-1)==2*(mu-1)/3
        for f in [F(1),F(3),q]:
            M=scalar(f,mu,ec);T=transform(f,C);I=inverse(f,C)
            N=[[F(0)]*8 for _ in range(8)]
            for start,speed2 in zip(range(0,8,2),[f,q,F(1),F(1)]):
                N[start][start+1]=1;N[start+1][start]=speed2
            assert mm(T,M)==mm(N,T)
            assert mm(T,I)==identity(8) and mm(I,T)==identity(8)
            rows.append({'f':str(f),'mu':str(mu),'ec':str(ec),'q':str(q),'C':str(C)})
    print(json.dumps({'passed':True,'exact_scalar_cases':len(rows),'cases':rows,
                      'scope':'Exact fixed-rational pencil algebra only; no kernel or evolution'},indent=2))


if __name__=='__main__':main()
