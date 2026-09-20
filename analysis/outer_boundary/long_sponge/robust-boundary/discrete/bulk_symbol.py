"""Periodic frozen flat full Z4c symbol, verified against corner_model volume."""
import numpy as np,json
from pathlib import Path

def symbol(angles,h=32.,kappa=.1,eta=2.,ldamp=.1,discrete=True):
    ang=np.array(angles);eye=np.eye(20,dtype=complex);S=list(eye)
    d=1j*(1.5*np.sin(ang)-.3*np.sin(2*ang)+np.sin(3*ang)/30)/h if discrete else 1j*ang/h
    d2=(2*np.cos(3*ang)/90-.3*np.cos(2*ang)+3*np.cos(ang)-49/18)/h**2 if discrete else -ang**2/h**2
    ko=-.5/256/h*np.sum((2*np.cos(4*ang)-16*np.cos(3*ang)+56*np.cos(2*ang)-112*np.cos(ang)+70)) if discrete else 0.
    H=[[S[1],S[3],S[4]],[S[3],S[2],S[5]],[S[4],S[5],-S[1]-S[2]]]
    A=[[S[8],S[10],S[11]],[S[10],S[9],S[12]],[S[11],S[12],-S[8]-S[9]]]
    c,k,t,a=S[0],S[6],S[7],S[16];G=S[13:16];B=S[17:20]
    lap=sum(d2);db=sum(d[i]*B[i] for i in range(3));dg=sum(d[i]*G[i] for i in range(3))
    dd=lambda i,j,u:(d2[i] if i==j else d[i]*d[j])*u
    V=np.zeros((20,20),complex)
    V[0]=2/3*(k+2*t)-2/3*db;V[6]=-lap*a+kappa*t
    V[7]=.5*dg+lap*c-2*kappa*t;V[16]=-2*k-ldamp*a
    at=[[None]*3 for _ in range(3)]
    for i in range(3):
      for j in range(3):at[i][j]=-dd(i,j,a)-.5*lap*H[i][j]+.5*(d[i]*G[j]+d[j]*G[i])+.5*dd(i,j,c)
    tr=sum(at[i][i] for i in range(3))
    for (i,j),hi,ai in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6),range(8,13)):
      V[hi]=-2*A[i][j]+d[i]*B[j]+d[j]*B[i]-(2/3*db if i==j else 0)
      V[ai]=at[i][j]-(tr/3 if i==j else 0)
    for i in range(3):
      V[13+i]=-4/3*d[i]*k-2/3*d[i]*t+lap*B[i]+sum(dd(i,j,B[j]) for j in range(3))/3-2*kappa*(G[i]-sum(d[j]*H[i][j] for j in range(3)))
      V[17+i]=G[i]-eta*B[i]
    return V+ko*eye

if __name__=='__main__':
    out=[]
    for angle in [np.pi/8,np.pi/4,np.pi/2,3*np.pi/4,np.pi]:
      for direction in [(1,0,0),(1,1,0)]:
       for disc in [True,False]:
        for kap in [.1,0.]:
         ang=[angle*j for j in direction];V=symbol(ang,kappa=kap,discrete=disc);val=np.linalg.eigvals(V);i=np.argmax(val.real)
         out.append(dict(angles=ang,kappa=kap,discrete=disc,real=float(val[i].real),imag=float(val[i].imag)))
    Path('bulk-symbol.json').write_text(json.dumps(out,indent=2)+'\n')
    for x in out:print(json.dumps(x))
