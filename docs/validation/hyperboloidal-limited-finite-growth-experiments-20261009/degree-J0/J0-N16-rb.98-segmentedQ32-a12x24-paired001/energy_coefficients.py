"""Source-matched normal matrices and independent reference coefficient derivatives."""
import math
import numpy as np
from canceled_basis_complex import canceled_basis,scalar_matrix

def mm(a,b):return np.einsum('ik,kj->ij',a,b,optimize=False)
def normal_matrix(alpha,w):
    f=1+(1-w)*2/alpha;mu=(1-w)*3/8+w
    a=np.zeros((20,20),dtype=np.result_type(alpha,w,float));a[:8,:8]=scalar_matrix(f,mu,w,w/2)
    for s in (8,12):a[s:s+4,s:s+4]=[[0,-2,0,1],[-.5,0,.5,0],[0,0,0,1],[0,0,mu,0]]
    for s in (16,18):a[s:s+2,s:s+2]=[[0,-2],[-.5,0]]
    return a
A1=normal_matrix(1.,1.)
HD=np.eye(20)+mm(A1.T,A1)
def cutoff(r,a=.85,b=.9):
    if r<=a:return 0.,0.
    if r>=b:return 1.,0.
    s=(r-a)/(b-a);t=(b-r)/(b-a);g=-1/s+1/t;e=math.exp(-abs(g))
    w=e/(1+e) if g<=0 else 1/(1+e)
    return w,e/(1+e)**2*(1/s**2+1/t**2)/(b-a)
def hc(alpha,w):
    left,_=canceled_basis(alpha,w)
    return mm(left.T,left) # algebraic transpose, not complex conjugation
def coefficients(r,ref):
    # ref=(alpha,dalpha,ddalpha,chi,...,Omega,...,c,...,beta_n,...,W,dW).
    alpha,ad=ref[0:2];c=ref[9];beta,bd=ref[12:14];w,wd=ref[15:17]
    z,zd=cutoff(r);step=1e-30
    C=hc(alpha,w);Cd=hc(alpha+1j*step*ad,w+1j*step*wd).imag/step
    H=(1-z)*C+z*HD;Hr=(1-z)*Cd+zd*(HD-C)
    A=normal_matrix(alpha,w);Ar=normal_matrix(alpha+1j*step*ad,w+1j*step*wd).imag/step
    Kn=beta*np.eye(20)+alpha*A;Knr=bd*np.eye(20)+ad*A+alpha*Ar
    DsH=c*Hr;DsKn=c*Knr;Q=mm(H,Kn);div=2*c/r
    Gamma=mm(DsH,Kn)+mm(H,DsKn)+div*Q
    return {'H':H,'A':A,'Kn':Kn,'Hr':Hr,'Knr':Knr,'DsH':DsH,'DsKn':DsKn,
        'Gamma':Gamma,'div_s':div,'zeta':z,'zeta_r':zd,'c':c,'c_r':ref[10]}

def screen_rotation(angle):
    c,s=math.cos(angle),math.sin(angle);R=np.array([[1,0,0],[0,c,s],[0,-s,c]])
    def rotate(x):
        z=x.copy()
        for indices in ((6,10,14),(7,11,15)):
            z[list(indices)]=np.einsum('ij,j->i',R,x[list(indices)],optimize=False)
        for indices in ((2,8,12,16,18),(5,9,13,17,19)):
            a,b,d,p,q=x[list(indices)];T=np.array([[a,b,d],[b,-a/2+p,q],[d,q,-a/2-p]])
            T=np.einsum('ik,kl,jl->ij',R,T,R,optimize=False)
            z[list(indices)]=[T[0,0],T[0,1],T[0,2],(T[1,1]-T[2,2])/2,T[1,2]]
        return z
    return np.column_stack([rotate(np.eye(20)[:,i]) for i in range(20)])
