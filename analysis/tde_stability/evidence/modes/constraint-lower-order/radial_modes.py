"""Radial continuum-constraint annulus diagnostic; no AthenaK code or jobs.

All lower-order coefficients are retained. A two-field wave reduction avoids
the equal-order H/M/Theta/Q Jordan reduction. First-order upwind method-of-lines
is used only for this independent spectrum; convergence/constraint defects must
be inspected before interpreting eigenvalues.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vacuum-preservation-20260918/python-deps'))
import sympy as s
import numpy as np
from scipy import sparse,linalg
from scipy.sparse import linalg as sla

r,k,mode=s.symbols('r k mode',real=True)
alpha=r/(1+r);chi=alpha**2;beta=r/(1+r)**2;K=1/(1+r)**2
sigma=k*(1-mode)*alpha+k*mode
ap=s.diff(alpha,r);cp=s.diff(chi,r);bp=s.diff(beta,r);sp=s.diff(sigma,r)
P=s.Matrix([[beta,alpha*chi/2],[2*alpha,beta]])
B=s.Matrix([[-2*sigma,alpha*chi/r],[0,-2*sigma]])
J=s.diag(alpha/2,2*alpha)
S=s.Matrix([[beta,-2*alpha*chi],[-alpha/2,beta]])
C=s.Matrix([[0,4*alpha*chi*K],[2*sigma,-alpha*cp]])
D=s.Matrix([[-4*sigma*K,0],[2*sp,-(2*chi*ap+alpha*cp)/r]])
E=s.Matrix([[2*alpha*K,-4*alpha*chi/r+alpha*cp-4*chi*ap],[-ap,bp+alpha*K]])
T=J*S*J.inv();F=J*E*J.inv()
assert s.simplify(P+T-2*beta*s.eye(2))==s.zeros(2)
assert s.simplify(-T*P-(alpha**2*chi-beta**2)*s.eye(2))==s.zeros(2)
L=B-ap/alpha*T+F
N=ap/alpha*T*P-T*P.diff(r)-T*B+J*C-F*P
O=ap/alpha*T*B-T*B.diff(r)+J*D-F*B
Pd=beta*bp*s.eye(2)+beta*L+N
c=alpha*s.sqrt(chi)
# r>0, so sqrt(chi)=alpha. Keep the expression explicitly rational.
c=alpha**2
coeff=s.lambdify((r,k,mode),[alpha,chi,beta,K,sigma,ap,cp,bp,s.diff(c,r)]+list(L)+list(Pd)+list(O),modules='numpy',cse=True)

def values(rad,rate,scaled):
    v=coeff(rad,rate,int(scaled))
    return np.array([np.broadcast_to(q,rad.shape) for q in v],float)

def build(n,inner,outer,rate,scaled):
    dx=(outer-inner)/n;rad=inner+(np.arange(n)+.5)*dx
    v=values(rad,rate,scaled);a,ch,b,kk,ss,da,dc,db,ccp=v[:9];cc=a*a
    ll=v[9:13].reshape(2,2,n);pp=v[13:17].reshape(2,2,n);oo=v[17:21].reshape(2,2,n)
    local=np.zeros((6,6,n))
    for i in range(2):
        local[i,2+i]=(1+b/cc)/2;local[i,4+i]=(1-b/cc)/2
    rows=[];cols=[];data=[]
    for sign,start in [(1,2),(-1,4)]:
        speed=b+sign*cc
        ff=pp+(sign*cc*db-sign*speed*ccp)*np.eye(2)[:,:,None]
        local[start:start+2,0:2]=oo
        local[start:start+2,2:4]=ll/2+ff/(2*cc)
        local[start:start+2,4:6]=ll/2-ff/(2*cc)
        for comp in range(2):
            off=(start+comp)*n
            for i in range(n):
                aa=speed[i]
                if aa>=0:
                    if i==n-1:
                        # Incoming value at the outer face is prescribed zero.
                        # At L>1 only plus fields use this case.
                        assert sign==1
                        rows.append(off+i);cols.append(off+i);data.append(-2*aa/dx)
                    else:
                        rows.extend([off+i,off+i]);cols.extend([off+i,off+i+1]);data.extend([-aa/dx,aa/dx])
                else:
                    # Inner face is inside r=1 and has no incoming wave mode.
                    assert i>0
                    rows.extend([off+i,off+i]);cols.extend([off+i-1,off+i]);data.extend([-aa/dx,aa/dx])
    for i in range(6):
        for j in range(6):
            for z in np.flatnonzero(local[i,j]):
                rows.append(i*n+z);cols.append(j*n+z);data.append(local[i,j,z])
    mat=sparse.coo_matrix((data,(rows,cols)),shape=(6*n,6*n)).tocsr()
    return mat,rad,v

def derivative(u,dx):
    return np.gradient(u,dx,axis=-1,edge_order=2)

def inspect(lam,evec,mat,rad,v,profiles=False):
    n=len(rad);dx=rad[1]-rad[0];u=evec.reshape(6,n)
    a,ch,b,kk,ss,da,dc,db,ccp=v[:9];cc=a*a
    W=u[:2];Pi=(u[2:4]+u[4:6])/2;Dw=(u[2:4]-u[4:6])/(2*cc)
    dW=derivative(W,dx);sl=slice(3,-3)
    defect=np.linalg.norm((Dw-dW)[:,sl])/max(np.linalg.norm(Dw[:,sl])+np.linalg.norm(dW[:,sl]),1e-300)
    H=2/a*(Pi[0]+2*ss*W[0])-ch*(Dw[1]+2*W[1]/rad)
    m=(Pi[1]+2*ss*W[1])/(2*a)-Dw[0]
    th,q=W;dt,dq=dW;dh,dm=derivative(np.array([H,m]),dx)
    sp=derivative(ss,dx)
    rhs=np.array([b*dt+a/2*H+a*ch/2*(dq+2*q/rad)-2*ss*th,
                  b*dq+2*a*(m+dt)-2*ss*q,
                  b*dh-2*a*ch*dm+(-4*a*ch/rad+a*dc-4*ch*da)*m+2*a*kk*H+4*a*ch*kk*dq-4*ss*kk*th,
                  b*dm+(db+a*kk)*m-a/2*dh-da*H-a*dc*dq-(2*ch*da+a*dc)*q/rad+2*ss*dt+2*sp*th])
    orig=np.array([th,q,H,m]);res=lam*orig-rhs
    component_res=[]
    for z in range(4):
        component_res.append(float(np.linalg.norm(res[z,sl])/max(np.linalg.norm((lam*orig[z])[sl])+np.linalg.norm(rhs[z,sl]),1e-300)))
    peak=np.argmax(abs(th));phase=np.exp(-1j*np.angle(th[peak]));scale=max(np.max(abs(th)),1e-300)
    result={'eigenvalue_real_per_M':float(lam.real),'eigenvalue_imag_per_M':float(lam.imag),
      'matrix_relative_residual':float(np.linalg.norm(mat@evec-lam*evec)/(max(sla.norm(mat,np.inf)+abs(lam),1e-300)*np.linalg.norm(evec))),
      'wave_reduction_L2_relative_defect_interior':float(defect),
      'original_subsystem_component_relative_residuals_interior':component_res,
      'Theta_peak_r_M':float(rad[peak]),
      'incoming_outer_face_nearest_cell_ratio':float(np.linalg.norm(u[2:4,-1])/max(np.max(np.linalg.norm(u[2:4],axis=0)),1e-300))}
    if profiles:result['profile']={'r_M':rad.tolist(),'Theta_real_normalized':(phase*th/scale).real.tolist(),'Theta_imag_normalized':(phase*th/scale).imag.tolist(),'Q_real_normalized':(phase*q/scale).real.tolist(),'H_real_normalized':(phase*H/scale).real.tolist(),'M_real_normalized':(phase*m/scale).real.tolist()}
    return result

def spectrum(n,inner,outer,rate,scaled,shifts=None):
    start=time.monotonic();mat,rad,v=build(n,inner,outer,rate,scaled)
    if shifts is None:
        vals,vecs=linalg.eig(mat.toarray(),check_finite=False)
        indices=np.argsort(vals.real)[::-1][:18]
    else:
        vv=[];xx=[]
        for shift in shifts:
            ee,zz=sla.eigs(mat,k=12,sigma=shift,tol=1e-10,maxiter=5000)
            for i,lam in enumerate(ee):
                if not any(abs(lam-z)<1e-7 for z in vv):vv.append(lam);xx.append(zz[:,i])
        vals=np.array(vv);vecs=np.array(xx).T;indices=np.argsort(vals.real)[::-1]
    rows=[inspect(vals[i],vecs[:,i],mat,rad,v,profiles=False) for i in indices]
    # Profiles are saved for each returned top mode, including defects, without
    # silently classifying a grid/reduction mode as physical.
    detailed=[inspect(vals[i],vecs[:,i],mat,rad,v,profiles=True) for i in indices[:3]]
    return {'n':n,'inner_M':inner,'outer_M':outer,'rate_per_M':rate,'lapse_scaled':scaled,
       'method':'dense all eigenvalues' if shifts is None else 'sparse shift-invert targeted subset',
       'shifts':None if shifts is None else [float(z) for z in shifts],'wall_seconds':time.monotonic()-start,'modes':rows,'leading_profiles':detailed}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=96);parser.add_argument('--inner',type=float,default=.2);parser.add_argument('--outer',type=float,default=4);parser.add_argument('--rate',type=float,default=.1);parser.add_argument('--scaled',action='store_true');parser.add_argument('--shifts',type=float,nargs='*');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=spectrum(args.n,args.inner,args.outer,args.rate,args.scaled,args.shifts)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('leading_profiles','modes')},indent=2));print(json.dumps(result['modes'][:6],indent=2))
