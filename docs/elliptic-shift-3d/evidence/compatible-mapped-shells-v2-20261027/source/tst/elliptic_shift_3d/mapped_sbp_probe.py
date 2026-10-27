"""Unadopted compatible Cartesian-core/cubed-sphere derivative experiment.

The core retains uniform cell centers. Shells use boundary-collocated Legendre
nodes so metric traces commute with multiplication, including at actual r=1.
This is a spatial component candidate, not a native grid or scri evolution.
All components use the common Cartesian basis. No evolved field is projected.
"""
import functools
import numpy as np
from numpy.polynomial import legendre as leg
from compatible_sbp import local, mortar, basis, mm


@functools.lru_cache(None)
def lobatto(n):
    coefficients=np.eye(n)[-1]
    z=np.r_[-1.,leg.legroots(leg.legder(coefficients)),1.]
    x=(z+1)/2;H=1/(n*(n-1)*leg.legval(z,coefficients)**2)
    V,dV=basis(x,n-1);inverse=np.linalg.inv(V);D=mm(dV,inverse)
    trace=np.zeros((2,n));trace[0,0]=trace[1,-1]=1
    # A symmetric high-mode penalty annihilates degree <=2 exactly. Its
    # dimensional factor is supplied by the physical cofactor below.
    norms=np.sum(H[:,None]*V*V,axis=0)
    penalty=(np.arange(n)/(n-1))**6;penalty[:3]=0
    R=mm(inverse.T*(norms*penalty)[None,:],inverse)
    return dict(x=x,H=H,D=D,R=R,trace=trace,degree=2)


def axis_apply(values,axis,operator):
    moved=np.moveaxis(values,axis,0)
    return np.moveaxis(mm(operator,moved.reshape(moved.shape[0],-1)).reshape((operator.shape[0],)+moved.shape[1:]),0,axis)


class ShellGrid:
    def __init__(self,n,radial=(0.,.45,1.),half_width=.25):
        if radial[0]!=0 or radial[-1]!=1 or min(np.diff(radial))<=0:
            raise ValueError('radial partition must span the actual sphere')
        self.n=n;self.radial=radial;self.half_width=half_width
        self.blocks=[(-1,0)]+[(p,k) for k in range(len(radial)-1) for p in range(6)]
        self.cells=len(self.blocks)*n**3
        if self.cells>50000:raise ValueError('mapped component cell budget')
        self.locals=[local(n,2)]+[lobatto(n)]*(len(self.blocks)-1)
        self.coordinates=[];self.cofactors=[];self.jacobians=[];self.weights=[]
        self.metric_identity_max=0.
        for b,(patch,layer) in enumerate(self.blocks):
            loc=self.locals[b];q=np.stack(np.meshgrid(*([loc['x']]*3),indexing='ij'),axis=-1)
            xyz=self.map(patch,layer,q);self.coordinates.append(xyz)
            dx=[axis_apply(xyz,a,loc['D']) for a in range(3)]
            cofactor=[]
            for a in range(3):
                j=(a+1)%3;k=(a+2)%3
                cofactor.append(.5*(axis_apply(np.cross(xyz,dx[k]),j,loc['D'])-axis_apply(np.cross(xyz,dx[j]),k,loc['D'])))
            C=np.stack(cofactor,axis=-2)
            jac=sum(np.sum(C[...,a,:]*dx[a],axis=-1) for a in range(3))/3
            orientation=1 if float(np.mean(jac))>0 else -1
            C*=orientation;jac*=orientation
            if np.min(jac)<=0:raise ValueError('folded discrete shell geometry')
            identity=sum(axis_apply(C[...,a,:],a,loc['D']) for a in range(3))
            self.metric_identity_max=max(self.metric_identity_max,float(np.max(abs(identity))))
            H=loc['H'];W=H[:,None,None]*H[None,:,None]*H[None,None,:]
            self.cofactors.append(C);self.jacobians.append(jac);self.weights.append(W*jac)
        self.coordinates=np.array(self.coordinates);self.weights=np.array(self.weights)
        self.links=[];self.outer=[];self.normal_mismatch=0.
        shell_id=lambda p,k:1+6*k+p
        # Nonconforming planar core/shell interfaces: the two quadratures have
        # matching moments, while their actual face nodes are different.
        cf=dict(x=self.locals[0]['x'],H=self.locals[0]['H'],V=basis(self.locals[0]['x'],2)[0],degree=2)
        sf=dict(x=lobatto(n)['x'],H=lobatto(n)['H'],V=basis(lobatto(n)['x'],2)[0],degree=2)
        P,Q=mortar(cf,sf)
        for patch in range(6):
            a=patch//2;side=1-patch%2;tangents=[j for j in range(3) if j!=a]
            swap=tangents!=[(a+1)%3,(a+2)%3]
            self.add_link((0,a,side),(shell_id(patch,0),0,0),P,Q,swap,None)
            for k in range(len(radial)-2):self.add_link((shell_id(patch,k),0,1),(shell_id(patch,k+1),0,0))
            self.outer.append((shell_id(patch,len(radial)-2),0,1))
        # Adjacent shell faces are paired by their physical coordinates, never
        # by assuming that local tensor axes have the same orientation.
        faces={}
        for b,(patch,k) in enumerate(self.blocks[1:],start=1):
            for axis in [1,2]:
                for side in [0,1]:
                    q=np.full(3,.5);q[axis]=side
                    key=tuple(np.round(self.map(patch,k,q),12))
                    faces.setdefault(key,[]).append((b,axis,side))
        for pair in faces.values():
            if len(pair)!=2:raise ValueError('unpaired angular face')
            L,R=pair;l=self.trace(self.coordinates,L).reshape(-1,3);r=self.trace(self.coordinates,R).reshape(-1,3)
            distance=np.sum((l[:,None,:]-r[None,:,:])**2,axis=-1);permutation=np.argmin(distance,axis=1)
            if len(set(permutation))!=n*n or np.max(distance[np.arange(n*n),permutation])>1e-24:
                raise ValueError('angular surface nodes do not coincide')
            self.add_link(L,R,permutation=permutation)
        if self.normal_mismatch>1e-9:raise ValueError('interface metric normals do not match')

    def map(self,patch,layer,q):
        if patch<0:return self.half_width*(2*q-1)
        a=patch//2;b=(a+1)%3;c=(a+2)%3;sgn=1 if patch%2==0 else -1
        s=self.radial[layer]+np.diff(self.radial)[layer]*q[...,0]
        u=2*q[...,1]-1;v=2*q[...,2]-1
        d=(1-s)*self.half_width+s/np.sqrt(1+u*u+v*v)
        xyz=np.zeros(q.shape);xyz[...,a]=sgn*d;xyz[...,b]=d*u;xyz[...,c]=d*v
        return xyz

    def trace(self,values,face):
        b,axis,side=face;v=np.moveaxis(values[b],axis,0)
        return np.einsum('i,i...->...',self.locals[b]['trace'][side],v,optimize=False)

    def normal(self,face):
        b,a,s=face
        return (2*s-1)*self.trace([C[...,a,:] for C in self.cofactors],face)

    def add_link(self,L,R,P=None,Q=None,swap=False,permutation=None):
        link=(L,R,P,Q,swap,permutation)
        self.normal_mismatch=max(self.normal_mismatch,float(np.max(abs(self.normal(L)+self.transfer(self.normal(R),link,True)))))
        self.links.append(link)

    def transfer(self,value,link,to_left):
        _,_,P,Q,swap,permutation=link
        if permutation is not None:
            index=permutation if to_left else np.argsort(permutation)
            return value.reshape((self.n*self.n,)+value.shape[2:])[index].reshape(value.shape)
        def tensor(v,M):
            if M is None:return v.copy()
            return np.einsum('jk,ik...->ij...',M,np.einsum('ij,jk...->ik...',M,v,optimize=False),optimize=False)
        if to_left:return tensor(np.swapaxes(value,0,1) if swap else value,P)
        v=tensor(value,Q);return np.swapaxes(v,0,1) if swap else v

    def lift(self,flux,face):
        b,a,s=face;loc=self.locals[b]
        normal=(loc['trace'][s]/loc['H']).reshape((self.n,)+(1,)*flux.ndim)
        return np.moveaxis(normal*flux[None,...],0,a)

    def derivative(self,values,physical_axis,sign):
        if sign not in [-1,0,1]:raise ValueError('paired derivative selector')
        out=[];tail=(1,)*(values.ndim-4)
        for b in range(len(self.blocks)):
            loc=self.locals[b];v=values[b];result=np.zeros_like(v)
            for a in range(3):
                c=self.cofactors[b][...,a,physical_axis];C=c.reshape(c.shape+tail)
                result+=.5*(C*axis_apply(v,a,loc['D'])+axis_apply(C*v,a,loc['D']))
                if sign:result+=sign*np.max(abs(c))*axis_apply(v,a,loc['R']/loc['H'][:,None])
            out.append(result)
        for link in self.links:
            L,R=link[:2];l=self.trace(values,L);r=self.trace(values,R)
            jl=l-self.transfer(r,link,True);jr=r-self.transfer(l,link,False)
            nl=self.normal(L)[...,physical_axis].reshape((self.n,self.n)+tail)
            nr=self.normal(R)[...,physical_axis].reshape((self.n,self.n)+tail)
            fl=-.5*nl*jl;fr=-.5*nr*jr
            if sign:
                fl+=sign*.25*abs(nl)*(jl-self.transfer(jr,link,True))
                fr+=sign*.25*abs(nr)*(jr-self.transfer(jl,link,False))
            out[L[0]]+=self.lift(fl,L);out[R[0]]+=self.lift(fr,R)
        return np.stack([v/self.jacobians[b].reshape(self.jacobians[b].shape+tail) for b,v in enumerate(out)])

    def boundary_integral(self,u,v,axis):
        result=0.
        for face in self.outer:
            H=self.locals[face[0]]['H'];normal=self.normal(face)[...,axis]
            result+=float(np.sum(H[:,None]*H[None,:]*normal*self.trace(u,face)*self.trace(v,face)))
        return result
