"""Experimental affine coefficient updates confined to complete shell layers.

Each correction is skew in the original physical norm. Only the block-local
positive high-mode form is modified; the full interface jump penalty is kept.
No physical field or source is projected and affine shift modes stay free.
Angular coupling within a layer remains nonlocal. This is not a native adapter.
"""
import numpy as np
from compatible_sbp import mm
from mapped_sbp_probe import axis_apply


class LayerAffineGrid:
    def __init__(self,base):
        self.base=base;W=base.weights;X=base.coordinates
        V=np.concatenate((np.ones(W.shape+(1,)),X),axis=-1)
        central=[base.derivative(V,a,0) for a in range(3)]
        groups=[[b for b,(p,l) in enumerate(base.blocks) if p<0]]
        groups += [[b for b,(p,l) in enumerate(base.blocks) if p>=0 and l==k] for k in range(len(base.radial)-1)]
        self.groups=[];self.diagnostic={'scope':__doc__,'groups':[]}
        for ids in groups:
            H=W[ids].ravel();v=V[ids].reshape(-1,4)
            transform=np.linalg.inv(np.linalg.cholesky(mm(v.T*H[None,:],v)).T)
            P=mm(v,transform);plans=[];diagnostics=[]
            for axis in range(3):
                target=np.zeros(4);target[axis+1]=1
                F=mm(target[None,:]-central[axis][ids].reshape(-1,4),transform)
                S=mm(P.T*H[None,:],F)
                # R_local is the sum of the original positive element forms.
                # Do not use R applied to a discontinuous group basis here:
                # that would also remove interface jump control.
                rp=[]
                for b in ids:
                    loc=base.locals[b];q=mm(V[b].reshape(-1,4),transform).reshape(V[b].shape)
                    r=np.zeros_like(q)
                    for a in range(3):
                        c=base.cofactors[b][...,a,axis]
                        r+=np.max(abs(c))*axis_apply(q,a,loc['R']/loc['H'][:,None])
                    rp.append(r/base.jacobians[b][...,None])
                rp=np.array(rp).reshape(-1,4);T=mm(P.T*H[None,:],rp)
                compatibility=float(np.max(abs(S+S.T)));symmetry=float(np.max(abs(T-T.T)))
                if compatibility>1e-10 or symmetry>1e-10:raise ValueError('layer affine moment compatibility failed')
                plans.append((F,rp,(S-S.T)/2,(T+T.T)/2))
                diagnostics.append({'axis':axis,'moment_compatibility_max':compatibility,'local_penalty_symmetry_max':symmetry})
            self.groups.append((ids,H,P,plans));self.diagnostic['groups'].append({'blocks':ids,'axes':diagnostics})

    def __getattr__(self,name):return getattr(self.base,name)

    def derivative(self,values,axis,sign):
        result=self.base.derivative(values,axis,sign)
        for ids,H,P,plans in self.groups:
            f=values[ids].reshape(len(H),-1);F,rp,S,T=plans[axis]
            moments=mm(P.T*H[None,:],f);fmoments=mm(F.T*H[None,:],f)
            update=mm(F,moments)-mm(P,fmoments+mm(S,moments))
            if sign:
                rmoments=mm(rp.T*H[None,:],f)
                update+=sign*(-mm(rp,moments)-mm(P,rmoments-mm(T,moments)))
            result[ids]+=update.reshape(values[ids].shape)
        return result
