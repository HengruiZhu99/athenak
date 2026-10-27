"""Unadopted global affine-moment correction of an existing SBP pair.

This changes derivative coefficients, not evolved fields, constraints or RHS
values. The four physical affine modes are differentiated rather than removed.
The update is global and low rank; its nonlocality and refinement behavior must
be qualified before any evolution adoption. No boundary rows are replaced here.
"""
import numpy as np
from compatible_sbp import mm


class AffineConsistentGrid:
    def __init__(self,base):
        self.base=base;H=base.weights.ravel();X=base.coordinates.reshape(-1,3)
        V=np.column_stack((np.ones(len(H)),X));gram=mm(V.T*H[None,:],V)
        self.transform=np.linalg.inv(np.linalg.cholesky(gram).T)
        self.P=mm(V,self.transform);self.H=H;self.plans=[]
        self.diagnostic={'scope':__doc__,'global_affine_modes':4,'axes':[]}
        values=self.P.reshape(base.weights.shape+(4,))
        for axis in range(3):
            central=base.derivative(values,axis,0).reshape(-1,4)
            rp=(base.derivative(values,axis,1).reshape(-1,4)-central)
            target=np.broadcast_to(self.transform[axis+1],central.shape)
            F=target-central;S=mm(self.P.T*H[None,:],F)
            T=mm(self.P.T*H[None,:],rp)
            compatibility=float(np.max(abs(S+S.T)))
            symmetry=float(np.max(abs(T-T.T)))
            # These are existence conditions, not quantities to force to zero
            # when the geometry/quadrature is incompatible with affine moments.
            if compatibility>1e-10 or symmetry>1e-10:
                raise ValueError('affine SBP moment compatibility failed')
            S=(S-S.T)/2;T=(T+T.T)/2
            self.plans.append((F,rp,S,T))
            self.diagnostic['axes'].append({'axis':axis,'moment_compatibility_max':compatibility,'high_mode_symmetry_max':symmetry,'uncorrected_affine_rms':float(np.sqrt(np.sum(H[:,None]*F*F)/np.sum(H)))})

    def __getattr__(self,name):return getattr(self.base,name)

    def derivative(self,values,axis,sign):
        result=self.base.derivative(values,axis,sign)
        f=values.reshape(len(self.H),-1);F,rp,S,T=self.plans[axis]
        moments=mm(self.P.T*self.H[None,:],f)
        fmoments=mm(F.T*self.H[None,:],f)
        # K = F P^* - P F^* - P(P^*F)P^* is skew-adjoint and makes
        # (D0+K)P equal the exact physical affine derivative.
        update=mm(F,moments)-mm(self.P,fmoments+mm(S,moments))
        if sign:
            rmoments=mm(rp.T*self.H[None,:],f)
            # Rnew=(I-PP^*)R(I-PP^*) stays nonnegative and annihilates only
            # these affine modes in the high-mode penalty. The full derivative
            # still returns their nonzero exact gradients. No state is reset.
            update+=sign*(-mm(rp,moments)-mm(self.P,rmoments-mm(T,moments)))
        return result+update.reshape(values.shape)
