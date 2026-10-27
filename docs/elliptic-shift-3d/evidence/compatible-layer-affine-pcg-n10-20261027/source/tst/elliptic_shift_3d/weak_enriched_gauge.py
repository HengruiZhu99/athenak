"""Flat-metric variational control, NOT a general-metric replacement gauge.

For g=I and tracefree h, DV[h]=div(h). On the homogeneous shift test space,
the weak equation is (1/2) S* [C+S beta]=0. The enriched test functions give
moment equations at the extra coefficients, not strong endpoint collocation.
Thus A=(1/2)S*S has an exact positive strain-energy identity in the declared
coefficient norm. The original strong DV is separately checked everywhere.
Variable-metric connection freezing requires its full nonlinear variation;
this restricted identity must NEVER be substituted for that formulation.
"""
import numpy as np
from adjoint_tensor_gauge import EnrichedGaugeAdjoint


class WeakEnrichedGauge:
    def __init__(self,enriched):
        if np.max(abs(enriched.gauge.g-np.eye(3)))>1e-14:
            raise ValueError('flat-metric control only; full nonlinear gauge not implemented here')
        self.e=enriched;self.adjoint=EnrichedGaugeAdjoint(enriched)

    def apply(self,value):return .5*self.adjoint.rate(self.e.rate(value))

    def source(self,metric_rate):
        if np.max(abs(np.trace(metric_rate,axis1=-2,axis2=-1)))>1e-9:
            raise ValueError('flat variational control requires tracefree total metric-rate source')
        return -.5*self.adjoint.rate(metric_rate)


class WeakBlockPreconditioner:
    """SPD block diagonal control; bubble blocks are exact, not row sums."""
    def __init__(self,enriched,base):
        self.e=enriched;self.base=base;self.inverse={};W=enriched.grid.weights
        self.diagnostic={'bubble_block_conditions':[],'operator_changed':False}
        for b,high,gradq,_ in enriched.plans:
            gram=np.sum(W[b,...,None,None]*high[:,None,None,None,None]**2*
                        (np.sum(gradq**2,axis=-1)[...,None,None]*np.eye(3)
                         +gradq[..., :,None]*gradq[...,None,:]/3),axis=0)/W[b,-1,...,None,None]
            eig=np.linalg.eigvalsh(gram)
            if np.min(eig)<=0:raise ValueError('nonpositive bubble block')
            self.inverse[b]=np.linalg.inv(gram)
            self.diagnostic['bubble_block_conditions'].append(float(np.max(eig[...,-1]/eig[...,0])))

    def __call__(self,rhs):
        v=rhs.copy();v[self.e.boundary]=0;out=self.base(v)
        for b,inverse in self.inverse.items():
            out[b,-1]=np.einsum('...ij,...j->...i',inverse,rhs[b,-1],optimize=False)
        return out


def conjugate_gradient(apply,rhs,pre,W,max_iterations=800,progress=None):
    if not 1<=max_iterations<=1200:raise ValueError('CG iteration budget')
    weight=W[...,None]/np.sum(W)
    dot=lambda a,b:float(np.sum(weight*a*b))
    x=np.zeros_like(rhs);r=rhs.copy();z=pre(r);p=z.copy();rz=dot(r,z)
    history=[];converged=False
    if rz<=0:raise ValueError('nonpositive preconditioned initial residual')
    for k in range(1,max_iterations+1):
        q=apply(p);den=dot(p,q)
        if den<=0:raise ValueError('nonpositive variational curvature')
        alpha=rz/den;x+=alpha*p;r-=alpha*q
        if k%20==0 or k==max_iterations:
            actual=rhs-apply(x);rms=np.sqrt(dot(actual,actual));point=float(np.max(abs(actual)))
            row={'iteration':k,'original_rms':float(rms),'original_point':point,
                 'recursive_residual_defect':float(np.sqrt(dot(actual-r,actual-r)))}
            history.append(row)
            if progress:progress(row,x)
            if rms<1e-9 and point<1e-7:converged=True;break
        z=pre(r);new=dot(r,z)
        if new<=0:raise ValueError('nonpositive preconditioned residual')
        p=z+(new/rz)*p;rz=new
    return x,{'iterations':k,'converged':converged,'history':history,
              'original_rms':history[-1]['original_rms'],'original_point':history[-1]['original_point']}
