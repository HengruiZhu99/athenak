"""Bounded restarted right-preconditioned GMRES with explicit residual checks."""
import numpy as np


def gmres(apply,rhs,precondition,weight,max_iterations=600,restart=60,
          rms_target=1e-9,point_target=1e-7,progress=None):
    if not 1<=max_iterations<=1200 or not 1<=restart<=120:
        raise ValueError('Krylov CPU/memory budget')
    shape=rhs.shape;root=np.sqrt(weight/np.sum(weight))[...,None]
    pack=lambda value:(root*value).ravel()
    unpack=lambda value:value.reshape(shape)/root
    norm=lambda value:float(np.sqrt(np.einsum('i,i->',value,value,optimize=False)))
    x=np.zeros_like(rhs);history=[];iterations=0;converged=False
    while iterations<max_iterations:
        residual=rhs-apply(x);v=pack(residual);beta=norm(v)
        if beta<rms_target and np.max(abs(residual))<point_target:
            converged=True;break
        m=min(restart,max_iterations-iterations)
        V=np.zeros((m+1,v.size));Z=np.zeros((m,v.size));H=np.zeros((m+1,m))
        V[0]=v/beta;best=None
        for j in range(m):
            Z[j]=pack(precondition(unpack(V[j])))
            w=pack(apply(unpack(Z[j])))
            # Two passes reduce loss of orthogonality from the broad mapped
            # spectrum. All cells and the explicit boundary rows participate.
            for _ in range(2):
                c=np.einsum('ij,j->i',V[:j+1],w,optimize=False)
                H[:j+1,j]+=c;w-=np.einsum('i,ij->j',c,V[:j+1],optimize=False)
            H[j+1,j]=norm(w)
            if H[j+1,j]>1e-30:V[j+1]=w/H[j+1,j]
            iterations+=1
            if iterations%20==0 or j==m-1 or H[j+1,j]<=1e-30:
                target=np.zeros(j+2);target[0]=beta
                y=np.linalg.lstsq(H[:j+2,:j+1],target,rcond=None)[0]
                best=x+unpack(np.einsum('i,ij->j',y,Z[:j+1],optimize=False))
                original=rhs-apply(best);rms=norm(pack(original));point=float(np.max(abs(original)))
                history.append({'iteration':iterations,'original_rms':rms,'original_point':point})
                if progress is not None:progress(history[-1])
                if rms<rms_target and point<point_target:
                    converged=True;break
            if H[j+1,j]<=1e-30:break
        if best is None:raise ValueError('Krylov iteration produced no checked update')
        x=best
        if converged:break
    original=rhs-apply(x)
    return x,{'iterations':iterations,'maximum_iterations':max_iterations,
              'restart':restart,'converged':converged,'history':history,
              'original_rms':norm(pack(original)),
              'original_point':float(np.max(abs(original)))}
