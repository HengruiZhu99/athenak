"""Bounded, fully reorthogonalized Golub--Kahan trial-space diagnostic.

The small rectangular matrix records A applied to every retained trial vector.
All solves are checked against the original unmodified RHS and operator. A
restricted singular value is not a full-space smallest singular value: its
condition ratio is only a lower bound on full conditioning. No physical source
or evolved field is filtered. The optional weak vectors are solver diagnostics.
"""
import numpy as np


def investigate(apply,adjoint,rhs,max_iterations=120,progress=None,rms_target=1e-9,point_check=None):
    if not 1<=max_iterations<=160:raise ValueError('bounded two-sided basis budget')
    n=rhs.size;m=max_iterations
    norm=lambda x:float(np.sqrt(np.einsum('i,i->',x,x,optimize=False)))
    U=np.zeros((m+1,n));V=np.zeros((m,n));H=np.zeros((m+1,m))
    beta=norm(rhs)
    if beta==0:raise ValueError('diagnostic requires nonzero RHS')
    U[0]=rhs/beta;history=[];solution=np.zeros(n);weak=None;converged=False
    for j in range(m):
        v=adjoint(U[j])
        for _ in range(2):
            if j:
                c=np.einsum('ij,j->i',V[:j],v,optimize=False)
                v-=np.einsum('i,ij->j',c,V[:j],optimize=False)
        length=norm(v)
        if length<1e-28:raise ValueError('right trial-space breakdown before assessment')
        V[j]=v/length
        w=apply(V[j])
        for _ in range(2):
            c=np.einsum('ij,j->i',U[:j+1],w,optimize=False)
            H[:j+1,j]+=c;w-=np.einsum('i,ij->j',c,U[:j+1],optimize=False)
        H[j+1,j]=norm(w)
        breakdown=H[j+1,j]<1e-28
        if not breakdown:U[j+1]=w/H[j+1,j]
        if (j+1)%20==0 or j==m-1 or breakdown:
            matrix=H[:j+2,:j+1];target=np.zeros(j+2);target[0]=beta
            left,singular,right=np.linalg.svd(matrix,full_matrices=False)
            y=np.einsum('ji,j->i',right,np.einsum('ij,i->j',left,target,optimize=False)/singular,optimize=False)
            solution=np.einsum('i,ij->j',y,V[:j+1],optimize=False)
            residual=rhs-apply(solution);original=norm(residual)
            normal=norm(adjoint(residual))
            weak=np.einsum('i,ij->j',right[-1],V[:j+1],optimize=False)
            weak_actual=norm(apply(weak));point=point_check(residual) if point_check else float(np.max(abs(residual)))
            row={'iteration':j+1,'original_rms':original,'original_point':point,
                 'adjoint_residual_norm':normal,'restricted_min_singular':float(min(singular)),
                 'restricted_max_singular':float(max(singular)),
                 'conditioning_lower_bound':float(max(singular)/min(singular)),
                 'weak_vector_norm':norm(weak),'weak_action_norm':weak_actual,
                 'weak_action_vs_svd_relative_error':abs(weak_actual-min(singular))/max(1.,weak_actual),
                 'small_system_residual':norm(target-np.einsum('ij,j->i',matrix,y,optimize=False))}
            history.append(row)
            if progress:progress(row)
            if original<rms_target and point<1e-7:converged=True;break
        if breakdown:break
    return solution,weak,{'iterations':j+1,'maximum_iterations':m,'history':history,'converged':converged,
                          'original_rms':history[-1]['original_rms'],'original_point':history[-1]['original_point']}
