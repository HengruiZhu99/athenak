#!/usr/bin/env python3
"""One-dimensional diagnostic of divergence modes hidden by Dirichlet rows.

The highest Legendre source varies with n and is not a convergence family.
No filtering/projection of the prescribed source is used. Penalty scales change
both members of the derivative pair; none is adopted by this diagnostic.
"""
import argparse,json
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from mapped_sbp_probe import lobatto
from compatible_sbp import mm,basis

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');rows=[]
for n in [10,12,14]:
    q=lobatto(n);x=q['x'];H=q['H'];degree=n-1
    high=leg.legval(2*x-1,np.eye(n)[-1]);integral=leg.legint(np.eye(n)[-1])/2
    truth_high=-leg.legval(2*x-1,integral)+leg.legval(-1,integral)
    for name,C,truth in [('highest_legendre',high,truth_high),('resolved_exponential',np.exp(x)-(np.e-1),1+(np.e-1)*x-np.exp(x))]:
        for label,scale in [('central',0.),('unit',1.),('n',float(n)),('n_squared',float(n*n))]:
            minus=q['D']-scale*q['R']/H[:,None];plus=q['D']+scale*q['R']/H[:,None]
            A=-mm(plus,minus);rhs=mm(plus,C)
            A[0]=0;A[-1]=0;A[0,0]=A[-1,-1]=1;rhs[0]=rhs[-1]=0
            beta=np.linalg.solve(A,rhs);rate=mm(minus,beta)+C;divergence=mm(plus,rate)
            rows.append({'n':n,'source':name,'penalty_scale':label,'bulk_rate_max':float(np.max(abs(divergence[1:-1]))),'endpoint_rate_max':float(max(abs(divergence[0]),abs(divergence[-1]))),'metric_rate_rms':float(np.sqrt(np.sum(H*rate*rate))),'solution_nodal_rms':float(np.sqrt(np.sum(H*(beta-truth)**2))),'matrix_condition':float(np.linalg.cond(A))})
        # One extra shift node represents the primitive of every degree-(n-1)
        # curvature polynomial. Keep the original n metric nodes and source;
        # enforce their divergence equations INCLUDING both endpoints. The
        # tall system has one dependent equation, checked by SVD/rank and the
        # unmodified full residual. This is a 1D design control, not a 3D scheme.
        shift=lobatto(n+1);Vs,_=basis(shift['x'],n);Vm,dVm=basis(x,n)
        inverse=np.linalg.inv(Vs);G=mm(dVm,inverse);interpolate=mm(Vm,inverse)
        matrix=mm(q['D'],G[:,1:-1]);rhs=-mm(q['D'],C)
        interior,_,rank,singular=np.linalg.lstsq(matrix,rhs,rcond=None)
        beta=np.r_[0.,interior,0.];rate=mm(G,beta)+C;divergence=mm(q['D'],rate)
        nodal=mm(interpolate,beta)
        rows.append({'n':n,'source':name,'penalty_scale':'enriched_shift_control','metric_nodes':n,'shift_nodes':n+1,'rank':int(rank),'expected_rank':n-1,'bulk_rate_max':float(np.max(abs(divergence[1:-1]))),'endpoint_rate_max':float(max(abs(divergence[0]),abs(divergence[-1]))),'metric_rate_rms':float(np.sqrt(np.sum(H*rate*rate))),'solution_nodal_rms':float(np.sqrt(np.sum(H*(nodal-truth)**2))),'matrix_condition':float(max(singular)/min(singular))})
result={'scope':__doc__,'rows':rows,'interpretation':'A small residual on interior rows need not control the endpoint metric-derived divergence. Source degree and penalty scaling require separate checks in the full tensor geometry.'}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
