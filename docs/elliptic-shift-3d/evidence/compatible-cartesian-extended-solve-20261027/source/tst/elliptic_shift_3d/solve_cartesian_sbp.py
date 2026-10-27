#!/usr/bin/env python3
"""Bounded original-residual solve of the compatible periodic 3D tensor gauge.

Discrete manufactured source; proves linear-solve behavior only, not continuum
MMS, nonlinear evolution, hyperboloidal scri or a general metric result.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from cartesian_sbp_probe import Grid
from cartesian_sbp_preconditioner import Preconditioner


def run(preconditioner,coarse_degree,max_iterations):
    grid=Grid(10,2,True)
    g=np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]])
    gi=np.linalg.inv(g);W=grid.weights*np.sqrt(np.linalg.det(g))
    def apply(b):
        grad=np.stack([grid.derivative(b,j,-1) for j in range(3)],axis=-2)
        div=np.einsum('...ii->...',grad,optimize=False)
        h=np.einsum('kj,...ik->...ij',g,grad,optimize=False)+np.einsum('ik,...jk->...ij',g,grad,optimize=False)-(2/3)*g*div[...,None,None]
        dh=np.stack([grid.derivative(h,j,1) for j in range(3)],axis=-3)
        return -np.einsum('jk,il,...jlk->...i',gi,gi,dh,optimize=False)+.5*np.einsum('jk,il,...ljk->...i',gi,gi,dh,optimize=False)
    def dot(b,c):return float(np.sum(W*np.einsum('...i,ij,...j->...',b,g,c,optimize=False))/np.sum(W))
    X=np.moveaxis(grid.coordinates,-1,0)
    exact=np.stack([np.sin(2*np.pi*X[0])*np.cos(2*np.pi*X[1]),.3*np.sin(2*np.pi*X[2]),.2*np.cos(2*np.pi*(X[0]+X[2]))],axis=-1)
    exact+=.001*np.random.default_rng(271127).normal(size=exact.shape)
    rhs=apply(exact);x=np.zeros_like(exact)
    scale=np.array([1/sum((grid.n/width)**2) for _,width in grid.blocks])[:,None,None,None,None]
    pre=Preconditioner(grid,g,apply,coarse_degree,preconditioner=='balanced') if preconditioner!='scaled' else lambda r:scale*r
    residual=rhs.copy();z=pre(residual);direction=z.copy();rz=dot(residual,z)
    history=[];curvatures=[];start=time.monotonic();converged=False
    for iteration in range(1,max_iterations+1):
        action=apply(direction);curvature=dot(direction,action);curvatures.append(curvature)
        if not np.isfinite(curvature) or curvature<=0:raise ValueError('nonpositive PCG curvature')
        step=rz/curvature;x+=step*direction;residual-=step*action
        if iteration%20==0 or np.sqrt(dot(residual,residual))<1e-9:
            original=rhs-apply(x);rms=np.sqrt(dot(original,original));point=np.max(abs(original))
            history.append({'iteration':iteration,'original_rms':float(rms),'original_point':float(point)})
            if rms<1e-9 and point<1e-7:converged=True;break
        z=pre(residual);new=dot(residual,z);direction=z+(new/rz)*direction;rz=new
    original=rhs-apply(x);error=x-exact
    mean=np.sum(W[...,None]*error,axis=(0,1,2,3))/np.sum(W)
    error=error-mean # Error measure modulo the three exact periodic constant modes.
    return {'scope':__doc__,'preconditioner':preconditioner,'preconditioner_diagnostics':getattr(pre,'diagnostic',{}),'n':grid.n,'cells':grid.cells,'coarse_fine_groups':grid.coarse_fine_links,'max_iterations':max_iterations,'iterations':iteration,'converged':converged,'original_rms':float(np.sqrt(dot(original,original))),'original_point':float(np.max(abs(original))),'solution_error_modulo_periodic_constants':float(np.sqrt(dot(error,error))),'unfixed_periodic_constant_offsets':mean.tolist(),'minimum_search_curvature':min(curvatures),'history':history,'solve_seconds':time.monotonic()-start,'gates':{'original_residual':converged,'discrete_solution':float(np.sqrt(dot(error,error)))<1e-7}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preconditioner',choices=['scaled','two-level','balanced'],default='two-level');p.add_argument('--coarse-degree',type=int,choices=[0,1,2],default=1);p.add_argument('--max-iterations',type=int,default=400);a=p.parse_args();assert 1<=a.max_iterations<=1200;a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    r=run(a.preconditioner,a.coarse_degree,a.max_iterations);(a.output/'assessment.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if all(r['gates'].values()) else 1)
