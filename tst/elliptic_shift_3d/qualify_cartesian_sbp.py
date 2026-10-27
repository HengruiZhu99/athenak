#!/usr/bin/env python3
"""Actual 3D Cartesian leaf-interface energy/accuracy checks, no evolution claim."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from cartesian_sbp_probe import Grid


def tensor(grid):
    g=np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]]);gi=np.linalg.inv(g)
    W=grid.weights*np.sqrt(np.linalg.det(g));X=np.moveaxis(grid.coordinates,-1,0)
    def rates(b):
        grad=np.stack([grid.derivative(b,j,-1) for j in range(3)],axis=-2)
        div=np.einsum('...ii->...',grad,optimize=False)
        return np.einsum('kj,...ik->...ij',g,grad,optimize=False)+np.einsum('ik,...jk->...ij',g,grad,optimize=False)-(2/3)*g*div[...,None,None]
    def variation(h):
        dh=np.stack([grid.derivative(h,j,1) for j in range(3)],axis=-3)
        return np.einsum('jk,il,...jlk->...i',gi,gi,dh,optimize=False)-.5*np.einsum('jk,il,...ljk->...i',gi,gi,dh,optimize=False)
    def connection(metric):
        inv=np.linalg.inv(metric);dg=np.stack([grid.derivative(metric,j,1) for j in range(3)],axis=-3)
        return np.einsum('...jk,...il,...jlk->...i',inv,inv,dg,optimize=False)-.5*np.einsum('...jk,...il,...ljk->...i',inv,inv,dg,optimize=False)
    def dot(b,c):return float(np.sum(W*np.einsum('...i,ij,...j->...',b,g,c,optimize=False)))
    smooth=np.stack([np.sin(2*np.pi*X[0])*np.cos(2*np.pi*X[1]),.3*np.sin(2*np.pi*X[2]),.2*np.cos(2*np.pi*(X[0]+X[2]))],axis=-1)
    rng=np.random.default_rng(271127);cases={}
    checker=np.zeros_like(smooth);checker[...,0]=np.broadcast_to((-1.)**np.indices((grid.n,)*3).sum(axis=0),grid.weights.shape)
    for name,b in [('smooth',smooth),('checker',checker),('random',rng.normal(size=smooth.shape))]:
        h=rates(b);Ab=-variation(h);strain=float(.5*np.sum(W*np.einsum('ik,jl,...ij,...kl->...',gi,gi,h,h,optimize=False)));energy=dot(b,Ab)
        c=rng.normal(size=b.shape);Ac=-variation(rates(c));left=dot(b,Ac);right=dot(Ab,c)
        cases[name]={'energy':energy,'strain':strain,'energy_relative_defect':abs(energy-strain)/max(1,abs(strain)),'rayleigh':energy/dot(b,b),'bilinear_symmetry_relative':abs(left-right)/max(1,abs(left),abs(right))}
    h=rates(smooth);exact=variation(h);metric=np.broadcast_to(g,h.shape).copy();fd=[]
    for eps in [1e-4,1e-5,1e-6]:
        plus=metric+eps*h;minus=metric-eps*h
        if min(np.min(np.linalg.det(plus)),np.min(np.linalg.det(minus)))<=0:raise ValueError('finite perturbation lost SPD')
        d=(connection(plus)-connection(minus))/(2*eps)-exact
        fd.append({'epsilon':eps,'scaled_rms':float(np.sqrt(np.sum(W[...,None]*d*d))/max(1,np.sqrt(np.sum(W[...,None]*exact*exact)))),'maximum':float(np.max(abs(d)))})
    return {'cases':cases,'frechet':fd,'metric':g.tolist()}


def run():
    report={'scope':'Periodic Cartesian 3D core with actual 2:1 leaf interfaces. Constant non-diagonal metric kernel and variable scalar coefficient controls only. No cubed-sphere, scri, general variable-metric gauge or evolution validation.','rows':[]}
    for n in [10,12,14]:
        grid=Grid(n,2,True);X=np.moveaxis(grid.coordinates,-1,0);W=grid.weights
        f=np.sin(2*np.pi*X[0])*np.cos(2*np.pi*X[1])+.2*np.sin(2*np.pi*X[2])
        truth=[2*np.pi*np.cos(2*np.pi*X[0])*np.cos(2*np.pi*X[1]),-2*np.pi*np.sin(2*np.pi*X[0])*np.sin(2*np.pi*X[1]),.4*np.pi*np.cos(2*np.pi*X[2])]
        rng=np.random.default_rng(711);u=rng.normal(size=f.shape);v=rng.normal(size=f.shape)
        errors=[];adjoint=[];constant=0;energy=0;strain=0
        a=1+.2*np.cos(2*np.pi*X[0])+.1*np.sin(2*np.pi*X[2])
        for axis in range(3):
            derivative=grid.derivative(f,axis,-1);errors.append(derivative-truth[axis]);constant=max(constant,float(np.max(abs(grid.derivative(np.ones_like(f),axis,-1)))))
            dm=grid.derivative(v,axis,-1);dp=grid.derivative(u,axis,1);b1=float(np.sum(W*u*dm));b2=float(np.sum(W*dp*v));adjoint.append(abs(b1+b2)/max(1,abs(b1),abs(b2)))
            energy-=float(np.sum(W*v*grid.derivative(a*dm,axis,1)));strain+=float(np.sum(W*a*dm*dm))
        row={'n':n,'blocks':len(grid.blocks),'cells':grid.cells,'coarse_fine_face_groups':grid.coarse_fine_links,'derivative_rms':float(np.sqrt(np.sum(W*np.sum(np.array(errors)**2,axis=0)))),'derivative_max':float(np.max(abs(np.array(errors)))),'constant_derivative_max':constant,'adjoint_relative_max':max(adjoint),'variable_scalar_energy_relative_defect':abs(energy-strain)/max(1,abs(strain))}
        max_transfer=max(max(float(np.max(np.sum(abs(P),axis=1)))**2,float(np.max(np.sum(abs(Q),axis=1)))**2) for links in grid.links for _,_,P,Q in links if P is not None)
        row['tensor_face_value_L1_max']=max_transfer;report['rows'].append(row)
        if n==10:report['tensor']=tensor(grid)
    errors=[r['derivative_rms'] for r in report['rows']];ns=[r['n'] for r in report['rows']]
    report['orders']=[float(np.log(errors[i-1]/errors[i])/np.log(ns[i]/ns[i-1])) for i in range(1,len(ns))]
    report['gates']={'all_cell_refinement':min(report['orders'])>2,'constant_and_adjoint':all(r['constant_derivative_max']<1e-9 and r['adjoint_relative_max']<1e-10 for r in report['rows']),'scalar_energy':all(r['variable_scalar_energy_relative_defect']<1e-10 for r in report['rows']),'interface_amplification':max(r['tensor_face_value_L1_max'] for r in report['rows'])<16,'tensor_energy':all(r['energy_relative_defect']<1e-10 and r['bilinear_symmetry_relative']<1e-10 and r['rayleigh']>0 for r in report['tensor']['cases'].values()),'tensor_frechet':min(r['scaled_rms'] for r in report['tensor']['frechet'])<1e-7}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    start=time.monotonic();r=run();r['seconds']=time.monotonic()-start;(a.output/'assessment.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if all(r['gates'].values()) else 1)
