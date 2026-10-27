#!/usr/bin/env python3
"""All-cell derivative and tensor checks on a cube plus actual spherical shells.

Unadopted spatial candidate. The outer nodes have r=1 and Omega=0 exactly up to
roundoff, but no hyperboloidal boundary equations or evolution are solved here.
The core is uniform Cartesian and unrefined in this first mapped test.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from compatible_tensor_gauge import TensorGauge


def zero_outer(value,grid):
    # Homogeneous Dirichlet test directions, not a reset of evolved fields.
    for b,a,s in grid.outer:
        index=[b,slice(None),slice(None),slice(None)];index[1+a]=-1 if s else 0
        value[tuple(index)]=0
    return value


def tensor_checks(grid):
    g=np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]])
    metric=np.broadcast_to(g,grid.weights.shape+(3,3)).copy();gauge=TensorGauge(grid,metric)
    W=grid.weights*np.sqrt(np.linalg.det(g));rng=np.random.default_rng(271128)
    xyz=grid.coordinates;r2=np.sum(xyz*xyz,axis=-1)
    smooth=zero_outer((1-r2)[...,None]**2*np.sin(xyz*2),grid)
    def dot(b,c):return float(np.sum(W*np.einsum('...i,ij,...j->...',b,g,c,optimize=False)))
    cases={}
    for name,b in [('smooth',smooth),('random',zero_outer(rng.normal(size=smooth.shape),grid))]:
        h=gauge.rate(b);action=gauge.apply(b)
        strain=.5*np.sum(W*np.einsum('...ik,...jl,...ij,...kl->...',gauge.gi,gauge.gi,h,h,optimize=False))
        c=zero_outer(rng.normal(size=b.shape),grid);left=dot(b,gauge.apply(c));right=dot(action,c)
        cases[name]={'energy':dot(b,action),'strain':float(strain),'energy_relative_defect':float(abs(dot(b,action)-strain)/max(1,abs(strain))),'symmetry_relative_defect':abs(left-right)/max(1,abs(left),abs(right)),'rayleigh':dot(b,action)/dot(b,b)}
    # A smooth non-diagonal, non-unit-determinant metric exercises every inverse
    # variation term. General-metric coercivity is not inferred from this check.
    variable=metric.copy()
    variable[...,0,0]+=.08*np.sin(xyz[...,0]+xyz[...,2])
    variable[...,1,1]+=.05*np.cos(2*xyz[...,1])
    variable[...,0,2]+=.03*np.sin(xyz[...,1]);variable[...,2,0]=variable[...,0,2]
    vg=TensorGauge(grid,variable);h=vg.rate(smooth);exact=vg.variation(h);fd=[]
    for eps in [1e-3,1e-4,1e-5]:
        plus=variable+eps*h;minus=variable-eps*h
        if min(np.min(np.linalg.eigvalsh(plus)),np.min(np.linalg.eigvalsh(minus)))<=0:raise ValueError('finite epsilon metric lost SPD')
        delta=(vg.connection(plus)-vg.connection(minus))/(2*eps)-exact
        fd.append({'epsilon':eps,'scaled_rms':float(np.sqrt(np.sum(W[...,None]*delta**2))/max(1,np.sqrt(np.sum(W[...,None]*exact**2)))),'maximum':float(np.max(abs(delta)))})
    return {'constant_metric':cases,'variable_metric_frechet':fd,'variable_metric_det_min':float(np.min(np.linalg.det(variable))),'variable_metric_det_max':float(np.max(np.linalg.det(variable)))}


def run():
    report={'scope':__doc__,'rows':[]}
    for n in [10,12,14]:
        grid=ShellGrid(n);X=grid.coordinates;W=grid.weights
        coefficients=np.array([.3,.2,-.1]);f=np.exp(np.einsum('...i,i->...',X,coefficients,optimize=False))
        rng=np.random.default_rng(421);u=rng.normal(size=f.shape);v=rng.normal(size=f.shape)
        errors=[];adjoint=[];constant=0.
        for a in range(3):
            errors.append(grid.derivative(f,a,-1)-coefficients[a]*f)
            constant=max(constant,float(np.max(abs(grid.derivative(np.ones_like(f),a,-1)))))
            left=float(np.sum(W*u*grid.derivative(v,a,-1)));right=float(np.sum(W*v*grid.derivative(u,a,1)))
            boundary=grid.boundary_integral(u,v,a)
            adjoint.append(abs(left+right-boundary)/max(1,abs(left),abs(right),abs(boundary)))
        error2=np.sum(np.array(errors)**2,axis=0)
        normals=[];omega=[]
        for face in grid.outer:
            pos=grid.trace(X,face);normal=grid.normal(face);unit=normal/np.linalg.norm(normal,axis=-1)[...,None]
            normals.append(float(np.max(np.linalg.norm(unit-pos,axis=-1))));omega.append(float(np.max(abs((1-np.sum(pos**2,axis=-1))/2))))
        amplification=max(max(float(np.max(np.sum(abs(P),axis=1)))**2,float(np.max(np.sum(abs(Q),axis=1)))**2) for _,_,P,Q,_,_ in grid.links if P is not None)
        report['rows'].append({'n':n,'cells':grid.cells,'blocks':len(grid.blocks),'radial_partition':list(grid.radial),'rms':float(np.sqrt(np.sum(W*error2)/np.sum(W))),'maximum':float(np.max(abs(np.array(errors)))),'core_rms':float(np.sqrt(np.sum(W[0]*error2[0])/np.sum(W[0]))),'shell_rms':float(np.sqrt(np.sum(W[1:]*error2[1:])/np.sum(W[1:]))),'constant_derivative_max':constant,'adjoint_relative_max':max(adjoint),'geometry_divergence_max':grid.metric_identity_max,'interface_normal_mismatch':grid.normal_mismatch,'surface_discrete_normal_error':max(normals),'outer_Omega_max':max(omega),'core_shell_value_L1_max':amplification,'volume':float(np.sum(W))})
        if n==10:report['tensor']=tensor_checks(grid)
    r=report['rows'];report['orders']=[float(np.log(r[i-1]['rms']/r[i]['rms'])/np.log(r[i]['n']/r[i-1]['n'])) for i in range(1,len(r))]
    report['gates']={'all_cell_refinement':min(report['orders'])>2,'constants_and_pair_identity':all(x['constant_derivative_max']<1e-8 and x['adjoint_relative_max']<1e-10 for x in r),'interface_amplification':max(x['core_shell_value_L1_max'] for x in r)<16,'actual_Omega_zero_nodes':max(x['outer_Omega_max'] for x in r)<1e-14,'constant_metric_energy':all(x['energy_relative_defect']<1e-9 and x['symmetry_relative_defect']<1e-9 and x['rayleigh']>0 for x in report['tensor']['constant_metric'].values()),'full_metric_frechet':min(x['scaled_rms'] for x in report['tensor']['variable_metric_frechet'])<1e-7}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    result=run();result['seconds']=time.monotonic()-start;(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
