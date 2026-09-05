"""Check all nonlinear reduction rows against independently differentiated RHS jets.

The production stencil differentiates nonlinear configuration RHS values, so
the remaining error must converge at order two as the oracle mesh is refined.
No GH, Einstein, reduction, or curl constraint is imposed on the input jets.
The conformal algebraic constraints are imposed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from frozen_reduction_source import coefficient
from make_inputs import parse


def gradients(u):
    return np.column_stack([u[22:25],u[25:43].reshape(3,6),u[43:46]/2,u[46:55].reshape(3,3)])


def configuration_rhs(u, rhs):
    return np.r_[rhs[:7],u[18]*rhs[0]+u[0]*rhs[18],rhs[19:22]]


def measure(run):
    data=np.genfromtxt(run/'nonlinear-jets.csv',delimiter=',',names=True)
    if not all(np.isfinite(data[n]).all() for n in data.dtype.names):
        raise ValueError('Nonfinite oracle data')
    pc=parse((run/'used_input.athinput').read_text())['pc_gh']
    rows=[]
    for case in range(3):
        selected=data[data['case']==case]
        U=np.empty((3,3,55));F=np.empty_like(U);h=np.empty(3)
        for d in range(3):
            for j in range(3):
                sample=selected[(selected['axis']==d)&(selected['offset']==j-1)]
                if len(sample)!=55 or not np.array_equal(sample['var'],np.arange(55)):
                    raise ValueError('Incomplete or reordered oracle jet')
                U[d,j]=sample['state'];F[d,j]=sample['rhs'];h[d]=sample['h'][0]
        if not np.array_equal(U[:,1],np.tile(U[0,1],(3,1))):
            raise ValueError('Repeated center state mismatch')
        u=U[0,1];rhs=F[0,1]
        du=(U[:,2]-U[:,0])/(2*h[:,None])
        dx=np.column_stack([du[:,:7],u[18]*du[:,0]+u[0]*du[:,18],du[:,19:22]])
        e=gradients(u)-dx
        de=np.array([gradients(du[d]) for d in range(3)])
        de[:,:,7]-=np.outer(du[:,0],du[:,18])+np.outer(du[:,18],du[:,0])
        actual=gradients(rhs)-np.array([(configuration_rhs(U[d,2],F[d,2])
                    -configuration_rhs(U[d,0],F[d,0]))/(2*h[d]) for d in range(3)])
        C=coefficient(u,float(pc.get('shift_eta',2)),float(pc.get('shift_switch_z0',.1)),
                      float(pc.get('shift_switch_z1',.5)))
        expected=np.einsum('d,dia->ia',u[19:22],de)+du[:,19:22]@e+e@C.T-float(pc['reduction_rate'])*e
        error=actual-expected
        row=dict(case=case,h=float(h[0]),max_error=float(abs(error).max()),
                 by_family={name:float(abs(error[:,lo:hi]).max())
                            for name,lo,hi in [('p',0,1),('Q',1,7),('L',7,8),('B',8,11)]},
                 error_components=error.tolist(),max_reduction=float(abs(e).max()))
        rows.append(row)
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();rows=[r for run in args.runs for r in measure(run)]
    checks=[]
    for case in range(3):
        series=sorted([r for r in rows if r['case']==case],key=lambda r:-r['h'])
        for a,b in zip(series[:-1],series[1:]):
            order=float(np.log(a['max_error']/b['max_error'])/np.log(a['h']/b['h']))
            checks.append(dict(case=case,coarse_h=a['h'],fine_h=b['h'],order=order))
            if order < 1.95 or b['max_error']>=a['max_error']:
                raise AssertionError(f'Nonlinear subsidiary discrepancy does not converge: {checks[-1]}')
    if len(checks)<3:
        raise ValueError('Need at least two spacings for all three backgrounds')
    args.output.write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
    print('PASS all 33 nonlinear reduction components; residual converges with the differentiated RHS stencil')
    for check in checks: print(check)
