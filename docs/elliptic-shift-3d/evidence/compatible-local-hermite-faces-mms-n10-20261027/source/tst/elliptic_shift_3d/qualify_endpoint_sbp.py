#!/usr/bin/env python3
"""Local closure, mapped tensor and continuum derivative checks at actual scri."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from endpoint_sbp import local,shell_grid
from affine_sbp_probe import AffineConsistentGrid
from compatible_sbp import mm
from qualify_mapped_sbp import tensor_checks


def run():
    result={'scope':__doc__,'local':[],'mapped':[]}
    for n in [10,12,14,20,40]:
        q=local(n);H=q['H'];D=q['D'];R=q['R'];B=mm(q['trace'].T,np.diag([-1.,1.])@q['trace'])
        error=mm(D-R/H[:,None],np.exp(q['x']))-np.exp(q['x'])
        moments=mm(leg.legvander(2*q['x']-1,4).T,H);moments[0]-=1
        result['local'].append({'n':n,'minimum_weight':float(min(H)),
            'sbp_max':float(np.max(abs(H[:,None]*D+D.T*H[None,:]-B))),
            'polynomial_max':float(np.max(abs(mm(D-R/H[:,None],q['V'])-q['dV']))),
            'quadrature_degree_four_error':float(np.max(abs(moments))),
            'smooth_rms':float(np.sqrt(sum(H*error*error))),
            'smooth_endpoint_max':float(max(abs(error[0]),abs(error[-1])))})
    for n in [10,12,14]:
        grid=AffineConsistentGrid(shell_grid(n,radial=(0,.45,1.)))
        X=grid.coordinates;W=grid.weights;u=np.exp(.3*X[...,0]+.2*X[...,1]-.1*X[...,2])
        coeff=[.3,.2,-.1];errors=[];affine=0.
        boundary=np.zeros(W.shape,dtype=bool)
        for b,_,_ in grid.outer:boundary[b,-1]=True
        for axis in range(3):
            errors.append(grid.derivative(u,axis,-1)-coeff[axis]*u)
            affine=max(affine,float(np.max(abs(grid.derivative(X,axis,-1)-np.eye(3)[axis]))))
        error=np.sqrt(np.sum(np.array(errors)**2,axis=0))
        amp=max(float(np.max(np.sum(abs(T),axis=1))) for link in grid.links for T in link[2:4] if T is not None)
        row={'n':n,'cells':grid.cells,'affine_max':affine,'mortar_amplification_max':amp,
             'smooth_all_node_rms':float(np.sqrt(np.sum(W*error**2)/np.sum(W))),
             'smooth_scri_max':float(np.max(error[boundary])),
             'metric_identity_max':grid.metric_identity_max,'normal_mismatch':grid.normal_mismatch,
             'actual_scri_radius_error':float(np.max(abs(np.linalg.norm(X[boundary],axis=-1)-1)))}
        if n==10:row['tensor']=tensor_checks(grid)
        result['mapped'].append(row)
    family=result['mapped'];orders=[float(np.log(family[i-1]['smooth_all_node_rms']/family[i]['smooth_all_node_rms'])/np.log((family[i]['n']-1)/(family[i-1]['n']-1))) for i in [1,2]]
    result['mapped_smooth_orders']=orders
    tensor=family[0]['tensor']
    result['gates']={'local_algebra':all(r['sbp_max']<1e-10 and r['polynomial_max']<1e-9 and r['quadrature_degree_four_error']<1e-12 and r['minimum_weight']>0 for r in result['local']),
        'mapped_affine_and_interfaces':all(r['affine_max']<1e-9 and r['normal_mismatch']<1e-9 and r['mortar_amplification_max']<16 and r['actual_scri_radius_error']<1e-14 for r in family),
        'smooth_mapped_refinement':min(orders)>1.5,
        'constant_metric_energy':all(r['energy_relative_defect']<1e-9 and r['symmetry_relative_defect']<1e-9 and r['rayleigh']>0 for r in tensor['constant_metric'].values()),
        'full_variable_metric_variation':min(r['scaled_rms'] for r in tensor['variable_metric_frechet'])<1e-7}
    result['evolution_adopted']=False
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    result=run();result['seconds']=time.monotonic()-start
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
