#!/usr/bin/env python3
"""First scalar-only global free-background projection proof; no PDE adoption."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse, hashlib, json, signal, sys, time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
signal.alarm(180);start=time.monotonic();root=Path.cwd();out=a.output;out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ev=root/'docs/elliptic-shift-3d/evidence';datafile=ev/'transition-complete-analysis-20261027/complete-all-cell-initial-sources.npz';background=ev/'transition-complete-balanced-20261027/background192.dat'
library=ev/'transition-complete-balanced-20261027/source/tst/elliptic_shift_1d'
sys.path.insert(0,str(library));from wormhole_isotropic import IsotropicWormhole
inputs={str(p):sha(p) for p in [datafile,background,library/'wormhole_isotropic.py',library/'wormhole_data.py']}
d=np.load(datafile);fields=list(d['fields']);cells=d['data'][:,0];at=lambda k:cells[:,fields.index(k)]
x=np.stack([at(k) for k in ['x','y','z']],axis=1);r=at('r');assert len(r)==47088 and np.all(r>0)
rr,inv=np.unique(r,return_inverse=True);model=IsotropicWormhole(mass=1/3,quadrature_order=96);_,_,_,meta=model.data(rr)
chi=np.array([row['chi_total'] for row in meta])[inv];volume=at('weight')*chi**1.5
assert np.all(volume>0);volume/=volume.sum();sw=np.sqrt(volume)
b192=np.loadtxt(background,skiprows=1);assert np.array_equal(b192[:,0],np.arange(len(r))) and np.allclose(b192[:,1],r,rtol=0,atol=2e-13)
powers=[(i,j,degree-i-j) for degree in range(6) for i in range(degree+1) for j in range(degree-i+1)]
base=np.stack([np.prod(x**power,axis=1) for power in powers]+[np.log(r),r],axis=1)
weighted=sw[:,None]*base;scales=np.linalg.norm(weighted,axis=0);weighted/=scales
Q,R=np.linalg.qr(weighted,mode='reduced');singular=np.linalg.svd(R,compute_uv=False);condition=float(singular[0]/singular[-1]);assert condition<1e8
results={};coefficients=[]
controls={'constant':np.ones(len(r)),'P2':x[:,0]+.3*x[:,1]-.7*x[:,2]+x[:,0]**2+.7*x[:,0]*x[:,1]+2*x[:,2]**2,'changing_logpower':1.09*np.log(r),'changing_radial':.5*r,'strong_angular':.1*np.cos(12*x[:,0]+7*x[:,1]-5*x[:,2]),'changed_radial':.2*np.log(r+.17),'fractional_angular':.01*x[:,0]**2/r**1.5,'unknown_noise':1e-12*np.sin(1.7*(np.arange(len(r))+1))}
for field,column in [('chi',2),('alpha',5)]:
    B=b192[:,column];residual=sw*B
    # Twice project to reduce cancellation, preserving the complete base span.
    for _ in range(2):residual-=np.einsum('ij,j->i',Q,np.einsum('ij,i->j',Q,residual,optimize=False),optimize=False)
    norm=float(np.linalg.norm(residual));assert norm>1e-11*np.linalg.norm(sw*B)
    functional=sw*residual/(norm*norm);coef=lambda q:float(np.sum(functional*q))
    base_response=np.array([coef(base[:,j]) for j in range(base.shape[1])]);assert max(abs(base_response))<1e-9
    row={'background_residual_norm':norm,'relative_residual':norm/float(np.linalg.norm(sw*B)),'coefficient_functional_L1':float(abs(functional).sum()),'coefficient_functional_L2':float(np.linalg.norm(functional)),'maximum_base_coefficient':float(max(abs(base_response))),'background_coefficient':coef(B),'controls':{name:coef(q) for name,q in controls.items()},'background_plus_angular_coefficient':coef(B+controls['strong_angular']),'coefficient_noise_bound_1e-12':float(abs(functional).sum()*1e-12)}
    assert abs(row['background_coefficient']-1)<1e-9
    # Nodal-background perturbation changes only the fitted functional.
    perturbed=sw*(B+controls['unknown_noise'])
    for _ in range(2):perturbed-=np.einsum('ij,j->i',Q,np.einsum('ij,i->j',Q,perturbed,optimize=False),optimize=False)
    wf=sw*perturbed/np.sum(perturbed*perturbed);row['B_nodal_1e-12_coefficient_on_unmodified_B']=float(np.sum(wf*B))
    results[field]=row;coefficients.append(functional)
with (out/'functionals.dat').open('w') as f:
    f.write(f'GLOBAL_BACKGROUND_FUNCTIONALS_V1 {len(r)}\n');np.savetxt(f,np.stack([np.arange(len(r)),*coefficients],axis=1),fmt=['%d','%.17g','%.17g'])
np.savez_compressed(out/'global-coefficient-functionals.npz',xyz=x,radius=r,conformal_volume=volume,chi=coefficients[0],alpha=coefficients[1])
m={'scope':'Scalar coefficient proof only; no derivative/operator/evolution change','all_active_cells':len(r),'volume':'Cartesian coordinate/conformal spatial cell volumes, normalized globally; not Penrose volume','basis':'all Cartesian P5 (56), log rho, rho; free B coefficient from twice orthogonalized residual','normalized_base_singular_values':singular.tolist(),'normalized_base_condition_2':condition,'fields':results,'input_sha256':inputs,'inputs_unchanged':all(sha(Path(p))==h for p,h in inputs.items()),'script_sha256':sha(Path(__file__)),'physical_effective_derivative_weights_qualified':False,'remaining_gates':['Same full halo/scri/puncture linear Dbase response on B and q','Effective gradient/Hessian physical functional bounds and all-cell source norms','Strong nonspherical/profile derivative error controls','Independent B96/192 derivative uncertainty and coupled pilots'],'scientific_acceptance':False,'production_adopted':False,'maximum_seconds':180,'one_CPU':True,'seconds':time.monotonic()-start};assert m['inputs_unchanged'];(out/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');print(json.dumps(m,indent=2));signal.alarm(0)
