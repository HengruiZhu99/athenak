#!/usr/bin/env python3
"""Finite-sphere quadratures and independent stored1D mass, no evolution."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tst/elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
from puncture_chart import PunctureChart
from telegraph import options
from compare_coupled_family import interpolate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--library',type=Path,required=True);a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True);radii=np.array([.05,.1,.2,.3,.5,.7,.85,.95]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=dict(no_evolution=True,radii=radii.tolist(),library_sha256=sha(a.library),quadratures={},reference={})
for nmu,nphi in [(8,16),(16,32),(32,64)]:
 mu,w=np.polynomial.legendre.leggauss(nmu);phi=2*np.pi*(np.arange(nphi)+.5)/nphi;rows=[]
 for i in range(nmu):
  for j in range(nphi):rows.append([i,j,mu[i],phi[j],np.sqrt(1-mu[i]**2)*np.cos(phi[j]),np.sqrt(1-mu[i]**2)*np.sin(phi[j]),mu[i],w[i]*2*np.pi/nphi])
 f=o/f'quadrature-{nmu*nphi}.dat';np.savetxt(f,np.array(rows),header=f'GLPHI {nmu} {nphi}',comments='',fmt='%.17g');manifest['quadratures'][str(nmu*nphi)]=dict(n_mu=nmu,n_phi=nphi,sha256=sha(f),solid_weight_sum=float(np.sum(np.array(rows)[:,-1])))
initial=IsotropicWormhole(mass=1/3,quadrature_order=96);q,d,dd,meta=initial.data(radii);oracle=np.column_stack([radii,1+q[:,0],d[:,0],q[:,2],q[:,3]-3,(1+radii*radii)/2+q[:,6]]);np.savetxt(o/'initial-oracle.dat',oracle,header='r chi chi_r A_rr Khat alpha',comments='',fmt='%.17g');manifest['initial_oracle']=dict(mass=initial.mass,initial_data='exact original hyperboloidal isotropic wormhole, not time-symmetric replacement',quadrature_order=96,sha256=sha(o/'initial-oracle.dat'))
params=json.loads((a.reference/'reference-parameters.json').read_text());assert params['time']==.0002 and params['telegraph_width']==.24 and params['reference']=='Minkowski';manifest['selected_parameters']=params
for n in [1025,2049]:
 f=a.reference/f'radial-n{n}.npz';data=np.load(f);args=options(n,a.library);args.advection='centered';args.puncture_ko=0;ev=PunctureChart(args);assert np.array_equal(ev.r,data['r']);state=data['state'][:,:8];_,_,diag,_=ev.parts(state);r=ev.r;O=(1-r*r)/2;chi=ev.total_chi;K=ev.total_khat+2*ev.geometry_state[:,4];Arr=ev.geometry_state[:,2];chip=ev.geometry_first[:,0];finite=(r>0)&(r<1);rr=r[finite];OO=O[finite];cc=chi[finite];R=rr/(OO*np.sqrt(cc));normal=R*(K[finite]/3-OO*Arr[finite]/2);spatial=(1+rr*rr)/(2*OO)-rr*chip[finite]/(2*cc);direct=R/2*(1+normal**2-spatial**2);pcurv=OO*Arr[finite]-2*K[finite]/3;kcurv=np.sqrt(cc)*(OO*(2/rr-chip[finite]/cc)+2*rr);via_expansion=R/2*(1+R*R*(pcurv*pcurv-kcurv*kcurv)/4)
 sampled=interpolate(rr,diag[finite,10],radii,7);direct_sample=interpolate(rr,direct,radii,7);expansion_sample=interpolate(rr,via_expansion,radii,7);physical_errors={name:float(np.max(abs(value-data[name]))) for name,value in [('chi',ev.total_chi),('alpha',ev.total_alpha),('Khat',ev.total_khat),('A_rr',Arr)]};assert max(physical_errors.values())<1e-12
 np.savez_compressed(o/f'reference-mass-n{n}.npz',r=rr,mass=diag[finite,10],direct=direct,expansion=via_expansion,requested_r=radii,requested_mass=sampled,requested_direct=direct_sample,requested_expansion=expansion_sample)
 manifest['reference'][str(n)]=dict(input=str(f),input_sha256=sha(f),physical_reconstruction_errors=physical_errors,mass_at_requested=sampled.tolist(),direct_at_requested=direct_sample.tolist(),expansion_at_requested=expansion_sample.tolist(),mass_direct_difference_at_requested=float(np.max(abs(sampled-direct_sample))),normalization_Rp_over2_matches_negative_normalR=float(np.max(abs(R*pcurv/2+normal))),normalization_Rk_over2_matches_spatial=float(np.max(abs(R*kcurv/2-spatial))),finite_domain_only=True,omitted_reference_endpoints='r=0 AF coefficient and actualscri r=1 are not requested finite surfaces; no requested radius omitted')
manifest['loaded_private_python_modules']={str(Path(v.__file__).resolve().relative_to(root)):sha(Path(v.__file__).resolve()) for v in list(sys.modules.values()) if getattr(v,'__file__',None) and Path(v.__file__).resolve().is_relative_to(root) and Path(v.__file__).suffix=='.py'};(o/'reference-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(quadratures=list(manifest['quadratures']),mass_reference=manifest['reference'])))
