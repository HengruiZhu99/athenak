#!/usr/bin/env python3
"""Independent selected-slice background log jets at actual active cells."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('catalog',type=Path);p.add_argument('output',type=Path);p.add_argument('--quadrature',type=int,default=192);a=p.parse_args();start=time.monotonic();points=np.genfromtxt(a.catalog,delimiter=',',names=True);points=points[points['kind']==0];assert np.array_equal(points['id'],np.arange(len(points))) and np.all((points['r']>0)&(points['r']<1));r,inverse=np.unique(points['r'],return_inverse=True);initial=IsotropicWormhole(mass=1/3,quadrature_order=a.quadrature);q,d,dd,meta=initial.data(r);chi=np.array([m['chi_total'] for m in meta]);alpha=np.array([m['physical_lapse'] for m in meta])*(1-r*r)/2;aref=(1+r*r)/2;ap=r+d[:,6];app=1+dd[:,6]
qc=np.log(chi);qcp=d[:,0]/chi;qcpp=dd[:,0]/chi-qcp*qcp;qa=np.log(alpha)-np.log(aref);qap=ap/alpha-r/aref;qapp=app/alpha-(ap/alpha)**2-1/aref+(r/aref)**2;values=np.column_stack([r,qc,qcp,qcpp,qa,qap,qapp])[inverse];assert np.all(np.isfinite(values))
with a.output.open('w') as f:
 f.write(f'BACKGROUND_LOG_JETS_V1 {len(points)} {a.quadrature}\n');np.savetxt(f,np.column_stack([points['id'],values]),fmt=['%d']+['%.17g']*7)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m={'mass':1/3,'reference':'Minkowski','quadrature':a.quadrature,'cells':len(points),'unique_radii':len(r),'k':initial.k,'r0':initial.r0,'catalog_sha256':sha(a.catalog),'script_sha256':sha(Path(__file__)),'isotropic_library_sha256':sha(Path(__file__).parents[1]/'elliptic_shift_1d/wormhole_isotropic.py'),'output_sha256':sha(a.output),'unknown_fields_modified':False,'seconds':time.monotonic()-start};a.output.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
