#!/usr/bin/env python3
"""Separate frozen continuum constraints from native derivative/halo error.

Consumes --dump-constraints output from the actual native zero-step audit.
Uses the frozen source's independent metric/longitudinal jet observer, and the
3D conformal transformation gamma=psi^4 h, psi=phi/sqrt(Omega). No field repair.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from convert_binary import ARCHIVE_SHA,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--initial-data-repo',type=Path,required=True)
 p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();root=a.initial_data_repo.resolve()
 archive=root/'validation/hyperboloidal/evidence/akv/build-hyperboloidal/akv/calibration-01/evaluation-01/solve/solution.npz'
 if sha(archive)!=ARCHIVE_SHA:raise ValueError('archive hash mismatch')
 report=json.loads((archive.parent/'report.json').read_text());snapshot=archive.parent/'source_snapshot'
 for name,want in report['implementation_sha256'].items():
  if sha(snapshot/name)!=want:raise ValueError('frozen source mismatch')
 sys.path.insert(0,str(snapshot));import hyperboloidal_hispid as hispid
 for receipt in (root/'build-hyperboloidal/hispid-bridge').glob('*.json'):
  if json.loads(receipt.read_text())['sources']==report['bridge_source_sha256']:
   native=receipt.with_name('libhispid-hyperboloidal-'+receipt.stem+'.so')
   if native.is_file() and sha(native)==report['native_bridge_sha256']:break
 else:raise ValueError('frozen ABI unavailable')
 hispid.build_bridge=lambda **kwargs:(native,report['bridge_source_sha256'])
 from hyperboloidal_binary_sampler import BinaryCTTSampler
 from hyperboloidal_curved_verify import metric_geometry_from_jets,longitudinal_from_jets
 sampler=BinaryCTTSampler.from_archive(archive)
 paths=sorted(a.native.glob('constraint-points-*.bin'))
 if not paths:raise ValueError('native constraints dump missing')
 sums=np.zeros((2,9));count=0
 for path in paths:
  rows=np.memmap(path,dtype='=f8',mode='r').reshape(-1,12)
  for start in range(0,len(rows),1024):
   q=np.asarray(rows[start:start+1024]);x=q[:,:3];d=sampler.conformal(x)
   g=metric_geometry_from_jets(d['h'],d['dh'],d['ddh']);inv=g['inverse'];G=g['Gamma']
   phi=d['phi'];o=d['omega'];dp=d['grad_phi'];do=d['domega']
   dl=dp/phi[:,None]-.5*do/o[:,None]
   ddl=d['hess_phi']/phi[:,None,None]-dp[:,:,None]*dp[:,None,:]/phi[:,None,None]**2
   ddl-=.5*(d['hess_omega']/o[:,None,None]-do[:,:,None]*do[:,None,:]/o[:,None,None]**2)
   lap=np.einsum('nij,nij->n',inv,ddl-np.einsum('nkij,nk->nij',G,dl))
   grad2=np.einsum('nij,ni,nj->n',inv,dl,dl)
   anorm=np.einsum('nik,njl,nij,nkl->n',d['h'],d['h'],d['Ahat'],d['Ahat'])
   H=o**2*phi**-4*(g['R']-8*lap-8*grad2)+(2/3)*d['K']**2-o**6*phi**-12*anorm
   div=longitudinal_from_jets(d['h'],d['dh'],d['ddh'],d['W'],d['grad_W'],d['hess_W'])['div_L']
   # The overlap reconstructs Ahat with the shell free provider; below the
   # lower blend join the original core split supplies M and its divergence.
   core=np.linalg.norm(x,axis=1)<sampler.blend_radii[0]
   for mask,patch in [(core,'core'),(~core,'shell')]:
    if mask.any():div[mask]+=sampler._free(x[mask],patch)['divM']
   M=o[:,None]**3*phi[:,None]**-6*np.einsum('nij,nj->ni',d['h'],div)-(2/3)*d['grad_K']
   physical_inverse=inv*(o**2*phi**-4)[:,None,None]
   norm=lambda v:np.einsum('nij,ni,nj->n',physical_inverse,v,v)
   nativeM=q[:,6:9]
   values=np.column_stack((np.ones(len(q)),H**2,norm(M),(q[:,5]-H)**2,
       norm(nativeM-M),q[:,5]**2,norm(nativeM),q[:,9],q[:,10]**2+q[:,11]**2))
   if not np.isfinite(values).all():raise ValueError('nonfinite observer')
   for region in range(2):sums[region]+=(values*(q[:,3]*(q[:,4] if region else 1))[:,None]).sum(axis=0)
   count+=len(q)
  print(path.name,'complete; points',count,flush=True)
 result={'archive_sha256':ARCHIVE_SHA,'native_dump_sha256':{p.name:sha(p) for p in paths},
   'observer_sha256':sha(__file__),'points':count,'regions':{},
   'interpretation':'source RMS is continuum residual of the frozen finite-resolution archive; native-minus-source includes derivative truncation and halo interpolation, not a separately solved constraint field; source Z=0 by exported contracted connection convention'}
 labels=['source_H_rms','source_M_rms','native_minus_source_H_rms','native_minus_source_M_rms',
         'native_H_rms','native_M_rms','native_Z_rms','algebraic_identity_rms']
 for name,row in zip(['full','horizon_exterior'],sums):
  result['regions'][name]={'volume':row[0],**dict(zip(labels,np.sqrt(row[1:]/row[0]).tolist()))}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
