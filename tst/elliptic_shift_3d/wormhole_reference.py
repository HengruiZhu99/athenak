#!/usr/bin/env python3
"""Independent existing spherical AF chart, selected lapse and vector-flux reduction.

Only differentiation/advection controls differ from the verified long runs:
centered derivatives and zero KO isolate continuum spatial error against the
3D pilot. The physical initial slice, gauge source and telegrapher are unchanged.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from puncture_chart import PunctureChart
from telegraph import options,window,python_rhs
from wormhole_isotropic import IsotropicWormhole

def evolve(n,library,dt=1e-4,steps=2):
 args=options(n,library);args.advection='centered';args.puncture_ko=0
 ev=PunctureChart(args);q,*_=ev.initial_wormhole(IsotropicWormhole(mass=1/3));B=np.zeros(n)
 support=window(ev.r,inner_profile='af-regular')
 def rhs(y):
  rate,flux,_=python_rhs(ev,y[:,:8],y[:,8],.005,.02,support)
  return np.column_stack([rate,flux])
 y=np.column_stack([q,B])
 for _ in range(steps):
  k1=rhs(y);k2=rhs(y+dt*k1/2);k3=rhs(y+dt*k2/2);k4=rhs(y+dt*k3);y+=dt*(k1+2*k2+2*k3+k4)/6
 ev.parts(y[:,:8]);physical=ev.to_physical(y[:,:8])
 return dict(r=ev.r,alpha=ev.total_alpha,chi=ev.total_chi,g_rr=np.ones(n),A_rr=physical[:,2],
  Khat=ev.total_khat,Theta=physical[:,4],Lambda_r=physical[:,5],beta_r=ev.shift_deviation-ev.r,B=y[:,8],state=y)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();results=[]
 for n in [129,257,513]:
  data=evolve(n,a.library);np.savez_compressed(a.output/f'radial-n{n}.npz',**data)
  results.append(dict(n=n,min_chi=float(min(data['chi'][1:])),min_alpha=float(min(data['alpha'][1:]))));print(n,flush=True)
 manifest=dict(method='existing independent spherical AF positive chart, centered/zero-KO control',
  source_reference='Minkowski',mass=1/3,dt=1e-4,steps=2,time=.0002,seconds=time.monotonic()-start,
  kappa=.005,tau=.02,nu=1,restoring='multiplicative physical-log',background_source='collapsed-inner',
  library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),results=results,
  source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((Path(__file__).parents[1]/'elliptic_shift_1d').glob('*.py'))})
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
