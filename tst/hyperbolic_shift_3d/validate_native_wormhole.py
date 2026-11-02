"""Independent Python validation only; never generates production initial fields."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole

def run(executable,out):
 out.mkdir(parents=True,exist_ok=True);samples={}
 for tol in (2e-11,2e-13,2e-14):
  path=out/f'native-tol-{tol}.csv'
  with path.open('w') as f:subprocess.run([str(executable),str(tol)],stdout=f,check=True)
  samples[tol]=np.genfromtxt(path,delimiter=',',names=True)
 x=samples[2e-13];w=IsotropicWormhole(1/3,192);use=x['r']<1;r=x['r'][use]
 q,d,dd,meta=w.data(r)
 ref={'chi':np.array([m['chi_total'] for m in meta]),'Ar':q[:,2],'K':q[:,3]-3,'alpha':np.array([m['physical_lapse'] for m in meta])*(1-r*r)/2}
 errors={k:float(max(abs(x[k][use]-v))) for k,v in ref.items()}
 outer=float(np.exp(-w.integral(np.array([w.y1]))[0]));lo=np.minimum(np.maximum(r,w.r0),outer)
 previous=None;shift={}
 for n in [96,192,384]:
  z,weights=np.polynomial.legendre.leggauss(n);t=(lo[:,None]+outer)/2+(outer-lo[:,None])/2*z
  qq,_,_,mm=w.data(t.ravel());alpha=np.array([m['physical_lapse'] for m in mm])*(1-t.ravel()**2)/2
  integrand=(-1.5*alpha*qq[:,2]/t.ravel()).reshape(t.shape)
  values=(outer-lo)/2*np.sum(integrand*weights,axis=1)
  shift[n]=dict(native_error=float(max(abs(values-x['shift_ratio'][use]))),reference_refinement=None if previous is None else float(max(abs(values-previous))))
  previous=values
 # All sampled radii are identical constants across tolerance runs.
 tolerance={}
 for tol in (2e-11,2e-14):
  assert np.array_equal(samples[tol]['r'],x['r'])
  tolerance[tol]={k:float(max(abs(samples[tol][k]-x[k]))) for k in ['chi','Ar','K','alpha','shift_ratio']}
 result=dict(scope='Pointwise native vs independent analytic Python values and original compactified-radius shift integral; not a spatial constraint convergence test.',geometric_errors=errors,shift_reference=shift,tolerance_differences=tolerance,matching_radius_error=abs(w.r0-x['r0'][0]),scale_error=abs(w.k-x['scale'][0]))
 assert max(errors.values())<5e-11 and shift[384]['native_error']<5e-11
 assert max(tolerance[2e-14].values())<5e-11
 (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.executable.resolve(),a.output)
