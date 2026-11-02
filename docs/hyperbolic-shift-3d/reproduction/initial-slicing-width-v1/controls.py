"""Independent initial-data controls. Run from repository root; no evolution."""
from pathlib import Path
import sys,json,subprocess,hashlib
import numpy as np
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
B=ROOT/'build-native-initial-smoothing-v1/tst/hyperbolic_shift_3d'
commands=[]
def run(args,path):
 commands.append([str(x) for x in args])
 with path.open('x') as f:subprocess.run(args,stdout=f,check=True)
 return np.genfromtxt(path,delimiter=',',names=True)
results={}
for profile,end in [('bump',8),('poly6',12)]:
 name=f'{profile}-4-{end}'
 a={}
 for q,tol in [(1024,2e-14),(2048,2e-14),(2048,2e-13)]:
  a[q,tol]=run([str(B/'hyperbolic_shift_3d_initial_smoothing'),'4',str(end),str(q),str(tol),profile],P/f'{name}-q{q}-t{tol}.csv')
 result={'quadrature_relative':{},'tolerance_relative':{}}
 for f in ['H_rms','M_rms']:
  result['quadrature_relative'][f]=float(max(abs(a[1024,2e-14][f]/a[2048,2e-14][f]-1)))
  result['tolerance_relative'][f]=float(max(abs(a[2048,2e-13][f]/a[2048,2e-14][f]-1)))
 assert max(result['quadrature_relative'].values())<1e-4
 assert max(result['tolerance_relative'].values())<1e-5
 for v in a.values():
  assert np.max(abs(v['H_gauge_difference']))==0 and np.max(abs(v['M_gauge_difference']))==0
 result['constraint_gauge_difference']=0
 samples={}
 for tol in [2e-13,2e-14]:
  samples[tol]=run([str(B/'hyperbolic_shift_3d_native_wormhole'),str(tol),'4',str(end),profile],P/f'{name}-fields-{tol}.csv')
 x=samples[2e-13];use=x['r']<1;r=x['r'][use]
 w=IsotropicWormhole(1/3,384,4,end,profile)
 q,d,dd,meta=w.data(r)
 ref={'chi':np.array([m['chi_total'] for m in meta]),'Ar':q[:,2],'K':q[:,3]-3,'alpha':np.array([m['physical_lapse'] for m in meta])*(1-r*r)/2}
 result['independent_native_field_abs']={k:float(max(abs(x[k][use]-v))) for k,v in ref.items()}
 result['native_tolerance_field_abs']={k:float(max(abs(x[k]-samples[2e-14][k]))) for k in ['chi','Ar','K','alpha','shift_ratio']}
 assert max(result['independent_native_field_abs'].values())<5e-11
 assert max(result['native_tolerance_field_abs'].values())<5e-11
 r=np.unique(np.r_[np.linspace(.005,.999,501),np.linspace(.25,.85,501),.063])
 q,d,dd,meta=w.data(r)
 jets=P/f'{name}-analytic-jets.txt'
 with jets.open('x') as f:
  f.write(str(len(r))+'\n');np.savetxt(f,np.c_[r,q,d,dd],fmt='%.17g')
 c=run([str(B/'hyperbolic_shift_3d_initial_continuum'),str(jets)],P/f'{name}-continuum.csv')
 result['continuum_exterior_abs']={k:float(max(abs(c[k][c['r']>.063]))) for k in ['H','M','Mr']}
 assert max(result['continuum_exterior_abs'].values())<1e-10
 results[name]=result
old=ROOT/'build-native-shared-transition/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_native_wormhole'
new=B/'hyperbolic_shift_3d_native_wormhole'
x=subprocess.check_output([old]);y=subprocess.check_output([new]);assert x==y
results['default_preservation']={'byte_identical':True,'output_sha256':hashlib.sha256(y).hexdigest(),'old_binary_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_binary_sha256':hashlib.sha256(new.read_bytes()).hexdigest()}
(P/'controls.json').write_text(json.dumps(results,indent=2)+'\n')
(P/'control-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print(json.dumps(results,indent=2))
