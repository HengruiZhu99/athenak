"""Reproduce geometry audits without touching an evolution or frozen build."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=ROOT/'output/refined-convergence';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=HERE/'audit_initial_geometry.cpp';original=OUT/'original-include/z4c/hyperboloidal';original.mkdir(parents=True,exist_ok=True)
ref='21676d39e'
for name in ['native_wormhole.hpp','native_boosted_wormhole.hpp']:
 (original/name).write_bytes(subprocess.check_output(['git','show',f'{ref}:src/z4c/hyperboloidal/{name}'],cwd=ROOT))
commands=[]
for label,inc,defs in [('initial',ROOT/'src',['-DUSE_GEOMETRY_AUDIT']),('original',OUT/'original-include',[])]:
 cmd=['c++','-std=c++17','-O2','-ffp-contract=off',*defs,'-I',str(inc),str(source),'-o',str(OUT/f'audit-{label}')];commands.append(cmd);subprocess.run(cmd,check=True)
cases=[(v,lo,hi) for lo,hi in [(4,12),(6,18),(8,24),(12,36),(20,40)] for v in [0,.1]]
def run(c):
 v,lo,hi=c;p=OUT/f'{"stationary" if v==0 else "boost"}-{lo}-{hi}.csv'
 with p.open('w') as f:subprocess.run([str(OUT/'audit-initial'),str(v),str(lo),str(hi)],stdout=f,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cases))
with (OUT/'boost-original.csv').open('w') as f:subprocess.run([str(OUT/'audit-original'),'.1'],stdout=f,check=True)
a=np.genfromtxt(OUT/'boost-original.csv',delimiter=',',names=True);b=np.genfromtxt(OUT/'boost-4-12.csv',delimiter=',',names=True)
assert a.shape==b.shape
errs={k:float(np.max(abs(a[k]-b[k]))) for k in a.dtype.names};assert max(errs.values())==0
summary={'reference_commit':ref,'compile_commands':commands,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [source,ROOT/'src/z4c/hyperboloidal/native_boosted_wormhole.hpp',ROOT/'src/z4c/hyperboloidal/native_wormhole.hpp']},'default_vs_original_max_difference':errs,'cases':{}}
for v,lo,hi in cases:
 p=OUT/f'{"stationary" if v==0 else "boost"}-{lo}-{hi}.csv';d=np.genfromtxt(p,delimiter=',',names=True)
 assert all(np.isfinite(d[k]).all() for k in d.dtype.names)
 assert np.min(d['Q'])>0 and np.min(d['detJ'])>0 and np.min(d['map_radial'])>0
 assert np.max(d['stationary_geometry_error'])<5e-10
 near=d[(d['r']>.3)&(d['r']<.4)]
 summary['cases'][p.name]={'sha256':sha(p),'points':len(d),'minimum_Q_over_Q0':float(np.min(d['Q']/d['Q0'])),'minimum_Q':float(np.min(d['Q'])),'minimum_boost_J':float(np.min(d['detJ'])),'minimum_map_radial':float(np.min(d['map_radial'])),'minimum_map_tangential':float(np.min(d['map_tangential'])),'affine_radius':float(d['affine_radius'][0]),'k':float(d['k'][0]),'stationary_geometry_error':float(np.max(d['stationary_geometry_error'])),'r03_04_min_Q_over_Q0':float(np.min(near['Q']/near['Q0']))}
(HERE/'initial-geometry-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: default point fields exactly preserved; all ten geometry samples regular; stationary geometry agrees with independent native initializer.')
