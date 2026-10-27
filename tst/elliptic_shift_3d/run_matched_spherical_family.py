#!/usr/bin/env python3
"""Bounded, source-bound 3D/1D matched-formulation pilot. All cells remain."""
import argparse,gzip,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,nargs='+',default=[6,8,10]);p.add_argument('--controls',action='store_true');p.add_argument('--steps',type=int,default=4);p.add_argument('--dt',type=float,default=.000048828125);a=p.parse_args()
if not (1<=a.steps<=16 and 0<a.dt<=.000048828125):p.error('bounded pilot requires 1..16 steps and 0<dt<=4.8828125e-5')
build=a.build.resolve();ref=a.reference.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
bm=json.loads((build/'manifest.json').read_text());binary=build/'probe';assert sha(binary)==bm['binary_sha256']
receipt=json.loads((ref/'receipt.json').read_text());assert all(sha(ref/'source'/f)==digest for f,digest in receipt['source_hashes'].items() if not f.endswith(('.dylib','.so'))) # Initial data use Python sources only; platform libraries are not in the portable archive.
sys.path.insert(0,str(ref/'source'))
from wormhole_isotropic import IsotropicWormhole
m={'scope':'Genuine 3D host research RK4 compared with the successful 1D gauge; spatial operators/KO differ; no scientific acceptance yet',
   'binary_sha256':sha(binary),'build_manifest_sha256':sha(build/'manifest.json'),'reference_receipt_sha256':sha(ref/'receipt.json'),
   'runner_sha256':sha(Path(__file__)),'predeclared':{'max_seconds_per_command':180,'max_cells':50000,'time':a.steps*a.dt,'steps':a.steps,'dt':a.dt,
   'metric_freeze_rms_target':1e-8,'scri_pole_target':1e-7,'all_cells_retained':True,'puncture_exclusions':False,'constant_linear_shift_not_pinned':True},'commands':[],'cases':{}}
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,cmd):
    row={'name':name,'argv':list(map(str,cmd)),'maximum_seconds':180};m['commands'].append(row);save();start=time.monotonic()
    with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
        child=subprocess.Popen(row['argv'],stdout=so,stderr=se,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
        row['owned_process_group']=child.pid;save()
        try:row['exit_code']=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            row['exit_code']=124
    row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True)
    return row['exit_code']
initial=IsotropicWormhole(mass=1/3,quadrature_order=96)
for n in a.n:
    coords=out/f'coordinates-n{n}.csv'
    if run(f'coordinates-n{n}',[binary,'coordinates',n,coords]):continue
    points=np.genfromtxt(coords,delimiter=',',names=True);unique,inverse=np.unique(points['r'],return_inverse=True)
    q,_,_,meta=initial.data(unique);chi=np.array([r['chi_total'] for r in meta]);alpha=np.array([r['physical_lapse'] for r in meta])*(1-unique**2)/2
    values=np.c_[unique,chi,alpha,q[:,2],q[:,3]-3][inverse];data=out/f'initial-n{n}.dat'
    with data.open('x') as f:f.write(f'ISOTROPIC_M1_3 {len(points)}\n');np.savetxt(f,values,fmt='%.17g')
    cases=[(f'n{n}',1e-4,1e-10)]
    if a.controls and n==6:cases += [('n6-floor-off',1e-4,0),('n6-unattenuated',0,1e-10)]
    for name,epsilon,floor in cases:
        code=run(name,[binary,'evolve',n,data,out/name,a.steps,a.dt,epsilon,floor])
        result={'child_exit_code':code,'initializer_sha256':sha(data),'coordinates_sha256':sha(coords),'lapse_attenuation':epsilon,'lapse_floor':floor}
        for label in ['initial','final']:
            f=out/name/(label+'.json')
            if f.exists():result[label]=json.loads(f.read_text())
        if 'final' in result:
            r=result['final'];result['gates']={'actual_metric_freeze':r['actual_metric_Vt_rms']<=1e-8,'scri_pole':r['scri_pole']<=1e-7}
        m['cases'][name]=result;save()
        for f in (out/name).glob('*.csv'):
            f.with_suffix('.csv.gz').write_bytes(gzip.compress(f.read_bytes(),mtime=0))
(out/'.gitignore').write_text('/*.csv\n/initial-*.dat\n/*/*.csv\n')
m['binary_unchanged']=sha(binary)==m['binary_sha256'];save();assert m['binary_unchanged']
# A failed physical gate is a failure, even when every child completed.
passed=bool(m['cases']) and all(r['child_exit_code']==0 and all(r.get('gates',{'missing':False}).values()) for r in m['cases'].values())
m['scientific_gates_passed']=passed;save();raise SystemExit(0 if passed else 1)
