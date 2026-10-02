"""Fixed-order angular quadrature refinement against an accepted exact control."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
from check_hispid_controls import ROOT,shape_error

p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--baseline',required=True)
p.add_argument('--case',required=True,choices=('boost885','kerr95_boost885'))
p.add_argument('--ntheta',default='74');p.add_argument('--output',required=True)
p.add_argument('--baseline-flow-alpha',type=float,
               help='explicit recovery of the input flow alpha for older records missing this metadata')
a=p.parse_args();exe=Path(a.executable).resolve();baseline=json.loads(Path(a.baseline).read_text())
sha=hashlib.sha256(exe.read_bytes()).hexdigest()
if baseline['executable_sha256']!=sha:raise ValueError('quadrature comparison requires the identical AthenaK executable')
source_rows=[r for r in baseline['records'] if r['case']==a.case]
if len(source_rows)<3 or not source_rows[-1]['passed']:raise ValueError('accepted angular-refined baseline is required')
last=source_rows[-1];source=last['source'];lmax=last['lmax'];root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
alpha=last.get('flow_alpha',a.baseline_flow_alpha)
if alpha is None or not np.isfinite(alpha) or alpha<=0:raise ValueError('baseline flow alpha must be recorded or explicitly recovered')
if 'flow_alpha' in last and a.baseline_flow_alpha is not None and alpha!=a.baseline_flow_alpha:
    raise ValueError('explicit alpha conflicts with recorded baseline')
result=dict(case=a.case,executable_sha256=sha,baseline=str(Path(a.baseline).resolve()),lmax=lmax,
            flow_alpha=alpha,flow_alpha_source='record' if 'flow_alpha' in last else 'explicit_input_recovery',records=[],passed=False)
for nt in map(int,a.ntheta.split(',')):
    if nt<=last['ntheta'] or nt%2:raise ValueError('use a finer even theta quadrature')
    run=root/f'n{nt}';run.mkdir(exist_ok=False)
    cmd=[str(exe),'-i',str(ROOT/'inputs/hispid.athinput'),'problem/hispid_filename='+source['path'],
         'problem/hispid_source_sha256='+source['source_library_sha256'],f'fastflow/lmax={lmax}',f'fastflow/ntheta={nt}',
         f'problem/hispid_horizon_guess_scale={last["initial_scale"]}',f'fastflow/flow_alpha_beta_const_0={alpha}',
         'fastflow/flow_iterations_0=3000']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    start=time.monotonic();r=subprocess.run(cmd,cwd=run,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=1200)
    (run/'run.log').write_text(r.stdout)
    row=dict(ntheta=nt,returncode=r.returncode,seconds=time.monotonic()-start,command=cmd,passed=False)
    if r.returncode==0:
        values=np.atleast_2d(np.loadtxt(run/'hispid.horizon_summary_0.txt'))[-1]
        row.update(area=float(values[7]),expansion_rms=float(np.sqrt(values[8])),zero_evolution_verified='MeshBlock-cycles = 0' in r.stdout)
        row['relative_area_change']=float(abs(row['area']/last['area']-1))
        row['expansion_rms_change']=float(abs(row['expansion_rms']-last['expansion_rms']))
        row['shape_sampled_relative_linf']=shape_error(run/'hispid.horizon_shape_0.txt',lmax,.95 if 'kerr95' in a.case else 0)
        row['passed']=bool(np.isfinite(values).all() and row['expansion_rms']<1e-7 and row['relative_area_change']<1e-7
                           and row['expansion_rms_change']<1e-8 and row['shape_sampled_relative_linf']<1e-6 and row['zero_evolution_verified'])
    result['records'].append(row);result['passed']=all(x['passed'] for x in result['records'])
    (root/'quadrature.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
raise SystemExit(0 if result['passed'] else 1)
