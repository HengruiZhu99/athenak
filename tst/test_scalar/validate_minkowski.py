#!/usr/bin/env python3
"""Evaluate declared scalar gates from preserved runs; never launches simulations."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def history(p):return {(float(r['time']),r['region']):r for r in csv.DictReader((p/'history.csv').open())}
def snapshot(p,t):return np.fromfile(p/f'fields-t{t:.3f}.f64').reshape(-1,12)
def norm(x,w):
 if x.ndim>1:x=np.linalg.norm(x,axis=1)
 return dict(L2=float(np.sqrt(np.sum(w*x*x)/np.sum(w))),Linf=float(max(abs(x))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence
 criteria=json.loads((E/'criteria.json').read_text());paths=[E/'cpu-v2/minkowski-n12',E/'cpu-v3/minkowski-n18',E/'cpu-v3/minkowski-n24'];h=[history(p) for p in paths];checks=[]
 complete=all((p/'complete.txt').exists() for p in paths)
 for region in ['all','bulk','shell']:
  for t in [2,6,12]:
   for field in ['psi','v','Dpsi','reduction']:
    for name in ['L2','Linf']:
     values=[float(r[t,region][field+'_'+name]) for r in h];orders=[math.log(values[k]/values[k+1])/math.log([1.5,4/3][k]) for k in [0,1]]
     checks.append(dict(time=t,region=region,field=field,norm=name,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 fine0=snapshot(paths[2],0);scale=float(np.max(np.linalg.norm(fine0[:,6:9],axis=1)));cmax=max(float(r['reduction_Linf']) for k,r in h[2].items() if k[1]=='all');reduction=dict(initial_derivative_scale=scale,max_reduction=cmax,normalized_max=cmax/scale,passed=cmax/scale<=.02)
 timechecks=[];timepaths=[paths[1],E/'cpu-v3/minkowski-n18-cfl0.09',E/'cpu-v3/minkowski-n18-cfl0.045'];timecomplete=all((p/'complete.txt').exists() for p in timepaths)
 if timecomplete:
  for t in [2,6,12]:
   arrays=[snapshot(p,t) for p in timepaths];w=arrays[0][:,3]
   for fld,cols in [('psi',slice(4,5)),('v',slice(5,6)),('Dpsi',slice(9,12))]:
    half=norm(arrays[0][:,cols]-arrays[1][:,cols],w);quarter=norm(arrays[1][:,cols]-arrays[2][:,cols],w)
    for name in ['L2','Linf']:
     gap=abs(float(h[1][t,'all'][fld+'_'+name])-float(h[2][t,'all'][fld+'_'+name]));timechecks.append(dict(time=t,field=fld,norm=name,half_difference=half[name],quarter_difference=quarter[name],conservative_spatial_gap=gap,fraction=half[name]/gap,quarter_ratio=quarter[name]/half[name],passed=half[name]<.1*gap and quarter[name]<.2*half[name]))
 outer=[];outer_complete=True
 for n,root in [(18,E/'cpu-v3'),(24,E/'cpu-v4')]:
  ref=np.genfromtxt(E/f'cpu-v3/minkowski-n{n}/waveform.csv',delimiter=',',names=True);refwave=np.c_[ref['real22'],ref['imag22']]
  for cap in [.9,.95]:
   directory=root/f'minkowski-n{n}-cap{cap}';done=(directory/'complete.txt').exists();outer_complete &= done
   if not done:continue
   f=np.genfromtxt(directory/'waveform.csv',delimiter=',',names=True);delta=np.c_[f['real22'],f['imag22']]-refwave;relative=float(np.linalg.norm(delta)/np.linalg.norm(refwave));outer.append(dict(n=n,cap=cap,relative_L2=relative,passed=relative<=.01,reference=str(E/f'cpu-v3/minkowski-n{n}')))
 charrows=[]
 for path in paths:
  rows=list(csv.DictReader((path/'characteristics.csv').open()));lo=max(abs(float(r['cminus'])) for r in rows);hi=min(float(r['cplus']) for r in rows);charrows.append(dict(run=str(path),max_abs_incoming=lo,min_outgoing=hi,passed=lo<=1e-10 and hi>0))
 frozen=json.loads((E/'frozen-baseline-sha256.json').read_text());mismatch=[p for p,v in frozen.items() if not Path(p).is_file() or sha(p)!=v];geometry=json.loads((E/'geometry-v2/review.json').read_text())
 backend=json.loads((E/'cpu-v3/backend-control.json').read_text())
 passed=complete and timecomplete and outer_complete and all(c['passed'] for c in checks+timechecks+charrows) and reduction['passed'] and all(c['passed'] for c in outer if c['n']==24) and geometry['passed'] and not mismatch and backend['passed']
 result=dict(stage='minkowski',passed=passed,complete=complete and timecomplete and outer_complete,qualification='Three-grid analytic field and derivative validation; finite-boundary sensitivity admission at finest n24 only. Failed medium boundary controls remain failed.',convergence=checks,timestep=timechecks,reduction=reduction,boundary=outer,characteristics=charrows,geometry=geometry,baseline_files=len(frozen),baseline_mismatches=mismatch,backend_control=backend,original_medium_boundary_passed=all(c['passed'] for c in outer if c['n']==18),source_sha256=sha(__file__))
 (E/'minkowski-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 (E/'cpu-admission.json').write_text(json.dumps({k:result[k] for k in ['stage','passed','complete','qualification','original_medium_boundary_passed']},indent=2)+'\n')
 fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for fld in ['psi','v','Dpsi']:
  check=next(c for c in checks if c['time']==6 and c['region']=='all' and c['field']==fld and c['norm']=='L2');axs[0].loglog([12,18,24],check['errors'],'o-',label=fld)
 axs[0].set(xlabel='Resolution N',ylabel='Volume L2 error at t=6',title='Angular Minkowski wave');axs[0].legend();axs[0].grid(alpha=.25)
 for path,label in [(paths[0],'N12 scri'),(paths[1],'N18 scri'),(paths[2],'N24 scri'),(E/'cpu-v4/minkowski-n24-cap0.9','N24 cap .9')]:
  if not (path/'waveform.csv').exists():continue
  q=np.genfromtxt(path/'waveform.csv',delimiter=',',names=True);axs[1].plot(q['time'],q['imag22'],label=label)
 axs[1].set(xlabel='Hyperboloidal time',ylabel='Im psi_22 at rho=10',title='Common extraction');axs[1].legend(fontsize=8);axs[1].grid(alpha=.25)
 for i,path in enumerate(paths):
  items=[(k[0],float(v['reduction_Linf'])) for k,v in h[i].items() if k[1]=='all'];axs[2].semilogy(*np.array(items).T,label=f'N{[12,18,24][i]}')
 axs[2].set(xlabel='Hyperboloidal time',ylabel='Max |q-D psi|',title='Scalar reduction constraint');axs[2].legend();axs[2].grid(alpha=.25)
 fig.tight_layout();fig.savefig(E/'minkowski-validation.png',dpi=160);fig.savefig(E/'minkowski-validation.pdf');plt.close(fig)
 print(json.dumps(dict(passed=passed,complete=result['complete'],min_convergence_order=min(min(c['orders']) for c in checks),max_timestep_fraction=max((c['fraction'] for c in timechecks),default=None),reduction=reduction,boundary=outer,baseline_mismatches=mismatch),indent=2))
if __name__=='__main__':main()
