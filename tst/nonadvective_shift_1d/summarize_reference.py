"""Read-only advective comparison; no evolution or change to the reference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hyperbolic_shift_1d'))
from continue_series import verify
from puncture import PunctureEvolution
from continue_puncture import digest
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=verify(a.case);h=json.loads((a.case/'history.json').read_text());t=np.array([r['t'] for r in h]);mass=np.array([r['apparent_horizon_mass'] for r in h]);radius=np.array([r['apparent_horizon_r'] for r in h]);late=3*t>=80
coef=np.polyfit(3*t[late],radius[late],1);res=radius[late]-np.polyval(coef,3*t[late])
metrics={}
for field in ['H','M','Theta','Z']:
 v=np.array([r[field+'_exterior_rms'] for r in h]);metrics[field]=dict(peak=float(max(v)),time_rms=float(np.sqrt(np.trapezoid(v*v,t)/(t[-1]-t[0]))),final=float(v[-1]))
# Only import code that matches the frozen implementation used by this case.
for name in ['puncture.py','evolution.py','gamma_driver.py','constraint_damping.py']:
 rel=Path('tst/hyperbolic_shift_1d')/name;assert digest(rel)==m['sources'][str(rel)]
lib=a.case/'source'/Path(m['arguments']['library']).name
import inspect
kwargs={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
e=PunctureEvolution(m['arguments']['n'],lib,**kwargs)
with np.load(a.case/'profiles.npz') as z:q=z['last'].copy()
f=e.rhs(q);x=e.last;physical=f[:,:9].copy();physical[:,0]*=x['chi'];w=e.chart.refs['w'][:,0]
for k in [2,3,4]:physical[:,k]*=w
weights=e.r**2*e.jac;weights[[0,-1]]*=.5;ext=e.r>h[-1]['apparent_horizon_r']
result=dict(case=str(a.case),manifest_sha256=digest(a.case/'manifest.json'),n=m['arguments']['n'],dt=m['dt'],t_over_M=float(3*t[-1]),constraints=metrics,final_relative_mass_error=float(3*mass[-1]-1),peak_mass_drift=float(max(abs(mass/mass[0]-1))),initial_horizon_r=float(radius[0]),final_horizon_r=float(radius[-1]),late_horizon_slope_per_M=float(coef[0]),late_horizon_fit_RMS=float(np.sqrt(np.mean(res**2))),late_interval_M=[float(min(3*t[late])),float(max(3*t[late]))],final_exterior_field_rate_RMS=float(np.sqrt(np.sum(weights[ext,None]*physical[ext]**2)/sum(weights[ext]))),final_exterior_beta_rate_RMS=float(np.sqrt(np.sum(weights[ext]*f[ext,7]**2)/sum(weights[ext]))),stage_stats=m['stage_stats'],scope='Verified coarse advective reference only; no convergence or stationary-coordinate claim.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
