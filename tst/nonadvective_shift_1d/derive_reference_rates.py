"""Read-only field-rate diagnostics on the exactly recovered reference replay."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);a=p.parse_args();m=json.loads((a.replay/'manifest.json').read_text());assert m['status']=='completed' and m['exact_checkpoint_recovery']['state']
sys.path.insert(0,str(a.replay.resolve()/'source/tst/hyperbolic_shift_1d'))
from puncture import PunctureEvolution
import inspect
ref=json.loads((Path(m['reference'])/'manifest.json').read_text());args=ref['arguments'];kwargs={k:v for k,v in args.items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
e=PunctureEvolution(args['n'],a.replay.resolve()/'source'/Path(args['library']).name,**kwargs);h=json.loads((a.replay/'history.json').read_text());records=[]
with np.load(a.replay/'profiles.npz') as z:
 for t,q in zip(z['t'],z['q']):
  row=next(r for r in h if abs(r['t']-t)<1e-13);f=e.rhs(q);x=e.last;physical=f[:,:9].copy();physical[:,0]*=x['chi'];w=e.chart.refs['w'][:,0]
  for field in [2,3,4]:physical[:,field]*=w
  weights=e.r**2*e.jac;weights[[0,-1]]*=.5;ext=e.r>row['apparent_horizon_r']
  records.append(dict(t=float(t),exterior_field_rate_rms=float(np.sqrt(np.sum(weights[ext,None]*physical[ext]**2)/sum(weights[ext]))),exterior_beta_rate_rms=float(np.sqrt(np.sum(weights[ext]*f[ext,7]**2)/sum(weights[ext])))))
(a.replay/'diagnostic-rates.json').write_text(json.dumps(dict(replay_manifest_sha256=hashlib.sha256((a.replay/'manifest.json').read_bytes()).hexdigest(),rows=records,scope='Additional read-only RHS diagnostics; original completed replay history and checkpoint hashes unchanged.'),indent=2)+'\n')
print('Computed',len(records),'reference rate samples without evolving.')
