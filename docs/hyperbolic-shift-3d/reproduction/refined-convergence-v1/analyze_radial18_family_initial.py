"""Existing admitted local t0 controls plus immutable partial GPU history snapshot."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
D=Path(__file__).resolve().parent;O=D/'radial18-family-initial-and-live';O.mkdir(exist_ok=True)
inputs={12:D/'radial18-resolution-check/n12/state.history.jsonl',14:D/'radial18-resolution-check/n14/state.history.jsonl',16:D/'radial-control-segment01/output/state.history.jsonl'}
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def fixed(r):
 q=[x for x in r['fixed'] if x['cut']==.3];assert len(q)==1;return q[0]
values={n:fixed(rows(p)[0]) for n,p in inputs.items()};ns=np.array([12.,14.,16.]);fit={};fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for key in ['H','M','Mr']:
 y=np.array([values[int(n)][key] for n in ns]);slope=-float(np.polyfit(np.log(ns),np.log(y),1)[0]);fit[key]=dict(values=y.tolist(),fitted_order=slope)
 axes[0].loglog(ns,y,'o-',label=f'{key}: p={slope:.3f}')
axes[0].set_xticks(ns,[str(int(n)) for n in ns]);axes[0].xaxis.set_minor_locator(NullLocator());axes[0].set_xlabel('N');axes[0].set_ylabel('Initial residual norm, r > 0.3');axes[0].legend();axes[0].grid(alpha=.25,which='both');axes[0].set_title('Initial-data residual trend')
live=rows(O/'n12-live-history.jsonl');fine=rows(inputs[16]);pairs=[]
for a in live:
 b=min(fine,key=lambda b:abs(b['time']-a['time']))
 if abs(a['time']-b['time'])>1e-12:continue
 q=dict(time_M=3*a['time'],time_mismatch_M=3*abs(a['time']-b['time']))
 for key in ['H','M','Mr']:q[key]=dict(N12=fixed(a)[key],N16=fixed(b)[key],order=float(np.log(fixed(a)[key]/fixed(b)[key])/np.log(16/12)))
 pairs.append(q)
for key in ['H','M']:
 axes[1].plot([q['time_M'] for q in pairs],[q[key]['order'] for q in pairs],'o-',label=key)
axes[1].axhline(4,color='gray',ls=':');axes[1].set_xlabel('t/M');axes[1].set_ylabel('Two-point residual order');axes[1].set_title('Early evolution: N12 live snapshot vs N16');axes[1].legend();axes[1].grid(alpha=.25)
fig.suptitle('Stationary radial18 family; fixed coordinate cutoff, not field self-convergence');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/('initial-and-early-orders.'+ext),dpi=170)
result=dict(initial=fit,initial_input_sha256={str(n):hashlib.sha256(p.read_bytes()).hexdigest() for n,p in inputs.items()},live_snapshot=json.loads((O/'snapshot.json').read_text()),matched_rows=pairs,qualification='N12/N14 local SERIAL production t0; N16 frozen GPU t0 and admitted .6M control. N12 live running snapshot is unadmitted as a terminal state. Exact common recorded times, no interpolation, same fixed coordinate cutoff r>.3. Space/time refined together. Constraint-residual order, not field self-convergence or long-time proof.')
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(initial=fit,last_matched=pairs[-1]),indent=2))
