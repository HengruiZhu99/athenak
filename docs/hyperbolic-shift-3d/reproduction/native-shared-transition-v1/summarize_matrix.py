"""Summarize already verified histories and admitted native endpoint reconstructions."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HERE/'aurora/matrix-final');OUT=parser.parse_args().output
summary=json.loads((OUT/'comparison.json').read_text());radial=json.loads((HERE/'aurora/abcd-versus-sg1/radial-comparison.json').read_text());result={}
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors={'B':'#1769aa','E':'#cc3344'}
for name,c in summary['cases'].items():
 raw=ROOT/('output/hyperbolic-core-shell/short-gauge-v1/aurora' if name=='SG1' else 'output/native-shared-transition/aurora')
 if name=='SG1':plan='campaigns/n16-short-gauge-fresh10m-v1-segment01/plan.json'
 else:plan=c['segments'][0]['plan']
 p=json.loads((raw/plan).read_text());hist=raw/p['commands'][0]['args'][1];data=np.atleast_1d(np.genfromtxt(str(hist)+'.history.csv',delimiter=',',names=True));accepted=np.where(data['horizon_found']==1)[0];last=int(accepted[-1]) if len(accepted) else None
 result[name]={k:c[k] for k in ['status','last_accepted_t_over_M','last_diagnostic_t_over_M','accepted_horizon_rows','rows','final_accepted_mass_error','counters','boundary_report_maxima']}
 result[name]['last_accepted_mass']=None if last is None else dict(t_over_M=float(data['t_over_M'][last]),relative_error=float(3*data['horizon_irreducible_mass'][last]-1))
 result[name]['matched_10M_endpoint']=radial.get(name)
 if name in colors:
  x=data['t_over_M'];r=data['exterior_cutoff']
  axes[0].plot(x,r,color=colors[name],label=name+' candidate cutoff')
  axes[0].plot(x[data['horizon_found']==1],r[data['horizon_found']==1],'.',color=colors[name],label=name+' accepted horizon')
  axes[1].semilogy(x,data['M_L2'],color=colors[name],label=name+' full M RMS')
  axes[1].semilogy(x,data['M_exterior_L2'],'--',color=colors[name],label=name+' candidate-exterior M RMS')
axes[0].axhline(.609375,color='black',ls=':',label='Rejected-stage failure radius')
axes[0].set(xlabel='t / M',ylabel='Compactified radius',title='Candidate cutoffs are not always accepted horizons')
axes[1].set(xlabel='t / M',ylabel='Momentum RMS',title='Candidate exclusion can hide the growing shell error')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Shared restoration: B and E both fail near9.39M; E also records a storage guard')
fig.savefig(OUT/'candidate-cutoff.png',dpi=170)
(OUT/'physics-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for k,v in result.items():print(k,v['last_accepted_mass'])
