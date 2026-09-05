"""Plot paired changes of scalar maxima; these are not vector-correction norms."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
report=dict(scope=__doc__,runs=[])
operations={1:'Prolongation',2:'Algebraic enforcement',7:'RK update',8:'Physical ghosts'}
for row,(path,title) in enumerate([(a.baseline,'Saved factorized target'),(a.corrected,'Direct product gradient')]):
    data=pd.read_csv(path/'operation-norm-increases.csv.gz');data=data[data.region=='all']
    item=dict(run=str(path),quantities={})
    for col,q in enumerate(['RQ','curl_Q','curl_L']):
        d=data[data.quantity==q];ax=axes[row,col]
        item['quantities'][q]={}
        for operation,label in operations.items():
            s=d[d.operation==operation]
            if s.empty:continue
            # Preserve the entire winning paired event, including distinct locations.
            winners=s.loc[s.groupby('half_M_bin').delta_max_norm.idxmax()].sort_values('half_M_bin')
            ax.plot(winners.half_M_bin*.5,winners.delta_max_norm,'o-',markersize=3,label=label)
            item['quantities'][q][label]=winners.to_dict(orient='records')
        ax.set_yscale('symlog',linthresh=1e-10)
        # Sparse signed logarithmic ticks keep the wide dynamic range legible.
        lo,hi=ax.get_ylim()
        ticks=[sign*10.**exponent for sign in [-1,1] for exponent in range(-9,10,3)]
        ax.set_yticks(sorted(t for t in ticks+[0.] if lo<=t<=hi))
        ax.set(title=f'{title}: {q}',xlabel='Step-start half-M bin',ylabel='max(after) - max(before)')
        ax.grid(alpha=.25)
    report['runs'].append(item)
axes[0,0].legend(fontsize=8)
fig.suptitle('Largest paired scalar-norm increase per bin and operation; not a vector correction or causal proof')
fig.savefig(a.output/'operation-norm-increases.png',dpi=170)
(a.output/'operation-norm-increases.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
