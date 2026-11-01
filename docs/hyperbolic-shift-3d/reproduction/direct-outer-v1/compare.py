"""Plot measured DO1 histories after verification; preserve unavailable diagnostics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--n16',type=Path,nargs='+',required=True);p.add_argument('--n24',type=Path,nargs='+');a=p.parse_args()
baseline=HERE.parent/'inner-telegrapher-v1/n16-completed/puncture.history.csv'
series=[('DO1 N16',a.n16,'#b55315'),('Previous IT1 N16',[baseline],'#777777')]
if a.n24: series.insert(1,('DO1 N24',a.n24,'#1769aa'))
stem='comparison' if a.n24 else 'comparison-n16'
fig,axes=plt.subplots(2,2,figsize=(11,7))
fields=[('H_exterior_L2','Exterior Hamiltonian RMS',1),('Mr_exterior_L2','Exterior radial-momentum RMS',1),('horizon_areal_radius','Accepted horizon areal radius / M',3),('null_surface','Running maximum scri null residual',1)]
sources={}; endpoints={}
for label,paths,color in series:
 records=[]
 for path in paths:
  sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
  records.append(np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True)))
 d=np.concatenate(records);d=d[np.argsort(d['t_over_M'],kind='stable')];_,i=np.unique(d['t_over_M'],return_index=True);d=d[i];d=d[d['t_over_M']<=10+1e-8]
 endpoints[label]=float(d['t_over_M'][-1])
 for ax,(key,title,scale) in zip(axes.flat,fields):
  ax.plot(d['t_over_M'],scale*d[key],'-o',ms=2,label=label,color=color)
  ax.set(xlabel='t / M',ylabel=title,xlim=(0,10));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Direct telegrapher-to-outer transition: fresh evolution to 10M')
fig.tight_layout(rect=(0,.08,1,.95))
fig.text(.5,.025,'Measured samples; missing diagnostics remain gaps. Exterior cutoffs use each run’s complete horizon candidate.\nLast diagnostic can precede last accepted step. Completion alone does not establish convergence.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(HERE/f'{stem}.{ext}',dpi=170)
(HERE/f'{stem}-sources.json').write_text(json.dumps(dict(sha256=sources,last_diagnostic_t_over_M=endpoints),indent=2)+'\n')
