#!/usr/bin/env python3
"""Plot independent sampling controls and preserved primary convergence failures."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();E=a.evidence;diagnostic=E/'reference-interpolation-diagnostic-v1.json';resolution=E/'reference-resolution-at-primary-points-v1.json';d=json.loads(diagnostic.read_text());r=json.loads(resolution.read_text());ns=[36,48,60]
if any(a.output.with_suffix(ext).exists() for ext in ['.png','.pdf','.json']):raise RuntimeError('Existing reference-audit plot is immutable')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);ax=axes[0,0];instrument=d['instrument'][1];n=[row['n'] for row in instrument['errors']];query=np.linspace(1.2,24,24001);function=lambda x:np.exp(-((x-6.137)/1.2)**2)*np.cos(.671*x);pchip=[]
for count in n:
 x=np.linspace(.04,24,count);pchip.append(np.max(abs(PchipInterpolator(x,function(x))(query)-function(query))))
ax.loglog(n,pchip,'o-',label='PCHIP');ax.loglog(n,[row['Linf'] for row in instrument['errors']],'s-',label='Six-point Lagrange');ax.set(xlabel='Independent profile nodes',ylabel='Maximum interpolation error',title='Independent smooth analytic instrument');ax.legend(fontsize=8)
ax=axes[0,1]
for field,color in [('psi','#0072b2'),('v','#d55e00')]:
 for method,marker in [('PCHIP','o--'),('Lagrange6','s-')]:
  record=next(c for c in d['checks'] if c['time']==2 and c['field']==field and c['norm']=='Linf' and c['method']==method);ax.loglog(ns,record['errors'],marker,color=color,label=field+' '+method)
ax.set(xlabel='Primary resolution N',ylabel='Maximum field error',title='Early t2 peak-sampling floor');ax.legend(fontsize=8)
ax=axes[1,0]
for method,marker in [('PCHIP','o--'),('Lagrange6','s-')]:
 record=next(c for c in d['checks'] if c['time']==100 and c['field']=='psi' and c['norm']=='L2' and c['method']==method);ax.loglog(ns,record['errors'],marker,label='1200 '+method)
record=next(c for c in r['checks'] if c['time']==100 and c['field']=='psi' and c['norm']=='L2');ax.loglog(ns,record['errors'],'^:',label='2400 Lagrange6');ax.set(xlabel='Primary resolution N',ylabel='Exterior field volume L2 error',title=f'Late order remains {record["orders"][1]:.3f} < 1.5');ax.legend(fontsize=8)
ax=axes[1,1];fields=['psi','v','Dpsi'];fraction=[max(c['fraction'] for c in r['uncertainty'] if c['field']==field) for field in fields];ax.bar(fields,fraction,color=['#0072b2','#d55e00','#009e73']);ax.axhline(.1,color='black',ls='--',lw=1,label='Predeclared 10% bound');ax.set(ylabel='Reference difference / finest primary error',title='Both norms and all four times');ax.legend(fontsize=8)
fig.suptitle('Sampling fixes early comparisons; the remaining late convergence gate still fails',fontsize=13)
for ax in axes.flat:ax.grid(alpha=.2)
for ext in ['.png','.pdf']:fig.savefig(a.output.with_suffix(ext),dpi=170)
a.output.with_suffix('.json').write_text(json.dumps(dict(input_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in [diagnostic,resolution]},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={ext:hashlib.sha256(a.output.with_suffix(ext).read_bytes()).hexdigest() for ext in ['.png','.pdf']},new_trajectories=0),indent=2)+'\n')
