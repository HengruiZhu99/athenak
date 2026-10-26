#!/usr/bin/env python3
"""Plot a preserved primary review without adding numerical trajectories."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scalar_evidence import history,waveform,snapshot

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.review.read_text());ns=r['resolutions'];runs=[Path(d) for d in r['runs']]
if any(a.output.with_suffix(ext).exists() for ext in ['.png','.pdf','.json']):raise RuntimeError('Existing scientific plot is immutable')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);colors=['#d55e00','#0072b2','#009e73'];ax=axes[0,0];kerr=r['stage']=='kerr'
for n,d,c in zip(ns,runs,colors):
 w=waveform(d);ax.plot(w['time'],w['imag22'],color=c,label=f'N{n}',lw=1)
ref=np.genfromtxt(r['oracle']['waveform'],delimiter=',',names=True);ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24
ax.plot(ref['time'],ref['imag22'] if kerr else factor*ref['radial_psi'],'k--',lw=1,label='Independent 2+1' if kerr else 'Independent radial');ax.set(xlabel='t/M',ylabel='Im scalar (2,2)',title=r['stage'].capitalize()+' trumpet waveform');ax.legend(fontsize=8)
ax=axes[0,1];late=next(q for q in r['records'] if q['time']==100)
for field,style in [('psi','o-'),('v','s-'),('Dpsi','^-'),('reduction','d-')]:
 values=[e[field]['Linf'] for e in late['errors']];ax.loglog(ns,values,style,label=field)
ax.set(xlabel='Resolution N',ylabel='Exterior maximum error',title='Errors at t = 100M');ax.legend(fontsize=8)
ax=axes[1,0]
checks=[q for q in r['field_checks'] if q['time']==100];x=np.arange(len(checks));width=.35
for k,label in enumerate([f'N{ns[0]}/N{ns[1]}',f'N{ns[1]}/N{ns[2]}']):ax.bar(x+(k-.5)*width,[q['orders'][k] for q in checks],width,label=label)
ax.axhline(1.5,color='black',ls='--',lw=1,label='Required 1.5');ax.set_xticks(x,[q['field']+' '+q['norm'] for q in checks],rotation=35,ha='right',fontsize=8);ax.set(ylabel='Measured pair order',title='Late field and derivative convergence');ax.legend(fontsize=8)
ax=axes[1,1]
for n,d,c in zip(ns,runs,colors):
 initial=snapshot(d,0);mask=np.linalg.norm(initial[:,:3],axis=1)>=1.2;scale=np.linalg.norm(initial[mask,6:9],axis=1).max();rows=sorted((t,float(q['reduction_Linf'])/scale) for (t,region),q in history(d).items() if region=='exterior');ax.semilogy(*np.array(rows).T,color=c,label=f'N{n}')
ax.axhline(.02,color='black',ls='--',lw=1,label='Finest required 0.02');ax.set(xlabel='t/M',ylabel='Reduction maximum / initial gradient peak',title='Every half-time reduction report');ax.legend(fontsize=8)
status='passes' if r['field_checks_passed'] and r['finest_accuracy_passed'] else 'fails';fig.suptitle(f'Preserved primary matrix {status}; full admission also requires controls',fontsize=13)
for ax in axes.flat:ax.grid(alpha=.2)
for ext in ['.png','.pdf']:fig.savefig(a.output.with_suffix(ext),dpi=170)
a.output.with_suffix('.json').write_text(json.dumps(dict(review=str(a.review.resolve()),review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={ext:hashlib.sha256(a.output.with_suffix(ext).read_bytes()).hexdigest() for ext in ['.png','.pdf']},native_trajectories_added=0),indent=2)+'\n')
