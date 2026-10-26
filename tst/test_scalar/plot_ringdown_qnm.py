#!/usr/bin/env python3
"""Plot existing free-frequency scalar fits and their independent QNM gates."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scalar_evidence import waveform

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.review.read_text());kerr=r['stage']=='kerr';ns=r['resolutions'];colors=['#d55e00','#0072b2','#009e73']
if any(a.output.with_suffix(ext).exists() for ext in ['.png','.pdf','.json']):raise RuntimeError('Existing QNM plot is immutable')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for n,record,color in zip(ns,r['QNM'],colors):
 t=[np.mean(f['window']) for f in record['fits']]
 for panel,field,bound in [(0,'real_error',.02),(1,'damping_error',.05)]:
  values=[f['relative_errors'][panel] if kerr else f[field] for f in record['fits']];axes[0,panel].plot(t,np.array(values)/bound,'o-',color=color,label=f'N{n} co' if kerr else f'N{n}')
  if kerr:axes[0,panel].plot(t,[f['relative_errors'][panel+2]/bound for f in record['fits']],'s--',mfc='white',color=color,label=f'N{n} counter')
 for key,value in record['window_spreads'].items():axes[1,0].scatter(n,value/.02,color=color,marker='s' if 'decay' in key else 'o',facecolors='none' if 'counterrotating' in key else color)
for panel,label in [(0,'Real-frequency error / 2% limit'),(1,'Decay error / 5% limit')]:
 ax=axes[0,panel];ax.axhline(1,color='black',ls='--',lw=1);ax.set(xlabel='Fit-window midpoint t/M',ylabel=label);ax.legend(fontsize=8)
ax=axes[1,0];ax.axhline(1,color='black',ls='--',lw=1);ax.set(xlabel='Resolution N',ylabel='Window spread / 2% limit',title='Circles: real frequency; squares: decay');ax.set_xticks(ns)
finest=r['QNM'][-1];f=next(f for f in finest['fits'] if f['window']==[40,70]);w=waveform(Path(r['runs'][-1]));select=(w['time']>=40)&(w['time']<=70);t=w['time'][select];tau=t-40;amp=np.array([complex(*v) for v in f['amplitudes']])
if kerr:
 basis=np.c_[np.exp((-f['decay']-1j*f['omega_real'])*tau),np.exp((-f['counterrotating_decay']+1j*f['counterrotating_real'])*tau)];z=w['real22'][select]+1j*w['imag22'][select]
else:
 basis=np.c_[np.exp(-f['decay']*tau)*np.cos(f['omega_real']*tau),np.exp(-f['decay']*tau)*np.sin(f['omega_real']*tau)];z=w['imag22'][select].astype(complex)
predicted=basis@amp;ax=axes[1,1];ax.plot(t,z.real,'o',ms=3,color='#0072b2',label='Field real' if kerr else 'Im (2,2) field');ax.plot(t,predicted.real,'-',color='#0072b2',label='Free fit real' if kerr else 'Free fit')
if kerr:ax.plot(t,z.imag,'s',ms=3,color='#d55e00',label='Field imaginary');ax.plot(t,predicted.imag,'-',color='#d55e00',label='Free fit imaginary')
ax.set(xlabel='t/M',ylabel='Scalar (2,2)',title=f'Finest N{ns[-1]}: free fit on 40–70M');ax.legend(fontsize=8)
fig.suptitle(r['stage'].capitalize()+' scalar: frequencies and amplitudes free; independent Leaver targets used for evaluation',fontsize=12)
for ax in axes.flat:ax.grid(alpha=.2)
for ext in ['.png','.pdf']:fig.savefig(a.output.with_suffix(ext),dpi=170)
a.output.with_suffix('.json').write_text(json.dumps(dict(review=str(a.review.resolve()),review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),finest_waveform_samples_sha256=hashlib.sha256(w.tobytes()).hexdigest(),files={ext:hashlib.sha256(a.output.with_suffix(ext).read_bytes()).hexdigest() for ext in ['.png','.pdf']},additional_trajectories=0,additional_fits=0),indent=2)+'\n')
