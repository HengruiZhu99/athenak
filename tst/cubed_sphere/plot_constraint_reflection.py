#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('output',type=Path);a=p.parse_args();j=json.loads(a.review.read_text());fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for index,r in enumerate(j['records']):
 color='tab:blue' if index<5 else 'tab:orange';label=('pulse speed samples' if index==0 else 'puncture speed samples') if index in [0,5] else None;t=r['tests'][1:];angles=[x['angle_degrees'] for x in t]
 for ax,key in zip(axes,['amplitude_reflection','normal_wavenumber_stretch','normal_derivative_amplitude_ratio']):ax.semilogy(angles,[x[key] for x in t],'o-',alpha=.45,color=color,label=label);ax.set_xlabel('Incidence angle (degrees)');ax.grid(alpha=.2)
for ax,title in zip(axes,['Reflected field amplitude','Reflected normal wavenumber stretch','Normal-derivative amplitude ratio']):ax.set_title(title,fontsize=10)
axes[0].legend(fontsize=8);fig.suptitle('Oblique frozen subsidiary-wave model: first-order Sommerfeld\nNative normal speeds; no actual native reflection or instability claim.',fontsize=11);fig.savefig(a.output.with_suffix('.png'),dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
