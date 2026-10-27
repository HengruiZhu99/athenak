"""Plot archived duration histories and the diagnosed coefficient-one crossing."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--cases',type=Path,nargs='+',required=True)
p.add_argument('--crossing',type=Path)
p.add_argument('--stem',default='gamma2-progress')
p.add_argument('--show-duration',action='store_true',help='Label each curve by its actual terminal time')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,ax=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
for case in a.cases:
 m=json.loads((case/'manifest.json').read_text());h=json.loads((case/'history.json').read_text())
 t=3*np.array([x['t'] for x in h]);n=m['arguments']['n'];mu=m['arguments']['gamma_coefficient']
 label=f'N={n}, bulk coefficient {mu:g}'
 if a.show_duration:label=f'N={n}, {3*m["t"]:.3g}M'
 line,=ax[0,0].plot(t,[x['shift_lapse_gap_times_g_min'] for x in h],label=label);color=line.get_color()
 ax[0,1].semilogy(t,[x['H_exterior_rms'] for x in h],color=color)
 ax[1,0].plot(t,1e6*(3*np.array([x['apparent_horizon_mass'] for x in h])-1),color=color)
 ax[1,1].plot(t,[x['g_min'] for x in h],color=color)
 if m['status']!='completed':
  for panel,key in [(ax[0,0],'shift_lapse_gap_times_g_min'),(ax[0,1],'H_exterior_rms'),(ax[1,1],'g_min')]:
   panel.plot(t[-1],h[-1][key],'x',color=color,ms=8)
for panel,title in zip(ax.flat,['Lapse/shift speed margin','Exterior Hamiltonian RMS','AH mass error [ppm]','Minimum radial metric']):
 panel.set(title=title,xlabel='Time / M');panel.grid(alpha=.2)
ax[0,0].axhline(0,color='black',lw=.8,ls=':');ax[0,0].set_ylabel(r'Minimum $g(c_S^2-c_L^2)$')
ax[0,0].legend(fontsize=10)
ax[1,0].set_yscale('symlog',linthresh=.01);ax[1,0].axhline(0,color='black',lw=.6)
for ext in ['pdf','png']:fig.savefig(a.output/f'{a.stem}.{ext}',dpi=170)

if a.crossing is None:
 raise SystemExit(0)

d=json.loads((a.crossing/'diagnosis.json').read_text());z=np.load(a.crossing/'profiles.npz')
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for key in z.files:
 i=int(key[1:]);t=d['rows'][i]['t'];q=z[key];r=q[:,0]
 line,=ax[0].plot(r[1:],q[1:,1],label=f't/M={3*t:.3g}')
 use=r>=.2;ax[1].semilogy(r[use],np.maximum(abs(q[use,3]),1e-15),color=line.get_color())
ax[0].axhline(0,color='black',ls=':',lw=.8)
ax[0].set(title='Coefficient-one speed coincidences',ylabel=r'$g(c_S^2-c_L^2)$',xlim=(.2,1))
ax[1].set(title='Outer-region Hamiltonian error',ylabel=r'$|H|$',xlim=(.2,1))
ax[0].legend(fontsize=10)
for panel in ax:panel.set_xlabel('Compact radius r');panel.grid(alpha=.2)
for ext in ['pdf','png']:fig.savefig(a.output/f'gamma-speed-crossing.{ext}',dpi=170)
