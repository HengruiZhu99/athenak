from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).parent
load=lambda name:json.loads((p/name).read_text())
a=load('balanced-N64-I32-tangent8.json')
b=load('balanced-N64-I32-tangent32.json')
c=load('protected-N64-I32-tangent8.json')
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained',facecolor='white')
for d,label,style in [(a,'Exterior BT, tangent π/8','o-'),(b,'Exterior BT, tangent π/32','s-'),(c,'Coupled metric, tangent π/8','^-')]:
    runs=[r for r in d['runs']if 'closed_growth'in r]
    axes[0].plot([r['order']for r in runs],[r['closed_growth']for r in runs],style,label=label,ms=4)
axes[0].axhline(0,color='black',lw=.7)
axes[0].set_yscale('symlog',linthresh=1e-10)
axes[0].set_ylim(-1e-4,1e-2)
axes[0].set(xlabel='Auxiliary exterior states (full: 640)',ylabel='Largest closed growth rate [1/M]')
axes[0].legend(fontsize=7)
for d,r,label in [(a,256,'Exterior BT 256'),(a,384,'Exterior BT 384'),(a,448,'Exterior BT 448'),(c,448,'Coupled metric 448')]:
    rec=next(x for x in d['runs']if x['order']==r)['records'][1:]
    axes[1].semilogy([x['time']for x in rec],[x['inside_relative_error']if 'inside_relative_error'in x else x['interior_relative_error']for x in rec],'o-',label=label,ms=4)
axes[1].axhline(.01,color='gray',ls='--',lw=.7)
axes[1].set(xlabel='Time [M]',ylabel='Signed interior state relative error')
axes[1].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Finite periodic memory fixture · 64 cells / 32 retained · RK3 dt=0.6M',fontsize=10)
fig.savefig(p/'memory-compression.png',dpi=180)
fig.savefig(p/'memory-compression.pdf')
