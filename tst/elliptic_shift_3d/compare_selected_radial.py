#!/usr/bin/env python3
"""All-cell and raw-angle comparison for the corrected selected taper pilot."""
import argparse,json,hashlib,os,tempfile
from pathlib import Path
import numpy as np
from radial_reference import interpolate
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
d=np.genfromtxt(root/'log-n8.csv',delimiter=',',names=True)
r=np.load(root/'radial-n2049.npz');coarse=np.load(root/'radial-n1025.npz');w=d['weight']
fields=['alpha','chi','g_rr','A_rr','Khat','Theta','Lambda_r','beta_r']
out=dict(scope='Selected .24 taper, full3D research pilot t=.0002; failed, no collapse reproduction',field_errors={},regions={},reference_resolution_control={},angular={})
def norm(e,weight):return dict(physical_rms=float(np.sqrt(np.average(e*e,weights=weight))),maximum=float(max(abs(e))))
for f in fields:
 truth=interpolate(r['r'],r[f],d['r'])
 out['field_errors'][f]=norm(d[f]-truth,w)
 out['reference_resolution_control'][f]=norm(interpolate(coarse['r'],coarse[f],d['r'])-truth,w)
for name,m in [('all',d['r']>=0),('core',d['r']<.1),('bulk',(d['r']>=.1)&(d['r']<.3)),('transition',(d['r']>=.3)&(d['r']<.8)),('outer',d['r']>=.8)]:
 out['regions'][name]=dict(cells=int(sum(m)),H=norm(d['H'][m],w[m]),M=norm(d['M_norm'][m],w[m]))
s=np.genfromtxt(root/'log-n8.csv.spheres.csv',delimiter=',',names=True)
for f in fields:
 q=s[f].reshape(-1,64);rs=np.unique(s['r'])
 out['angular'][f]=dict(radii=rs.tolist(),mean=q.mean(axis=1).tolist(),scatter=q.std(axis=1).tolist(),mean_error=(q.mean(axis=1)-interpolate(r['r'],r[f],rs)).tolist())
out['input_hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'log-n8.csv',root/'log-n8.csv.spheres.csv',root/'radial-n1025.npz',root/'radial-n2049.npz']}
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
with tempfile.TemporaryDirectory(prefix='shift-selected-plot-') as config:
 os.environ.setdefault('MPLCONFIGDIR',config)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for ax,f in zip(axes.flat,['alpha','chi','A_rr','beta_r']):
  q=out['angular'][f];rs=np.array(q['radii']);mean=np.array(q['mean']);sd=np.array(q['scatter'])
  ax.plot(r['r'],r[f],color='k',label='Independent spherical N2049')
  ax.plot(rs,mean,'o-',ms=3,label='3D N8 angular mean')
  ax.fill_between(rs,mean-sd,mean+sd,alpha=.25,label='Raw angular scatter')
  ax.set(xlabel='compact radius',ylabel=f);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Selected taper .24, t=0.0002: coarse 3D pilot fails scientific gates')
 fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps(out['field_errors']))
