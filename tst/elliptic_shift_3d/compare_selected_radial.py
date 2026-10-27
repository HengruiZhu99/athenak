#!/usr/bin/env python3
"""All-cell and raw-angle comparison for the corrected selected taper pilot."""
import argparse,json,hashlib
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
print(json.dumps(out['field_errors']))
