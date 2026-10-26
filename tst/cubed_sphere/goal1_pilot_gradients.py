#!/usr/bin/env python3
"""All-cell first-derivative temporal differences from measured CK02 endpoints.

The native measure() fills ghosts before checkpoint output. Use those actual
ghosts and the same fourth-order centered D1, including boundary/seam cells.
This is a temporal sensitivity norm, not an independent solution error.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from compare_closure_field_norms import read
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
first,second=read(a.first),read(a.second);s=first['signature'];assert first['time']==second['time'];assert first['tag']==second['tag'] and first['fingerprint']==second['fingerprint']
assert all(x==y for i,(x,y) in enumerate(zip(s,second['signature'])) if i!=15)
assert s[15]/2==second['signature'][15]
records=[]
for region,(x,y) in enumerate(zip(first['packs'],second['packs'])):
    z=np.asarray(x)-np.asarray(y);active=z[:,:,4:-4,4:-4,4:-4];m,fields,nk,nj,ni=active.shape
    spacings=[1/ni,2/nj,2/nk] if region else [2*s[0]/(ni*m**(1/3))]*3
    if not region:assert m==1,'Pilot deliberately uses a single bulk block; do not infer other bulk layouts.'
    logical=[]
    for axis,h in zip([4,3,2],spacings):
        v=0.
        for off,c in [(-2,1/12),(-1,-2/3),(1,2/3),(2,-1/12)]:
            selection=[slice(None),slice(None),slice(4,-4),slice(4,-4),slice(4,-4)]
            length=z.shape[axis];selection[axis]=slice(4+off,length-4+off)
            v=v+c*z[tuple(selection)]/h
        logical.append(v)
    masks={'all':np.ones((m,nk,nj,ni),dtype=bool)}
    if region:
        vv,uu,ss=np.meshgrid(-1+2*(np.arange(nk)+.5)/nk,-1+2*(np.arange(nj)+.5)/nj,(np.arange(ni)+.5)/ni,indexing='ij')
        L,R,k=s[:3];norm=np.sqrt(1+uu**2+vv**2);t=ss if k==0 else -np.expm1(-k*ss)/(-np.expm1(-k));tp=np.ones_like(ss) if k==0 else k*np.exp(-k*ss)/(-np.expm1(-k));d=(1-t)*L+t*R/norm
        ds=tp*(R/norm-L);du=-t*R*uu/norm**3;dv=-t*R*vv/norm**3
        radius=d*norm;weight=radius**2*(R-L*norm)*tp/norm**3/ni*(2/nj)*(2/nk)
        weight=np.broadcast_to(weight,(m,nk,nj,ni));gradient=np.zeros((m,fields,3,nk,nj,ni))
        for patch in range(6):
            F=np.zeros((nk,nj,ni,3,3));axis=patch//2;b=(axis+1)%3;c=(axis+2)%3;sign=1 if patch%2==0 else -1
            for A,dA in enumerate([ds,du,dv]):
                F[...,axis,A]=sign*dA;F[...,b,A]=uu*dA;F[...,c,A]=vv*dA
            F[...,b,1]+=d;F[...,c,2]+=d;J=np.linalg.inv(F)
            for A in range(3):
                for i in range(3):gradient[patch,:,i]+=logical[A][patch]*J[...,A,i]
        masks.update(transition=np.broadcast_to((radius>=s[4])&(radius<s[5]),masks['all'].shape),outer_collar=np.broadcast_to(radius>.9,masks['all'].shape))
        seam=(np.abs(uu)>.8).astype(int)+(np.abs(vv)>.8).astype(int)
        for label,n in [('face',0),('edge',1),('corner',2)]:masks[label]=np.broadcast_to(seam==n,masks['all'].shape)
    else:
        gradient=np.stack(logical,axis=2);weight=np.full(masks['all'].shape,spacings[0]**3)
    for label,mask in masks.items():
        assert mask.any();vol=float(np.sum(weight[mask]));components=[]
        for field in range(fields):
            squared=np.sum(gradient[:,field]**2,axis=1)
            components.append({'component':field,'l2':float(np.sqrt(np.sum(weight[mask]*squared[mask])/vol)),
                'linf':float(np.sqrt(np.max(squared[mask])))})
        records.append({'region':('shell/' if region else 'bulk/')+label,'volume':vol,'cells':int(mask.sum()),'components':components})
report={'scope':__doc__,'time':first['time'],'input_payloads':[first['payload'],second['payload']],
    'all_active_cells_included':True,'gradient_contraction':'Euclidean Cartesian sum of three squared derivatives; compact coordinate-volume RMS; L0=1',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,
    'maximum_regional_component_l2':max(v['l2'] for r in records for v in r['components']),
    'maximum_regional_component_linf':max(v['linf'] for r in records for v in r['components'])}
a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:report[k] for k in ['maximum_regional_component_l2','maximum_regional_component_linf']})
