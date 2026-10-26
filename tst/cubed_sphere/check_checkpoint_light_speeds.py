#!/usr/bin/env python3
"""All-active-shell physical radial speeds, with selected live symbol samples."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('receipt',type=Path);p.add_argument('output',type=Path);p.add_argument('--logical-normal',action='store_true');a=p.parse_args()
receipt=json.loads(a.receipt.read_text());payload=receipt['checkpoint_payload'];digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
assert digest==payload['sha256'] and a.checkpoint.stat().st_size==payload['bytes']
with a.checkpoint.open('rb') as f:
 assert f.read(16) in [b'ATHENA_CSP_CK01!',b'ATHENA_CSP_CK02!']
 tag=f.read(64).decode();sig=struct.unpack('<17d',f.read(136));fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24));packs=[]
 for _ in range(2):
  shape=struct.unpack('<5Q',f.read(40));packs.append(np.memmap(a.checkpoint,mode='r',dtype='<f8',offset=f.tell(),shape=shape));f.seek(int(np.prod(shape))*8,1)
 assert f.tell()+8==a.checkpoint.stat().st_size
L,R,stretch,curvature,r0,r1,mass=sig[:7];assert .9<R<=1 and sig[-1]==8
domain=a.receipt.with_name('cubed_sphere_conformal_domain.json')
if R!=1 or domain.exists():
 descriptor=json.loads(domain.read_text())
 assert descriptor['domain_radius']==R and descriptor['reference_scri_radius']==1
u=np.array(packs[1][:,:,4:-4,4:-4,4:-4]);na,nr=u.shape[2],u.shape[4]
v,w,s=np.meshgrid(-1+2*(np.arange(na)+.5)/na,-1+2*(np.arange(na)+.5)/na,(np.arange(nr)+.5)/nr,indexing='ij')
t=s if stretch==0 else -np.expm1(-stretch*s)/(-np.expm1(-stretch))
radius=(1-t)*L*np.sqrt(1+w*w+v*v)+t*R
n=np.zeros((6,)+radius.shape+(3,))
for patch in range(6):
 aa=patch//2;bb=(aa+1)%3;cc=(aa+2)%3
 n[patch,...,aa]=1 if patch%2==0 else -1;n[patch,...,bb]=w;n[patch,...,cc]=v
n/=np.sqrt(1+w*w+v*v)[None,...,None]
radial_normal=n.copy()
if a.logical_normal:
 xyz=n*radius[None,...,None]
 for patch in range(6):
  axis=patch//2;sign=1 if patch%2==0 else -1;e=np.zeros(3);e[axis]=sign
  rr=radius;dd=sign*xyz[patch,...,axis];den=R*dd/rr-L
  dden=R*(e/rr[...,None]-dd[...,None]*xyz[patch]/rr[...,None]**3)
  gradient=(e*den[...,None]-(dd-L)[...,None]*dden)/den[...,None]**2
  n[patch]=gradient/np.linalg.norm(gradient,axis=-1)[...,None]
g=np.zeros(n.shape[:-1]+(3,3))
for field,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)],1):g[...,i,j]=g[...,j,i]=u[:,field]
assert np.isfinite(u).all() and np.min(np.linalg.eigvalsh(g))>0
inv=np.linalg.inv(g);alpha=u[:,18];chi=u[:,0];beta=np.moveaxis(u[:,19:22],1,-1)
assert np.min(alpha)>0 and np.min(chi)>0
bn=np.einsum('...i,...i->...',beta,n);light=alpha*np.sqrt(chi*np.einsum('...i,...ij,...j->...',n,inv,n))
minus=-bn-light;plus=-bn+light
mask=np.broadcast_to(radius>max(.9,r1),minus.shape)
selected=[];regions={}
for label,region in [('outer_collar',mask),('last_radial_cell',np.broadcast_to(np.arange(nr)==nr-1,minus.shape))]:
 locations={}
 for name,values in [('minus_min',minus),('minus_max',-minus),('plus_min',plus)]:
  loc=np.unravel_index(np.argmin(np.where(region,values,np.inf)),values.shape);locations[name]=list(map(int,loc));selected.append(loc)
 regions[label]=dict(cells=int(region.sum()),incoming_light_cells=int(np.count_nonzero(region&(minus<0))),both_outgoing_cells=int(np.count_nonzero(region&(minus>=0))),minus_range=[float(minus[region].min()),float(minus[region].max())],plus_range=[float(plus[region].min()),float(plus[region].max())],locations=locations)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(checkpoint_sha256=digest,binary_sha256=tag,time=time,reference_fingerprint=fingerprint,collar_radial_threshold=max(.9,r1),scope='All active cells beyond the stated collar radial threshold, physical light speeds only; '+('constant logical radial surface normal' if a.logical_normal else 'Cartesian radial normal')+'; no gauge/constraint classification. This threshold can differ from the native fixed r>0.9 constraint collar.',regions=regions),indent=2)+'\n')
with a.output.with_suffix('.logical-samples' if a.logical_normal else '.samples').open('w') as f:
 for loc in dict.fromkeys(selected):
  patch,k,j,i=loc;xyz=radial_normal[loc]*radius[k,j,i]
  values=[int(mass>0),curvature,r0,r1,*xyz,*u[patch,:,k,j,i]]
  if a.logical_normal:values+=list(n[loc])
  f.write(' '.join(format(float(x),'.17g') for x in values)+'\n')
print(json.dumps(regions))
