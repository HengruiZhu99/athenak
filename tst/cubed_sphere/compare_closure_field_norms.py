#!/usr/bin/env python3
"""Within-grid closure differences, no cross-resolution interpolation.
Coordinate-volume component norms are diagnostics, not invariant field norms.
"""
import argparse,csv,hashlib,json,struct
from pathlib import Path
import numpy as np

def read(case):
 r=json.loads((case/'receipt.json').read_text());p=case/'probe_checkpoint.bin';payload=r['checkpoint_payload']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['sha256']
 with p.open('rb') as f:
  assert f.read(16) in [b'ATHENA_CSP_CK01!',b'ATHENA_CSP_CK02!'];tag=f.read(64).decode();sig=struct.unpack('<17d',f.read(136));fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24));packs=[]
  for _ in range(2):
   shape=struct.unpack('<5Q',f.read(40));packs.append(np.memmap(p,mode='r',dtype='<f8',offset=f.tell(),shape=shape));f.seek(int(np.prod(shape))*8,1)
  assert f.tell()+8==p.stat().st_size
 assert r['reference_unchanged'] and sig[-1]==8
 return dict(signature=sig,time=time,cycle=cycle,tag=tag,payload=payload,packs=packs,receipt=r,fingerprint=fingerprint)

def main():
 p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('output',type=Path)
 p.add_argument('--variation',choices=['closure','timestep','cfl','binary'],default='closure');a=p.parse_args()
 first=read(a.first);second=read(a.second);s=first['signature']
 assert first['time']==second['time']
 allowed={'closure':{12},'timestep':{15},'cfl':{13,14},'binary':set()}[a.variation]
 assert all(x==y for i,(x,y) in enumerate(zip(s,second['signature'])) if i not in allowed)
 if a.variation!='binary':
  assert first['fingerprint']==second['fingerprint'] and first['tag']==second['tag']
 if a.variation=='timestep':
  assert s[15]>0 and second['signature'][15]==s[15]/2
 elif a.variation=='cfl':
  assert all(s[i]>0 and second['signature'][i]==s[i]/2 for i in [13,14])
 else:
  # Adaptive CFL and restart milestone truncation can change cycle counts
  # even for identical closure-pair grids; physical endpoint time must match.
  if a.variation=='closure':assert s[12]!=second['signature'][12]
  else:assert first['tag']!=second['tag']
 L,R,stretch=s[:3];results={};total_volume=0.;global_squared=np.zeros(25)
 for region,(x,y) in enumerate(zip(first['packs'],second['packs'])):
  assert x.shape==y.shape;u=np.asarray(x[:,:,4:-4,4:-4,4:-4]);v=np.asarray(y[:,:,4:-4,4:-4,4:-4]);assert np.isfinite(u).all() and np.isfinite(v).all()
  na,nr=u.shape[2],u.shape[4];shape=u[:,0].shape
  masks={'all':np.ones(shape,dtype=bool)}
  if region:
   vv,uu,ss=np.meshgrid(-1+2*(np.arange(na)+.5)/na,-1+2*(np.arange(na)+.5)/na,(np.arange(nr)+.5)/nr,indexing='ij')
   norm=np.sqrt(1+uu*uu+vv*vv);t=ss if stretch==0 else -np.expm1(-stretch*ss)/(-np.expm1(-stretch));dt=1 if stretch==0 else stretch*np.exp(-stretch*ss)/(-np.expm1(-stretch))
   radius=(1-t)*L*norm+t*R
   weight=np.broadcast_to(radius**2*(R-L*norm)*dt/norm**3*(1/nr)*(2/na)**2,shape)
   masks.update(fixed_inner_shell=np.broadcast_to(radius<.5,shape),fixed_outer_shell=np.broadcast_to(radius>=.5,shape),fixed_outer_collar=np.broadcast_to(radius>.9,shape),fixed_transition=np.broadcast_to((radius>=s[4])&(radius<s[5]),shape))
   # Supplementary partitions retain the entire collar. Fixed logical bands
   # support refinement comparisons; two-cell bands diagnose stencil contact
   # but shrink physically with angular refinement and cannot replace them.
   for prefix,threshold in [('fixed_band',.8),('two_cell_band',1-4/na)]:
    seam_axes=(np.abs(uu)>threshold).astype(int)+(np.abs(vv)>threshold).astype(int)
    parts=[np.broadcast_to((radius>.9)&(seam_axes==number),shape) for number in range(3)]
    assert np.array_equal(sum(part.astype(int) for part in parts),masks['fixed_outer_collar'].astype(int))
    for label,part in zip(['face_interior','edge','corner'],parts):masks['collar_'+prefix+'_'+label]=part
  else:
   # Obtain block ordering/coordinates from the immutable reference record,
   # rather than assuming a Morton ordering or a single bulk block.
   ref=a.first/'cubed_sphere_reference_before.bin';refpayload=first['receipt']['reference_payloads'][ref.name]
   counts=[int(np.prod([z.shape[0],z.shape[2]-8,z.shape[3]-8,z.shape[4]-8])) for z in first['packs']]
   compact=a.first/'bulk_geometry.bin'
   if not ref.exists():
    meta=json.loads((a.first/'bulk_geometry.json').read_text())
    assert meta['format']=='bulk-index-xyz-le-v1'
    assert meta['reference_payload']==refpayload
    assert all(meta['checkpoint_payload'][key]==first['payload'][key] for key in ['sha256','bytes'])
    assert meta['count']==counts[0] and tuple(meta['shape'])==x.shape
    assert compact.stat().st_size==meta['compact_payload']['bytes']==40*counts[0]
    assert hashlib.sha256(compact.read_bytes()).hexdigest()==meta['compact_payload']['sha256']
    cellsize=40;count=counts[0];offset=0;cellpath=compact
   else:
    assert hashlib.sha256(ref.read_bytes()).hexdigest()==refpayload['sha256']
    with ref.open('rb') as f:
     magic=f.readline();assert magic==b'CUBED_SPHERE_REFERENCE_V1\n'
     residual=ref.stat().st_size-len(magic)-96-sum(z.size*8 for z in first['packs'])
     assert residual%sum(counts)==0;cellsize=residual//sum(counts)
     count=struct.unpack('<Q',f.read(8))[0];refshape=struct.unpack('<5Q',f.read(40));assert count==counts[0] and refshape==x.shape
     offset=f.tell();cellpath=ref
   dtype=np.dtype({'names':['indices','xyz'],'formats':[('<i4',(4,)),('<f8',(3,))],'offsets':[0,16],'itemsize':cellsize})
   cells=np.memmap(cellpath,mode='r',dtype=dtype,offset=offset,shape=(count,))
   extent=np.zeros(shape);radius=np.zeros(shape);seen=np.zeros(shape,dtype=bool)
   for cell in cells:
    m,k,j,i=map(int,cell['indices']);index=(m,k-4,j-4,i-4);assert not seen[index];seen[index]=True;extent[index]=max(abs(cell['xyz']));radius[index]=np.linalg.norm(cell['xyz'])
   assert seen.all()
   bulk_n=first['receipt']['bulk'];weight=np.full(shape,(2*L/bulk_n)**3);masks.update(fixed_interior=extent<.18,fixed_interface=extent>=.18,fixed_transition=(radius>=s[4])&(radius<s[5]))
  for label,mask in masks.items():
   vol=float(weight[mask].sum());assert vol>0
   name=('shell/' if region else 'bulk/')+label;norms=[]
   for n in range(25):
    delta=u[:,n]-v[:,n];square=float(np.sum(weight[mask]*delta[mask]**2));location=np.unravel_index(np.argmax(np.where(mask,abs(delta),-1)),delta.shape)
    norms.append(dict(component=n,l2=float(np.sqrt(square/vol)),linf=float(abs(delta[location])),squared_integral=square,maximum_active_patch_kji=list(map(int,location))))
    if label=='all':global_squared[n]+=square
   results[name]=dict(cells=int(mask.sum()),volume=vol,components=norms)
   # Independent Jacobian formula must agree with the actual native quadrature.
   for case in ([] if label.startswith('collar_') else [a.first,a.second]):
    rows=csv.DictReader((case/'cubed_sphere_conformal_regions.csv').open());matches=[r for r in rows if r['region']==('shell' if region else 'bulk') and r['subset']==label and float(r['time'])==first['time']]
    assert len(matches)==1 and int(matches[0]['cells'])==int(mask.sum())
    assert abs(float(matches[0]['volume'])/vol-1)<1e-10,(name,vol,matches[0]['volume'])
   if label=='all':total_volume+=vol
 for prefix in ['fixed_band','two_cell_band']:
  parts=[results['shell/collar_'+prefix+'_'+label] for label in ['face_interior','edge','corner']]
  collar=results['shell/fixed_outer_collar']
  assert sum(part['cells'] for part in parts)==collar['cells']
  assert abs(sum(part['volume'] for part in parts)/collar['volume']-1)<1e-12
  for n in range(25):
   total=collar['components'][n]['squared_integral'];subtotal=sum(part['components'][n]['squared_integral'] for part in parts)
   assert abs(total-subtotal)<=1e-12*max(abs(total),1e-300)
   for part in parts:part['components'][n]['collar_squared_fraction']=part['components'][n]['squared_integral']/total if total else None
 out=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seam_partition_definition='On r>0.9, classify zero, one or two angular coordinates with abs(q)>0.8 (fixed band) or abs(q)>1-4/Na (shrinking two-cell stencil band). These are supplementary complete partitions; all global and native fixed regions remain included.',time=first['time'],cycle=first['cycle'],payloads=[first['payload'],second['payload']],binary_hashes=[first['tag'],second['tag']],closure_signatures=[s[12],second['signature'][12]],scope='Coordinate-volume componentwise closure differences on identical grids and parameters. Across resolutions compare these fixed-region norms directly; no cross-grid interpolation. Cell-center near-scri sampling still changes with resolution.',regions=results,global_component_l2=np.sqrt(global_squared/total_volume).tolist())
 out.update(variation=a.variation,cycles=[first['cycle'],second['cycle']],scope='Coordinate-volume componentwise differences at identical physical time and grid. Variation checks enforce only a closure change, factor-two fixed timestep or CFL change, or a binary change. Adaptive timesteps and restart milestones may differ; these differences are included, not isolated. No cross-grid interpolation; near-scri sampling changes with refinement.')
 a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print({k:max(c['l2'] for c in v['components']) for k,v in results.items()})
if __name__=='__main__':main()
