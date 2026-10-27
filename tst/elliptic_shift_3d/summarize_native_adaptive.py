#!/usr/bin/env python3
"""Complete-region diagnostics; overlapping interface bands never exclude cells."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_native_adaptive import sidecar
p=argparse.ArgumentParser();p.add_argument('controls',type=Path,nargs='+');p.add_argument('--cycle',type=int,choices=[2,4],default=4);a=p.parse_args()
for root in a.controls:
 full=root/'full';layout=sidecar(full/'rst'/f'native_adaptive.{a.cycle:05d}.rst.composite');q=np.genfromtxt(full/f'native_adaptive.{a.cycle:05d}.csv',delimiter=',',names=True);n=layout['n'];per=n**3;points=len(q);cart=np.arange(points)<layout['native_count']*per;stencil=np.zeros(points,bool);coarsefine=stencil.copy();cart_shell=stencil.copy();shell_radial=stencil.copy();angular=stencil.copy();scri=stencil.copy();topology=layout['topology'];width=2
 # Exact closures include face, edge and corner touching leaves.
 for b in topology:
  base=b['composite']*per;ids=np.arange(base,base+per);xyz=np.column_stack([q[k][ids] for k in ['x','y','z']]);h=(np.array(b['hi'])-b['lo'])/n
  for c in topology:
   if b is c:continue
   if all(min(b['hi'][d],c['hi'][d])>=max(b['lo'][d],c['lo'][d]) for d in range(3)):
    mask=np.ones(per,bool);touch=0
    for d in range(3):
     if b['hi'][d]==c['lo'][d]:mask&=b['hi'][d]-xyz[:,d]<width*h[d];touch+=1
     elif b['lo'][d]==c['hi'][d]:mask&=xyz[:,d]-b['lo'][d]<width*h[d];touch+=1
    if touch:stencil[ids[mask]]=True
    if touch and b['refinement']!=c['refinement']:coarsefine[ids[mask]]=True
  for d in range(3):
   if b['lo'][d]==-layout['half']:cart_shell[ids[xyz[:,d]-b['lo'][d]<width*h[d]]]=True
   if b['hi'][d]==layout['half']:cart_shell[ids[b['hi'][d]-xyz[:,d]<width*h[d]]]=True
 shells=np.arange(points)[~cart];block=shells//per-layout['native_count'];layer=block//6;local=shells%per;i=local%n;j=local//n%n;k=local//(n*n);layers=len(layout['radii'])
 cart_shell[shells[(layer==0)&(i<width)]]=True;shell_radial[shells[((layer>0)&(i<width))|((layer<layers-1)&(i>=n-width))]]=True;angular[shells[(j<width)|(j>=n-width)|(k<width)|(k>=n-width)]]=True;scri[shells[(layer==layers-1)&(i>=n-width)]]=True
 interface=stencil|cart_shell|shell_radial|angular|scri;regions={'all':np.ones(points,bool),'bulk':~interface,'any_interface_stencil_band':interface,'Cartesian':cart,'shells':~cart,'Cartesian_block_seams':stencil,'Cartesian_coarse_fine_seams':coarsefine,'Cartesian_shell_coupling':cart_shell,'radial_shell_seams':shell_radial,'angular_face_seams':angular,'scri_stencil_band':scri};report={'weight':'existing Penrose proper-volume cell weight; M/Z retain kernel conformal norm conventions','all_cells_retained':True,'disjoint_acceptance_partition':'bulk + any_interface_stencil_band; detailed bands overlap','stencil_width_cells':width,'puncture':'not applicable: regular Minkowski only','regions':{}}
 for name,mask in regions.items():
  row={'cells':int(mask.sum())}
  if mask.any():
   w=q['weight'][mask];assert np.all(np.isfinite(w)) and np.all(w>0)
   for field in ['H','M','Theta','Z']:v=q[field][mask];row[field+'_rms']=float(np.sqrt(np.sum(w*v*v)/np.sum(w)));row[field+'_max']=float(max(abs(v)))
  report['regions'][name]=row
 assert report['regions']['bulk']['cells']+report['regions']['any_interface_stencil_band']['cells']==points
 for field in ['H','M','Theta','Z']:
  index=int(np.argmax(abs(q[field])));report[field+'_maximum_point']={'ordinal':index,'x':[float(q[k][index]) for k in ['x','y','z']],'value':float(q[field][index]),'region_membership':[name for name,mask in regions.items() if mask[index]]}
 (root/('region-diagnostics.json' if a.cycle==4 else 'region-diagnostics-refined-cycle2.json')).write_text(json.dumps(report,indent=2)+'\n');print(root.name,'all',points,'bulk',report['regions']['bulk']['cells'],'interfaces',report['regions']['any_interface_stencil_band']['cells'])
