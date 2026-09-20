"""Read a saved uniform checkpoint; active spatial diagnostics, never evolution."""
from pathlib import Path
import argparse,sys,json,struct,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--regression',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();sys.path.insert(0,str(a.regression));from z4c_background_restart import checkpoint,cohort
out=Path(__file__).resolve().parent;run=a.run;first=max((run/'rst/rank_00000000').glob('*.rst'),key=lambda p:checkpoint(p)['cycle']);meta=checkpoint(first);_,records=cohort(run,4,meta['cycle']);raw=first.read_bytes();end=raw.index(b'<par_end>\n')+len(b'<par_end>\n');params={};sec=''
for line in raw[:end].decode().splitlines():
 line=line.split('#')[0].strip()
 if line.startswith('<'):sec=line[1:-1];params[sec]={}
 elif '='in line:k,v=line.split('=',1);params[sec][k.strip()]=v.strip()
ng,nx,ny,nz=struct.unpack_from('<19i',raw,end+8+72+76)[:4];start=end+8+72+2*76+20;loc=[struct.unpack_from('<4i',raw,start+16*b)for b in range(meta['total'])];u=np.asarray([s for r in records for s in r['state']]).reshape(-1,25,nz+2*ng,ny+2*ng,nx+2*ng);owners=[r for r,rec in enumerate(records)for _ in rec['state']]
lo=np.array([float(params['mesh'][f'x{i}min'])for i in(1,2,3)]);hi=np.array([float(params['mesh'][f'x{i}max'])for i in(1,2,3)]);ns=np.array([int(params['mesh'][f'nx{i}'])for i in(1,2,3)]);dx=(hi-lo)/ns;coords=[];fields=[];volumes=[]
for b,ll in enumerate(loc):
 axes=[lo[i]+(ll[i]*n+np.arange(n)+.5)*dx[i]for i,n in enumerate((nx,ny,nz))];z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij');xyz=np.stack([x,y,z],axis=-1);r=np.linalg.norm(xyz,axis=-1);v=u[b,:,ng:ng+nz,ng:ng+ny,ng:ng+nx];chi=(r/(r+1))**2+v[0];xx,xy,xz,yy,yz,zz=1+v[1],v[2],v[3],1+v[4],v[5],1+v[6];det=xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy;assert np.all(chi>0)and np.all(det>0);dv=np.prod(dx)*np.sqrt(det)/chi**1.5;coords.append(xyz);fields.append(v);volumes.append(dv)
xyz=np.asarray(coords);v=np.asarray(fields);dv=np.asarray(volumes);distance=np.min(np.minimum(xyz-lo,hi-xyz),axis=-1);theta=v[:,17];theta2=dv*theta**2;maxima={}
for name,val in [('Theta',np.abs(theta)),('lapse_residual',np.abs(v[:,18])),('shift_residual',np.max(np.abs(v[:,19:22]),axis=1))]:
 idx=np.unravel_index(np.argmax(val),val.shape);b=idx[0];maxima[name]={'value':float(val[idx]),'rank':owners[b],'gid':int(b),'logical_level':loc[b][3],'xyz_M':xyz[idx].tolist(),'nearest_physical_face_M':float(distance[idx])}
cut=[0,32,64,96,128,192,257];bins=[]
for left,right in zip(cut[:-1],cut[1:]):
 m=(distance>=left)&(distance<right);bins.append({'distance_to_face_M':[left,right],'proper_volume':float(np.sum(dv[m])),'Theta2_proper_integral':float(np.sum(theta2[m])),'Theta_max':float(np.max(abs(theta[m])))if np.any(m)else None})
result={'scope':'Saved active-cell residual profile; not the subsequent failed/evolved state and not a stability proof.','checkpoint_name':first.name,'time_M':meta['time'],'cycle':meta['cycle'],'ranks':4,'blocks':meta['total'],'matching_rank_headers':True,'rank_file_sha256':{str(rank):hashlib.sha256((run/'rst'/f'rank_{rank:08d}'/first.name).read_bytes()).hexdigest()for rank in range(4)},'Theta_RMS':float(np.sqrt(np.sum(theta2)/np.sum(dv))),'proper_volume':float(np.sum(dv)),'maxima':maxima,'distance_bins':bins}
# Independently compare the coincident history's proper-volume normalization.
h=np.loadtxt(run/'ks_background.z4c.user.hst');ind=np.argmin(abs(h[:,0]-meta['time']));result['nearest_history_time_M']=float(h[ind,0]);result['history_theta_rms']=float(np.sqrt(h[ind,9]/h[ind,10]));result['history_same_time']=bool(abs(h[ind,0]-meta['time'])<1e-7)
if result['history_same_time']:result['RMS_relative_difference_vs_history']=float(abs(result['Theta_RMS']-result['history_theta_rms'])/max(result['history_theta_rms'],1e-300))
(out/(a.name+'-checkpoint-profile.json')).write_text(json.dumps(result,indent=2)+'\n');print(a.name,result['time_M'],result['Theta_RMS'],result.get('RMS_relative_difference_vs_history'),maxima['Theta'])
