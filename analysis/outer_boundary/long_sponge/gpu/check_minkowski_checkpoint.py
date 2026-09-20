#!/usr/bin/env python3
"""Fail-closed validator for uniform residual-Z4c Minkowski diagnostic restarts.

All ghost cells and all physical fields are checked. Geometry comes from the
actual common checkpoint header; partition order is contiguous global gid.
Supports multiple uniform blocks per rank, no AMR and no forced metric override.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,struct,sys
import numpy as np
here=Path(__file__).resolve().parent
regression_path=os.environ.get('ATHENA_REGRESSION_PATH')
if regression_path is None:
 regression_path=next((str(p/'tst/regression') for p in here.parents if (p/'tst/regression/z4c_background_restart.py').is_file()),None)
if regression_path is None:raise RuntimeError('Set ATHENA_REGRESSION_PATH to the AthenaK tst/regression directory')
sys.path.insert(0,regression_path)
from z4c_background_restart import checkpoint

def checkpoint_header(path):
 """Read only metadata for cohort selection; selected payloads are still validated."""
 with path.open('rb') as stream:
  prefix=stream.read(262144);marker=b'<par_end>\n';stop=prefix.find(marker)
  assert stop>=0,'Missing checkpoint parameter header'
  end=stop+len(marker);stream.seek(end)
  total,level=struct.unpack('<ii',stream.read(8));stream.seek(72+2*76,1)
  time,dt,cycle=struct.unpack('<ddi',stream.read(20))
  assert np.isfinite(time) and np.isfinite(dt) and total>0
  return {'time':time,'dt':dt,'cycle':cycle,'total':total,'level':level}

def cohort(run,ranks,cycle):
 paths=sorted((run/'rst/rank_00000000').glob('*.rst'))
 matches=[p for p in paths if checkpoint_header(p)['cycle']==cycle]
 assert matches,'No checkpoint at cycle '+str(cycle)
 first=matches[-1];files=[run/'rst'/('rank_%08d'%r)/first.name for r in range(ranks)]
 assert set((run/'rst').glob('rank_*/'+first.name))==set(files)
 records=[checkpoint(p) for p in files]
 assert len({(r['cycle'],r['time'],r['dt'],r['total']) for r in records})==1
 assert len({r['header_hash'] for r in records})==1,'Rank metadata differs'
 assert sum(len(r['state']) for r in records)==records[0]['total']
 return first,records

def boolean(d,k,default=False):
 v=str(d.get(k,default)).lower()
 if v in ('true','1'):return True
 if v in ('false','0'):return False
 raise ValueError('Unsupported boolean '+k+'='+v)

def supported(p):
 assert float(p['problem']['bh_mass'])==0 and float(p['problem'].get('bh_spin','0'))==0
 assert p['problem']['bh_background']=='kerr_schild'
 assert boolean(p['coord'],'minkowski') and not boolean(p['problem'],'force_minkowski_metric')
 assert boolean(p['problem'],'use_direct_z4c_background',True)
 assert boolean(p['z4c'],'use_analytic_background') and float(p['z4c']['chi_psi_power'])==-4
 z=p['z4c'];assert boolean(z,'evolve_lapse_residual',boolean(z,'evolve_gauge_residual',True)) or boolean(z,'preserve_lapse_residual')
 assert p['mesh_refinement']['refinement']=='none'

def validate(run,ranks,cycle=None,exact_zero=False,check_evolution=True):
 if cycle is None:cycle=max(checkpoint_header(p)['cycle'] for p in (run/'rst/rank_00000000').glob('*.rst'))
 first,records=cohort(run,ranks,cycle);q=records[0];raw=first.read_bytes();end=raw.index(b'<par_end>\n')+len(b'<par_end>\n');params={};section=None
 for line in raw[:end].decode().splitlines():
  line=line.split('#',1)[0].strip()
  if line.startswith('<'):section=line[1:-1];params[section]={}
  elif '=' in line:k,v=line.split('=',1);params[section][k.strip()]=v.strip()
 supported(params)
 ng,nx,ny,nz=struct.unpack_from('<19i',raw,end+8+72+76)[:4];start=end+8+72+2*76+20
 locs=[struct.unpack_from('<4i',raw,start+16*i) for i in range(q['total'])];assert all(loc[3]==q['level'] for loc in locs)
 payload_start=start+20*q['total']+24;shape=(25,nz+2*ng,ny+2*ng,nx+2*ng)
 mesh=params['mesh'];lo=np.array([float(mesh['x%dmin'%a]) for a in (1,2,3)]);hi=np.array([float(mesh['x%dmax'%a]) for a in (1,2,3)]);ns=np.array([int(mesh['nx%d'%a]) for a in (1,2,3)]);dx=(hi-lo)/ns
 assert all(ns[a]%n==0 for a,n in enumerate((nx,ny,nz)))
 minima={k:float('inf') for k in ('alpha','chi','gxx','minor2','detg')};bad=0;samples=[];files=[];gid=0;peak=0.;nonzero=0;all_finite=True
 for rank,rec in enumerate(records):
  f=run/'rst'/('rank_%08d'%rank)/first.name;b=f.read_bytes();finite=bool(np.isfinite(np.frombuffer(b[payload_start:],dtype='<f8')).all());all_finite &= finite
  files.append({'rank':rank,'name':str(f),'blocks':len(rec['state']),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'all_payload_finite':finite})
  for values in rec['state']:
   u=np.asarray(values).reshape(shape);loc=locs[gid];axes=[lo[a]+(loc[a]*n+np.arange(n+2*ng)-ng+.5)*dx[a] for a,n in enumerate((nx,ny,nz))]
   xx,xy,xz,yy,yz,zz=1+u[1],u[2],u[3],1+u[4],u[5],1+u[6]
   fields={'alpha':1+u[18],'chi':1+u[0],'gxx':xx,'minor2':xx*yy-xy*xy,'detg':xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy}
   invalid=np.zeros(shape[1:],dtype=bool)
   for key,v in fields.items():invalid|=~np.isfinite(v)|(v<=0);minima[key]=min(minima[key],float(v.min()))
   bad+=int(invalid.sum());peak=max(peak,float(abs(u).max()));nonzero+=int(np.count_nonzero(u))
   for k,j,i in np.argwhere(invalid)[:max(0,12-len(samples))]:
    xyz=[float(axes[a][v]) for a,v in enumerate((i,j,k))];depth=[max(ng-int(v),int(v)-(ng+n-1),0) for v,n in zip((i,j,k),(nx,ny,nz))]
    samples.append({'rank':rank,'gid':gid,'logical_location':loc,'relative_level':loc[3]-q['level'],'xyz':xyz,'local_ijk':[int(i),int(j),int(k)],'local_ghost_depth':depth,'physical_ghost_codimension':int(sum(x<l or x>h for x,l,h in zip(xyz,lo,hi))),'values':{key:float(v[k,j,i]) for key,v in fields.items()}})
   gid+=1
 assert gid==q['total']
 passed=all_finite and bad==0 and (not exact_zero or nonzero==0)
 result={'time_code':q['time'],'cycle':q['cycle'],'ranks':ranks,'blocks':q['total'],'matching_headers':True,'all_payload_finite':all_finite,'invalid_metric_cells_including_ghosts':bad,'invalid_samples':samples,'minimum':minima,'residual_max':peak,'exact_zero_required':exact_zero,'all_residuals_zero':nonzero==0,'files':files,'dx':dx.tolist(),'stopping_reason':None,'passed':passed}
 if check_evolution:
  log=(run/'run.log').read_text();stops=re.findall(r'Terminating on ([^\n]+)',log);assert stops
  result['stopping_reason']=stops[-1];assert result['stopping_reason'] in ('time limit','wall clock limit','cycle limit')
  final=re.findall(r'^time=([^ ]+) cycle=(\d+)\s*$',log,re.M);assert final
  reported_time,reported_cycle=float(final[-1][0]),int(final[-1][1]);assert reported_cycle==q['cycle']
  assert abs(reported_time-q['time'])<=5e-7*max(1,abs(q['time']))
  result['checkpoint_matches_final_application_record']=True
  if exact_zero and int(params['time']['nlim'])>=0:assert q['cycle']==int(params['time']['nlim'])
  result['invalid_state_marker']=bool(re.search(r'Z4C_INVALID_STATE|An error occurred during the primitive solve|### FATAL ERROR',log));passed &= not result['invalid_state_marker']
  histories={}
  for path in run.glob('*.hst'):
   lines=path.read_text().splitlines();names=re.findall(r'\[\d+\]=(\S+)',lines[1]);a=np.atleast_2d(np.loadtxt(path));finite=bool(np.isfinite(a).all());item={'last_time':float(a[-1,0]),'all_finite':finite};passed &= finite
   if 'bad-metric' in names:item['bad_metric_max']=float(a[:,names.index('bad-metric')].max());passed &= item['bad_metric_max']==0
   histories[path.name]=item
  assert histories;result['histories']=histories
  if (run/'exit_code.txt').exists():result['application_exit']=int((run/'exit_code.txt').read_text());passed &= result['application_exit']==0
  result['target_time']=float(params['time']['tlim']);result['target_reached']=q['time']>=result['target_time']-1e-10
 result['passed']=bool(passed);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--ranks',type=int,required=True);p.add_argument('--cycle',type=int);p.add_argument('--exact-zero',action='store_true');p.add_argument('--checkpoint-only',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 r=validate(a.run.resolve(),a.ranks,a.cycle,a.exact_zero,not a.checkpoint_only);out=a.output or a.run/'checkpoint-validation.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2));sys.exit(0 if r['passed'] else 1)
