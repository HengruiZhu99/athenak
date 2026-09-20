"""Read-only collection of existing local controls; never runs AthenaK or changes jobs."""
import argparse,datetime,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--regression',type=Path,required=True);p.add_argument('--validate-clean',action='store_true');a=p.parse_args();out=Path(__file__).resolve().parent;study=a.study.resolve();os.environ['ATHENA_REGRESSION_PATH']=str(a.regression.resolve());sys.path.insert(0,str(a.regression.resolve()))
from z4c_background_restart import checkpoint,cohort
assert (study/'local/kappa01/run.log').is_file() and (study/'fast-weakfield/run.log').is_file(), 'Pass --study pointing to the complete external raw run collection; packaged summaries alone cannot be recollected'
spec=importlib.util.spec_from_file_location('saved_checker',study/'check_trumpet_checkpoint.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
NUM=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
windows=[(0,500),(500,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,15000),(15000,20000),(20000,30000),(30000,40000),(40000,50000)]
thresholds={'theta_rms':1e-14,'H_rms':1e-14,'theta_max':1e-13,'lapse_max':1e-11,'shift_max':1e-11}

def params(path):
 d={};b=''
 for l in path.read_text().splitlines():
  l=l.split('#')[0].strip()
  if l.startswith('<'):b=l[1:-1]
  elif '='in l:
   k,v=l.split('=',1);d[b+'/'+k.strip()]=v.strip()
 return d

def hst(path):
 raw=path.read_bytes();lines=raw.decode().splitlines()
 if raw and not raw.endswith(b'\n'):lines=lines[:-1]
 labels=[];rows=[];malformed=0
 for l in lines:
  if l.startswith('#'):
   if '[1]='in l:labels=re.findall(r'\[\d+\]=(\S+)',l)
   continue
  if not l.strip():continue
  try:row=[float(x)for x in l.split()]
  except ValueError:malformed+=1;continue
  if len(row)!=len(labels):malformed+=1;continue
  rows.append(row)
 arr=np.asarray(rows,float)
 # Preserve last complete entry at duplicated times, retaining chronological order.
 if len(arr):
  _,ind=np.unique(arr[::-1,0],return_index=True);arr=arr[len(arr)-1-ind];arr=arr[np.argsort(arr[:,0])]
 return labels,arr,malformed,hashlib.sha256(raw).hexdigest()

def fits(t,y,name):
 rows=[]
 for lo,hi in windows:
  mask=(t>=lo-1e-8)&(t<=hi+1e-8);x=t[mask];v=y[mask]
  if len(x)<10 or x[-1]-x[0]<.8*(hi-lo):continue
  record={'window_M':[lo,hi],'actual_window_M':[float(x[0]),float(x[-1])],'samples':len(x),'min':float(np.min(v)),'max':float(np.max(v)),'endpoint_ratio':float(v[-1]/v[0])if v[0]>0 else None}
  if not(np.isfinite(v).all()and np.all(v>thresholds[name])):record['fit_status']='omitted: values at/below conservative analysis threshold or nonfinite'
  else:
   z=np.log(v);m,b=np.polyfit(x,z,1);res=z-(m*x+b);den=np.sum((z-z.mean())**2);record.update(fit_status='fitted',gamma_M_inverse=float(m),e_fold_M=float(1/m)if m>0 else None,R_squared=float(1-np.sum(res**2)/den)if den>0 else None)
  rows.append(record)
 return rows

results={'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing local fixed-binary vacuum weak-field controls; no production jobs or source changed.','fit_thresholds':thresholds,'norm_definition':'Theta_rms=sqrt(Theta-norm2/Volume), H_rms=sqrt(H-norm2/Volume). History label Theta-norm is truncated; its stored value is the integral of Theta squared, not a norm. Volume is proper active-cell volume.','cases':{}}
for name,rel in [('kappa01','local/kappa01'),('all_weak','local/all_weak'),('no_damping','local/no_damping'),('gauge_weak','local/gauge_weak'),('fast_weakfield','fast-weakfield')]:
 d=study/rel
 if not(d/'input.athinput').exists():continue
 par=params(d/'input.athinput');log=(d/'run.log').read_text(errors='replace')if(d/'run.log').exists()else'';lines=log.splitlines();ex=json.loads((d/'execution.json').read_text())if(d/'execution.json').exists()else None
 cfg={k:par.get(k)for k in ['time/tlim','time/cfl_number','z4c/damp_kappa1','z4c/shift_eta','z4c/residual_lapse_damping','z4c/shift_Gamma','problem/outer_sponge_enabled','problem/vacuum_gauge_pulse_amplitude']};rec={'path':str(d),'config':cfg,'input_sha256':hashlib.sha256((d/'input.athinput').read_bytes()).hexdigest(),'execution':ex}
 if not(d/'ks_background.user.hst').exists():rec['status']='not_started';results['cases'][name]=rec;continue
 ul,u,ubad,uhash=hst(d/'ks_background.user.hst');zl,z,zbad,zhash=hst(d/'ks_background.z4c.user.hst');np.savez_compressed(out/(name+'-histories.npz'),user=u,user_labels=ul,z4c=z,z4c_labels=zl)
 iu={n:i for i,n in enumerate(ul)};iz={n:i for i,n in enumerate(zl)};common=min(len(u),len(z));assert np.allclose(u[:common,0],z[:common,0],rtol=0,atol=1e-8),(name,'mismatched history times')
 u=u[:common];z=z[:common];t=u[:,0];thlabel='Theta-norm2'if'Theta-norm2'in iz else'Theta-norm';vol=z[:,iz['Volume']]
 fields={'theta_rms':np.sqrt(z[:,iz[thlabel]]/vol),'H_rms':np.sqrt(z[:,iz['H-norm2']]/vol),'theta_max':u[:,iu['Theta-max']],'lapse_max':u[:,iu['alpha-res']],'shift_max':u[:,iu['beta-res']]}
 invalid=[]
 for line in lines:
  if 'Z4C_INVALID_STATE 'in line:
   vals=dict(re.findall(r'(time|cycle|rank|gid|relative_level|x|y|z|local_bad_cells)=('+NUM+r')',line));invalid.append({k:float(v)for k,v in vals.items()})
 failures=[i for i,l in enumerate(lines)if 'An error occurred during the primitive solve:'in l];first_failure=None
 if failures:
  idx=failures[0];chunk='\n'.join(lines[idx:idx+24]);xy=re.search(r'\n\s*\(('+NUM+r'),\s*('+NUM+r'),\s*('+NUM+r')\)',chunk);det=re.search(r'detg\s*=\s*('+NUM+')',chunk)
  prev=[float(x)for l in lines[:idx]for x in re.findall(r'Z4C_EXT_HMAX time=('+NUM+')',l)];following=[float(x)for l in lines[idx:]for x in re.findall(r'Z4C_EXT_HMAX time=('+NUM+')',l)]
  first_failure={'message':lines[idx],'xyz_M':[float(x)for x in xy.groups()]if xy else None,'determinant':float(det[1])if det else None,'preceding_printed_H_time_M':prev[-1]if prev else None,'following_printed_H_time_M':following[0]if following else None,'time_scope':'Only stdout order; no exact first-event time/stage/rank in this primitive message.'}
  if xy:
   xx=np.asarray(first_failure['xyz_M']);lo=np.array([float(par[f'mesh/x{i}min'])for i in (1,2,3)]);hi=np.array([float(par[f'mesh/x{i}max'])for i in (1,2,3)]);dx=(hi-lo)/np.array([int(par[f'mesh/nx{i}'])for i in (1,2,3)]);depth=np.maximum(np.maximum(np.ceil((lo-xx)/dx),np.ceil((xx-hi)/dx)),0).astype(int);first_failure['physical_ghost_depth_xyz']=depth.tolist()
 progress=[(float(e),int(c),float(tt),float(dt))for e,c,tt,dt in re.findall(r'elapsed=('+NUM+r')\s+cycle=(\d+)\s+time=('+NUM+r')\s+dt=('+NUM+r')',log)]
 final_match=re.search(r'Terminating on ([^\n]+)\n\s*time=('+NUM+r')\s+cycle=(\d+)',log)
 final_log={'reason':final_match[1],'time_M':float(final_match[2]),'cycle':int(final_match[3])}if final_match else None
 target=float(par['time/tlim']);status='running_or_unconfirmed'
 if invalid or (ex and ex['exit_code']!=0):status='failed'
 elif first_failure and ex:status='finished_with_evolution_errors'
 elif ex and ex['exit_code']==0 and final_log:status='completed_target'if final_log['time_M']>=target-1e-7 else'clean_stop_below_target'
 rec.update(status=status,final_log=final_log,last_history_M=float(t[-1]),target_M=target,history_rows=common,all_history_fields_finite=bool(np.isfinite(u).all()and np.isfinite(z).all()),discarded_malformed_rows={'user':ubad,'z4c':zbad},history_sha256={'user':uhash,'z4c':zhash},last={key:float(value[-1])for key,value in fields.items()},badmetrics_max=float(max(u[:,iu['bad-metric']])),last_metric_history={n:float(u[-1,iu[n]])for n in ['alpha-min','chi-min','detg-min','detg-max']},fits={n:fits(t,v,n)for n,v in fields.items()},first_printed_primitive_failure=first_failure,active_invalid_events=invalid,last_progress=dict(zip(['wall_elapsed_s','cycle','time_M','dt_M'],progress[-1]))if progress else None)
 hmax=[]
 for line in lines:
  if'Z4C_EXT_HMAX'in line:
   vals=dict(re.findall(r'(time|H|x|y|z|r_bh)=('+NUM+r')',line));
   if len(vals)==6:hmax.append({k:float(v)for k,v in vals.items()})
 rec['latest_H_max_location']=hmax[-1]if hmax else None
 # Checkpoint availability is separate from the latest evolved state, especially on failure.
 cps=list((d/'rst/rank_00000000').glob('*.rst'))
 if cps:
  readable=[];unreadable=[]
  for cp in cps:
   try:readable.append((cp,checkpoint(cp)))
   except(Exception)as exc:unreadable.append({'path':str(cp),'error':str(exc)})
  rec['unreadable_checkpoint_candidates']=unreadable
  if readable:
   cp,meta=max(readable,key=lambda item:item[1]['cycle']);rec['latest_checkpoint']={'rank0_path':str(cp),'time_M':meta['time'],'cycle':meta['cycle'],'older_than_latest_history':bool(meta['time']<t[-1]-1e-8),'represents_failure_state':False}
  marker=out/(name+'-checkpoint.json')
  if a.validate_clean and status in ['completed_target','clean_stop_below_target']:
   try:
    valid=checker.validate(d,4);tol=5e-7*max(1,abs(final_log['time_M']));valid['matches_final_log_time_cycle']=bool(valid['cycle']==final_log['cycle']and abs(valid['time_M']-final_log['time_M'])<=tol);valid['log_time_print_tolerance_M']=tol
    marker.write_text(json.dumps(valid,indent=2)+'\n');rec['checkpoint_validation']=valid
    if not(valid['passed']and valid['matches_final_log_time_cycle']):rec['status']='checkpoint_validation_failed'
   except(Exception)as exc:rec['checkpoint_validation_error']=str(exc);rec['status']='checkpoint_validation_failed'
  elif marker.exists():rec['checkpoint_validation']=json.loads(marker.read_text())
 results['cases'][name]=rec
if(out/'timestep-review.json').exists():results['timestep_comparison']=json.loads((out/'timestep-review.json').read_text())
(out/'status.json').write_text(json.dumps(results,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n')
for name,r in results['cases'].items():print(name,r['status'],r.get('last_history_M'),r.get('last',{}).get('theta_rms'),r.get('last',{}).get('theta_max'),flush=True)
