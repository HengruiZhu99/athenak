#!/usr/bin/env python3
"""Fail-closed check of a fresh232-block/24-rank strong-field diagnostic."""
import argparse,json,math,re
from pathlib import Path
import numpy as np
from check_smr_trumpet_checkpoint import validate,require

def check(run,zero=False,ranks=24):
 run=Path(run);require(int((run/'exit_code.txt').read_text())==0,'Application did not exit0')
 log=(run/'run.log').read_text(errors='replace')
 require(not re.search(r'Z4C_INVALID_STATE|DYNGRMHD_ERROR|MPI_Abort|MPI_ABORT|segmentation fault|SYCL.*exception',log,re.I),'Recorded evolution failure')
 stops=re.findall(r'^Terminating on (cycle limit|time limit|wall clock limit)\s*$',log,re.M);require(len(stops)==1,'Missing/ambiguous clean stopping reason')
 end=re.findall(r'^time=([+\-\d.eE]+) cycle=(\d+)\s*$',log,re.M);require(len(end)==1,'Missing/ambiguous terminal time/cycle');t,cycle=float(end[0][0]),int(end[0][1])
 result=validate(run,ranks,3 if zero else None,zero,Path(__file__).with_name('mesh-audit.json'))
 require(result['passed'],'Checkpoint validity failed');require(result['blocks']==232,'Wrong meshblock count');require(result['cycle']==cycle,'Final checkpoint/log cycles disagree')
 require(abs(t-result['time_M'])<=max(1e-9,abs(t)*6e-7),'Final checkpoint/log times disagree')
 if zero:require(cycle==3 and stops[0]=='cycle limit' and result['residual_exactly_zero'],'Zero gate did not finish exactly3cycles')
 else:
  require(stops[0]in('time limit','wall clock limit'),'Unexpected pulse stopping reason')
  require(not result['residual_exactly_zero'],'Pulse checkpoint has no nonzero response')
 histories={}
 for suffix in ['mhd.hst','user.hst','z4c.user.hst']:
  path=run/('ks_background.'+suffix);text=path.read_text();header=next((x for x in text.splitlines()if '[1]='in x),None);require(header,'Missing history columns')
  columns=re.findall(r'\[(\d+)\]=([^\s]+)',header);a=np.loadtxt(path,ndmin=2);require(a.size>0 and a.shape[1]==len(columns),'Bad history shape');require(np.isfinite(a).all(),'Nonfinite history')
  require(abs(a[-1,0]-result['time_M'])<=max(1e-9,abs(result['time_M'])*1e-10),'Final history/checkpoint times disagree')
  require(np.all(np.diff(a[:,0])>=0),'History time decreases')
  names={name:int(i)-1 for i,name in columns}
  for name,j in names.items():
   if name=='bad-metric':require(np.all(a[:,j]==0),'Invalid metrics in history')
  histories[suffix]={'rows':len(a),'last':{name:float(a[-1,j])for name,j in names.items()}}
 if zero:require(histories['user.hst']['last']['Theta-max']==0,'Zero gate has nonzero Theta history')
 target=1000.;completed=result['time_M']>=target-1e-9
 if stops[0]=='time limit':require(completed,'Time-target stop below1000M')
 result.update(stopping_reason=stops[0],target_M=target,target_completed=completed,histories=histories,classification='exact_zero_gate_pass'if zero else 'target_completed'if completed else 'clean_walltime_stop_below_target',stability_clearance=False,continuation_authorized=False)
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--zero',action='store_true');ap.add_argument('--ranks',type=int,default=24);a=ap.parse_args()
 try:r=check(a.run,a.zero,a.ranks)
 except Exception as e:r={'passed':False,'validation_error':str(e),'error_type':type(e).__name__}
 (a.run/'run-validation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in ['files','histories','residual_field_max_active','residual_field_max_including_ghosts']},indent=2));raise SystemExit(0 if r['passed']else 1)
