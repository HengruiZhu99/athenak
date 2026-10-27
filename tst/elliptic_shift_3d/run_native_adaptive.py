#!/usr/bin/env python3
"""Actual Driver adaptive refine/coarsen controls, regular Minkowski only."""
import argparse,hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--integrator',choices=['rk3','rk4'],default='rk4');p.add_argument('--shells',type=int,choices=[1,2],default=2);p.add_argument('--amplitude',type=float,default=1e-6);p.add_argument('--cfl',type=float,default=.4);p.add_argument('--ceiling',type=float,default=.0001);p.add_argument('--build-evidence',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-adaptive-build-preflight-20261027'));a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);build=a.build_evidence.resolve();binary=build/'athena-frozen';prefix='native_adaptive';m={'scope':'Actual regular Minkowski native adaptive Driver, canonical22 transaction; no wormhole/MPI/device/midstage claim','binary_sha256':sha(binary),'build_manifest_sha256':sha(build/'manifest.json'),'integrator':a.integrator,'shells':a.shells,'amplitude':a.amplitude,'cfl':a.cfl,'ceiling':a.ceiling,'maximum_seconds_per_process':180,'maximum_active_cells':50000,'cases':[]}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 def run(name,args,success=True):
  folder=out/name;folder.mkdir();command=list(map(str,args))+['-d',str(folder)];row={'name':name,'command':command,'expected_success':success};m['cases'].append(row);save();t=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid;save()
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('refusing unowned group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  row['seconds']=time.monotonic()-t;row['expectation_met']=(row['exit_code']==0)==success;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True)
  if not row['expectation_met']:raise RuntimeError('unexpected native adaptive result '+name)
  return folder
 template=(root/'inputs/z4c/composite_minkowski.athinput').read_text().replace('nx1=6','nx1=12',1).replace('nx2=6','nx2=12',1).replace('nx3=6','nx3=12',1).replace('nghost=3','nghost=4').replace('refinement=none','refinement=adaptive\nmax_nmb_per_rank=32\nnum_levels=2\nrefinement_interval=1\nncycle_check=1').replace('integrator=rk4','integrator='+a.integrator).replace('nlim=2','nlim=4').replace('tlim=.0002','tlim=.0004').replace('basename=native_composite','basename='+prefix).replace('composite_output=native_composite','composite_output='+prefix).replace('composite_initial_lapse_amplitude=0','composite_initial_lapse_amplitude='+format(a.amplitude,'.17g')).replace('composite_initial_angular=false','composite_initial_angular=true\ncomposite_adaptive_experimental=true\ncomposite_validation_refine_cycle=1\ncomposite_validation_coarsen_cycle=3')
 if a.shells==1:template=template.replace('composite_radii=.65,1','composite_radii=1')
 template=template.replace('cfl_number=.4','cfl_number='+format(a.cfl,'.17g')).replace('composite_dt=.0001','composite_dt='+format(a.ceiling,'.17g'))
 if a.cfl!=.4 or a.ceiling!=.0001:
  template=template.replace('tlim=.0004','tlim=.002')
  template=template.replace('\ndt=.0001','\ndt=0\ndcycle=1')
 template+='\n<amr_criterion1>\nmethod=location\nlocation_rad=0\n';input=out/'adaptive.athinput';input.write_text(template);m['input_sha256']=sha(input);save()
 full=run('full',[binary,'-i',input]);events=[json.loads(s) for s in (full/(prefix+'.regrid.jsonl')).read_text().splitlines()];m['events']=events;save();assert [(e['cycle'],e['native_leaves']) for e in events]==[(1,15),(3,8)]
 trace=[json.loads(s) for s in (full/(prefix+'.jsonl')).read_text().splitlines()];m['trace']=trace;save();assert [r['cycle'] for r in trace]==[0,1,2,3,4]
 again=run('restart',[binary,'-r',full/'rst'/(prefix+'.00002.rst')]);checks={}
 for name in [prefix+'.00004.csv',prefix+'.00004.scri.csv',prefix+'.regrid-2.00003.csv',prefix+'.regrid-2.00003.scri.csv']:
  x,y=full/name,again/name;checks[name]={'bitwise':x.read_bytes()==y.read_bytes(),'full_sha256':sha(x),'restart_sha256':sha(y)}
 m['restart']=checks;save();assert all(r['bitwise'] for r in checks.values())
 run('reject-unopted-adaptive',[binary,'-i',input,'z4c/composite_adaptive_experimental=false'],False)
 run('reject-wormhole-adaptive',[binary,'-i',input,'problem/pgen_name=z4c_composite_wormhole','problem/mass=.3333333333333333333'],False)
 run('reject-schedule-reinterpretation',[binary,'-r',full/'rst'/(prefix+'.00002.rst'),'z4c/composite_validation_coarsen_cycle=4'],False)
 m['binary_unchanged']=sha(binary)==m['binary_sha256'];m['input_unchanged']=sha(input)==m['input_sha256'];m['actual_refine_and_coarsen_completed']=True;m['scientific_collapse_claim']=False;save();assert m['binary_unchanged'] and m['input_unchanged']
 print(json.dumps({'actual_events':len(events),'final_restart_bitwise':True,'cells':[r['cells'] for r in trace]}))
if __name__=='__main__':main()
