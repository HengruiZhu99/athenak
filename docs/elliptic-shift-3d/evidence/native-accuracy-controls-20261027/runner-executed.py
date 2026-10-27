#!/usr/bin/env python3
"""Bounded native accuracy controls using a frozen executable/header capsule.
No live PDE header, production executable or original job is modified/read for execution.
"""
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--capsule',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',choices=['dt_half','tolerance','n8','n10'],default=['dt_half','tolerance','n8','n10']);a=p.parse_args();root=Path.cwd();capsule=a.capsule.resolve();out=a.output.resolve();out.mkdir(exist_ok=False);resource.setrlimit(resource.RLIMIT_CORE,(0,0));cm=json.loads((capsule/'manifest.json').read_text());binary=capsule/'athena-qualified';reference=capsule/'native_wormhole_reference';assert sha(binary)==cm['production_binary_sha256'] and sha(reference)==cm['matched_reference_binary_sha256'];assert all(sha(capsule/f)==h for f,h in cm['capsule_source_sha256'].items())
 budgets={'dt_half':dict(n=6,dt=5e-5,cycles=4,tolerance=1e-11,cells=10800),'tolerance':dict(n=6,dt=1e-4,cycles=2,tolerance=1e-12,cells=10800),'n8':dict(n=8,dt=1e-4,cycles=2,tolerance=1e-11,cells=25600),'n10':dict(n=10,dt=1e-4,cycles=2,tolerance=1e-11,cells=50000)}
 m={'scope':'Actual frozen-v14 Serial native Driver accuracy controls; all scientific gates separately retained, no tolerance relaxation or generic regularity claim','capsule_manifest_sha256':sha(capsule/'manifest.json'),'production_archive_sha256':cm['production_archive_sha256'],'binary_sha256':sha(binary),'matched_reference_sha256':sha(reference),'predeclared_budget':{'one_CPU_per_native_process':True,'maximum_seconds_per_process':180,'cases':{k:budgets[k] for k in a.cases},'n10_fallback':'Only after timed-out owned process is reaped, inspect accepted checkpoint1; use it or run fresh one-cycle segment, then resume cycle2 within separate180s cap; never duplicate uncertain execution.'},'cases':{}}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 save()
 def run(row,name,argv):
  r={'name':name,'argv':list(map(str,argv)),'maximum_seconds':180};row['commands'].append(r);save();start=time.monotonic()
  with (out/(row['name']+'-'+name+'.stdout')).open('w') as so,(out/(row['name']+'-'+name+'.stderr')).open('w') as se:
   process=subprocess.Popen(list(map(str,argv)),cwd=root,stdout=so,stderr=se,start_new_session=True);r['owned_process_group']=process.pid;save()
   try:r['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    r.update(exit_code=124,owned_process_reaped=True,cleanup='only newly owned process group terminated')
  r['seconds']=time.monotonic()-start;save();print(row['name'],name,r['exit_code'],round(r['seconds'],3),flush=True);return r['exit_code']
 for key in a.cases:
  spec=budgets[key];case=out/key;case.mkdir();row={'name':key,**spec,'commands':[],'scientific_acceptance':False};m['cases'][key]=row;save();n=spec['n'];coords=case/'coordinates.csv';initial=case/'initial.dat'
  assert run(row,'coordinates',[reference,'--coordinates',n,coords])==0
  assert run(row,'initializer',[sys.executable,capsule/'tst/elliptic_shift_3d/wormhole_initial.py',coords,initial])==0
  row['initializer']=json.loads(initial.with_suffix('.json').read_text());row['initializer']['sha256']=sha(initial);save()
  overrides=['mesh/nx'+str(i)+'='+str(2*n) for i in [1,2,3]]+['meshblock/nx'+str(i)+'='+str(n) for i in [1,2,3]]+['z4c/composite_dt='+format(spec['dt'],'.17g'),'time/nlim='+str(spec['cycles']),'z4c/composite_tolerance='+format(spec['tolerance'],'.17g')]
  full=case/'full';code=run(row,'full',[binary,'-i',capsule/'inputs/z4c/composite_wormhole_static8.athinput',*overrides,'z4c/composite_initial_file='+str(initial),'-d',full]);final=full
  if code==124 and key=='n10':
   checkpoint=full/'rst/native_wormhole.00001.rst';sidecar=Path(str(checkpoint)+'.composite')
   row['timeout_checkpoint_inspection']={'native_checkpoint1_exists':checkpoint.exists(),'paired_sidecar_exists':sidecar.exists(),'owned_process_reaped':True};save()
   if not checkpoint.exists() or not sidecar.exists():
    first=case/'segment1';assert run(row,'segment1',[binary,'-i',capsule/'inputs/z4c/composite_wormhole_static8.athinput',*overrides,'time/nlim=1','z4c/composite_initial_file='+str(initial),'-d',first])==0;checkpoint=first/'rst/native_wormhole.00001.rst'
   final=case/'segment2';code=run(row,'segment2',[binary,'-r',checkpoint,'time/nlim=2','-d',final]);row['segmented_after_timeout']=True
  if code!=0:row['execution_failed']=True;save();continue
  trace=[json.loads(s) for s in (final/'native_wormhole.jsonl').read_text().splitlines()];row['final_folder']=str(final.relative_to(out));row['trace']=trace;assert trace[-1]['cycle']==spec['cycles'] and abs(trace[-1]['time']-.0002)<1e-15 and trace[-1]['cells']==spec['cells'];save()
  # Bind same-data restart without reopening initializer. Full successful runs
  # restart from penultimate cycle; segmented N10 reuses its existing checkpoint1.
  hidden=case/'initial.hidden-after-fresh-run.dat';initial.rename(hidden);row['initializer_hidden_before_restart']=True;save();cycle=spec['cycles']-1;checkpoint=full/f'rst/native_wormhole.{cycle:05d}.rst'
  if not checkpoint.exists():checkpoint=case/'segment1/rst/native_wormhole.00001.rst'
  again=case/'restart';code=run(row,'restart',[binary,'-r',checkpoint,'-d',again])
  if code==0:
   checks={}
   for suffix in ['.csv','.scri.csv']:
    f=final/f'native_wormhole.{spec["cycles"]:05d}{suffix}';g=again/f.name;checks[suffix]={'bitwise_identical':f.read_bytes()==g.read_bytes(),'full_sha256':sha(f),'restart_sha256':sha(g)}
   row['restart']=checks;assert all(v['bitwise_identical'] for v in checks.values())
  save()
  # Native actual RK receipt/time sequence; independent header capsule remains
  # the same physical engine. No classical-RK substitution.
  dts=case/'dt.dat';dts.write_text(''.join(format(spec['dt'],'.17g')+'\n' for _ in range(spec['cycles'])));receipt=full/'native_wormhole.rk.dat'
  code=run(row,'matched-reference',[reference,hidden,receipt,dts,case/'reference',2,n,spec['tolerance']])
  if code==0:
   comparisons={}
   for cycle in range(spec['cycles']+1):
    f=full/f'native_wormhole.{cycle:05d}.csv'
    if not f.exists() and cycle==spec['cycles']:f=final/f.name
    if not f.exists():continue
    q=np.genfromtxt(f,delimiter=',',names=True);r=np.genfromtxt(case/f'reference.{cycle:05d}.csv',delimiter=',',names=True);fields=[f'state_{i}' for i in range(22)]+[f'delta_beta_{i}' for i in range(3)];errors={name:float(np.max(abs(q[name]-r[name]))) for name in fields};comparisons[str(cycle)]={'cells':len(q),'field_errors':errors,'all_bitwise':all(np.array_equal(q[f],r[f]) for f in fields)}
   row['matched_actual_RK']=comparisons;assert all(r['all_bitwise'] for r in comparisons.values())
  else:row['matched_reference_incomplete']='Exceeded prescribed cap or failed; not claimed passing'
  save()
 m['capsule_unchanged']=all(sha(capsule/f)==h for f,h in cm['capsule_source_sha256'].items());m['binary_unchanged']=sha(binary)==m['binary_sha256'];assert m['capsule_unchanged'] and m['binary_unchanged'];save()

if __name__=='__main__':main()
