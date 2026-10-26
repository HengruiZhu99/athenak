#!/usr/bin/env python3
"""Fresh or resumed native diagnosis, at most four validated short segments per GPU branch."""
import argparse,csv,json,shutil,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
support=Path(__file__).resolve().parent/'source/tst/cubed_sphere'
if support.is_dir():sys.path.insert(0,str(support))
from run_conformal_spherical import run_case,sha
from run_finite_radius_long_stage import checkpoint
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('grids',type=Path);p.add_argument('output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args();source=a.source.resolve();out=a.output.resolve();grids=json.loads(a.grids.read_text());tag=sha(a.binary);assert 1<=len(grids)<=4
parents={}
for g in grids:
 assert 0<g['extra']['cubed_sphere/conformal_wall_seconds']<=360 and g['tlim']==.2 and g['checkpoint']
 assert 0<g['cfl']<=.15 and 0<g['pole_cfl']<=.1 and abs(g['amplitude'])<=.5
 if g.get('restart'):
  c=json.loads((Path(g['restart']).parent/'receipt.json').read_text());ck=checkpoint(Path(g['restart']).parent,c,tag);assert ck['sha256']==g['restart_sha256'];assert c['bulk']==g['bulk'] and c['nr']==g['nr'] and c['nangular']==g['na'];assert c['reference_unchanged'] and c['all_rows_finite'];assert c['input_sha256']==sha(source/'inputs/z4c/cubed_sphere_conformal_probe.athinput');parents[g['name']]=(c,ck)
 else:parents[g['name']]=(None,dict(time=0.,reference_fingerprint=None))
if a.check_only:print(json.dumps(dict(preflight_passed=True,cases=list(parents),maximum_segments=4,maximum_native_wall=300)));sys.exit(0)
out.mkdir(parents=True,exist_ok=False);binary=out/'athena_frozen';shutil.copy2(a.binary,binary);hashes=json.loads((source.parent/'gpu-source-index.json').read_text());assert all(sha(source/f)==h for f,h in hashes.items())
record=dict(binary_sha256=tag,runner_sha256=sha(Path(__file__)),source_sha256=hashes,grids=grids,maximum_segments=4,scientific_acceptance=False,cases={},histories={},errors={});(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
def run(item):
 i,g=item;name=g['name'];old,oldck=parents[name];history=[];branch=out/name;branch.mkdir();started=time.monotonic()
 try:
  for n in range(4):
   if oldck['time']>=.2 or time.monotonic()-started>1200:break
   label='segment'+str(n).zfill(2);extra=dict(g['extra']);extra.update({'cubed_sphere/conformal_fixed_dt':g['dt'],'cubed_sphere/conformal_checkpoint':str(branch/label/'probe_checkpoint.bin'),'cubed_sphere/conformal_checkpoint_tag':tag})
   if old is not None:extra['cubed_sphere/conformal_restart']=old['checkpoint_payload']['path']
   c=run_case(binary,source,branch,label,str(i)+'.0',g['bulk'],g['nr'],g['na'],g['amplitude'],g['kappa'],g['cfl'],g['tlim'],g['degree'],420,g['pole_cfl'],extra)
   assert c['outcome'] in ['complete','checkpointed'] and c['all_rows_finite'] and c['reference_unchanged'] and c.get('initial_profile_unchanged',True)
   
   for filename,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
    assert all(0<=float(row[field])<=1 for row in csv.DictReader((branch/label/filename).open()) for field in fields), 'absolute native constraint monitor exceeded; no successor'
   ck=checkpoint(branch/label,c,tag);assert (oldck['reference_fingerprint'] is None or ck['reference_fingerprint']==oldck['reference_fingerprint']) and ck['time']>oldck['time']
   history.append(dict(relative_directory=str((branch/label).relative_to(out)),receipt=c,checkpoint=ck));old,oldck=c,ck
  return name,old,history,None
 except Exception:return name,old,history,traceback.format_exc()
with ThreadPoolExecutor(max_workers=len(grids)) as pool:results=list(pool.map(run,enumerate(grids)))
for name,c,history,error in results:record['cases'][name]=c;record['histories'][name]=history;record['errors'][name]=error
record['integrity_passed']=all(e is None for e in record['errors'].values());(out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(dict(integrity_passed=record['integrity_passed'],times={n:c['last_rows'][-1]['time'] for n,c in record['cases'].items()}));assert record['integrity_passed']
