#!/usr/bin/env python3
"""Bounded continuation matrix; old extrapolation evidence is never overwritten."""
import argparse,datetime,gzip,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from validate import ROOT,sources,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--build',type=Path,default=Path('build-shift-3d'));a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);build=a.build.resolve();frozen=sources()
 tag=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
 m=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_commits=['487f750d8b8503b4944ee75d0b3b1efdcee1b72a','0149acb26'],
        source_hashes=frozen,source_tag=tag,tests={},worker_limit=1,case_timeout=180,
        executable_sha256=sha(build/'elliptic_shift_3d_coupled_pilot'),acceptance='actual nodal and independently recovered first-ghost boundary equations; raw active-only extrapolation and raw clock-rate corner errors retained separately')
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 save()
 for name,command in [('native-and-radial-poles',['python3',ROOT/'tst/elliptic_shift_3d/test_scri_radial.py',build/'elliptic_shift_3d_tensor_scri_tests']),('surface-and-checkpoint',[build/'elliptic_shift_3d_lifecycle_tests',out/'test.bin'])]:
  m['tests'][name]=run(command,out,name);save()
 cases=[('initial-n6',[6,1e-6,.0005,0,0,1,1e-11]),('minkowski',[6,0,.0005,4,0,1,1e-11])]
 cases +=[(f'radial-n{n}',[n,1e-6,.0005,4,0,1,1e-11]) for n in (6,8,12)]
 cases +=[(f'angular-two-shell-n{n}',[n,1e-6,.0005,4,1,2,1e-11]) for n in (6,8,12)]
 cases +=[('half-dt',[8,1e-6,.00025,8,0,1,1e-11]),('quarter-dt',[8,1e-6,.000125,16,0,1,1e-11]),('tight-tolerance',[8,1e-6,.0005,4,0,1,1e-13]),('no-boundary-penalty',[8,1e-6,.0005,4,0,1,1e-11]),('extended-radial',[8,1e-6,.0005,40,0,1,1e-11])]
 for name,case in cases:
  profile=out/(name+'.csv');checkpoint=out/(name+'.bin') if name=='radial-n6' else ''
  command=[build/'elliptic_shift_3d_coupled_pilot',*case,profile,checkpoint,tag if checkpoint else '',4,1]
  if name=='no-boundary-penalty':command +=[0,0]
  record=run(command,out,name)
  for suffix,key in [('', 'profile'),('.spheres.csv','spheres')]:
   path=Path(str(profile)+suffix)
   if path.exists():record[key]=path.name;record[key+'_sha256']=sha(path)
  if checkpoint and Path(checkpoint).exists():record['checkpoint']=dict(file=Path(checkpoint).name,sha256=sha(checkpoint),schema=3)
  m['tests'][name]=record;save();print(name,record['returncode'],record['wall_seconds'],flush=True)
 # Independent like-for-like complete state/time/tolerance controls.
 controls={}
 baseline=np.genfromtxt(out/'radial-n8.csv',delimiter=',',names=True)
 for name in ('half-dt','quarter-dt','tight-tolerance','no-boundary-penalty'):
  table=np.genfromtxt(out/(name+'.csv'),delimiter=',',names=True)
  for k in ('x','y','z'):np.testing.assert_array_equal(table[k],baseline[k])
  controls[name]={k:float(np.max(np.abs(table[k]-baseline[k]))) for k in table.dtype.names if k.startswith('state_') or k.startswith('delta_beta')}
 m['controls']=controls
 for receipt in m['tests'].values():
  for key in ('profile','spheres'):
   if key not in receipt:continue
   path=out/receipt[key];target=path.with_name(path.name+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
   receipt[key+'_uncompressed_sha256']=receipt[key+'_sha256'];receipt[key]=target.name;receipt[key+'_sha256']=sha(target);path.unlink()
 m['sources_unchanged']=sources()==frozen;m['executable_unchanged']=sha(build/'elliptic_shift_3d_coupled_pilot')==m['executable_sha256']
 m['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 assert m['sources_unchanged'] and m['executable_unchanged']
 print(out/'manifest.json')

if __name__=='__main__':main()
