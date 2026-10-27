#!/usr/bin/env python3
"""Final shared-athena default/uniform and static-SMR bitwise qualification."""
import argparse,hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_CORE,(0,0));binary=root/'build-shift-3d/src/athena';evidence=root/'docs/elliptic-shift-3d/evidence';build=evidence/'native-static-smr-final-shared-build-20261027';m={'scope':'Final actual shared-athena qualification against previously independently validated uniform and static-SMR Minkowski controls; no new wormhole scientific claim','binary_sha256':sha(binary),'frozen_build_manifest_sha256':sha(build/'manifest.json'),'maximum_seconds_per_process':180,'maximum_active_cells':50000,'cases':[]}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 cases=[('uniform-ng3',root/'inputs/z4c/composite_minkowski.athinput',evidence/'native-composite-cfl-controls-20261027/two-shell','native_composite',2808),('static-SMR-ng4',evidence/'native-static-smr-controls-r3-20261027/native_static.athinput',evidence/'native-static-smr-controls-r3-20261027/full','native_smr',12312)]
 for name,input,baseline,prefix,cells in cases:
  folder=out/name;folder.mkdir();command=[str(binary),'-i',str(input),'-d',str(folder)];row={'name':name,'command':command,'input_sha256':sha(input),'expected_active_cells':cells,'reference':str(baseline),'files':{}};m['cases'].append(row);save();t=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid;save()
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('unowned group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  row['seconds']=time.monotonic()-t;save();assert row['exit_code']==0
  for old in sorted(baseline.glob('*.csv')):
   new=folder/old.name;row['files'][old.name]={'bitwise':old.read_bytes()==new.read_bytes(),'reference_sha256':sha(old),'shared_sha256':sha(new)}
  row['trace']=[json.loads(line) for line in (folder/(prefix+'.jsonl')).read_text().splitlines()];row['active_cells']=row['trace'][-1]['cells'];row['snapshot_CSV_count']=sum(name.startswith(prefix+'.0') for name in row['files']);row['all_six_CSVs_bitwise']=row['snapshot_CSV_count']==6 and all(r['bitwise'] for r in row['files'].values());row['input_unchanged']=sha(input)==row['input_sha256'];save();assert row['active_cells']==cells and row['all_six_CSVs_bitwise'] and row['input_unchanged'];print(name,row['exit_code'],round(row['seconds'],3),'6CSVbitwise',flush=True)
 m['binary_unchanged']=sha(binary)==m['binary_sha256'];m['qualified']=all(r['all_six_CSVs_bitwise'] for r in m['cases']) and m['binary_unchanged'];save();assert m['qualified'];print(json.dumps({'qualified':True,'shared_binary_sha256':m['binary_sha256']}))
if __name__=='__main__':main()
