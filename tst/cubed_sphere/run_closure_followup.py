#!/usr/bin/env python3
"""Prepare/run restart and ghost-independence controls after six-case admission."""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('parent',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
summary=json.loads((a.parent/'summary.json').read_text());digest=hashlib.sha256(a.binary.read_bytes()).hexdigest();assert digest==summary['binary_sha256']
for r in summary['cases'].values():assert r['reached_step_target'] and r['reference_unchanged'] and r['exit_code']==0
base=next(g for g in summary['grids'] if g['name']=='pulse-interior');grids=[]
for name in ['resumed-eight','fresh-eight','ghost-degree5']:
 g=copy.deepcopy(base);g['name']=name;g['device']=str(len(grids))+'.0';g['extra']['cubed_sphere/conformal_probe_steps']=4 if name=='ghost-degree5' else 8
 if name=='resumed-eight':
  payload=summary['cases']['pulse-interior']['checkpoint_payload'];g['restart']=str((a.parent/'pulse-interior/probe_checkpoint.bin').resolve());g['restart_sha256']=payload['sha256']
 if name=='ghost-degree5':g['degree']=5
 grids.append(g)
plan=a.output.with_suffix('.grids.json');assert not plan.exists() and not a.output.exists();plan.write_text(json.dumps(grids,indent=2)+'\n')
subprocess.run([sys.executable,str(a.source/'tst/cubed_sphere/run_balanced_hyperboloidal.py'),str(a.binary),str(a.source),str(a.output),'--phase','pulse','--grids',str(plan),'--timeout','300','--tlim','.001'],check=True)
