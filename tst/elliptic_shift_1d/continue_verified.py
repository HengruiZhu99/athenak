#!/usr/bin/env python3
"""Continue an exact source/library-bound checkpoint using a preserved source copy.

Copy only files in the parent's source manifest, verifying each hash. This avoids
invalidating a trajectory merely because unrelated analysis scripts were added.
The numerical source and library must match byte for byte; no compatibility bypass.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--t',type=float,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[2];a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve()
with np.load(a.checkpoint,allow_pickle=False) as z:
 parent=json.loads(str(z['metadata']));parent_t=float(z['t']);q=z['q']
 n=parent['arguments']['n'];r=z['r'] if 'r' in z else (np.arange(n)+.5)/(n-.5)
 if (parent.get('status')!='finished' or not np.all(np.isfinite(q)) or
     min(np.min(1+q[:,0]),np.min(1+q[:,1]),np.min((1+r*r)/2+q[:,6]))<=0):
  raise SystemExit('Checkpoint is not a completed finite positive state')
if parent_t>=a.t:raise SystemExit('No unfinished continuation segment')
if a.output.exists():raise SystemExit('Never overwrite an existing continuation')
bundle=a.output.with_name(a.output.name+'-source');bundle.mkdir(parents=True,exist_ok=False)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
original_script=Path(parent['command'][0])
source_repo=original_script.parents[2] if original_script.is_absolute() and original_script.name=='evolve.py' else repo
if (a.checkpoint.parent/'frozen-source').is_dir():source_repo=a.checkpoint.parent/'frozen-source'
for name,digest in parent['binding']['source_sha256'].items():
 src=source_repo/name
 if not src.is_file() or sha(src)!=digest:raise SystemExit('Source differs from saved trajectory: '+name)
 dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
library=Path(parent['arguments']['library']);library=library if library.is_absolute() else repo/library
candidates=[library,source_repo/'runtime'/library.name]
library=next((f for f in candidates if f.is_file() and sha(f)==parent['binding']['executable_sha256']),None)
if library is None:raise SystemExit('No exact saved library is available')
runtime=bundle/'runtime';runtime.mkdir();saved_library=runtime/library.name;shutil.copy2(library,saved_library);library=saved_library
cmd=[sys.executable,str(bundle/'tst/elliptic_shift_1d/evolve.py'),'--library',str(library),'--output',str(a.output),'--resume',str(a.checkpoint),'--t',str(a.t)]
for key,value in parent['arguments'].items():
 if key in ('library','output','resume','t') or value is None:continue
 if isinstance(value,bool):
  if value:cmd+=['--'+key.replace('_','-')]
  continue
 cmd+=['--'+key.replace('_','-'),str(value)]
receipt={'parent_checkpoint':str(a.checkpoint),'parent_sha256':sha(a.checkpoint),'parent_time':parent_t,'target_time':a.t,'verified_binding':parent['binding'],'command':cmd,'source_copy':str(bundle)}
(bundle/'continuation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
result=subprocess.run(cmd,cwd=repo);sys.exit(result.returncode)
