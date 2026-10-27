#!/usr/bin/env python3
"""Run a diagnostic/analysis from an immutable source and optional binary copy.

Numerical evolution matrices and verified continuations already freeze their
own closures. This launcher extends that discipline to standalone analyses.
"""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--library',type=Path);p.add_argument('--fast-library',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
repo=Path(__file__).resolve().parents[2];script=a.script.resolve();relative=script.relative_to(repo)
if script.suffix!='.py' or script.parent!=Path(__file__).parent:raise ValueError('Select a repository spherical diagnostic/analysis script')
output=a.output.resolve();bundle=output.with_name(output.name+'-source')
if output.exists() or bundle.exists():raise ValueError('Never overwrite an analysis or its source bundle')
bundle.mkdir(parents=True);hashes={}
files=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cpp'),
       *Path(repo/'src/z4c/hyperboloidal').glob('*.hpp'),*Path(repo/'docs/elliptic-shift-1d').glob('*.json')]
for source in files:
 target=bundle/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 hashes[str(source.relative_to(repo))]=hashlib.sha256(target.read_bytes()).hexdigest()
command=[sys.executable,str(bundle/relative),'--output',str(output)]
for key in ['library','fast_library']:
 source=getattr(a,key)
 if source is None:continue
 source=source.resolve();target=bundle/'runtime'/source.name;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
 hashes['runtime/'+source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
 command+=['--'+key.replace('_','-'),str(target)]
extra=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
if '--output' in extra:raise ValueError('Use the launcher output argument once')
command+=extra
receipt=dict(pid=os.getpid(),source_sha256=hashes,command=command,cwd=str(repo),python=sys.version,
             numpy=np.__version__,platform=platform.platform(),long_double_epsilon=float(np.finfo(np.longdouble).eps),status='running')
path=bundle/'launch.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
result=subprocess.run(command,cwd=repo);receipt.update(status='finished' if result.returncode==0 else 'failed',returncode=result.returncode)
path.write_text(json.dumps(receipt,indent=2)+'\n');sys.exit(result.returncode)
