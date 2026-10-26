#!/usr/bin/env python3
"""Execute an explicit input matrix sequentially, preserving every outcome."""
import argparse,json,subprocess,sys,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--fast-library')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
data=json.loads(a.input.read_text());(a.output/'input.json').write_text(a.input.read_text());receipts=[]
# Freeze once per matrix: edits made during a long matrix cannot silently
# change the equations/operators between its resolution or parameter cases.
repo=Path(__file__).resolve().parents[2];bundle=a.output/'matrix-source'
files=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cpp'),
       *Path(repo/'src/z4c/hyperboloidal').glob('*.hpp')]
hashes={}
for source in files:
    relative=source.relative_to(repo);target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target);hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
runtime=bundle/'runtime';runtime.mkdir()
for key in ['library','fast_library']:
    value=getattr(a,key)
    if value:
        source=Path(value).resolve();target=runtime/source.name;shutil.copy2(source,target)
        hashes['runtime/'+source.name]=hashlib.sha256(target.read_bytes()).hexdigest();setattr(a,key,str(target))
(a.output/'matrix-binding.json').write_text(json.dumps(hashes,indent=2)+'\n')
for case in data['cases']:
    options={**data['common'],**case['arguments']}
    command=[sys.executable,str(bundle/'tst/elliptic_shift_1d/evolve.py'),
        '--library',a.library,'--output',str(a.output/case['name'])]
    if a.fast_library and not options.get('puncture_chart'):command+=['--fast-library',a.fast_library]
    for key,value in options.items():
        if isinstance(value,bool):
            if value:command+=['--'+key.replace('_','-')]
        elif value is not None:command+=['--'+key.replace('_','-'),str(value)]
    with (a.output/(case['name']+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipts.append(dict(name=case['name'],command=command,returncode=result.returncode))
    (a.output/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts[-1]),flush=True)
