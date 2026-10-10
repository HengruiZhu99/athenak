#!/usr/bin/env python3
"""Capture source and metadata only before independent content review."""
from pathlib import Path
import hashlib,json,os,shutil,sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
OWNER=BASE/'continuum/exact-rational-backend-source002-held-20261009'
ROOT=BASE/'exact-rational-backend-source002-root-release-20261009'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(v,s):
    if not v:raise ValueError(s)
def load(p):return json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
require(sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE')=='0','isolated stdlib source capture')
require(sha(OWNER/'source-index.json')=='ee41e069eee322e93d126699ec5a6251444fc843394c2c8f5d324de8f54ed5e9','owner exact index')
require(sha(OWNER/'recipe.json')=='2835509160432e29e6608c81591c44cd3420e03e1b32a0c7fa44336a10f30289','owner exact recipe')
paths={OWNER/'source-index.json',ROOT/'source-index.json'}
for index in (OWNER/'source-index.json',ROOT/'source-index.json'):
    for row in load(index)['files']:
        p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'indexed input drift')
        paths.add(p)
(HERE/'inputs').mkdir(exist_ok=False);copied=[]
for n,p in enumerate(sorted(paths)):
    require(p.stat().st_size<=1048576 and p.suffix not in ('.jsonl','.npz','.npy'),'compact source input')
    rel=(str(p.relative_to(OWNER)) if OWNER in p.parents else 'root/'+str(p.relative_to(ROOT)))
    target=HERE/'inputs'/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,target)
    require(sha(p)==sha(target),'byte-preserving capture')
    copied.append({'origin':str(p),'copy':str(target.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
with (HERE/'capture.json').open('x') as f:json.dump({'source_inputs_frozen_before_content_read':True,'owner_source_index_sha256':sha(OWNER/'source-index.json'),'root_source_index_sha256':sha(ROOT/'source-index.json'),'inputs':copied,'candidate_imports_or_scientific_evaluation':False},f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'copies':len(copied),'root_index_sha256':sha(ROOT/'source-index.json')}))
