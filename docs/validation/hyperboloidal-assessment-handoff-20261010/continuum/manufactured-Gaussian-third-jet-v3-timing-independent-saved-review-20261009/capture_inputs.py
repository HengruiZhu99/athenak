#!/usr/bin/env python3
"""Freeze compact reviewer inputs before reading their scientific content."""
from pathlib import Path
import hashlib, json, os, shutil, sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OWNER=BASE/'continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009'
ROOT=BASE/'Gaussian-third-jet-oracle-v3-timing-root-release-20261009'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def check(v,s):
    if not v: raise ValueError(s)
check(sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 and sys.flags.optimize==0,'stdlib isolated nonoptimized invocation required')
check(os.environ.get('PYTHONOPTIMIZE')=='0','explicit optimization guard')
expected={OWNER/'source-index.json':'41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a',OWNER/'recipe.json':'847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf',OWNER/'attempts/timing001/receipt.json':'b5d3a0263dfeb536a9a71b8263b46e61b7d26b124ed6620bcc85e9b6ea939e44',OWNER/'attempts/timing001/result.json':'e9efd42805d6eca7d3e537cae219b22049a1063056c4b9abda3e0b225ebd6139',ROOT/'timing-outer001/receipt.json':'cef24693f5090354d0aa557871fbacd5fba2bf626749983d1c1fa2edadcca018'}
for p,h in expected.items(): check(sha(p)==h,'root-reported exact input '+str(p))
paths=set(ROOT.rglob('*'))
paths={p for p in paths if p.is_file()}
for name in ('source-index.json','recipe.json','run_oracle.py','outer_once.py','PLAN.md','SCHEMA.md','measured-timing-review-schema.json','diagnostics.py','oracle.py','reference3.py','geometry.py'):
    paths.add(OWNER/name)
for name in ('receipt.json','result.json','source-before.json','source-after.json'):
    paths.add(OWNER/'attempts/timing001'/name)
for pref in ('manufactured-Gaussian-third-jet-v3-independent-source-review-20261009','manufactured-Gaussian-third-jet-v3-units-independent-saved-review-20261009','manufactured-Gaussian-third-jet-v2-timing-failure-independent-saved-review-20261009'):
    for name in ('index.json','receipt.json','saved-readback.json','REVIEW.md'):
        p=BASE/'continuum'/pref/name
        if p.exists(): paths.add(p)
(HERE/'inputs').mkdir(exist_ok=False)
copies=[]; external=[]
for n,p in enumerate(sorted(paths)):
    check(p.is_file(),'input file absent')
    row={'origin':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    check(p.suffix not in ('.npz','.npy','.jsonl') and row['bytes']<=1048576,'unexpected compact input '+str(p))
    target=HERE/'inputs'/('%03d-'%n+p.parent.name+'-'+p.name)
    shutil.copyfile(p,target)
    check(sha(target)==row['sha256'],'copy drift')
    row['copy']=str(target.relative_to(HERE));copies.append(row)
for name in ('oracle.jsonl','precision-checks.jsonl','height-context.json'):
    p=OWNER/'attempts/timing001'/name
    external.append({'origin':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'scope':'metadata only; content not decoded'})
with (HERE/'capture.json').open('x') as f:
    json.dump({'compact_inputs_frozen_before_content_read':True,'inputs':copies,'external_metadata_only':external,'invocation':{'argv':sys.orig_argv if hasattr(sys,'orig_argv') else ['python3','-I','-B',str(Path(__file__).resolve())],'executable':sys.executable,'flags':{'isolated':sys.flags.isolated,'dont_write_bytecode':sys.flags.dont_write_bytecode,'optimize':sys.flags.optimize},'environment':{k:os.environ.get(k) for k in ('PYTHONOPTIMIZE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}},f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'compact_inputs':len(copies),'external_metadata_only':len(external)}))
