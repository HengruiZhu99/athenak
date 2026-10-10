"""Capture static review inputs; no candidate import or numerical evaluation."""
from pathlib import Path
import hashlib,json,ast
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
SRC=BASE/'continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
index=SRC/'source-index.json';assert sha(index)=='92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3'
idx=load(index);recipe=load(SRC/'recipe.json')
pins={str(index):sha(index)}
for p,h in list(idx['files'].items())+list(recipe['protected_inputs'].items()):
 assert p not in pins or pins[p]==h,p
 assert sha(p)==h,p
 pins[p]=h
assert len(idx['files'])==38 and len(recipe['protected_inputs'])==192
inventory=[]
for p in sorted(SRC.glob('*.py')):
 t=p.read_text();ast.parse(t);inventory.append({'path':str(p),'bytes':p.stat().st_size,'lines':len(t.splitlines()),'sha256':sha(p)})
write(HERE/'source-pins002.json',pins)
write(HERE/'review-preparation002.json',{'metadata_passed':True,'mathematical_source_review_complete':False,'candidate_imports':False,'numeric_calls':False,'protected_inputs':len(pins),'source_index_sha256':sha(index),'inventory':inventory,'read_order':'Source index identity supplied by author; all supplied hashes verified here before v2 diff read; prior v1 full mathematical source review retained as held FAIL'})
print(json.dumps({'metadata_passed':True,'protected_inputs':len(pins),'inventory':inventory}))
