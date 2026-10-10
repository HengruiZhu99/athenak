"""Narrow root source002 driver correction review, no scientific execution."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
OLD=REPO/'build-layer-research/continuum/exact-signed-product-sum-source001-held-20261009'
SRC=REPO/'build-layer-research/continuum/exact-signed-product-sum-source002-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
assert sha(SRC/'source-index.json')=='36c72f687bad50fa61c3f1ac684cf5b9b28cfa99348e71e627ea60bad5489a09'
assert sha(SRC/'recipe.json')=='8199914356681f54022745c40ec46513a0eb806cf6937c15d8641f770ae2a986'
assert load(HERE/'source001-review.json')['source_math_algorithm_review_passed']
pins={r['path']:r['sha256'] for r in load(SRC/'source-index.json')['files']+load(SRC/'external-pins.json')}
pins[str(SRC/'source-index.json')]=sha(SRC/'source-index.json')
for p,h in load(HERE/'source001-pins.json').items():pins[p]=h
for p,h in pins.items():assert sha(p)==h,p
proof=load(SRC/'driver-reverse-proof.json')
for row in proof['unchanged_files']:assert (SRC/row['name']).read_bytes()==(OLD/row['name']).read_bytes() and sha(SRC/row['name'])==row['sha256']
new=(SRC/'run_gate.py').read_text();old=(OLD/'run_gate.py').read_text()
assert new.count(proof['added_guard'])==1 and new.replace(proof['added_guard'],'')==old
assert ast.dump(ast.parse(new.replace(proof['added_guard'],'')))==ast.dump(ast.parse(old))
r=load(SRC/'recipe.json');prior=load(OLD/'recipe.json')
for k in prior:
 if k not in ('compiler','local_sources'):assert r[k]==prior[k],k
c=r['compiler'];assert c['path']=='/Library/Developer/CommandLineTools/usr/bin/clang++' and str(Path(c['path']).resolve())==c['resolved_path']
for key in ('path','resolved_path'):
 p=c[key];expected=c['sha256' if key=='path' else 'resolved_sha256'];assert sha(p)==expected and pins[p]==expected
review=load(HERE/'source001-review.json')
review.update(passed_source_review=True,source_math_algorithm_review_passed=True,execution_released=False,reviewed_source_index_sha256=sha(SRC/'source-index.json'),recipe_sha256=sha(SRC/'recipe.json'),protected_paths=len(pins),finding=None,
 source001_review_sha256=sha(HERE/'source001-review.json'),source001_mechanical_finding_preserved=True,
 narrow_correction='Only literal clang++ invocation and resolved target/content guards, recipe/external/history metadata; all arithmetic/probe/oracle/70controls unchanged. Exact reverse bytes and AST verified.',
 compiler_invocation_path=c['path'],compiler_resolved_path=c['resolved_path'])
write(HERE/'source002-pins.json',pins);write(HERE/'source002-review.json',review)
print(json.dumps({'source_review_passed':True,'execution_released':False,'pins':len(pins),'review_sha256':sha(HERE/'source002-review.json')}))
