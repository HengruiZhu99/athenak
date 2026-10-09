"""Hash and saved local algebra receipt readback only; never compiles or queries."""
import argparse,hashlib,json,math,pathlib
parser=argparse.ArgumentParser();parser.add_argument('capsule');parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args();P=pathlib.Path(args.capsule);idx=json.loads((P/'index.json').read_text());checked=0;skipped=[]
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for e in idx['files']:
 p=P/e['path']
 if not p.exists() and args.metadata_only and e['role']=='large_payload':skipped.append(e['path']);continue
 assert p.exists(),e['path'];b=p.read_bytes();assert len(b)==e['bytes'];assert hashlib.sha256(b).hexdigest()==e['sha256'];checked+=1
 if p.suffix=='.json':finite(json.loads(b))
A=P/idx['accepted_attempt'];r=json.loads((A/'receipt.json').read_text());recipe=json.loads((A/'release-recipe.json').read_text());out=json.loads((A/'release.json').read_text())
assert r['passed_local_gate'] and r['sources_unchanged'] and r['release_debug_equal'] and not r['operators_or_evolution_run']
assert out==json.loads((A/'debug.json').read_text())
assert all(c['returncode']==0 for c in r['commands']);assert all(c['stderr_sha256']==hashlib.sha256(b'').hexdigest()for c in r['commands'])
assert r['source_before']==r['source_after']==recipe['inputs'];assert len(recipe['inputs'])==374
for k,tol in recipe['thresholds'].items():
 v=out[k];v=v[-1]if isinstance(v,list)else v;assert math.isfinite(v)and v<=tol,k
assert out['factored_reference_max']==0
print(json.dumps({'passed':True,'checked_files':checked,'skipped_large_payload':skipped,'scope':'saved hashes/finiteJSON/declared thresholds; no kernel query, compilation, spectrum or evolution'},indent=2))
