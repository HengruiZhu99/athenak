import argparse,hashlib,json,math,pathlib
p=argparse.ArgumentParser();p.add_argument('capsule');p.add_argument('--metadata-only',action='store_true');a=p.parse_args();P=pathlib.Path(a.capsule);index=json.loads((P/'index.json').read_text());skip=[]
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for f in index['files']:
 q=P/f['path']
 if not q.exists()and a.metadata_only and f['role']=='large_payload':skip.append(f['path']);continue
 b=q.read_bytes();assert len(b)==f['bytes'];assert hashlib.sha256(b).hexdigest()==f['sha256']
 if q.suffix=='.json':finite(json.loads(b))
A=P/index['accepted_attempt'];r=json.loads((A/'receipt.json').read_text());assert r['passed_AB']and r['sources_unchanged']and r['release_debug_numeric_JSON_equal']and not r['C_queries_or_evolution_run'];assert len(r['source_before'])==375
assert all(c['returncode']==0 and c['stderr_sha256']==hashlib.sha256(b'').hexdigest()for c in r['commands'])
for m in ['release','debug']:
 q=json.loads((A/('check-principal-'+m+'.json')).read_text());assert q['passed']and q['actual_principal_cases']==792 and q['exact_projector_ranks']==[10,10]
 assert q['matrix_expected_max_absolute']<=2e-12 and q['actual_involution_max_absolute']<=2e-11 and q['actual_symmetrizer_max_absolute']<=2e-11 and q['normal_max_scaled']<=5e-11
 c=json.loads((A/('core-'+m+'.json')).read_text());assert c['core_cases']==204 and c['witnesses']==17
 assert all(v<=5e-11 for k,v in c.items()if k not in ['core_cases','witnesses'])
print(json.dumps({'passed':True,'checked_or_declared_files':len(index['files']),'skipped_large_payload':skip,'scope':'saved hashes/finiteJSON/proof receipts/thresholds only; no compilation, matrices, eigensolve or evolution'},indent=2))
