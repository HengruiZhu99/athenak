"""Read-only API build/source/dependency/binary and saved-case review."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
P=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
F=ROOT/'build-layer-research/continuum/finite-rb-constraint-rate-oracle/immutable-finite-rb-C0-constraint-rates-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(F/'index.json')=='3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f'
records=[];inputs={};production=[]
for mode,attempt in [('release','release-005'),('debug','debug-003')]:
 b=P/'build-attempts'/attempt;r=json.loads((b/'receipt.json').read_text())
 assert r['exit_code']==0 and r['sources_before']==r['sources_after']
 assert r['runtime_source_commit']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
 for name,digest in r['sources_before'].items():
  q=b/name
  if not q.exists():q=P/name
  assert sha(q)==digest,name
 dep=r['compiler_dependency_hashes']|r['link_archive_hashes']
 for name,digest in dep.items():
  q=Path(name)
  if q.parent==P and (b/q.name).is_file():q=b/q.name
  assert sha(q)==digest,name
  if name in inputs:assert inputs[name]==digest
  inputs[name]=digest
  original=Path(name)
  if original.is_relative_to(ROOT/'src'):
   relative=str(original.relative_to(ROOT));content=subprocess.check_output(['git','show',r['runtime_source_commit']+':'+relative],cwd=ROOT)
   assert hashlib.sha256(content).hexdigest()==digest
   production.append(relative)
 executable=b/('radial-bridge-'+mode)
 assert sha(executable)==r['executable_sha256']==sha(F/'executables'/executable.name)
 assert (b/'stderr').stat().st_size==0 and (b/'dependency.stderr').stat().st_size==0
 records.append({'mode':mode,'attempt':attempt,'compiler_dependencies':len(r['compiler_dependency_hashes']),'link_archives':len(r['link_archive_hashes']),'compiler_seconds':r['seconds'],'receipt_sha256':sha(b/'receipt.json'),'retained_executable_sha256':sha(executable),'frozen_executable_readback_matches':True})
start=time.monotonic();cmd=[sys.executable,str(F/'verify_frozen.py'),str(F)]
run=subprocess.run(cmd,capture_output=True,text=True)
(HERE/'root-saved-data-readback.stdout.json').write_text(run.stdout);(HERE/'root-saved-data-readback.stderr').write_text(run.stderr)
assert run.returncode==0 and not run.stderr
saved=json.loads(run.stdout);assert saved['total_cases']==7896
review={'status':'PASS_read_only_API_build_dependency_retained_binary_and_all_saved_case_arithmetic','review_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scientific_rerun_performed':False,'root_review_source_sha256':sha(__file__),'frozen_index_sha256':sha(F/'index.json'),'draft_sha256':sha(HERE/'audit-draft.md'),'builds':records,'unique_dependency_and_link_inputs':len(inputs),'compiled_production_headers_matching_27c19':sorted(set(production)),'full_saved_data_readback':{'command':cmd,'seconds':time.monotonic()-start,'exit_code':run.returncode,'stderr_bytes':0,'stdout_sha256':sha(HERE/'root-saved-data-readback.stdout.json'),'cases':7896,'files':saved['verified_index_files']},'source_math_review':'Read exact algebraic-chart core proof, physical8 API and both Cartesian FD drivers; no higher reference jets, primitive off-normal closure, integrated-energy or stability claims. Original launch provenance and Debug permission failure preserved.'}
(HERE/'root-provenance-review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':review['status'],'unique_inputs':len(inputs),'builds':records}))
