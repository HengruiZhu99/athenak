"""Freeze the reused p1 wide negative screen, with exact compiled provenance."""
from pathlib import Path
import hashlib,json,math,shlex,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];out=w/'immutable-flat-penrose-wide-screen-20261009';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
prep=read(w/'preparation.json')
for name,h in prep['inputs'].items():assert sha(root/name)==h,name
for name,h in prep['generated'].items():assert sha(w/name)==h,name
assert sha(w/'frozen-flat_power.hpp')=='7d3c22bb5e23627a5da83d542fd4100d109fad43612ee3f995c223bd50df1135'
local=read(w/'local-build-receipt.json');tangent=read(w/'tangent-results.json');checks=[]
# The radial exporter used the same saved Release command; capture its complete
# compiler dependency list now without recompiling or modifying its executable.
row=local['builds'][-1];cmd=row['command'];source=str(w/'export_radial.cpp');prefix=cmd[:cmd.index(source)];dep=subprocess.run(prefix+[source,'-M','-MT','audit'],cwd=root,capture_output=True,text=True);assert dep.returncode==0
(w/'export-dependencies.make').write_text(dep.stdout);paths=shlex.split(dep.stdout.replace('\\\n',' ').split(':',1)[1]);row['compiler_dependency_hashes']={str(Path(k).resolve()):sha(Path(k)) for k in paths}
row['link_archives']={str(root/f'build-layer-release/kokkos/{k}/src/libkokkos{k}.a'):sha(root/f'build-layer-release/kokkos/{k}/src/libkokkos{k}.a') for k in ['containers','algorithms','core','simd']}
(w/'local-build-receipt.json').write_text(json.dumps(local,indent=2)+'\n')
for row in local['builds']+tangent['builds']:
 cmd=row['command'];exe=Path(cmd[cmd.index('-o')+1]);assert sha(exe)==row['executable_sha256']
 for name,h in row['compiler_dependency_hashes'].items():assert sha(Path(name))==h,name
 for name,h in row['link_archives'].items():assert sha(Path(name))==h,name
 checks.append({'executable':str(exe),'sha256':sha(exe),'compiler_dependency_count':len(row['compiler_dependency_hashes']),'archives':len(row['link_archives']),'unchanged':True})
assert all(row['exit_status']==0 for row in local['tests'])
oracle=read(w/'oracle-wide.json');assert oracle['checked_scalar_values']==1936 and oracle['derivative_survives_boost_underflow']
assert sha(w/'oracle_wide.py')==oracle['source_sha256'] and sha(w/'export-radial')==oracle['export_executable_sha256']
assert sha(w/'constraint_tangent.cpp')==tangent['source_sha256']
assert all(row['flat_over_original']['Hdot_rms']>2 for row in tangent['summary'])
assert subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''
identity={'scope':'Previously rejected exact exponent-one family, fresh wide local/instantaneous supplement, stopped before global/native evolution',
 'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','launch_HEAD':prep['head'],
 'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'prototype_sha256':sha(w/'frozen-flat_power.hpp'),'overlay_sha256':sha(w/'overlay/z4c/hyperboloidal/layer_reference.hpp'),
 'compiled_identity_checks':checks,'local_release_and_asan_pass':True,'independent_100digit_oracle_pass':True,
 'native_instantaneous_source_worsens':True,'no_global_native_or_production_change':True,
 'independent_math_review_index_sha256':'0925f36f411bf934342f24ce76876f9d95232de5309bb665590cb2705442bbf6',
 'oracle_summary_correction':'First optional normalized-error denominator corrected; original source/output/log retained, raw comparisons/gates unchanged.'}
(w/'source-identity-verification.json').write_text(json.dumps(identity,indent=2)+'\n')
out.mkdir()
for p in w.iterdir():
 if p.is_file() and p.suffix in ['.py','.hpp','.cpp','.json','.jsonl','.log','.stderr','.md','.txt','.make']:shutil.copy2(p,out/p.name)
shutil.copytree(w/'overlay',out/'overlay');shutil.copytree(w/'first-oracle-summary-normalization',out/'first-oracle-summary-normalization')
large={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size,'copied':False} for p in [w/'local-gate-release',w/'local-gate-asan',w/'export-radial',w/'tangent-original',w/'tangent-flat']}
(out/'executables-metadata-only.json').write_text(json.dumps(large,indent=2)+'\n')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for p in out.rglob('*.json'):finite(read(p))
files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
(out/'index.json').write_text(json.dumps({'scope':identity['scope'],'files':files},indent=2)+'\n')
for name,row in files.items():assert sha(out/name)==row['sha256']
print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'finite_json':len(list(out.rglob('*.json'))),'index_sha256':sha(out/'index.json'),'tangent_results_sha256':sha(out/'tangent-results.json'),'REPORT_sha256':sha(out/'REPORT.md')},indent=2))
