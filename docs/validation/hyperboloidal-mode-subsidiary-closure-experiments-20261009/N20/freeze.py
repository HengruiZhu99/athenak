"""Freeze the independent N20 saved-mode comparator and N16 comparison."""
from pathlib import Path
import hashlib,json,math,os,platform,shutil,subprocess,sys
import numpy as np,scipy
W=Path(__file__).resolve().parent;R=W.parents[2]
F=W/'immutable-N20-mode-subsidiary-defect-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not F.exists()
results=json.loads((W/'results.json').read_text());build=json.loads((W/'build-provenance.json').read_text());export=json.loads((W/'candidate-metadata.json').read_text())
assert sha(W/'comparator')==build['executable_sha256']
assert sha(W/'comparator.cpp')==json.loads((W/'source-review.json').read_text())['reviewed_comparator_cpp_sha256']
assert sha(W/'candidate-vectors.npz')==export['candidate_vectors_sha256']
for p,h in build['compiler_dependency_hashes'].items():assert sha(p)==h,p
for p,h in build['link_archive_hashes'].items():assert sha(p)==h,p
for p,h in results['input_sha256'].items():assert sha(p)==h,p
for p,h in export['input_sha256'].items():assert sha(p)==h,p
originals={
 R/'build-layer-research/boundary/full-tensor-global-final/manifest.json':'4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2',
 R/'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009/index.json':'396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69',
 R/'build-layer-research/continuum/constraint-propagation/immutable-constraint-propagation-20261009/manifest.json':'1ef4735a5b1136c7836daf4fc8783a24f5b83e42eca8281fa63bff8735c4040a',
 W.parent/'mode-subsidiary-defect/immutable-mode-subsidiary-defect-20261009/index.json':'0ccc0eb70207cfdc7ba14d7da156063902c0850119690215f65bc0c8b3c3321b',
 R/'build-layer-research/boundary/full-tensor-C0-N20-20261009/immutable-C0-N20-default-span-v2-20261009/index.json':'8ec4c4dd84898d19b055831696de6e3608542b1fd10a88e04f087b1f689f99aa'}
verified=[]
for p,h in originals.items():
 assert sha(p)==h,p
 for name,row in json.loads(p.read_text())['files'].items():assert sha(p.parent/name)==row['sha256'],name
 verified.append({'path':str(p),'sha256':h,'small_files_reverified':len(json.loads(p.read_text())['files'])})
prep=json.loads((W/'source-preparation.json').read_text())
P=W.parent/'mode-subsidiary-defect/immutable-mode-subsidiary-defect-20261009'
cpp=(W/'comparator.cpp').read_text()
for old,new in prep['callback_literal_only_change'].items():assert cpp.count(new)==1;cpp=cpp.replace(new,old)
assert cpp==(P/'comparator.cpp').read_text()
assert not subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=R,text=True)
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,list):
  for q in x:finite(q)
 elif isinstance(x,dict):
  for q in x.values():finite(q)
finite_count=0
for p in W.rglob('*.json'):finite(json.loads(p.read_text()));finite_count+=1
m=results['masks'];assert m['centered_active']['cells']==1088 and m['centered_Lx_active']['cells']==1064 and m['fully_nested_native']['cells']==184
assert all(x==0 for x in results['strict_vs_same_ray_max'].values())
assert max(results['C_Jv_minus_lambda_v_over_CJv'])<1e-6
for row in json.loads((W/'manufactured.json').read_text()):assert row['K_error_over_max1_scale']<1e-12 and row['Lx_correction_max']<1e-10 and row['KO_max']<1e-10
for p in [W/'diagnostic-arrays.npz',W/'candidate-vectors.npz']:
 for v in np.load(p).values():assert np.isfinite(v).all()
common=json.loads((W/'common-region-comparison.json').read_text());assert [c['cells'] for c in common['cases']]==[32,32]
for row in common['cases']:
 for p,h in row['input_sha256'].items():assert sha(p)==h,p
assert export['runtime_warnings_promoted_to_errors'] and export['phase_aligned_candidate_distance']<1e-6
assert export['candidates'][0]['singular_value_fraction_at_rank']>1e-10
verification={'passed_readback_and_source_pins':True,'sources_unchanged':True,'originals':verified,
 'compiler_dependencies_reverified':len(build['compiler_dependency_hashes']),'finite_JSON_count_before_freeze':finite_count,
 'grid_only_callback_change_verified':True,'runtime_implementation':build['runtime_implementation'],'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
 'platform':platform.platform(),'environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','PYTHONPATH']},
 'commands':[{'argv':['python3',str(W/'export_selected_mode.py')],'output':'export.log','scope':'Replay fixed prior dense reduced exports only.'},
 {'argv':['python3',str(W/'prepare_sources.py')]},{'argv':['python3',str(W/'prepare_build.py')],'output':'build-driver.log'},
 {'argv':[str(W/'comparator'),'--manufactured'],'output':'manufactured.json'},
 {'argv':['python3',str(W/'run_comparator.py')],'output':'run.log'},
 {'argv':['python3',str(W/'compare_common_region.py')],'output':'common-region.log'}],
 'claims':'Fixed-grid saved approximate-mode intertwining defect only. Different modes/grid phase forbid order. Chosen constraint extension not induced primitive closure. No new evolution/global eigensolve/matrix/native-long/production edits.'}
(W/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
large=[];small=[]
for p in sorted(W.rglob('*')):
 if not p.is_file() or p.name=='freeze.log':continue
 if p.name=='comparator' or p.suffix=='.npz':large.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'metadata_only':True})
 else:small.append(p)
external={}
for pins in [results['input_sha256'],export['input_sha256']]+[r['input_sha256'] for r in common['cases']]:
 for p,h in pins.items():
  if not str(p).startswith(str(W)+'/'):external[p]={'path':p,'bytes':Path(p).stat().st_size,'sha256':h,'metadata_only':True}
for p in [R/'build-layer-research/boundary/full-tensor-C0-N20-20261009/full22/spatialnorm-cache0.0001-lift.bin']+list(originals):
 external[str(p)]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'metadata_only':True}
F.mkdir();files={}
for p in small:
 rel=p.relative_to(W);target=F/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 files[str(rel)]={'bytes':p.stat().st_size,'sha256':sha(p)}
index={'scope':verification['claims'],'files':files,'large_artifacts_metadata_only':large,'external_comparison_records':list(external.values()),
 'count':len(files),'bytes':sum(x['bytes'] for x in files.values())}
(F/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
for name,row in files.items():assert sha(F/name)==row['sha256'],name
for p in F.rglob('*'):
 if p.is_file():p.chmod(0o444)
print('FROZEN',str(F/'index.json'),sha(F/'index.json'),len(files),index['bytes'],len(large),len(external),flush=True)
