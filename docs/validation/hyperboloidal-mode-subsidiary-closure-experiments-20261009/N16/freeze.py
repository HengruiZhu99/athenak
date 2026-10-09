"""Readback/hash freeze of this comparator and all original source pins."""
from pathlib import Path
import hashlib,json,math,os,platform,shutil,subprocess,sys
import numpy as np,scipy
W=Path(__file__).resolve().parent;R=W.parents[2]
F=W/'immutable-mode-subsidiary-defect-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not F.exists()
results=json.loads((W/'results.json').read_text());build=json.loads((W/'build-provenance.json').read_text())
assert sha(W/'comparator')==build['executable_sha256']
assert sha(W/'comparator.cpp')==json.loads((W/'source-review.json').read_text())['reviewed_comparator_cpp_sha256']
for p,h in build['compiler_dependency_hashes'].items():assert sha(p)==h,p
for p,h in build['link_archive_hashes'].items():assert sha(p)==h,p
for p,h in results['input_sha256'].items():assert sha(p)==h,p
originals=[R/'build-layer-research/boundary/full-tensor-global-final/manifest.json',
 R/'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009/index.json',
 R/'build-layer-research/continuum/constraint-propagation/immutable-constraint-propagation-20261009/manifest.json']
original_pins=['4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2',
 '396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69',
 '1ef4735a5b1136c7836daf4fc8783a24f5b83e42eca8281fa63bff8735c4040a']
verified=[]
for p,h in zip(originals,original_pins):
 assert sha(p)==h,p
 for name,row in json.loads(p.read_text())['files'].items():assert sha(p.parent/name)==row['sha256'],name
 verified.append({'path':str(p),'sha256':h,'small_files_reverified':len(json.loads(p.read_text())['files'])})
assert not subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=R,text=True)
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,list):
  for q in x:finite(q)
 elif isinstance(x,dict):
  for q in x.values():finite(q)
finite_count=0
for p in W.rglob('*.json'):
 finite(json.loads(p.read_text()));finite_count+=1
m=results['masks'];assert m['centered_active']['cells']==432 and m['centered_Lx_active']['cells']==408 and m['fully_nested_native']['cells']==32
assert all(x==0 for x in results['strict_vs_same_ray_max'].values())
assert max(results['C_Jv_minus_lambda_v_over_CJv'])<2e-7
for row in json.loads((W/'manufactured.json').read_text()):assert row['K_error_over_max1_scale']<1e-12 and row['Lx_correction_max']<1e-10 and row['KO_max']<1e-10
for v in np.load(W/'diagnostic-arrays.npz').values():assert np.isfinite(v).all()
verification={'passed_readback_and_source_pins':True,'sources_unchanged':True,'originals':verified,
 'compiler_dependencies_reverified':len(build['compiler_dependency_hashes']),'finite_JSON_count_before_freeze':finite_count,
 'runtime_implementation':build['runtime_implementation'],'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
 'platform':platform.platform(),'environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','PYTHONPATH']},
 'commands':[{'argv':['python3',str(W/'prepare_build.py')],'output':'build.log'},
 {'argv':[str(W/'comparator'),'--manufactured'],'output':'manufactured.json'},
 {'argv':['python3',str(W/'run_comparator.py')],'environment':{'OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')},'output':'run.log'}],
 'claims':'Sampled fixed-grid intertwining defect only; chosen same-ray constraint extension is not induced primitive closure. No propagation/eigensolve/native-long/production edits or global/continuum stability claims.'}
(W/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
large=[];small=[]
for p in sorted(W.rglob('*')):
 if not p.is_file() or p.name=='freeze.log':continue
 rel=p.relative_to(W)
 if p.name=='comparator' or p.suffix=='.npz':large.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'metadata_only':True})
 else:small.append(p)
external=[]
for p,h in results['input_sha256'].items():external.append({'path':p,'bytes':Path(p).stat().st_size,'sha256':h,'metadata_only':True})
for p in [R/'build-layer-research/boundary/full-tensor-propagator/full22-v2/spatialnorm-cache0.0001-lift.bin']+originals:
 external.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'metadata_only':True})
F.mkdir();files={}
for p in small:
 rel=p.relative_to(W);target=F/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 files[str(rel)]={'bytes':p.stat().st_size,'sha256':sha(p)}
index={'scope':verification['claims'],'files':files,'large_artifacts_metadata_only':large,'external_comparison_records':external,
 'count':len(files),'bytes':sum(x['bytes'] for x in files.values())}
(F/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
for name,row in files.items():assert sha(F/name)==row['sha256'],name
for p in F.rglob('*'):
 if p.is_file():p.chmod(0o444)
print('FROZEN',str(F/'index.json'),sha(F/'index.json'),len(files),index['bytes'],len(large),len(external),flush=True)
