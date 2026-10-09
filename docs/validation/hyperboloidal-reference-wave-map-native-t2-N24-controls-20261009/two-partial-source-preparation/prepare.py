"""Standard-library-only metadata preparation; never decode native outputs or import scientific packages."""
from pathlib import Path
import ast, hashlib, json, subprocess, sysconfig, time
R=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=R/'build-layer-research/reference-wave-map-partial-N24-held-20261009'
OBSERVER='c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3'
TEMPLATE='9228f05a83abbb3f0e6c047946ced7d8ffd37509cae64f058d431780627b80d6'
PYTHON='/Library/Developer/CommandLineTools/usr/bin/python3'
CASES=[('wave-map-half-N24-large-t2','reference-wave-map-partial-half-N24-held-20261009',.79280598958281545),('wave-map-N24-small-t2','reference-wave-map-partial-small-N24-held-20261009',1.0180338541661471)]
cache={}
def sha(p):
 p=Path(p).resolve();key=(str(p),p.stat().st_mtime_ns,p.stat().st_size)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for s in iter(lambda:f.read(1048576),b''):h.update(s)
  cache[key]=h.hexdigest()
 return cache[key]
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def checked(d):
 for p,s in d.items():assert sha(p)==s,p
assert sha(OLD/'observe_partial.py')==OBSERVER
assert sha(OLD/'recipe.json')==TEMPLATE
old=load(OLD/'recipe.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
# File inventory only. No import of numpy, restart reader, analyzer, or observer.
runtime={};runtime_roots=[Path(sysconfig.get_paths()['stdlib']),R/'build-layer-research/boundary/python-deps/numpy',R/'build-layer-research/boundary/python-deps/numpy-2.0.2.dist-info']
for root in runtime_roots:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and 'site-packages' not in p.relative_to(root).parts and p.suffix not in ['.pyc','.pyo']:
   runtime[str(p.resolve())]=sha(p)
runtime[PYTHON]=sha(PYTHON)
runner='''"""Held invocation wrapper. Requires exact root authorization path and SHA on CLI."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\\n')
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
A=Path(sys.argv[1]).resolve();expected_auth=sys.argv[2]
OUT=P/'invocations001';OUT.mkdir(exist_ok=False)
record={'partial_diagnostic_only':True,'accepted_native_run':False,'observer_completed':False,'outer_wrapper_completed':False,'scope':'Exactly one original failed N24 process; no native advance or completed analyzer.'}
started=time.monotonic();protected={str(Path(__file__).resolve()):sha(__file__),str(A):sha(A)}
try:
 assert sha(A)==expected_auth,'root authorization SHA differs'
 idx=load(P/'source-index.json')
 for item in idx['files']:
  assert sha(item['path'])==item['sha256'],item['path'];protected[item['path']]=item['sha256']
 Q=P/'recipe.json';S=P/'observe_partial.py';q=load(Q);a=load(A)
 assert a['partial_native_snapshot_observation_authorized'] is True
 assert a['observer_sha256']==sha(S) and a['recipe_sha256']==sha(Q)
 assert set(a['cases'])==set(q['cases']) and len(q['cases'])==1
 name=next(iter(q['cases']));protected.update(q['fixed_pins'])
 for p,s in protected.items():assert sha(p)==s,p
 dump(OUT/'pins-before.json',protected)
 python='/Library/Developer/CommandLineTools/usr/bin/python3'
 overrides={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')}
 env=os.environ.copy();env.update(overrides)
 command=[python,'-B',str(S),str(A),name]
 context={'command':command,'cwd':str(R),'environment_overrides':overrides,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'case':name,'root_release_sha256':sha(A),'scope':record['scope']}
 dump(OUT/'invocation-context.json',context);(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
 so=OUT/'observer.stdout';se=OUT/'observer.stderr'
 with so.open('wb') as f,se.open('wb') as g:result=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=g)
 record.update(context,returncode=result.returncode,stdout=pin if False else str(so),stdout_sha256=sha(so),stderr=str(se),stderr_sha256=sha(se))
 for p,s in protected.items():assert sha(p)==s,p
 dump(OUT/'pins-after.json',protected)
 record.update(pins_unchanged=True,outer_wrapper_completed=True,observer_completed=result.returncode==0)
except Exception as exc:record['protocol_error']=repr(exc)
record['seconds']=time.monotonic()-started;dump(OUT/'receipt.json',record);print(json.dumps(record),flush=True)
assert record['outer_wrapper_completed'] and record.get('returncode')==0,'partial diagnostic wrapper stopped; never native acceptance'
'''
# This odd-looking expression is not needed; preserve a clean standard-library source before pinning.
runner=runner.replace("stdout=pin if False else str(so)","stdout=str(so)")
prepared=[]
for case,prefix,abort_time in CASES:
 P=R/'build-layer-research'/prefix;P.mkdir(exist_ok=False)
 failed=R/'build-layer-research/wave-map-native-t2-root-20261009/batch001'/case/'launch-receipt.json';f=load(failed)
 assert f['returncode']==-6 and f['passed_native_process_and_provenance'] is False and f['sources_before_after_equal'] is True
 assert f['native_execution_authorization_sha256']==old['native_execution_release_sha256']
 b=load(f['build_receipt']);assert b['passed_compile_link'] and b['compiled_implementation']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
 assert sha(f['build_receipt'])==f['build_receipt_sha256'] and sha(f['executable'])==f['executable_sha256'] and sha(f['input_path'])==f['input_sha256']
 assert f['executable']==b['executable'] and f['executable_sha256']==b['executable_sha256']
 (P/'observe_partial.py').write_bytes((OLD/'observe_partial.py').read_bytes());assert sha(P/'observe_partial.py')==OBSERVER
 (P/'run_released_observation001.py').write_text(runner)
 (P/'release-schema.json').write_bytes((OLD/'release-schema.json').read_bytes())
 ctx=P/'source-context';ctx.mkdir()
 copies={}
 for name,source in {'template-recipe.json':OLD/'recipe.json','template-observer.py':OLD/'observe_partial.py','original-failed-launch-receipt.json':failed,'exact-input.athinput':Path(f['input_path'])}.items():
  dst=ctx/name;dst.write_bytes(source.read_bytes());copies[str(dst)]={'source':str(source),'sha256':sha(source)}
 plan=f'''# Held partial observations of {case}\n\nThis fresh source-only prefix names exactly the original failed process {case}. The copied observer is byte-exact c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3; the accepted probe, reader, ABI, formulas and thresholds are unchanged. No scientific module is imported by preparation, and no native array/history/log content is interpreted. File bytes are hashed only for preservation. No scientific attempt has run here.\n\nThe original process returned -6. Root reports its unsaved abort at t={abort_time:.17g}, at the same xyz/Omega as the earlier wave-map-N24-large-t2 failure. This root-supplied failure time is context, not a state reconstructed from prior snapshots. Original target t=2 and the failed launch receipt remain authoritative. Saved RSTs precede the abort and cannot certify its unsaved state.\n\nAll 29 prior shared pins remain context inputs. Additional pins cover this exact failed launch, full original output inventory, native source/build/compiler dependencies and original protected-input inventories; accepted probe dependencies; interpreter, standard library and NumPy candidate runtime file inventory. Runtime files are inventoried without importing them. Input/build/launch copies are source context only. The case spec is taken from its own failed launch, including mode {f['mode']} and its exact executable.\n\nExecution is HELD. Root must supply a release binding this recipe, exact copied observer and failed launch receipt, with a fresh direct child of this prefix/attempts. The held command is Python -B run_released_observation001.py ROOT_AUTHORIZATION_PATH ROOT_AUTHORIZATION_SHA. Use PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1; the wrapper records full command/env/outer logs/return code and pre/post protected pins. It invokes only the unchanged partial observer and accepted snapshot probe. It cannot invoke the completed-result analyzer or advance native evolution.\n\nEvery result remains partial_diagnostic_only=true and accepted_native_run=false. Observer completion means bookkeeping completion, never native-run completion, repaired-state admission, future preservation, or stability acceptance. Diagnostic guard failures and invalid arrays are retained without floors/SPD repair and invalid fields are not sent to the geometry probe. No original frozen prefix or radial006 source is changed.\n'''
 (P/'PLAN.md').write_text(plan)
 dump(P/'runtime-inputs.json',{'scope':'Unexecuted observer candidate runtime inventory; standard-library file enumeration, no scientific imports. Bytecode/site-packages excluded from stdlib inventory; exact NumPy wheel sources/binaries included.','roots':[str(p) for p in runtime_roots],'files':runtime})
 protected=dict(old['fixed_pins']);protected.update(runtime)
 protected.update(b['source_before']);protected.update(b['all_compiler_dependency_sha256'])
 before=failed.with_name('protected-inputs-before.json');after=failed.with_name('protected-inputs-after.json')
 assert before.read_bytes()==after.read_bytes();protected.update(load(before))
 for p in [failed,before,after,failed.with_name('launch-before.json'),Path(f['build_receipt']),Path(f['input_path']),Path(f['executable']),Path(f['run_log']),Path(f['stderr_path'])]:protected[str(p)]=sha(p)
 for key,expected in [(f['run_log'],f['run_log_sha256']),(f['stderr_path'],f['stderr_sha256'])]:assert protected[key]==expected
 outdir=Path(f['output_directory']);actual={str(p.relative_to(outdir)) for p in outdir.rglob('*') if p.is_file()};assert actual==set(f['outputs'])
 for relative,item in f['outputs'].items():
  p=outdir/relative;assert p.stat().st_size==item['bytes'];protected[str(p)]=item['sha256']
 seam=load(old['seam_receipt']);pr=load(old['probe_recipe']);protected.update(pr['source_before']);protected.update(seam['source_before']);protected.update(seam['all_compiler_dependency_sha256'])
 for p in [P/'PLAN.md',P/'release-schema.json',P/'runtime-inputs.json',P/'run_released_observation001.py',Path(__file__).resolve(),OLD/'recipe.json',OLD/'observe_partial.py']:
  protected[str(p)]=sha(p)
 for p,item in copies.items():protected[p]=item['sha256']
 checked(protected)
 output_inventory={'scope':'Original failed-process inventories only. No RST/BIN/history/log decoding.','case':case,'original_failed_launch_receipt':pin(failed),'original_outputs':f['outputs'],'output_files':len(f['outputs']),'rst_count':sum(k.endswith('.rst') for k in f['outputs']),'all_original_bytes_match_receipt':True,'original_native_returncode':f['returncode'],'original_target_time':2,'root_reported_unsaved_abort_time':abort_time}
 dump(P/'original-output-inventory.json',output_inventory);protected[str(P/'original-output-inventory.json')]=sha(P/'original-output-inventory.json')
 spec={'name':case,'mode':f['mode'],'input_path':f['input_path'],'build_receipt':f['build_receipt'],'required_hashes':{f['input_path']:f['input_sha256'],f['build_receipt']:f['build_receipt_sha256'],f['executable']:f['executable_sha256']}}
 q=dict(old);q.update(prepared_HEAD=head,cases={case:spec},fixed_pins=protected,initial_review_failures={case:{'original_failed_launch_receipt':str(failed),'sha256':sha(failed),'returncode':f['returncode'],'output_inventory_files':len(f['outputs'])}},source_context_copies=copies,scope=f'Held observations of only original failed process {case}. No new scientific calls or native acceptance.',original_unchanged_N24_template_recipe_sha256=TEMPLATE)
 dump(P/'recipe.json',q)
 readiness={'source_only':True,'execution':'HELD','case':case,'observer_sha256':sha(P/'observe_partial.py'),'observer_byte_equal_to_template':(P/'observe_partial.py').read_bytes()==(OLD/'observe_partial.py').read_bytes(),'recipe_sha256':sha(P/'recipe.json'),'original_failed_launch_receipt':pin(failed),'original_input':pin(f['input_path']),'original_executable':pin(f['executable']),'original_build_receipt':pin(f['build_receipt']),'original_returncode':f['returncode'],'original_output_files':len(f['outputs']),'original_saved_rst_count':sum(k.endswith('.rst') for k in f['outputs']),'shared_prior_fixed_pins':len(old['fixed_pins']),'protected_file_count':len(protected),'candidate_runtime_files':len(runtime),'new_probe_calls':0,'new_native_calls':0,'native_arrays_histories_or_logs_interpreted':False,'scientific_modules_imported':False,'accepted_native_run':False,'partial_diagnostic_only':True}
 dump(P/'source-only-readiness.json',readiness)
 sources=[P/n for n in ['PLAN.md','observe_partial.py','run_released_observation001.py','release-schema.json','runtime-inputs.json','recipe.json','source-only-readiness.json','original-output-inventory.json']]+list(ctx.iterdir())
 for p in [P/'observe_partial.py',P/'run_released_observation001.py']:ast.parse(p.read_text(),filename=str(p))
 index={'source_only':True,'execution':'HELD','files':[pin(p) for p in sources],'protected_external_file_count':len(protected),'scope':readiness}
 dump(P/'source-index.json',index)
 checked(protected)
 prepared.append({'case':case,'prefix':str(P),'source_index':pin(P/'source-index.json'),'recipe':pin(P/'recipe.json'),'runner':pin(P/'run_released_observation001.py'),'readiness':pin(P/'source-only-readiness.json'),'failed_launch':pin(failed),'outputs':len(f['outputs']),'saved_rst':sum(k.endswith('.rst') for k in f['outputs']),'protected_file_count':len(protected),'execution':'HELD'})
dump(HERE/'receipt.json',{'source_only':True,'prepared_HEAD':head,'preparation_source':pin(__file__),'prepared':prepared,'scientific_imports':False,'arrays_or_logs_decoded':False,'probe_or_native_calls':0,'all_pins_unchanged':True,'execution':'HELD'})
print(json.dumps(prepared,indent=2))
