"""One-shot metadata, source copying/diff and AST proof only; no arithmetic imports."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'build-layer-research/continuum/manufactured-Gaussian-a2-interval-cache-v5-cap-held-20261009'
RUN=OLD/'attempts/certificate001'
DIAG=ROOT/'build-layer-research/continuum/Gaussian-a2-cache-v5-capped-independent-diagnosis-20261009'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def pin(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def marked(lines,indent):
 prefix=' '*indent
 return prefix+'# BEGIN_OBSERVATION_ONLY\n'+''.join(prefix+line+'\n' for line in lines)+prefix+'# END_OBSERVATION_ONLY\n'
def insert(source,anchor,lines,indent,before=False):
 if source.count(anchor)!=1:raise RuntimeError('nonunique instrumentation anchor '+anchor)
 block=marked(lines,indent)
 return source.replace(anchor,block+anchor if before else anchor+block,1)
def strip(source):
 out=[];inside=False
 for line in source.splitlines(keepends=True):
  if line.strip()=='# BEGIN_OBSERVATION_ONLY':
   if inside:raise RuntimeError('nested marker')
   inside=True
  elif line.strip()=='# END_OBSERVATION_ONLY':
   if not inside:raise RuntimeError('unmatched marker')
   inside=False
  elif not inside:out.append(line)
 if inside:raise RuntimeError('unclosed marker')
 return ''.join(out)

def main():
 if (HERE/'source-index.json').exists():raise RuntimeError('one-shot source freeze')
 if sha(OLD/'source-index.json')!='da66e0687dea4832913b975c4079b11263c072024f038d27d56873ceee428341':raise RuntimeError('old source index')
 if sha(RUN/'receipt.json')!='46bbdd2c0883b0f64d951cdb1152a8bc4de5fad6239d7a7d266b02d476626156':raise RuntimeError('actual timeout receipt')
 oldindex=load(OLD/'source-index.json');oldrecipe=load(OLD/'recipe.json');prior=load(RUN/'receipt.json')
 if not(prior['completed'] is False and prior['passed'] is False and prior['inputs_unchanged'] is True and prior['returncode']==1):raise RuntimeError('timeout classification')
 if (RUN/'report.json').exists():raise RuntimeError('unexpected producer report')
 protected={}
 for r in oldrecipe['protected_inputs']+oldindex['files']:
  p=Path(r['path'])
  if sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise RuntimeError('old protected source drift')
  protected[str(p)]=r
 history=HERE/'history/v5';history.mkdir(parents=True)
 for r in oldindex['files']:
  p=Path(r['path']);q=history/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix in ('.jsonl','.npz','.npy') or p.stat().st_size>1048576:raise RuntimeError('source-only history contains a payload')
  shutil.copyfile(p,q)
 shutil.copyfile(OLD/'source-index.json',history/'source-index.json')
 top=['interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py','replay_stage.py',
      'unit_stage.py','cache_units.py','run_once.py','admission.py','certificate_stage.py',
      'ARITHMETIC-AND-REPLAY.md','CACHE-PLAN.md','CERTIFICATE-PLAN.md','ENCODING-ADDENDUM.md']
 for name in top:shutil.copyfile(OLD/name,HERE/name)
 cert=(OLD/'certificate_stage.py').read_text()
 new=insert(cert,'    from producer_bounds import coefficients\n',['from instrumentation import ObservedEndpointCache, ProgressObserver'],4)
 new=insert(new,'    ctx, K = Context(recipe["bits"]), recipe["series_order"]\n',
            ['cache = ObservedEndpointCache()','ctx._exp_endpoint_cache = cache'],4)
 new=insert(new,'    counts = {"regular": 0, "separated": 0, "upper_tail": 0, "lower_tail": 0}\n',
            ['observer = ProgressObserver(out, roots, started, cache)'],4)
 new=insert(new,'    with target.open("xb") as handle:\n',
            ['observer.progress(nodes, leaves, stack, counts, minimum, "initial")'],8)
 new=insert(new,'                raise RuntimeError("UNRESOLVED: declared domain time limit")\n',
            ['observer.progress(nodes, leaves, stack, counts, minimum, "domain_time_cap")'],16,True)
 new=insert(new,'            result = None\n',['observer.active(root_id, path, box, depth)'],12,True)
 new=insert(new,'            low = directed(low, ctx.bits, False)\n',['observer.lower(method, low)'],12)
 new=insert(new,'                counts[method] += 1\n',['observer.leaf(root_id)'],16)
 new=insert(new,'                    "pending": len(stack), "elapsed_seconds": time.monotonic() - started}, allow_nan=False) + "\\n")\n',
            ['observer.progress(nodes, leaves, stack, counts, minimum, "checkpoint")'],16)
 new=insert(new,'    report = {"passed": True, "stage": "certificate", "source_index_sha256": index_sha,\n',
            ['observer.progress(nodes, leaves, stack, counts, minimum, "producer_finished")'],4,True)
 if strip(new)!=cert:raise RuntimeError('observation-erased producer byte equality failed')
 (HERE/'certificate_stage.py').write_text(new)
 admission=(OLD/'admission.py').read_text()
 guard_old='    if stage not in ("certificate", "replay"):\n        raise RuntimeError("cache v5 admits certificate/replay only after exact release")\n'
 guard_new='    if stage != "certificate":\n        raise RuntimeError("cache v6 instrumentation admits only a fresh diagnostic producer")\n'
 if admission.count(guard_old)!=1:raise RuntimeError('old stage guard anchor')
 admission_new=admission.replace(guard_old,guard_new,1)
 extra=[
  'v5_pin = recipe["cached_v5_timeout_receipt"]',
  'if authorization.get("prior_v5_timeout_receipt") != v5_pin:',
  '    raise RuntimeError("exact failed cached-v5 receipt pin required")',
  'verify_pin(v5_pin)',
  'v5 = load(v5_pin["path"])',
  'if not (v5.get("completed") is False and v5.get("passed") is False',
  '        and type(v5.get("returncode")) is int and v5.get("returncode") == 1',
  '        and v5.get("inputs_unchanged") is True and v5.get("stage") == "certificate"',
  '        and v5.get("source_index_sha256") == recipe["cached_v5_source_index_sha256"]',
  '        and v5.get("recipe_sha256") == recipe["cached_v5_recipe_sha256"]):',
  '    raise RuntimeError("cached-v5 timeout history classification differs")',
  'for entry in (recipe["cached_v5_progress"], recipe["cached_v5_stderr"],',
  '              recipe["cached_v5_command"], recipe["cached_v5_partial_certificate_metadata"]):',
  '    if entry not in v5.get("outputs", []):',
  '        raise RuntimeError("v5 failed receipt does not bind history output")',
  'for entry in (recipe["cached_v5_progress"], recipe["cached_v5_stderr"], recipe["cached_v5_command"]):',
  '    verify_pin(entry)',
  'if "UNRESOLVED: declared domain time limit" not in Path(recipe["cached_v5_stderr"]["path"]).read_text():',
  '    raise RuntimeError("v5 failure is not the declared time cap")',
  'if (Path(v5_pin["path"]).parent / "report.json").exists():',
  '    raise RuntimeError("failed v5 unexpectedly has a producer report")',
 ]
 admission_new=insert(admission_new,'    dependencies = {"prior_timeout": prior}\n',extra,4,True)
 if strip(admission_new).replace(guard_new,guard_old,1)!=admission:raise RuntimeError('admission reverse equality')
 (HERE/'admission.py').write_text(admission_new)
 # No partial payload is copied/decoded. Its immutable failed-receipt metadata is retained.
 byname={Path(r['path']).name:r for r in prior['outputs']}
 recipe=dict(oldrecipe)
 recipe.update(status='HELD instrumentation-only fresh 600-second cached producer; no resume or replay',
  domain_wall_seconds=600,cached_v5_timeout_receipt=pin(RUN/'receipt.json'),
  cached_v5_source_index_sha256=sha(OLD/'source-index.json'),cached_v5_recipe_sha256=sha(OLD/'recipe.json'),
  cached_v5_progress=byname['progress.json'],cached_v5_stderr=byname['stderr.log'],
  cached_v5_command=byname['command.json'],cached_v5_partial_certificate_metadata=byname['certificate.jsonl'],
  instrumentation_only=True,instrumentation_cadence_nodes=128,replay_admitted_by_this_candidate=False,
  no_resume=True,no_completion_fraction_claim=True)
 recipe['stage_timeouts']=dict(oldrecipe['stage_timeouts']);recipe['stage_timeouts']['certificate']=660
 added=[OLD/'source-index.json',RUN/'receipt.json',RUN/'progress.json',RUN/'stderr.log',RUN/'command.json',
        DIAG/'index.json',DIAG/'receipt.json',DIAG/'ASSESSMENT.md',
        ROOT/'build-layer-research/Gaussian-a2-interval-cache-v5-root-release-20261009/certificate-invocation001/receipt.json']
 for p in added:protected[str(p)]=pin(p)
 recipe['protected_inputs']=list(sorted(protected.values(),key=lambda r:r['path']))
 # Prove every original scientific recipe setting unchanged; only the fixed resource bounds differ.
 exempt={'status','domain_wall_seconds','stage_timeouts','protected_inputs'}
 if any(recipe[k]!=v for k,v in oldrecipe.items() if k not in exempt):raise RuntimeError('scientific recipe drift')
 if recipe['stage_timeouts']['replay']!=oldrecipe['stage_timeouts']['replay']:raise RuntimeError('replay cap drift')
 save(HERE/'recipe.json',recipe)
 save(HERE/'authorization-schema.json',{'allow_execution':False,'stage':'certificate',
  'source_index_sha256':'exact new index','recipe_sha256':'exact new recipe','output':'fresh immediate attempts child',
  'units_receipt':recipe['cached_v3_units_receipt'],'prior_timeout_receipt':recipe['cached_v4_timeout_receipt'],
  'prior_v5_timeout_receipt':recipe['cached_v5_timeout_receipt'],'scope':'instrumented observational producer only; no resume/replay/global acceptance'})
 diffs={}
 for name in ('certificate_stage.py','admission.py','recipe.json'):
  before=(OLD/name).read_text();after=(HERE/name).read_text()
  filename=name.replace('.','-')+'-v5-v6.diff'
  (HERE/filename).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='v5/'+name,tofile='v6/'+name)))
  diffs[name]=sha(HERE/filename)
 proofs=[]
 for name in ('interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py','replay_stage.py','unit_stage.py','cache_units.py','run_once.py'):
  before=(OLD/name).read_text();after=(HERE/name).read_text()
  if before!=after:raise RuntimeError('unchanged mathematical/wrapper source drift')
  proofs.append({'file':name,'byte_equal':True,'AST_equal':ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False)})
 proofs.append({'file':'certificate_stage.py','observation_erased_byte_equal':strip(new)==cert,
  'observation_erased_AST_equal':ast.dump(ast.parse(strip(new)),include_attributes=False)==ast.dump(ast.parse(cert),include_attributes=False)})
 save(HERE/'source-equalities.json',{'proofs':proofs,'admission_reverse_byte_equal':True,
  'scientific_recipe_equal_except_resource_fields':True,'original_domain_cap':1800,'new_domain_cap':600,
  'original_wrapper_cap':1860,'new_wrapper_cap':660,'diffs':diffs,'numerical_execution':False,
  'cache_protocol_scope':'dict membership/delete return/exception/order unchanged; observer integer counters only'})
 for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for r in recipe['protected_inputs']:
  if sha(r['path'])!=r['sha256']:raise RuntimeError('final external drift')
 files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='source-index.json')
 save(HERE/'source-index.json',{'source_only':True,'execution_authorized':False,'file_count':len(files),'files':[pin(p) for p in files]})
 print(json.dumps({'source_index':sha(HERE/'source-index.json'),'recipe':sha(HERE/'recipe.json'),
  'certificate':sha(HERE/'certificate_stage.py'),'instrumentation':sha(HERE/'instrumentation.py'),
  'files':len(files),'protected_inputs':len(recipe['protected_inputs']),'scientific_imports':False}))
if __name__=='__main__':main()
