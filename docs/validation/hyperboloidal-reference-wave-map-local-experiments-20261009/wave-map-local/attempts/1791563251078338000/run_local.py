"""Pinned private local algebra/reference/gauge/dual gate; no operators/evolution."""
import hashlib,json,pathlib,shlex,shutil,subprocess,time,sys
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[2]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
recipe=json.loads((P/'release-recipe.json').read_text())
A=P/'attempts'/str(time.time_ns());A.mkdir(parents=True)
shutil.copyfile(P/'release-recipe.json',A/'release-recipe.json')
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':recipe['production_implementation'],'recipe_sha256':sha(A/'release-recipe.json'),'commands':[],'passed_local_gate':False,'operators_or_evolution_run':False}
def save(): (A/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def run(cmd,name):
 t=time.monotonic();q=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
 (A/(name+'.stdout')).write_text(q.stdout);(A/(name+'.stderr')).write_text(q.stderr)
 r['commands'].append({'command':cmd,'returncode':q.returncode,'seconds':time.monotonic()-t,'stdout_sha256':sha(A/(name+'.stdout')),'stderr_sha256':sha(A/(name+'.stderr'))})
 save()
 if q.returncode:raise RuntimeError(name+' failed')
 return q.stdout
try:
 r['source_before']={p:sha(ROOT/p) for p in recipe['inputs']}
 if r['source_before']!=recipe['inputs']:raise RuntimeError('precompile source hash mismatch')
 if sha(recipe['compiler_path'])!=recipe['compiler_sha256']:raise RuntimeError('compiler hash mismatch')
 for f in recipe['local_sources']:shutil.copyfile(P/f,A/f)
 r['compiler_version']=run([recipe['compiler_path'],'--version'],'compiler-version')
 results={}
 for mode in ['release','debug']:
  flags=recipe[mode+'_flags'];exe=A/('local-'+mode);dep=A/(mode+'.d')
  run([recipe['compiler_path']]+flags+['-MD','-MF',str(dep),str(A/'local_gate.cpp'),'-o',str(exe)],'compile-'+mode)
  # Full compiler-emitted dependencies include SDK, Kokkos/config, actual source.
  names=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);deps={}
  for n in names:
   q=pathlib.Path(n);q=q if q.is_absolute() else ROOT/q;deps[str(q.resolve())]=sha(q)
  r[mode+'_compiler_dependencies']=deps;r[mode+'_executable_sha256']=sha(exe);save()
  out=run([str(exe)],'run-'+mode);results[mode]=json.loads(out)
  (A/(mode+'.json')).write_text(json.dumps(results[mode],indent=2)+'\n')
  for key,lim in recipe['thresholds'].items():
   value=results[mode][key]
   if isinstance(value,list):value=value[-1]
   if not isinstance(value,(int,float)) or not __import__('math').isfinite(value) or value>lim:raise RuntimeError('threshold '+key+': '+str(value)+' > '+str(lim))
  if results[mode]['factored_reference_max']!=0:raise RuntimeError('factored reference nonzero')
 if results['release']!=results['debug']:raise RuntimeError('Release/Debug result mismatch')
 r['source_after']={p:sha(ROOT/p) for p in recipe['inputs']};r['sources_unchanged']=r['source_before']==r['source_after']
 if not r['sources_unchanged']:raise RuntimeError('source changed during gate')
 r['passed_local_gate']=True;r['release_debug_equal']=True;save()
 print(json.dumps({'attempt':str(A),'passed':True,'receipt_sha256':sha(A/'receipt.json'),'results':results['release']},indent=2))
except Exception as e:
 r['exception']=repr(e);save();print(json.dumps({'attempt':str(A),'passed':False,'exception':repr(e),'receipt_sha256':sha(A/'receipt.json')}));raise
