"""Only root-released A/B local principal/core gates. C held."""
import hashlib,json,math,pathlib,shlex,shutil,subprocess,time
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
recipe=json.loads((P/'release-recipe.json').read_text());A=P/'attempts'/str(time.time_ns());A.mkdir(parents=True);shutil.copyfile(P/'release-recipe.json',A/'release-recipe.json')
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'production_implementation':recipe['production_implementation'],'recipe_sha256':sha(A/'release-recipe.json'),'commands':[],'passed_AB':False,'C_queries_or_evolution_run':False}
def save():(A/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def run(cmd,n):
 t=time.monotonic();p=subprocess.run(cmd,cwd=R,text=True,capture_output=True);(A/(n+'.stdout')).write_text(p.stdout);(A/(n+'.stderr')).write_text(p.stderr);r['commands'].append({'command':cmd,'returncode':p.returncode,'seconds':time.monotonic()-t,'stdout_sha256':sha(A/(n+'.stdout')),'stderr_sha256':sha(A/(n+'.stderr'))});save()
 if p.returncode:raise RuntimeError(n+' failed')
 return p.stdout
try:
 r['source_before']={p:sha(R/p)for p in recipe['inputs']};assert r['source_before']==recipe['inputs']
 assert sha(recipe['compiler_path'])==recipe['compiler_sha256'];assert sha(recipe['python_path'])==recipe['python_sha256']
 for n in recipe['local_sources']:shutil.copyfile(P/n,A/n)
 r['compiler_version']=run([recipe['compiler_path'],'--version'],'compiler-version');results={}
 for mode in ['release','debug']:
  results[mode]={}
  for name in ['principal','core']:
   exe=A/(name+'-'+mode);dep=A/(name+'-'+mode+'.d')
   run([recipe['compiler_path']]+recipe[mode+'_flags']+['-MD','-MF',str(dep),str(A/(name+'.cpp')),'-o',str(exe)],'compile-'+name+'-'+mode)
   paths=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);deps={}
   for p in paths:
    q=pathlib.Path(p);q=q if q.is_absolute()else R/q;deps[str(q.resolve())]=sha(q)
   r[name+'_'+mode+'_dependencies']=deps;r[name+'_'+mode+'_executable_sha256']=sha(exe);save()
   out=run([str(exe)],'run-'+name+'-'+mode);data=json.loads(out);(A/(name+'-'+mode+'.json')).write_text(json.dumps(data,indent=2)+'\n');results[mode][name]=data
   if name=='core':
    assert data['core_cases']==204 and data['witnesses']==17
    for k,v in data.items():
     if k in ['core_cases','witnesses']:continue
     assert isinstance(v,(int,float))and math.isfinite(v)and v<=5e-11,(k,v)
  check=json.loads(run([recipe['python_path'],str(A/'check_principal.py'),str(A/('principal-'+mode+'.json'))],'check-principal-'+mode));(A/('check-principal-'+mode+'.json')).write_text(json.dumps(check,indent=2)+'\n')
 pr,pg=results['release']['principal'],results['debug']['principal'];assert len(pr)==len(pg)==792;mx=0
 for a,b in zip(pr,pg):
  assert {k:v for k,v in a.items()if k not in ['M','normal_scaled']}=={k:v for k,v in b.items()if k not in ['M','normal_scaled']}
  mx=max(mx,max(abs(a['M'][i][j]-b['M'][i][j])for i in range(20)for j in range(20)))
 assert mx<=2e-12
 r['release_debug_principal_max_difference']=mx;r['release_debug_numeric_JSON_equal']=results['release']==results['debug'];r['source_after']={p:sha(R/p)for p in recipe['inputs']};r['sources_unchanged']=r['source_before']==r['source_after'];assert r['sources_unchanged'];r['passed_AB']=True;save()
 print(json.dumps({'attempt':str(A),'passed':True,'receipt_sha256':sha(A/'receipt.json'),'core':results['release']['core'],'principal':json.loads((A/'check-principal-release.json').read_text())},indent=2))
except Exception as e:r['exception']=repr(e);save();print(json.dumps({'attempt':str(A),'passed':False,'receipt_sha256':sha(A/'receipt.json'),'exception':repr(e)}));raise
