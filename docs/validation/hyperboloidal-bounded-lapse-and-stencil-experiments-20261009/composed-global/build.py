from pathlib import Path
import subprocess,json,hashlib,time,concurrent.futures
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prod=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],text=True).splitlines()
paths=[Path(p) for p in prod]+[p for p in w.rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.hpp'] and 'history' not in p.parts and 'inputs' not in p.parts]
before={str(p):sha(p) for p in paths};t0=time.time()
def run(name):
 cmd=json.loads((w/('build-'+name+'.json')).read_text());t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True);(w/(name+'-build.stdout')).write_text(r.stdout);(w/(name+'-build.stderr')).write_text(r.stderr)
 return {'name':name,'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-t,'executable_sha256':sha(w/name) if r.returncode==0 else None}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['server-spatialnorm','diagnostic-composed']))
after={str(p):sha(p) for p in paths};receipt={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'compiled_production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':after,'sources_unchanged':before==after,'results':results,'seconds':time.time()-t0};(w/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'results':results,'sources_unchanged':receipt['sources_unchanged'],'source_count':len(before)},indent=2));assert before==after and all(r['returncode']==0 for r in results)
