from pathlib import Path
import subprocess,hashlib,json,time
p=Path(__file__).resolve().parent;repo=p.parents[2];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'live-damping-control/receipt.json').read_text());sources={k:v for k,v in base['source_before'].items() if k.startswith('src/') or k=='CMakeLists.txt' or k.endswith(('profile_helpers.hpp','live_damping_profile.hpp','dual_helpers.hpp'))}
for f in ['taylor_kernel.cpp','run_audit.py']:sources[str((p/f).relative_to(repo))]=sha(p/f)
assert all(sha(repo/k)==v for k,v in sources.items());flags=base['commands'][0]['command'][:-3];commands=[flags+[str(p/'taylor_kernel.cpp'),'-o',str(p/'taylor_kernel')],[str(p/'taylor_kernel')]];rows=[]
for i,cmd in enumerate(commands):
 t=time.monotonic()
 if i:
  with (p/'kernel.json').open('w') as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.PIPE,text=True)
 else:r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
 rows.append({'command':cmd,'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t});r.check_returncode();assert not r.stderr
 if i:rows[-1]['stdout_file']='kernel.json';rows[-1]['stdout_sha256']=sha(p/'kernel.json')
 else:rows[-1]['stdout']=r.stdout
assert all(sha(repo/k)==v for k,v in sources.items())
(p/'compile-receipt.json').write_text(json.dumps({'sources':sources,'sources_unchanged':True,'commands':rows,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'binary_sha256':sha(p/'taylor_kernel'),'no_native_evolution':True},indent=2)+'\n');print('PASS actual Taylor kernel commands')
