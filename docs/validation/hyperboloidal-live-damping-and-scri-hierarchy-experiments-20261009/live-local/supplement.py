from pathlib import Path
import json,hashlib,subprocess,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p/'compile-receipt.json').read_text())
files=[repo/f for f in base['source_before']]+[p/'full20_RK_all_a.cpp',p/'supplement.py']
before={str(f.relative_to(repo)):sha(f) for f in files};assert all(before[k]==v for k,v in base['source_before'].items())
commands=[(base['commands'][0]['command'][:-3]+[str(p/'full20_RK_all_a.cpp'),'-o',str(p/'full20_RK_all_a')],None),([str(p/'full20_RK_all_a')],'full20-RK-all-a.json')]
results=[]
for cmd,out in commands:
 t=time.monotonic()
 if out:
  with (p/out).open('w') as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.PIPE,text=True)
 else:r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
 result={'command':cmd,'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t}
 if out:result.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:result['stdout']=r.stdout
 results.append(result);r.check_returncode();assert not r.stderr
assert all(sha(repo/k)==v for k,v in before.items())
receipt={'passed_supplement_commands':True,'source_before':before,'sources_unchanged':True,'commands':results,'binary_sha256':sha(p/'full20_RK_all_a'),'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
(p/'supplement-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS supplemental actual20 all-a commands',len(json.loads((p/'full20-RK-all-a.json').read_text())))
