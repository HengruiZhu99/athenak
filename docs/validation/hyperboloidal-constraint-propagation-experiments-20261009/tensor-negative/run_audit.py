from pathlib import Path
import hashlib,json,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/receipt.json').read_text())
cmd=base['commands'][0]['command'][:]
cmd[-3]=str(p/'tensor_kernel.cpp');cmd[-1]=str(p/'tensor_kernel')
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]
files += [p/f for f in ['tensor_kernel.cpp','check_tensor.py','run_audit.py','REPORT.md']]
files += [p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009'/f for f in ['discrete_kernel.cpp','dual_helpers.hpp']]
files += [p.parent/'discrete-bianchi/check_discrete.py',repo/'tst/hyperboloidal/check_kernel_symbol.py']
before={str(f.relative_to(repo)):sha(f) for f in files};results=[];start=time.monotonic()
for command,out in [(cmd,None),([str(p/'tensor_kernel')],'kernel.json'),([sys.executable,str(p/'check_tensor.py')],'check.log')]:
 if out:
  with (p/out).open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 else:r=subprocess.run(command,cwd=repo,text=True,capture_output=True)
 result={'command':command,'returncode':r.returncode,'stderr':r.stderr}
 if out:result.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:result['stdout']=r.stdout
 results.append(result);r.check_returncode()
after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
receipt={'passed_negative_audit':True,'candidate_accepted':False,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,'binary_sha256':sha(p/'tensor_kernel'),'check_report_sha256':sha(p/'check-report.json'),'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'seconds':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS negative audit',sha(p/'receipt.json'))
