from pathlib import Path
import hashlib,json,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/receipt.json').read_text())
flags=base['commands'][0]['command'][:-3]
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]
files += [p/f for f in ['c1_additions.hpp','covariant_gate.cpp','principal_gate.cpp','check_transform.py','run_audit.py','DERIVATION.md']]
files += [p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',repo/'tst/hyperboloidal/kernel_symbol.cpp',repo/'tst/hyperboloidal/check_kernel_symbol.py']
before={str(f.relative_to(repo)):sha(f) for f in files};results=[];start=time.monotonic()
commands=[]
for name in ['covariant_gate','principal_gate']:
 commands.append((flags+[str(p/f'{name}.cpp'),'-o',str(p/name)],None))
debug=flags[:];debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG');debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
commands.append((debug+[str(p/'covariant_gate.cpp'),'-o',str(p/'covariant_gate_sanitize')],None))
commands += [([str(p/'covariant_gate')],'tensor.json'),([str(p/'covariant_gate'),'--affine'],'affine.json'),([str(p/'principal_gate')],'principal.json'),([sys.executable,str(repo/'tst/hyperboloidal/check_kernel_symbol.py'),str(p/'principal_gate')],'principal.log'),([str(p/'covariant_gate_sanitize')],'tensor-sanitize.json'),([sys.executable,str(p/'check_transform.py')],'check.log')]
for command,out in commands:
 if out:
  with (p/out).open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 else:r=subprocess.run(command,cwd=repo,text=True,capture_output=True)
 result={'command':command,'returncode':r.returncode,'stderr':r.stderr}
 if out:result.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:result['stdout']=r.stdout
 results.append(result)
 (p/'commands-in-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 r.check_returncode()
debugrows=json.loads((p/'tensor-sanitize.json').read_text());assert len(debugrows)==384
for row in debugrows:
 for k,v in row.items():
  if k.endswith('_error'):assert v<2e-10,(k,v)
after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
receipt={'passed_tensor_identity_gate':True,'native_or_scri_stability_accepted':False,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,'binary_sha256':{f:sha(p/f) for f in ['covariant_gate','principal_gate','covariant_gate_sanitize']},'check_report_sha256':sha(p/'check-report.json'),'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'seconds':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS tensor identity',sha(p/'receipt.json'))
