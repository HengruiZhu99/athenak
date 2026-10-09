from pathlib import Path
import hashlib,json,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'covariant-z4-candidate/receipt.json').read_text());flags=base['commands'][0]['command'][:-3]
names=['live_damping_profile.hpp','profile_helpers.hpp','subsidiary_c0.hpp','subsidiary_profile.hpp','constraint_tangent.cpp','full20_profile.cpp','tensor_gate.cpp','principal_profile.cpp','run_audit.py','kernel_symbol_copy.cpp','gradient_gate.cpp']
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]+[p/f for f in names]+[p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',repo/'tst/hyperboloidal/kernel_symbol.cpp',repo/'tst/hyperboloidal/check_kernel_symbol.py']
before={str(f.relative_to(repo)):sha(f) for f in files};results=[];start=time.monotonic()
assert (p/'kernel_symbol_copy.cpp').read_text().replace('\n  live_radius=radius; // prescribed scratch V(r) only','')==(repo/'tst/hyperboloidal/kernel_symbol.cpp').read_text()
commands=[(flags+[str(p/f'{f}.cpp'),'-o',str(p/f)],None) for f in ['constraint_tangent','full20_profile','tensor_gate','principal_profile','gradient_gate']]
debug=flags[:];debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG');debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
commands += [(debug+[str(p/'tensor_gate.cpp'),'-o',str(p/'tensor_gate_debug')],None),([str(p/'tensor_gate')],'tensor-gate.json'),([str(p/'tensor_gate_debug')],'tensor-gate-debug.json'),([str(p/'full20_profile'),'pole'],'leading-pole.json'),([str(p/'full20_profile')],'full20.json'),([str(p/'constraint_tangent')],'constraint-profile.json'),([sys.executable,str(repo/'tst/hyperboloidal/check_kernel_symbol.py'),str(p/'principal_profile')],'principal.log'),([str(p/'gradient_gate')],'gradient-gate.json')]
for command,out in commands:
 t=time.monotonic()
 if out:
  with (p/out).open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 else:r=subprocess.run(command,cwd=repo,text=True,capture_output=True)
 result={'command':command,'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t}
 if out:result.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:result['stdout']=r.stdout
 results.append(result);(p/'commands-in-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 if r.returncode:
  (p/'failed-receipt.json').write_text(json.dumps({'source_before':before,'commands':results},indent=2)+'\n')
  r.check_returncode()
after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
assert all(before.get(k)==v for k,v in base['source_before'].items() if k.startswith('src/') or k=='CMakeLists.txt')
receipt={'passed_compiled_commands':True,'mathematical_checker_pending':True,'global_native_or_scri_stability_accepted':False,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,'binary_sha256':{f:sha(p/f) for f in ['constraint_tangent','full20_profile','tensor_gate','tensor_gate_debug','principal_profile','gradient_gate']},'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'python':sys.version,'numpy':__import__('numpy').__version__,'seconds':time.monotonic()-start}
(p/'compile-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS compiled profile commands',sha(p/'compile-receipt.json'))
