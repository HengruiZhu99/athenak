from pathlib import Path
import hashlib,json,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'covariant-z4-candidate/receipt.json').read_text())
flags=base['commands'][0]['command'][:-3]
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]
names=['constraint_tangent.cpp','subsidiary.hpp','subsidiary_c0.hpp','blend.hpp','bulk_c1_additions.hpp','c1_additions.hpp','blend_gate.cpp','principal_blend.cpp','kernel_symbol_copy.cpp','check_subsidiary.py','run_audit.py']
files += [p/f for f in names]
files += [p.parent/'covariant-z4-candidate/full20.cpp',p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',repo/'tst/hyperboloidal/kernel_symbol.cpp',repo/'tst/hyperboloidal/check_kernel_symbol.py']
before={str(f.relative_to(repo)):sha(f) for f in files};results=[];start=time.monotonic()
# The copied extractor differs only by the prescribed-radius assignment.
copy=(p/'kernel_symbol_copy.cpp').read_text().replace('\n  bulk_radius=radius; // prescribed scratch C(r) only','')
assert copy==(repo/'tst/hyperboloidal/kernel_symbol.cpp').read_text()
assert sha(p/'c1_additions.hpp')=='908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7'
commands=[(flags+[str(p/f'{f}.cpp'),'-o',str(p/f)],None) for f in ['constraint_tangent','blend_gate','principal_blend']]
debug=flags[:];debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG');debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
commands.append((debug+[str(p/'blend_gate.cpp'),'-o',str(p/'blend_gate_debug')],None))
commands += [([str(p/'constraint_tangent'),'10'],'tangent-C1-kappa10.json'),
 ([str(p/'constraint_tangent'),'10','blend'],'tangent-blend-kappa10.json'),
 ([str(p/'blend_gate')],'blend-gate.json'),([str(p/'blend_gate_debug')],'blend-gate-debug.json'),
 ([sys.executable,str(repo/'tst/hyperboloidal/check_kernel_symbol.py'),str(p/'principal_blend')],'principal.log'),
 ([sys.executable,str(p/'check_subsidiary.py')],'check.log')]
for command,out in commands:
 if out:
  with (p/out).open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 else:r=subprocess.run(command,cwd=repo,text=True,capture_output=True)
 result={'command':command,'returncode':r.returncode,'stderr':r.stderr}
 if out:result.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:result['stdout']=r.stdout
 results.append(result);(p/'commands-in-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 r.check_returncode()
after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
assert all(before.get(k)==v for k,v in base['source_before'].items() if k.startswith('src/') or k=='CMakeLists.txt')
receipt={'passed_reference_tangent_and_blend_gates':True,'global_native_or_scri_stability_accepted':False,
 'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,
 'binary_sha256':{f:sha(p/f) for f in ['constraint_tangent','blend_gate','blend_gate_debug','principal_blend']},
 'check_report_sha256':sha(p/'check-report.json'),'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),
 'python':sys.version,'numpy':__import__('numpy').__version__,'seconds':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS C1/blend subsidiary',sha(p/'receipt.json'))
