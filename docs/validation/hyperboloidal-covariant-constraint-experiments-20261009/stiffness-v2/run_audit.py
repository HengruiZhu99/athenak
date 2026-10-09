from pathlib import Path
import hashlib,json,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
identity=p.parent/'covariant-z4/immutable-C1-tensor-identity-20261009'
base=json.loads((identity/'receipt.json').read_text())
flags=base['commands'][0]['command'][:-3]
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]
files += [p/f for f in ['full20.cpp','check_stiffness.py','run_audit.py']]
files += [identity/'c1_additions.hpp',p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp']
before={str(f.relative_to(repo)):sha(f) for f in files};results=[];start=time.monotonic()
commands=[(flags+[str(p/'full20.cpp'),'-o',str(p/'full20')],None),
          ([str(p/'full20')],'Fourier.json'),([str(p/'full20'),'--small-Omega'],'small-Omega.json'),
          ([sys.executable,str(p/'check_stiffness.py')],'check.log')]
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
receipt={'passed_finite_Omega_local_numerical_gate':True,'native_or_scri_stability_accepted':False,
 'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,
 'binary_sha256':sha(p/'full20'),'check_report_sha256':sha(p/'check-report.json'),
 'identity_index_sha256':sha(identity/'index.json'),
 'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),
 'python':sys.version,'numpy':__import__('numpy').__version__,'seconds':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS local C1 candidate',sha(p/'receipt.json'))
