"""Build and record the analytic coefficient oracle in the private native tree."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=json.loads((w/'build-spatialnorm.json').read_text())
cmd=[str(w/'reference_coefficients.cpp') if x==str(w/'full22_server.cpp') else str(w/'reference-coefficients') if x==str(w/'server-spatialnorm') else x for x in cmd]
assert cmd[cmd.index('-o')+1]==str(w/'reference-coefficients')
t0=time.monotonic();p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(w/'build-reference.log').write_text(p.stdout);assert p.returncode==0,p.stdout
dc=[];skip=False
for x in cmd:
 if skip:skip=False;continue
 if x=='-o':skip=True;continue
 if x.endswith('.a'):continue
 dc.append(x)
dc+=['-M','-MT','candidate'];output=subprocess.check_output(dc,text=True)
(w/'dependencies-reference.make').write_text(output)
deps=sorted(set(str(Path(x).resolve()) for x in shlex.split(output.replace('\\\n',' ').split(':',1)[1])))
out=subprocess.run([str(w/'reference-coefficients')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert out.returncode==0 and not out.stderr,out.stderr
(w/'reference-coefficients.json').write_bytes(out.stdout)
r={'command':cmd,'seconds':time.monotonic()-t0,'dependency_command':dc,'dependency_hashes':{p:sha(p) for p in deps},'executable_sha256':sha(w/'reference-coefficients'),'source_sha256':sha(w/'reference_coefficients.cpp'),'link_archive_hashes':{p:sha(p) for p in cmd if p.endswith('.a')},'run_command':[str(w/'reference-coefficients')],'run_returncode':out.returncode,'run_stderr':out.stderr.decode(),'reference_coefficients_sha256':sha(w/'reference-coefficients.json')}
(w/'reference-build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'seconds':r['seconds'],'executable_sha256':r['executable_sha256'],'coefficients_sha256':r['reference_coefficients_sha256']},indent=2))
