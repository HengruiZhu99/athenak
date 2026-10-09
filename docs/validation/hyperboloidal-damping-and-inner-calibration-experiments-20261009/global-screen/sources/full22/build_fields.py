"""Build actual ghosted field-derivative callback in this fixed private tree."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=json.loads((w/'build-production.json').read_text())
cmd=[str(w/'diagnostic_fields.cpp') if x==str(w/'full22_server.cpp') else str(w/'diagnostic-fields') if x==str(w/'server-production') else x for x in cmd]
assert cmd[cmd.index('-o')+1]==str(w/'diagnostic-fields')
(w/'build-diagnostic-fields.json').write_text(json.dumps(cmd,indent=2)+'\n')
t0=time.monotonic();p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(w/'build-diagnostic-fields.log').write_text(p.stdout)
assert p.returncode==0,p.stdout
dc=[];skip=False
for x in cmd:
 if skip:skip=False;continue
 if x=='-o':skip=True;continue
 if x.endswith('.a'):continue
 dc.append(x)
dc+=['-M','-MT','candidate'];output=subprocess.check_output(dc,text=True)
(w/'dependencies-fields.make').write_text(output)
deps=sorted(set(str(Path(x).resolve()) for x in shlex.split(output.replace('\\\n',' ').split(':',1)[1])))
r={'command':cmd,'seconds':time.monotonic()-t0,'dependency_command':dc,'dependency_hashes':{p:sha(p) for p in deps},'executable_sha256':sha(w/'diagnostic-fields'),'link_archive_hashes':{p:sha(p) for p in cmd if p.endswith('.a')}}
(w/'field-diagnostic-build.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'seconds':r['seconds'],'exe_sha256':r['executable_sha256']},indent=2))
