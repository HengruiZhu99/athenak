"""Unique executable build/provenance for local reference/synthetic gates."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
target=sys.argv[1];mode=sys.argv[2];assert target in ('reference','synthetic') and mode in ('release','debug')
recipe=json.loads((P/'reference-local-recipe.json').read_text())
for item in recipe['sources']:assert sha(P/item['path'])==item['sha256']
base=json.loads((ROOT/'build-layer-research/boundary/mode-subsidiary-defect/build-command.json').read_text())
attempts=P/'build-attempts';attempts.mkdir(exist_ok=True);number=1
while (attempts/f'{target}-{mode}-{number:03}').exists():number+=1
A=attempts/f'{target}-{mode}-{number:03}';A.mkdir();exe=A/'probe'
cmd=[]
for a in base:
 if a in ('-O3','-DNDEBUG'):continue
 if a.endswith('/comparator.cpp'):a=str(P/(target+'_probe.cpp'))
 if a.endswith('/comparator'):a=str(exe)
 cmd.append(a)
cmd[1:1]=['-O3','-DNDEBUG'] if mode=='release' else ['-O0','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
names=['taylor.hpp','complete_reference.hpp','reference_probe.cpp','synthetic_probe.cpp','build_reference.py','reference-local-recipe.json','reference-build-admission.json','reference-radii.txt']
for n in names:shutil.copyfile(P/n,A/n)
r={'kind':'local complete-reference/synthetic build','target':target,'mode':mode,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'runtime_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','command':cmd,'source_hashes_before':{n:sha(P/n) for n in names},
 'compiler_path':str(Path(cmd[0]).resolve()),'compiler_sha256':sha(Path(cmd[0]).resolve()),'compiler_version':subprocess.check_output([cmd[0],'--version'],text=True)}
(A/'launch.json').write_text(json.dumps(r,indent=2)+'\n');start=time.monotonic();run=subprocess.run(cmd,capture_output=True)
(A/'stdout').write_bytes(run.stdout);(A/'stderr').write_bytes(run.stderr)
r.update(exit_code=run.returncode,seconds=time.monotonic()-start)
(A/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
if run.returncode:print(run.stderr.decode());sys.exit(run.returncode)
dep=[];skip=False
for a in cmd:
 if skip:skip=False;continue
 if a=='-o':skip=True;continue
 if a.endswith('.a'):continue
 dep.append(a)
dep+=['-M','-MT','probe'];q=subprocess.run(dep,capture_output=True);(A/'dependencies.make').write_bytes(q.stdout);(A/'dependency.stderr').write_bytes(q.stderr);assert q.returncode==0
paths=sorted(set(str(Path(s).resolve()) for s in shlex.split(q.stdout.decode().replace('\\\n',' ').split(':',1)[1])))
r.update(executable_path=str(exe),executable_sha256=sha(exe),dependency_command=dep,compiler_dependency_hashes={p:sha(p) for p in paths},link_archive_hashes={p:sha(p) for p in cmd if p.endswith('.a')},source_hashes_after={n:sha(P/n) for n in names})
assert r['source_hashes_after']==r['source_hashes_before']
for p,h in r['compiler_dependency_hashes'].items():
 if p.startswith(str(ROOT/'src')+'/'):
  b=subprocess.check_output(['git','show',r['runtime_source_commit']+':'+str(Path(p).relative_to(ROOT))]);assert hashlib.sha256(b).hexdigest()==h
(A/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(A),'receipt_sha256':sha(A/'receipt.json'),'executable_path':str(exe),'executable_sha256':sha(exe),'seconds':r['seconds'],'dependencies':len(paths)},indent=2))
