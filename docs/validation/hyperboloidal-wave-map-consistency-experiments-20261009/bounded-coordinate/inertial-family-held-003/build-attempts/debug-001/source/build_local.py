from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode=sys.argv[1];assert mode in ('release','debug')
recipe=json.loads((P/'local-recipe.json').read_text())
for s in recipe['sources']:assert sha(P/s['path'])==s['sha256']
for s in recipe['reference_input_pins']:assert sha(P/s['path'])==s['sha256']
base=json.loads((ROOT/'build-layer-research/boundary/mode-subsidiary-defect/build-command.json').read_text())
A=P/'build-attempts';A.mkdir(exist_ok=True);n=1
while(A/f'{mode}-{n:03}').exists():n+=1
A=A/f'{mode}-{n:03}';A.mkdir();exe=A/'probe';cmd=[]
for arg in base:
 if arg in ('-O3','-DNDEBUG'):continue
 if arg.endswith('/comparator.cpp'):arg=str(P/'coordinate_probe.cpp')
 if arg.endswith('/comparator'):arg=str(exe)
 cmd.append(arg)
cmd[1:1]=['-O3','-DNDEBUG']if mode=='release'else['-O0','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
for s in recipe['sources']:
 dest=A/'source'/s['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/s['path'],dest)
shutil.copyfile(P/'local-recipe.json',A/'local-recipe.json')
r={'kind':'private Cartesian Einstein-coordinate local build','mode':mode,'command':cmd,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runtime_commit':recipe['runtime_commit'],'recipe_sha256':sha(P/'local-recipe.json'),'compiler_path':str(Path(cmd[0]).resolve()),'compiler_sha256':sha(Path(cmd[0]).resolve()),'compiler_version':subprocess.check_output([cmd[0],'--version'],text=True)}
(A/'launch.json').write_text(json.dumps(r,indent=2)+'\n');start=time.monotonic();q=subprocess.run(cmd,capture_output=True);(A/'stdout').write_bytes(q.stdout);(A/'stderr').write_bytes(q.stderr);r.update(exit_code=q.returncode,seconds=time.monotonic()-start);(A/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
if q.returncode:print(q.stderr.decode());sys.exit(q.returncode)
dep=[];skip=False
for arg in cmd:
 if skip:skip=False;continue
 if arg=='-o':skip=True;continue
 if arg.endswith('.a'):continue
 dep.append(arg)
dep+=['-M','-MT','probe'];d=subprocess.run(dep,capture_output=True);(A/'dependencies.make').write_bytes(d.stdout);(A/'dependency.stderr').write_bytes(d.stderr);assert d.returncode==0
paths=sorted(set(str(Path(x).resolve())for x in shlex.split(d.stdout.decode().replace('\\\n',' ').split(':',1)[1])))
r.update(executable_path=str(exe),executable_sha256=sha(exe),dependency_command=dep,dependency_hashes={p:sha(p)for p in paths},archive_hashes={p:sha(p)for p in cmd if p.endswith('.a')})
for p,h in r['dependency_hashes'].items():
 if p.startswith(str(ROOT/'src')+'/'):assert hashlib.sha256(subprocess.check_output(['git','show',r['runtime_commit']+':'+str(Path(p).relative_to(ROOT))])).hexdigest()==h
for s in recipe['sources']:assert sha(P/s['path'])==s['sha256']
(A/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(A),'receipt_sha256':sha(A/'receipt.json'),'exe_sha256':sha(exe),'seconds':r['seconds'],'dependencies':len(paths)}))
