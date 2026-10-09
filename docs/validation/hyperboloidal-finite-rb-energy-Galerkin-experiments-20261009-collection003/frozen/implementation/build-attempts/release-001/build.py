"""Fresh staged bridge build; save every attempted source and command."""
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode = sys.argv[1] if len(sys.argv)>1 else 'release'
assert mode in ('release','debug')
base = json.loads((ROOT/'build-layer-research/boundary/mode-subsidiary-defect/build-command.json').read_text())
cmd=[]
for a in base:
    if a in ('-O3','-DNDEBUG'):continue
    if a.endswith('/comparator.cpp'):a=str(P/'radial_bridge.cpp')
    if a.endswith('/comparator'):a=str(P/('radial-bridge-'+mode))
    cmd.append(a)
flags=['-O3','-DNDEBUG'] if mode=='release' else ['-O0','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
cmd[1:1]=flags
attempts=P/'build-attempts';attempts.mkdir(exist_ok=True)
k=1
while (attempts/('%s-%03d'%(mode,k))).exists():k+=1
A=attempts/('%s-%03d'%(mode,k));A.mkdir()
for name in ('radial_bridge.cpp','actual_bridge.cpp','baseline_dual_spatial.hpp','spatial_dual.hpp','generic_gauge.hpp','configuration_rows.hpp','radial_normalization.hpp','point_energy.hpp','build.py'):
    shutil.copyfile(P/name,A/name)
record={'mode':mode,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runtime_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','command':cmd,'sources_before':{n:sha(P/n) for n in ('radial_bridge.cpp','actual_bridge.cpp','baseline_dual_spatial.hpp','spatial_dual.hpp','generic_gauge.hpp','configuration_rows.hpp','radial_normalization.hpp','point_energy.hpp','all_m_data.hpp')},'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True)}
started=time.monotonic();r=subprocess.run(cmd,text=True,capture_output=True)
(A/'stdout').write_text(r.stdout);(A/'stderr').write_text(r.stderr)
record.update(exit_code=r.returncode,seconds=time.monotonic()-started)
(A/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
if r.returncode:
    print(r.stderr);sys.exit(r.returncode)
dep=[];skip=False
for a in cmd:
    if skip:skip=False;continue
    if a=='-o':skip=True;continue
    if a.endswith('.a'):continue
    dep.append(a)
dep+=['-M','-MT','bridge']
r=subprocess.run(dep,text=True,capture_output=True);(A/'dependencies.make').write_text(r.stdout);(A/'dependency.stderr').write_text(r.stderr);assert r.returncode==0
paths=sorted(set(str(Path(s).resolve()) for s in shlex.split(r.stdout.replace('\\\n',' ').split(':',1)[1])))
record.update(executable_sha256=sha(P/('radial-bridge-'+mode)),dependency_command=dep,compiler_dependency_hashes={p:sha(p) for p in paths},link_archive_hashes={p:sha(p) for p in cmd if p.endswith('.a')},sources_after={n:sha(P/n) for n in record['sources_before']})
assert record['sources_before']==record['sources_after']
for p,h in record['compiler_dependency_hashes'].items():
    if p.startswith(str(ROOT/'src')+'/'):
        rel=str(Path(p).relative_to(ROOT));assert hashlib.sha256(subprocess.check_output(['git','show',record['runtime_source_commit']+':'+rel])).hexdigest()==h,rel
(A/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
(P/('build-'+mode+'-latest.json')).write_text(json.dumps({'attempt':str(A),'receipt_sha256':sha(A/'receipt.json'),'executable_sha256':record['executable_sha256']},indent=2)+'\n')
print(json.dumps({'built':True,'mode':mode,'seconds':record['seconds'],'dependency_count':len(paths),'executable_sha256':record['executable_sha256'],'attempt':str(A)},indent=2))
