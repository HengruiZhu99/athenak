"""Compile independent flat-core formulas, capture dependencies and runs."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'bridge.cpp')=='825d5219d18a0548e82d4692ee0301a78b91d470be58741e1d25962dace31222'
for mode in ('release','debug'):
    old=json.loads((P/('build-'+mode+'-latest.json')).read_text());build=json.loads((Path(old['attempt'])/'receipt.json').read_text())
    cmd=[str(P/'core_oracle.cpp') if x==str(P/'bridge.cpp') else str(P/('core-oracle-'+mode)) if x==str(P/('bridge-'+mode)) else x for x in build['command']]
    A=P/'core-oracle-attempts'/mode;assert not A.exists();A.mkdir(parents=True)
    for name in ('core_oracle.cpp','run_core_oracle.py'):
        shutil.copyfile(P/name,A/name)
    r0=time.monotonic();r=subprocess.run(cmd,text=True,capture_output=True);(A/'build.stdout').write_text(r.stdout);(A/'build.stderr').write_text(r.stderr)
    receipt={'command':cmd,'compiler_exit_code':r.returncode,'compile_seconds':time.monotonic()-r0,'compiler_version':build['compiler_version'],'source_sha256':sha(P/'core_oracle.cpp'),'unchanged_bridge_sha256':sha(P/'bridge.cpp')}
    (A/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');assert r.returncode==0,r.stderr
    dep=[];skip=False
    for a in cmd:
        if skip:skip=False;continue
        if a=='-o':skip=True;continue
        if a.endswith('.a'):continue
        dep.append(a)
    dep+=['-M','-MT','core-oracle'];d=subprocess.check_output(dep,text=True);(A/'dependencies.make').write_text(d)
    paths=sorted(set(str(Path(x).resolve()) for x in shlex.split(d.replace('\\\n',' ').split(':',1)[1])))
    started=time.monotonic();r=subprocess.run([str(P/('core-oracle-'+mode))],text=True,capture_output=True);(A/'stdout.json').write_text(r.stdout);(A/'stderr').write_text(r.stderr)
    receipt.update(run_exit_code=r.returncode,run_seconds=time.monotonic()-started,executable_sha256=sha(P/('core-oracle-'+mode)),compiler_dependency_hashes={p:sha(p) for p in paths},link_archive_hashes=build['link_archive_hashes'])
    (A/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(mode,r.returncode,r.stdout,r.stderr,flush=True);assert r.returncode==0
