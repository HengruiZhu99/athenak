from pathlib import Path
import hashlib, json, shlex, subprocess, sys
p=Path(__file__).resolve().parent; repo=p.parents[3]; build=repo/"build-layer-feedback-native"
flags={}
for line in (build/"src/CMakeFiles/athena.dir/flags.make").read_text().splitlines():
    if line.startswith(("CXX_INCLUDES = ","CXX_FLAGS = ")):
        k,v=line.split(" = ",1);flags[k]=shlex.split(v)
common=["/usr/bin/c++","-DKOKKOS_DEPENDENCE"]+flags["CXX_FLAGS"]+flags["CXX_INCLUDES"]
libs=[build/"kokkos"/part/"src"/("libkokkos"+part+".a") for part in ("containers","algorithms","core","simd")]
commands=[]
def run(args,log):
    args=list(map(str,args));commands.append(args)
    r=subprocess.run(args,cwd=repo,text=True,capture_output=True)
    (p/log).write_text(r.stdout+r.stderr)
    if r.returncode: print(r.stdout+r.stderr);raise SystemExit(r.returncode)
    return r.stdout
for name in ("overlay_audit","overlay_symbol"):
    run(common+[p/(name+".cpp"),"-o",p/name]+libs,name+"-build.log")
run([p/"overlay_audit"],"overlay-reference.log")
actual=json.loads(run([p/"overlay_audit","--pole"],"overlay-poles.json"))
expected=json.loads((p.parent/"feedback_pole.json").read_text())
import numpy as np
assert len(actual)==len(expected)==4
maximum=0
for aa,ee in zip(actual,expected):
    assert aa["a"]==ee["a"]
    maximum=max(maximum,float(np.max(abs(np.asarray(aa["M"])-np.asarray(ee["M"])))))
assert maximum<1e-12,maximum
run([sys.executable,repo/"tst/hyperboloidal/check_kernel_symbol.py",p/"overlay_symbol"],"overlay-symbol.log")
receipt={"commands":commands,"full20_difference_from_audited_candidate":maximum,
"source_sha256":{str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.suffix in (".cpp",".hpp",".py")}}
(p/"overlay-audit-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print("PASS exact overlay full20 equivalence",maximum)
print((p/"overlay-reference.log").read_text());print((p/"overlay-symbol.log").read_text())
