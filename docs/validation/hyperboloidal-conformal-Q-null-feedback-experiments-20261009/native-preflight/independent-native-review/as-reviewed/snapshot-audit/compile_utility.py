"""Compile the values-only null-feedback snapshot utility with pinned native headers."""
import hashlib,json,subprocess,time,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'native-build/build-receipt.json')=='8f7e60ef5d5df9dad31f8feba359b244b36859aa115d525d2f52b057a8813508'
cmd=json.loads((HERE/'compile-command.json').read_text());started=time.monotonic()
with (HERE/'compile.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
r.check_returncode()
deps={}
for name in shlex.split((HERE/'check_snapshot.d').read_text().replace('\\\n',' '))[1:]:
 p=Path(name)
 if p.is_file() and p.is_relative_to(ROOT):deps[str(p.relative_to(ROOT))]=sha(p)
libs={str(Path(x).relative_to(ROOT)):sha(Path(x)) for x in cmd if x.endswith('.a')}
receipt={'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-started,'source_sha256':sha(HERE/'check_snapshot.cpp'),'executable_sha256':sha(HERE/'check_snapshot'),'all_repository_dependencies_sha256':deps,'libraries_sha256':libs,'scientific_gate_index_sha256':'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96','private_build_receipt_sha256':sha(BASE/'native-build/build-receipt.json'),'recipe_sha256':sha(Path(__file__))}
(HERE/'compile-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS independent snapshot utility compile',len(deps),'dependencies',len(libs),'libraries',receipt['executable_sha256'])
