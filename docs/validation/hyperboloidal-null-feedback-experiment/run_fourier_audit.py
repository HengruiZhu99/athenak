from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
p=Path(__file__).resolve().parent;repo=p.parents[3];build=repo/'build-layer-feedback-native'
f={}
for line in (build/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_FLAGS = ','CXX_INCLUDES = ')):
        k,v=line.split(' = ',1);f[k]=shlex.split(v)
flags=f['CXX_FLAGS'];idx=flags.index('-include');del flags[idx:idx+2]
libs=[build/'kokkos'/part/'src'/('libkokkos'+part+'.a') for part in ('containers','algorithms','core','simd')]
command=list(map(str,['/usr/bin/c++','-DKOKKOS_DEPENDENCE']+flags+f['CXX_INCLUDES']+[p/'fourier_audit.cpp','-o',p/'fourier_audit']+libs))
r=subprocess.run(command,cwd=repo,text=True,capture_output=True);(p/'fourier-build.log').write_text(r.stdout+r.stderr)
if r.returncode:print(r.stdout+r.stderr);raise SystemExit(r.returncode)
with (p/'fourier-matrices.json').open('w') as log:r=subprocess.run([p/'fourier_audit'],cwd=repo,stdout=log)
if r.returncode:raise SystemExit(r.returncode)
receipt={'compile_command':command,'run_command':[str(p/'fourier_audit')],
         'source_sha256':{str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p/'fourier_audit.cpp',p/'null_feedback.hpp')},
         'scope':'Frozen local continuum full20 operator from actual tensor/gauge kernel with analytic value/d/dd perturb jets; no grid stencils, boundary, KO or global spectral claim.'}
(p/'fourier-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
r=subprocess.run([sys.executable,p/'check_fourier.py'],cwd=repo)
if r.returncode:raise SystemExit(r.returncode)
receipt['extra_run_commands']=[]
for flag,filename in [('--modes','fourier-mode-matrices.json'),('--k0-pole','fourier-k0-poles.json')]:
    receipt['extra_run_commands'].append([str(p/'fourier_audit'),flag])
    with (p/filename).open('w') as log:r=subprocess.run([p/'fourier_audit',flag],cwd=repo,stdout=log)
    if r.returncode:raise SystemExit(r.returncode)
r=subprocess.run([sys.executable,p/'check_fourier_modes.py'],cwd=repo)
if r.returncode:raise SystemExit(r.returncode)
receipt['executable_sha256']=hashlib.sha256((p/'fourier_audit').read_bytes()).hexdigest()
receipt['artifact_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.glob('fourier-*.json') if x.name!='fourier-receipt.json'}
receipt['source_sha256'].update({str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p/'run_fourier_audit.py',p/'check_fourier.py',p/'check_fourier_modes.py')})
(p/'fourier-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
