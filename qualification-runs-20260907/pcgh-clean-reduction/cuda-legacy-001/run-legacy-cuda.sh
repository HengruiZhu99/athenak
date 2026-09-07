#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-legacy
echo $$ > test-controller.pid
while test ! -f build.exit; do
  kill -0 143849 || { echo 'build controller absent without exit record'; exit 2; }
  sleep 10
done
test "$(cat build.exit)" = 0
python3 - <<'PY'
import json,hashlib
from pathlib import Path
for arm in ['collision','current']:
 root=Path(arm+'-source'); manifest=json.loads(Path(arm+'-legacy-build-source.json').read_text())
 for name,expected in manifest.items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print('PASS: both source manifests match before CUDA oracles')
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv > test-gpu-before.csv
python3 - <<'PY'
import subprocess
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>2048,'insufficient free GPU memory for this small test'
PY
timeout 600s python3 run_legacy_equivalence.py --collision "$PWD/build-collision/src/athena" --current "$PWD/build-current/src/athena" --output "$PWD/cuda-zero-001" > cuda-zero-001.log 2>&1
cp build-collision/src/athena collision-exact-cuda
python3 - <<'PY'
from pathlib import Path
import difflib,json,hashlib
p=Path('collision-source/src/pc_gh/pc_gh_update.cpp');old=p.read_text()
needle='  par_for("PC-GH RK update", DevExeSpace(),'
assert old.count(needle)==1
new=old.replace(needle,'  auto state = u0;\n  auto accumulator = u1;\n  auto source = u_rhs;\n'+needle)
new=new.replace('u0(m, n, k, j, i) = gam0*u0(m, n, k, j, i)','state(m, n, k, j, i) = gam0*state(m, n, k, j, i)').replace('gam1*u1(m, n, k, j, i)','gam1*accumulator(m, n, k, j, i)').replace('beta_dt*u_rhs(m, n, k, j, i)','beta_dt*source(m, n, k, j, i)')
assert new!=old
Path('collision-update-original.cpp').write_text(old)
Path('collision-cuda-host-capture.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/src/pc_gh/pc_gh_update.cpp',tofile='b/src/pc_gh/pc_gh_update.cpp')))
p.write_text(new)
Path('collision-hostfix-source.json').write_text(json.dumps({'base':'b81b44d658f3b81584e94ce79b92656c112ff908','only_production_change':'src/pc_gh/pc_gh_update.cpp','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'CUDA host-this capture repair only; identical RK arithmetic'},indent=2)+'\n')
PY
cmake --build build-collision -j4 > build-collision-hostfix.log 2>&1
timeout 600s python3 run_legacy_equivalence.py --collision "$PWD/build-collision/src/athena" --current "$PWD/build-current/src/athena" --output "$PWD/cuda-one-step-001" --steps 1 > cuda-one-step-001.log 2>&1
