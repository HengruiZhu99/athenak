#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001
trap 'printf "%s\n" "$?" > repeat-fix.exit' EXIT
while ! test -f test.exit; do
 if ! kill -0 1505096 2>/dev/null; then echo 'Original test controller missing without terminal status'; exit 3; fi
 sleep 5
done
test "$(cat test.exit)" = 0
python3 - <<'PYVERIFY'
import json,hashlib
from pathlib import Path
for f,h in json.loads(Path('repeat-fix/manifest.json').read_text()).items():assert hashlib.sha256((Path('repeat-fix')/f).read_bytes()).hexdigest()==h,f
PYVERIFY
mkdir before-repeat-fix
cp build/src/athena before-repeat-fix/athena
cp source/src/pc_gh/pc_gh_intrinsic.cpp before-repeat-fix/pc_gh_intrinsic.cpp
cp repeat-fix/pc_gh_intrinsic.cpp source/src/pc_gh/pc_gh_intrinsic.cpp
python3 - <<'PYSOURCE'
import json,hashlib
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
m['parent_commit']=m.pop('commit');m['change']='accept false restart_tracker_state marker; reject true; no equation changes'
m['files']['src/pc_gh/pc_gh_intrinsic.cpp']=hashlib.sha256(Path('source/src/pc_gh/pc_gh_intrinsic.cpp').read_bytes()).hexdigest()
for f,h in m['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
Path('source-repeat-fix-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PYSOURCE
cmake --build build -j4 > repeat-fix-build.log 2>&1
sha256sum build/src/athena > repeat-fix-binary.sha256
python3 -W error repeat-fix/tests/check_intrinsic_mesh.py --rate-profile lapse_scaled --binary "$PWD/build/src/athena" --reference "$PWD/repeat-fix/tests/candidate.py" --output "$PWD/oracle-cuda-repeat-fix-001" > oracle-cuda-repeat-fix-001.log 2>&1
python3 -W error repeat-fix/tests/check_intrinsic_decomposition.py --seeded --binary "$PWD/build/src/athena" --reference-binary "$PWD/build/src/athena" --launcher '/usr/local/openmpi/cuda-12.6/4.1.6/nvhpc2411/bin/mpiexec -n 2' --ranks 2 --output "$PWD/decomposition-cuda-seeded-001" > decomposition-cuda-seeded-001.log 2>&1
python3 -W error repeat-fix/tests/check_intrinsic_mesh_controls.py --binary "$PWD/build/src/athena" --input "$PWD/oracle-cuda-repeat-fix-001/fd6-2d/used.athinput" --restart "$PWD/oracle-cuda-repeat-fix-001/fd6-2d/rst/legacy_equivalence.00001.rst" --output "$PWD/repeat-fix-controls-001" > repeat-fix-controls-001.log 2>&1
