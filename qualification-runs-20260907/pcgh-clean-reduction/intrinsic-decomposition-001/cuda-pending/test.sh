#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001
trap 'printf "%s\n" "$?" > test.exit' EXIT
while ! test -f build.exit; do
 if ! kill -0 1455661 2>/dev/null; then echo 'Build controller missing without terminal status'; exit 3; fi
 sleep 5
done
test "$(cat build.exit)" = 0
python3 - <<'PYVERIFY'
import json,hashlib
from pathlib import Path
for f,h in json.loads(Path('tests-v2/manifest.json').read_text()).items():assert hashlib.sha256((Path('tests-v2')/f).read_bytes()).hexdigest()==h,f
for f,h in json.loads(Path('source-manifest.json').read_text())['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
PYVERIFY
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader > gpu-before.csv
python3 - <<'PYRESOURCE'
from pathlib import Path
assert int(Path('gpu-before.csv').read_text().split(',')[-1].split()[0])>=4096
PYRESOURCE
python3 -W error tests-v2/check_intrinsic_mesh.py --rate-profile lapse_scaled --binary "$PWD/build/src/athena" --reference "$PWD/tests-v2/candidate.py" --output "$PWD/oracle-cuda-001" > oracle-cuda-001.log 2>&1
python3 -W error tests-v2/check_intrinsic_decomposition.py --binary "$PWD/build/src/athena" --reference-binary "$PWD/build/src/athena" --output "$PWD/decomposition-cuda-serial-001" > decomposition-cuda-serial-001.log 2>&1
python3 -W error tests-v2/check_intrinsic_decomposition.py --binary "$PWD/build/src/athena" --reference-binary "$PWD/build/src/athena" --launcher '/usr/local/openmpi/cuda-12.6/4.1.6/nvhpc2411/bin/mpiexec -n 2' --ranks 2 --output "$PWD/decomposition-cuda-mpi-001" > decomposition-cuda-mpi-001.log 2>&1
python3 -W error tests-v2/analyze_intrinsic_restart.py --input "$PWD/decomposition-cuda-mpi-001/fd6-3d-multi/rst" --output "$PWD/snapshot-diagnostics-cuda-001" > snapshot-diagnostics-cuda-001.log 2>&1
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader > gpu-after.csv
