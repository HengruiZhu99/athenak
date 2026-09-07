#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001
trap 'printf "%s\n" "$?" > stage-001.exit' EXIT
test "$(cat test.exit)" = 0
test "$(cat repeat-fix.exit)" = 0
python3 - <<'PY'
import hashlib,json,shutil
from pathlib import Path
p=Path('stage-001')
for f,h in json.loads((p/'manifest.json').read_text()).items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
m=json.loads(Path('source-repeat-fix-manifest.json').read_text())
for f,h in m['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
old=Path('before-stage-001');old.mkdir()
shutil.copy2('build/src/athena',old/'athena')
for f in (p/'src').rglob('*'):
 if f.is_file():
  name=f.relative_to(p);saved=old/name;saved.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(Path('source')/name,saved);shutil.copy2(f,Path('source')/name)
  m['files'][str(name)]=hashlib.sha256(f.read_bytes()).hexdigest()
m['commit']='fc46936088945a2beeb858763e713e3c701c2fb8'
m['change']='Optional float64 RK/exchange stage dumps; no equation changes'
for f,h in m['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
Path('source-stage-001-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PY
cmake --build build -j4 > stage-001-build.log 2>&1
sha256sum build/src/athena > stage-001-binary.sha256
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > stage-001-gpu-before.csv
python3 - <<'PY'
import csv
with open('stage-001-gpu-before.csv') as f:
 rows=list(csv.reader(f));assert int(rows[1][-1].strip().split()[0])>=4096
PY
python3 -W error stage-001/tests/check_intrinsic_stage_dump.py --binary "$PWD/build/src/athena" --fixtures "$PWD/decomposition-cuda-seeded-001" --output "$PWD/stage-cuda-serial-001" > stage-cuda-serial-001.log 2>&1
python3 -W error stage-001/tests/check_intrinsic_stage_dump.py --binary "$PWD/build/src/athena" --fixtures "$PWD/decomposition-cuda-seeded-001" --launcher '/usr/local/openmpi/cuda-12.6/4.1.6/nvhpc2411/bin/mpiexec -n 2' --ranks 2 --output "$PWD/stage-cuda-mpi-001" > stage-cuda-mpi-001.log 2>&1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > stage-001-gpu-after.csv
