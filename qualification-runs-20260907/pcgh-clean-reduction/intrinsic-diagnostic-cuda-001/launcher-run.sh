#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001
trap 'printf "%s\n" "$?" > diagnostic-001.exit' EXIT
test "$(cat stage-001.exit)" = 0
python3 - <<'PY'
import hashlib,json,shutil
from pathlib import Path
p=Path('diagnostic-001')
for f,h in json.loads((p/'manifest.json').read_text()).items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
m=json.loads(Path('source-stage-001-manifest.json').read_text())
for f,h in m['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
old=Path('before-diagnostic-001');old.mkdir();shutil.copy2('build/src/athena',old/'athena')
for f in (p/'src').rglob('*'):
 if f.is_file():
  name=f.relative_to(p);saved=old/name;saved.parent.mkdir(parents=True,exist_ok=True)
  if (Path('source')/name).exists():shutil.copy2(Path('source')/name,saved)
  shutil.copy2(f,Path('source')/name);m['files'][str(name)]=hashlib.sha256(f.read_bytes()).hexdigest()
m['commit']='5d9de0d864cc998dd16869aaefa681b6c524b8ad'
m['change']='Optional in-process primary physical and reduction component histories; no equation changes'
for f,h in m['files'].items():assert hashlib.sha256((Path('source')/f).read_bytes()).hexdigest()==h,f
Path('source-diagnostic-001-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PY
cmake --build build -j4 > diagnostic-001-build.log 2>&1
sha256sum build/src/athena > diagnostic-001-binary.sha256
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > diagnostic-001-gpu-before.csv
python3 - <<'PY'
import csv
with open('diagnostic-001-gpu-before.csv') as f:
 rows=list(csv.reader(f));assert int(rows[1][-1].strip().split()[0])>=4096
PY
python3 -W error diagnostic-001/tests/check_intrinsic_diagnostic_task.py --binary "$PWD/build/src/athena" --fixtures "$PWD/decomposition-cuda-seeded-001" --output "$PWD/diagnostic-cuda-serial-001" > diagnostic-cuda-serial-001.log 2>&1
python3 -W error diagnostic-001/tests/check_intrinsic_diagnostic_task.py --binary "$PWD/build/src/athena" --fixtures "$PWD/decomposition-cuda-seeded-001" --launcher '/usr/local/openmpi/cuda-12.6/4.1.6/nvhpc2411/bin/mpiexec -n 2' --ranks 2 --output "$PWD/diagnostic-cuda-mpi-001" > diagnostic-cuda-mpi-001.log 2>&1
python3 -W error diagnostic-001/tests/check_intrinsic_diagnostic_controls.py --binary "$PWD/build/src/athena" --fixtures "$PWD/decomposition-cuda-seeded-001" --output "$PWD/diagnostic-cuda-controls-001" > diagnostic-cuda-controls-001.log 2>&1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > diagnostic-001-gpu-after.csv
