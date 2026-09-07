#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-transfer-001
trap 'printf "%s\n" "$?" > test-accessfix.exit' EXIT
while [ ! -f build-accessfix.exit ]; do
  if ! kill -0 605504 2>/dev/null; then
    echo "Build controller missing without terminal status; stop for diagnosis"
    exit 3
  fi
  sleep 5
done
test "$(cat build-accessfix.exit)" = 0
python3 - <<'PY'
import json,hashlib,pathlib
p=pathlib.Path('source'); manifest=json.load(open('source-accessfix-manifest.json'))
assert all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
print('PASS: source snapshot unchanged before GPU tests',flush=True)
PY
for ranks in 1 2; do
  for topology in uniform smr; do
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "gpu-before-${ranks}-${topology}-002.csv"
    python3 - "$ranks" "$topology" <<'PY'
import pathlib,sys
row=pathlib.Path(f'gpu-before-{sys.argv[1]}-{sys.argv[2]}-002.csv').read_text().strip().split(',')
assert len(row)==5 and int(row[3].split()[0])>=4096, 'insufficient GPU memory for bounded fixtures'
PY
    label="cuda-${ranks}rank-${topology}-002"
    extra=()
    if [ "$topology" = smr ]; then extra+=(--smr); fi
    python3 source/analysis/pc_gh_clean_reduction/run_transfer_mesh.py \
      --binary "$PWD/build/src/athena" --output "$PWD/$label" --ranks "$ranks" \
      "${extra[@]}" > "$label.log" 2>&1
    python3 - "$label" <<'PY'
import json,pathlib,sys
rows=json.loads((pathlib.Path(sys.argv[1])/'results.json').read_text())
assert len(rows)==6 and all(r['status']=='PASS' for r in rows), 'transfer fixture failure; no further promotion'
print('PASS:',sys.argv[1],flush=True)
PY
  done
done
