#!/bin/bash
# Execute only after the recorded Gate 1 review. No downstream runs are launched.
set -e
source /home/hz0693/athenak_env
set -u
root=/scratch/gpfs/FPRETORI/hz0693/pcgh-direct-lapse-20260905
cd "$root/source"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('..')
assert json.loads((root/'gate1.json').read_text())['decision']=='PASS'
assert hashlib.sha256((root/'inputs/stress/core256-R16.athinput').read_bytes()).hexdigest()=='ff69d9d88a75e4156735454b19b7cd91522d133ce4ff0f8fd9ced1e2d7105f80'
assert hashlib.sha256(Path('build-direct-cuda/src/athena').read_bytes()).hexdigest()=='0bfd4c5ce9c9e55285614fb988ebd90624a12640f21a41eb60d674148e4e1788'
PY
set +e
python3 analysis/pc_gh_regular_extension/cuda_driver.py run \
  --build build-direct-cuda --input "$root/inputs/stress/core256-R16.athinput" \
  --output "$root/stress/core256-R16" --wall-segment 00:15:00
run_exit=$?
set -e
RUN_EXIT="$run_exit" python3 - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
Path('../stress-exit.json').write_text(json.dumps(dict(exit_code=int(os.environ['RUN_EXIT']),completed_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
PY
exit "$run_exit"
