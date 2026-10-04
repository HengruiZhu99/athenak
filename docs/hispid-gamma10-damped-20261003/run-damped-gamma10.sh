#!/usr/bin/env bash
set -euo pipefail
PUNCTURE_FINDER_ROOT=/pscratch/sd/h/hzhu/codex-hispid-kokkos-20261002
PUNCTURE_FINDER_SOURCE="$PUNCTURE_FINDER_ROOT/athenak-gamma10-damped-2aada0e6"
PUNCTURE_FINDER_OUTPUT="$PUNCTURE_FINDER_ROOT/gamma10-damped-physics-20261003"
cd "$PUNCTURE_FINDER_SOURCE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
scontrol show job "$SLURM_JOB_ID" > "$PUNCTURE_FINDER_OUTPUT/allocation.txt"
PUNCTURE_FINDER_STATUS=0
srun --mpi=none --resv-ports=0 -n1 -c1 --gpus=1 --cpu-bind=cores \
  python3 tst/test_suite/z4c/check_hispid_controls.py \
  --executable "$PUNCTURE_FINDER_ROOT/athenak-build-extreme-serial-20261003/src/athena" \
  --manifest "$PUNCTURE_FINDER_ROOT/extreme-human-override-20261003/consumer-analytic-seeds-v2/manifest.json" \
  --cases gamma10 --boost-levels 96 --diagnostic-single-boost-level \
  --boost-flow-alpha .02 --flow-iterations 200 --worker-timeout 900 \
  --harmonic-storage factorized --full-precision-trace \
  --output "$PUNCTURE_FINDER_OUTPUT/controls" \
  > "$PUNCTURE_FINDER_OUTPUT/control.log" 2>&1 || PUNCTURE_FINDER_STATUS=$?
printf '%s\n' "$PUNCTURE_FINDER_STATUS" > "$PUNCTURE_FINDER_OUTPUT/wrapper.exit"
printf '%s\n' 'single_Gamma10_diagnosis_terminal' > "$PUNCTURE_FINDER_OUTPUT/phase.txt"
python3 - "$PUNCTURE_FINDER_OUTPUT/controls/controls.json" <<'REPORT'
import json,sys
evidence=json.load(open(sys.argv[1]))
print(json.dumps(dict(aggregate_qualified=evidence['passed'],records=evidence['records'])),flush=True)
REPORT
