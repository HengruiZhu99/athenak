#!/bin/bash
set -euo pipefail

prior_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${prior_root}/source/athenak
build_root=${prior_root}/build/current-cuda-mpi
checkpoint_root=${campaign_root}/runs/exact-localization
run_root=${campaign_root}/runs/axis-phase-replay-tau0
evidence_root=${campaign_root}/evidence/axis-phase-replay-tau0
athena=${build_root}/src/athena
required_commit=4bf4dabb94b9306c680540dbb87924edef9b8fcf
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "${SLURM_GPUS:-0}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${required_commit}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}"
test ! -e "${run_root}"
test ! -e "${evidence_root}"
mkdir -p "${run_root}" "${evidence_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
git -C "${source_root}" rev-parse HEAD > "${evidence_root}/source-commit.txt"
sha256sum "${athena}" > "${evidence_root}/authority-products.sha256"

read_state_time_cycle() {
  "${python_bin}" - "$1" <<'PY'
from pathlib import Path
import sys

with Path(sys.argv[1]).open("rb") as stream:
    stream.readline()
    count = int(stream.readline().split(b"=")[-1])
    header = {}
    for _ in range(count - 1):
        key, value = stream.readline().decode().split("=")
        header[key.strip()] = value.strip()
print(header["time"], header["cycle"])
PY
}

for resolution in 128 256 512; do
  exact=${checkpoint_root}/N${resolution}/tau0000/exact
  restart=$(find "${exact}/rst" -name '*.rst' -print | sort | tail -1)
  state=$(find "${exact}/bin" -name '*.state.*.bin' -print | sort | tail -1)
  test -n "${restart}" && test -f "${restart}"
  test -n "${state}" && test -f "${state}"
  read -r target cycle <<< "$(read_state_time_cycle "${state}")"
  diagnostic=${run_root}/N${resolution}/tau0000
  mkdir -p "${diagnostic}"
  (
    cd "${diagnostic}"
    "${athena}" -r "${restart}" -d "${diagnostic}" \
      job/basename="axis_phase_N${resolution}_tau0000" \
      time/nlim="$((cycle + 1))" time/tlim=1.0 \
      z4c/rhs_stage_diagnostics=true \
      z4c/rhs_stage_diagnostics_start_time="${target}" \
      z4c/rhs_stage_diagnostics_rho_max=16.0 \
      z4c/rhs_stage_diagnostics_abs_z_max=16.0 \
      output1/dcycle=-1 output2/dt=-1 output3/dt=-1 output4/dt=-1 \
      > "${evidence_root}/N${resolution}.stdout.log" \
      2> "${evidence_root}/N${resolution}.stderr.log"
  )
  log=${diagnostic}/z4c_rhs_stage_rank0.log
  test -s "${log}"
  grep -q '^Z4C_AXIS_RHS_PHASE_DIAGNOSTIC ' "${log}"
  grep -q '^Z4C_AXIS_TERM_POINT_DIAGNOSTIC ' "${log}"
  gzip -9 "${log}"
  sha256sum "${restart}" "${state}" "${log}.gz" \
    > "${evidence_root}/N${resolution}.products.sha256"
  printf '%s %.17g %s\n' "${resolution}" "${target}" "${cycle}" \
    >> "${evidence_root}/checkpoint-inventory.txt"
done

printf '%s\n' AXIS_PHASE_AND_TERM_TAU0_PROVENANCE_CAPTURED \
  > "${evidence_root}/verdict.txt"
