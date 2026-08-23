#!/bin/bash
set -euo pipefail

authority_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${authority_root}/source/athenak
build_root=${authority_root}/build/current-cuda-mpi
run_root=${campaign_root}/runs/outer-onesided-n512
evidence_root=${campaign_root}/evidence/outer-onesided-n512
input=${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/fixed_grid_brill_dense.athinput
coefficient=${authority_root}/authority/brill_global_48x32.coefficients
athena=${build_root}/src/athena
required_commit=ba7daebc

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "${SLURM_GPUS:-0}" -eq 1
test "$(git -C "${source_root}" rev-parse --short=8 HEAD)" = "${required_commit}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}" && test -f "${input}" && test -f "${coefficient}"
test ! -e "${run_root}" && test ! -e "${evidence_root}"
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
sha256sum "${athena}" "${input}" "${coefficient}" \
  > "${evidence_root}/authority-products.sha256"

"${athena}" -i "${input}" -d "${run_root}" \
  job/basename=outer_onesided_N512 \
  mesh/nx1=512 mesh/nx2=1024 meshblock/nx1=128 meshblock/nx2=128 \
  problem/brill_global_coefficients_file="${coefficient}" \
  problem/constraint_summary_file="${run_root}/constraints.dat" \
  output2/dt=-1 output3/dt=-1 output4/dt=-1 \
  > "${evidence_root}/stdout.log" 2> "${evidence_root}/stderr.log"

test ! -e "${run_root}/z4c_state_failure.json"
history=${run_root}/outer_onesided_N512.z4c.user.hst
test -s "${history}"
tail -1 "${history}" > "${evidence_root}/terminal-history-row.txt"
printf '%s\n' OUTER_ONESIDED_N512_REACHED_T5 \
  > "${evidence_root}/verdict.txt"
