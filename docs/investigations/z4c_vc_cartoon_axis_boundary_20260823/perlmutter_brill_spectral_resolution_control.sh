#!/bin/bash
set -euo pipefail

authority_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${authority_root}/source/athenak
build_root=${authority_root}/build/current-cuda-mpi
input=${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/fixed_grid_brill_dense.athinput
coefficient=${ATHENA_BRILL_SPECTRAL_CONTROL_COEFFICIENT:?set coefficient path}
tag=${ATHENA_BRILL_SPECTRAL_CONTROL_TAG:?set deterministic tag}
run_root=${campaign_root}/runs/brill-spectral-resolution-control-${tag}
evidence_root=${campaign_root}/evidence/brill-spectral-resolution-control-${tag}
athena=${build_root}/src/athena
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "${SLURM_GPUS:-0}" -eq 1
git -C "${source_root}" merge-base --is-ancestor \
  bd1ba697ed4d3315844deae7d10ef89b9cad2106 HEAD
test -z "$(git -C "${source_root}" status --short)"
test -x "${athena}"
test -f "${input}"
test -f "${coefficient}"
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
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
git -C "${source_root}" rev-parse HEAD > "${evidence_root}/source-commit.txt"
sha256sum "${athena}" "${build_root}/CMakeCache.txt" "${input}" \
  "${coefficient}" > "${evidence_root}/authority-products.sha256"

for resolution in 128 256 512; do
  meshblock=$((resolution / 4))
  stride=$((resolution / 128))
  run=${run_root}/N${resolution}
  basename=brill_spectral_${tag}_N${resolution}
  mkdir -p "${run}"
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC=${run}/rhs
  export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE=${stride}
  (
    cd "${run}"
    "${athena}" -i "${input}" -d "${run}" \
      job/basename="${basename}" \
      mesh/nx1="${resolution}" mesh/nx2="$((2 * resolution))" \
      meshblock/nx1="${meshblock}" meshblock/nx2="${meshblock}" \
      problem/brill_global_coefficients_file="${coefficient}" \
      problem/constraint_summary_file="${run}/${basename}.constraints.dat" \
      time/nlim=1 time/tlim=1.0 \
      z4c/rhs_stage_diagnostics=true \
      z4c/rhs_stage_diagnostics_start_time=0.0 \
      z4c/rhs_stage_diagnostics_rho_max=16.0 \
      z4c/rhs_stage_diagnostics_abs_z_max=16.0 \
      output1/dcycle=1 output2/dt=-1 output3/dt=-1 output4/dt=-1 \
      > "${evidence_root}/N${resolution}.stdout.log" \
      2> "${evidence_root}/N${resolution}.stderr.log"
  )
  test ! -e "${run}/z4c_state_failure.json"
  test -s "${run}/rhs.rank000000.csv"
  test -s "${run}/z4c_rhs_stage_rank0.log"
  test -s "${run}/${basename}.constraints.dat"
  grep -q '^Z4C_AXIS_RHS_PHASE_DIAGNOSTIC ' \
    "${run}/z4c_rhs_stage_rank0.log"
  grep -q '^Z4C_AXIS_TERM_POINT_DIAGNOSTIC ' \
    "${run}/z4c_rhs_stage_rank0.log"
  gzip -9 "${run}/rhs.rank000000.csv" \
    "${run}/z4c_rhs_stage_rank0.log"
  sha256sum "${run}/rhs.rank000000.csv.gz" \
    "${run}/z4c_rhs_stage_rank0.log.gz" \
    "${run}/${basename}.constraints.dat" \
    "${run}/${basename}.z4c.user.hst" \
    > "${evidence_root}/N${resolution}.products.sha256"
done

printf '%s\n' BRILL_SPECTRAL_RESOLUTION_ZERO_TIME_RHS_CONTROL_CAPTURED \
  > "${evidence_root}/verdict.txt"
