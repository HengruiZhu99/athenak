#!/bin/bash
set -euo pipefail

prior_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${prior_root}/source/athenak
build_root=${prior_root}/build/current-cuda-mpi
tag=${ATHENA_DOMAIN_CC_TAG:-default}
run_root=${campaign_root}/runs/domain-cc-controls-${tag}
evidence_root=${campaign_root}/evidence/domain-cc-controls-${tag}
input=${source_root}/docs/investigations/z4c_vc_cartoon_axis_boundary_20260823/fixed_grid_brill_dense.athinput
coefficient=${ATHENA_DOMAIN_CC_COEFFICIENT:-${prior_root}/authority/brill_global_48x32.coefficients}
extrap_order=${ATHENA_DOMAIN_CC_EXTRAP_ORDER:-2}
athena=${build_root}/src/athena
required_source_fix=1392f5c472353fec1cdc44108b403a316f33fc46
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
target_time=2.0295751268186133

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
git -C "${source_root}" merge-base --is-ancestor "${required_source_fix}" HEAD
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
sha256sum "${athena}" "${build_root}/CMakeCache.txt" "${input}" "${coefficient}" \
  > "${evidence_root}/authority-products.sha256"

validate() {
  local state=$1
  local expected_nx1=$2
  "${python_bin}" - "${state}" "${target_time}" "${expected_nx1}" \
      "${source_root}/vis/python" <<'PY'
from pathlib import Path
import math, sys
sys.path.insert(0, sys.argv[4])
import bin_convert
data = bin_convert.read_binary(sys.argv[1])
assert math.isclose(data["time"], float(sys.argv[2]), rel_tol=0.0, abs_tol=2e-14)
assert data["Nx1"] == int(sys.argv[3]) and data["Nx3"] == 1
assert all(math.isfinite(float(value))
           for name in data["var_names"]
           for block in data["mb_data"][name]
           for value in block.ravel())
print(data["time"], data["cycle"], data["Nx1"], data["Nx2"], data["n_mbs"])
PY
}

for geometry in vc_large cc_base; do
  for resolution in 128 256 512; do
    meshblock=$((resolution / 4))
    run=${run_root}/${geometry}/N${resolution}
    basename=${geometry}_N${resolution}
    mkdir -p "${run}"
    common=(
      job/basename="${basename}"
      time/tlim="${target_time}"
      meshblock/nx1="${meshblock}" meshblock/nx2="${meshblock}"
      problem/brill_global_coefficients_file="${coefficient}"
      problem/constraint_summary_file="${run}/${basename}.constraints.dat"
      z4c/extrap_order="${extrap_order}"
      output1/dcycle=4
      output2/variable=z4c output2/id=state output2/dt=1.0e99
      output3/variable=con output3/id=constraints output3/dt=1.0e99
      output4/id=restart output4/dt=1.0e99
    )
    if test "${geometry}" = vc_large; then
      expected_nx1=$((2 * resolution))
      geometry_args=(
        z4c/grid_centering=vertex
        mesh/nx1="${expected_nx1}" mesh/nx2="$((4 * resolution))"
        mesh/x1min=0.0 mesh/x1max=32.0
        mesh/x2min=-32.0 mesh/x2max=32.0
      )
    else
      expected_nx1=${resolution}
      geometry_args=(
        z4c/grid_centering=cell
        mesh/nx1="${resolution}" mesh/nx2="$((2 * resolution))"
        mesh/x1min=0.0 mesh/x1max=16.0
        mesh/x2min=-16.0 mesh/x2max=16.0
      )
    fi
    start=$(date +%s)
    "${athena}" -i "${input}" -d "${run}" \
      "${common[@]}" "${geometry_args[@]}" \
      > "${evidence_root}/${geometry}.N${resolution}.stdout.log" \
      2> "${evidence_root}/${geometry}.N${resolution}.stderr.log"
    end=$(date +%s)
    printf '%s\n' "$((end - start))" \
      > "${evidence_root}/${geometry}.N${resolution}.wall-seconds.txt"
    test ! -e "${run}/z4c_state_failure.json"
    state=$(find "${run}/bin" -name '*.state.*.bin' -print | sort | tail -1)
    constraints=$(find "${run}/bin" -name '*.constraints.*.bin' -print | sort | tail -1)
    restart=$(find "${run}/rst" -name '*.rst' -print | sort | tail -1)
    test -n "${state}" && test -n "${constraints}" && test -n "${restart}"
    validate "${state}" "${expected_nx1}" \
      > "${evidence_root}/${geometry}.N${resolution}.validation.txt"
    sha256sum "${state}" "${constraints}" "${restart}" \
      "${run}/${basename}.z4c.user.hst" \
      "${run}/${basename}.constraints.dat" \
      > "${evidence_root}/${geometry}.N${resolution}.products.sha256"
  done
done

printf '%s\n' 'LARGE_DOMAIN_VC_AND_MATCHED_CC_CONTROLS_REACHED_TARGET_TIME' \
  > "${evidence_root}/verdict.txt"
