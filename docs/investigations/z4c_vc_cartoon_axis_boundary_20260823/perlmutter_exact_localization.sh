#!/bin/bash
set -euo pipefail

prior_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
campaign_root=/pscratch/sd/h/hzhu/z4c-vc-cartoon-axis-boundary-20260823
source_root=${prior_root}/source/athenak
build_root=${prior_root}/build/current-cuda-mpi
base_root=${ATHENA_EXACT_BASE_ROOT:-${campaign_root}/runs/fixed-grid-gate}
run_root=${ATHENA_EXACT_RUN_ROOT:-${campaign_root}/runs/exact-localization}
evidence_root=${ATHENA_EXACT_EVIDENCE_ROOT:-${campaign_root}/evidence/exact-localization}
athena=${build_root}/src/athena
required_source_fix=1392f5c472353fec1cdc44108b403a316f33fc46
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
git -C "${source_root}" merge-base --is-ancestor "${required_source_fix}" HEAD
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
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
nvidia-smi -L > "${evidence_root}/nvidia-smi.txt"
git -C "${source_root}" rev-parse HEAD > "${evidence_root}/source-commit.txt"
sha256sum "${athena}" "${build_root}/CMakeCache.txt" \
  > "${evidence_root}/authority-products.sha256"

targets=(
  '0000 0'
  '0125 0.2029627528643316'
  '0250 0.40592555529852264'
  '0500 0.81185053572369925'
  '0750 1.2177724040009463'
  '1000 1.6236846569991397'
  '1250 2.0295751268186133'
)

select_restart() {
  local resolution=$1
  local target=$2
  "${python_bin}" - "${base_root}/N${resolution}" "${target}" <<'PY'
from pathlib import Path
import sys

root, target = Path(sys.argv[1]), float(sys.argv[2])

def read_time(path):
    with path.open("rb") as stream:
        stream.readline()
        count = int(stream.readline().split(b"=")[-1])
        header = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split("=")
            header[key.strip()] = value.strip()
    return float(header["time"])

states = sorted((root / "bin").glob("*.state.*.bin"))
eligible = [path for path in states if read_time(path) <= target + 1.0e-14]
if not eligible:
    raise RuntimeError("no restart precedes exact target")
index = eligible[-1].name.split(".")[-2]
restarts = sorted((root / "rst").glob(f"*.{index}.rst"))
if len(restarts) != 1:
    raise RuntimeError(f"restart index {index} is not unique")
print(restarts[0])
PY
}

read_state_time_cycle() {
  "${python_bin}" - "$1" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
with path.open("rb") as stream:
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
  case "${resolution}" in
    128) stride=1 ;;
    256) stride=2 ;;
    512) stride=4 ;;
  esac
  for specification in "${targets[@]}"; do
    read -r tag target <<< "${specification}"
    base_restart=$(select_restart "${resolution}" "${target}")
    exact=${run_root}/N${resolution}/tau${tag}/exact
    diagnostic=${run_root}/N${resolution}/tau${tag}/diagnostic
    mkdir -p "${exact}" "${diagnostic}"
    basename=exact_N${resolution}_tau${tag}

    (
      cd "${exact}"
      "${athena}" -r "${base_restart}" -d "${exact}" \
        job/basename="${basename}" time/tlim="${target}" time/nlim=-1 \
        output1/dcycle=1000000007 \
        output2/file_type=bin output2/variable=z4c output2/id=state \
        output2/dt=1.0e99 output2/last_time="${target}" \
        output3/file_type=bin output3/variable=con output3/id=constraints \
        output3/dt=1.0e99 output3/last_time="${target}" \
        output4/file_type=rst output4/id=restart \
        output4/dt=1.0e99 output4/last_time="${target}" \
        > "${evidence_root}/N${resolution}.tau${tag}.exact.stdout.log" \
        2> "${evidence_root}/N${resolution}.tau${tag}.exact.stderr.log"
    )
    state=$(find "${exact}/bin" -name '*.state.*.bin' -print | sort | tail -1)
    constraints=$(find "${exact}/bin" -name '*.constraints.*.bin' -print | sort | tail -1)
    restart=$(find "${exact}/rst" -name '*.rst' -print | sort | tail -1)
    test -n "${state}" && test -n "${constraints}" && test -n "${restart}"
    read -r observed_time observed_cycle <<< "$(read_state_time_cycle "${state}")"
    "${python_bin}" - "${target}" "${observed_time}" <<'PY'
import math, sys
target, observed = map(float, sys.argv[1:])
assert math.isclose(target, observed, rel_tol=0.0, abs_tol=2.0e-14), (target, observed)
PY

    export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC=${diagnostic}/rhs
    export ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE=${stride}
    (
      cd "${diagnostic}"
      "${athena}" -r "${restart}" -d "${diagnostic}" \
        time/nlim="$((observed_cycle + 1))" time/tlim="$(${python_bin} - "${target}" <<'PY'
import sys
print(float(sys.argv[1]) + 1.0)
PY
)" \
        z4c/rhs_stage_diagnostics=true \
        z4c/rhs_stage_diagnostics_start_time="${target}" \
        z4c/rhs_stage_diagnostics_rho_max=16.0 \
        z4c/rhs_stage_diagnostics_abs_z_max=16.0 \
        output1/dcycle=-1 output2/dt=-1 output3/dt=-1 output4/dt=-1 \
        > "${evidence_root}/N${resolution}.tau${tag}.diagnostic.stdout.log" \
        2> "${evidence_root}/N${resolution}.tau${tag}.diagnostic.stderr.log"
    )
    test -f "${diagnostic}/rhs.rank000000.csv"
    test -f "${diagnostic}/z4c_rhs_stage_rank0.log"
    gzip -9 "${diagnostic}/rhs.rank000000.csv" \
      "${diagnostic}/z4c_rhs_stage_rank0.log"
    sha256sum "${state}" "${constraints}" "${restart}" \
      "${diagnostic}/rhs.rank000000.csv.gz" \
      "${diagnostic}/z4c_rhs_stage_rank0.log.gz" \
      > "${evidence_root}/N${resolution}.tau${tag}.products.sha256"
    printf '%s %s %s %s %s\n' "${resolution}" "${tag}" "${target}" \
      "${observed_time}" "${observed_cycle}" \
      >> "${evidence_root}/exact-targets.txt"
  done
done

printf '%s\n' 'EXACT_COMMON_TIME_STATE_CONSTRAINT_RHS_DIAGNOSTICS_CAPTURED' \
  > "${evidence_root}/verdict.txt"
