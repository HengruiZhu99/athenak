#!/bin/bash
# Run inside one already-granted four-node Perlmutter GPU allocation.
# The two matched continuations run sequentially, each on all sixteen A100s,
# so the generic controlled-reference cache does not exceed 40-GB device
# memory.  This is a bounded discriminator, not a stability campaign.

set -euo pipefail

: "${CAMPAIGN_ROOT:?set a fresh directory under PSCRATCH}"
: "${EXPECTED_COMMIT:?set the exact pushed source commit}"

readonly SRC="${CAMPAIGN_ROOT}/src_clean"
readonly BUILD="${CAMPAIGN_ROOT}/build_${SLURM_JOB_ID}"
readonly OUT="${CAMPAIGN_ROOT}/reference_motion_freeze_${SLURM_JOB_ID}"
readonly EXE="${BUILD}/src/athena"
readonly INPUT="${SRC}/inputs/ref_gh/ref_gh_relative_damped_hard_freeze_outer24.athinput"
readonly SOURCE_INPUT="${SRC}/inputs/ref_gh/ref_gh_source_unit.athinput"
readonly ANALYZER="${SRC}/scripts/ref_gh/analyze_reference_motion_freeze.py"

if [[ $(hostname) != nid* ]]; then
  echo "refusing substantial work outside a Perlmutter compute node" >&2
  exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-0} -ne 4 ]]; then
  echo "this script requires exactly four allocated nodes" >&2
  exit 2
fi

mkdir -p "${OUT}"
cd "${OUT}"
exec > >(tee -a bootstrap.log) 2>&1
trap 'status=$?; echo "reference_motion_freeze_failure status=${status} line=${LINENO} command=${BASH_COMMAND}"; exit "${status}"' ERR

module load cudatoolkit
export CRAYPE_LINK_TYPE=dynamic
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

test "$(git -C "${SRC}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${SRC}" status --porcelain)"
test -f "${SRC}/kokkos/CMakeLists.txt"

{
  date -Is
  hostname
  echo "allocation=${SLURM_JOB_ID}"
  echo "qos=${SLURM_JOB_QOS:-unknown}"
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "nodelist=${SLURM_JOB_NODELIST}"
  echo "account=${SLURM_JOB_ACCOUNT:-unknown}"
  echo "campaign_root=${CAMPAIGN_ROOT}"
  echo "source_commit=${EXPECTED_COMMIT}"
  echo "common_restart=fresh moving-reference trajectory generated in this allocation"
  echo "fork_physical_time=2.0M"
  echo "freeze_xi=0.25"
  echo "continued_xi_dot=0.125/M"
  echo "hard_freeze_xi_dot=0"
  echo "target_time=4.2M"
  echo "cases_run_sequentially_on_all_16_A100s=true"
  git -C "${SRC}" status --short --branch
  git -C "${SRC}" submodule status
  module -t list 2>&1
  cc --version | head -1
  CC --version | head -1
  nvcc --version | tail -1
  cmake --version | head -1
  scontrol show job "${SLURM_JOB_ID}"
  sha256sum "${INPUT}" "${SOURCE_INPUT}" "${ANALYZER}" "$0"
} > provenance.txt

cmake -S "${SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=built_in_pgens \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_OPENMP=OFF \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER="${SRC}/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF > configure.log 2>&1
cmake --build "${BUILD}" --parallel 32 > build.log 2>&1

{
  grep -E 'CMAKE_(C|CXX)_COMPILER:|CMAKE_BUILD_TYPE:|Athena_ENABLE_MPI:|Athena_ENABLE_OPENMP:|Kokkos_ENABLE_(CUDA|SERIAL|OPENMP|DEBUG_BOUNDS_CHECK):|Kokkos_ARCH_AMPERE80:|PROBLEM:' \
    "${BUILD}/CMakeCache.txt"
  sha256sum "${BUILD}/CMakeCache.txt" "${EXE}"
  ldd "${EXE}"
} >> provenance.txt
grep -q '^Athena_ENABLE_MPI:BOOL=ON$' "${BUILD}/CMakeCache.txt"
grep -q '^Kokkos_ENABLE_CUDA:BOOL=ON$' "${BUILD}/CMakeCache.txt"
grep -q '^Kokkos_ARCH_AMPERE80:BOOL=ON$' "${BUILD}/CMakeCache.txt"

srun -N 4 -n 16 -c 32 --gpus-per-task=1 --gpu-bind=none \
  bash -lc 'echo "host=$(hostname) rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader' \
  > rank_gpu_mapping.txt 2>&1
test "$(grep -c '^host=' rank_gpu_mapping.txt)" -eq 16
test "$(grep 'GPU-' rank_gpu_mapping.txt | sed -n 's/.*\(GPU-[0-9a-f-]*\).*/\1/p' | sort -u | wc -l)" -eq 16
test "$(grep '^host=' rank_gpu_mapping.txt | sed -n 's/^host=\([^ ]*\).*/\1/p' | sort -u | wc -l)" -eq 4
awk '/^host=/ {split($1, field, "="); count[field[2]]++}
     END {for (host in count) if (count[host] != 4) exit 1}' \
  rank_gpu_mapping.txt

mkdir source_oracle
(
  cd source_oracle
  srun -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=none \
    "${EXE}" --kokkos-map-device-id-by=mpi_rank -i "${SOURCE_INPUT}" \
    > source_unit.log 2>&1
  grep -q 'controlled hard-freeze time-derivative oracle passed exactly' \
    source_unit.log
  ! grep -Eiq 'FATAL ERROR|CUDA error|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
    source_unit.log
)

# Generate the common pre-growth state on this machine.  Both branches below
# read this exact file; no cross-machine restart transfer is involved.
mkdir seed_to_t2
(
  cd seed_to_t2
  /usr/bin/time -p srun -N 4 -n 16 -c 32 \
    --gpus-per-task=1 --gpu-bind=none \
    "${EXE}" --kokkos-map-device-id-by=mpi_rank -i "${INPUT}" \
    job/basename=refgh_reference_motion_seed \
    ref_gh/continuation_mode=legacy_time \
    time/nlim=-1 time/tlim=2.0 output1/dt=0.02 output2/dt=2.0 \
    > run.log 2>&1
  latest_time=$(awk 'NF && $1 !~ /^#/ {value=$1} END {print value+0}' \
    refgh_reference_motion_seed.ref_gh.hst)
  python3 - "${latest_time}" <<'PY'
import math
import sys
time = float(sys.argv[1])
if not math.isfinite(time) or abs(time-2.0) > 1.0e-12:
    raise SystemExit(f"common seed did not reach t=2M: {time}")
PY
)
readonly RESTART_FILE=$(find "${OUT}/seed_to_t2/rst" -maxdepth 1 \
  -type f -name '*.rst' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
test -r "${RESTART_FILE}"
readonly RESTART_SHA256=$(sha256sum "${RESTART_FILE}" | awk '{print $1}')
{
  echo "restart_file=${RESTART_FILE}"
  echo "restart_sha256=${RESTART_SHA256}"
  stat --printf='restart_size_bytes=%s\n' "${RESTART_FILE}"
} >> provenance.txt

run_case() {
  local label=$1
  local mode=$2
  local directory="${OUT}/${label}"
  local basename="refgh_reference_motion_${label}"
  local status

  mkdir "${directory}"
  cd "${directory}"
  set +e
  if [[ ${mode} == hard_freeze ]]; then
    /usr/bin/time -p srun -N 4 -n 16 -c 32 \
      --gpus-per-task=1 --gpu-bind=none \
      "${EXE}" --kokkos-map-device-id-by=mpi_rank \
      -r "${RESTART_FILE}" -i "${INPUT}" \
      job/basename="${basename}" \
      ref_gh/continuation_mode=hard_freeze \
      ref_gh/continuation_xi=0.25 \
      ref_gh/continuation_xi_dot=0.0 \
      ref_gh/hard_freeze_previous_xi_dot=0.125 \
      ref_gh/hard_freeze_previous_xi_ddot=0.0 \
      ref_gh/hard_freeze_reference_already_static=false \
      time/nlim=-1 time/tlim=4.2 output1/dt=0.02 output2/dt=0.5 \
      > run.log 2>&1
    status=$?
  else
    /usr/bin/time -p srun -N 4 -n 16 -c 32 \
      --gpus-per-task=1 --gpu-bind=none \
      "${EXE}" --kokkos-map-device-id-by=mpi_rank \
      -r "${RESTART_FILE}" -i "${INPUT}" \
      job/basename="${basename}" \
      ref_gh/continuation_mode=legacy_time \
      time/nlim=-1 time/tlim=4.2 output1/dt=0.02 output2/dt=0.5 \
      > run.log 2>&1
    status=$?
  fi
  set -e
  {
    echo "case=${label}"
    echo "continuation_mode=${mode}"
    echo "run_exit_status=${status}"
    echo "latest_ref_history_time=$(awk 'NF && $1 !~ /^#/ {value=$1} END {print value+0}' "${basename}.ref_gh.hst" 2>/dev/null || echo 0)"
    echo "latest_power_history_time=$(awk 'NF && $1 ~ /^[0-9.+-]/ {value=$1} END {print value+0}' "${basename}.ref_gh_power.hst" 2>/dev/null || echo 0)"
    echo "remote_directory=${directory}"
  } > run_status.txt
  find rst -maxdepth 1 -type f -name '*.rst' -printf '%p\t%s bytes\n' \
    2>/dev/null | sort > restart_sizes.tsv
  cd "${OUT}"
}

run_case continued legacy_time
run_case hard_freeze hard_freeze

python3 "${ANALYZER}" \
  --case "continued=${OUT}/seed_to_t2,${OUT}/continued" \
  --case "hard_freeze=${OUT}/seed_to_t2,${OUT}/hard_freeze" \
  --freeze-time 2.0 --post-start 2.15 --post-end 4.2 \
  --output-prefix "${OUT}/reference_motion_freeze" > analysis.log 2>&1
python3 -m json.tool reference_motion_freeze.json >/dev/null

python3 - reference_motion_freeze.json > scientific_status.txt <<'PY'
import json
import math
import pathlib
import sys

cases = json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]
for label in ("continued", "hard_freeze"):
    case = cases[label]
    final = case["final"]
    motion = case["reference_motion_at_first_post_freeze"]
    print(f"{label}_final_time={final['time']:.17e}")
    print(f"{label}_GH_RMS={final['GH_RMS']:.17e}")
    print(f"{label}_reduction_RMS={final['reduction_RMS']:.17e}")
    print(f"{label}_curl_RMS={final['curl_RMS']:.17e}")
    print(f"{label}_reference_dt_frame={motion['dt_frame_max']}")
    print(f"{label}_reference_dt_connection={motion['dt_connection_max']}")
    for quantity in ("GH_RMS", "reduction_RMS", "curl_RMS"):
        fit = case["growth_fits"][quantity]["full_post_freeze"]
        print(f"{label}_{quantity}_growth_per_M={fit['slope_per_M']}")
print("claim_boundary=bounded matched reference-motion discriminator only")
print("stable_or_convergent_trumpet_claim=false")
PY

find seed_to_t2 continued hard_freeze source_oracle -type f \
  \( -name '*.hst' -o -name '*.tsv' -o -name '*.txt' -o -name '*.log' \) \
  -print0 | sort -z | xargs -0 sha256sum > compact_sha256.txt
sha256sum provenance.txt rank_gpu_mapping.txt configure.log build.log \
  analysis.log scientific_status.txt reference_motion_freeze.json \
  reference_motion_freeze_growth.tsv reference_motion_freeze.png \
  >> compact_sha256.txt
echo "reference_motion_freeze_complete=$(date -Is)"
