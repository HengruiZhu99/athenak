#!/bin/bash
# Run inside one already-granted four-node Perlmutter GPU allocation.  This is
# the bounded direct-fixed plus smooth-stop discriminator only.

set -euo pipefail

: "${CAMPAIGN_ROOT:?set a fresh directory under PSCRATCH}"
: "${EXPECTED_COMMIT:?set the exact pushed source commit}"
: "${COMMON_RESTART:?set the existing clean t=2M moving-reference restart}"
: "${COMMON_RESTART_SHA256:?set its authoritative SHA-256}"

readonly SRC="${CAMPAIGN_ROOT}/src_clean"
readonly BUILD="${CAMPAIGN_ROOT}/build_${SLURM_JOB_ID}"
readonly OUT="${CAMPAIGN_ROOT}/fixed_reference_discriminator_${SLURM_JOB_ID}"
readonly EXE="${BUILD}/src/athena"
readonly DIRECT_INPUT="${SRC}/inputs/ref_gh/ref_gh_relative_damped_direct_fixed_outer24.athinput"
readonly SMOOTH_INPUT="${SRC}/inputs/ref_gh/ref_gh_relative_damped_smooth_stop_outer24.athinput"
readonly SOURCE_INPUT="${SRC}/inputs/ref_gh/ref_gh_source_unit.athinput"
readonly ANALYZER="${SRC}/scripts/ref_gh/analyze_fixed_reference_discriminator.py"
readonly OLD="${SRC}/artifacts/ref_gh_reference_motion_freeze_perlmutter_20260831"

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
trap 'status=$?; echo "fixed_reference_discriminator_failure status=${status} line=${LINENO} command=${BASH_COMMAND}"; exit "${status}"' ERR

module load cudatoolkit
export CRAYPE_LINK_TYPE=dynamic
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

test "$(git -C "${SRC}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${SRC}" status --porcelain)"
test -f "${SRC}/kokkos/CMakeLists.txt"
test -r "${COMMON_RESTART}"
test "$(sha256sum "${COMMON_RESTART}" | awk '{print $1}')" = \
  "${COMMON_RESTART_SHA256}"

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
  echo "common_restart=${COMMON_RESTART}"
  echo "common_restart_sha256=${COMMON_RESTART_SHA256}"
  echo "direct_fixed_xi=0.25"
  echo "smooth_stop_start=2.0M"
  echo "smooth_stop_end=3.0M"
  echo "smooth_stop_final_xi=0.3125"
  echo "direct_target_time=4.2M"
  echo "smooth_target_time=5.2M"
  echo "cases_run_sequentially_on_all_16_A100s=true"
  git -C "${SRC}" status --short --branch
  git -C "${SRC}" submodule status
  module -t list 2>&1
  cc --version | head -1
  CC --version | head -1
  nvcc --version | tail -1
  cmake --version | head -1
  scontrol show job "${SLURM_JOB_ID}"
  sha256sum "${DIRECT_INPUT}" "${SMOOTH_INPUT}" "${SOURCE_INPUT}" \
    "${ANALYZER}" "$0"
  stat --printf='common_restart_size_bytes=%s\n' "${COMMON_RESTART}"
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
  bash -lc 'echo "host=$(hostname) rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --id="${SLURM_LOCALID}" --query-gpu=index,uuid,name,memory.total --format=csv,noheader' \
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
  grep -q 'controlled direct-fixed projection passed' source_unit.log
  grep -q 'controlled smooth-stop oracle passed' source_unit.log
  ! grep -Eiq 'FATAL ERROR|CUDA error|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
    source_unit.log
)

run_direct() {
  mkdir direct_fixed
  cd direct_fixed
  set +e
  /usr/bin/time -p srun -N 4 -n 16 -c 32 \
    --gpus-per-task=1 --gpu-bind=none \
    "${EXE}" --kokkos-map-device-id-by=mpi_rank -i "${DIRECT_INPUT}" \
    job/basename=refgh_direct_fixed_xi025 \
    time/nlim=-1 time/tlim=4.2 output1/dt=0.02 output2/dt=0.5 \
    > run.log 2>&1
  status=$?
  set -e
  {
    echo "case=direct_fixed"
    echo "run_exit_status=${status}"
    echo "latest_ref_history_time=$(awk 'NF && $1 !~ /^#/ {value=$1} END {print value+0}' refgh_direct_fixed_xi025.ref_gh.hst 2>/dev/null || echo 0)"
    echo "latest_power_history_time=$(awk 'NF && $1 ~ /^[0-9.+-]/ {value=$1} END {print value+0}' refgh_direct_fixed_xi025.ref_gh_power.hst 2>/dev/null || echo 0)"
    echo "remote_directory=${OUT}/direct_fixed"
  } > run_status.txt
  find rst -maxdepth 1 -type f -name '*.rst' -printf '%p\t%s bytes\n' \
    2>/dev/null | sort > restart_sizes.tsv
  cd "${OUT}"
}

run_smooth() {
  mkdir smooth_stop
  cd smooth_stop
  set +e
  /usr/bin/time -p srun -N 4 -n 16 -c 32 \
    --gpus-per-task=1 --gpu-bind=none \
    "${EXE}" --kokkos-map-device-id-by=mpi_rank \
    -r "${COMMON_RESTART}" -i "${SMOOTH_INPUT}" \
    job/basename=refgh_smooth_stop \
    ref_gh/continuation_mode=smooth_stop \
    ref_gh/continuation_xi=0.25 ref_gh/continuation_xi_dot=0.125 \
    ref_gh/continuation_frozen=false \
    ref_gh/continuation_smooth_stop_start=2.0 \
    ref_gh/continuation_smooth_stop_duration=1.0 \
    ref_gh/continuation_smooth_stop_initial_xi=0.25 \
    ref_gh/continuation_smooth_stop_initial_rate=0.125 \
    time/nlim=-1 time/tlim=5.2 output1/dt=0.02 output2/dt=0.5 \
    > run.log 2>&1
  status=$?
  set -e
  {
    echo "case=smooth_stop"
    echo "run_exit_status=${status}"
    echo "latest_ref_history_time=$(awk 'NF && $1 !~ /^#/ {value=$1} END {print value+0}' refgh_smooth_stop.ref_gh.hst 2>/dev/null || echo 0)"
    echo "latest_power_history_time=$(awk 'NF && $1 ~ /^[0-9.+-]/ {value=$1} END {print value+0}' refgh_smooth_stop.ref_gh_power.hst 2>/dev/null || echo 0)"
    echo "remote_directory=${OUT}/smooth_stop"
  } > run_status.txt
  find rst -maxdepth 1 -type f -name '*.rst' -printf '%p\t%s bytes\n' \
    2>/dev/null | sort > restart_sizes.tsv
  cd "${OUT}"
}

run_direct
# The local startup already established a large discrete projection/gauge
# impulse, so the controlling decision tree requires the smooth-stop branch.
run_smooth

python3 "${ANALYZER}" \
  --case "direct_fixed=${OUT}/direct_fixed" \
  --case "continued=${OLD}/seed_to_t2,${OLD}/continued" \
  --case "hard_freeze=${OLD}/seed_to_t2,${OLD}/hard_freeze" \
  --case "smooth_stop=${OLD}/seed_to_t2,${OUT}/smooth_stop" \
  --window direct_fixed:startup=0.15:1.0 \
  --window direct_fixed:late=2.8:4.2 \
  --window continued:post_fork=2.15:4.2 \
  --window hard_freeze:post_fork=2.15:4.2 \
  --window smooth_stop:during_stop=2.05:2.9 \
  --window smooth_stop:post_clear=3.8:5.2 \
  --stop-start 2.0 --stop-end 3.0 \
  --output-prefix "${OUT}/fixed_reference_discriminator" > analysis.log 2>&1
python3 -m json.tool fixed_reference_discriminator.json >/dev/null

python3 - fixed_reference_discriminator.json > scientific_status.txt <<'PY'
import json
import pathlib
import sys

cases = json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]
for label in ("direct_fixed", "continued", "hard_freeze", "smooth_stop"):
    case = cases[label]
    final = case["final"]
    print(f"{label}_final_time={final['time']:.17e}")
    print(f"{label}_GH_RMS={final['GH_RMS']:.17e}")
    print(f"{label}_reduction_RMS={final['reduction_RMS']:.17e}")
    print(f"{label}_curl_RMS={final['curl_RMS']:.17e}")
    print(f"{label}_final_xi={final['xi']:.17e}")
    print(f"{label}_final_xi_dot={final['xi_dot']:.17e}")
print("claim_boundary=bounded fixed-reference causal discriminator only")
print("stable_or_convergent_trumpet_claim=false")
PY

find direct_fixed smooth_stop source_oracle -type f \
  \( -name '*.hst' -o -name '*.tsv' -o -name '*.txt' -o -name '*.log' \
     -o -name '*controlled-transition.dat' \) \
  -print0 | sort -z | xargs -0 sha256sum > compact_sha256.txt
sha256sum provenance.txt rank_gpu_mapping.txt configure.log build.log \
  analysis.log scientific_status.txt fixed_reference_discriminator.json \
  fixed_reference_discriminator_growth.tsv >> compact_sha256.txt
if [[ -f fixed_reference_discriminator.png ]]; then
  sha256sum fixed_reference_discriminator.png >> compact_sha256.txt
fi
echo "fixed_reference_discriminator_complete=$(date -Is)"
