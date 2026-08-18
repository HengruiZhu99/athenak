#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-amr-history-ac75c8d3-v15-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
evidence=${root}/build/evidence
run=${root}/run/exact_timestep
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
iris_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/source/iris
iris_archive=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/build/iris-host/src/libiris_athenak_interpolator.a
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
coeff=${root}/input/brill_global_48x32.coefficients
input=${root}/bundle/pilot.athinput
identity_analyzer=${root}/bundle/analyze_n128_replay_identity.py
v14=/pscratch/sd/h/hzhu/axisymmetric-cartoon-amr-history-0396d281-v14-20260818

test "${SLURM_JOB_NAME:-}" = amrhist-ac75-v15
test "${SLURM_JOB_QOS:-}" = gpu_shared_interactive
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_TRES_PER_TASK:-}" = 'cpu=32,gres/gpu=1'
test "${SLURM_TRES_BIND:-}" = 'gres/gpu:map_gpu:0'
test "$(git -C "${source_root}" rev-parse HEAD)" = \
  ac75c8d348da91b38cbc6855b5fba51cd3089663
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = \
  6284882bd06e8db379495675aba7a4f153fb4afa
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = \
  6739bc623081648af9e752b616d9671527922cbf
test -z "$(git -C "${source_root}" status --short)"
test -z "$(git -C "${source_root}/kokkos" status --short)"
test "$(sha256sum "${coeff}" | awk '{print $1}')" = \
  ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
test "$(sha256sum "${input}" | awk '{print $1}')" = \
  edced480bbd934347aa80152dda4c164c4b6fd59c2a7abe764ac990983004791
test "$(sha256sum "${identity_analyzer}" | awk '{print $1}')" = \
  ed758cba7267022aa02fbc05d72480fa0db3720ca66076e63f7cddf3471c9c14
test "$(sha256sum "${v14}/run/n256_physical_time/SHA256SUMS" | awk '{print $1}')" = \
  151e12469229711b17a30d6aa99266d21e7996ed781994e38c34da2d2f3449f1
test "$(sha256sum "${v14}/run/n256_physical_time/SHA256SUMS.sha256" | awk '{print $1}')" = \
  d258ee11af7c171ff78b8bbebbe5ef15407e916f724bd2f8f358fcb83043e947
(cd "${v14}/run/n256_physical_time" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "${root}/build"
test ! -e "${run}"

mkdir -p "${evidence}" "${run}/evidence"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${run}/evidence/source-status.final"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${run}/evidence/kokkos-status.final"
  find "${run}" -type f \( -path '*/bin/*' -o -path '*/rst/*' \) \
    -printf '%s %p\n' | sort > "${run}/evidence/heavy-output-inventory.txt"
  find "${run}" -type f ! -path '*/bin/*' ! -path '*/rst/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${run}/SHA256SUMS"
  sha256sum "${run}/SHA256SUMS" > "${run}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${run}/evidence/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${run}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${run}/evidence/node.txt"

configure=(cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC
  -DPROBLEM=built_in_pgens -DBUILD_TESTING=ON -DAthena_BUILD_UNIT_TESTS=ON
  -DAthena_ENABLE_IRISK_INTERPOLATOR=ON -DIRISK_ROOT="${iris_root}"
  -DIRISK_INTERPOLATOR_LIBRARY="${iris_archive}"
  -DAthena_SINGLE_PRECISION=OFF -DAthena_ENABLE_MPI=ON
  -DAthena_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_OPENMP=ON
  -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ARCH_AMPERE80=ON)
printf '%q ' "${configure[@]}" > "${evidence}/configure-command.txt"
printf '\n' >> "${evidence}/configure-command.txt"
/usr/bin/time -p -o "${evidence}/configure-time.txt" "${configure[@]}" \
  > "${evidence}/configure.log" 2> "${evidence}/configure.err"

build=(cmake --build "${build_root}" --parallel 8 --target athena
  athena_amr_history_format_unit_test athena_z4c_chi_prolongation_unit_test
  athena_z4c_coarse_cache_ownership_unit_test)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" "${build[@]}" \
  > "${evidence}/build.log" 2> "${evidence}/build.err"

ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.(amr_history_format|z4c_chi_prolongation|z4c_coarse_cache_ownership|z4c_cartoon_amr_static|z4c_amr_derefine_factor_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 5' \
  "${evidence}/focused-tests.log" >/dev/null
integration=("${python_bin}"
  "${source_root}/tst/unit/mesh/amr_history_integration_test.py"
  --athena "${build_root}/src/athena"
  --input "${source_root}/tst/inputs/amr_history_lwave.athinput"
  --work-dir "${root}/tests/amr_history_integration")
printf '%q ' "${integration[@]}" > "${evidence}/integration-command.txt"
printf '\n' >> "${evidence}/integration-command.txt"
timeout 600 "${integration[@]}" > "${evidence}/integration.log" 2>&1
grep -Fx 'AMR_HISTORY_INTEGRATION_TEST_PASS' "${evidence}/integration.log" >/dev/null
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" \
  > "${evidence}/build-products.sha256"
printf 'BUILD_AND_FOCUSED_TESTS_PASS\n' > "${evidence}/status.txt"
find "${evidence}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${run}/evidence/build-and-tests.sha256"
sha256sum "${v14}/run/n256_physical_time/SHA256SUMS" \
  "${v14}/run/n256_physical_time/SHA256SUMS.sha256" \
  "${v14}/run/n256_physical_time/P-H_N256_replay/n256-replay-summary.json" \
  "${v14}/run/n256_physical_time/P-H_N256_replay/run.log" \
  > "${run}/evidence/v14-roundoff-failure.sha256"

run_case() {
  local name=$1
  local mode=$2
  local history=$3
  local wall_limit=$4
  shift 4
  local case_root=${run}/${name}
  mkdir -p "${case_root}/bindings"
  local command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1
    --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=map_gpu:0
    --cpu-bind=cores --exact --kill-on-bad-exit=1 --time="${wall_limit}"
    "${python_bin}" "${wrapper}" --evidence-dir "${case_root}/bindings"
    --require-cuda -- "${build_root}/src/athena" -i "${input}" -d "${case_root}"
    job/basename="${name}"
    problem/brill_global_coefficients_file="${coeff}"
    problem/constraint_summary_file="${name}-constraints.dat"
    mesh_refinement/amr_history_mode="${mode}"
    mesh_refinement/amr_history_file="${history}"
    output2/dcycle=4096 output3/dcycle=1024 output4/dcycle=1024
    output5/dcycle=2048 output6/dcycle=2048 output7/dcycle=1024
    output8/dcycle=1024 "$@")
  printf '%q ' "${command[@]}" > "${case_root}/command.txt"
  printf '\n' >> "${case_root}/command.txt"
  local status=0
  "${command[@]}" > "${case_root}/run.log" 2>&1 || status=$?
  printf '%s\n' "${status}" > "${case_root}/exit-status.txt"
  sha256sum "${case_root}/command.txt" "${case_root}/run.log" \
    "${case_root}/exit-status.txt" > "${case_root}/terminal.sha256"
  return "${status}"
}

record=${run}/P-R_N128_record
history=${record}/hierarchy.jsonl
record_status=0
run_case P-R_N128_record record "${history}" 01:10:00 || record_status=$?
test "${record_status}" = 1
grep -F 'AMR_HISTORY_RECORD event=210 time_hex=0x1.b3ec5e9999acfp+3 cycle=14715 leaves=977 max_level=22 checksum=9a6db3afa653129c' \
  "${record}/run.log" >/dev/null
grep -F 'cycle=14722' "${record}/run.log" | \
  grep -F 'invalid_parent_stencils=107' >/dev/null

read -r endpoint endpoint_hex final_event < <("${python_bin}" - "${history}" <<'PY'
import json
import sys
events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
last = [row for row in events if row.get("type") == "event"][-1]
print(last["time"], last["time_hex"], last["event"])
PY
)
test "${endpoint_hex}" = 0x1.b3ec5e9999acfp+3
test "${final_event}" = 210
printf '%s\n' \
  "endpoint_decimal=${endpoint}" \
  "endpoint_hex=${endpoint_hex}" \
  "last_recorded_event=${final_event}" \
  'authority=physical_time_not_cycle' \
  'record_terminal=strict_chi_failure_after_last_event' \
  > "${run}/evidence/common_endpoint.txt"
sha256sum "${history}" "${history}.ledger.jsonl" "${record}/run.log" \
  "${record}/terminal.sha256" > "${run}/evidence/authority.sha256"

pi_status=0
run_case P-I_N128_replay replay "${history}" 01:10:00 \
  time/tlim="${endpoint}" || pi_status=$?
if test "${pi_status}" -ne 0; then
  printf 'N128_REPLAY_FAILED\n' > "${run}/verdict.txt"
  exit 0
fi

"${python_bin}" "${identity_analyzer}" \
  --history "${history}" \
  --ledger "${run}/P-I_N128_replay/P-I_N128_replay.amr_history_replay.jsonl" \
  --authority "${record}" \
  --replay "${run}/P-I_N128_replay" \
  --output "${run}/n128-identity.json"

ph_status=0
run_case P-H_N256_replay replay "${history}" 03:10:00 \
  mesh/nx1=128 mesh/nx2=256 meshblock/nx1=64 meshblock/nx2=64 \
  time/tlim="${endpoint}" || ph_status=$?

"${python_bin}" - "${history}" "${run}/P-H_N256_replay" "${ph_status}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

history = pathlib.Path(sys.argv[1])
case = pathlib.Path(sys.argv[2])
status = int(sys.argv[3])
events = [json.loads(line) for line in history.read_text().splitlines()]
events = [row for row in events if row.get("type") == "event"][1:]
ledger_path = case / "P-H_N256_replay.amr_history_replay.jsonl"
ledger = ([json.loads(line) for line in ledger_path.read_text().splitlines()]
          if ledger_path.exists() else [])
expected = [(row["event"], row["time_hex"], row["leaf_count"],
             row["max_level"], row["tree_checksum"])
            for row in events[:len(ledger)]]
observed = [(row["event"], row["time_hex"], row["leaves"],
             row["max_level"], row["tree_checksum"])
            for row in ledger]
if expected != observed:
    raise SystemExit("N256 replay ledger is not an exact authority prefix")
log = (case / "run.log").read_text(errors="replace")
clips = log.count("AMR_HISTORY_TIMESTEP_CLIP")
cycles = [int(value) for value in re.findall(r"cycle=([0-9]+)", log)]
times = [float(value) for value in
         re.findall(r"time=([0-9]+\.[0-9]+e[+-][0-9]+)", log)]
completed = status == 0 and len(ledger) == len(events)
if status == 0 and not completed:
    raise SystemExit("N256 exited zero without replaying the complete schedule")
summary = {
    "schema": "brill_amr_history_n256_physical_time_v1",
    "disposition": "N256_REPLAY_ENDPOINT_PASS" if completed
                   else "N256_REPLAY_TERMINAL_FAILURE",
    "n128_identity": "pass",
    "authority_history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
    "authority_events": len(events),
    "endpoint_time_hex": events[-1]["time_hex"],
    "n256": {
        "status": status,
        "events_applied": len(ledger),
        "last_event": ledger[-1]["event"] if ledger else None,
        "last_event_time_hex": ledger[-1]["time_hex"] if ledger else None,
        "ledger_exact_authority_prefix": True,
        "physical_time_authority": True,
        "timestep_clip_count": clips,
        "last_reported_cycle": cycles[-1] if cycles else None,
        "last_reported_time": times[-1] if times else None,
    },
}
(case / "n256-replay-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(summary["disposition"])
PY
printf 'N128_IDENTITY_PASS_N256_EXECUTED\n' > "${run}/verdict.txt"
cat "${run}/P-H_N256_replay/n256-replay-summary.json"
exit "${ph_status}"
