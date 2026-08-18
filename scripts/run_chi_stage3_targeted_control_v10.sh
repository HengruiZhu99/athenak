#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-stage3-ac75c8d3-v14-20260818
v9=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-stage3-ac75c8d3-v9-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
exe=${build_root}/src/athena
cache=${build_root}/CMakeCache.txt
v9_exe=${v9}/build/athena-cuda/src/athena
v9_cache=${v9}/build/athena-cuda/CMakeCache.txt
evidence=${root}/evidence
control=${root}/run/conditional-control
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
iris_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/source/iris
iris_archive=/pscratch/sd/h/hzhu/axisymmetric-cartoon-r3-r4-brill-f3-9448c1e6-v6-20260812/build/iris-host/src/libiris_athenak_interpolator.a
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
restart=${root}/input/n256-replay-cycle4096.rst
history=${root}/input/n128-authority-hierarchy.jsonl
shadow_requests=${root}/input/n256-shadow-amr-requests.jsonl
coeff=${root}/input/brill_global_48x32.coefficients

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS:-}" = 1
scontrol show job "${SLURM_JOB_ID}" -o | grep -F 'QOS=gpu_shared_interactive' >/dev/null
test "$(git -C "${source_root}" rev-parse HEAD)" = \
  ac75c8d348da91b38cbc6855b5fba51cd3089663
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = \
  6284882bd06e8db379495675aba7a4f153fb4afa
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = \
  6739bc623081648af9e752b616d9671527922cbf
test "$(git -C "${source_root}" diff --binary HEAD | sha256sum | awk '{print $1}')" = \
  d4e0db8efdac39317a33dce18a9df4c86b86b30d177e96174fb615304ab8c328
test "$(sha256sum "${v9}/SHA256SUMS" | awk '{print $1}')" = \
  104fc41525880c39b76e4e8d08e04e1cca5fae7ad3177f1c03c9703f4e1a7dbb
test "$(sha256sum "${v9}/SHA256SUMS.sha256" | awk '{print $1}')" = \
  e284a4a4797523aebf0ad774fde25e9e507867603a6717ee0e1f3f9a9dcd46e4
test "$(sha256sum "${v9_exe}" | awk '{print $1}')" = \
  aab5704fa8684aea5cbdb5a1dfcbd89cc0b7b243d3f9ac4310dfe81a6d266d28
test "$(sha256sum "${v9_cache}" | awk '{print $1}')" = \
  95531979a24734bc2450eb38962fc29f9bdf7134c0583d55c6cb9553f8033035
test "$(sha256sum "${restart}" | awk '{print $1}')" = \
  2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9
test "$(sha256sum "${history}" | awk '{print $1}')" = \
  d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555
test "$(sha256sum "${shadow_requests}" | awk '{print $1}')" = \
  6d28a3743cc84dc3a111869f86ae8bc764e3c0db55a53196ff6b5461050ad483
test "$(sha256sum "${coeff}" | awk '{print $1}')" = \
  ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
test ! -e "${root}/run" && test ! -e "${build_root}"
mkdir -p "${evidence}" "${control}/bindings"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${evidence}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  find "${root}" -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 \
    -print0 | sort -z | xargs -0 sha256sum > "${root}/SHA256SUMS"
  sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${evidence}/node.txt"
nvidia-smi -L > "${evidence}/nvidia-smi-L.txt"
sha256sum "${v9_exe}" "${v9_cache}" "${v9}/SHA256SUMS" \
  "${v9}/SHA256SUMS.sha256" > "${evidence}/v9-products.sha256"

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
  athena_z4c_coarse_cache_ownership_unit_test
  athena_z4c_chi_preupdate_diagnostic_unit_test)
printf '%q ' "${build[@]}" > "${evidence}/build-command.txt"
printf '\n' >> "${evidence}/build-command.txt"
/usr/bin/time -p -o "${evidence}/build-time.txt" "${build[@]}" \
  > "${evidence}/build.log" 2> "${evidence}/build.err"
ctest --test-dir "${build_root}" --output-on-failure \
  -R '^athena\.(amr_history_format|amr_history_extension_static|z4c_chi_prolongation|z4c_coarse_cache_ownership|z4c_chi_preupdate_diagnostic|z4c_chi_preupdate_diagnostic_static|z4c_cartoon_amr_static|z4c_amr_derefine_factor_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 8' \
  "${evidence}/focused-tests.log" >/dev/null
sha256sum "${exe}" "${cache}" > "${evidence}/build-products.sha256"

# GPU history reductions are not bitwise deterministic.  Require exact time,
# timestep, cycle, topology/count/location columns, and a 1024-epsilon envelope
# for every floating reduction.  The first differences precede diagnostic
# activation, so they cannot have been caused by the diagnostic branch.
"${python_bin}" - \
  "${v9}/run/same-executable-diagnostic-off/same-executable-diagnostic-off.z4c.user.hst" \
  "${v9}/run/phase1-stage3-preupdate/phase1-stage3-preupdate.z4c.user.hst" \
  "${v9}/run/same-executable-diagnostic-off/same-executable-diagnostic-off.amr_history_replay.jsonl" \
  "${v9}/run/phase1-stage3-preupdate/phase1-stage3-preupdate.amr_history_replay.jsonl" \
  "${v9}/run/phase1-stage3-preupdate/chi_stage3_diagnostic/rk_stage3_candidate_summary.json" \
  "${v9}/run/phase1-stage3-preupdate/chi_stage3_diagnostic/phase1_disposition.json" \
  "${evidence}/diagnostic-qualification.json" <<'PY'
import json, math, pathlib, sys
off_hst, on_hst, off_replay, on_replay, summary_path, disposition_path, output = \
    map(pathlib.Path, sys.argv[1:])
def rows(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines()
            if line and not line.startswith("#")]
off, on = rows(off_hst), rows(on_hst)
assert len(off) == len(on) == 362
exact_columns = {0, 1, 12, 14, 15, 16, 17, 18, 26, 36, 41, 46, 51, 56,
                 58, 59, 61, 62, 64, 65, 67, 68}
maximum_scaled = 0.0
first_difference = None
for row_index, (left, right) in enumerate(zip(off, on)):
    assert len(left) == len(right) == 71
    for column, (a, b) in enumerate(zip(left, right)):
        if column in exact_columns:
            assert a == b
        difference = abs(a - b)
        scaled = difference / max(abs(a), abs(b), 1.0)
        maximum_scaled = max(maximum_scaled, scaled)
        if difference and first_difference is None:
            first_difference = [row_index, column]
limit = 1024.0 * sys.float_info.epsilon
assert first_difference is not None and off[first_difference[0]][0] < 10.53
assert maximum_scaled <= limit
off_events = [json.loads(line) for line in off_replay.read_text().splitlines() if line]
on_events = [json.loads(line) for line in on_replay.read_text().splitlines() if line]
assert len(off_events) == len(on_events)
for left, right in zip(off_events, on_events):
    for key in ("action", "event", "time_hex", "leaves", "max_level",
                "tree_checksum", "ranks", "exact_match"):
        assert left[key] == right[key]
summary = json.loads(summary_path.read_text())
disposition = json.loads(disposition_path.read_text())
assert summary["classification"] == disposition["classification"] == \
       "ADVECTION_DOMINATED_FAILURE"
assert summary["cycle"] == 5546 and summary["stage"] == 3
assert summary["invalid_candidates"] == 2
assert summary["copy_exact"] is True and summary["stencil_owner_exact"] is True
result = {
    "schema": "athenak_chi_diagnostic_qualification_v1",
    "same_executable": True,
    "history_rows": len(on),
    "exact_discrete_topology_and_location_columns": sorted(exact_columns),
    "floating_reduction_limit": limit,
    "maximum_scaled_floating_reduction_difference": maximum_scaled,
    "first_difference_row_column": first_difference,
    "first_difference_precedes_diagnostic_start": True,
    "replay_events_exact": True,
    "phase1_classification": summary["classification"],
    "qualification_pass": True,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("V9_DIAGNOSTIC_ULP_QUALIFICATION_PASS")
PY

derived_history=${evidence}/targeted-refinement-history.jsonl
derived_summary=${evidence}/targeted-refinement-schedule.json
"${python_bin}" "${source_root}/scripts/build_targeted_refinement_control.py" \
  --authority "${history}" --shadow-requests "${shadow_requests}" \
  --output "${derived_history}" --summary "${derived_summary}" > \
  "${evidence}/targeted-refinement-schedule.log"
sha256sum "${derived_history}" "${derived_summary}" > \
  "${evidence}/targeted-refinement-schedule.sha256"

export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=10.53
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_targeted_refinement_control
unset ATHENA_CHI_CONTROL_TARGET_TRACE
export ATHENA_AMR_HISTORY_EXTENSION_FILE="${derived_history}"
export ATHENA_AMR_HISTORY_BRANCH_BASE_EVENT=11
control_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
  --gpu-bind=single:1 "${python_bin}" "${wrapper}"
  --evidence-dir "${control}/bindings" --require-cuda --
  "${exe}" -r "${restart}" -d "${control}"
  job/basename=targeted-refinement-control
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=targeted-refinement-control-constraints.dat
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${history}"
  output1/dcycle=99999999 output2/dcycle=99999999
  output3/dcycle=99999999 output4/dcycle=99999999
  output5/dcycle=99999999 output6/dcycle=99999999
  output7/dcycle=99999999 output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${control_command[@]}" > "${control}/command.txt"
printf '\n' >> "${control}/command.txt"
control_status=0
(cd "${control}" && "${control_command[@]}") > "${control}/run.log" 2>&1 || \
  control_status=$?
printf '%s\n' "${control_status}" > "${control}/exit-status.txt"
test "${control_status}" = 0 || test "${control_status}" = 1
test -s "${control}/chi_targeted_refinement_control/preupdate_candidate_minima.csv"
"${python_bin}" - "${control}" "${derived_summary}" "${control_status}" \
  "${evidence}/conditional_control_summary.json" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
schedule = json.loads(pathlib.Path(sys.argv[2]).read_text())
status = int(sys.argv[3])
output = pathlib.Path(sys.argv[4])
diag = root / "chi_targeted_refinement_control"
minima = list(csv.DictReader((diag / "preupdate_candidate_minima.csv").open()))
assert minima
ledger_paths = list(root.glob("*.amr_history_replay.jsonl"))
assert len(ledger_paths) == 1
ledger = [json.loads(line) for line in ledger_paths[0].read_text().splitlines() if line]
target = [row for row in ledger if row["event"] == schedule["target_event"]]
assert len(target) == 1 and target[0]["exact_match"] is True
assert target[0]["tree_checksum"] == schedule["tree_checksum"]
all_positive = all(int(row["nonpositive"]) == 0 and
                   int(row["nonfinite"]) == 0 and
                   float(row["candidate_min"]) > 0.0 for row in minima)
if status == 0:
    assert all_positive
    disposition = "ADVECTION_UNDERRESOLUTION_CONFIRMED"
    outcome = "targeted_earlier_refinement_reached_endpoint_with_positive_candidates"
else:
    assert (diag / "phase1_disposition.json").is_file()
    disposition = "NOT_ESTABLISHED"
    outcome = "targeted_earlier_refinement_did_not_remove_candidate_failure"
result = {
    "schema": "athenak_chi_conditional_control_v1",
    "phase1_classification": "ADVECTION_DOMINATED_FAILURE",
    "control": "single_targeted_refinement_at_first_retained_native_request",
    "control_native_status": status,
    "target_event": schedule["target_event"],
    "target_time_hex": schedule["target_time_hex"],
    "target_tree_checksum": schedule["tree_checksum"],
    "preupdate_samples": len(minima),
    "minimum_candidate": min(float(row["candidate_min"]) for row in minima),
    "all_sampled_candidates_finite_positive": all_positive,
    "outcome": outcome,
    "disposition": disposition,
    "production_adoption_claim": False,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("TARGETED_REFINEMENT_CONDITIONAL_CONTROL_STRICT_PASS")
PY
sha256sum "${control}/command.txt" "${control}/run.log" \
  "${control}/exit-status.txt" "${control}/chi_targeted_refinement_control/"* \
  "${evidence}/conditional_control_summary.json" > "${control}/terminal.sha256"
"${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["disposition"])' \
  "${evidence}/conditional_control_summary.json" > "${evidence}/final-disposition.txt"
