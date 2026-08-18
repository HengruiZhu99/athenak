#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-stage3-ac75c8d3-v9-20260818
prior=/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-parent-provenance-ac75c8d3-v1-20260818
source_root=${root}/source/athenak
build_root=${root}/build/athena-cuda
evidence=${root}/evidence
reference=${root}/run/same-executable-diagnostic-off
phase1=${root}/run/phase1-stage3-preupdate
prestate=${root}/run/precycle5546-state
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
input=${root}/input/pilot.athinput

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
test "$(sha256sum "${restart}" | awk '{print $1}')" = \
  2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9
test "$(sha256sum "${history}" | awk '{print $1}')" = \
  d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555
test "$(sha256sum "${shadow_requests}" | awk '{print $1}')" = \
  6d28a3743cc84dc3a111869f86ae8bc764e3c0db55a53196ff6b5461050ad483
test "$(sha256sum "${coeff}" | awk '{print $1}')" = \
  ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b
test "$(sha256sum "${input}" | awk '{print $1}')" = \
  edced480bbd934347aa80152dda4c164c4b6fd59c2a7abe764ac990983004791
test ! -e "${build_root}"
test ! -e "${root}/run"
mkdir -p "${evidence}" "${reference}/bindings" "${phase1}/bindings"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${evidence}/orchestration-status.txt"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final"
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
git -C "${source_root}" diff --binary HEAD > "${evidence}/diagnostic.patch"
sha256sum "${evidence}/diagnostic.patch" > "${evidence}/diagnostic.patch.sha256"
git -C "${source_root}" status --short > "${evidence}/source-status.initial"

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
  -R '^athena\.(amr_history_format|z4c_chi_prolongation|z4c_coarse_cache_ownership|z4c_chi_preupdate_diagnostic|z4c_chi_preupdate_diagnostic_static|z4c_cartoon_amr_static|z4c_amr_derefine_factor_static)$' \
  > "${evidence}/focused-tests.log" 2>&1
grep -F '100% tests passed, 0 tests failed out of 7' \
  "${evidence}/focused-tests.log" >/dev/null
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" > \
  "${evidence}/build-products.sha256"
printf 'CUDA_BUILD_AND_SEVEN_FOCUSED_TESTS_PASS\n' > "${evidence}/build-status.txt"

# Qualify the diagnostic against an off run made by the same executable.  This
# is a non-perturbation reference, not the diagnosis-selected numerical control.
unset ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC
unset ATHENA_CHI_PARENT_PROVENANCE_START_TIME
unset ATHENA_CHI_PARENT_PROVENANCE_OUTPUT
unset ATHENA_CHI_CONTROL_TARGET_TRACE
reference_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
  --gpu-bind=single:1 "${python_bin}" "${wrapper}"
  --evidence-dir "${reference}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}" -d "${reference}"
  job/basename=same-executable-diagnostic-off
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=same-executable-diagnostic-off-constraints.dat
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${reference_command[@]}" > "${reference}/command.txt"
printf '\n' >> "${reference}/command.txt"
reference_status=0
(cd "${reference}" && "${reference_command[@]}") > "${reference}/run.log" 2>&1 || \
  reference_status=$?
printf '%s\n' "${reference_status}" > "${reference}/exit-status.txt"
test "${reference_status}" = 1
test -s "${reference}/same-executable-diagnostic-off.z4c.user.hst"
sha256sum "${reference}/command.txt" "${reference}/run.log" \
  "${reference}/exit-status.txt" \
  "${reference}/same-executable-diagnostic-off.z4c.user.hst" > \
  "${reference}/terminal.sha256"

export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=10.53
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_stage3_diagnostic
phase1_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
  --gpu-bind=single:1 "${python_bin}" "${wrapper}"
  --evidence-dir "${phase1}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}" -d "${phase1}"
  job/basename=phase1-stage3-preupdate
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=phase1-stage3-preupdate-constraints.dat
  output2/dcycle=99999999 output3/dcycle=99999999
  output4/dcycle=99999999 output5/dcycle=99999999
  output6/dcycle=99999999 output7/dcycle=99999999
  output8/dcycle=99999999 time/tlim=10.54)
printf '%q ' "${phase1_command[@]}" > "${phase1}/command.txt"
printf '\n' >> "${phase1}/command.txt"
phase1_status=0
(cd "${phase1}" && "${phase1_command[@]}") > "${phase1}/run.log" 2>&1 || \
  phase1_status=$?
printf '%s\n' "${phase1_status}" > "${phase1}/exit-status.txt"
test "${phase1_status}" = 1
for name in rk_stage3_candidate_summary.json rk_accumulator_audit.csv \
  chi_rhs_term_decomposition.csv chi_candidate_counterfactuals.csv \
  local_stiffness_metrics.csv chi_stencil_values.csv \
  chi_stencil_owner_comparison.csv derivative_order_comparison.csv \
  ko_directional_comparison.csv local_patch_all_fields.csv phase1_disposition.json; do
  test -s "${phase1}/chi_stage3_diagnostic/${name}"
done

"${python_bin}" - "${phase1}/chi_stage3_diagnostic" \
  "${reference}/same-executable-diagnostic-off.z4c.user.hst" \
  "${prior}/run/n256-replay-provenance-terminal-v6/N256-replay-provenance-terminal-v6.z4c.user.hst" \
  "${phase1}/phase1-stage3-preupdate.z4c.user.hst" \
  "${evidence}/cross-build-history-comparison.json" <<'PY'
import csv, json, math, pathlib, struct, sys
root, reference_hist, old_hist, new_hist, comparison_out = map(pathlib.Path, sys.argv[1:])
summary = json.loads((root / "rk_stage3_candidate_summary.json").read_text())
assert summary["cycle"] == 5546 and summary["stage"] == 3
assert summary["time_hex"] == "0x1.5124ccccccd9bp+3"
assert summary["invalid_candidates"] == 2
for path in root.glob("*.json"):
    json.loads(path.read_text())
for path in root.glob("*.csv"):
    list(csv.reader(path.open()))
def rows(path):
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
reference, old, new = rows(reference_hist), rows(old_hist), rows(new_hist)
assert len(new) > 100 and new == reference[:len(new)]

def numbers(line):
    return [float(token) for token in line.split()]

def ordered_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (0xffffffffffffffff - bits) if bits & (1 << 63) else bits + (1 << 63)

common = min(len(reference), len(old))
max_abs = 0.0
max_rel = 0.0
max_ulp = 0
first_difference = None
for row_index, (left_line, right_line) in enumerate(zip(reference[:common], old[:common])):
    if left_line != right_line and first_difference is None:
        first_difference = row_index
    left, right = numbers(left_line), numbers(right_line)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        max_abs = max(max_abs, abs(a - b))
        max_rel = max(max_rel, abs(a - b) / max(abs(a), abs(b), 1.0))
        if math.isfinite(a) and math.isfinite(b):
            max_ulp = max(max_ulp, abs(ordered_bits(a) - ordered_bits(b)))
comparison = {
    "schema": "athenak_chi_cross_build_history_comparison_v1",
    "same_executable_off_on_exact_common_prefix": True,
    "same_executable_reference_rows": len(reference),
    "diagnostic_rows": len(new),
    "old_v6_rows": len(old),
    "old_v6_common_rows": common,
    "old_v6_exact_match": reference == old,
    "old_v6_first_differing_row": first_difference,
    "old_v6_max_absolute_difference": max_abs,
    "old_v6_max_scaled_difference": max_rel,
    "old_v6_max_ulp_difference": max_ulp,
    "old_v6_role": "cross_build_roundoff_context_not_identity_authority",
}
comparison_out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print("PHASE1_STRICT_SCHEMA_AND_SAME_EXECUTABLE_PREFailure_HISTORY_IDENTITY_PASS")
PY
sha256sum "${phase1}/command.txt" "${phase1}/run.log" \
  "${phase1}/exit-status.txt" "${phase1}/chi_stage3_diagnostic/"* > \
  "${phase1}/terminal.sha256"

classification=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' \
  "${phase1}/chi_stage3_diagnostic/phase1_disposition.json")
printf '%s\n' "${classification}" > "${evidence}/phase1-classification.txt"
if test "${classification}" = ADVECTION_DOMINATED_FAILURE || \
   test "${classification}" = REPLAY_TREE_UNDERREFINES_N256; then
  derived_history=${evidence}/targeted-refinement-history.jsonl
  derived_schedule_summary=${evidence}/targeted-refinement-schedule.json
  "${python_bin}" "${source_root}/scripts/build_targeted_refinement_control.py" \
    --authority "${history}" --shadow-requests "${shadow_requests}" \
    --output "${derived_history}" --summary "${derived_schedule_summary}" \
    > "${evidence}/targeted-refinement-schedule.log"
  sha256sum "${derived_history}" "${derived_schedule_summary}" > \
    "${evidence}/targeted-refinement-schedule.sha256"

  export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
  export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=10.53
  export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_targeted_refinement_control
  unset ATHENA_CHI_CONTROL_TARGET_TRACE
  mkdir -p "${control}/bindings"
  control_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
    --gpu-bind=single:1 "${python_bin}" "${wrapper}"
    --evidence-dir "${control}/bindings" --require-cuda --
    "${build_root}/src/athena" -r "${restart}" -d "${control}"
    job/basename=targeted-refinement-control
    problem/brill_global_coefficients_file="${coeff}"
    problem/constraint_summary_file=targeted-refinement-control-constraints.dat
    mesh_refinement/amr_history_mode=replay
    mesh_refinement/amr_history_file="${derived_history}"
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
  "${python_bin}" - "${control}" "${derived_schedule_summary}" \
    "${control_status}" "${evidence}/conditional_control_summary.json" <<'PY'
import csv, glob, json, pathlib, sys
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
assert len(target) == 1
assert target[0]["exact_match"] is True
assert target[0]["tree_checksum"] == schedule["tree_checksum"]
all_positive = all(int(row["nonpositive"]) == 0 and int(row["nonfinite"]) == 0
                   and float(row["candidate_min"]) > 0.0 for row in minima)
if status == 0:
    assert all_positive
    disposition = "ADVECTION_UNDERRESOLUTION_CONFIRMED"
    outcome = "targeted_early_refinement_reached_endpoint_with_positive_candidates"
else:
    assert (diag / "phase1_disposition.json").is_file()
    disposition = "NOT_ESTABLISHED"
    outcome = "targeted_early_refinement_did_not_remove_first_candidate_failure"
result = {
    "schema": "athenak_chi_conditional_control_v1",
    "phase1_classification": "ADVECTION_DOMINATED_FAILURE",
    "control": "single_targeted_refinement_at_first_recorded_request",
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
    "${evidence}/conditional_control_summary.json" > \
    "${control}/terminal.sha256"
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["disposition"])' \
    "${evidence}/conditional_control_summary.json" > \
    "${evidence}/final-disposition.txt"
  exit 0
fi
if test "${classification}" != RK_AFFINE_COMBINATION_FAILURE; then
  printf 'PHASE1_COMPLETE_CONTROL_REQUIRES_SEPARATE_IMPLEMENTATION\n' > \
    "${evidence}/phase1-only-status.txt"
  exit 42
fi

# Materialize the exact accepted state at the start of cycle 5546.  This is an
# unchanged RK4/replay continuation ending before the failing cycle, not a control arm.
unset ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC
unset ATHENA_CHI_PARENT_PROVENANCE_START_TIME
unset ATHENA_CHI_PARENT_PROVENANCE_OUTPUT
unset ATHENA_CHI_CONTROL_TARGET_TRACE
mkdir -p "${prestate}/bindings"
prestate_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
  --gpu-bind=single:1 "${python_bin}" "${wrapper}"
  --evidence-dir "${prestate}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${restart}" -d "${prestate}"
  job/basename=precycle5546-state
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=precycle5546-state-constraints.dat
  output1/dcycle=99999999 output2/dcycle=99999999
  output3/dcycle=99999999 output4/dcycle=99999999
  output5/dcycle=99999999 output6/dcycle=99999999
  output7/dcycle=99999999 output8/dcycle=99999999
  time/tlim=10.535742187500366)
printf '%q ' "${prestate_command[@]}" > "${prestate}/command.txt"
printf '\n' >> "${prestate}/command.txt"
(cd "${prestate}" && "${prestate_command[@]}") > "${prestate}/run.log" 2>&1
grep -F 'time=1.053574e+01 cycle=5546' "${prestate}/run.log" >/dev/null
mapfile -t prestate_restarts < <(find "${prestate}" -type f -name '*.rst' | sort)
test "${#prestate_restarts[@]}" = 1
precycle_restart=${prestate_restarts[0]}
sha256sum "${precycle_restart}" > "${prestate}/restart.sha256"

# Exactly one diagnosis-selected bounded control: one SSPRK3 step over the same
# physical dt and frozen replay hierarchy, ending at the matching physical time.
export ATHENA_CHI_PARENT_PROVENANCE_DIAGNOSTIC=1
export ATHENA_CHI_PARENT_PROVENANCE_START_TIME=10.535742187500366
export ATHENA_CHI_PARENT_PROVENANCE_OUTPUT=chi_ssprk3_control
export ATHENA_CHI_CONTROL_TARGET_TRACE=1
mkdir -p "${control}/bindings"
control_command=(srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1
  --gpu-bind=single:1 "${python_bin}" "${wrapper}"
  --evidence-dir "${control}/bindings"
  --require-cuda -- "${build_root}/src/athena" -r "${precycle_restart}"
  -d "${control}" job/basename=ssprk3-one-step-control time/integrator=rk3
  problem/brill_global_coefficients_file="${coeff}"
  problem/constraint_summary_file=ssprk3-one-step-control-constraints.dat
  output1/dcycle=99999999 output2/dcycle=99999999
  output3/dcycle=99999999 output4/dcycle=99999999
  output5/dcycle=99999999 output6/dcycle=99999999
  output7/dcycle=99999999 output8/dcycle=99999999
  time/tlim=10.536328125000366)
printf '%q ' "${control_command[@]}" > "${control}/command.txt"
printf '\n' >> "${control}/command.txt"
(cd "${control}" && "${control_command[@]}") > "${control}/run.log" 2>&1
test -s "${control}/chi_ssprk3_control/preupdate_candidate_minima.csv"
test -s "${control}/chi_ssprk3_control/control_target_stage_decomposition.csv"
"${python_bin}" - "${control}/chi_ssprk3_control" \
  "${evidence}/conditional_control_summary.json" <<'PY'
import csv, json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
minima = list(csv.DictReader((root / "preupdate_candidate_minima.csv").open()))
targets = list(csv.DictReader((root / "control_target_stage_decomposition.csv").open()))
assert len(minima) == 3
assert {int(row["rk_stage"]) for row in minima} == {1, 2, 3}
assert all(int(row["nonpositive"]) == 0 and int(row["nonfinite"]) == 0 for row in minima)
assert len(targets) == 6
assert {int(row["gid"]) for row in targets} == {35, 60}
result = {
    "schema": "athenak_chi_conditional_control_v1",
    "phase1_classification": "RK_AFFINE_COMBINATION_FAILURE",
    "control": "one_ssprk3_step_same_dt_same_tree",
    "cycle": 5546,
    "stages": 3,
    "all_stage_candidates_finite_positive": True,
    "minimum_candidate": min(float(row["candidate_min"]) for row in minima),
    "disposition": "RK_AFFINE_COMBINATION_CONFIRMED",
    "production_adoption_claim": False,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("SSPRK3_ONE_STEP_CONTROL_STRICT_PASS")
PY
sha256sum "${control}/command.txt" "${control}/run.log" \
  "${control}/chi_ssprk3_control/"* "${evidence}/conditional_control_summary.json" > \
  "${control}/terminal.sha256"
printf 'RK_AFFINE_COMBINATION_CONFIRMED\n' > "${evidence}/final-disposition.txt"
