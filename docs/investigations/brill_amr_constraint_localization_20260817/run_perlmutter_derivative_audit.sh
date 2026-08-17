#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-constraint-localization-55f9147b-v4-20260817
source_root=${root}/source/athenak
build_root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-constraint-localization-55f9147b-v3-20260817/build/athena-cuda
run_root=${root}/run
evidence=${run_root}/evidence
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
expected_commit=55f9147bc80d574636c47bcd1dac86178d921988
expected_tree=cb2ad270f0675230b77023877dc0fdf93b52cd59
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
expected_restart=83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea

test "${SLURM_JOB_NAME:-}" = cartoon-brill-localize-55f914-v4
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 32
test "${SLURM_GPUS_PER_NODE:-}" = 1
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${root}/build"
test ! -e "${root}/run"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_commit}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -z "$(git -C "${source_root}/kokkos" status --short)"
test "$(sha256sum "${root}/input/brill_n256_pre_event_c1721.00014.rst" | awk '{print $1}')" = \
  "${expected_restart}"
test "$(sha256sum "${build_root}/src/athena" | awk '{print $1}')" = \
  638bdf0d60daba67f2c20cbfbe127a3e0f65991832fe632af0085b2639aa2e4d
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = \
  d8dc2031d44734891ea3d9e7aea9c2a5d473e8e23eaed4e23707bc776bc6ac1f
test "$(sha256sum "${root}/predecessor_v3_build/focused-tests.log" | awk '{print $1}')" = \
  b210e04728a3ddb2b730887280dfec42ab1f27f6224d8eb836e3adad0377775d
grep -F '100% tests passed, 0 tests failed out of 5' \
  "${root}/predecessor_v3_build/focused-tests.log" >/dev/null

mkdir -p "${evidence}"
finish() {
  status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${status}" > "${run_root}/allocation.status"
  git -C "${source_root}" status --porcelain=v1 > "${evidence}/source-status.final"
  git -C "${source_root}/kokkos" status --porcelain=v1 > \
    "${evidence}/kokkos-status.final"
  find "${run_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > "${run_root}/SHA256SUMS"
  sha256sum "${run_root}/SHA256SUMS" > "${run_root}/SHA256SUMS.sha256"
  exit "${status}"
}
trap finish EXIT

export COLLAPSE_ROOT=${root}
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU
env | sort > "${evidence}/environment.txt"
module -t list > "${evidence}/modules.txt" 2>&1
scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence}/hosts.txt"
node=$(head -1 "${evidence}/hosts.txt")
scontrol show node "${node}" -o > "${evidence}/node.txt"
grep -E 'ActiveFeatures=[^ ]*gpu' "${evidence}/node.txt" >/dev/null
grep -E 'ActiveFeatures=[^ ]*hbm40g' "${evidence}/node.txt" >/dev/null

printf 'v3_fresh_build_and_focused_tests_verified|0\n' >> \
  "${evidence}/phase-status.tsv"

case_root=${run_root}/zero_pde_derivative_audit
mkdir -p "${case_root}/bindings"
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
executable=${build_root}/src/athena
command=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 --time=00:25:00 "${python_bin}" "${wrapper}"
  --evidence-dir "${case_root}/bindings" --require-cuda -- "${executable}"
  -r "${root}/input/brill_n256_pre_event_c1721.00014.rst"
  -i "${root}/input/continuation_derivative_audit.athinput" -d "${case_root}"
  job/basename=brill_n256_constraint_localization_zero_pde
  problem/brill_global_coefficients_file="${root}/input/brill_global_48x32.coefficients"
  problem/constraint_summary_file=brill_n256_constraint_localization-constraints.dat
  z4c/amr_jump_output_basename=z4c_amr_constraint_localization
  z4c/amr_jump_derivative_order_audit=true
  z4c/amr_jump_post_cycles=0)
printf '%q ' "${command[@]}" > "${case_root}/command.txt"
printf '\n' >> "${case_root}/command.txt"
"${command[@]}" > "${case_root}/run.log" 2>&1
printf 'zero_pde_derivative_audit|0\n' >> "${evidence}/phase-status.tsv"

event=$(find "${case_root}/z4c_amr_constraint_localization/rank0000" -maxdepth 1 \
  -type d -name 'event_c00001722_*' -print -quit)
test -n "${event}"
grep -F 'after T5 and before the next RHS' "${case_root}/run.log" >/dev/null
for phase in t3_06_PHYSICAL_OR_AXIS_BC t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION; do
  for order in o2 o4 o6; do
    test -s "${event}/${phase}/constraints_${order}.bin"
  done
  cmp -s "${event}/${phase}/constraints.bin" "${event}/${phase}/constraints_o6.bin"
  grep -F '"derivative_order_audit":true' "${event}/${phase}/phase.json" >/dev/null
done
sha256sum "${executable}" "${build_root}/CMakeCache.txt" \
  "${root}/input/brill_n256_pre_event_c1721.00014.rst" \
  "${root}/input/continuation_derivative_audit.athinput" \
  "${event}/t3_06_PHYSICAL_OR_AXIS_BC/constraints_o2.bin" \
  "${event}/t3_06_PHYSICAL_OR_AXIS_BC/constraints_o4.bin" \
  "${event}/t3_06_PHYSICAL_OR_AXIS_BC/constraints_o6.bin" \
  > "${evidence}/qualification.sha256"
printf 'terminal|0\n' >> "${evidence}/phase-status.tsv"
