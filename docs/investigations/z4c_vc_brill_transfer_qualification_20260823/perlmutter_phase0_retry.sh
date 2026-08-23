#!/bin/bash
set -euo pipefail

campaign_root=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
source_root=${campaign_root}/source/athenak
build_root=${campaign_root}/build/phase0-host-base
evidence_root=${campaign_root}/evidence/phase0-host-retry
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
python_packages=${campaign_root}/python-packages
expected_source=2d59f85c11cb0da4614c84a695d64f032fb9eec7
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf

test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES}" -eq 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --short)"
test -x "${build_root}/src/athena"
test ! -e "${evidence_root}"
mkdir -p "${evidence_root}"

finish() {
  status=$?
  set +e
  printf '%s\n' "${status}" > "${evidence_root}/exit-status.txt"
  find "${evidence_root}" -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 -r sha256sum > "${evidence_root}/SHA256SUMS"
  exit "${status}"
}
trap finish EXIT

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=false
export KOKKOS_NUM_THREADS=8
export PYTHONPATH=${python_packages}

test "$("${python_bin}" -c 'import sympy; print(sympy.__version__)')" = 1.14.0
scontrol show job "${SLURM_JOB_ID}" > "${evidence_root}/slurm-job.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${evidence_root}/hosts.txt"
env | sort > "${evidence_root}/environment.txt"
"${python_bin}" -c \
  'import sympy,h5py,numpy; print(sympy.__version__,h5py.__version__,numpy.__version__); print(sympy.__file__)' \
  > "${evidence_root}/python-packages.txt"
sha256sum "${build_root}/src/athena" "${build_root}/CMakeCache.txt" \
  > "${evidence_root}/build-products.sha256"

ctest_command=(
  ctest --test-dir "${build_root}" --output-on-failure -j 4
  -E '^athena[.](z4c_cc_selector_equivalence|z4c_rhs_policy_production_host_exact|z4c_restart_carrier)$'
)
printf '%q ' "${ctest_command[@]}" > "${evidence_root}/ctest-command.txt"
printf '\n' >> "${evidence_root}/ctest-command.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 bash -c \
  'set -euo pipefail; export OMP_NUM_THREADS=8 OMP_PROC_BIND=false KOKKOS_NUM_THREADS=8 PYTHONPATH="$1"; shift; "$@"' \
  bash "${python_packages}" "${ctest_command[@]}" \
  > "${evidence_root}/ctest-portable.log" 2>&1
grep -F '100% tests passed' "${evidence_root}/ctest-portable.log" >/dev/null

# Preserve the known compiler-bound literal fingerprint result separately.  This
# test is not weakened or treated as portable relative-selector evidence.
set +e
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --exact \
  --kill-on-bad-exit=1 bash -c \
  'export OMP_NUM_THREADS=8 OMP_PROC_BIND=false KOKKOS_NUM_THREADS=8 PYTHONPATH="$1"; ctest --test-dir "$2" --output-on-failure -R "^athena[.]z4c_rhs_policy_production_host_exact$"' \
  bash "${python_packages}" "${build_root}" \
  > "${evidence_root}/compiler-bound-fingerprint.log" 2>&1
fingerprint_status=$?
set -e
printf '%s\n' "${fingerprint_status}" > "${evidence_root}/compiler-bound-fingerprint.status"
test "${fingerprint_status}" -ne 0
grep -F 'exact-base final state payload changed' \
  "${evidence_root}/compiler-bound-fingerprint.log" >/dev/null

"${python_bin}" "${campaign_root}/phase0_compare_cc_selector.py" \
  --source "${source_root}" \
  --work-dir "${build_root}/z4c_cc_selector_equivalence_test" \
  --output "${evidence_root}/cc-selector-relative-exact.json" \
  > "${evidence_root}/cc-selector-relative-exact.log" 2>&1
grep -F 'PASS: implicit/default and explicit cell-centered Z4c are exact' \
  "${evidence_root}/cc-selector-relative-exact.log" >/dev/null

convergence=${source_root}/tst/unit/z4c/z4c_vertex_dynamic_linear_wave_test.py
input2=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_2d_cartesian.athinput
input3=${source_root}/tst/inputs/z4c_vc_dynamic_linear_wave_3d_cartesian.athinput
athena=${build_root}/src/athena
for order in 2 4 6; do
  if test "${order}" -eq 6; then
    resolutions=(24 36 48)
  else
    resolutions=(16 32 64)
  fi
  for dimensions in 2 3; do
    input=${input2}
    test "${dimensions}" -eq 3 && input=${input3}
    label=o${order}-${dimensions}d
    command=(
      "${python_bin}" "${convergence}"
      --athena "${athena}"
      --input "${input}"
      --work-dir "${campaign_root}/runs/phase0-${label}"
      --dimensions "${dimensions}"
      --order "${order}"
      --integrator rk4
      --resolutions "${resolutions[@]}"
    )
    printf '%q ' "${command[@]}" > "${evidence_root}/${label}-command.txt"
    printf '\n' >> "${evidence_root}/${label}-command.txt"
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --exact \
      --kill-on-bad-exit=1 "${command[@]}" \
      > "${evidence_root}/${label}.log" 2>&1
    grep -F 'PASS: native-VC' "${evidence_root}/${label}.log" >/dev/null
  done
done

ctest --test-dir "${build_root}" -N > "${evidence_root}/ctest-inventory.txt"
grep -F 'athena.z4c_cc_selector_equivalence' "${evidence_root}/ctest-inventory.txt" >/dev/null
printf 'PHASE0_BASE_REPRODUCED_WITH_COMPILER_BOUND_LITERAL_FINGERPRINT_LIMITATION\n' \
  > "${evidence_root}/verdict.txt"
