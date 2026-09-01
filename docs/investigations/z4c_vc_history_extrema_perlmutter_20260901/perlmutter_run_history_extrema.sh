#!/usr/bin/env bash
set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901
source_root=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/supercritical-horizon-20260830/source
build_root=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/supercritical-horizon-20260830/build-cuda-mpi
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py
python=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
athena=${HISTORY_EXTREMA_ATHENA:-${build_root}/src/athena}
authority=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/source/docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/authority/n256_reference_shock_authority.jsonl

module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
mkdir -p "${campaign}/build" "${campaign}/runs/smoke" \
  "${campaign}/runs/subcritical_Aminus0p047" \
  "${campaign}/runs/subcritical_Aminus0p047_dynamic" \
  "${campaign}/runs/subcritical_Aminus0p047_dynamic_retry1" \
  "${campaign}/runs/subcritical_Aminus0p047_dynamic_retry2" \
  "${campaign}/runs/subcritical_Aminus0p047_dynamic_retry3" \
  "${campaign}/runs/supercritical_Aminus0p050"

exec > >(tee -a "${campaign}/allocation.log") 2>&1
date -Is
hostname
scontrol show job "${SLURM_JOB_ID}"

if [[ ${HISTORY_EXTREMA_SKIP_BUILD:-0} != 1 ]]; then
  # Compile only the changed history translation unit, bypassing the unusually
  # slow whole-tree CMake dependency scan.  The old object remains authoritative
  # until the temporary object has compiled successfully.
  flags_file=${build_root}/src/CMakeFiles/athena.dir/flags.make
  defines=$(sed -n 's/^CXX_DEFINES = //p' "${flags_file}")
  includes=$(sed -n 's/^CXX_INCLUDES = //p' "${flags_file}")
  cxx_flags=$(sed -n 's/^CXX_FLAGS = //p' "${flags_file}")
  cd "${build_root}/src"
  compile_command="${source_root}/kokkos/bin/kokkos_launch_compiler ${source_root}/kokkos/bin/nvcc_wrapper /opt/cray/pe/craype/2.7.36/bin/CC /opt/cray/pe/craype/2.7.36/bin/CC ${defines} ${includes} ${cxx_flags} -MD -MT src/CMakeFiles/athena.dir/outputs/history.cpp.o -MF CMakeFiles/athena.dir/outputs/history.cpp.o.new.d -o CMakeFiles/athena.dir/outputs/history.cpp.o.new -c ${source_root}/src/outputs/history.cpp"
  echo HISTORY_EXTREMA_COMPILE_BEGIN
  eval "${compile_command}"
  echo HISTORY_EXTREMA_COMPILE_OK
  cp -p CMakeFiles/athena.dir/outputs/history.cpp.o \
    CMakeFiles/athena.dir/outputs/history.cpp.o.pre_min_lapse
  cp -p CMakeFiles/athena.dir/outputs/history.cpp.o.d \
    CMakeFiles/athena.dir/outputs/history.cpp.o.pre_min_lapse.d
  mv CMakeFiles/athena.dir/outputs/history.cpp.o.new \
    CMakeFiles/athena.dir/outputs/history.cpp.o
  mv CMakeFiles/athena.dir/outputs/history.cpp.o.new.d \
    CMakeFiles/athena.dir/outputs/history.cpp.o.d

  cp -p "${athena}" "${campaign}/build/athena.pre_min_lapse"
  bash CMakeFiles/athena.dir/link.txt
  cp -p "${athena}" "${campaign}/build/athena.history_extrema"
  sha256sum "${source_root}/src/outputs/history.cpp" "${athena}" \
    "${campaign}/build/athena.history_extrema" | tee "${campaign}/build/SHA256SUMS"
else
  test -x "${athena}"
  test -f "${campaign}/build/athena.history_extrema"
  cmp -s "${athena}" "${campaign}/build/athena.history_extrema"
  test "$(sha256sum "${source_root}/src/outputs/history.cpp" | awk '{print $1}')" = \
    "383f041cda8a16eea02d421feccc21a019c6960ced5a439a67841f80cf609e96"
  test "$(sha256sum "${athena}" | awk '{print $1}')" = \
    "c3ef1c8b371eb3a447108d3b0acc115b34cfeaeb71337fda277e18d409a5b8c0"
  echo HISTORY_EXTREMA_PREBUILT_BINARY_OK
fi

run_athena() {
  local run_dir=$1
  shift
  mkdir -p "${run_dir}/rank-bindings"
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 \
    --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
    --kill-on-bad-exit=1 \
    "${python}" "${wrapper}" --evidence-dir "${run_dir}/rank-bindings" \
    --require-cuda -- "${athena}" "$@"
}

if [[ ${HISTORY_EXTREMA_SUBCRITICAL_ONLY:-0} != 1 ]]; then
  # One-cycle smoke gate: verify that both extrema labels are emitted by the
  # rebuilt CUDA executable before starting either requested rerun.
  smoke=${campaign}/runs/smoke
  cp "${campaign}/inputs/brill_vc_reference_shock_gauge.athinput" "${smoke}/"
  cp "${campaign}/inputs/brill_global_128x32.coefficients" "${smoke}/"
  cd "${smoke}"
  run_athena "${smoke}" -i brill_vc_reference_shock_gauge.athinput \
    mesh/nx1=256 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64 \
    mesh_refinement/max_nmb_per_rank=256 \
    mesh_refinement/amr_history_mode=off time/nlim=1 time/tlim=1.0 \
    output1/dcycle=1 output2/dt=1000 output3/dt=1000 output4/dt=1000 output5/dt=1000 \
    job/basename=history_extrema_smoke \
    problem/brill_global_coefficients_file=brill_global_128x32.coefficients \
    problem/constraint_summary_file=history_extrema_smoke-constraints.dat
  grep -q 'maxAbsKret' history_extrema_smoke.z4c.user.hst
  grep -q 'minLapse' history_extrema_smoke.z4c.user.hst
  echo HISTORY_EXTREMA_SMOKE_OK
fi

# Figure-3-amplitude telegrapher case (A=-0.047), using the same native
# dchi-controlled dynamic-AMR policy as the A=-0.050 comparison run.
sub=${campaign}/runs/subcritical_Aminus0p047_dynamic_retry3
cp "${campaign}/inputs/brill_vc_n512_native_amr.athinput" "${sub}/"
cp "${campaign}/inputs/brill_global_128x32.coefficients" "${sub}/"
cd "${sub}"
set +e
run_athena "${sub}" -i brill_vc_n512_native_amr.athinput -t 01:50:00 \
  mesh/nx1=256 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64 \
  mesh_refinement/amr_history_mode=record \
  mesh_refinement/amr_history_file="${sub}/n512_Aminus0p047_history_extrema.amr_history.jsonl" \
  time/tlim=38.652331986867424 time/nlim=-1 time/cfl_number=0.15 \
  z4c/history_constraint_radius=64.0 z4c/boundary_rhs=sommerfeld \
  z4c/extrap_order=3 z4c/lapse_shock_avoiding=false z4c/lapse_oplog=2.0 \
  z4c/lapse_harmonic=0.0 z4c/telegraph_lapse=true z4c/telegraph_max_K=true \
  z4c/telegraph_damping_prescription=max_domain_abs_K \
  z4c/telegraph_tau=0.01 z4c/telegraph_kappa=0.01 \
  z4c/shift_mode=prescribed_zero z4c/shift_Gamma=0.0 \
  z4c/shift_alpha2Gamma=0.0 z4c/shift_H=0.0 z4c/shift_eta=0.0 \
  z4c/shift_eta_max_K=false z4c/shift_advect=0.0 \
  z4c/shift_invariant_diagnostic=false z4c/diss=0.50 \
  z4c/damp_kappa1=0 z4c/damp_kappa2=0 \
  output2/dt=1000 output3/dt=1000 output4/dt=1000 output5/dt=1000 \
  job/basename=n512_Aminus0p047_history_extrema \
  fastflow/num_horizons=1 fastflow/cartoon_adaptive_initial_radius_0=true \
  fastflow/cartoon_origin_lapse_radius_factor_0=3.0 \
  fastflow/cartoon_pair_disjoint_fraction_0=0.8 \
  problem/stop_on_horizon=true \
  problem/brill_direct_initial_lapse=unit \
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients \
  problem/constraint_summary_file=n512_Aminus0p047_history_extrema-constraints.dat \
  > stdout.log 2> stderr.log
sub_status=$?
set -e
printf '%s\n' "${sub_status}" > run-status
echo SUBCRITICAL_STATUS=${sub_status}

if [[ ${HISTORY_EXTREMA_SUBCRITICAL_ONLY:-0} == 1 ]]; then
  cd "${campaign}"
  find runs -maxdepth 2 -type f -print0 | sort -z | xargs -0 sha256sum \
    > SHA256SUMS
  date -Is
  exit 0
fi

# Faithful rerun of the latest supercritical telegrapher case (A=-0.050).
sup=${campaign}/runs/supercritical_Aminus0p050
cp "${campaign}/inputs/brill_vc_n512_native_amr.athinput" "${sup}/"
cp "${campaign}/inputs/brill_Aminus0p050_global_128x32.coefficients" "${sup}/"
cd "${sup}"
set +e
run_athena "${sup}" -i brill_vc_n512_native_amr.athinput -t 00:55:00 \
  mesh_refinement/amr_history_mode=record \
  mesh_refinement/amr_history_file="${sup}/n512_Aminus0p050_history_extrema.amr_history.jsonl" \
  fastflow/num_horizons=1 fastflow/cartoon_adaptive_initial_radius_0=true \
  fastflow/cartoon_origin_lapse_radius_factor_0=3.0 \
  fastflow/cartoon_pair_disjoint_fraction_0=0.8 \
  output2/dt=1000 output3/dt=1000 output4/dt=1000 output5/dt=1000 \
  job/basename=n512_Aminus0p050_history_extrema \
  problem/brill_global_coefficients_file=brill_Aminus0p050_global_128x32.coefficients \
  problem/constraint_summary_file=n512_Aminus0p050_history_extrema-constraints.dat \
  problem/stop_on_horizon=true z4c/lapse_shock_avoiding=false \
  z4c/lapse_oplog=2.0 z4c/lapse_harmonic=0.0 z4c/telegraph_lapse=true \
  z4c/telegraph_max_K=true \
  z4c/telegraph_damping_prescription=max_domain_abs_K \
  z4c/telegraph_tau=0.01 z4c/telegraph_kappa=0.01 \
  > stdout.log 2> stderr.log
sup_status=$?
set -e
printf '%s\n' "${sup_status}" > run-status
echo SUPERCRITICAL_STATUS=${sup_status}

cd "${campaign}"
find runs -maxdepth 2 -type f -print0 | sort -z | xargs -0 sha256sum \
  > SHA256SUMS
date -Is
