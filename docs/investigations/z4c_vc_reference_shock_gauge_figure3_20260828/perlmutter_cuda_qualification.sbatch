#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=z4c-shock-qual
#SBATCH --output=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/cuda-qual-%j.out
#SBATCH --error=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/cuda-qual-%j.err

set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828
source_root=${campaign}/source
build_root=${campaign}/build-cuda-mpi
athena=${build_root}/src/athena
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
expected_source=00e66dfa5e0e7f7f2c711166998806a05decbd55
expected_tree=578bfd4ede74ddc7464f82e4c7b76f4111a4ad76
expected_exe=c5153a5c25c4b2aba22737061baa00628badcd6eade2c66461ac182708677e55
expected_cache=273fe5f79f18d39f19748aff573c1ac204ebe9aae14f61533ca99c42a3c13f8a
evidence=${campaign}/evidence/cuda-qualification-${SLURM_JOB_ID}

test "${SLURM_NNODES}" = 1
test "${SLURM_NTASKS}" = 1
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"

mkdir -p "${evidence}" "${evidence}/shift-smoke"
export COLLAPSE_ROOT=${campaign}
# shellcheck source=/dev/null
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta

scontrol show job "${SLURM_JOB_ID}" > "${evidence}/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${evidence}/node.txt"
env | LC_ALL=C sort > "${evidence}/environment.txt"
{
  date --iso-8601=seconds
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt"
  module list
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${evidence}/provenance.txt" 2>&1

test_regex='^athena[.](z4c_state_admissibility|z4c_state_admissibility_static|z4c_timestep_contract|z4c_timestep_contract_static|z4c_shift_control_static|z4c_shock_avoiding_gauge_static|z4c_vc_dynamic_amr_2d_cartoon_o4|z4c_vc_static_multilevel_2d_cartoon_axis_o4|z4c_vc_amr_history|z4c_cartoon_amr_transfer_qualification|z4c_cartoon_production_kernels)$'
ctest --test-dir "${build_root}" --output-on-failure -V -R "${test_regex}" \
  > "${evidence}/ctest.log" 2>&1

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
  "${athena}" --z4c-cartoon-kernel-test --require-cuda \
  > "${evidence}/cuda-kernel.log" 2>&1

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gpus-per-task=1 --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
  "${athena}" -i "${source_root}/tst/inputs/z4c_shock_avoiding_gauge.athinput" \
  -d "${evidence}/shift-smoke" job/basename=shock_zero_shift \
  time/nlim=2 z4c/telegraph_lapse=false z4c/lapse_oplog=0 \
  z4c/lapse_harmonic=0 z4c/lapse_shock_avoiding=true \
  z4c/lapse_shock_avoiding_kappa=1 z4c/shift_mode=prescribed_zero \
  z4c/shift_Gamma=0 z4c/shift_alpha2Gamma=0 z4c/shift_H=0 \
  z4c/shift_advect=0 z4c/shift_eta=0 z4c/shift_invariant_diagnostic=true \
  > "${evidence}/shift-smoke.log" 2>&1

grep -Fq 'for collapsed NGHOST 2/3/4 on Cuda' "${evidence}/cuda-kernel.log"
grep -Fq 'cycle=2' "${evidence}/shift-smoke.log"
grep -Fq '100% tests passed' "${evidence}/ctest.log"

find "${evidence}" -type f ! -name SHA256SUMS -print0 | \
  LC_ALL=C sort -z | xargs -0r sha256sum > "${evidence}/SHA256SUMS"
(cd "${evidence}" && sha256sum -c SHA256SUMS >/dev/null)
printf 'PERLMUTTER_CUDA_QUALIFICATION_PASS\n' > "${evidence}/disposition"
