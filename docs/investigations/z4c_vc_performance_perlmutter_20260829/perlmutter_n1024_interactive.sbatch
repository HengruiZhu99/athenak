#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=z4c-n1024
#SBATCH --output=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/slurm/n1024-%j.out
#SBATCH --error=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/slurm/n1024-%j.err

set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829
source_root=${campaign}/source
build_root=${campaign}/build-cuda-mpi
athena=${build_root}/src/athena
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh
python_bin=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3
input=${source_root}/docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/brill_vc_reference_shock_gauge.athinput
history=${source_root}/docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/authority/n256_reference_shock_authority.jsonl
coeff=${source_root}/docs/investigations/z4c_vc_performance_aurora_20260829/evidence/optimized_one_tile_v7/brill_global_128x32.coefficients
wrapper=${source_root}/tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py

expected_source=02704c79ffe95312cfeba9acde3d38f8b9677dec
expected_tree=9605962eed2b74fba6f1d70e6fc526a8f5f56bba
expected_kokkos=6739bc623081648af9e752b616d9671527922cbf
expected_exe=a13a58be5e45b37c4c29f65eab2f6fa0d43f8d235d0c51e7ed3459b18e9e247e
expected_cache=079d2d8b3bb18ac716293b8c7498f6564ac257b8f8099b6778a74adf01076721
expected_input=d86595adeaf7d4c72e1f150d044cd5277f7b4e3507eea4e6d03f427c8dc38838
expected_history=7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde
expected_coeff=1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10

variant=${N1024_VARIANT:-baseline}
case "${variant}" in
  baseline)
    run_root=${campaign}/runs/n1024_reference_shock_seg000_retry2
    basename=n1024_reference_shock_replay
    run_cfl=0.15
    run_diss=0.50
    athena_wall_limit=${N1024_ATHENA_WALL_LIMIT:-03:35:00}
    ;;
  cfl040_ko010)
    run_root=${campaign}/runs/n1024_reference_shock_cfl040_ko010_retry1
    basename=n1024_reference_shock_cfl040_ko010_replay
    run_cfl=0.40
    run_diss=0.10
    athena_wall_limit=${N1024_ATHENA_WALL_LIMIT:-01:45:00}
    ;;
  *)
    printf 'unsupported N1024_VARIANT=%s\n' "${variant}" >&2
    exit 2
    ;;
esac

test "${SLURM_NNODES}" = 1
test "${SLURM_NTASKS}" = 4
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_source}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test "$(git -C "${source_root}/kokkos" rev-parse HEAD)" = "${expected_kokkos}"
test -z "$(git -C "${source_root}" status --porcelain=v1)"
test "$(sha256sum "${athena}" | awk '{print $1}')" = "${expected_exe}"
test "$(sha256sum "${build_root}/CMakeCache.txt" | awk '{print $1}')" = "${expected_cache}"
test "$(sha256sum "${input}" | awk '{print $1}')" = "${expected_input}"
test "$(sha256sum "${history}" | awk '{print $1}')" = "${expected_history}"
test "$(sha256sum "${coeff}" | awk '{print $1}')" = "${expected_coeff}"
test ! -e "${run_root}"

mkdir -p "${run_root}/evidence" "${run_root}/rank-bindings"
cp "${input}" "${run_root}/"
cp "${coeff}" "${run_root}/brill_global_128x32.coefficients"
printf 'variant=%s\ncfl=%s\ndiss=%s\nathena_wall_limit=%s\n' "${variant}" \
  "${run_cfl}" "${run_diss}" "${athena_wall_limit}" \
  > "${run_root}/evidence/effective-variant.txt"

export COLLAPSE_ROOT=${campaign}
# shellcheck source=/dev/null
source "${profile}"
export OMP_NUM_THREADS=8 KOKKOS_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0
export MPICH_OFI_NIC_POLICY=GPU SLURM_MPI_TYPE=cray_shasta
export ATHENA_AMR_HISTORY_COMPATIBLE_SOURCE_ID=athena-0.1-git-f8303c6be7eb214fa1e91b646123ee0d434b3698

scontrol show job "${SLURM_JOB_ID}" > "${run_root}/evidence/slurm-job.txt"
scontrol show node "${SLURM_NODELIST}" -o > "${run_root}/evidence/node.txt"
env | LC_ALL=C sort > "${run_root}/evidence/environment.txt"
{
  date --iso-8601=seconds
  hostname
  git -C "${source_root}" rev-parse HEAD 'HEAD^{tree}'
  git -C "${source_root}/kokkos" rev-parse HEAD
  sha256sum "${athena}" "${build_root}/CMakeCache.txt" "${input}" \
    "${history}" "${coeff}"
  module list
  CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi -L
  CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader
} > "${run_root}/evidence/provenance.txt" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits --loop=10 \
  > "${run_root}/evidence/gpu-samples.csv" 2>&1 &
monitor_pid=$!
cleanup() {
  set +e
  kill "${monitor_pid}" 2>/dev/null
  wait "${monitor_pid}" 2>/dev/null
  git -C "${source_root}" status --porcelain=v1 > "${run_root}/evidence/source-status.final"
  scontrol show job "${SLURM_JOB_ID}" > "${run_root}/evidence/slurm-job.final.txt" 2>&1
}
trap cleanup EXIT

command=(srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=32
  --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores --exact
  --kill-on-bad-exit=1 "${python_bin}" "${wrapper}"
  --evidence-dir "${run_root}/rank-bindings" --require-cuda -- "${athena}"
  -i brill_vc_reference_shock_gauge.athinput -t "${athena_wall_limit}"
  mesh/nx1=512 mesh/nx2=1024 meshblock/nx1=128 meshblock/nx2=128
  mesh_refinement/max_nmb_per_rank=128
  mesh_refinement/amr_history_mode=replay
  mesh_refinement/amr_history_file="${history}"
  time/tlim=38.652331986867424 time/nlim=-1 time/cfl_number="${run_cfl}"
  z4c/lapse_shock_avoiding=true z4c/lapse_shock_avoiding_kappa=1
  z4c/telegraph_lapse=false z4c/shift_mode=prescribed_zero
  z4c/shift_invariant_diagnostic=false z4c/diss="${run_diss}"
  z4c/damp_kappa1=0 z4c/damp_kappa2=0
  job/basename="${basename}"
  problem/brill_global_coefficients_file=brill_global_128x32.coefficients
  problem/constraint_summary_file="${basename}-constraints.dat")
printf '%q ' "${command[@]}" > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

cd "${run_root}"
set +e
"${command[@]}" > stdout.log 2> stderr.log &
launcher_pid=$!
terminal_seen=0
launcher_interrupted=0
while kill -0 "${launcher_pid}" 2>/dev/null; do
  sleep 30
  if grep -Fq 'Terminating on time limit' stdout.log; then
    terminal_seen=$((terminal_seen + 1))
    if ((terminal_seen >= 2)); then
      kill "${launcher_pid}" 2>/dev/null
      launcher_interrupted=1
      break
    fi
  fi
done
wait "${launcher_pid}"
launcher_status=$?
set -e

printf 'launcher_status=%s\nlauncher_interrupted_after_terminal=%s\n' \
  "${launcher_status}" "${launcher_interrupted}" > run-status

if grep -Fq 'Terminating on time limit' stdout.log; then
  disposition=REACHED_TLIM
elif grep -Fq 'Terminating on wall clock limit' stdout.log; then
  disposition=HEALTHY_WALLTIME_RESTART_REQUIRED
elif [[ ${launcher_status} -ne 0 ]]; then
  disposition=FAIL_CLOSED_NUMERICAL_OR_RUNTIME_FAILURE
else
  disposition=UNCLASSIFIED_CLEAN_EXIT
fi
printf '%s\n' "${disposition}" > disposition

if compgen -G '*.amr_history_replay.jsonl' >/dev/null; then
  if grep -Fq '"exact_match":false' ./*.amr_history_replay.jsonl; then
    printf 'AMR replay exact-match failure\n' >&2
    exit 3
  fi
fi

grep -E 'Terminating|time=|cycle=|MeshBlock-cycles|cpu time used|zone-cycles' \
  stdout.log | tail -40 > evidence/terminal-summary.txt
find rst -maxdepth 1 -type f -printf '%f %s\n' | LC_ALL=C sort \
  > evidence/restart-inventory.txt
find bin -type f -printf '%f %s\n' | LC_ALL=C sort \
  > evidence/binary-inventory.txt
sha256sum stdout.log stderr.log ./*.z4c.user.hst ./*.amr_history_replay.jsonl \
  > evidence/compact-results.sha256

if [[ "${disposition}" = REACHED_TLIM ]]; then
  exit 0
fi
if [[ "${disposition}" = HEALTHY_WALLTIME_RESTART_REQUIRED ]]; then
  exit 0
fi
exit 2
