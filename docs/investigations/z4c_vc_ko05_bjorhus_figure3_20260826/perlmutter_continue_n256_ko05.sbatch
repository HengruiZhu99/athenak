#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=shared_interactive
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=z4c-n256-ko05-cont
#SBATCH --output=/pscratch/sd/h/hzhu/z4c-vc-ko05-bjorhus-figure3-20260826/slurm/n256-ko05-cont-%j.out
#SBATCH --error=/pscratch/sd/h/hzhu/z4c-vc-ko05-bjorhus-figure3-20260826/slurm/n256-ko05-cont-%j.err

set -euo pipefail

campaign=/pscratch/sd/h/hzhu/z4c-vc-ko05-bjorhus-figure3-20260826
source_root=${campaign}/source
build_root=${campaign}/build-cuda-mpi
profile=/pscratch/sd/h/hzhu/collapse-critical-perlmutter/workflow/profiles/perlmutter-a100.sh

export COLLAPSE_ROOT=/pscratch/sd/h/hzhu/collapse-critical-perlmutter
# shellcheck source=/dev/null
source "${profile}"

# Qualify the endpoint-aware native-vertex diagnostic on the allocated A100
# before consuming the authenticated restart.
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  --gpu-bind=map_gpu:0 --cpu-bind=cores --exact \
  "${build_root}/athena_z4c_shared_geometry_policy_unit_test"

export CASE_LABEL=n256
export HISTORY_MODE=record
export HISTORY_FILE=${campaign}/authority/n256_ko05_authority_active.jsonl
export RUN_ROOT=${campaign}/runs/n256_record_seg002
export RUN_TLIM=30.0
export CAMPAIGN_BUNDLE=${campaign}/bundle
export SOURCE_ROOT=${source_root}
export BUILD_ROOT=${build_root}
export RESTART_FILE=${campaign}/runs/n256_record_seg000/rst/n256_ko05_record.00027.rst
export EXPECTED_SOURCE=892e1e59b2a29e8f4e2b628ed57cd52c88639032
export EXPECTED_TREE=6e5425f6b9507b14d5444a767eaa50a2dd6e66f5
export EXPECTED_EXE=274d7836b00d9a5fa22d59de2f174f5b75c543f71c536afa8c365ac6455f0f6a
export EXPECTED_CMAKE_CACHE=d076ab301909f14b0649957e03b59f32a96d9c707db9684b10ca8e1e5feb3e96
export IRISK_LIBRARY=/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823/source/irisk-authority/build/serial-gcc/src/libiris_athenak_interpolator.a
export EXPECTED_IRISK_LIBRARY_SHA256=d4afad6d3a20a8dd8197eb7d70d5a23903a7e2401a5d8b034d32005bf07f3f39

exec "${campaign}/bundle/perlmutter_run_ko05_case.sh"
