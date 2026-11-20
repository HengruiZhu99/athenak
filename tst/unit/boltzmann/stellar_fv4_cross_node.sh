#!/bin/bash
# Request two GPU nodes with one rank and one GPU per node.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll
export UCX_TLS=${FV4_UCX_TLS:-self,sm,tcp,cuda_copy,cuda_ipc}
root=$PWD
dir=$root/qualification/$SLURM_JOB_ID/coupled-2-4-48-12
mkdir -p "$dir"
cd "$dir"
cp "$root/inputs/boltzmann/photon_torus.athinput" input
cat >> input <<'DECK'
<output2>
file_type = rst
dcycle = 1
DECK
sha256sum "$root/build/src/athena" input
set -x
srun --exact -N2 -n2 --ntasks-per-node=1 -c4 --gpus-per-task=1 --gpu-bind=single:1 \
  --mpi=pmix --kill-on-bad-exit=1 bash -c 'echo "rank=$SLURM_PROCID host=$(hostname) node=$SLURM_NODEID gpu=$(nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader)" >&2; exec "$@"' \
  launcher "$root/build/src/athena" -i input time/integrator=ssprk104 boltzmann/transport_order=4 \
  boltzmann/nmu_half=12 boltzmann/npsi=16 problem/torus_file="$root/strong.adm" >run.log 2>&1
tail -12 run.log
