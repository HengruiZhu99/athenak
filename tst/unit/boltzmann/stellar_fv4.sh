#!/bin/bash
# Submit from a staged source checkout with flat.adm and strong.adm.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy,cuda_ipc
root=$PWD
mode=$1
ranks=$2
order=$3
nx=$4
bands=$5
nz=$((nx/2))
sectors=$((bands*4/3))
block=$6
work=$root/qualification/$SLURM_JOB_ID/$mode-$ranks-$order-$nx-$bands
mkdir -p "$work"
cd "$work"
cp "$root/inputs/boltzmann/photon_torus.athinput" input
case $order in
  2) rk=rk2;;
  4) rk=ssprk104;;
esac
args=(-i input "time/integrator=$rk" "boltzmann/transport_order=$order"
  "boltzmann/nmu_half=$bands" "boltzmann/npsi=$sectors" "mesh/nx1=$nx"
  "meshblock/nx1=$block" "meshblock/nx2=$block")
case $mode in
  coupled)
    args+=(problem/torus_file="$root/strong.adm" "mesh/nx2=$nx")
    cat >> input <<'DECK'
<output2>
file_type = rst
dcycle = 1
DECK
    ;;
  turn|bench)
    args+=(problem/torus_file="$root/flat.adm" "mesh/nx2=$nz" mesh/x1max=12 mesh/x2min=-3
      mesh/x2max=3 boltzmann/fixed_metric=true boltzmann/backreaction=false time/tlim=6.5)
    if [[ $mode == turn ]]; then
      cat >> input <<'DECK'
<output2>
file_type = bin
variable = tmunu
dt = 0.5
DECK
      args+=(output1/dcycle=100 time/ndiag=1000)
    else
      args+=(time/nlim=40 output1/dcycle=20 time/ndiag=10)
    fi;;
esac
nvidia-smi --query-gpu=index,uuid,name --format=csv
sha256sum "$root/build/src/athena" input
visible=${FV4_ALLOCATED_GPUS:-$ranks}
mask=$(printf '0x%x' "$(( (1 << visible) - 1 ))")
set -x
srun --exact -N1 -n "$ranks" -c 4 --gres=gpu:"$visible" --gpu-bind=mask_gpu:"$mask" --cpu-bind=none \
  --mpi=pmix --kill-on-bad-exit=1 python3 "$root/tst/unit/boltzmann/stellar_photon_rank.py" \
  "$root/build/src/athena" "${args[@]}" > run.log 2>&1
tail -12 run.log
