case "${1##*/}" in
    worker.py|replay.py)
        if [[ $1 == */worker.py || ${3:-} == evolve ]]; then
            module purge
            module load anaconda3/2025.12 cudatoolkit/13.2 hpcx-latest/hpcx
            export CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0
            export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy,cuda_ipc
        else
            module load anaconda3/2025.12
        fi ;;
    *) module load anaconda3/2025.12 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
