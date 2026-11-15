#!/usr/bin/env python3
import os
import socket
keys = ['ZE_AFFINITY_MASK', 'OMP_NUM_THREADS', 'MPIR_CVAR_ENABLE_GPU',
        'MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE', 'MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE',
        'MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES']
print(socket.gethostname(), os.environ.get('MPI_LOCALRANKID',os.environ.get('PALS_LOCAL_RANKID','unknown')),
      *[os.environ.get(key,'unset') for key in keys],
      ','.join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),flush=True)
