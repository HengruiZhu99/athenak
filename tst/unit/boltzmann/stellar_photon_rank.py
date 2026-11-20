"""Select one primary GPU per rank, retaining the allocated peer GPUs for CUDA IPC."""

import ctypes
import json
import os
from pathlib import Path
import socket
import sys

rank = int(os.environ["SLURM_LOCALID"])
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
assert len(devices) >= int(os.environ["SLURM_NTASKS"]), devices
cuda = ctypes.CDLL("libcudart.so")
address = ctypes.create_string_buffer(64)
assert cuda.cudaDeviceGetPCIBusId(address, len(address), rank) == 0
pci = address.value.decode().lower()
node = int(Path("/sys/bus/pci/devices", pci, "numa_node").read_text())
available = os.sched_getaffinity(0)
near = [
    cpu
    for cpu in sorted(available)
    if Path(f"/sys/devices/system/node/node{node}/cpu{cpu}").exists()
]
if near:
    os.sched_setaffinity(0, near)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[rank:] + devices[:rank])
print(
    json.dumps(
        dict(
            host=socket.gethostname(), rank=rank, pci=pci, peers=os.environ["CUDA_VISIBLE_DEVICES"]
        )
    ),
    flush=True,
)
os.execv(sys.argv[1], sys.argv[1:])
