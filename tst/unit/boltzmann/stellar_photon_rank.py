"""Select the primary CUDA GPU from a Slurm per-task or shared-node binding."""

import ctypes
import json
import os
from pathlib import Path
import socket
import sys

rank = int(os.environ["SLURM_LOCALID"])
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
rank_bound = sys.argv[1] == "--rank-bound"
if rank_bound:
    if len(devices) != 1:
        raise RuntimeError(f"Expected one GPU assigned to rank {rank}, got {devices}")
    device = 0
else:
    if rank >= len(devices):
        raise RuntimeError(f"Local rank {rank} cannot select from {devices}")
    device = rank
cuda = ctypes.CDLL("libcudart.so")
address = ctypes.create_string_buffer(64)
assert cuda.cudaDeviceGetPCIBusId(address, len(address), device) == 0
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
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[device:] + devices[:device])
print(
    json.dumps(
        dict(
            host=socket.gethostname(), rank=rank, pci=pci, peers=os.environ["CUDA_VISIBLE_DEVICES"]
        )
    ),
    flush=True,
)
command = sys.argv[2:] if rank_bound else sys.argv[1:]
# Queued segments can already have a Slurm-spooled copy of the batch
# script. Read the campaign release's input overlay at restart launch.
if "-r" in command and "-i" not in command:
    overlay = os.environ.get("EQUATOR_RUNTIME_INPUT")
    if overlay is None and os.environ.get("EQUATOR_RELEASE"):
        candidate = Path(os.environ["EQUATOR_RELEASE"]).with_name("runtime.in")
        if candidate.is_file():
            overlay = str(candidate)
    if overlay:
        if rank == 0:
            print(f"Restart runtime input: {overlay}", flush=True)
        command.extend(["-i", overlay])
os.execv(command[0], command)
