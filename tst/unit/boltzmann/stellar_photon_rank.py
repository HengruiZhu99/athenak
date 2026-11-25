"""Select the primary visible CUDA GPU on launchers that expose peer devices."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket


def local_rank():
    for key in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        if key in os.environ:
            return int(os.environ[key])
    raise RuntimeError("The CUDA rank wrapper needs a node-local MPI rank")


def bind_numa(device):
    # PCI locality is optional. Honor the scheduler cpuset on hosts without sysfs.
    address = ctypes.create_string_buffer(64)
    cuda = ctypes.CDLL("libcudart.so")
    status = cuda.cudaDeviceGetPCIBusId(address, len(address), device)
    if status:
        raise RuntimeError(f"cudaDeviceGetPCIBusId({device}) failed with status {status}")
    pci = address.value.decode().lower()
    numa = Path("/sys/bus/pci/devices", pci, "numa_node")
    if numa.is_file() and hasattr(os, "sched_getaffinity"):
        node = int(numa.read_text())
        near = {cpu for cpu in os.sched_getaffinity(0)
                if Path(f"/sys/devices/system/node/node{node}/cpu{cpu}").exists()}
        if near:
            os.sched_setaffinity(0, near)
    return pci


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank-bound", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.command:
        p.error("supply an AthenaK executable and arguments")
    rank = local_rank()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    device = 0 if args.rank_bound else rank
    if (args.rank_bound and len(devices) != 1) or not 0 <= device < len(devices):
        raise RuntimeError(f"Rank {rank} has incompatible CUDA visibility {devices}")
    pci = bind_numa(device)
    peers = ",".join(devices[device:] + devices[:device])
    os.environ["CUDA_VISIBLE_DEVICES"] = peers
    print(json.dumps(dict(host=socket.gethostname(), rank=rank, pci=pci, peers=peers)), flush=True)
    command = args.command
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
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
