"""Collect Level Zero kernel execution times for every launched tile rank."""
import os
from pathlib import Path
import re
import sys


def device_totals(directory, ranks):
    rows = []
    for rank in range(ranks):
        # unitrace can append the process ID before the filename extension.
        logs = list((directory/f'device-rank-{rank}').glob('*'))
        text = '\n'.join(path.read_text(errors='replace') for path in logs if path.is_file())
        totals = re.findall(r'Total [^\n]+ Time for L0 backend \(ns\):\s*(\d+)',text)
        if len(totals)!=1 or int(totals[0])<=0:
            raise RuntimeError(f'missing or ambiguous Level Zero execution time for rank {rank}: {logs}')
        rows.append(dict(rank=rank,logs=[str(p) for p in logs],
                         kernel_seconds=int(totals[0])*1e-9))
    return rows


if __name__ == '__main__':
    tool, directory, *command = sys.argv[1:]
    rank = next(os.environ[key] for key in ('PMI_RANK','PALS_RANKID','PMIX_RANK') if key in os.environ)
    path = Path(directory)/f'device-rank-{int(rank)}'
    path.mkdir()
    os.execv(tool,[tool,'--device-timing','--output',str(path/'kernels.log'),*command])
