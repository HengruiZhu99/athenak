"""Collect Level Zero kernel execution times for every launched tile rank."""
import csv
import os
from pathlib import Path
import re
import sys


def timing_table(text):
    """Read the execution table, excluding the separate compiler-properties table."""
    lines = iter(text.splitlines())
    for line in lines:
        columns = [s.strip() for s in line.split(',')]
        if columns[:3]==['Kernel','Calls','Time (ns)']:
            break
    operations = {}
    for line in lines:
        if not line.strip():
            break
        name, calls, duration, *_ = next(csv.reader([line.strip()],skipinitialspace=True))
        # Preserve MPI copy directions. For SYCL compute kernels, group template
        # specializations by the native function enclosing their device lambda.
        method = re.search(r'(?:z4c_hyperboloidal|bvals)::[\w:]+(?=\()',name)
        key = method[0] if method else name
        row = operations.setdefault(key,dict(calls=0,nanoseconds=0))
        row['calls'] += int(calls)
        row['nanoseconds'] += int(duration)
    return operations


def device_totals(directory, ranks):
    rows = []
    for rank in range(ranks):
        # unitrace can append the process ID before the filename extension.
        logs = list((directory/f'device-rank-{rank}').glob('*'))
        text = '\n'.join(path.read_text(errors='replace') for path in logs if path.is_file())
        totals = re.findall(r'Total [^\n]+ Time for L0 backend \(ns\):\s*(\d+)',text)
        if len(totals)!=1 or int(totals[0])<=0:
            raise RuntimeError(f'missing or ambiguous Level Zero execution time for rank {rank}: {logs}')
        total = int(totals[0])
        operations = timing_table(text)
        if sum(op['nanoseconds'] for op in operations.values())!=total:
            raise RuntimeError(f'incomplete Level Zero timing table for rank {rank}')
        driver = sum(op['nanoseconds'] for name,op in operations.items()
                     if name.startswith('zeCommandList'))
        rows.append(dict(rank=rank,logs=[str(p) for p in logs],device_seconds=total*1e-9,
                         kernel_seconds=(total-driver)*1e-9,driver_command_seconds=driver*1e-9,
                         operations=operations))
    return rows


if __name__ == '__main__':
    tool, directory, skip_steps, *command = sys.argv[1:]
    rank = next(os.environ[key] for key in ('PMI_RANK','PALS_RANKID','PMIX_RANK') if key in os.environ)
    path = Path(directory)/f'device-rank-{int(rank)}'
    path.mkdir()
    os.environ["MR_TRACE_SKIP_STEPS"]=skip_steps
    os.execv(tool,[tool,'--start-paused','--device-timing','--output',str(path/'kernels.log'),*command])
