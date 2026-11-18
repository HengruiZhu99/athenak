"""Measure engine activity across a complete process, including its initialization."""
import os
from pathlib import Path
import re
import subprocess


def snapshot(root, directory, label, nodes):
    probe=root/'engine-activity'
    if not probe.is_file():
        return {}
    env=dict(os.environ)
    env.pop('ZE_AFFINITY_MASK',None)
    log=directory/f'engine-{label}.log'
    with log.open('w') as stream:
        result=subprocess.run(['mpiexec','-n',str(nodes),'-ppn','1',str(probe)],env=env,
                              stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        return {}
    records={}
    for line in log.read_text().splitlines():
        row=re.fullmatch(r'(\S+) ([0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]+) (-?\d+) (\d+) (\d+) (\d+)',line)
        if row:
            key=tuple(row.groups()[:4])
            if key in records:
                raise ValueError('duplicate hardware engine counter '+str(key))
            records[key]=tuple(map(int,row.groups()[4:]))
    return records


def utilization(before,after):
    measured=[]
    for key,(active,start) in sorted(before.items()):
        if key not in after:
            continue
        end_active,end=after[key]
        if end>start and 0<=end_active-active<=end-start:
            host,pci,tile,group=key
            measured.append(dict(host=host,pci=pci,tile=int(tile),engine_group=int(group),
                                 busy_fraction=(end_active-active)/(end-start)))
    return measured
