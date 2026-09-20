#!/usr/bin/env python3
"""Run read-only binary aggregation on Aurora and fetch only compact profiles.

Only the separate analysis output directory is written remotely. This never
queries, submits, cancels, or changes scheduler jobs or simulation settings.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--host', default='aurora')
p.add_argument('--socket')
p.add_argument('--run', required=True)
p.add_argument('--remote-analysis', required=True)
p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
p.add_argument('--case', action='append', metavar='NAME=PREFIX',
               help='Restrict/override case subdirectory and file prefix; repeatable')
a = p.parse_args()
here = Path(__file__).resolve().parent
a.output.mkdir(parents=True, exist_ok=True)
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
if a.socket:
    opts += ['-o', 'ControlPath='+a.socket]


def remote(argv):
    return subprocess.run(['ssh']+opts+[a.host, shlex.join(argv)], check=True, capture_output=True, text=True)


def fetch(relative, target):
    subprocess.run(['scp']+opts+[a.host+':'+a.remote_analysis+'/'+relative, str(target)], check=True)


identity = remote(['hostname', '-f']).stdout.strip()
if not identity.endswith('.alcf.anl.gov'):
    raise RuntimeError('Refusing remote writes: not a verified ALCF hostname: '+identity)
remote(['test', '-d', '/lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review'])
remote(['mkdir', '-p', a.remote_analysis])
subprocess.run(['scp']+opts+[str(here/'aggregate.py'), a.host+':'+a.remote_analysis+'/aggregate.py'], check=True)
cmd = ['env', 'OPENBLAS_NUM_THREADS=1', 'python3', a.remote_analysis+'/aggregate.py', a.run, a.remote_analysis]
for case in a.case or []:
    cmd.extend(['--case', case])
result = remote(cmd)
print(result.stdout.strip())
status = json.loads(result.stdout)
fetch('status.json', a.output/'status.json')
for item in status:
    case = item['case']
    folder = a.output/case
    folder.mkdir(parents=True, exist_ok=True)
    fetch(case+'/manifest.json', folder/'manifest.json')
    if item['snapshots']:
        fetch(case+'/profiles.npz', folder/'profiles.npz')
subprocess.run([sys.executable, str(here/'plot.py'), str(a.output)], check=True)
