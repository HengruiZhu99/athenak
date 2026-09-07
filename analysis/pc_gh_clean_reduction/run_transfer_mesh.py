#!/usr/bin/env python3
"""Run actual periodic mesh transfer fixtures without evolving a physical state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from run_legacy_equivalence import input_text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--smr', action='store_true')
parser.add_argument('--ranks', type=int, default=1)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
results = []
for order in [2, 4, 6]:
    for dim in [2, 3]:
        folder = args.output/f'fd{order}-{dim}d'
        folder.mkdir()
        content = input_text(order, dim, 'collision_factorized', 0)
        mesh, tail = content.split('<meshblock>')
        mesh = mesh.replace('nx1 = 8', 'nx1 = 16').replace('nx2 = 8', 'nx2 = 16')
        if dim == 3:
            mesh = mesh.replace('nx3 = 8', 'nx3 = 16')
        content = (mesh+'<meshblock>'+tail).replace('<pc_gh>', '<pc_gh>\ncoherent_transfer = residual_shifted')
        if args.smr:
            content += '''\n<mesh_refinement>
refinement = static
<refined_region1>
x1min = 0.125
x1max = 0.375
x2min = 0.1625
x2max = 0.4875
x3min = 0.2125
x3max = 0.6375
level = 1
'''
        used = folder/'used_input.athinput'
        used.write_text(content)
        command = [str(args.binary.resolve()), '-i', str(used.resolve())]
        if args.ranks > 1:
            command = ['mpiexec', '-n', str(args.ranks)]+command
        start = time.time()
        with (folder/'run.log').open('w') as log:
            run = subprocess.run(command, cwd=folder, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=120)
        row = dict(order=order, dimension=dim, smr=args.smr, ranks=args.ranks, command=command,
                   returncode=run.returncode, wall_seconds=time.time()-start,
                   binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256(used.read_bytes()).hexdigest())
        if run.returncode == 0:
            records = [json.loads(line) for file in folder.glob('transfer-mesh-rank*.jsonl')
                       for line in file.read_text().splitlines()]
            assert records
            row['max_fixed_change'] = max(r['fixed_change'] for r in records)
            row['max_residual_error'] = max(r['after_constant_residual_error'] for r in records)
            row['blocks'] = len(records)//3
            row['status'] = 'PASS' if row['max_fixed_change'] == 0 and row['max_residual_error'] <= 2e-12 else 'FAIL'
        else:
            row['status'] = 'FAIL'
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
