#!/usr/bin/env python3
"""Run actual periodic mesh transfer fixtures without evolving a physical state."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time
from run_legacy_equivalence import input_text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--smr', action='store_true')
parser.add_argument('--ranks', type=int, default=1)
parser.add_argument('--mpiexec', default='mpiexec')
parser.add_argument('--block-n', type=int, default=8)
parser.add_argument('--orders', type=int, nargs='+', default=[2, 4, 6])
parser.add_argument('--profile', choices=['constant', 'smooth','boundary_linear'], default='constant')
parser.add_argument('--boundary', choices=['periodic','outflow','reflect','mixed'], default='periodic')
parser.add_argument('--extrap-order',type=int,default=2)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
results = []
for order in args.orders:
    for dim in [2, 3]:
        folder = args.output/f'fd{order}-{dim}d'
        folder.mkdir()
        content = input_text(order, dim, 'collision_factorized', 0)
        mesh, tail = content.split('<meshblock>')
        mesh = mesh.replace('nx1 = 8', f'nx1 = {2*args.block_n}').replace('nx2 = 8', f'nx2 = {2*args.block_n}')
        tail = tail.replace('nx1 = 8', f'nx1 = {args.block_n}').replace('nx2 = 8', f'nx2 = {args.block_n}')
        if dim == 3:
            mesh = mesh.replace('nx3 = 8', f'nx3 = {2*args.block_n}')
            tail = tail.replace('nx3 = 8', f'nx3 = {args.block_n}')
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
        if args.boundary != 'periodic':
            flags = [args.boundary]*6 if args.boundary != 'mixed' else ['reflect','outflow','outflow','reflect','reflect','outflow']
            for face,flag in zip(['ix1','ox1','ix2','ox2','ix3','ox3'],flags):
                content = content.replace(f'{face}_bc = periodic',f'{face}_bc = {flag}')
        content = content.replace('<pc_gh>',f'<pc_gh>\nextrap_order = {args.extrap_order}')
        content = content.replace('<problem>', f'<problem>\nresidual_profile = {args.profile}')
        used = folder/'used_input.athinput'
        used.write_text(content)
        command = [str(args.binary.resolve()), '-i', str(used.resolve())]
        if args.ranks > 1:
            command = [args.mpiexec, '-n', str(args.ranks)]+command
        start = time.time()
        with (folder/'run.log').open('w') as log:
            run = subprocess.run(command, cwd=folder, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=120)
        row = dict(order=order, dimension=dim, smr=args.smr, ranks=args.ranks, profile=args.profile, boundary=args.boundary, extrap_order=args.extrap_order, block_n=args.block_n, command=command,
                   returncode=run.returncode, wall_seconds=time.time()-start,
                   binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256(used.read_bytes()).hexdigest())
        reported_ranks = [int(n) for n in re.findall(
            r'Number of parallel ranks = (\d+)', (folder/'run.log').read_text())]
        rank_files = sorted(folder.glob('transfer-mesh-rank*.jsonl'))
        row['reported_ranks'] = reported_ranks
        row['rank_files'] = [f.name for f in rank_files]
        runtime_valid = (reported_ranks == [args.ranks]
                         and {f.name for f in rank_files} == {
                             f'transfer-mesh-rank{r}.jsonl' for r in range(args.ranks)})
        row['runtime_status'] = 'PASS' if runtime_valid else 'FAIL'
        if run.returncode == 0 and runtime_valid:
            records = [json.loads(line) for file in folder.glob('transfer-mesh-rank*.jsonl')
                       for line in file.read_text().splitlines()]
            assert records
            row['max_fixed_change'] = max(r['fixed_change'] for r in records)
            row['max_residual_error'] = max(r.get('after_reference_residual_error', r.get('after_constant_residual_error')) for r in records)
            row['blocks'] = len(records)//3
            row['reflection_error'] = max(r.get('reflection_error_after',0) for r in records)
            row['invariance_status'] = 'PASS' if row['max_fixed_change'] == 0 else 'FAIL'
            row['status'] = ('PASS' if row['max_fixed_change'] == 0 and row['max_residual_error'] <= 2e-12 and row['reflection_error'] <= 2e-12 else 'FAIL') if args.profile != 'smooth' or not args.smr else 'NOT_RUN'
            if args.profile == 'smooth' and args.smr:
                row['status_note'] = 'convergence gate evaluated by resolution ladder, not this single mesh'
        else:
            row['status'] = 'FAIL'
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
