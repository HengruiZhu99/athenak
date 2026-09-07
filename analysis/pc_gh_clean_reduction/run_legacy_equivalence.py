#!/usr/bin/env python3
"""Run independently built production kernels with the identical legacy adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_text(order, dim, target, steps):
    n3 = 8 if dim == 3 else 1
    return f'''<job>
basename = legacy_equivalence
<mesh>
nghost = 4
nx1 = 8
nx2 = 8
nx3 = {n3}
x1min = 0
x1max = 1
x2min = 0
x2max = 1.3
x3min = 0
x3max = 1.7
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 8
nx2 = 8
nx3 = {n3}
<time>
evolution = dynamic
integrator = rk3
cfl_number = 0.2
nlim = {steps}
tlim = {0.0001 if steps else 0}
ndiag = 1
<pc_gh>
spatial_order = {order}
gauge = z4c_mp_hyperbolic
reduction_system = legacy
reduction_rate = 0
lapse_projection_target = {target}
kappa = 0
shift_eta = 2
dissipation = {0.3 if steps else 0}
project_gauge_constraints = true
project_reduction_constraints = true
boundedness_output = false
<problem>
pgen_name = legacy_equivalence
'''


def run(binary, path, text):
    path.mkdir(parents=True, exist_ok=False)
    used = path/'used_input.athinput'
    used.write_text(text)
    start = time.time()
    command = [str(binary.resolve()), '-i', str(used.resolve())]
    with (path/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=path, stdout=log, stderr=subprocess.STDOUT)
    manifest = dict(command=command, returncode=result.returncode,
                    wall_seconds=time.time()-start, binary_sha256=digest(binary),
                    input_sha256=digest(used))
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'{path}: process failed {result.returncode}; see run.log')
    values = np.loadtxt(path/'legacy-oracle.csv', delimiter=',', skiprows=1)
    assert np.isfinite(values).all()
    manifest['output_sha256'] = digest(path/'legacy-oracle.csv')
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collision', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for order in [2, 4, 6]:
        for dim in [2, 3]:
            name = f'fd{order}-{dim}d'
            text = input_text(order, dim, 'collision_factorized', args.steps)
            old = run(args.collision, args.output/(name+'-collision'), text)
            new = run(args.current, args.output/(name+'-current'), text)
            assert np.array_equal(old[:, :6], new[:, :6])
            difference = abs(old[:, 6]-new[:, 6])
            normalized = difference/(1+abs(old[:, 6])+abs(new[:, 6]))
            winner = int(np.argmax(normalized))
            record = dict(fixture=name, components=len(old), steps=args.steps,
                          max_normalized_error=float(normalized[winner]),
                          max_absolute_error=float(difference.max()),
                          worst_cell_operation=old[winner, :6].tolist(),
                          status='PASS' if normalized[winner]<=2e-12 else 'FAIL')
            if not args.steps:
                direct = run(args.current, args.output/(name+'-direct'),
                             input_text(order, dim, 'direct_product', 0))
                assert np.array_equal(old[:, :6], direct[:, :6])
                record['direct_product_negative_control_max_difference'] = float(abs(direct[:, 6]-old[:, 6]).max())
                assert record['direct_product_negative_control_max_difference']>1e-8
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    assert all(r['status']=='PASS' for r in records)


if __name__ == '__main__':
    main()
