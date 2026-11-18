#!/usr/bin/env python3
"""Measure native Scheme A endpoint order with the spatial grid held fixed."""
import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from adaptive import adaptive_input

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'z4c_hyperboloidal'))
from restart_roundtrip import parse, payload


def seed_radial_probe(checkpoint):
    """Seed a compact smooth lapse perturbation in copies of test-only initial data."""
    values = payload(checkpoint, mode='r+')
    n, overlap = 12, 3
    index = np.arange(n + 2*overlap) - overlap
    k, j, i = np.meshgrid(index, index, index, indexing='ij')
    slots = i.size
    topology = np.frombuffer(checkpoint['topology'], dtype='=i4').reshape(-1,5)
    root_level = int(topology[:,3].min())
    root_cart = 1+int(topology[(topology[:,3]==root_level) & (topology[:,4]==0),:3].max())
    if values.shape[1] != slots+n*n or root_cart not in (1,3):
        raise ValueError('Unsupported radial probe fixture')
    radii = [.08, .24, .48, .72, 1.]
    for gid,(lx,ly,lz,level,chart) in enumerate(topology):
        if chart == 0:
            blocks = root_cart*2**int(level-root_level)
            coords = [(-.1+.2*(block+(axis+.5)/n)/blocks) for block,axis in zip((lx,ly,lz),(i,j,k))]
            radius = np.sqrt(sum(axis**2 for axis in coords))
        else:
            region = int(lx)
            offset = 0 if region == 3 else .5
            radius = radii[region]+(radii[region+1]-radii[region])*(i+offset)/n
        radius = radius.ravel()
        perturbation = np.zeros_like(radius)
        bumps = [(.14,.065), (.72,.095), (.94,.055)]
        if root_cart == 3:
            bumps.append((.04,.02))
        for center, width in bumps:
            z = (radius-center)/width
            inside = abs(z)<1
            perturbation[inside] += 1e-4 * np.exp(1-1/(1-z[inside]**2))
        evolved = values[gid, :slots, 0]>0
        values[gid, :slots, 18] += np.where(evolved, perturbation, 0)
    values.flush()


def evolve(args, name, steps, uniform):
    directory = args.output / name
    directory.mkdir()
    fixture = directory / 'fixture.athinput'
    inp = Path(__file__).with_name('small.athinput').read_text()
    if args.adaptive:
        inp = adaptive_input(args.ranks, uniform=uniform, balanced=False)
        inp += f'\n<mesh_refinement>\nncycle_check = {steps//2}\n'
    if not args.adaptive:
        inp += """
<output2>
file_type = hst
dt = 100
data_format = %24.17e
"""
    fixture.write_text(inp + """
<output3>
file_type = scri
dt = 100
nodal = true
modes = true
lmax = 4
boundary_diagnostics = true
""")
    command = [word.format(ranks=args.ranks) for word in shlex.split(args.launcher)]
    command.extend([str(args.binary.resolve()), '-i', str(fixture),
                    f'time/nlim={steps}', f'time/dt={args.duration / steps:.17g}',
                    'z4c_hyperboloidal/chi_floor=0',
                    'multirate/uniform_reference=' + str(uniform).lower()])
    if args.radial_probe:
        initialize = [*command, 'time/nlim=0']
        with (directory / 'initialize.log').open('w') as stream:
            subprocess.run(initialize, cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True)
        path = next((directory / 'rst').glob('*.rst'))
        seed_radial_probe(parse(path))
        command = command[:command.index('-i')] + ['-r', str(path), f'time/nlim={steps}',
                                                   'output1/last_time=-1']
    with (directory / 'run.log').open('w') as stream:
        subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True)
    (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    checkpoint = parse(sorted((directory / 'rst').glob('*.rst'))[-1])
    if checkpoint['cycle'] != steps or abs(checkpoint['time'] - args.duration) > 1e-14 * args.duration:
        raise ValueError('The runs must end at the same physical time')
    data = payload(checkpoint)
    if not np.isfinite(data).all() or np.any(data[..., 33:36]):
        raise ValueError('Nonfinite state, active floor, or nonpositive chi in convergence run')
    if args.adaptive:
        expected = f'MULTIRATE_REGRID cycle={steps//2} created=7 deleted=0 blocks=32'
        if expected not in (directory/'run.log').read_text():
            raise ValueError('missing refinement at the shared physical midpoint')
    return checkpoint


def errors(coarse, fine):
    if coarse['topology'] != fine['topology'] or coarse['atlas'] != fine['atlas']:
        raise ValueError('comparison requires identical accepted meshes')
    state, reference = payload(coarse), payload(fine)
    n, overlap = 12, 3   # Native fixture geometry
    bulk_slots = (n + 2 * overlap)**3
    evolved = reference[..., 0] > 0
    edge = np.zeros_like(evolved)
    edge[:, bulk_slots:] = True
    # Block IDs with a surface patch form the fast region. Its first three
    # radial nodes and the last three slow nodes straddle its inner interface.
    gid_is_fast = np.any(evolved & edge, axis=1)
    positions = np.arange(bulk_slots) % (n + 2 * overlap) - overlap
    shell_inner = (positions >= 0) & (positions < 3)
    shell_outer = (positions >= n - 3) & (positions < n)
    interface = np.zeros_like(evolved)
    locations = np.frombuffer(fine['topology'],dtype='=i4').reshape(-1,5)
    shell = locations[:,4]>0
    outer = shell & (locations[:,0]==3)
    penultimate = shell & (locations[:,0]==2)
    if not np.array_equal(gid_is_fast,outer) or sum(outer)!=6 or sum(penultimate)!=6:
        raise ValueError('unexpected physical shell layout')
    interface[outer,:bulk_slots] = shell_inner
    interface[penultimate,:bulk_slots] = shell_outer
    result = {}
    for group, mask in [('bulk', ~edge), ('radial_interface', interface), ('scri', edge)]:
        rows = mask & evolved
        # Use all 26 independent evolved variables in their stored units.
        fields = [i for i in range(28) if i not in (6, 12)]
        difference = state[rows][:, fields] - reference[rows][:, fields]
        result[group] = float(np.sqrt(np.mean(difference**2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=2.0**-8)
    parser.add_argument('--adaptive', action='store_true', help='refine the Cartesian parent at half the simulated duration')
    parser.add_argument('--radial-probe', action='store_true')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--launcher', default='mpiexec -n {ranks}')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    immutable_binary = args.output / 'athena'
    shutil.copy2(args.binary, immutable_binary)
    args.binary = immutable_binary
    mixed = [evolve(args, f'mixed-{steps}', steps, False) for steps in (2, 4, 8)]
    reference = evolve(args, 'uniform-32', 32, True)
    norms = [errors(state, reference) for state in mixed]
    orders = {group: [math.log2(norms[i][group] / norms[i+1][group]) if norms[i+1][group] else None
                      for i in range(2)] for group in norms[0]}
    passed = all(value is not None and value >= 3.7 for row in orders.values() for value in row)
    report = dict(adaptive=args.adaptive,radial_probe=args.radial_probe,binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),duration=args.duration, steps=[2, 4, 8], uniform_reference_steps=32,
                  rms_errors=norms, observed_orders=orders, passed=passed)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
