"""Prepare the user-requested R16 SMR ladder; never submit or run evolution."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from make_inputs import parse, write, ROOT
from make_hybrid_controls import candidate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'qualification-runs-20260904/regular-extension/rate16-qualification/pcgh-l16-o6-rk3-smr-r20-R64.athinput'
    base = parse(source.read_text())
    cases = []
    for n in (16, 20, 24):
        p = deepcopy(base)
        candidate(p, 'R16')
        name = f'R16-smr128-h{n//2}-t20'
        p['job']['basename'] = name
        p['time'].update(tlim='20', ndiag='1')
        p['pc_gh']['constraint_excise_chi'] = '0.0625'
        p['problem']['expected_finest_spacing'] = str(2/n)
        for d in (1, 2, 3):
            p['mesh'].update({f'nx{d}': str(n), f'x{d}min': '-128', f'x{d}max': '128'})
            p['meshblock'][f'nx{d}'] = str(n//2)
        for level in range(1, 8):
            radius = 128/(2**level)-1e-6
            p[f'refined_region{level}'] = {'level': str(level)}
            for d in (1, 2, 3):
                p[f'refined_region{level}'].update({f'x{d}min': str(-radius), f'x{d}max': str(radius)})
        # Same Cartesian sampling grid at all h for exterior self-convergence.
        for output in ('output2', 'output3'):
            p[output].update(extent_x='8', extent_y='8', numpoints_x='128', numpoints_y='128', dt='0.5')
        assert 256/n/2**7 == 2/n
        assert p['pc_gh']['project_reduction_constraints'] == 'false'
        assert p['pc_gh']['reduction_inner_rate'] == '16'
        path = args.output/'smr'/f'{name}.athinput'
        write(path, p)
        cases.append(dict(input=str(path.relative_to(args.output)), finest_spacing=2/n,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidate='R16', tlim=20, boundary=[-128, 128], refinement_levels=7,
        constraint_selection='chi=w^2 >= 0.0625; diagnostic selection only, no evolution excision',
        qualification='Requested exterior convergence study despite M/256 strict failure; binary gate remains closed.',
        cases=cases), indent=2)+'\n')


if __name__ == '__main__':
    main()
