"""Read-only interpolation sensitivity of horizon mass on a frozen trajectory.

Uses the run's verified frozen numerical code and library. Varying extraction
order on one grid does not measure evolution convergence.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    case = a.case.resolve()
    m = json.loads((case/'manifest.json').read_text())
    assert m['status'] in ['completed', 'failed', 'bounded_stop']
    assert m['arguments']['puncture']
    for rel, sha in m['sources'].items():
        assert digest(case/'source'/rel) == sha, rel
    for name, sha in m['checkpoint_hashes'].items():
        assert digest(case/name) == sha, name
    lib = case/'source'/Path(m['arguments']['library']).name
    assert digest(lib) == m['library_sha256']
    sys.path.insert(0, str(case/'source/tst/hyperbolic_shift_1d'))
    from puncture import PunctureEvolution
    from horizon import apparent_horizons
    assert Path(inspect.getfile(PunctureEvolution)).resolve().parent == case/'source/tst/hyperbolic_shift_1d'
    args = {k:v for k,v in m['arguments'].items()
            if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
    e = PunctureEvolution(m['arguments']['n'], lib, **args)
    z = np.load(case/'profiles.npz')
    assert np.array_equal(e.r, z['r'])
    indices = sorted(set([0, len(z['q'])//2, len(z['q'])-1]))
    snapshots = [(float(z['t'][i]), z['q'][i].copy()) for i in indices]
    if snapshots[-1][0] != float(z['last_time']):
        snapshots.append((float(z['last_time']), z['last'].copy()))
    rows, profiles = [], {}
    for i, (t, q) in enumerate(snapshots):
        original = q.copy()
        e.rhs(q, diagnostics=True)
        assert np.array_equal(q, original), 'Diagnostic mutated the checkpoint'
        v = e.last
        r = e.r[1:-1]
        angular_chi = v['chi'][1:-1]*np.sqrt(v['g'][1:-1])
        expansion = v['diag'][1:-1,11]
        methods = []
        for width in [3,5,7,9,11]:
            h = apparent_horizons(r, angular_chi, expansion, width=width)
            assert h['apparent_horizon_count'] >= 1
            methods.append(dict(width=width, **h,
                                relative_mass_error=3*h['apparent_horizon_mass']-1))
        masses = np.array([x['apparent_horizon_mass'] for x in methods])
        reference = methods[2]
        high_order_spread = float(3*(max(masses[2:])-min(masses[2:])))
        rows.append(dict(t=t, t_over_M=3*t, methods=methods,
                         relative_spread_widths_7_9_11=high_order_spread,
                         width7_relative_error=reference['relative_mass_error'],
                         horizon_radius=reference['apparent_horizon_r'],
                         outgoing_crossing_brackets=[[float(r[j]),float(r[j+1])]
                             for j in np.flatnonzero((expansion[:-1]<=0)&(expansion[1:]>0))]))
        profiles[f'frame{i}'] = np.c_[r, v['diag'][1:-1,10], expansion,
            v['diag'][1:-1,0], v['diag'][1:-1,1], angular_chi]
    result = dict(case=str(case), status=m['status'], n=len(e.r),
                  hashes={name:digest(case/name) for name in ['manifest.json','profiles.npz','history.json']},
                  frozen_source_library_and_checkpoints_verified=True,
                  profile_columns=['r','geometric_mass','outgoing_expansion','H','M_r','angular_chi'],
                  rows=rows,
                  scope='Read-only extraction-order sensitivity on fixed saved states. Agreement across widths does not prove the evolved geometry accurate. No puncture mass enters an acceptance criterion.')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(a.output/'profiles.npz', **profiles)
    print(json.dumps([{k:v for k,v in row.items() if k!='methods'} for row in rows], indent=2))


if __name__ == '__main__':
    main()
