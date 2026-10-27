"""Compare verified puncture profiles at one exact, saved common time.

Spatial inputs must be nested grids at fixed CFL; temporal inputs must be the
same grid with successive timestep halvings. No time interpolation is used.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from puncture import PunctureEvolution

FIELDS = ['chi', 'g', 'Arr', 'p+3', 'Theta', 'Lambda', 'alpha', 'beta', 'B']
PHYSICS = ['ratio', 'damping', 'ko', 'restoring', 'inner_shift', 'lapse_floor',
           'shift_source', 'connection_driver', 'shift_gauge', 'gamma_coefficient',
           'grid_stretch', 'shift_inner_stencil', 'inner_z4_damping', 'puncture']
SOURCES = ['evolution.py', 'puncture.py', 'gamma_driver.py', 'constraint_damping.py',
           'principal.py', 'boundary.py', 'shift_source.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', type=Path, required=True)
    p.add_argument('--time', type=float, required=True)
    p.add_argument('--kind', choices=['spatial', 'temporal'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert len(a.cases)==3 or (a.kind=='spatial' and len(a.cases)==2)
    rows, states, manifests, snapshots = [], [], [], {}
    for i, case in enumerate(a.cases):
        m = json.loads((case/'manifest.json').read_text())
        assert m['status'] == 'completed' and m['t']+1e-13 >= a.time
        assert m['arguments']['puncture']
        for rel, sha in m['sources'].items():
            assert digest(case/'source'/rel) == sha, rel
        for fn, sha in m['checkpoint_hashes'].items():
            assert digest(case/fn) == sha, fn
        lib = case/'source'/Path(m['arguments']['library']).name
        assert digest(lib) == m['library_sha256']
        if manifests:
            ref = manifests[0]
            for key in PHYSICS:
                assert m['arguments'].get(key) == ref['arguments'].get(key), key
            for name in SOURCES:
                key = 'tst/hyperbolic_shift_1d/'+name
                assert m['sources'][key] == ref['sources'][key], name
            assert m['library_sha256'] == ref['library_sha256']
            if a.kind == 'spatial':
                assert m['arguments']['cfl'] == ref['arguments']['cfl']
        with np.load(case/'profiles.npz') as z:
            hits = np.flatnonzero(abs(z['t']-a.time) < 1e-13)
            assert len(hits) == 1, (case, 'Need exactly one saved common-time profile')
            q = z['q'][hits[0]].copy()
        kwargs = {k:v for k,v in m['arguments'].items()
                  if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
        e = PunctureEvolution(m['arguments']['n'], lib.resolve(), **kwargs)
        report = e.report(q, a.time)
        v = e.last
        physical = v['physical'][:,:7].copy()
        physical[:,0], physical[:,1], physical[:,6] = v['chi'], v['g'], v['alpha']
        physical = np.c_[physical, v['beta'], q[:,8]]
        weight = e.r**2*e.jac
        weight[[0,-1]] *= .5
        states.append((e.r.copy(), physical, weight, report['apparent_horizon_r']))
        manifests.append(m)
        snapshots[f'r{i}'], snapshots[f'fields{i}'] = e.r, physical
        rows.append(dict(case=str(case), n=len(q), dt=m['dt'],
                         hashes={fn:digest(case/fn) for fn in ['manifest.json','history.json','profiles.npz']},
                         diagnostics={k:report[k] for k in ['H_exterior_rms','M_exterior_rms',
                                      'apparent_horizon_mass','apparent_horizon_r','H_interior_max','g_min']}))
    comparisons = []
    for i, ((r,q,w,ah),(rf,qf,wf,ahf)) in enumerate(zip(states,states[1:])):
        if a.kind == 'spatial':
            assert len(rf) == 2*len(r)-1 and np.array_equal(r,rf[::2])
            d = q-qf[::2]
        else:
            assert np.array_equal(r,rf)
            assert manifests[i]['dt'] == 2*manifests[i+1]['dt']
            d = q-qf
        item = {}
        for scope, mask in [('full', r>=0), ('exterior', r>=max(ah,ahf))]:
            rms = np.sqrt(np.sum(w[mask,None]*d[mask]**2,axis=0)/sum(w[mask]))
            item[scope] = dict(field_rms=rms.tolist(), mixed_rms=float(np.linalg.norm(rms)))
        comparisons.append(item)
    orders = {scope:float(np.log2(comparisons[0][scope]['mixed_rms']/comparisons[1][scope]['mixed_rms']))
              for scope in ['full','exterior']} if len(comparisons)==2 else None
    result = dict(kind=a.kind,t=a.time,t_over_M=3*a.time,fields=FIELDS,runs=rows,
                  comparisons=comparisons,orders=orders,
                  norm='Radial r^2 times Jacobian trapezoidal weights; exterior mask begins at the larger interpolated AH radius in each pair.',
                  scope='Exact saved common-time profiles, verified identical equation sources; no convergence claim beyond the sampled time.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'physical-profiles.npz',**snapshots)
    print(json.dumps(dict(kind=a.kind,t=a.time,orders=orders,comparisons=comparisons),indent=2))


if __name__ == '__main__':
    main()
