#!/usr/bin/env python3
"""Separate extraction interpolation from genuine 3D angular scatter.

Project independent 1D Cartesian tensors through the SAME sampling as the 3D
state; retain raw exact-radius discrepancies too. No evolution data are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from radial_reference import interpolate


def weights(n, x):
    owner = np.argmax(np.abs(x), axis=1)
    dominant = np.take_along_axis(x, owner[:, None], axis=1)[:, 0]
    face = 2*owner+(dominant < 0)
    bulk = np.max(np.abs(x), axis=1) <= .25
    local = np.empty_like(x)
    block = 1+face
    for p in range(len(x)):
        if bulk[p]:
            block[p] = 0
            local[p] = (x[p]+.25)/.5
        else:
            a = owner[p]
            d = abs(x[p, a])
            u, v = x[p, (a+1)%3]/d, x[p, (a+2)%3]/d
            local[p] = [(d-.25)/(1/np.sqrt(1+u*u+v*v)-.25), (u+1)/2, (v+1)/2]
    t = n*local-.5
    first = np.clip(np.floor(t).astype(int)-2, 0, n-6)
    w = np.ones((len(x), 3, 6))
    for a in range(3):
        for i in range(6):
            for j in range(6):
                if i != j:
                    w[:, a, i] *= (t[:, a]-first[:, a]-j)/(i-j)
    indices, coefficients = [], []
    for k in range(6):
        for j in range(6):
            for i in range(6):
                indices.append(((block*n+first[:, 2]+k)*n+first[:, 1]+j)*n+first[:, 0]+i)
                coefficients.append(w[:, 0, i]*w[:, 1, j]*w[:, 2, k])
    return np.array(indices).T, np.array(coefficients).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    root = parser.parse_args().evidence
    manifest = json.loads((root/'manifest.json').read_text())
    ref = np.load(root/'radial-reference.npz')
    report = dict(scope='extraction audit only; raw errors retained in frozen manifest',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(), cases={})
    for n in (6, 8, 12):
        receipt = manifest['tests'][f'radial-n{n}']
        for key in ('profile', 'spheres'):
            assert hashlib.sha256((root/receipt[key]).read_bytes()).hexdigest() == receipt[key+'_sha256']
        cells = np.genfromtxt(root/receipt['profile'], delimiter=',', names=True)
        spheres = np.genfromtxt(root/receipt['spheres'], delimiter=',', names=True)
        radius = spheres['r']
        z = 1-2*(spheres['direction']+.5)/64
        phi = spheres['direction']*2.39996322972865332
        normal = np.stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z], axis=1)
        index, w = weights(n, radius[:, None]*normal)
        sample = lambda f: np.sum(w*f[index], axis=1)
        cell_x = np.stack([cells['x'], cells['y'], cells['z']], axis=1)
        cell_n = cell_x/cells['r'][:, None]
        delta_alpha = interpolate(ref['r'], ref['alpha']-(1+ref['r']**2)/2, cells['r'])
        arr = interpolate(ref['r'], ref['A_rr'], cells['r'])
        reference_arr = np.zeros_like(radius)
        reproduced_arr = np.zeros_like(radius)
        for t, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            factor = normal[:, i]*normal[:, j]*(1 if i == j else 2)
            tensor = arr*(1.5*cell_n[:, i]*cell_n[:, j]-(.5 if i == j else 0))
            reference_arr += factor*sample(tensor)
            reproduced_arr += factor*sample(cells[f'state_{7+t}'])
        reproduced_alpha = (1+radius**2)/2+sample(cells['state_18'])
        extraction_error = max(float(np.max(np.abs(reproduced_alpha-spheres['alpha']))),
                               float(np.max(np.abs(reproduced_arr-spheres['A_rr']))))
        assert extraction_error < 2e-14, extraction_error
        matched = dict(alpha=(1+radius**2)/2+sample(delta_alpha), A_rr=reference_arr)
        rows = []
        for r in np.unique(radius):
            mask = radius == r
            row = dict(r=float(r))
            for field in ('alpha', 'A_rr'):
                corrected = spheres[field]-matched[field]
                exact = float(interpolate(ref['r'], ref[field], np.array([r]))[0])
                raw = spheres[field]-exact
                row[field] = dict(mean_error=float(np.mean(corrected[mask])),
                                  angular_scatter=float(np.std(corrected[mask])),
                                  raw_mean_error=float(np.mean(raw[mask])),
                                  raw_angular_scatter=float(np.std(raw[mask])),
                                  extraction_bias=float(np.mean(matched[field][mask])-exact))
            rows.append(row)
        report['cases'][str(n)] = dict(extraction_reproduction_maximum=extraction_error, rows=rows)
    (root/'surface-interpolation-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({n: r['extraction_reproduction_maximum'] for n, r in report['cases'].items()}))


if __name__ == '__main__':
    main()
