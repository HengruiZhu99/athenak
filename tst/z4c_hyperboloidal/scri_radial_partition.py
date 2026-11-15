#!/usr/bin/env python3
"""Partition native single-hole exterior constraint integrals without interpolation.

The inner exterior below r=.8 is the complement of four native outer shells.
This tool requires the archived input to prove an origin-centered fixed horizon,
and checks a conservative bound on every saved harmonic surface. It must not be
applied to a moving/binary horizon or different shell definitions by assumption.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scri_run_summary import history

KEYS = ['H_M0_squared', 'M_M0_squared', 'Theta_M0', 'Z_M0']
UNITS = np.array([1/9, 1/9, 1/3, 1/3])
REGIONS = ['inner_exterior', 'shell_0', 'shell_1', 'shell_2', 'shell_3']


def input_path(root):
    path = root/'input.athinput'
    if not path.exists():
        manifest = json.loads((root/'merge-manifest.json').read_text())
        path = Path(manifest['segments'][0]['directory'])/'input.athinput'
    assert path.is_file(), ('missing original input', path)
    return path


def horizon_bound(root, inp):
    text = inp.read_text()
    for key, value in [('npunct', 0), ('use_puncture_0', -1),
                       ('center_x_0', 0), ('center_y_0', 0), ('center_z_0', 0)]:
        matches = re.findall(r'^\s*'+key+r'\s*=\s*([^#\s]+)', text, re.M)
        assert len(matches) == 1 and float(matches[0]) == value, ('unsupported horizon center', key)
    shape = next(root.glob('*.horizon_shape_0.txt'))
    a = np.atleast_2d(np.loadtxt(shape))
    assert np.isfinite(a).all() and a.shape[1] == 169, 'requires lmax=12 native real-harmonic shape'
    upper = a[:,0]/np.sqrt(4*np.pi)
    for l in range(1,13):
        row = a[:,l*l:(l+1)*(l+1)]
        upper += np.sqrt((2*l+1)/(4*np.pi))*(abs(row[:,0])+
                    np.sqrt(2)*np.sum(abs(row[:,1:]),axis=1))
    # Native text prints six decimal digits; enlarge the conservative bound.
    bound = float(np.max(upper)*1.00001)
    assert 0 < bound < .8, ('horizon intersects fixed shells', bound)
    return bound, shape


def partition(root):
    root = Path(root)
    inp = input_path(root)
    bound, shape = horizon_bound(root, inp)
    proof_paths = [inp,shape]
    manifest = json.loads((root/'merge-manifest.json').read_text()) if (root/'merge-manifest.json').exists() else None
    if manifest:
        for segment in manifest['segments']:
            segment_input = Path(segment['directory'])/'input.athinput'
            horizon_bound(root,segment_input)
            proof_paths.append(segment_input)
    paths = [next(root.glob('*.z4c_hyperboloidal.hst'))]
    paths += [next(root.glob(f'*.scri-shell-{b}.hst')) for b in range(4)]
    if manifest:
        record = manifest['streams'][paths[0].name]
        assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == record['sha256']
    ext, _ = history(paths[0])
    shells = []
    for b, path in enumerate(paths[1:]):
        text = path.read_text()
        edges = ['[0.8,0.88)', '[0.88,0.94)', '[0.94,0.97)', '[0.97,1)']
        if manifest:
            record = manifest['streams'][path.name]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            assert len(manifest['segments']) == len(record['source_sha256'])
            for segment, sha in zip(manifest['segments'],record['source_sha256']):
                raw = Path(segment['directory'])/path.name
                assert hashlib.sha256(raw.read_bytes()).hexdigest() == sha
                assert 'Compact radius band '+edges[b] in raw.read_text(), ('unverified source shell bounds',raw)
                proof_paths.append(raw)
        else:
            assert 'Compact radius band '+edges[b] in text, ('unverified shell bounds', path)
        a, _ = history(path)
        np.testing.assert_array_equal(a[:,0], ext[:,0])
        shells.append(a)
    shell_v = np.stack([a[:,2] for a in shells],axis=1)
    shell_e = np.stack([a[:,2,None]*a[:,3:7]**2 for a in shells],axis=1)
    ext_e = ext[:,7,None]*ext[:,8:12]**2
    inner_v = ext[:,7]-shell_v.sum(axis=1)
    inner_e = ext_e-shell_e.sum(axis=1)
    tol = 100*np.finfo(float).eps*np.maximum(ext_e,shell_e.sum(axis=1))
    assert np.all(inner_v>0) and np.all(shell_v>0), 'invalid partition volume'
    assert np.all(inner_e>=-tol), 'negative inner squared integral'
    inner_e = np.maximum(inner_e,0)
    volumes = np.column_stack([inner_v,shell_v])
    energies = np.concatenate([inner_e[:,None,:],shell_e],axis=1)
    np.testing.assert_allclose(energies.sum(axis=1),ext_e,rtol=2e-14,atol=1e-30)
    rms = np.sqrt(energies/volumes[:,:,None])*UNITS
    fractions = np.divide(energies,ext_e[:,None,:],out=np.zeros_like(energies),
                          where=ext_e[:,None,:]>0)
    return dict(time_M0=(3*ext[:,0]).tolist(), regions=REGIONS, constraint_keys=KEYS,
        rms_by_time_region_constraint=rms.tolist(),
        volume_by_time_region=volumes.tolist(),
        squared_integral_fractions=fractions.tolist(),
        zero_exterior_integral=(ext_e==0).tolist(),
        maximum_horizon_radius_bound=bound,
        final_subtraction_conditioning=[float(e/i) if i>0 else None
                                        for e,i in zip(ext_e[-1],inner_e[-1])],
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in proof_paths+paths},
        meaning='Compact-coordinate weighted constraint norms. Inner exterior is a complement of disjoint native shells; not solution-field self-convergence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = partition(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['maximum_horizon_radius_bound','final_subtraction_conditioning']}))
