#!/usr/bin/env python3
"""Independently verify the chi-masked ball integrals against full 3D dumps."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vis/python'))
sys.path.insert(0, str(ROOT / 'experiments/single_hole_gauges_20261001'))
from bin_convert import read_binary
from generate import make, parse, dump


def main():
    _, text = make('pulse', 'tel', 32)
    base = parse(text)
    for key in list(base):
        if key.startswith(('refined_region', 'output')):
            del base[key]
    # dx=dy=dz=0.5 resolves both excluded puncture cells and retained r<2 cells.
    # The rectangular domain also contains cells outside the radius-16 sphere.
    for a, n, extent in ((1, 64, 16), (2, 32, 8), (3, 32, 8)):
        base['mesh'].update({f'nx{a}': n, f'x{a}min': -extent, f'x{a}max': extent})
        base['meshblock'][f'nx{a}'] = 16
    base['time'].update(nlim=1, tlim=0.1)
    base['z4c']['horizon_finder'] = 'none'
    # Exercise the new defaults, not just explicit input overrides.
    for key in ('history_inner_radius', 'history_interior_radius', 'excise_chi'):
        base['z4c'].pop(key, None)
    base['output1'] = dict(file_type='hst', dt=1e-8, data_format='%24.16e')
    for n, var in enumerate(('adm', 'z4c', 'con'), 2):
        base[f'output{n}'] = dict(file_type='bin', variable=var, dt=1e-8)

    with tempfile.TemporaryDirectory(prefix='chi-ball-history-') as tmp:
        def run(name, updates=None, fail=False):
            config = copy.deepcopy(base)
            config['z4c'].update(updates or {})
            dest = Path(tmp) / name
            dest.mkdir()
            (dest / 'input').write_text(dump(config))
            result = subprocess.run([str(ROOT / 'build/local-serial/src/athena'),
                                     '-i', 'input'], cwd=dest, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if fail:
                assert result.returncode != 0 and 'history_inner_radius=0' in result.stdout
                return
            assert result.returncode == 0, result.stdout[-4000:]
            return dest, np.loadtxt(next(dest.rglob('*.hst')), ndmin=2)

        dest, hist = run('default')
        # The final output duplicates the last scheduled evolved snapshot.
        assert hist.shape[1] == 20 and len(hist) >= 2, hist.shape
        assert hist[0, 0] == 0 and hist[-1, 0] > 0
        assert 'boundary-exposed-volume' in next(dest.rglob('*.hst')).read_text()
        inner_integrals = []
        for step, row in enumerate(hist):
            dumps = [read_binary(str(p)) for p in sorted(dest.rglob(f'*.{step:05d}.bin'))]
            assert len(dumps) == 3
            assert all(abs(d['time'] - row[0]) < 1e-12 for d in dumps)
            geometry = dumps[0]['mb_geometry']
            fields = {}
            for d in dumps:
                np.testing.assert_array_equal(d['mb_geometry'], geometry)
                fields.update({k: np.asarray(v, dtype=float) for k, v in d['mb_data'].items()})
            chi = fields['z4c_chi']
            nz, ny, nx = chi.shape[1:]
            radius2, dvol = [], []
            for xmin, xmax, ymin, ymax, zmin, zmax in geometry:
                dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz
                x = xmin + (np.arange(nx)+0.5)*dx
                y = ymin + (np.arange(ny)+0.5)*dy
                z = zmin + (np.arange(nz)+0.5)*dz
                radius2.append(x[None, None, :]**2 + y[None, :, None]**2 + z[:, None, None]**2)
                dvol.append(dx*dy*dz)
            radius2 = np.asarray(radius2)
            dvol = np.broadcast_to(np.asarray(dvol)[:, None, None, None], chi.shape)
            xx, xy, xz, yy, yz, zz = [fields['adm_g'+v] for v in ('xx','xy','xz','yy','yz','zz')]
            det = xx*yy*zz + 2*xy*xz*yz - xx*yz**2 - yy*xz**2 - zz*xy**2
            vol = dvol*np.sqrt(np.abs(det))
            ball = radius2 < 16**2
            keep = ball & (chi >= 0.0625)
            inner = keep & (radius2 < 2**2)
            assert np.any(inner) and np.any(ball & ~keep) and np.any(~ball)
            quantities = [fields['con_H']**2, fields['con_M'], fields['con_C']]
            sums = np.array([np.sum((q*vol)[keep]) for q in quantities])
            np.testing.assert_allclose(row[11:14], sums, rtol=2e-6, atol=1e-14)
            expected = [np.sum(vol[keep]), np.sum(dvol[keep]), np.sum(vol[ball & ~keep])]
            np.testing.assert_allclose(row[[14, 17, 18]], expected, rtol=2e-6)
            inner_sums = np.array([np.sum((q*vol)[inner]) for q in quantities])
            # Time-symmetric initial data have identically zero momentum constraint.
            assert np.all(inner_sums >= 0) and inner_sums[0] > 0 and inner_sums[2] > 0
            assert np.all(inner_sums[[0, 2]] > sums[[0, 2]]*1e-3), (inner_sums, sums)
            inner_integrals.append(inner_sums.tolist())

        _, exposed = run('exposed', dict(history_boundary_buffer=100))
        np.testing.assert_array_equal(exposed[:, [11,12,13,14,17,18]], hist[:, [11,12,13,14,17,18]])
        np.testing.assert_array_equal(exposed[:, 15], exposed[:, 14])
        _, unmasked = run('unmasked', dict(excise_chi=0))
        assert np.all(unmasked[:, 18] == 0)
        assert np.all(unmasked[:, 14] > hist[:, 14])
        np.testing.assert_allclose(unmasked[:, 14], hist[:, 14]+hist[:, 18], rtol=1e-12)
        _, disabled = run('disabled', dict(history_interior_radius=0))
        assert disabled.shape == (len(hist), 11)
        np.testing.assert_array_equal(disabled, hist[:, :11])
        run('legacy', dict(history_inner_radius=2), fail=True)
        print('PASS: 3D initial/evolved integrals, r<2 contributions, chi and outer cuts,')
        print('      boundary reporting, unchanged legacy columns, disable and legacy-cut guard.')
        print('Inner r<2 H2/M2/C2 contributions:', inner_integrals)


if __name__ == '__main__':
    main()
