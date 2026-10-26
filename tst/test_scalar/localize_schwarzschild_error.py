#!/usr/bin/env python3
"""Read-only regional and scri diagnostics of preserved scalar snapshots.

No evolution, coefficient construction, checkpoint modification or admission.
Plots use compact coordinates. Quantitative spatial norms use all active nodes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from scalar_evidence import parent
from reference_interpolation import lagrange6

FIELDS = ['psi', 'v', 'Dpsi', 'q', 'reduction']
REGIONS = ['inner exterior', 'cube interface', 'inner shell', 'transition', 'outer shell', 'near scri']
COLORS = ['#d55e00', '#0072b2', '#009e73']
INPUTS = set()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    INPUTS.add(Path(path).resolve())
    return json.loads(Path(path).read_text())


def snapshot(run, time):
    path = run / f'fields-t{time:.3f}.f64'
    if not path.exists():
        previous = parent(run)
        if previous is None:
            raise FileNotFoundError(path)
        for name in ['analysis-parent.json', 'restart-invocation.json']:
            if (run/name).exists():
                INPUTS.add((run/name).resolve())
        return snapshot(previous, time)
    INPUTS.add(path.resolve())
    return np.memmap(path, dtype='float64', mode='r').reshape(-1, 12)


def profile(base, n, time):
    path = base / f'waveform-upwind-n{n}.csv-profile-{time}.csv'
    INPUTS.add(path.resolve())
    return np.genfromtxt(path, delimiter=',', names=True)


def truth(data, reference):
    rho = np.linalg.norm(data[:, :3], axis=1)
    normal = data[:, :3]/rho[:, None]
    angular = normal[:, 0]*normal[:, 1]
    p, v, dp = (lagrange6(reference['rho'], reference[key], rho) for key in ['psi', 'v', 'Dpsi'])
    da = -2*angular[:, None]*normal
    da[:, 0] += normal[:, 1]
    da[:, 1] += normal[:, 0]
    return angular*p, angular*v, da*p[:, None]/rho[:, None]+angular[:, None]*normal*dp[:, None]


def region_ids(xyz):
    rho = np.linalg.norm(xyz, axis=1)
    cube = np.max(abs(xyz), axis=1)
    # Fixed physical subsets, identical across resolutions; partition rho>=1.2.
    return np.select([cube < 2.5, cube < 3.5, rho < 8, rho < 16, rho < 22], range(5), default=5)


def stats(data, refs, edges):
    # accumulator columns: volume, five squared errors, five maxima,
    # three reference-difference squares, three reference-difference maxima.
    accum = np.zeros((2, 7, 17))
    bins = np.zeros((len(edges)-1, 7))
    max_locations = {key: [0., None] for key in FIELDS}
    for start in range(0, len(data), 65536):
        d = np.asarray(data[start:start+65536])
        r = np.linalg.norm(d[:, :3], axis=1)
        a, b = (truth(d, ref) for ref in refs)
        errors = np.column_stack((d[:, 4]-b[0], d[:, 5]-b[1],
                                  np.linalg.norm(d[:, 9:12]-b[2], axis=1),
                                  np.linalg.norm(d[:, 6:9]-b[2], axis=1),
                                  np.linalg.norm(d[:, 6:9]-d[:, 9:12], axis=1)))
        delta = np.column_stack((a[0]-b[0], a[1]-b[1], np.linalg.norm(a[2]-b[2], axis=1)))
        regions = region_ids(d[:, :3])
        for extent in [0, 1]:
            eligible = r >= 1.2 if extent == 0 else r < 1.2
            for region in range(7):
                mask = eligible & ((regions == region) if region < 6 else True)
                if not mask.any():
                    continue
                w, e, de = d[mask, 3], errors[mask], delta[mask]
                ac = accum[extent, region]
                ac[0] += sum(w)
                ac[1:6] += np.sum(w[:, None]*e**2, axis=0)
                ac[6:11] = np.maximum(ac[6:11], np.max(abs(e), axis=0))
                ac[11:14] += np.sum(w[:, None]*de**2, axis=0)
                ac[14:17] = np.maximum(ac[14:17], np.max(abs(de), axis=0))
        eligible = r >= 1.2
        for j, key in enumerate(FIELDS):
            e = np.where(eligible, abs(errors[:, j]), -1)
            k = np.argmax(e)
            if e[k] > max_locations[key][0]:
                max_locations[key] = [float(e[k]), d[k, :3].tolist()]
        bi = np.minimum(np.searchsorted(edges, r, side='right')-1, len(edges)-2)
        for k in range(len(edges)-1):
            mask = eligible & (bi == k)
            w = d[mask, 3]
            bins[k, 0] += sum(w)
            bins[k, 1:6] += np.sum(w[:, None]*errors[mask]**2, axis=0)
            bins[k, 6] += np.sum(w*(np.max(abs(d[mask, :3]), axis=1)-3)**2)
    def unpack(ac):
        return {key: dict(L2=float(np.sqrt(ac[1+j]/ac[0])), Linf=float(ac[6+j]),
                          squared_error=float(ac[1+j]),
                          reference_fraction_L2=float(np.sqrt(ac[11+j]/ac[1+j])) if j < 3 else None,
                          reference_fraction_Linf=float(ac[14+j]/ac[6+j]) if j < 3 else None)
                for j, key in enumerate(FIELDS)}
    result = {name: dict(volume=float(accum[0, j, 0]), errors=unpack(accum[0, j]))
              for j, name in enumerate(REGIONS+['all exterior'])}
    result['excluded inner region'] = dict(volume=float(accum[1, 6, 0]), errors=unpack(accum[1, 6]))
    for name in REGIONS:
        for key in FIELDS:
            result[name]['errors'][key]['squared_error_fraction'] = result[name]['errors'][key]['squared_error']/result['all exterior']['errors'][key]['squared_error']
    return result, bins, max_locations


class Sampler:
    """Six-point tensor sampling of active patch data, for illustrations only."""
    def __init__(self, data, n):
        self.n = n
        shell_size = (2*n+1)*n*n
        bulk_size = len(data)-6*shell_size
        bulk = np.full((2*n, 2*n, 2*n, 12), np.nan)
        d = data[:bulk_size]
        index = np.rint((d[:, :3]+3)/(3/n)-.5).astype(int)
        bulk[index[:, 2], index[:, 1], index[:, 0]] = d
        self.blocks = [bulk] + [data[bulk_size+k*shell_size:bulk_size+(k+1)*shell_size].reshape(n, n, 2*n+1, 12) for k in range(6)]
        self.data = data
        self.bulk_size = bulk_size

    def __call__(self, xyz, columns=(4,)):
        xyz = np.atleast_2d(xyz)
        axis = np.argmax(abs(xyz), axis=1)
        block = 1+2*axis+(xyz[np.arange(len(xyz)), axis] < 0)
        block[np.max(abs(xyz), axis=1) < 3] = 0
        out = np.empty((len(xyz), len(columns)))
        for bid in range(7):
            take = np.where(block == bid)[0]
            if not len(take):
                continue
            x = xyz[take]
            if bid == 0:
                q = (x+3)/(3/self.n)-.5
            else:
                a = (bid-1)//2
                d = abs(x[:, a]); u = x[:, (a+1)%3]/d; v = x[:, (a+2)%3]/d
                t = (d-3)/(24/np.sqrt(1+u*u+v*v)-3)
                s = np.log1p(np.expm1(2)*t)/2
                q = np.column_stack((s*2*self.n, (u+1)*self.n/2-.5, (v+1)*self.n/2-.5))
            shape = np.array(self.blocks[bid].shape[:3])[::-1]
            start = np.clip(np.floor(q).astype(int)-2, 0, shape-6)
            weights = np.ones((3, 6, len(take)))
            for a in range(3):
                for i in range(6):
                    for j in range(6):
                        if i != j:
                            weights[a, i] *= (q[:, a]-start[:, a]-j)/(i-j)
            anchor = self.blocks[bid][start[:, 2]+2, start[:, 1]+2, start[:, 0]+2][:, columns]
            value = anchor.copy()
            for k in range(6):
                for j in range(6):
                    for i in range(6):
                        w = weights[0, i]*weights[1, j]*weights[2, k]
                        node = self.blocks[bid][start[:, 2]+k, start[:, 1]+j, start[:, 0]+i][:, columns]
                        value += w[:, None]*(node-anchor)
            out[take] = value
        return out

    def scri(self):
        return np.concatenate([block[:, :, -1, :] for block in self.blocks[1:]]) .reshape(-1, 12)


def scri_values(data, n, refs):
    shell_size = (2*n+1)*n*n
    start = len(data)-6*shell_size
    d = np.concatenate([data[start+k*shell_size:start+(k+1)*shell_size].reshape(n, n, 2*n+1, 12)[:, :, -1, :].reshape(-1, 12) for k in range(6)])
    rho = np.linalg.norm(d[:, :3], axis=1)
    if np.max(abs(rho-24)) > 1e-11:
        raise RuntimeError('Expected the exact scri endpoint')
    normal = d[:, :3]/rho[:, None]
    Y = normal[:, 0]*normal[:, 1]
    # du dv/(1+u^2+v^2)^(3/2) on each face. No radial/volume weight here.
    w = (2/n)**2*np.max(abs(normal), axis=1)**3
    y22 = np.sqrt(15/(32*np.pi))*(normal[:, 0]-1j*normal[:, 1])**2
    continuum_factor = np.sqrt(2*np.pi/15)
    result = dict(points=len(d), max_radius_error=float(np.max(abs(rho-24))), area_weight_sum=float(sum(w)), fields={})
    for key, col, refcol in [('psi', 4, 'psi'), ('v', 5, 'v')]:
        p = refs[1][refcol][-1]
        refdelta = refs[0][refcol][-1]-p
        amplitude = np.sum(w*Y*d[:, col])/np.sum(w*Y*Y)
        error = d[:, col]-Y*p
        raw = sum(w*d[:, col]*y22)
        matched = sum(w*Y*p*y22)
        result['fields'][key] = dict(radial_amplitude=float(amplitude), reference_radial_amplitude=float(p),
            radiation_mode_imag=float(-24*continuum_factor*amplitude), reference_radiation_mode_imag=float(-24*continuum_factor*p),
            mode_error=float(24*continuum_factor*abs(amplitude-p)),
            angular_L2=float(np.sqrt(sum(w*error**2)/sum(w))), angular_Linf=float(max(abs(error))),
            angular_contamination_L2=float(np.sqrt(sum(w*(d[:, col]-Y*amplitude)**2)/sum(w))),
            reference_mode_difference=float(24*continuum_factor*abs(refdelta)),
            raw_projected_radiation_mode=[float(24*raw.real), float(24*raw.imag)],
            raw_projection_error=float(24*abs(raw+1j*continuum_factor*p)),
            matched_quadrature_projection_error=float(24*abs(raw-matched)),
            raw_quadrature_bias=float(24*abs(matched+1j*continuum_factor*p)))
    return result


def orders(values, ns):
    return [float(np.log(values[j]/values[j+1])/np.log(ns[j+1]/ns[j])) if min(values[j:j+2]) > 0 else None for j in range(2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    E, out = args.evidence.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError('Choose a new diagnostic directory; preserve existing evidence')
    out.mkdir(parents=True)
    original = read(E/'cpu-schwarzschild-commensurate-finer/primary-resolved-reference-review.json')
    runs = [Path(p) for p in original['runs']]
    parameters = [read(run/'parameters.json') for run in runs]
    ns = [p['n'] for p in parameters]
    for p, run in zip(parameters, runs):
        if p['cap'] != 1 or p['M'] != 1 or p['a_over_M'] != 0:
            raise RuntimeError('This diagnostic expects exact-scri Schwarzschild runs')
        if not read(run/'frozen-integrity.json')['unchanged']:
            raise RuntimeError('Frozen background fingerprint changed')
    O = E/'radial-oracle/smooth'
    edges = np.linspace(0, 24, 97)
    result = dict(resolutions=ns, times=[2, 6, 12, 100], regional={}, maxima={}, scri={}, scope='Read-only postprocessing; no new trajectories or scientific admission',
                  regions=dict(zip(REGIONS, ['rho>=1.2, max|y_i|<2.5', '2.5<=max|y_i|<3.5', 'max|y_i|>=3.5, rho<8', '8<=rho<16', '16<=rho<22', '22<=rho<=24'])),
                  scri_method='Exact endpoint, chart area weights, normalized known nx*ny harmonic projection; continuum radiation r*Phi=S*psi. Raw and quadrature-matched projections also recorded. Sparse saved times only, not a resolved ringdown trace.')
    arrays = dict(bin_edges=edges)
    plane = np.linspace(-24, 24, 361)
    xx, yy = np.meshgrid(plane, plane)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    valid = (np.linalg.norm(points, axis=1) <= 24-1e-10) & (np.linalg.norm(points, axis=1) >= .9)
    arrays.update(plane=plane, plane_valid=valid.reshape(xx.shape))
    ray = np.array([.8, .6, 0.])
    radii = np.linspace(1.2, 24, 1801)
    arrays['ray_rho'] = radii
    for time in [2, 6, 12, 100]:
        refs = [profile(O, n, time) for n in [2400, 4800]]
        result['scri'][time] = {}
        for n, run in zip(ns, runs):
            data = snapshot(run, time)
            result['scri'][time][n] = scri_values(data, n, refs)
            if time == 100:
                print(f'Full spatial diagnostic N={n}', flush=True)
                regional, bins, maxima = stats(data, refs, edges)
                result['regional'][n], result['maxima'][n] = regional, maxima
                arrays[f'bins_n{n}'] = bins
                sampler = Sampler(data, n)
                # Check exact-coordinate interpolation on actual non-excised nodes.
                sample = np.linspace(sampler.bulk_size, len(data)-1, 36).astype(int)
                nodal_error = float(np.max(abs(sampler(data[sample, :3], (4,))-data[sample, 4:5])))
                result.setdefault('sampler_nodal_errors', {})[n] = nodal_error
                if nodal_error > 1e-12:
                    raise RuntimeError('Illustration sampler failed node reproduction')
                query = points[valid]
                values = sampler(query, (4, 9, 10, 11))
                ref = truth(np.column_stack((query, np.zeros((len(query), 9)))), refs[1])
                for key, error in [('psi', abs(values[:, 0]-ref[0])), ('Dpsi', np.linalg.norm(values[:, 1:]-ref[2], axis=1))]:
                    image = np.full(xx.size, np.nan); image[valid] = error
                    arrays[f'plane_{key}_n{n}'] = image.reshape(xx.shape)
                del sampler
            if time in [12, 100]:
                sampler = Sampler(data, n)
                arrays[f'ray_psi_t{time}_n{n}'] = sampler(radii[:, None]*ray)[:, 0]/(ray[0]*ray[1])
                # Radial shell nodes along this ray, sampled only in angle.
                s = np.arange(2*n+1)/(2*n)
                node_r = 3/.8+(24-3/.8)*np.expm1(2*s)/np.expm1(2)
                arrays[f'ray_nodes_n{n}'] = node_r
                arrays[f'ray_node_psi_t{time}_n{n}'] = sampler(node_r[:, None]*ray)[:, 0]/(ray[0]*ray[1])
                del sampler
            del data
        if time in [12, 100]:
            arrays[f'ray_reference_t{time}'] = lagrange6(refs[1]['rho'], refs[1]['psi'], radii)
            arrays[f'ray_reference_difference_t{time}'] = abs(lagrange6(refs[0]['rho'], refs[0]['psi'], radii)-arrays[f'ray_reference_t{time}'])
    result['regional_orders'] = {region: {key: {norm: orders([result['regional'][n][region]['errors'][key][norm] for n in ns], ns) for norm in ['L2', 'Linf']} for key in FIELDS} for region in REGIONS+['all exterior', 'excluded inner region']}
    result['scri_orders'] = {time: {key: {norm: orders([result['scri'][time][n]['fields'][key][norm] for n in ns], ns) for norm in ['mode_error', 'angular_L2', 'angular_Linf']} for key in ['psi', 'v']} for time in result['times']}
    # Reproduce the original failing numbers using its 2400-point reference.
    verification = []
    from validate_schwarzschild import compare
    reference2400 = profile(O, 2400, 100)
    old_check = next(c for c in original['field_checks'] if c['time'] == 100 and c['field'] == 'psi' and c['norm'] == 'L2')
    for n, run, expected in zip(ns, runs, old_check['errors']):
        measured = compare(snapshot(run, 100), reference2400, sampler='lagrange6')['psi']['L2']
        verification.append(dict(n=n, old=expected, measured=measured, relative_difference=abs(measured/expected-1)))
    if max(v['relative_difference'] for v in verification) > 1e-10:
        raise RuntimeError('Did not reproduce preserved failing field gate')
    result['original_gate_reproduction'] = verification
    np.savez_compressed(out/'plot-data.npz', **arrays)
    plot(result, arrays, out)
    result['additional_trajectories'] = 0
    result['input_sha256'] = {str(p): sha(p) for p in sorted(INPUTS)}
    result['source_sha256'] = {str(Path(__file__).resolve()): sha(__file__), str(Path(__file__).with_name('reference_interpolation.py').resolve()): sha(Path(__file__).with_name('reference_interpolation.py'))}
    result['output_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), regional_psi_orders=result['regional_orders'], scri_orders=result['scri_orders']), indent=2), flush=True)


def save(fig, out, name):
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'{name}.{ext}', dpi=170)
    plt.close(fig)


def plot(r, a, out):
    ns = r['resolutions']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for row, key in enumerate(['psi', 'Dpsi']):
        vmax = max(np.nanpercentile(a[f'plane_{key}_n{n}'], 99.8) for n in ns[1:])
        for col, n in enumerate(ns[1:]):
            ax = axes[row, col]
            im = ax.imshow(a[f'plane_{key}_n{n}'], origin='lower', extent=[-24, 24, -24, 24], cmap='magma', norm=LogNorm(vmax/1e5, vmax), interpolation='nearest')
            for radius, ls in [(1.2, ':'), (8, '--'), (16, '--'), (22, ':')]:
                ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor='cyan', linestyle=ls, linewidth=.65))
            ax.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], color='cyan', lw=1)
            ax.set(xlabel='compact x/M', ylabel='compact y/M', title=f'N={n}: '+('$|\\psi-\\psi_{ref}|$' if key == 'psi' else '$|D\\psi-D\\psi_{ref}|$'), aspect='equal')
        fig.colorbar(im, ax=axes[row, :], label='absolute error; common scale in row', shrink=.85)
    fig.suptitle('Schwarzschild at 100M: equatorial error, exact scri at radius 24M\nSix-point sampling for display; all-node 3D norms measured separately', fontsize=12)
    save(fig, out, 'spatial-error-map')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    mid = (a['bin_edges'][1:]+a['bin_edges'][:-1])/2
    for col, key in enumerate(['psi', 'Dpsi']):
        for n, color in zip(ns, COLORS):
            b = a[f'bins_n{n}']; j = FIELDS.index(key)+1
            axes[0, col].semilogy(mid, np.sqrt(np.divide(b[:, j], b[:, 0], out=np.full(len(b), np.nan), where=b[:, 0]>0)), color=color, label=f'N={n}')
        ax = axes[0, col]; ax.set(xlabel='compact radius rho/M', ylabel='radial-bin RMS error', title=key+' at 100M');ax.legend()
        for x in [8, 16, 22]:ax.axvline(x, color='gray', ls=':', lw=.8)
    x = np.arange(len(REGIONS)); width = .24
    for k, (n, color) in enumerate(zip(ns, COLORS)):
        share = [100*r['regional'][n][reg]['errors']['psi']['squared_error_fraction'] for reg in REGIONS]
        axes[1, 0].bar(x+(k-1)*width, share, width=width, color=color, label=f'N={n}')
    axes[1, 0].set(ylabel='share of total field squared error (%)', xticks=x, xticklabels=REGIONS, title='Where the full exterior field error comes from')
    axes[1, 0].tick_params(axis='x', labelrotation=25);axes[1, 0].legend()
    for key, marker in [('psi', 'o'), ('Dpsi', 's'), ('reduction', '^')]:
        axes[1, 1].plot(x, [r['regional_orders'][reg][key]['L2'][1] for reg in REGIONS], marker+'-', label=key)
    axes[1, 1].axhline(1.5, color='black', ls='--', label='declared minimum')
    axes[1, 1].set(xticks=x, xticklabels=REGIONS, ylabel='observed order, N48/60', title='Fixed-region convergence of RMS errors')
    axes[1, 1].tick_params(axis='x', labelrotation=25);axes[1, 1].legend()
    fig.suptitle('Localization by fixed regions: interface = max|coordinate| in [2.5,3.5]M; near scri = rho>=22M', fontsize=12)
    save(fig, out, 'regional-convergence')

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for col, time in enumerate([12, 100]):
        rho, ref = a['ray_rho'], a[f'ray_reference_t{time}']
        axes[0, col].plot(rho, ref, 'k-', lw=1.4, label='independent 4800')
        for n, color in zip(ns, COLORS):
            axes[0, col].plot(a[f'ray_nodes_n{n}'], a[f'ray_node_psi_t{time}_n{n}'], '.', ms=2.8, color=color, label=f'N={n} shell radial nodes')
            axes[1, col].semilogy(rho, abs(a[f'ray_psi_t{time}_n{n}']-ref), color=color, label=f'N={n}')
            nodes = a[f'ray_nodes_n{n}']
            axes[2, col].plot((nodes[:-1]+nodes[1:])/2, np.diff(nodes), color=color, label=f'N={n}')
        axes[1, col].semilogy(rho, np.maximum(a[f'ray_reference_difference_t{time}'], 1e-20), 'k:', label='2400/4800 reference difference')
        axes[0, col].set(title=f't={time}M: radial amplitude psi/(nx ny)', ylabel='radial amplitude')
        axes[1, col].set(ylabel='absolute radial-amplitude error')
        axes[2, col].set(ylabel='shell radial grid spacing delta rho/M', xlabel='compact radius rho/M')
        for row in range(3):
            ax=axes[row, col];ax.axvline(3.75, color='gray', ls=':', lw=1);ax.axvspan(8,16,color='gray',alpha=.08);ax.axvline(22,color='gray',ls=':',lw=.8);ax.set_xlim(1.2,24)
            ax.legend(fontsize=8)
    fig.suptitle('Ray direction (nx,ny,nz)=(0.8,0.6,0): cube exit at rho=3.75M; scri at 24M\nDots retain actual shell radial node spacing; angular and dense-ray display use six-point sampling', fontsize=12)
    save(fig, out, 'radial-profiles')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    times = r['times']; palette = ['#cc79a7', '#e69f00', '#56b4e9', '#000000']
    for time, color in zip(times, palette):
        values = [r['scri'][time][n]['fields']['psi'] for n in ns]
        axes[0, 0].loglog(ns, [v['mode_error'] for v in values], 'o-', color=color, label=f't={time}M; p48/60={r["scri_orders"][time]["psi"]["mode_error"][1]:.2f}')
        axes[0, 1].loglog(ns, [v['angular_L2'] for v in values], 'o-', color=color, label=f't={time}M; p48/60={r["scri_orders"][time]["psi"]["angular_L2"][1]:.2f}')
        axes[1, 0].loglog(ns, [r['scri'][time][n]['fields']['v']['mode_error'] for n in ns], 'o-', color=color, label=f't={time}M; p48/60={r["scri_orders"][time]["v"]["mode_error"][1]:.2f}')
    for ax, label in zip([axes[0,0],axes[0,1],axes[1,0]], ['radiation (2,2) mode absolute error', 'endpoint angular RMS field error', 'radiation time-derivative mode error']):
        ax.set(xlabel='resolution N', ylabel=label, xticks=ns);ax.set_xticklabels(ns);ax.legend(fontsize=8)
    ax = axes[1, 1]
    for n, color in zip(ns, COLORS):
        ax.scatter(times, [r['scri'][t][n]['fields']['psi']['radiation_mode_imag'] for t in times], color=color, label=f'N={n}', s=30)
    ax.scatter(times, [r['scri'][t][ns[-1]]['fields']['psi']['reference_radiation_mode_imag'] for t in times], marker='x', color='black', label='independent reference', s=65)
    ax.set(xlabel='t/M', ylabel='Im radiation mode at exact scri', title='Saved endpoint samples only; no temporal interpolation');ax.legend(fontsize=8)
    fig.suptitle('Exact-scri radiation: r Phi -> 24 psi; normalized harmonic projection removes quadrature bias\nSparse snapshot convergence is measurable; these samples do not resolve the ringdown time series', fontsize=12)
    save(fig, out, 'scri-convergence')


if __name__ == '__main__':
    main()
