"""Measure the requested three-resolution R16 ladder without automatic promotion."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_inputs import ROOT, parse
from read_complete_history import read_history
sys.path.insert(0, str(ROOT/'analysis/pc_gh_localization'))
from plot_qualification import regional_rms, read_cartesian


def fitted_order(h, ratio):
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    a, b = np.log(h[0]/h[1]), np.log(h[1]/h[2])
    def model(p):
        return a/b if abs(p) < 1e-8 else np.expm1(p*a)/(-np.expm1(-p*b))
    lo, hi = -20., 20.
    if not model(lo) <= ratio <= model(hi):
        return None
    for _ in range(80):
        mid = (lo+hi)/2
        if model(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path, nargs=3)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.runs, key=lambda p: -float(parse((p/'used_input.athinput').read_text())['problem']['expected_finest_spacing']))
    params = [parse((r/'used_input.athinput').read_text()) for r in runs]
    h = np.array([float(p['problem']['expected_finest_spacing']) for p in params])
    if not np.allclose(h, [1/8, 1/10, 1/12], rtol=0, atol=1e-15):
        raise ValueError('Not the requested resolution ladder')
    for p in params:
        if any(float(p['mesh'][f'x{d}{s}']) != v for d in (1,2,3) for s,v in [('min',-128),('max',128)]):
            raise ValueError('Wrong domain')
        expected = dict(reduction_system='advective', reduction_profile='smooth_core',
                        project_reduction_constraints='false', project_gauge_constraints='false')
        if any(p['pc_gh'][k] != v for k,v in expected.items()):
            raise ValueError('Wrong candidate')
        for key,value in [('reduction_rate',1),('reduction_inner_rate',16),('reduction_core_radius',.125),
                          ('reduction_taper_radius',.5),('constraint_excise_chi',.0625)]:
            if float(p['pc_gh'][key]) != value:
                raise ValueError(f'Wrong {key}')
    histories, audits, statuses = [], [], []
    for r in runs:
        hist, audit = read_history(next(r.glob('*.pcgh.hst')))
        histories.append(hist)
        audits.append(audit)
        fatals = [line for log in sorted(r.glob('segment-*.log')) for line in log.read_text().splitlines() if 'FATAL ERROR' in line]
        statuses.append(dict(run=str(r), history_end=float(hist['time'][-1]),
                             completed=(r/'completed.json').exists(), fatal_errors=fatals))
    end = min(20., min(x['time'][-1] for x in histories))
    times = np.arange(.5, end+1e-6, .5)
    norms = [regional_rms(hist, 'chi') for hist in histories]
    for hist, norm in zip(histories, norms):
        if np.any(hist['Volume'] <= 0):
            raise ValueError('Empty chi-excised native volume')
        norm['H'] = np.sqrt(hist['H-norm2']/hist['Volume'])
        norm['M'] = np.sqrt(hist['Mhat-norm2']/hist['Volume'])
    report = dict(runs=statuses, finest_spacing=h.tolist(), history_audits=audits,
                  common_end=end, chi_threshold=.0625, constraints={}, field_self_convergence=[],
                  qualification='No automatic promotion; M/256 R16 failed metric positivity at 5.187818M.',
                  scope='Native 3D chi-excised coordinate-volume RMS constraints; moving masks can differ across h. Field differences use a common chi-mask intersection on interpolated Cartesian slices.')
    fig, axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    families = ['GH','H','M','reduction','curl','algebraic']
    titles = dict(GH='GH', H='Hamiltonian', M='Alpha-weighted momentum',
                  reduction='Reduction', curl='Curl', algebraic='Algebraic')
    for ax,family in zip(axes.flat,families):
        sampled = np.stack([np.interp(times, hist['time'], norm[family]) for hist,norm in zip(histories,norms)])
        pair_orders = []
        for i in (0,1):
            valid = (sampled[i] > 1e-14) & (sampled[i+1] > 1e-14)
            orders = np.full(len(times), np.nan)
            orders[valid] = np.log(sampled[i,valid]/sampled[i+1,valid])/np.log(h[i]/h[i+1])
            pair_orders.append([float(x) if np.isfinite(x) else None for x in orders])
        volumes = np.stack([np.interp(times,hist['time'],hist['Volume']) for hist in histories])
        report['constraints'][family] = dict(times=times.tolist(), rms=sampled.tolist(),
            l2=(sampled*np.sqrt(volumes)).tolist(), coordinate_volume=volumes.tolist(), pair_orders=pair_orders)
        for spacing,hist,norm in zip(h,histories,norms):
            ax.semilogy(hist['time'],norm[family],label=f'h=M/{1/spacing:g}')
        ax.set_title(titles[family]); ax.set_xlabel('t/M'); ax.set_ylabel('Coordinate-volume RMS'); ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle('R16: chi >= 0.0625, SMR, outer boundary +/-128M')
    fig.savefig(args.output/'chi-constraints.png',dpi=170)
    plt.close(fig)
    # Store exactly which sampled times are available; do not substitute nearest late data.
    carts = []
    for r in runs:
        by_time = {}
        for path in sorted((r/'cart').glob('*.pcgh_slice.*.bin')):
            state = read_cartesian(path)
            by_time[round(float(state['time']),6)] = (path,state)
        carts.append(by_time)
    groups = dict(w=(0,1),gtilde=(1,7),K=(7,8),Atilde=(8,14),Z=(14,17),Cperp=(17,18),
                  rho=(18,19),beta=(19,22),p=(22,25),Q=(25,43),L=(43,46),B=(46,55),all=(0,55))
    for t in times:
        key = round(float(t),6)
        if any(key not in c for c in carts):
            report['field_self_convergence'].append(dict(time=float(t),missing=True))
            continue
        states = [c[key][1] for c in carts]
        if any(list(s['data']) != list(states[0]['data']) for s in states[1:]):
            raise ValueError('Different field ordering in Cartesian outputs')
        if any(not np.array_equal(states[0][d],s[d]) for s in states[1:] for d in ['x','y','z']):
            raise ValueError('Different Cartesian sample coordinates')
        u = [np.stack(list(s['data'].values()))[:,0].astype(float) for s in states]
        if any(x.shape[0] != 55 or not np.isfinite(x).all() for x in u):
            raise ValueError('Incomplete/nonfinite regular fields')
        mask = np.logical_and.reduce([s['data']['pcgh_w'][0]**2 >= .0625 for s in states])
        if not mask.any():
            raise ValueError('Empty common chi mask')
        row = dict(time=float(t),points=int(mask.sum()),groups={})
        for name,(lo,hi) in groups.items():
            diff = [(u[i][lo:hi]-u[i+1][lo:hi])[:,mask].ravel() for i in (0,1)]
            magnitudes = np.array([np.linalg.norm(d) for d in diff])
            ratio = magnitudes[0]/magnitudes[1] if magnitudes[1] > 0 else np.nan
            row['groups'][name] = dict(rms_differences=(magnitudes/np.sqrt(mask.sum())).tolist(),
                fitted_order=fitted_order(h,ratio),
                alignment=float(np.dot(*diff)/np.prod(magnitudes)) if magnitudes.min() > 0 else None)
        report['field_self_convergence'].append(row)
    (args.output/'convergence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=statuses,common_end=end,outputs=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
