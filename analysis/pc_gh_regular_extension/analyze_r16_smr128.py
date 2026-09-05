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


def align_snapshot(samples, target):
    """Bracket in time; compare linear and quadratic interpolation explicitly."""
    ts = np.array(sorted(samples))
    close = int(np.argmin(abs(ts-target)))
    def unpack(t):
        path,state = samples[float(t)]
        return path,state,np.stack(list(state['data'].values()))[:,0].astype(float)
    if abs(ts[close]-target) < 1e-6:
        path,state,u = unpack(ts[close])
        return state,u,np.zeros_like(u),dict(method='exact saved time',times=[float(ts[close])],files=[str(path)])
    j = int(np.searchsorted(ts,target))
    if j == 0 or j == len(ts):
        return None
    indices = [j-1,j]
    extra = [i for i in (j-2,j+1) if 0 <= i < len(ts)]
    if not extra:
        return None
    indices.append(min(extra,key=lambda i: abs(ts[i]-target)))
    chosen = [unpack(ts[i]) for i in indices]
    template = chosen[0][1]
    if any(list(s['data']) != list(template['data']) or
           any(not np.array_equal(template[d],s[d]) for d in ('x','y','z')) for _,s,_ in chosen):
        raise ValueError('Different fields or spatial coordinates within time bracket')
    a = (target-ts[j-1])/(ts[j]-ts[j-1])
    linear = (1-a)*chosen[0][2]+a*chosen[1][2]
    weights=[]
    for i in indices:
        weights.append(float(np.prod([(target-ts[k])/(ts[i]-ts[k]) for k in indices if k != i])))
    quadratic = sum(w*item[2] for w,item in zip(weights,chosen))
    return template,quadratic,quadratic-linear,dict(method='bracketed quadratic interpolation',
        times=[float(ts[i]) for i in indices],weights=weights,files=[str(item[0]) for item in chosen],
        sensitivity='Quadratic-minus-linear change is an interpolation sensitivity estimate, not a rigorous error bound.')


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
        # The legacy regional helper combines residuals with the most recent
        # algebraic map correction. A map size is not a residual constraint.
        norm['algebraic'] = np.sqrt((hist['detg-norm2']+hist['trA-norm2']+hist['trQ-norm2'])/hist['Volume'])
    report = dict(runs=statuses, finest_spacing=h.tolist(), history_audits=audits,
                  common_end=end, chi_threshold=.0625, constraints={}, field_self_convergence=[],
                  qualification='No automatic promotion; M/256 R16 failed metric positivity at 5.187818M.',
                  scope='Native 3D chi-excised coordinate-volume RMS constraints; moving masks can differ across h. Field differences use a common chi-mask intersection on Cartesian slices. Asynchronous snapshots are explicitly interpolated in time with a linear/quadratic sensitivity measurement; endpoint results use exact saved time when present.')
    report['algebraic_correction_diagnostic'] = dict(
        scope='RMS of the most recently applied algebraic correction, separate from det/trace residuals. This is neither reduction projection nor a cumulative correction per unit time; last-step sizes differ.',
        runs=[dict(times=hist['time'].tolist(),rms=np.sqrt(hist['proj-norm2']/hist['Volume']).tolist()) for hist in histories])
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
    fig,axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,family in zip(axes.flat,families):
        entry = report['constraints'][family]
        for i,label in enumerate(['M/8 versus M/10','M/10 versus M/12']):
            ax.plot(times,[np.nan if p is None else p for p in entry['pair_orders'][i]],label=label)
        if all(p is None for row in entry['pair_orders'] for p in row):
            ax.text(.5,.7,'Residuals at roundoff;\norder not resolved',ha='center',transform=ax.transAxes)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xlim(0,end)
        ax.set_title(titles[family]); ax.set_xlabel('t/M'); ax.set_ylabel('Observed norm order'); ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle('R16 chi-excised constraint convergence: unequal resolution ratios')
    fig.savefig(args.output/'constraint-orders.png',dpi=170)
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
        aligned = [align_snapshot(c,key) for c in carts]
        if any(a is None for a in aligned):
            report['field_self_convergence'].append(dict(time=float(t),missing=True))
            continue
        states = [a[0] for a in aligned]
        if any(list(s['data']) != list(states[0]['data']) for s in states[1:]):
            raise ValueError('Different field ordering in Cartesian outputs')
        if any(not np.array_equal(states[0][d],s[d]) for s in states[1:] for d in ['x','y','z']):
            raise ValueError('Different Cartesian sample coordinates')
        u = [a[1] for a in aligned]
        if any(x.shape[0] != 55 or not np.isfinite(x).all() for x in u):
            raise ValueError('Incomplete/nonfinite regular fields')
        mask = np.logical_and.reduce([x[0]**2 >= .0625 for x in u])
        if not mask.any():
            raise ValueError('Empty common chi mask')
        row = dict(time=float(t),points=int(mask.sum()),time_alignment=[a[3] for a in aligned],groups={})
        for name,(lo,hi) in groups.items():
            diff = [(u[i][lo:hi]-u[i+1][lo:hi])[:,mask].ravel() for i in (0,1)]
            magnitudes = np.array([np.linalg.norm(d) for d in diff])
            ratio = magnitudes[0]/magnitudes[1] if magnitudes[1] > 0 else np.nan
            interpolation = [float(np.linalg.norm(a[2][lo:hi,mask])) for a in aligned]
            row['groups'][name] = dict(rms_differences=(magnitudes/np.sqrt(mask.sum())).tolist(),
                fitted_order=fitted_order(h,ratio),
                time_interpolation_rms_sensitivity=(np.array(interpolation)/np.sqrt(mask.sum())).tolist(),
                time_sensitivity_over_fine_difference=max(interpolation)/magnitudes[1] if magnitudes[1] > 0 else None,
                alignment=float(np.dot(*diff)/np.prod(magnitudes)) if magnitudes.min() > 0 else None)
        report['field_self_convergence'].append(row)
    (args.output/'convergence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    field_rows=[row for row in report['field_self_convergence'] if not row.get('missing')]
    if field_rows:
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        tt=[row['time'] for row in field_rows]
        for group in ['all','gtilde','Q']:
            axes[0].plot(tt,[row['groups'][group]['fitted_order'] for row in field_rows],label=group)
            axes[1].plot(tt,[row['groups'][group]['alignment'] for row in field_rows],label=group)
            axes[2].plot(tt,[row['groups'][group]['time_sensitivity_over_fine_difference'] for row in field_rows],label=group)
        for ax,title in zip(axes,['Fitted field-difference order','Successive difference alignment','Time-interpolation sensitivity / fine difference']):
            ax.set_title(title); ax.set_xlabel('t/M'); ax.axhline(0,color='black',linewidth=.8); ax.grid(alpha=.25)
        axes[0].legend();axes[1].set_ylim(-1.05,1.05)
        fig.suptitle('Common chi-mask Cartesian slice: shrinking norms alone do not establish asymptotic convergence')
        fig.savefig(args.output/'field-convergence.png',dpi=170)
        plt.close(fig)
    print(json.dumps(dict(status=statuses,common_end=end,outputs=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
