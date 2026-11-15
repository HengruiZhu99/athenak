#!/usr/bin/env python3
"""Validate and summarize a downloaded native scri experiment segment.

No interpolation of stage-window extrema and no reconstruction of pointwise
identities from independently reduced extrema. Native histories are retained.
"""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np

def history(path):
    text = path.read_text()
    header = next(s for s in text.splitlines() if '[1]=time' in s)
    names = re.findall(r'\[\d+\]=(\S+)', header)
    a = np.loadtxt(path, ndmin=2)
    assert len(names) == a.shape[1] and np.isfinite(a).all(), path
    # Native Finalize can repeat a periodic dump when nlim ends on that cycle.
    # Accept only exactly identical same-time rows; never conceal disagreement.
    steps=np.diff(a[:,0]);assert np.all(steps>=0), path
    for i in np.flatnonzero(steps==0):
        assert np.array_equal(a[i],a[i+1]),('nonidentical duplicate history',path,i)
    a=a[np.r_[True,steps>0]]
    return a, dict(zip(names, a.T))

def performance(path):
    records = re.findall(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', path.read_text())
    if len(records) < 2:
        return None
    values = np.array(records, float)
    assert np.all(np.diff(values[:, 0]) > 0) and np.all(np.diff(values[:, 1]) > 0), path
    cost = np.diff(values[:, 0])/np.diff(values[:, 1])
    return dict(intervals=len(cost), last_cycle=int(values[-1, 1]),
        last_logged_M0=float(values[-1, 2]*3),
        interval_s_per_step_min=float(np.min(cost)),
        interval_s_per_step_median=float(np.median(cost)),
        interval_s_per_step_max=float(np.max(cost)),
        measured_loop_s_per_step=float((values[-1,0]-values[0,0])/(values[-1,1]-values[0,1])),
        note='Intervals include any history/checkpoint costs; minimum is not a controlled no-output benchmark.')

def summarize(directory, baseline=None):
    p = Path(directory)
    raw = next(p.glob('*.z4c_hyperboloidal.hst'))
    a, labels = history(raw)
    result = dict(last_history_M0=float(a[-1, 0]*3), samples=len(a),
                  constraint_weight='compact-coordinate volume; accepted horizon surface mask',
                  exterior=dict(zip(['H_M0_squared', 'M_M0_squared', 'Theta_M0', 'Z_M0'],
                                    (a[-1, 8:]*np.array([1/9, 1/9, 1/3, 1/3])).tolist())))
    signed, absolute = {}, {}
    for group in range(4):
        path = next(p.glob(f'*.scri-investigation-{group}.hst'))
        b, columns = history(path)
        # A live copy may include more rows in a later-copied file. Summarize
        # each stream at its own recorded time, never pretend they are aligned.
        for name, data in columns.items():
            if name in ('time', 'dt'):
                continue
            if name.endswith('-min'):
                counterpart = columns[name[:-4]+'-max']
                assert np.all(data <= counterpart), (path, name)
                signed[name[:-4]] = dict(min=float(data[-1]), max=float(counterpart[-1]),
                                        t_M0=float(b[-1, 0]*3))
            elif not name.endswith('-max'):
                assert np.all(data >= 0), (path, name)
                absolute[name] = dict(max=float(data[-1]), t_M0=float(b[-1, 0]*3))
    result['scri_native_signed_extrema'] = signed
    result['scri_native_absolute_maxima'] = absolute
    result['shells'] = []
    for group in range(4):
        path = next(p.glob(f'*.scri-shell-{group}.hst'))
        b, columns = history(path)
        assert np.all(b[:, 2:10] >= 0), path
        result['shells'].append(dict(band=group, t_M0=float(b[-1, 0]*3),
                                     native_values={k:float(v[-1]) for k,v in columns.items()
                                                    if k not in ('time', 'dt')}))
    horizon = np.loadtxt(next(p.glob('*.horizon_summary_0.txt')), ndmin=2)
    assert np.isfinite(horizon).all() and np.all(horizon[:, 8] >= 0)
    # FastFlow's legacy "hrms" column stores integral(H^2 dA)/area.
    # fastflow.cpp::result() takes its square root for expansion_rms.
    result['horizon'] = dict(t_M0=float(horizon[-1, 1]*3),
                            mass_M0=float(horizon[-1, 2]*3),
                            expansion_mean_square_native=float(horizon[-1, 8]),
                            expansion_rms_native=float(np.sqrt(horizon[-1, 8])))
    if (p/'merge-manifest.json').exists():
        manifest = json.loads((p/'merge-manifest.json').read_text())
        result['performance_by_segment'] = [dict(directory=s['directory'],
            measurements=performance(Path(s['directory'])/'run.log')) for s in manifest['segments']]
        result['incomplete_extrema_windows'] = {name:[row['time_native']*3 for row in stream['rows']
            if not row['complete_window']] for name,stream in manifest['streams'].items() if stream['windowed']}
    else:
        measured = performance(p/'run.log')
        if measured is not None:
            result['performance'] = measured
    if baseline:
        b, _ = history(Path(baseline)); differences = []
        for row in a:
            idx = np.argmin(abs(b[:, 0]-row[0]))
            if abs(b[idx, 0]-row[0]) < 1e-12:
                differences.append(row[8:]-b[idx, 8:])
        assert differences
        result['historical_exterior_comparison'] = dict(matched_samples=len(differences),
            max_abs_native=float(np.max(np.abs(differences))),
            equal_at_all_matched_samples=bool(np.all(np.array(differences)==0)))
    receipt = p/'exit-code.txt'
    result['recorded_exit_code'] = int(receipt.read_text()) if receipt.exists() else None
    result['history_sha256'] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in sorted(p.glob('*.hst'))}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory, args.baseline)
    (args.directory/'segment-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in
        ('last_history_M0', 'exterior', 'horizon', 'performance', 'historical_exterior_comparison',
         'recorded_exit_code')}, indent=2))
