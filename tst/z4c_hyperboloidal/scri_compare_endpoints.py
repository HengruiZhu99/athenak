#!/usr/bin/env python3
"""Report a matched A/B endpoint comparison; this is not a convergence test."""
import argparse
import json
from pathlib import Path
import numpy as np
from scri_run_summary import summarize, history


def extrema_window(root, begin_M0, end_M0):
    """Reduce an exact union of recorded stage windows; never interpolate."""
    root=Path(root)
    manifest_path=root/'merge-manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    signed,absolute,provenance={},{},{}
    for group in range(4):
        path=next(root.glob(f'*.scri-investigation-{group}.hst'))
        array,columns=history(path);times=3*array[:,0]
        first=np.flatnonzero(abs(times-begin_M0)<1e-9)
        last=np.flatnonzero(abs(times-end_M0)<1e-9)
        assert len(first)==len(last)==1 and first[0]<last[0], 'window endpoints must be recorded dumps'
        indices=range(int(first[0])+1,int(last[0])+1)
        if manifest:
            rows=manifest['streams'][path.name]['rows']
            assert len(rows)==len(times)
            for i in indices:
                row=rows[i]
                assert row['complete_window'], 'cannot reconstruct a lost restart window'
                assert abs(3*row['time_native']-times[i])<1e-9
                assert abs(3*row['window_start_native']-times[i-1])<1e-9, 'gap or overlap in stage windows'
        selection=slice(int(first[0])+1,int(last[0])+1)
        for name,values in columns.items():
            if name in ('time','dt') or name.endswith('-max'):
                continue
            if name.endswith('-min'):
                high=columns[name[:-4]+'-max']
                assert np.all(values<=high)
                signed[name[:-4]]=dict(min=float(np.min(values[selection])),
                                      max=float(np.max(high[selection])),t_M0=float(times[last[0]]))
            else:
                assert np.all(values>=0)
                absolute[name]=dict(max=float(np.max(values[selection])),t_M0=float(times[last[0]]))
        provenance[path.name]=dict(begin_M0=float(times[first[0]]),end_M0=float(times[last[0]]),
                                   source_rows=list(indices),subwindows=len(indices))
    return signed,absolute,provenance


def compare(a_path, b_path):
    roots = [Path(a_path), Path(b_path)]
    data = [summarize(root) for root in roots]
    final = [d['last_history_M0'] for d in data]
    assert abs(final[0]-final[1]) < 1e-9, ('unmatched endpoints', final)
    identities = []
    modes = []
    windows = []
    for root, item in zip(roots, data):
        receipt = json.loads((root/'final-checkpoint.json').read_text())
        assert abs(receipt['time_M0']-item['last_history_M0']) < 1e-9
        horizon = np.loadtxt(next(root.glob('*.horizon_summary_0.txt')), ndmin=2)
        assert horizon[-1, 0] == receipt['cycle']
        identities.append(';'.join(s for s in receipt['identity'].split(';')
                                   if not s.startswith('scri-treatment-v1=')))
        modes.append(next((s.split('=',1)[1] for s in receipt['identity'].split(';')
                           if s.startswith('scri-treatment-v1=')), '0'))
        times = []
        for group in range(4):
            h, _ = history(next(root.glob(f'*.scri-investigation-{group}.hst')))
            assert abs(h[-1, 0]*3-final[0]) < 1e-9
            times.append(h[:, 0]*3)
        assert all(np.array_equal(t, times[0]) for t in times)
        windows.append(times[0])
        assert all(abs(s['t_M0']-final[0]) < 1e-9 for s in item['shells'])
    assert identities[0] == identities[1], 'A/B differ beyond the boundary treatment'
    assert modes == ['0','1'], ('expected modes A, B', modes)
    common=[time for time in windows[0][:-1] if np.any(abs(windows[1][:-1]-time)<1e-9)]
    assert common, 'no shared recorded beginning for final extrema window'
    begin=max(common)
    window_sources=[]
    for root,item in zip(roots,data):
        signed,absolute,source=extrema_window(root,begin,final[0])
        item['scri_native_signed_extrema']=signed
        item['scri_native_absolute_maxima']=absolute
        window_sources.append(source)
    windows=[[source[next(iter(source))]['begin_M0'],source[next(iter(source))]['end_M0']]
             for source in window_sources]

    def pair(a, b):
        return dict(A=float(a), B=float(b), B_minus_A=float(b-a),
                    B_over_A=float(b/a) if a else None)

    constraints = {name:pair(data[0]['exterior'][name], data[1]['exterior'][name])
                   for name in data[0]['exterior']}
    scri = {}
    for name in data[0]['scri_native_signed_extrema']:
        values = [d['scri_native_signed_extrema'][name] for d in data]
        scri[name] = pair(*[max(abs(v['min']), abs(v['max'])) for v in values])
    for name in data[0]['scri_native_absolute_maxima']:
        scri[name] = pair(*[d['scri_native_absolute_maxima'][name]['max'] for d in data])
    shells = []
    for group in range(4):
        values = [d['shells'][group]['native_values'] for d in data]
        shells.append({name:pair(values[0][name], values[1][name]) for name in values[0]})
    return dict(time_M0=final, extrema_window_M0=windows,extrema_window_sources=window_sources,
                scope='One matched resolution and timestep; no convergence claim. Boundary reflections unresolved.',
                exterior_constraints_M0=constraints, scri_native_magnitudes=scri,
                fixed_shells_native=shells,
                horizon={name:pair(data[0]['horizon'][name], data[1]['horizon'][name])
                         for name in ('mass_M0', 'expansion_rms_native')},
                performance={label:d.get('performance', d.get('performance_by_segment'))
                             for label,d in zip(['A','B'], data)},
                source_histories={label:d['history_sha256'] for label,d in zip(['A','B'], data)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--A', type=Path, required=True)
    parser.add_argument('--B', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.A, args.B)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('time_M0','exterior_constraints_M0','horizon')}, indent=2))
