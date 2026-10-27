"""Compare verified terminal histories at exact shared diagnostic times.

Reports pointwise exceptions as well as time-integrated and peak errors. No
interpolation or removal of near-zero mass-error samples is permitted.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_refinement import PHYSICS, SOURCES
from continue_puncture import digest
from continue_series import verify


def plot_histories(output, manifests, t, data):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.95,hspace=.30)
    for ax,field,title in zip(axes,['H_exterior_rms','M_exterior_rms','absolute_relative_AH_mass_error'],
                              ['Exterior Hamiltonian RMS','Exterior momentum RMS','Absolute relative AH mass error']):
        for m,values in zip(manifests,data[field]):
            ax.semilogy(3*t,np.ma.masked_less_equal(values,0),label=f'N={m["arguments"]["n"]}')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel('Time / M')
    fig.savefig(output/'matched-histories.png',dpi=170,bbox_inches=None)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert len(a.cases) in [2,3]
    manifests = [verify(c) for c in a.cases]
    histories = [json.loads((c/'history.json').read_text()) for c in a.cases]
    base = manifests[0]
    for i, m in enumerate(manifests):
        for key in PHYSICS+['cfl']:
            assert m['arguments'].get(key) == base['arguments'].get(key), key
        assert m['library_sha256'] == base['library_sha256']
        for name in SOURCES:
            key = 'tst/hyperbolic_shift_1d/'+name
            assert m['sources'][key] == base['sources'][key], name
        if i:
            assert m['arguments']['n'] == 2*manifests[i-1]['arguments']['n']-1
            assert manifests[i-1]['dt'] == 2*m['dt']
    dt = base['dt']
    maps = []
    for history in histories:
        index = {}
        for record in history:
            step = round(record['t']/dt)
            assert abs(step*dt-record['t']) < 1e-12
            assert step not in index, 'Duplicate diagnostic time'
            index[step] = record
        maps.append(index)
    keys = sorted(set.intersection(*(set(m) for m in maps)))
    assert len(keys) > 1 and keys[0] == 0
    # Every diagnostic through the shortest completed history must be shared.
    end = min(max(m) for m in maps)
    for m in maps:
        assert set(k for k in m if k <= end) == set(keys), 'Missing diagnostic samples'
    t = np.array(keys)*dt
    fields = ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms']
    data = {field:np.array([[m[k][field] for k in keys] for m in maps]) for field in fields}
    mass = np.array([[m[k]['apparent_horizon_mass'] for k in keys] for m in maps])
    data['absolute_relative_AH_mass_error'] = abs(3*mass-1)
    assert all(np.isfinite(v).all() for v in data.values())
    metrics = {}
    for field, values in data.items():
        peak = np.max(values, axis=1)
        time_rms = np.sqrt(np.trapezoid(values**2,t,axis=1)/(t[-1]-t[0]))
        exceptions = []
        for pair in range(len(values)-1):
            bad = np.flatnonzero(values[pair+1] >= values[pair])
            exceptions.append(dict(coarse_n=manifests[pair]['arguments']['n'],
                                   fine_n=manifests[pair+1]['arguments']['n'],
                                   count=len(bad), samples=[dict(t=float(t[j]),t_over_M=float(3*t[j]),
                                     coarse=float(values[pair,j]),fine=float(values[pair+1,j])) for j in bad]))
        metrics[field] = dict(peak=peak.tolist(),time_rms=time_rms.tolist(),
                             peak_decreases=bool(np.all(np.diff(peak)<0)),
                             time_rms_decreases=bool(np.all(np.diff(time_rms)<0)),
                             pointwise_exceptions=exceptions)
    result = dict(t_start=float(t[0]),t_end=float(t[-1]),t_end_over_M=float(3*t[-1]),
                  exact_common_samples=len(keys), metrics=metrics,
                  peak_relative_AH_mass_drift=np.max(abs(mass/mass[:,0,None]-1),axis=1).tolist(),
                  runs=[dict(case=str(c),n=m['arguments']['n'],dt=m['dt'],
                             hashes={name:digest(c/name) for name in ['manifest.json','history.json','profiles.npz']})
                        for c,m in zip(a.cases,manifests)],
                  scope='Matched history diagnostics, not field self-convergence. All shared samples retained, including zero crossings of mass error. Horizon-exterior spatial norms come from the unchanged run diagnostic; time RMS uses trapezoidal time weights. Two-grid results do not establish a three-grid convergence order.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'matched-histories.npz',t=t,AH_mass=mass,**data)
    plot_histories(a.output,manifests,t,data)
    print(json.dumps(dict(t_end_over_M=result['t_end_over_M'],samples=len(keys),
                         metrics={k:dict(peak=v['peak'],time_rms=v['time_rms'],
                           pointwise_exception_counts=[e['count'] for e in v['pointwise_exceptions']]) for k,v in metrics.items()}),indent=2))


if __name__ == '__main__':
    main()
