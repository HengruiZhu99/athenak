"""Archive terminal damping controls and compare their measured histories.

No failed/bounded run is relabelled and no unequal-time profile order is fitted.
"""
import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from puncture import PunctureEvolution


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--archive-cases', nargs='*', default=[])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    records = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), layout='constrained')
    profiles, pax = plt.subplots(3, 1, figsize=(8, 8.5), layout='constrained')
    for name in dict.fromkeys(a.cases + a.archive_cases):
        src = a.root / name
        m = json.loads((src/'manifest.json').read_text())
        assert m['status'] != 'running', name
        for rel, sha in m['sources'].items():
            assert digest(src/'source'/rel) == sha, (name, rel)
        lib = src/'source'/Path(m['arguments']['library']).name
        assert digest(lib) == m['library_sha256']
        for fn, sha in m['checkpoint_hashes'].items():
            assert digest(src/fn) == sha, (name, fn)
        if 'parent' in m:
            for fn, sha in m['parent']['hashes'].items():
                assert digest(Path(m['parent']['path'])/fn) == sha
            assert digest(src/'continuation_harness.py') == m['continuation_harness_sha256']
        dest = a.output/name
        dest.mkdir(parents=True)
        for fn in ['manifest.json', 'history.json', 'profiles.npz',
                   'failed-stage.npz', 'continuation_harness.py']:
            if (src/fn).exists():
                shutil.copy2(src/fn, dest/fn)
        h = json.loads((src/'history.json').read_text())
        records.append(dict(case=name, status=m['status'], t=m['t'], error=m['error'],
                            sources_checkpoints_parents_verified=True, final=h[-1]))
        if not m['arguments']['puncture']:
            continue
        mass = np.array([x['apparent_horizon_mass'] for x in h])
        records[-1].update(peak_relative_mass_drift=float(max(abs(mass/mass[0]-1))),
                           final_relative_mass_error=float(3*mass[-1]-1))
        if name not in a.cases:
            continue
        t = 3*np.array([x['t'] for x in h])
        n = m['arguments']['n']
        adjusted = m['arguments'].get('inner_z4_damping', 0) != 0
        label = f'N={n}, '+('adjusted damping' if adjusted else 'original damping')
        ls = '-' if adjusted else '--'
        color = {257: 'tab:blue', 513: 'tab:orange', 1025: 'tab:green'}[n]
        for ax, key in zip([axes[0,0], axes[0,1], axes[2,0]],
                           ['H_exterior_rms', 'M_exterior_rms', 'H_interior_max']):
            ax.semilogy(t, [max(x[key], 1e-16) for x in h], ls, color=color, label=label)
        axes[1,0].plot(t, 1e6*(3*mass-1), ls, color=color, label=label)
        axes[1,1].plot(t, [x['g_min'] for x in h], ls, color=color, label=label)
        axes[2,1].semilogy(t, [max(x['pole'], 1e-16) for x in h], ls, color=color, label=label)
        if m['status'] != 'completed':
            axes[1,1].plot(t[-1], h[-1]['g_min'], 'x', color=color, ms=8)
        args = {k:v for k,v in m['arguments'].items()
                if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
        e = PunctureEvolution(n, lib.resolve(), **args)
        z = np.load(src/'profiles.npz')
        e.rhs(z['last'])
        v = e.last
        snapshot = label+f', {3*m["t"]:g}M'
        for ax, key in zip(pax, ['alpha', 'beta', 'g']):
            mask = (e.r > 0) & (e.r <= .3)
            ax.plot(e.r[mask], v[key][mask], ls, color=color, label=snapshot)
        records[-1]['final_inner_beta_slope'] = v['inner_beta_slope']
        records[-1]['final_AF_damping_margin'] = args.get('inner_z4_damping',0)-2*v['inner_beta_slope']
    titles = ['Exterior Hamiltonian RMS', 'Exterior momentum RMS', 'AH mass error [ppm]',
              'Minimum radial metric', 'Interior Hamiltonian maximum', 'Scri numerator maximum']
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Time / M')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    axes[1,0].set_yscale('symlog', linthresh=.01)
    axes[1,0].axhline(0, color='black', lw=.6)
    for ax, title in zip(pax, ['Conformal lapse', 'Radial shift', 'Radial metric']):
        ax.set(title=title, xlabel='Compact radius r', xlim=(0,.3))
        ax.grid(alpha=.2)
    pax[0].set_ylim(bottom=0)
    pax[0].legend(fontsize=9)
    for ext in ['pdf', 'png']:
        fig.savefig(a.output/f'damping-histories.{ext}', dpi=170)
        profiles.savefig(a.output/f'damping-profiles.{ext}', dpi=170)
    result = dict(runs=records, scope='Terminal histories; unequal-time profiles are labelled. No long-time convergence claim.', goal_complete=False)
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'hashes.json').write_text(json.dumps({str(f.relative_to(a.output)):digest(f)
        for f in a.output.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k!='final'} for x in records],indent=2))


if __name__ == '__main__':
    main()
