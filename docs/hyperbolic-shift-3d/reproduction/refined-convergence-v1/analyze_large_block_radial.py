"""Verify read-only native reconstructions and localize the admitted short errors."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reader', type=Path)
    p.add_argument('campaign', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    terminal = json.loads((a.campaign / 'terminal-verification.json').read_text())
    assert terminal['terminal'] and terminal['scheduler_exit'] == 0
    results = {}
    profiles = {}
    for name, admitted in zip(['block12', 'block24'], terminal['cases']):
        native = json.loads((a.reader / (name + '.json')).read_text())
        root = a.campaign / 'output' / name
        report = json.loads((root / 'state.report.json').read_text())
        cp_hash = sha(root / 'state.checkpoint')
        assert admitted['verified'] and admitted['sha256'] == cp_hash
        assert native['identity'] == report['identity']
        assert native['steps'] == report['steps'] == 600
        assert abs(native['t_over_M'] - .075) < 1e-12
        assert native['analysis_override'] == 'none'
        full = native['integrals'][0]
        norms = [math.sqrt(full[i] / full[0]) for i in [1, 2]]
        reported = [report['H'], report['M']]
        assert all(abs(x-y) <= 5e-11 * max(abs(y), 1e-30)
                   for x, y in zip(norms, reported))
        assert native['native_volume_integral_relative_difference'] < 1e-12
        bins = native['radial_bins']
        exterior = [b for b in bins if b[0] >= .3 - 1e-12]
        totals = [sum(b[i] for b in exterior) for i in [3, 4]]
        regions = {}
        for lo, hi in [(.3,.6),(.6,.78),(.78,.9),(.9,.94),(.94,.95),(.95,.97),(.97,1)]:
            selected = [b for b in bins if b[0] >= lo-1e-12 and b[1] <= hi+1e-12]
            regions[f'{lo:g}-{hi:g}'] = {
                'fraction_H2': sum(b[3] for b in selected)/totals[0],
                'fraction_M2': sum(b[4] for b in selected)/totals[1]}
        results[name] = dict(native_full_norms=norms, report_norms=reported,
            checkpoint_unchanged_sha256=cp_hash, regions=regions,
            fixed03_norms=[math.sqrt(native['integrals'][1][i]/native['integrals'][1][0]) for i in [1,2]])
        (a.output / (name+'-verification.json')).write_text(json.dumps(results[name],indent=2)+'\n')
        for suffix in ['.json','-command.json','.log']:
            shutil.copy2(a.reader / (name+suffix), a.output / (name+suffix))
        profiles[name] = exterior
    results['qualification'] = ('Same production constraint operator on each distinct admitted mesh; '
        'fixed coordinate cut r>.3. Radial squared-integral distribution, no ray interpolation. '
        '600 steps/.075M only; no long stability, convergence, or pure packing attribution.')
    (a.output / 'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax, index, label in zip(axes,[3,4],['Hamiltonian','Momentum']):
        for name,bins in profiles.items():
            ax.semilogy([(b[0]+b[1])/2 for b in bins],
                        [max(b[index]/(b[1]-b[0]),1e-30) for b in bins],'-o',ms=3,label=name)
        ax.axvline(.95,color='k',ls=':',label='original mesh transition')
        ax.axvspan(.94,.97,color='grey',alpha=.12)
        ax.set(xlim=(.78,1),ylim=(1e-14,1e5),xlabel='compact radius r',ylabel=label+' squared integral / bin width')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Native volume constraints at .075M; no interpolation of fields')
    for ext in ['png','pdf']:
        fig.savefig(a.output/('radial-comparison.'+ext),dpi=170)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
