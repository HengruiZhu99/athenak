"""Localize spatial self-differences using already verified common-time frames.

Retains all positive-radius nodes and adds the auxiliary shift driver D, which
is not included in the historical nine-field mixed norm. Regional orders are
diagnostics, not a replacement for full/exterior qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    trajectory = json.loads(a.trajectory.read_text())
    results = []
    for index, row in enumerate(trajectory['rows']):
        assert 'D' not in row['fields'], 'Regional audit expects the historical nine-field input'
        archived = a.trajectory.parent / f'frame-{index:02d}' / 'physical-profiles.npz'
        states = []
        with np.load(archived) as physical:
            for k, run in enumerate(row['runs']):
                case = Path(run['case'])
                for name, sha in run['hashes'].items():
                    assert digest(case/name) == sha, (case, name)
                with np.load(case/'profiles.npz') as raw:
                    hits = np.flatnonzero(abs(raw['t']-row['t']) < 1e-13)
                    assert len(hits) == 1
                    r = physical[f'r{k}'].copy()
                    assert np.array_equal(r, raw['r'])
                    fields = np.c_[physical[f'fields{k}'], raw['q'][hits[0], :, 9]]
                states.append((r, fields, run['diagnostics']['apparent_horizon_r']))
        pairs = []
        for k in range(2):
            r, q, ah = states[k]
            rf, qf, ahf = states[k+1]
            assert np.array_equal(r, rf[::2])
            # Recover the exact mapping Jacobian from the frozen run input.
            # Existing stored regional quadrature is not inferred from dr.
            from puncture import PunctureEvolution
            import inspect
            case = Path(row['runs'][k]['case'])
            m = json.loads((case/'manifest.json').read_text())
            kwargs = {key: value for key, value in m['arguments'].items()
                      if key in inspect.signature(PunctureEvolution).parameters
                      and key not in ['n', 'library']}
            lib = case/'source'/Path(m['arguments']['library']).name
            assert digest(lib) == m['library_sha256']
            # Geometry mapping code must match the archived source, too.
            import puncture
            source_key = 'tst/hyperbolic_shift_1d/puncture.py'
            assert digest(Path(puncture.__file__)) == m['sources'][source_key]
            evolution = PunctureEvolution(len(r), lib.resolve(), **kwargs)
            assert np.array_equal(evolution.r, r)
            w = r**2 * evolution.jac
            w[[0, -1]] *= .5
            d = q-qf[::2]
            masks = dict(full=r >= 0, exterior=r >= max(ah, ahf),
                         interior=r < max(ah, ahf), puncture=r <= .02,
                         scri_collar=r >= .9)
            pair = {}
            for name, mask in masks.items():
                rms = np.sqrt(np.sum(w[mask, None]*d[mask]**2, axis=0)/sum(w[mask]))
                pair[name] = dict(field_rms=rms.tolist(),
                                  historical_mixed_rms=float(np.linalg.norm(rms[:-1])),
                                  all_field_mixed_rms=float(np.linalg.norm(rms)),
                                  nodes=int(sum(mask)), min_r=float(min(r[mask])),
                                  max_r=float(max(r[mask])))
            for name in ['full', 'exterior']:
                assert np.isclose(pair[name]['historical_mixed_rms'],
                                  row['comparisons'][k][name]['mixed_rms'], rtol=1e-13, atol=0)
            pairs.append(pair)
        regions = {}
        for name in pairs[0]:
            x, y = pairs[0][name], pairs[1][name]
            field_orders = [float(np.log2(u/v)) if u > 0 and v > 0 else None
                            for u, v in zip(x['field_rms'], y['field_rms'])]
            regions[name] = dict(pairs=[x, y], field_orders=field_orders,
                                 all_field_order=float(np.log2(x['all_field_mixed_rms']/y['all_field_mixed_rms'])))
        results.append(dict(t=row['t'], t_over_M=row['t_over_M'], regions=regions))
    result = dict(trajectory=str(a.trajectory), trajectory_sha256=digest(a.trajectory),
                  fields=trajectory['rows'][0]['fields']+['D'], rows=results,
                  quadrature='r^2 times mapping Jacobian with trapezoidal endpoint weights; each region normalized by its own weights.',
                  validation='Nine-field full and exterior norms reproduce the parent collector at relative tolerance 1e-13.',
                  scope='Regional self-differences only. Individual near-roundoff orders are not accuracy evidence. Interior masks use the larger AH radius in each pair; no nodes are removed from the full-domain assessment.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    time = [r['t_over_M'] for r in results]
    sizes = [r['n'] for r in trajectory['rows'][0]['runs']]
    colors = dict(full='tab:blue', exterior='tab:green', interior='tab:orange',
                  puncture='tab:red', scri_collar='tab:purple')
    labels = dict(full='Full domain', exterior='AH exterior', interior='AH interior',
                  puncture='Puncture: r <= 0.02', scri_collar='Scri collar: r >= 0.9')
    for name in colors:
        axes[0, 0].plot(time, [r['regions'][name]['all_field_order'] for r in results],
                        color=colors[name], label=labels[name])
    for name in ['full', 'exterior']:
        for pair, style in [(0, '-'), (1, '--')]:
            grid = f'{sizes[pair]} vs {sizes[pair+1]}'
            values = [r['regions'][name]['pairs'][pair]['all_field_mixed_rms'] for r in results]
            axes[0, 1].semilogy(time, values, style, color=colors[name],
                                label=labels[name]+', '+grid)
    for name in ['puncture', 'scri_collar']:
        for pair, style in [(0, '-'), (1, '--')]:
            axes[1, 0].semilogy(time, [r['regions'][name]['pairs'][pair]['all_field_mixed_rms'] for r in results],
                                style, color=colors[name], label=labels[name]+f', {sizes[pair]}/{sizes[pair+1]}')
    for name in ['full', 'exterior', 'scri_collar']:
        axes[1, 1].plot(time, [r['regions'][name]['field_orders'][-1] for r in results],
                        color=colors[name], label=labels[name])
    for ax, title in zip(axes.flat, ['Spatial order, including driver D', 'Ten-field self-differences',
                                    'Local ten-field self-differences', 'Auxiliary shift driver D: spatial order']):
        ax.set(title=title, xlabel='Time / M')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(a.output.with_suffix('.'+ext), dpi=170)
    plt.close(fig)
    print(json.dumps([dict(t_over_M=r['t_over_M'], orders={k:v['all_field_order'] for k,v in r['regions'].items()}) for r in results], indent=2))


if __name__ == '__main__':
    main()
