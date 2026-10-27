"""Plot matched inner profiles from one archived refinement assessment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--radius', type=float, default=.06)
    a = p.parse_args()
    if not 0 < a.radius < 1:
        raise ValueError('Need an inner radius in (0,1)')
    assessment = a.frame/'assessment.json'
    profiles = a.frame/'physical-profiles.npz'
    data = json.loads(assessment.read_text())
    assert data['kind'] == 'spatial'
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    for run in data['runs']:
        for name, sha in run['hashes'].items():
            assert digest(Path(run['case'])/name) == sha
    a.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    columns = [('g', 'Radial conformal metric'), ('alpha', 'Conformal lapse'),
               ('Arr', r'Radial curvature $A_{rr}$'), ('p+3', r'Reduced curvature $p+3$')]
    with np.load(profiles) as z:
        for k, run in enumerate(data['runs']):
            r, q = z[f'r{k}'], z[f'fields{k}']
            assert np.isfinite(q).all()
            mask = (r > 0) & (r <= a.radius)
            for ax, (field, title) in zip(axes.flat, columns):
                index = data['fields'].index(field)
                ax.plot(r[mask], q[mask, index], '-o', markersize=2,
                        linewidth=1.1, label=f'N={run["n"]}')
                ax.set(title=title, xlabel='Compact radius r', xlim=(0, a.radius))
                ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_yscale('symlog', linthresh=1)
    axes[0, 0].legend()
    fig.suptitle(f'Matched inner profiles at {data["t_over_M"]:g}M; all positive-radius nodes shown')
    fig.savefig(a.output/'inner-profiles.png', dpi=180)
    plt.close(fig)
    provenance = dict(t=data['t'], t_over_M=data['t_over_M'], radius=a.radius,
                      inputs={str(f):digest(f) for f in [assessment, profiles]},
                      script_sha256=digest(Path(__file__)),
                      scope='Same-time coordinate profiles, not invariant comparisons; no smoothing or node removal within the plotted positive-radius interval. Curvature panels use symmetric logarithmic axes.')
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
