"""Describe completed endpoint experiments; never automatically certify reproduction."""
import argparse
import json
import re
from pathlib import Path
import numpy as np


def read_history(path):
    labels = None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            pairs = re.findall(r'\[(\d+)\]=([^\s]+)', line)
            if pairs:
                current = [label for _, label in pairs]
                if labels is not None and current != labels:
                    raise ValueError('history schema changed within file')
                labels = current
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    if labels is None or not rows:
        raise ValueError('missing history header or samples')
    values = np.asarray(rows)
    if values.shape[1] != len(labels) or not np.all(np.isfinite(values)):
        raise ValueError('invalid history data')
    result = {name: values[:, i] for i, name in enumerate(labels)}
    if np.any(np.diff(result['time']) < 0):
        raise ValueError('history time goes backwards')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    metrics = ['minLapse', 'maxAbsKret', 'C-norm2', 'axisLapse', 'nmb_total', 'maxRefLev']
    cases = []
    for row in m['cases']:
        directory = Path(row['directory'])
        run = json.loads((directory/'result.json').read_text())
        if run['returncode'] != 0:
            raise ValueError('failed experiment: '+row['name'])
        paths = list(directory.glob('*.user.hst'))
        if len(paths) != 1:
            raise ValueError('expected one user history')
        history = read_history(paths[0])
        if abs(history['time'][-1]-m['end']) > 1e-12*max(1, abs(m['end'])):
            raise ValueError('experiment did not reach matched endpoint')
        final = {key: float(history[key][-1]) for key in metrics}
        cases.append((row, run, history, final))
    reference = next(case for case in cases if case[0]['name'] == 'classical_sync')
    report = dict(start=m['start'], end=m['end'],
                  scope='Short late-checkpoint segment only; scientific reproduction not certified.',
                  original_final_history_row=m['original_final_history_row'], cases=[])
    for row, run, history, final in cases:
        difference = {key: final[key]-reference[3][key] for key in metrics}
        relative = {key: difference[key]/abs(reference[3][key])
                    if reference[3][key] != 0 else None for key in metrics}
        report['cases'].append(dict(name=row['name'], final=final,
            difference_from_classical_sync=difference,
            relative_difference_from_classical_sync=relative,
            wall_seconds=run['wall_seconds'],
            speedup_vs_classical_sync=reference[1]['wall_seconds']/run['wall_seconds']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        for row, _, history, _ in cases:
            ax.plot(history['time']-m['start'], history[metric], '.-', label=row['name'])
        ax.set_ylabel(metric)
        ax.set_xlabel('Coordinate time since checkpoint')
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle('Brill endpoint experiment: native sample times; no temporal interpolation\n'
                 'Short continuation; reproduction and spatial accuracy require separate assessment')
    a.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(a.output/'diagnostics.png', dpi=160)
    plt.close(fig)
    (a.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
