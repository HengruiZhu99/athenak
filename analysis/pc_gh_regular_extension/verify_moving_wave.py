"""Compare production RK mask centers with the exact shifted-wave normal flow."""
import argparse
import json
from pathlib import Path
import numpy as np
from make_inputs import parse


def measure(run):
    p = parse((run/'used_input.athinput').read_text())
    amp = float(p['problem']['amp'])
    length = float(p['mesh']['x1max'])-float(p['mesh']['x1min'])
    k = 2*np.pi/length
    rows = []
    for n, path in enumerate(sorted(run.glob('*.co_*.txt'))):
        data = np.atleast_2d(np.loadtxt(path))
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite tracker output')
        x0 = float(p['pc_gh'].get(f'co_{n}_x', 0))
        t = data[:, 1]
        x = np.full(len(t), x0)
        for _ in range(12):
            residual = x-2*amp/k*np.cos(k*(x-t))-x0+2*amp/k*np.cos(k*x0)
            x -= residual/(1+2*amp*np.sin(k*(x-t)))
        err = data[:, 2]-x
        rows.append(dict(file=path.name, time=float(t[-1]), max_x_error=float(abs(err).max()),
                         final_x_error=float(err[-1]), transverse_max=float(abs(data[:,3:5]).max())))
    if len(rows) != 2:
        raise ValueError('Expected two complete production trackers')
    result = dict(nx1=int(p['mesh']['nx1']), integrator=p['time']['integrator'], trackers=rows,
        equation='X - (2 A/k) cos(k(X-t)) = X0 - (2 A/k) cos(k X0)',
        scope='Exact normal flow in shifted gauge wave; errors include field evolution and interpolation')
    (run/'moving-wave-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',type=Path)
    for run in ap.parse_args().runs:
        print(run,json.dumps(measure(run),sort_keys=True),flush=True)
