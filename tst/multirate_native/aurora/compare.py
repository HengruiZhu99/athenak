"""Compare accepted native records by GID and report measured physical-GPU scaling."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'z4c_hyperboloidal'))
from restart_roundtrip import parse, payload


def compare(left, right):
    a, b = (parse(Path(x['checkpoint'])) for x in (left, right))
    for key in ('topology', 'atlas', 'time'):
        if a[key] != b[key]:
            raise ValueError(f'unmatched {key}: {left["checkpoint"]}, {right["checkpoint"]}')
    x, y = payload(a), payload(b)
    worst = 0.
    squared = 0.
    count = 0
    for u, v in zip(x, y):
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite accepted state or persistent diagnostic')
        u, v = u[:, :31], v[:, :31]
        difference = np.abs(u-v)/np.maximum(1., np.maximum(np.abs(u), np.abs(v)))
        worst = max(worst, float(difference.max()))
        squared += np.sum(difference**2)
        count += difference.size
    return dict(max_scaled=worst, rms_scaled=float(np.sqrt(squared/count)), field_scalars=count)


def main():
    root = Path(sys.argv[1])
    data = json.loads((root/'results.json').read_text())
    rows = {(v['mesh'],v['tiles'],v['mode']):v for v in data['results']}
    checks, scaling, methods = [], [], []
    for (mesh,tiles,mode), v in rows.items():
        if mode == 'metis':
            for name, key in [('uniform_reference',(mesh,tiles,'uniform')),
                              ('one_gpu_decomposition',(mesh,2,mode))]:
                if key not in rows or rows[key] is v:
                    continue
                checks.append(dict(mesh=mesh,gpus=tiles//2,comparison=name,**compare(v,rows[key])))
        uniform = rows.get((mesh,tiles,'uniform'))
        if uniform and mode != 'uniform':
            methods.append(dict(mesh=mesh,method=mode,gpus=tiles//2,
                                steady_speedup_vs_uniform=uniform['steady_seconds_per_M0']/v['steady_seconds_per_M0'],
                                elapsed_speedup_vs_uniform=uniform['seconds_per_M0']/v['seconds_per_M0']))
        one = rows.get((mesh,2,mode))
        if one:
            speedup = one['steady_seconds_per_M0']/v['steady_seconds_per_M0']
            scaling.append(dict(mesh=mesh,method=mode,gpus=tiles//2,nodes=v['nodes'],
                                seconds_per_M0=v['steady_seconds_per_M0'],speedup=speedup,
                                efficiency=speedup/(tiles/2)))
    (root/'comparison.json').write_text(json.dumps(dict(checks=checks,scaling=scaling,methods=methods),indent=2)+'\n')


if __name__=='__main__':
    main()
