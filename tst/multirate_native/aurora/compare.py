"""Compare accepted native records by GID and report measured physical-GPU scaling."""
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'z4c_hyperboloidal'))
from restart_roundtrip import parse, payload


def compare(left, right):
    a, b = (parse(Path(x['checkpoint'])) for x in (left, right))
    if left['simulated_M0']!=right['simulated_M0']:
        raise ValueError('different requested physical durations')
    expected=left['simulated_M0']/3
    for endpoint in (a,b):
        # The native driver accumulates one addition per step. Even dyadically
        # related non-binary-exact inputs can end a few ulps apart.
        if abs(endpoint['time']-expected)>max(1,endpoint['cycle'])*math.ulp(expected):
            raise ValueError('checkpoint time differs from requested endpoint beyond summation roundoff')
    for key in ('topology', 'atlas'):
        if a[key] != b[key]:
            raise ValueError(f'unmatched {key}: {left["checkpoint"]}, {right["checkpoint"]}')
    x, y = payload(a), payload(b)
    worst = 0.
    squared = 0.
    count = 0
    identical = True
    for u, v in zip(x, y):
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite accepted state or persistent diagnostic')
        identical = identical and np.array_equal(u.view(np.uint64),v.view(np.uint64))
        u, v = u[:, :31], v[:, :31]
        difference = np.abs(u-v)/np.maximum(1., np.maximum(np.abs(u), np.abs(v)))
        worst = max(worst, float(difference.max()))
        squared += np.sum(difference**2)
        count += difference.size
    return dict(endpoint_time_delta=abs(a['time']-b['time']),max_scaled=worst, rms_scaled=float(np.sqrt(squared/count)), field_scalars=count, identical_checkpoint_payload=bool(identical))


def main():
    root = Path(sys.argv[1])
    filename = sys.argv[2] if len(sys.argv)>2 else 'results.json'
    data = json.loads((root/filename).read_text())
    reference_rows = {}
    for previous in ('results.json','larger-results.json','throughput-results.json'):
        if previous != filename and (root/previous).exists():
            baseline=json.loads((root/previous).read_text())
            reference_rows.update({(v['mesh'],v['tiles'],v['mode']):v for v in baseline['results']})
    rows = {(v['mesh'],v['tiles'],v['mode']):v for v in data['results']}
    checks, scaling, methods = [], [], []
    for (mesh,tiles,mode), v in rows.items():
        if mode in ('metis','spread','posted_fixed','polled_fixed','zorder','scalar','cohort','locality'):
            for name, key in [('uniform_reference',(mesh,tiles,'uniform')),
                              ('one_gpu_decomposition',(mesh,2,mode))]:
                reference=rows.get(key,reference_rows.get(key))
                if not reference or reference is v or reference['simulated_M0']!=v['simulated_M0']:
                    continue
                checks.append(dict(mesh=mesh,method=mode,gpus=tiles//2,comparison=name,**compare(v,reference)))
        uniform = rows.get((mesh,tiles,'uniform'),reference_rows.get((mesh,tiles,'uniform')))
        if uniform and mode != 'uniform' and uniform['simulated_M0']==v['simulated_M0']:
            methods.append(dict(mesh=mesh,method=mode,gpus=tiles//2,
                                steady_speedup_vs_uniform=uniform['steady_seconds_per_M0']/v['steady_seconds_per_M0'],
                                elapsed_speedup_vs_uniform=uniform['seconds_per_M0']/v['seconds_per_M0']))
        for candidate, control in [('posted_fixed','polled_fixed'),('spread','zorder'),('spread','metis'),('cohort','spread'),('cohort','classical'),('locality','cohort')]:
            other=rows.get((mesh,tiles,control),reference_rows.get((mesh,tiles,control)))
            if mode==candidate and other and other['simulated_M0']==v['simulated_M0']:
                methods.append(dict(mesh=mesh,method=mode,control=control,gpus=tiles//2,
                                    steady_speedup=other['steady_seconds_per_M0']/v['steady_seconds_per_M0']))
                checks.append(dict(mesh=mesh,method=mode,gpus=tiles//2,comparison=control,**compare(v,other)))
        one = rows.get((mesh,2,mode))
        if one:
            speedup = one['steady_seconds_per_M0']/v['steady_seconds_per_M0']
            scaling.append(dict(mesh=mesh,method=mode,gpus=tiles//2,nodes=v['nodes'],
                                seconds_per_M0=v['steady_seconds_per_M0'],speedup=speedup,
                                efficiency=speedup/(tiles/2)))
    (root/filename.replace('results','comparison')).write_text(json.dumps(dict(checks=checks,scaling=scaling,methods=methods),indent=2)+'\n')


if __name__=='__main__':
    main()
