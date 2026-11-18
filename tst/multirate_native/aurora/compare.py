"""Compare accepted native records by GID and report measured physical-GPU scaling."""
import json
import math
import re
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'z4c_hyperboloidal'))
from restart_roundtrip import parse, payload


def spatial_identity(identity):
    # The native restart identity includes the integrator and fixed timestep,
    # which intentionally differ in a temporal comparison. Keep the full initial
    # data identity, geometry, physical coefficients and agreement operator.
    spatial, marker, initial = identity.partition(b';initial-data=')
    timing = (b'integrator=',b'p.requested_dt=',b'multirate=',b'spherical-levels=',b'uniform-reference=')
    fields = [value for value in spatial.split(b';')
              if value and not value.startswith(timing) and value != b'agreement=shared-stage-v1']
    return b';'.join(fields)+marker+initial


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
    for key in ('blocks','record_blocks','stride','topology','atlas'):
        if a[key] != b[key]:
            raise ValueError(f'unmatched {key}: {left["checkpoint"]}, {right["checkpoint"]}')
    if spatial_identity(a['identity']) != spatial_identity(b['identity']):
        raise ValueError('N12 spatial physics or initial-data identities differ')
    x, y = payload(a), payload(b)
    worst = 0.
    diagnostic_worst = 0.
    squared = 0.
    count = 0
    identical = True
    for u, v in zip(x, y):
        same=np.array_equal(u.view(np.uint64),v.view(np.uint64))
        if not np.isfinite(u).all() or (not same and not np.isfinite(v).all()):
            raise ValueError('nonfinite accepted state or persistent diagnostic')
        count += u.shape[0]*31
        if same:
            continue
        identical = False
        diag_u, diag_v = u[:,31:], v[:,31:]
        diagnostic_worst=max(diagnostic_worst,float((np.abs(diag_u-diag_v) /
                             np.maximum(1.,np.maximum(np.abs(diag_u),np.abs(diag_v)))).max()))
        u, v = u[:, :31], v[:, :31]
        difference = np.abs(u-v)/np.maximum(1., np.maximum(np.abs(u), np.abs(v)))
        worst = max(worst, float(difference.max()))
        squared += np.sum(difference**2)
    return dict(endpoint_time_delta=abs(a['time']-b['time']),max_scaled=worst, max_scaled_persistent_diagnostics=diagnostic_worst, rms_scaled=float(np.sqrt(squared/count)), field_scalars=count, identical_checkpoint_payload=bool(identical))


def main():
    from transport_modes import TRANSPORT, PACKET_CONTROLS
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
        if mode in TRANSPORT or mode in ('metis','spread','posted_fixed','polled_fixed','zorder','scalar','spread_fixed'):
            for name, key in [('uniform_reference',(mesh,tiles,'uniform')),
                              ('one_gpu_decomposition',(mesh,2,mode))]:
                reference=rows.get(key,reference_rows.get(key))
                if not reference or reference is v or reference['simulated_M0']!=v['simulated_M0']:
                    continue
                checks.append(dict(mesh=mesh,method=mode,gpus=tiles//2,comparison=name,**compare(v,reference)))
        if mode.endswith('_sibling_batch'):
            baseline = mode.removesuffix('_sibling_batch')
            control = rows[(mesh,tiles,baseline)]
            for key in ('job_id','binary_sha256','rank_binding','intervals'):
                if v[key]!=control[key]:
                    raise ValueError('unmatched sibling-tile trial '+key)
            placement=json.loads(Path(v['log']).with_name('sibling-placement.json').read_text())
            observed={int(re.search(r' batches=(\d+)',line)[1]) for line in v['work']
                      if line.startswith('MULTIRATE_WORK cycle=')}
            if observed!={placement['predicted_batches']}:
                raise ValueError('executed sibling batches differ from the audited schedule')
            fields=compare(v,control)
            # Reuse the native multirate ownership qualification tolerance.
            if max(fields['max_scaled'],fields['max_scaled_persistent_diagnostics'])>1e-11:
                raise ValueError('sibling-tile placement changed the numerical N12 trajectory')
            checks.append(dict(mesh=mesh,method=mode,gpus=tiles//2,comparison=baseline,**fields))
            methods.append(dict(mesh=mesh,method=mode,control=baseline,gpus=tiles//2,
                                steady_speedup=control['steady_seconds_per_M0']/v['steady_seconds_per_M0']))
        if mode.endswith('_whole_gpu'):
            baseline=mode.removesuffix('_whole_gpu')
            control=rows[(mesh,tiles,baseline)]
            for key in ('job_id','binary_sha256','intervals','physical_gpus'):
                if v[key]!=control[key]:
                    raise ValueError('unmatched whole-GPU control '+key)
            for rank,(host,tile) in control['rank_binding'].items():
                if v['rank_binding'][str(int(rank)//2)]!=[host,tile.split('.')[0]]:
                    raise ValueError('whole-GPU trial changed the physical rank mapping')
            if baseline!='classical':
                def owners(receipt):
                    with Path(receipt['log']).with_name('partition.log').open('rb') as stream:
                        rows=stream.read(int(stream.readline())).decode().splitlines()
                    return [int(line.split()[1]) for line in rows[1:]]
                if owners(v)!=[rank//2 for rank in owners(control)]:
                    raise ValueError('whole-GPU trial moved a block between physical GPUs')
            fields=compare(v,control)
            if max(fields['max_scaled'],fields['max_scaled_persistent_diagnostics'])>1e-11:
                raise ValueError('whole-GPU binding changed the numerical trajectory')
            checks.append(dict(mesh=mesh,method=mode,gpus=tiles//2,comparison=baseline,**fields))
            methods.append(dict(mesh=mesh,method=mode,gpus=tiles//2,control=baseline,
                                evolution_speedup=control['evolution_seconds_per_M0']/v['evolution_seconds_per_M0']))
        native_key = (mesh,tiles,'classical_whole_gpu' if mode.endswith('_whole_gpu') else 'classical')
        native = rows.get(native_key,reference_rows.get(native_key))
        if (native and not mode.endswith('_trace') and mode not in ('classical','classical_whole_gpu','uniform') and native['simulated_M0']==v['simulated_M0']
                and native['binary_sha256']==v['binary_sha256']):
            methods.append(dict(mesh=mesh,method=mode,gpus=tiles//2,control='native_uniform_rk4',
                                binary_sha256=v['binary_sha256'],
                                elapsed_speedup=native['seconds']/v['seconds'],
                                steady_speedup=native['steady_seconds_per_M0']/v['steady_seconds_per_M0']))
        uniform = rows.get((mesh,tiles,'uniform'),reference_rows.get((mesh,tiles,'uniform')))
        if uniform and not mode.endswith('_trace') and mode != 'uniform' and uniform['simulated_M0']==v['simulated_M0']:
            methods.append(dict(mesh=mesh,method=mode,gpus=tiles//2,
                                steady_speedup_vs_uniform=uniform['steady_seconds_per_M0']/v['steady_seconds_per_M0'],
                                elapsed_speedup_vs_uniform=uniform['seconds_per_M0']/v['seconds_per_M0']))
        for candidate, control in [('posted_fixed','polled_fixed'),('spread','zorder'),('spread','metis'),('cohort','classical'),('locality','cohort'),('spread_fixed','spread'),*PACKET_CONTROLS.items(),('cohort_cached_packets','cohort_receive_cache')]:
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
    for (mesh,tiles,mode),trial in rows.items():
        if not mode.endswith('_trace') or mode=='classical_trace':
            continue
        control=rows.get((mesh,tiles,'classical_trace'))
        if not control or trial['binary_sha256']!=control['binary_sha256'] or trial['intervals']!=control['intervals']:
            continue
        from device_profile import device_totals
        def summary(receipt):
            # Re-read the rank traces. Older receipts include the driver's
            # copy/fill commands in the field originally named kernel_seconds.
            ranks=device_totals(Path(receipt['log']).parent,tiles)
            operations={}
            for rank in ranks:
                for name,timing in rank['operations'].items():
                    item=operations.setdefault(name,dict(calls=0,nanoseconds=0))
                    for field in item: item[field]+=timing[field]
            return dict(device_seconds=sum(r['device_seconds'] for r in ranks),
                        kernel_seconds=sum(r['kernel_seconds'] for r in ranks),
                        driver_command_seconds=sum(r['driver_command_seconds'] for r in ranks),
                        operations=operations)
        mixed,uniform=summary(trial),summary(control)
        methods.append(dict(mesh=mesh,gpus=tiles//2,method=mode,control='classical_trace',
                            multirate=mixed,uniform=uniform,
                            summed_device_time_ratio=mixed['device_seconds']/uniform['device_seconds'],
                            summed_kernel_time_ratio=mixed['kernel_seconds']/uniform['kernel_seconds'],
                            coarse_intervals=trial['device_profile_coarse_intervals'],
                            scope='driver integrator tasks after two coarse warmup intervals'))
    if filename=='corrected-transport-results.json':
        trial=rows.get(('current754',24,'cohort'))
        if trial:
            for path in sorted(root.glob('control-*-results.json')):
                for control in json.loads(path.read_text())['results']:
                    if (control['mesh'],control['tiles'],control['mode'],control['intervals'],control['job_id']) != (
                            trial['mesh'],trial['tiles'],trial['mode'],trial['intervals'],trial['job_id']):
                        continue
                    first=Path(trial['log']).with_name('partition.log').read_bytes()
                    second=Path(control['log']).with_name('partition.log').read_bytes()
                    if first != second:
                        raise ValueError('different replay placement in transport comparison')
                    fields=compare(trial,control)
                    if not fields['identical_checkpoint_payload']:
                        raise ValueError('transport changed accepted Scheme A state or diagnostics')
                    checks.append(dict(mesh=trial['mesh'],method='cohort',gpus=12,
                                       comparison=path.stem,**fields))
                    methods.append(dict(mesh=trial['mesh'],method='cohort',control=path.stem,
                                        control_sha256=control['binary_sha256'],gpus=12,
                                        steady_speedup=control['steady_seconds_per_M0']/trial['steady_seconds_per_M0']))
    (root/filename.replace('results','comparison')).write_text(json.dumps(dict(checks=checks,scaling=scaling,methods=methods),indent=2)+'\n')


if __name__=='__main__':
    main()
